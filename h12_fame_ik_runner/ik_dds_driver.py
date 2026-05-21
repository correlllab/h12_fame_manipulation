"""FrameController driver for the safety_split topology.

Two modes are supported via --mode:

  batch (default)
    Step through the goal sequence declared in the runner YAML
    (``ik.goals``). Used by the orchestrator's headless smoke test.

  interactive
    Drop into a REPL where the user can request:
      - a named configuration (e.g. ``home``)
      - a 6-DOF frame target on a specific link
        (``left x y z roll pitch yaw`` / ``right ...`` / ``frame <name> ...``)
      - ``show`` to print current left / right wrist poses
      - ``q`` / ``quit`` to exit
    RPY values are entered in DEGREES (converted to radians internally), x/y/z
    in metres. This mirrors the input format used by ``frame_controller_goto.py``
    in the controller repo.

Both modes publish to whatever ``low_cmd`` topic the controller config maps to
(typically ``rt/safety/lowcmd_upper_in`` via ``sim_split_controller.yaml``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from h12_ros2_controller.core.controller.frame_controller import FrameController
from h12_ros2_controller.utility.controller_config import (
    initialize_channel_factory,
    load_controller_config,
)
from h12_ros2_controller.utility.named_config import NAMED_CONFIGS

# Built-in frame aliases. The keys are short names users can type at the REPL,
# the values are the corresponding link names in the H1-2 URDF.
FRAME_ALIASES = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
    "l_wrist": "left_wrist_yaw_link",
    "r_wrist": "right_wrist_yaw_link",
}


def _resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _make_frame_controller(runner_yaml: dict, cfg_dir: Path) -> FrameController:
    """Instantiate FrameController + DDS using the runner YAML's ik section."""
    ik = runner_yaml.get("ik", {})
    controller_cfg_name = ik.get("controller_config", "safety_split")
    cfg_arg = Path(controller_cfg_name)
    if not cfg_arg.is_absolute() and ("/" in controller_cfg_name or cfg_arg.suffix):
        cfg_arg = _resolve_path(controller_cfg_name, cfg_dir)
    controller_config = load_controller_config(str(cfg_arg))

    # ARM_KP_MULT env override: scales the arm kp values FrameController
    # will embed in every LowCmd. Indices 13..26 in BODY_JOINTS are the
    # 7 left + 7 right arm joints; 12 is the torso. We scale 13..26 (both
    # arms) so the PD applies more torque for the same position error —
    # the most direct knob for "push harder when the arm is at workspace
    # limit". Wrists scale too but they're typically already low (20).
    mult_env = os.environ.get("ARM_KP_MULT", "").strip()
    if mult_env:
        try:
            mult = float(mult_env)
            kp = controller_config.get("gains", {}).get("kp", None)
            if kp is not None and len(kp) >= 27:
                old = list(kp[13:27])
                for i in range(13, 27):
                    kp[i] = float(kp[i]) * mult
                new = list(kp[13:27])
                print(f"[ik] ARM_KP_MULT={mult:.3f} : arm kp scaled\n"
                      f"     old (13..26): {old}\n"
                      f"     new (13..26): {new}", flush=True)
        except ValueError:
            print(f"[ik] ignoring invalid ARM_KP_MULT={mult_env!r}", flush=True)

    initialize_channel_factory(controller_config)

    urdf_path = _resolve_path(ik["urdf_path"], cfg_dir)
    urdf_sphere = _resolve_path(ik["urdf_sphere_path"], cfg_dir)
    srdf_sphere = _resolve_path(ik["srdf_sphere_path"], cfg_dir)

    print(
        f"[ik] urdf={urdf_path.name} controller_config={controller_cfg_name}",
        flush=True,
    )

    return FrameController(
        str(urdf_path), str(urdf_sphere), str(srdf_sphere),
        handless=bool(ik.get("handless", True)),
        visualize=bool(ik.get("visualize", False)),
        config=controller_config,
    )


def _goal_label(goal: dict) -> str:
    if "name" in goal:
        return f"named:{goal['name']}"
    if "frame_delta" in goal:
        return f"frame_delta:{goal['frame_delta']}"
    if "frame" in goal:
        return f"frame:{goal['frame']}"
    if "q_reduced" in goal:
        return f"q_reduced[{len(goal['q_reduced'])}]"
    return f"unknown:{goal}"


def _resolve_frame_link(frame: str) -> str:
    return FRAME_ALIASES.get(frame, frame)


def _parse_yaml_pose(pose: list[float]) -> np.ndarray:
    """YAML pose is [x, y, z, R_deg, P_deg, Y_deg] — convert RPY to radians."""
    if len(pose) != 6:
        raise ValueError(f"frame goal pose must have 6 values; got {len(pose)}")
    vals = [float(v) for v in pose]
    vals[3:] = list(np.deg2rad(vals[3:]))
    return np.asarray(vals, dtype=np.float64)


def run_batch(
    frame_controller: FrameController,
    goals: list[dict],
    duration_cap: float,
) -> None:
    """Step through ``goals``. Each goal is one of

        {name: <named_config>, seconds: <float>}
        {frame: <alias_or_link>, pose: [x y z R_deg P_deg Y_deg], seconds: <float>,
         linear_thresh: <float, opt>, angular_thresh: <float, opt>}
        {frame_delta: <alias_or_link>, delta: [dx dy dz dR_deg dP_deg dY_deg],
         seconds: <float>, linear_thresh: <float, opt>, angular_thresh: <float, opt>}
        {q_reduced: <len-14 list of floats>, seconds: <float>}

    Named goals hold the reduced configuration for the full ``seconds``.
    Frame goals push the wrist toward the absolute 6-DOF pose using the IK
    frame_task path and exit early on convergence.
    frame_delta goals read the link's CURRENT IK-frame pose, add the delta,
    and drive frame_task toward that target. Use this for "move 20 cm in +x
    from wherever you are now" — the result is a clean straight-line motion
    in IK-frame coordinates regardless of starting pose. Absolute frame
    goals are sensitive to URDF/MJCF kinematic mismatch and body attitude.
    q_reduced goals bypass IK entirely and command an explicit upper-body
    joint vector (ENABLED_JOINTS order: 7 left arm + 7 right arm).
    """
    start = time.time()
    last_named_q = None
    print(f"[ik] batch mode; goals={[_goal_label(g) for g in goals]}", flush=True)
    for goal in goals:
        seconds = float(goal.get("seconds", 5.0))
        if "name" in goal:
            name = goal["name"]
            if name not in NAMED_CONFIGS:
                print(
                    f"[ik] skipping unknown named goal {name}; valid: {list(NAMED_CONFIGS)}",
                    flush=True,
                )
                continue
            q_reduced = NAMED_CONFIGS[name]
            last_named_q = q_reduced
            print(f"[ik] goal=named:{name} for {seconds:.1f}s", flush=True)
            t_goal = time.time()
            while time.time() - t_goal < seconds:
                step_start = time.time()
                frame_controller.goto_reduced_configuration(q_reduced)
                if duration_cap > 0 and (time.time() - start) >= duration_cap:
                    return
                sleep_for = frame_controller.dt - (time.time() - step_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        elif "q_reduced" in goal:
            try:
                q_reduced = np.asarray(goal["q_reduced"], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                print(f"[ik] skipping malformed q_reduced goal: {exc}", flush=True)
                continue
            last_named_q = q_reduced
            print(
                f"[ik] goal=q_reduced[{q_reduced.shape[0]}] for {seconds:.1f}s  "
                f"q={np.array2string(q_reduced, precision=3, separator=',')}",
                flush=True,
            )
            t_goal = time.time()
            while time.time() - t_goal < seconds:
                step_start = time.time()
                frame_controller.goto_reduced_configuration(q_reduced)
                if duration_cap > 0 and (time.time() - start) >= duration_cap:
                    return
                sleep_for = frame_controller.dt - (time.time() - step_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        elif "frame_delta" in goal:
            link = _resolve_frame_link(goal["frame_delta"])
            try:
                delta = _parse_yaml_pose(goal["delta"])
            except (KeyError, ValueError) as exc:
                print(f"[ik] skipping malformed frame_delta goal {goal}: {exc}",
                      flush=True)
                continue
            # PUSH_DELTA_X env override: lets a sweep script vary the push
            # amplitude without rewriting YAML for each trial. Replaces the
            # x-component of the delta only (y, z, RPY unchanged).
            env_dx = os.environ.get("PUSH_DELTA_X", "").strip()
            if env_dx:
                try:
                    old = float(delta[0])
                    delta[0] = float(env_dx)
                    print(f"[ik] PUSH_DELTA_X override: delta_x {old:.3f} -> {delta[0]:.3f}",
                          flush=True)
                except ValueError:
                    print(f"[ik] ignoring invalid PUSH_DELTA_X={env_dx!r}", flush=True)
            try:
                current = frame_controller.get_frame_pose(link)
            except Exception as exc:
                print(f"[ik] could not read current pose of {link}: {exc}",
                      flush=True)
                continue
            target = np.asarray(current, dtype=np.float64) + delta
            lin_thr = float(goal.get("linear_thresh", 5e-3))
            ang_thr = float(goal.get("angular_thresh", 2e-2))
            print(
                f"[ik] goal=frame_delta:{link}  "
                f"current=({current[0]:+.3f},{current[1]:+.3f},{current[2]:+.3f})  "
                f"delta=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})  "
                f"target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})  "
                f"max={seconds:.1f}s",
                flush=True,
            )
            _drive_frame_task(
                frame_controller, link, target, seconds,
                linear_thresh=lin_thr, angular_thresh=ang_thr,
            )
            if duration_cap > 0 and (time.time() - start) >= duration_cap:
                return
        elif "frame" in goal:
            link = _resolve_frame_link(goal["frame"])
            try:
                pose = _parse_yaml_pose(goal["pose"])
            except (KeyError, ValueError) as exc:
                print(f"[ik] skipping malformed frame goal {goal}: {exc}", flush=True)
                continue
            lin_thr = float(goal.get("linear_thresh", 5e-3))
            ang_thr = float(goal.get("angular_thresh", 2e-2))
            print(
                f"[ik] goal=frame:{link} pose_xyz=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) "
                f"rpy_deg=({np.rad2deg(pose[3]):+.1f},{np.rad2deg(pose[4]):+.1f},{np.rad2deg(pose[5]):+.1f}) "
                f"max={seconds:.1f}s",
                flush=True,
            )
            _drive_frame_task(
                frame_controller, link, pose, seconds,
                linear_thresh=lin_thr, angular_thresh=ang_thr,
            )
            if duration_cap > 0 and (time.time() - start) >= duration_cap:
                return
        else:
            print(f"[ik] skipping goal with no name/frame/frame_delta/q_reduced key: {goal}",
                  flush=True)
            continue

    # After the schedule, hold either the last named config (if any) or the
    # last IK solution (frame_task already left the controller at that pose).
    if last_named_q is not None:
        print("[ik] holding last named goal until interrupted", flush=True)
        while True:
            step_start = time.time()
            frame_controller.goto_reduced_configuration(last_named_q)
            if duration_cap > 0 and (time.time() - start) >= duration_cap:
                return
            sleep_for = frame_controller.dt - (time.time() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    else:
        print("[ik] holding last frame_task pose until interrupted", flush=True)
        while True:
            step_start = time.time()
            frame_controller.control_step_reduced(com=False)
            if duration_cap > 0 and (time.time() - start) >= duration_cap:
                return
            sleep_for = frame_controller.dt - (time.time() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


# ----------------------------------------------------------------- interactive

def _print_help() -> None:
    aliases = ", ".join(f"{k}->{v}" for k, v in FRAME_ALIASES.items())
    print(
        "\n[ik] commands\n"
        f"  show                              - print current left/right wrist poses\n"
        f"  <named_config>                    - run a NAMED_CONFIGS entry "
        f"({list(NAMED_CONFIGS)})\n"
        f"  left  x y z  R P Y                - target left wrist; xyz in m, RPY in deg\n"
        f"  right x y z  R P Y                - target right wrist; xyz in m, RPY in deg\n"
        f"  frame <link_name>  x y z  R P Y   - target an arbitrary link\n"
        f"  timeout <seconds>                 - max time per goal (default 6 s)\n"
        f"  help                              - print this help\n"
        f"  q / quit                          - leave the REPL (stack keeps running)\n"
        f"\nFrame aliases: {aliases}\n",
        flush=True,
    )


def _parse_pose(tokens: list[str]) -> np.ndarray:
    """Parse 6 numbers (x y z R P Y) — RPY in degrees → radians."""
    if len(tokens) != 6:
        raise ValueError(f"need 6 values (x y z R P Y); got {len(tokens)}")
    vals = [float(t) for t in tokens]
    vals[3:] = list(np.deg2rad(vals[3:]))
    return np.asarray(vals, dtype=np.float64)


def _show_pose(frame_controller: FrameController) -> None:
    try:
        left = frame_controller.left_ee_pose
        right = frame_controller.right_ee_pose
        print(
            f"[ik] left_wrist  xyz=({left[0]:+.3f},{left[1]:+.3f},{left[2]:+.3f}) "
            f"rpy_deg=({np.rad2deg(left[3]):+.1f},{np.rad2deg(left[4]):+.1f},{np.rad2deg(left[5]):+.1f})",
            flush=True,
        )
        print(
            f"[ik] right_wrist xyz=({right[0]:+.3f},{right[1]:+.3f},{right[2]:+.3f}) "
            f"rpy_deg=({np.rad2deg(right[3]):+.1f},{np.rad2deg(right[4]):+.1f},{np.rad2deg(right[5]):+.1f})",
            flush=True,
        )
    except Exception as exc:
        print(f"[ik] could not read EE poses: {exc}", flush=True)


def _drive_named(frame_controller: FrameController, name: str, timeout: float) -> None:
    q_reduced = NAMED_CONFIGS[name]
    print(f"[ik] driving named '{name}' for up to {timeout:.1f}s", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        step_start = time.time()
        frame_controller.goto_reduced_configuration(q_reduced)
        sleep_for = frame_controller.dt - (time.time() - step_start)
        if sleep_for > 0:
            time.sleep(sleep_for)


def _drive_frame_task(
    frame_controller: FrameController,
    frame_name: str,
    pose: np.ndarray,
    timeout: float,
    linear_thresh: float = 5e-3,
    angular_thresh: float = 2e-2,
) -> None:
    task_name = f"{frame_name}_task"
    frame_controller.clear_frame_tasks()
    frame_controller.add_frame_task(task_name, frame_name, pose)
    frame_controller.update_ik_solver()
    print(
        f"[ik] driving frame_task on {frame_name}; "
        f"xyz=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) "
        f"rpy_deg=({np.rad2deg(pose[3]):+.1f},{np.rad2deg(pose[4]):+.1f},{np.rad2deg(pose[5]):+.1f}) "
        f"timeout={timeout:.1f}s",
        flush=True,
    )

    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < timeout:
        step_start = time.time()
        frame_controller.control_step_reduced(com=False)
        err = frame_controller.get_frame_task_error(task_name)
        lin = float(np.linalg.norm(err[:3]))
        ang = float(np.linalg.norm(err[3:]))
        now = time.time()
        if now - last_print > 0.5:
            print(f"[ik]   linear={lin:.4f} m  angular={ang:.4f} rad", flush=True)
            last_print = now
        if lin < linear_thresh and ang < angular_thresh:
            print("[ik]   target reached", flush=True)
            break
        sleep_for = frame_controller.dt - (time.time() - step_start)
        if sleep_for > 0:
            time.sleep(sleep_for)
    # settle a few extra steps so the publisher latches the final pose
    for _ in range(50):
        step_start = time.time()
        frame_controller.control_step_reduced(com=False)
        sleep_for = frame_controller.dt - (time.time() - step_start)
        if sleep_for > 0:
            time.sleep(sleep_for)


def run_interactive(frame_controller: FrameController) -> None:
    print("[ik] entering interactive mode", flush=True)
    _print_help()
    timeout = 6.0
    while True:
        try:
            line = input("ik> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ik] leaving REPL", flush=True)
            return
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            print("[ik] leaving REPL", flush=True)
            return
        if cmd in ("h", "help", "?"):
            _print_help()
            continue
        if cmd == "show":
            _show_pose(frame_controller)
            continue
        if cmd == "timeout":
            if len(parts) != 2:
                print("[ik] usage: timeout <seconds>", flush=True)
                continue
            try:
                timeout = float(parts[1])
                print(f"[ik] timeout set to {timeout:.2f}s", flush=True)
            except ValueError as exc:
                print(f"[ik] invalid timeout: {exc}", flush=True)
            continue
        if cmd in NAMED_CONFIGS:
            try:
                _drive_named(frame_controller, cmd, timeout)
            except Exception as exc:
                print(f"[ik] error driving '{cmd}': {exc}", flush=True)
            continue
        if cmd in FRAME_ALIASES:
            try:
                pose = _parse_pose(parts[1:])
            except ValueError as exc:
                print(f"[ik] {exc}", flush=True)
                continue
            try:
                _drive_frame_task(frame_controller, FRAME_ALIASES[cmd], pose, timeout)
            except Exception as exc:
                print(f"[ik] error driving frame task: {exc}", flush=True)
            continue
        if cmd == "frame":
            if len(parts) < 8:
                print("[ik] usage: frame <link_name> x y z R P Y", flush=True)
                continue
            link = parts[1]
            try:
                pose = _parse_pose(parts[2:])
            except ValueError as exc:
                print(f"[ik] {exc}", flush=True)
                continue
            try:
                _drive_frame_task(frame_controller, link, pose, timeout)
            except Exception as exc:
                print(f"[ik] error driving frame task: {exc}", flush=True)
            continue

        print(
            f"[ik] unrecognised command: {line!r}. Type 'help' for usage.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameController driver for safety_split")
    parser.add_argument("--config", required=True, type=str, help="Path to runner YAML")
    parser.add_argument(
        "--mode", choices=("batch", "interactive"), default="batch",
        help="batch: step through YAML goals (default); "
             "interactive: stdin REPL for named/frame_task commands",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Shorthand for --mode interactive",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Optional cap (batch mode only); 0 = run sequence then hold forever",
    )
    args = parser.parse_args()
    if args.interactive:
        args.mode = "interactive"

    config_path = Path(args.config).resolve()
    top = _load_yaml(config_path)
    cfg_dir = config_path.parent

    frame_controller = _make_frame_controller(top, cfg_dir)
    try:
        if args.mode == "interactive":
            run_interactive(frame_controller)
        else:
            goals = list(top.get("ik", {}).get("goals", [{"name": "home", "seconds": 10.0}]))
            run_batch(frame_controller, goals, args.duration)
    except KeyboardInterrupt:
        print("[ik] interrupted, shutting down", flush=True)
    finally:
        try:
            frame_controller.shutdown()
        except Exception as exc:
            print(f"[ik] shutdown error: {exc}", flush=True)


if __name__ == "__main__":
    main()

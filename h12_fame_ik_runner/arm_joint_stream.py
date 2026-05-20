"""High-bandwidth joint-command streamer for the H1-2 upper body.

Publishes a 27-motor LowCmd to ``rt/safety/lowcmd_upper_in`` at 500 Hz with
``mode=1`` on motors 12..26 (torso + 14 arm joints) and ``mode=0`` on legs.
The published target is whatever the current ``q_des`` is — there's no goal
state machine, no IK, no convergence wait. Update ``q_des`` from the REPL
(or from a Python driver via ``ArmJointStream.set_joint(...)`` /
``set_all(...)``) and the next tick (within 2 ms) sends it.

Use this when you want bare joint control of the arm — e.g. teleop, scripted
trajectories, calibration moves — bypassing the FrameController IK loop. The
safety_layer's split-mode merge still clips the command and drops the legs.

Joint name aliases (case-insensitive):

  torso                       torso_joint                       (index 0)
  lsp lsr lsy le lwr lwp lwy  left  shoulder p/r/y, elbow,      (1..7)
                              left  wrist roll/pitch/yaw
  rsp rsr rsy re rwr rwp rwy  right shoulder p/r/y, elbow,      (8..14)
                              right wrist roll/pitch/yaw

Indices in this script are 0..14 within the 15-element ``arm_q`` vector,
corresponding to motor IDs 12..26 in the 27-motor LowCmd ordering.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as LowCmdDefault,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


NUM_H12_MOTORS = 27
NUM_LEG_MOTORS = 12
ARM_OFFSET = NUM_LEG_MOTORS  # motor IDs 12..26 in LowCmd
NUM_ARM = NUM_H12_MOTORS - NUM_LEG_MOTORS  # 15

ARM_JOINT_NAMES = [
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_ALIASES = {
    "torso": 0, "torso_joint": 0,
    "lsp": 1, "left_shoulder_pitch": 1, "left_shoulder_pitch_joint": 1,
    "lsr": 2, "left_shoulder_roll": 2, "left_shoulder_roll_joint": 2,
    "lsy": 3, "left_shoulder_yaw": 3, "left_shoulder_yaw_joint": 3,
    "le":  4, "left_elbow":         4, "left_elbow_joint":         4,
    "lwr": 5, "left_wrist_roll":    5, "left_wrist_roll_joint":    5,
    "lwp": 6, "left_wrist_pitch":   6, "left_wrist_pitch_joint":   6,
    "lwy": 7, "left_wrist_yaw":     7, "left_wrist_yaw_joint":     7,
    "rsp": 8, "right_shoulder_pitch": 8, "right_shoulder_pitch_joint": 8,
    "rsr": 9, "right_shoulder_roll":  9, "right_shoulder_roll_joint":  9,
    "rsy": 10, "right_shoulder_yaw":  10, "right_shoulder_yaw_joint":  10,
    "re":  11, "right_elbow":        11, "right_elbow_joint":        11,
    "rwr": 12, "right_wrist_roll":   12, "right_wrist_roll_joint":   12,
    "rwp": 13, "right_wrist_pitch":  13, "right_wrist_pitch_joint":  13,
    "rwy": 14, "right_wrist_yaw":    14, "right_wrist_yaw_joint":    14,
}


def _resolve_index(token: str) -> int:
    """Map a user-supplied joint identifier to an arm index 0..14."""
    t = token.strip().lower()
    if t in ARM_ALIASES:
        return ARM_ALIASES[t]
    try:
        idx = int(t)
    except ValueError:
        raise ValueError(f"unknown joint '{token}'. type 'names' for the list")
    if not (0 <= idx < NUM_ARM):
        raise ValueError(f"index {idx} out of range 0..{NUM_ARM - 1}")
    return idx


class ArmJointStream:
    """Background publisher streaming a 27-motor LowCmd to lowcmd_upper_in.

    The publisher thread runs at ``publish_hz`` (default 500 Hz). It always
    writes the latest ``arm_q`` / ``arm_kp`` / ``arm_kd`` to motors 12..26.
    Mutations from the REPL (or from external callers via ``set_joint`` /
    ``set_all``) take effect on the next tick.
    """

    def __init__(
        self,
        topic: str,
        domain_id: int,
        interface: Optional[str],
        default_arm_q: np.ndarray,
        default_arm_kp: np.ndarray,
        default_arm_kd: np.ndarray,
        publish_hz: float = 500.0,
        lowstate_topic: str = "rt/lowstate",
    ):
        if default_arm_q.shape != (NUM_ARM,):
            raise ValueError(f"default_arm_q must have shape ({NUM_ARM},)")
        if default_arm_kp.shape != (NUM_ARM,) or default_arm_kd.shape != (NUM_ARM,):
            raise ValueError("default_arm_kp/kd must have shape (15,)")

        if interface:
            ChannelFactoryInitialize(domain_id, interface)
        else:
            ChannelFactoryInitialize(domain_id)
        self._crc = CRC()
        self._pub = ChannelPublisher(topic, LowCmd_)
        self._pub.Init()

        self._publish_hz = float(publish_hz)
        self._topic = topic
        self._default_q = default_arm_q.astype(np.float32).copy()
        self._default_kp = default_arm_kp.astype(np.float32).copy()
        self._default_kd = default_arm_kd.astype(np.float32).copy()

        self._lock = threading.Lock()
        self.arm_q = self._default_q.copy()
        self.arm_dq = np.zeros(NUM_ARM, dtype=np.float32)
        self.arm_tau = np.zeros(NUM_ARM, dtype=np.float32)
        self.arm_kp = self._default_kp.copy()
        self.arm_kd = self._default_kd.copy()

        # Latest state from rt/lowstate (for `show` command). The publisher
        # thread does not use this; it only reflects what the robot reports.
        self._state_lock = threading.Lock()
        self._state_q = np.zeros(NUM_H12_MOTORS, dtype=np.float32)
        self._got_state = False
        self._state_sub = ChannelSubscriber(lowstate_topic, LowState_)
        self._state_sub.Init(self._on_low_state, 10)

        # Pre-built LowCmd; only the upper-body fields will be mutated.
        self._cmd = LowCmdDefault()
        self._cmd.mode_machine = 6
        for i in range(NUM_ARM):
            mc = self._cmd.motor_cmd[ARM_OFFSET + i]
            mc.mode = 1
            mc.q = float(self._default_q[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self._default_kp[i])
            mc.kd = float(self._default_kd[i])

        self._running = False
        self._thread = threading.Thread(target=self._publish_loop, daemon=True, name="arm_stream")

    # ------------------------------------------------------------------ DDS
    def _on_low_state(self, msg: LowState_) -> None:
        with self._state_lock:
            for i in range(NUM_H12_MOTORS):
                self._state_q[i] = float(msg.motor_state[i].q)
            self._got_state = True

    def _publish_loop(self) -> None:
        dt = 1.0 / self._publish_hz
        while self._running:
            t0 = time.time()
            with self._lock:
                q = self.arm_q.copy()
                dq = self.arm_dq.copy()
                tau = self.arm_tau.copy()
                kp = self.arm_kp.copy()
                kd = self.arm_kd.copy()
            for i in range(NUM_ARM):
                mc = self._cmd.motor_cmd[ARM_OFFSET + i]
                mc.mode = 1
                mc.q = float(q[i])
                mc.dq = float(dq[i])
                mc.tau = float(tau[i])
                mc.kp = float(kp[i])
                mc.kd = float(kd[i])
            self._cmd.crc = self._crc.Crc(self._cmd)
            self._pub.Write(self._cmd)
            sleep_for = dt - (time.time() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ---------------------------------------------------------- public API
    def start(self) -> None:
        self._running = True
        self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass
        for c in (self._pub, self._state_sub):
            try:
                c.Close()
            except Exception:
                pass

    def set_joint(self, idx: int, q_rad: float) -> None:
        with self._lock:
            self.arm_q[idx] = float(q_rad)

    def set_all(self, q_rad: np.ndarray) -> None:
        if q_rad.shape != (NUM_ARM,):
            raise ValueError(f"set_all expects shape ({NUM_ARM},)")
        with self._lock:
            self.arm_q[:] = q_rad.astype(np.float32)

    def reset_defaults(self) -> None:
        with self._lock:
            self.arm_q[:] = self._default_q
            self.arm_kp[:] = self._default_kp
            self.arm_kd[:] = self._default_kd
            self.arm_dq[:] = 0.0
            self.arm_tau[:] = 0.0

    def set_uniform_gain(self, kind: str, value: float) -> None:
        with self._lock:
            if kind == "kp":
                self.arm_kp[:] = float(value)
            elif kind == "kd":
                self.arm_kd[:] = float(value)
            else:
                raise ValueError("gain must be 'kp' or 'kd'")

    def snapshot(self) -> dict:
        with self._lock:
            q = self.arm_q.copy()
            kp = self.arm_kp.copy()
            kd = self.arm_kd.copy()
        with self._state_lock:
            state_q = self._state_q.copy() if self._got_state else None
        return {"q_des": q, "kp": kp, "kd": kd, "state_q": state_q}


# ------------------------------------------------------------------- REPL

def _print_help() -> None:
    print(
        "\n[arm] commands\n"
        "  show                           - print q_des, kp/kd, and current q from rt/lowstate\n"
        "  names                          - list all 15 arm-joint names and indices\n"
        "  set <joint> <radians>          - set ONE joint by name or index (in radians)\n"
        "  setdeg <joint> <degrees>       - same in degrees\n"
        "  nudge <joint> <delta_radians>  - add delta_radians to ONE joint (sign matters)\n"
        "  setall  <q0> <q1> ... <q14>    - set all 15 joints (radians)\n"
        "  setall_deg <q0> ... <q14>      - set all 15 joints (degrees)\n"
        "  home                           - all zeros (arms straight down, torso=0)\n"
        "  default                        - restore default_angles_arms from the YAML\n"
        "  gain kp <value>                - set kp on all 15 motors\n"
        "  gain kd <value>                - set kd on all 15 motors\n"
        "  help                           - this message\n"
        "  quit / exit                    - stop streaming and exit\n",
        flush=True,
    )


def _print_names() -> None:
    print("[arm] index | alias | full name", flush=True)
    aliases_by_idx = {idx: [] for idx in range(NUM_ARM)}
    for alias, idx in ARM_ALIASES.items():
        if alias != ARM_JOINT_NAMES[idx]:
            aliases_by_idx[idx].append(alias)
    for idx, full in enumerate(ARM_JOINT_NAMES):
        short = next((a for a in aliases_by_idx[idx] if len(a) <= 4), "")
        print(f"  {idx:>2} | {short:<5} | {full}", flush=True)


def _show(stream: ArmJointStream) -> None:
    snap = stream.snapshot()
    print("[arm] q_des (rad / deg):", flush=True)
    for i in range(NUM_ARM):
        q = float(snap["q_des"][i])
        line = (
            f"  {i:>2} {ARM_JOINT_NAMES[i]:<28} q_des={q:+.3f} rad "
            f"({np.rad2deg(q):+6.1f} deg)  kp={snap['kp'][i]:.1f}  kd={snap['kd'][i]:.2f}"
        )
        if snap["state_q"] is not None:
            cur = float(snap["state_q"][ARM_OFFSET + i])
            line += f"  q_meas={cur:+.3f}"
        print(line, flush=True)


def run_repl(stream: ArmJointStream) -> None:
    print(
        f"[arm] streaming on {stream._topic} at {stream._publish_hz:.0f} Hz. "
        f"Type 'help' for commands, 'quit' to exit.",
        flush=True,
    )
    while True:
        try:
            line = input("arm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[arm] leaving REPL", flush=True)
            return
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("quit", "exit"):
                print("[arm] leaving REPL", flush=True)
                return
            elif cmd in ("h", "help", "?"):
                _print_help()
            elif cmd == "names":
                _print_names()
            elif cmd == "show":
                _show(stream)
            elif cmd == "set":
                if len(parts) != 3:
                    raise ValueError("usage: set <joint> <radians>")
                idx = _resolve_index(parts[1])
                stream.set_joint(idx, float(parts[2]))
                print(f"[arm] {ARM_JOINT_NAMES[idx]} -> {float(parts[2]):.4f} rad", flush=True)
            elif cmd == "setdeg":
                if len(parts) != 3:
                    raise ValueError("usage: setdeg <joint> <degrees>")
                idx = _resolve_index(parts[1])
                val = np.deg2rad(float(parts[2]))
                stream.set_joint(idx, val)
                print(f"[arm] {ARM_JOINT_NAMES[idx]} -> {val:.4f} rad ({float(parts[2])} deg)", flush=True)
            elif cmd == "nudge":
                if len(parts) != 3:
                    raise ValueError("usage: nudge <joint> <delta_radians>")
                idx = _resolve_index(parts[1])
                with stream._lock:
                    new = float(stream.arm_q[idx]) + float(parts[2])
                stream.set_joint(idx, new)
                print(f"[arm] {ARM_JOINT_NAMES[idx]} -> {new:.4f} rad", flush=True)
            elif cmd == "setall":
                if len(parts) != 1 + NUM_ARM:
                    raise ValueError(f"setall needs {NUM_ARM} values (got {len(parts) - 1})")
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
                stream.set_all(vec)
                print("[arm] q_des updated (radians)", flush=True)
            elif cmd == "setall_deg":
                if len(parts) != 1 + NUM_ARM:
                    raise ValueError(f"setall_deg needs {NUM_ARM} values (got {len(parts) - 1})")
                vec = np.deg2rad(np.asarray([float(x) for x in parts[1:]], dtype=np.float32))
                stream.set_all(vec)
                print("[arm] q_des updated (converted from degrees)", flush=True)
            elif cmd == "home":
                stream.set_all(np.zeros(NUM_ARM, dtype=np.float32))
                print("[arm] q_des -> home (all zeros)", flush=True)
            elif cmd == "default":
                stream.reset_defaults()
                print("[arm] q_des, kp, kd -> YAML defaults", flush=True)
            elif cmd == "gain":
                if len(parts) != 3 or parts[1].lower() not in ("kp", "kd"):
                    raise ValueError("usage: gain kp <value>  OR  gain kd <value>")
                stream.set_uniform_gain(parts[1].lower(), float(parts[2]))
                print(f"[arm] {parts[1].lower()} -> {float(parts[2]):.2f} on all 15 motors", flush=True)
            else:
                print(f"[arm] unknown command: {line!r}. type 'help' for usage.", flush=True)
        except Exception as exc:
            print(f"[arm] error: {exc}", flush=True)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main():
    p = argparse.ArgumentParser(description="High-bandwidth arm-joint streamer")
    p.add_argument("--config", required=True, type=str, help="Runner YAML")
    p.add_argument("--topic", type=str, default=None,
                   help="Override the topic (default: topics.low_cmd_upper_in from YAML)")
    p.add_argument("--publish-hz", type=float, default=500.0,
                   help="Publish rate (default 500)")
    args = p.parse_args()

    cfg = _load_yaml(Path(args.config).resolve())
    fame = cfg.get("fame", {})
    robot = cfg.get("robot", {})
    topics = cfg.get("topics", {})
    net = cfg.get("network", {})

    default_arms = np.asarray(fame["default_angles_arms"], dtype=np.float32)
    default_kp = np.asarray(robot["default_kp"][NUM_LEG_MOTORS:], dtype=np.float32)
    default_kd = np.asarray(robot["default_kd"][NUM_LEG_MOTORS:], dtype=np.float32)

    topic = args.topic or topics.get("low_cmd_upper_in", "rt/safety/lowcmd_upper_in")
    lowstate_topic = topics.get("low_state", "rt/lowstate")
    domain_id = int(net.get("domain_id", 0))
    interface = net.get("interface") or None

    stream = ArmJointStream(
        topic=topic,
        domain_id=domain_id,
        interface=interface,
        default_arm_q=default_arms,
        default_arm_kp=default_kp,
        default_arm_kd=default_kd,
        publish_hz=args.publish_hz,
        lowstate_topic=lowstate_topic,
    )
    stream.start()
    try:
        run_repl(stream)
    finally:
        stream.shutdown()


if __name__ == "__main__":
    main()

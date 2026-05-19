"""Launch the four processes that comprise the FAME + IK + safety sim stack.

The stack is:
  1. mujoco_dds_bridge   - publishes rt/lowstate, subscribes rt/lowcmd
  2. h12_safety_layer    - merges lower/upper LowCmds, publishes rt/lowcmd
  3. fame_dds_runner     - subscribes rt/lowstate, publishes rt/safety/lowcmd_lower_in
  4. ik_dds_driver       - drives FrameController -> rt/safety/lowcmd_upper_in

Each child runs in its own process group so a Ctrl-C tears them down cleanly.
Per-process stdout/stderr is line-prefixed and forwarded to the orchestrator's
streams so logs from all four are readable in a single terminal.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]  # .../src/
RUNNER_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _python() -> str:
    venv = REPO_ROOT / "h12_adaptive_policy" / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _safety_layer_main() -> Path:
    return REPO_ROOT / "h12_safety_layer" / "h12_safety_layer" / "script" / "safety_layer_main.py"


def _start_process(name: str, argv: list[str], cwd: Optional[Path] = None) -> subprocess.Popen:
    print(f"[orchestrator] starting {name}: {' '.join(argv)}", flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True,  # new process group so SIGTERM tears down children
    )

    def _pump(stream, prefix: str) -> None:
        for line in stream:
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()

    threading.Thread(target=_pump, args=(proc.stdout, name), daemon=True).start()
    return proc


def _stop_all(procs: list[tuple[str, subprocess.Popen]], grace: float = 3.0) -> None:
    for name, proc in procs:
        if proc.poll() is None:
            print(f"[orchestrator] sending SIGINT to {name} (pgid={os.getpgid(proc.pid)})", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

    deadline = time.time() + grace
    for name, proc in procs:
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[orchestrator] {name} still running, sending SIGTERM", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    for name, proc in procs:
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            print(f"[orchestrator] {name} not responding; SIGKILL", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate FAME+IK+safety sim stack")
    parser.add_argument("--config", required=True, type=str, help="Runner YAML")
    parser.add_argument(
        "--safety-config", type=str,
        default=str(RUNNER_ROOT / "configs" / "sim_safety_split.yaml"),
        help="h12_safety_layer config (must be split_mode). Bare names are "
             "resolved against the safety_layer package config dir; absolute "
             "paths are used as-is. Defaults to the runner's sim_safety_split.yaml.",
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run mujoco without viewer")
    parser.add_argument("--viewer", action="store_true",
                        help="Force mujoco viewer (overrides YAML)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Stop the bridge after N seconds (0 = run until Ctrl-C)")
    parser.add_argument("--no-ik", action="store_true",
                        help="Skip the IK driver (FAME-only smoke test)")
    parser.add_argument("--no-fame", action="store_true",
                        help="Skip the FAME runner (IK-only smoke test)")
    parser.add_argument("--bridge-delay", type=float, default=0.0,
                        help="Delay before starting the bridge (lets DDS settle)")
    parser.add_argument("--fame-delay", type=float, default=1.0,
                        help="Delay before starting FAME (let bridge/safety attach)")
    parser.add_argument("--ik-delay", type=float, default=2.0,
                        help="Delay before starting IK driver")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    top = _load_yaml(config_path)
    if "topics" not in top:
        raise ValueError("runner YAML must declare a topics section")

    python = _python()
    pkg_main = lambda mod: [python, "-m", f"h12_fame_ik_runner.{mod}", "--config", str(config_path)]
    safety_argv = [python, str(_safety_layer_main()), "--config", args.safety_config]

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        # 1) safety layer first so it is already merging when FAME/IK come up.
        procs.append(("safety", _start_process("safety", safety_argv)))
        time.sleep(0.5)

        # 2) mujoco bridge (DDS-side requires the safety layer subscriber).
        bridge_args = list(pkg_main("mujoco_dds_bridge"))
        if args.headless:
            bridge_args.append("--headless")
        if args.viewer:
            bridge_args.append("--viewer")
        if args.duration > 0.0:
            bridge_args += ["--duration", str(args.duration)]
        if args.bridge_delay > 0.0:
            time.sleep(args.bridge_delay)
        procs.append(("bridge", _start_process("bridge", bridge_args, cwd=RUNNER_ROOT)))

        # 3) FAME publisher.
        if not args.no_fame:
            time.sleep(args.fame_delay)
            procs.append((
                "fame",
                _start_process("fame", list(pkg_main("fame_dds_runner")), cwd=RUNNER_ROOT),
            ))

        # 4) IK driver.
        if not args.no_ik:
            time.sleep(args.ik_delay)
            procs.append((
                "ik",
                _start_process("ik", list(pkg_main("ik_dds_driver")), cwd=RUNNER_ROOT),
            ))

        # Watch for any child exit.
        start_t = time.time()
        while True:
            for name, proc in procs:
                rc = proc.poll()
                if rc is not None:
                    print(f"[orchestrator] {name} exited with code {rc}", flush=True)
                    return
            if args.duration > 0 and (time.time() - start_t) > args.duration + 5.0:
                print(f"[orchestrator] duration cap reached, stopping", flush=True)
                return
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[orchestrator] Ctrl-C received", flush=True)
    finally:
        _stop_all(procs)
        print("[orchestrator] done", flush=True)


if __name__ == "__main__":
    main()

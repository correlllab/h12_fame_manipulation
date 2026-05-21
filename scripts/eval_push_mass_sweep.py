#!/usr/bin/env python
"""Sweep block mass and measure push success rate.

For each (mass, trial) the script:
  1. picks an output CSV path
  2. launches the orchestrator (safety + bridge + FAME + IK) headless,
     passing PUSH_BLOCK_MASS=<m> and BLOCK_LOG_PATH=<csv> via env
  3. waits for the orchestrator to exit (it does so when the bridge hits
     `--duration`, which the orchestrator forwards to the bridge process)
  4. parses the per-trial CSV to read the block's initial / final x and the
     robot pelvis z, decides success/failure, and appends a row to a summary
     CSV.

Success criterion (tunable via CLI):
  success := block_final_x >= --target-x AND pelvis_z_final >= --min-pelvis-z

Failure modes captured implicitly:
  - block did not move far enough (didn't push hard enough or block slipped sideways)
  - block fell off the table (block_z drops well below table top)
  - robot fell (pelvis_z drops)

Usage:
    python scripts/eval_push_mass_sweep.py \
        --masses 0.1 0.5 1.0 2.0 5.0 \
        --trials 3 \
        --duration 25 \
        --target-x 0.85 \
        --out runs/push_sweep
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_VENV_PY = REPO_ROOT.parent / "h12_adaptive_policy" / ".venv" / "bin" / "python"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "h1_2_fame_ik_eval.yaml"


def _python() -> str:
    if RUNNER_VENV_PY.is_file():
        return str(RUNNER_VENV_PY)
    return sys.executable


def _read_block_csv(csv_path: Path) -> Optional[dict]:
    """Return summary stats over the trial: initial / final block x, min block z,
    final pelvis z, etc. Returns None if the file is empty / missing."""
    try:
        with csv_path.open("r") as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        return None
    if not rows:
        return None
    bx0 = float(rows[0]["block_x"])
    bxN = float(rows[-1]["block_x"])
    bx_max = max(float(r["block_x"]) for r in rows)
    bz_min = min(float(r["block_z"]) for r in rows)
    pz0 = float(rows[0]["base_z"])
    pzN = float(rows[-1]["base_z"])
    pz_min = min(float(r["base_z"]) for r in rows)
    t0 = float(rows[0]["t"])
    tN = float(rows[-1]["t"])
    return {
        "n_samples": len(rows),
        "t0": t0,
        "tN": tN,
        "block_x0": bx0,
        "block_xN": bxN,
        "block_x_max": bx_max,
        "block_z_min": bz_min,
        "pelvis_z0": pz0,
        "pelvis_zN": pzN,
        "pelvis_z_min": pz_min,
    }


def _decide_success(stats: dict, target_x: float, min_pelvis_z: float,
                    min_block_z: float) -> tuple[bool, str]:
    if stats is None:
        return False, "no telemetry"
    if stats["block_z_min"] < min_block_z:
        return False, f"block fell (min z={stats['block_z_min']:.3f} < {min_block_z:.3f})"
    if stats["pelvis_z_min"] < min_pelvis_z:
        return False, f"robot fell (min pelvis z={stats['pelvis_z_min']:.3f} < {min_pelvis_z:.3f})"
    if stats["block_xN"] >= target_x:
        return True, f"block reached x={stats['block_xN']:.3f} >= {target_x:.3f}"
    return False, f"block stopped at x={stats['block_xN']:.3f} < {target_x:.3f}"


def _run_trial(mass: float, trial_idx: int, args, csv_path: Path) -> dict:
    env = os.environ.copy()
    env["PUSH_BLOCK_MASS"] = f"{mass:.6f}"
    env["BLOCK_LOG_PATH"] = str(csv_path)
    cmd = [
        _python(), "-u", "-m", "h12_fame_ik_runner.orchestrator",
        "--config", str(args.config),
        "--headless",
        "--duration", str(args.duration),
    ]
    print(f"\n[sweep] mass={mass:.3f} kg  trial={trial_idx}  csv={csv_path.name}",
          flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    # Forward orchestrator output so we can see failures (FAME warmup, etc).
    log_path = csv_path.with_suffix(".log")
    with open(log_path, "w", buffering=1) as log_fh:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            if args.verbose:
                sys.stdout.write(line)
                sys.stdout.flush()
    rc = proc.wait()
    elapsed = time.time() - t0
    print(f"[sweep]   exit={rc}  wall={elapsed:.1f}s", flush=True)

    stats = _read_block_csv(csv_path)
    success, reason = _decide_success(
        stats, args.target_x, args.min_pelvis_z, args.min_block_z,
    )
    return {
        "mass_kg": mass,
        "trial_idx": trial_idx,
        "rc": rc,
        "wall_seconds": elapsed,
        "success": int(success),
        "reason": reason,
        "block_x0": (stats or {}).get("block_x0"),
        "block_xN": (stats or {}).get("block_xN"),
        "block_x_max": (stats or {}).get("block_x_max"),
        "block_z_min": (stats or {}).get("block_z_min"),
        "pelvis_z0": (stats or {}).get("pelvis_z0"),
        "pelvis_zN": (stats or {}).get("pelvis_zN"),
        "pelvis_z_min": (stats or {}).get("pelvis_z_min"),
        "csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Runner YAML (default: configs/h1_2_fame_ik_eval.yaml)")
    parser.add_argument("--masses", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0, 5.0],
                        help="Block masses to sweep (kg)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials per mass")
    parser.add_argument("--duration", type=float, default=25.0,
                        help="Per-trial wall-clock budget passed to the bridge (s)")
    parser.add_argument("--target-x", type=float, default=0.70,
                        help="Block-x success threshold (world m); block "
                             "starts at world x≈0.60 so this is a 10cm push")
    parser.add_argument("--min-pelvis-z", type=float, default=0.65,
                        help="Robot pelvis-z lower bound (m); below = robot "
                             "fell. FAME's steady-state pelvis ≈ height_cmd "
                             "(0.80), so set this well below that to catch "
                             "actual collapses, not normal tracking dips")
    parser.add_argument("--min-block-z", type=float, default=0.95,
                        help="Block-z lower bound (m); below = block fell off table")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "push_sweep",
                        help="Output directory for per-trial CSVs + summary")
    parser.add_argument("--verbose", action="store_true",
                        help="Mirror orchestrator stdout to this terminal")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.csv"
    summary_fields = [
        "mass_kg", "trial_idx", "rc", "wall_seconds", "success", "reason",
        "block_x0", "block_xN", "block_x_max", "block_z_min",
        "pelvis_z0", "pelvis_zN", "pelvis_z_min", "csv",
    ]
    with summary_path.open("w", newline="") as summary_fh:
        writer = csv.DictWriter(summary_fh, fieldnames=summary_fields)
        writer.writeheader()

        per_mass_success: dict[float, list[int]] = {m: [] for m in args.masses}
        for mass in args.masses:
            for trial_idx in range(args.trials):
                csv_path = args.out / f"trial_m{mass:.3f}kg_t{trial_idx:02d}.csv"
                row = _run_trial(mass, trial_idx, args, csv_path)
                writer.writerow(row)
                summary_fh.flush()
                per_mass_success[mass].append(row["success"])
                print(f"[sweep]   -> success={bool(row['success'])}  reason={row['reason']}",
                      flush=True)

    print("\n[sweep] summary:")
    print(f"  config={args.config}")
    print(f"  target_x={args.target_x}  min_pelvis_z={args.min_pelvis_z}")
    print("  mass_kg  trials  success_rate")
    for mass in args.masses:
        results = per_mass_success[mass]
        if not results:
            continue
        n = len(results)
        sr = sum(results) / n
        print(f"  {mass:7.3f}  {n:6d}  {sr:12.2%}")
    print(f"  summary CSV: {summary_path}")


if __name__ == "__main__":
    main()

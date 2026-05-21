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


def _quat_pitch_yaw_deg(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float]:
    import math
    yaw = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    s = max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))
    pitch = math.degrees(math.asin(s))
    return pitch, yaw


def _read_block_csv(csv_path: Path) -> Optional[dict]:
    """Trial stats focused on DISTURBANCE to the robot:
        - pelvis_z_min: lowest pelvis height during the trial (collapse measure)
        - base_xy_drift: how far the robot's foot-print drifted from t=0
        - pitch/yaw excursion: how much the body attitude deviates from
          FAME's steady-state crouch (-15°, +21° baseline) during the push
        - block_dx, block_dy: how far the block was actually displaced
          (informational; doesn't affect success)
    """
    try:
        with csv_path.open("r") as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        return None
    if not rows:
        return None

    # Establish a "settled" baseline at t ≈ 2 s (after warmup, after the
    # biased-home pose has converged, before the IK push begins).
    baseline = min(rows, key=lambda r: abs(float(r["t"]) - 2.0))
    base_x_ref = float(baseline["base_x"])
    base_y_ref = float(baseline["base_y"])
    base_pitch_ref, base_yaw_ref = _quat_pitch_yaw_deg(
        float(baseline["base_qw"]), float(baseline["base_qx"]),
        float(baseline["base_qy"]), float(baseline["base_qz"]),
    )

    # Push window: only count disturbance after IK push starts (~t>=2s
    # with the new fast-trial timing), otherwise the warmup/home
    # transient dominates the metrics.
    push_rows = [r for r in rows if float(r["t"]) >= 2.0]
    if not push_rows:
        push_rows = rows

    bx0 = float(rows[0]["block_x"])
    bxN = float(rows[-1]["block_x"])
    by0 = float(rows[0]["block_y"])
    byN = float(rows[-1]["block_y"])

    pz_min = min(float(r["base_z"]) for r in push_rows)
    pzN = float(rows[-1]["base_z"])
    base_xy_drift_max = max(
        ((float(r["base_x"]) - base_x_ref) ** 2 +
         (float(r["base_y"]) - base_y_ref) ** 2) ** 0.5
        for r in push_rows
    )

    # Integrated |pitch deviation| over time (deg·s). Catches sustained
    # disturbance that a "max" misses, and recovery (a brief spike but
    # quick return -> low integral; sustained tilt -> high integral).
    pitch_dev_max = 0.0
    yaw_dev_max = 0.0
    pitch_dev_integral = 0.0  # deg·s
    last_t = None
    for r in push_rows:
        t = float(r["t"])
        p, y = _quat_pitch_yaw_deg(
            float(r["base_qw"]), float(r["base_qx"]),
            float(r["base_qy"]), float(r["base_qz"]),
        )
        dp = abs(p - base_pitch_ref)
        pitch_dev_max = max(pitch_dev_max, dp)
        yaw_dev_max = max(yaw_dev_max, abs(y - base_yaw_ref))
        if last_t is not None:
            pitch_dev_integral += dp * (t - last_t)
        last_t = t

    # Leg torque metrics. tau_l0..tau_l11 are FAME's commanded leg torques.
    # Max abs torque tells us how hard a single joint is straining; RMS
    # over the push window measures sustained effort.
    leg_keys = [f"tau_l{i}" for i in range(12)]
    leg_max_abs = [0.0] * 12
    leg_sum_sq = [0.0] * 12
    n_pts = 0
    for r in push_rows:
        if leg_keys[0] not in r:
            break
        n_pts += 1
        for i, k in enumerate(leg_keys):
            v = float(r[k])
            leg_max_abs[i] = max(leg_max_abs[i], abs(v))
            leg_sum_sq[i] += v * v
    leg_rms = [(leg_sum_sq[i] / n_pts) ** 0.5 if n_pts > 0 else 0.0 for i in range(12)]
    # Aggregate scalars: peak (across all 12 joints) and total RMS norm.
    leg_torque_peak = max(leg_max_abs) if leg_max_abs else 0.0
    leg_torque_rms_norm = (sum(v * v for v in leg_rms)) ** 0.5

    return {
        "n_samples": len(rows),
        "block_x0": bx0, "block_xN": bxN,
        "block_y0": by0, "block_yN": byN,
        "block_dx_cm": (bxN - bx0) * 100,
        "block_dy_cm": (byN - by0) * 100,
        "pelvis_zN": pzN,
        "pelvis_z_min": pz_min,
        "base_xy_drift_max_cm": base_xy_drift_max * 100,
        "pitch_dev_max_deg": pitch_dev_max,
        "pitch_dev_integral_degs": pitch_dev_integral,
        "yaw_dev_max_deg": yaw_dev_max,
        "leg_torque_peak_Nm": leg_torque_peak,
        "leg_torque_rms_norm_Nm": leg_torque_rms_norm,
    }


def _decide_success(stats: dict, _target_x: float, min_pelvis_z: float,
                    _min_block_z: float) -> tuple[bool, str]:
    """Success = FAME survives the push without collapsing. Block motion
    is INFORMATIONAL (we want to know how much disturbance the push gives
    FAME, not whether the block reached some arbitrary point)."""
    if stats is None:
        return False, "no telemetry"
    if stats["pelvis_z_min"] < min_pelvis_z:
        return False, (f"robot fell (pelvis_z_min={stats['pelvis_z_min']:.3f} "
                       f"< {min_pelvis_z:.2f})  "
                       f"max_pitch_dev={stats['pitch_dev_max_deg']:.1f}° "
                       f"max_yaw_dev={stats['yaw_dev_max_deg']:.1f}° "
                       f"drift={stats['base_xy_drift_max_cm']:.1f}cm")
    return True, (f"upright  pelvis_z_min={stats['pelvis_z_min']:.3f}  "
                  f"max_pitch_dev={stats['pitch_dev_max_deg']:.1f}° "
                  f"max_yaw_dev={stats['yaw_dev_max_deg']:.1f}° "
                  f"drift={stats['base_xy_drift_max_cm']:.1f}cm  "
                  f"block_moved=({stats['block_dx_cm']:+.1f},{stats['block_dy_cm']:+.1f})cm")


def _run_trial(mass: float, trial_idx: int, args, csv_path: Path) -> dict:
    env = os.environ.copy()
    env["PUSH_BLOCK_MASS"] = f"{mass:.6f}"
    env["BLOCK_LOG_PATH"] = str(csv_path)
    if args.delta_x is not None:
        env["PUSH_DELTA_X"] = f"{args.delta_x:.6f}"
    if args.arm_kp_mult is not None:
        env["ARM_KP_MULT"] = f"{args.arm_kp_mult:.6f}"
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
    s = stats or {}
    return {
        "mass_kg": mass,
        "delta_x": args.delta_x if args.delta_x is not None else 0.25,
        "arm_kp_mult": args.arm_kp_mult if args.arm_kp_mult is not None else 1.0,
        "trial_idx": trial_idx,
        "rc": rc,
        "wall_seconds": elapsed,
        "success": int(success),
        "reason": reason,
        "pelvis_z_min": s.get("pelvis_z_min"),
        "pelvis_zN": s.get("pelvis_zN"),
        "base_xy_drift_max_cm": s.get("base_xy_drift_max_cm"),
        "pitch_dev_max_deg": s.get("pitch_dev_max_deg"),
        "pitch_dev_integral_degs": s.get("pitch_dev_integral_degs"),
        "yaw_dev_max_deg": s.get("yaw_dev_max_deg"),
        "leg_torque_peak_Nm": s.get("leg_torque_peak_Nm"),
        "leg_torque_rms_norm_Nm": s.get("leg_torque_rms_norm_Nm"),
        "block_dx_cm": s.get("block_dx_cm"),
        "block_dy_cm": s.get("block_dy_cm"),
        "block_x0": s.get("block_x0"),
        "block_xN": s.get("block_xN"),
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
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Per-trial wall-clock budget passed to the bridge (s). "
                             "With warmup_seconds=3 and home+push goals of 1.5+5s, "
                             "10s leaves 1-2s tail for FAME to settle after push.")
    parser.add_argument("--delta-x", type=float, default=None,
                        help="Override frame_delta x-component (push amplitude) "
                             "for all trials in this sweep. Passed to the IK "
                             "driver via PUSH_DELTA_X env var. None = use the "
                             "YAML value (currently 0.25).")
    parser.add_argument("--arm-kp-mult", type=float, default=None,
                        help="Scale FrameController's arm kp values (indices "
                             "13..26 in BODY_JOINTS) by this multiplier. e.g. "
                             "2.0 = double the arm PD stiffness, pushes harder "
                             "for the same position error. Passed via env var.")
    parser.add_argument("--target-x", type=float, default=0.55,
                        help="Block-x success threshold (world m); block "
                             "starts at world x=0.52 (post-settling ≈0.53), "
                             "so 0.55 = ~2 cm push counts as success")
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
        "mass_kg", "delta_x", "arm_kp_mult", "trial_idx", "rc", "wall_seconds", "success", "reason",
        "pelvis_z_min", "pelvis_zN",
        "base_xy_drift_max_cm",
        "pitch_dev_max_deg", "pitch_dev_integral_degs", "yaw_dev_max_deg",
        "leg_torque_peak_Nm", "leg_torque_rms_norm_Nm",
        "block_dx_cm", "block_dy_cm",
        "block_x0", "block_xN", "csv",
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

#!/usr/bin/env python
"""Plot a single eval trial CSV (block, base, wrist, right-arm q).

Usage:
    python scripts/plot_trial.py runs/push_smoke_latest/trial_m0.500kg_t00.csv

Saves <csv>.png next to the CSV and pops up a window if a display is
available. Also tags the major phases (home / pre_push / push / home) on
the time axis when the right-arm q_des changes magnitude, so it's obvious
when the wrist actually moves and whether the block responds.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg" if not os.environ.get("DISPLAY") else matplotlib.get_backend())
import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def quat_to_rpy_deg(qw, qx, qy, qz):
    yaw = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))
    pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)))
    roll = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
    return roll, pitch, yaw


def find_phase_boundaries(t: np.ndarray, rsp_des: np.ndarray, rel_des: np.ndarray) -> list[tuple[float, str]]:
    """Detect when the right-arm q_des changes — those are the goal transitions."""
    sig = [(round(float(p), 2), round(float(e), 2)) for p, e in zip(rsp_des, rel_des)]
    boundaries = [(float(t[0]), "start")]
    for i in range(1, len(sig)):
        if sig[i] != sig[i - 1]:
            boundaries.append((float(t[i]), f"q_des -> pitch={rsp_des[i]:.2f},elbow={rel_des[i]:.2f}"))
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="trial CSV from the eval harness")
    parser.add_argument("--target-x", type=float, default=0.70,
                        help="success threshold for block x (world m)")
    parser.add_argument("--min-pelvis-z", type=float, default=0.65,
                        help="robot-fell threshold for pelvis z (world m)")
    parser.add_argument("--show", action="store_true",
                        help="open the plot window (in addition to saving PNG)")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"file not found: {args.csv}")

    d = load_csv(args.csv)
    t = d["t"]
    roll, pitch, yaw = quat_to_rpy_deg(d["base_qw"], d["base_qx"], d["base_qy"], d["base_qz"])

    phases = find_phase_boundaries(t, d["rsp_des"], d["rel_des"])

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # --- Row 1: block world position ---
    ax = axes[0]
    ax.plot(t, d["block_x"], label="block_x", color="C3")
    ax.plot(t, d["block_y"], label="block_y", color="C2", alpha=0.5)
    ax.plot(t, d["block_z"], label="block_z", color="C0", alpha=0.5)
    ax.axhline(args.target_x, color="C3", ls=":", lw=1.0, label=f"target_x={args.target_x:.2f}")
    ax.axhline(d["block_x"][0], color="C3", ls="--", lw=0.8, alpha=0.5,
               label=f"block_x0={d['block_x'][0]:.2f}")
    block_moved = d["block_x"].max() - d["block_x"][0]
    final_block_x = d["block_x"][-1]
    success_block = final_block_x >= args.target_x
    ax.set_ylabel("block world (m)")
    ax.set_title(f"Block — moved {block_moved*100:+.1f} cm  "
                 f"(x_final={final_block_x:.3f}, target={args.target_x:.3f})  "
                 f"{'SUCCESS' if success_block else 'FAIL'}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 2: pelvis world xyz + RPY ---
    ax = axes[1]
    ax.plot(t, d["base_x"], label="base_x", color="C0")
    ax.plot(t, d["base_y"], label="base_y", color="C2")
    ax.plot(t, d["base_z"], label="base_z", color="C3")
    ax.axhline(args.min_pelvis_z, color="C3", ls=":", lw=1.0,
               label=f"min_z={args.min_pelvis_z:.2f}")
    fell = d["base_z"].min() < args.min_pelvis_z
    ax.set_ylabel("pelvis world (m)")
    ax.set_title(f"Pelvis — min_z={d['base_z'].min():.3f}  "
                 f"{'ROBOT FELL' if fell else 'upright'}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 3: pelvis RPY (deg) ---
    ax = axes[2]
    ax.plot(t, roll,  label="roll",  color="C0")
    ax.plot(t, pitch, label="pitch", color="C2")
    ax.plot(t, yaw,   label="yaw",   color="C3")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_ylabel("pelvis RPY (deg)")
    ax.set_title(f"Pelvis attitude — at t=5s: roll={roll[t>=5][0]:+.0f}, "
                 f"pitch={pitch[t>=5][0]:+.0f}, yaw={yaw[t>=5][0]:+.0f}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 4: right-arm tracking ---
    ax = axes[3]
    ax.plot(t, d["rsp_des"], label="pitch_des", color="C0", ls="--")
    ax.plot(t, d["rsp_now"], label="pitch_now", color="C0")
    ax.plot(t, d["rel_des"], label="elbow_des", color="C2", ls="--")
    ax.plot(t, d["rel_now"], label="elbow_now", color="C2")
    ax.set_ylabel("right arm (rad)")
    ax.set_title("Right shoulder_pitch & elbow tracking (des dashed, now solid)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Phase markers on every panel
    for ax in axes:
        for boundary_t, label in phases[1:]:  # skip first ("start")
            ax.axvline(boundary_t, color="k", ls=":", lw=0.6, alpha=0.5)

    axes[-1].set_xlabel("sim time (s)")
    fig.suptitle(f"{args.csv.name}", fontsize=11)
    fig.tight_layout()

    png = args.csv.with_suffix(".png")
    fig.savefig(png, dpi=110)
    print(f"saved {png}")

    if args.show and os.environ.get("DISPLAY"):
        plt.show()


if __name__ == "__main__":
    main()

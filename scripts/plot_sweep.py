#!/usr/bin/env python
"""Plot the mass-sweep summary as disturbance-vs-mass curves.

Reads the summary.csv produced by eval_push_mass_sweep.py and produces a
4-panel figure showing how each mass perturbed FAME:

  1. Success rate (robot stayed upright) vs mass
  2. Max pelvis pitch deviation vs mass        — body rotation under push
  3. Max base xy drift vs mass                  — robot moved on the floor
  4. Block displacement (informational)         — sanity check on contact

Usage:
    python scripts/plot_sweep.py runs/mass_sweep/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path, x_col: str) -> dict[float, list[dict]]:
    by_x: dict[float, list[dict]] = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                m = float(r[x_col])
            except (KeyError, ValueError):
                continue
            by_x[m].append(r)
    return by_x


def agg(rows: list[dict], key: str) -> tuple[float, float, float]:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(vals)
    return a.mean(), a.min(), a.max()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--x", default="mass_kg",
                        help="Column to sweep on the x-axis (mass_kg, "
                             "delta_x, arm_kp_mult). Default: mass_kg.")
    parser.add_argument("--xlabel", default=None,
                        help="X-axis label (defaults to --x with units guessed)")
    parser.add_argument("--logx", action="store_true", default=None,
                        help="Force log x scale (default: log if all values > 0)")
    parser.add_argument("--linx", action="store_true",
                        help="Force linear x scale")
    args = parser.parse_args()

    by_mass = load(args.summary, args.x)
    masses = sorted(by_mass.keys())
    xlabel_map = {
        "mass_kg": "block mass (kg)",
        "delta_x": "IK push amplitude delta_x (m)",
        "arm_kp_mult": "arm kp multiplier",
    }
    xlabel = args.xlabel or xlabel_map.get(args.x, args.x)
    use_logx = args.logx if args.logx is not None else (not args.linx and all(m > 0 for m in masses) and (max(masses) / max(min(masses), 1e-9)) >= 10)

    panels = [
        ("success_rate",            "success rate",            "Robot survived push?  (success = pelvis_z > 0.65 m)",          "C0", False),
        ("pitch_dev_max_deg",       "max pitch deviation (°)", "Peak pelvis pitch excursion during push",                      "C2", True),
        ("pitch_dev_integral_degs", "∫|Δpitch| dt (°·s)",      "Sustained pitch deviation (area under |Δpitch| curve)",        "C5", True),
        ("base_xy_drift_max_cm",    "max xy drift (cm)",       "How far the robot's base translated during the push",          "C3", True),
        ("leg_torque_peak_Nm",      "peak leg torque (Nm)",    "Single-joint peak leg torque — hardest a leg is working",      "C6", True),
        ("leg_torque_rms_norm_Nm",  "RMS leg torque (Nm)",     "Sustained leg effort: RMS over 12 legs, integrated over push", "C7", True),
        ("block_dx_cm",             "block dx (cm)",           "Block displacement (informational — contact sanity check)",     "C4", True),
    ]

    # success rate computed separately (it's a fraction, not from agg)
    sr = []
    for m in masses:
        rs = by_mass[m]
        sr.append(sum(int(r["success"]) for r in rs) / len(rs))

    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    axes = axes.flatten()

    # Panel 0: success rate
    ax = axes[0]
    ax.plot(masses, sr, "o-", color="C0", markersize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("success rate")
    ax.set_xlabel(xlabel)
    ax.set_title("Robot survived push?  (success = pelvis_z > 0.65 m)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Panels 1..6: continuous metrics from CSV
    for slot, (key, ylabel, title, color, is_metric) in enumerate(panels[1:], start=1):
        ax = axes[slot]
        means, mins, maxes = [], [], []
        for m in masses:
            mean_v, min_v, max_v = agg(by_mass[m], key)
            means.append(mean_v); mins.append(min_v); maxes.append(max_v)
        means = np.asarray(means); mins = np.asarray(mins); maxes = np.asarray(maxes)
        if np.all(np.isnan(means)):
            ax.text(0.5, 0.5, f"no data for '{key}'", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
        else:
            ax.errorbar(masses, means, yerr=[means - mins, maxes - means],
                        fmt="o-", color=color, markersize=10, capsize=4)
        if use_logx: ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    # hide unused axes
    for slot in range(len(panels), len(axes)):
        axes[slot].set_visible(False)

    fig.suptitle(f"FAME disturbance vs {args.x}  |  {args.summary.parent.name}", fontsize=12)
    fig.tight_layout()

    png = args.summary.with_suffix(".png")
    fig.savefig(png, dpi=110)
    print(f"saved {png}")

    # Compact text table
    def col(key):
        return [agg(by_mass[m], key)[0] for m in masses]
    pitch_max_v   = col("pitch_dev_max_deg")
    pitch_int_v   = col("pitch_dev_integral_degs")
    drift_v       = col("base_xy_drift_max_cm")
    tau_peak_v    = col("leg_torque_peak_Nm")
    tau_rms_v     = col("leg_torque_rms_norm_Nm")
    block_dx_v    = col("block_dx_cm")
    print(f"\n{'mass(kg)':>9}  {'succ':>5}  {'Δpitch(°)':>9}  {'∫Δpitch(°·s)':>13}  "
          f"{'drift(cm)':>9}  {'τ_peak(Nm)':>11}  {'τ_rms(Nm)':>10}  {'blk_dx(cm)':>11}")
    for i, m in enumerate(masses):
        print(f"{m:>9.2f}  {sr[i]*100:>4.0f}%  "
              f"{pitch_max_v[i]:>9.1f}  {pitch_int_v[i]:>13.2f}  "
              f"{drift_v[i]:>9.1f}  {tau_peak_v[i]:>11.1f}  "
              f"{tau_rms_v[i]:>10.1f}  {block_dx_v[i]:>+11.2f}")


if __name__ == "__main__":
    main()

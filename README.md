# h12_fame_manipulation

FAME locomotion + FrameController IK + safety_layer (split mode) on top of a
MuJoCo sim, plus a **push-disturbance evaluation harness** (a table-and-block
scene with scripts that script the arm into the block while sweeping mass /
amplitude / arm stiffness, logging FAME's response).

```
┌──────┐ lowcmd_lower_in ─┐
│ FAME │ ───────────────► │
└──────┘                  ▼
                   ┌────────────┐  lowcmd  ┌───────────────┐
                   │ safety_layer │ ──────►│ mujoco bridge │ ──► rt/lowstate ─► (back to FAME / IK / safety)
                   └────────────┘          └───────────────┘
┌──────┐ lowcmd_upper_in ─▲
│  IK  │ ─────────────────
└──────┘
```

## File map

| file | role |
| --- | --- |
| [`h12_fame_ik_runner/mujoco_dds_bridge.py`](h12_fame_ik_runner/mujoco_dds_bridge.py) | MuJoCo sim. Publishes `rt/lowstate` at 500 Hz, subscribes merged `rt/lowcmd`, applies PD `τ = τ_ff + kp·(q_des−q) + kd·(dq_des−dq)`. Has env-var hooks (`PUSH_BLOCK_MASS`, `BLOCK_LOG_PATH`) for the eval. |
| [`h12_fame_ik_runner/fame_dds_runner.py`](h12_fame_ik_runner/fame_dds_runner.py) | FAME RMA policy → `rt/safety/lowcmd_lower_in` (legs, mode=1 only on motors 0..11). |
| [`h12_fame_ik_runner/ik_dds_driver.py`](h12_fame_ik_runner/ik_dds_driver.py) | FrameController driver. Batch (YAML `ik.goals`) or interactive REPL. Supports `name:`, `frame:`, `frame_delta:`, `q_reduced:` goal types. Honors `PUSH_DELTA_X` and `ARM_KP_MULT` env vars for eval sweeps. |
| [`h12_fame_ik_runner/orchestrator.py`](h12_fame_ik_runner/orchestrator.py) | Spawns the 4 processes (safety, bridge, FAME, IK); Ctrl-C tears them down. |
| [`scripts/eval_push_mass_sweep.py`](scripts/eval_push_mass_sweep.py) | Push-eval driver. Loops trials, sets env vars per trial, parses CSV, writes `summary.csv` with disturbance metrics. |
| [`scripts/plot_trial.py`](scripts/plot_trial.py) | Per-trial 4-panel plot (block xyz, pelvis xyz, pelvis RPY, arm tracking). |
| [`scripts/plot_sweep.py`](scripts/plot_sweep.py) | Per-sweep 7-panel plot. `--x mass_kg` / `--x delta_x` / `--x arm_kp_mult`. |
| [`scenes/h1_2_table.xml`](scenes/h1_2_table.xml) | H1-2 + workbench table + 30×30×15 cm push block (45° rotated about z, freejoint). |
| [`configs/h1_2_fame_ik_eval.yaml`](configs/h1_2_fame_ik_eval.yaml) | Eval runner config: biased-home q_reduced → `frame_delta` push, no retract, `height_cmd=0.90`. |

## Prerequisites

Sibling repos `h12_adaptive_policy`, `h12_safety_layer`, `h12_ros2_controller`
cloned with submodules. Runner uses the `uv` env in adaptive_policy:

```bash
cd src/h12_adaptive_policy && uv sync
# colcon overlay must expose h12_safety_layer + h12_ros2_controller; source the
# workspace's install/setup.bash if a fresh shell can't find them.
```

FAME weights: `src/h12_adaptive_policy/data/rma_hand/{policy.pt,encoder_3999.pt}`.

## Quick start (control stack only)

```bash
# Headless smoke (20 s, all 4 processes, no viewer):
./scripts/headless_smoke.sh configs/h1_2_fame_ik.yaml

# Live viewer:
./scripts/viewer.sh configs/h1_2_fame_ik.yaml

# Just bridge+FAME, arms held at warmup defaults (debug):
python -m h12_fame_ik_runner.orchestrator --config configs/h1_2_fame_ik.yaml --no-ik --headless --duration 15
```

Sim-only configs ([sim_safety_split.yaml](configs/sim_safety_split.yaml),
[sim_split_controller.yaml](configs/sim_split_controller.yaml)) relax estop
ranges so FAME warm-up transients don't trip the safety thread.
**Don't ship these to hardware.**

## Push-disturbance evaluation

### What it tests

The arm is scripted through a brief "biased home → straight-line forward IK
push" trajectory while the block sits in the wrist's path. Across each
trial we log block + base + wrist pose and the 12 leg-actuator torques,
then aggregate disturbance metrics per trial. Sweep any of:

* **block mass** via `PUSH_BLOCK_MASS=<kg>` (bridge rescales `body_mass` + `body_inertia` at model load).
* **IK push amplitude** via `PUSH_DELTA_X=<m>` (overrides the x-component of any `frame_delta` goal).
* **arm PD stiffness** via `ARM_KP_MULT=<float>` (scales arm kp 13..26 in the controller config before `FrameController` init).

### One trial

```bash
PUSH_BLOCK_MASS=0.5 BLOCK_LOG_PATH=$(pwd)/runs/one.csv \
  python -m h12_fame_ik_runner.orchestrator \
  --config configs/h1_2_fame_ik_eval.yaml --headless --duration 10

python scripts/plot_trial.py runs/one.csv          # -> runs/one.png
```

### Mass sweep

```bash
python scripts/eval_push_mass_sweep.py \
  --masses 0.1 0.5 1.0 2.0 5.0 10 20 50 --trials 3 --out runs/mass_sweep
python scripts/plot_sweep.py runs/mass_sweep/summary.csv --x mass_kg
```

### Amplitude / arm-kp sweep

```bash
# amplitude
for dx in 0.25 0.40 0.60 0.80; do
  python scripts/eval_push_mass_sweep.py --masses 5.0 --trials 3 \
    --delta-x $dx --out runs/sweeps/amp_dx${dx}_m5
done

# arm kp
for k in 1.0 2.0 4.0 8.0; do
  python scripts/eval_push_mass_sweep.py --masses 5.0 --trials 3 \
    --arm-kp-mult $k --out runs/sweeps/kp_x${k}_m5
done

# concatenate the summary.csv files per axis, then plot:
python scripts/plot_sweep.py runs/sweeps/amplitude_sweep.csv --x delta_x --linx
python scripts/plot_sweep.py runs/sweeps/armkp_sweep.csv  --x arm_kp_mult --linx
```

### Metrics

* **success** = `pelvis_z_min > 0.65 m` over the push window. FAME is
  considered to have survived the disturbance.
* Continuous: `pitch_dev_max_deg`, `pitch_dev_integral_degs` (∫\|Δpitch\| dt
  — sustained tilt), `yaw_dev_max_deg`, `base_xy_drift_max_cm`,
  `leg_torque_peak_Nm`, `leg_torque_rms_norm_Nm`, `block_dx_cm`.

### Findings

* FAME is **100% upright** across all sweeps tried (mass 0.1–50 kg,
  delta_x 0.25–0.80, arm kp ×1–×8).
* **Amplitude and arm-kp sweeps are flat on every metric** — increasing
  either does not raise contact force.
* Root cause: the IK is **velocity-limited** by `dq_lim: 1.0` rad/s in
  [sim_split_controller.yaml](configs/sim_split_controller.yaml).
  `limit_joint_vel` clamps every joint, so the arm sweeps at the same
  velocity regardless of target distance or PD stiffness, and peak contact
  force is bounded by `v × arm_inertia`.
* Mass-dependent block dx (1.7 cm at 0.1 kg → 0.02 cm at 50 kg) confirms
  contact is happening, just at a velocity-limited magnitude.
* Next axis worth sweeping: `dq_lim` itself — needs an env-var hook in the
  controller config loader (analogous to `ARM_KP_MULT`).

## Known limitations

* FAME's training distribution assumes arms at `default_angles_arms`; any
  IK pose puts the policy off-distribution and the body sits in a tilted
  crouch (pitch ≈ −15°, yaw ≈ +21°) instead of upright. The eval was
  designed around this resting attitude rather than trying to fix it.
* `tau_est` reported via DDS is MuJoCo's actuator force, not friction-aware
  measured torque.
* Magpie / eFlesh gripper actuators stay passive at midrange — out of
  scope for both the locomotion stack and the push eval.

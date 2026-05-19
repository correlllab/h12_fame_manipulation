# h12_fame_ik_runner

Thin orchestration layer that runs **FAME** (the RMA locomotion policy from
[h12_adaptive_policy](../h12_adaptive_policy)) together with the
**FrameController** upper-body IK from
[h12_ros2_controller](../h12_ros2_controller) on top of a Mujoco simulation,
relaying both through the
[h12_safety_layer](../h12_safety_layer) in **split mode**.

Because the integration touches three repos that should not depend on each
other, the runner lives in its own folder. It only imports the three packages
(plus `unitree_sdk2py`) and adds:

| file | role |
| --- | --- |
| [`h12_fame_ik_runner/mujoco_dds_bridge.py`](h12_fame_ik_runner/mujoco_dds_bridge.py) | Mujoco sim. Publishes `rt/lowstate` at 500 Hz, subscribes the merged `rt/lowcmd`, applies `τ = τ_ff + kp·(q_des−q) + kd·(dq_des−dq)` per actuator. Supports headless or `mujoco.viewer` modes. |
| [`h12_fame_ik_runner/fame_dds_runner.py`](h12_fame_ik_runner/fame_dds_runner.py) | FAME RMA policy. Subscribes `rt/lowstate`, runs encoder+policy at 50 Hz, publishes a 27-motor LowCmd with `mode=1` only for legs (0..11) to `rt/safety/lowcmd_lower_in`. |
| [`h12_fame_ik_runner/ik_dds_driver.py`](h12_fame_ik_runner/ik_dds_driver.py) | FrameController driver. Two modes: `--mode batch` (default — steps through YAML `ik.goals`, used by the orchestrator) and `--mode interactive` (a stdin REPL like `frame_task_client`: send named configs or 6-DOF frame targets). Publishes to `rt/safety/lowcmd_upper_in`. |
| [`h12_fame_ik_runner/orchestrator.py`](h12_fame_ik_runner/orchestrator.py) | Spawns the four processes (safety, bridge, FAME, IK) in dedicated process groups; tears them down cleanly on Ctrl-C. |

```
┌──────────┐  rt/safety/lowcmd_lower_in ─┐
│   FAME   │ ────────────────────────►   │
└──────────┘                             ▼
                                  ┌────────────────┐  rt/lowcmd  ┌──────────────┐
                                  │ safety_layer   │ ──────────► │  Mujoco DDS  │
                                  │ (split_mode)   │             │   bridge     │
                                  └────────────────┘             └─────┬────────┘
┌──────────┐  rt/safety/lowcmd_upper_in ─▲                             │ rt/lowstate
│   IK     │ ────────────────────────────                              ▼
└──────────┘                                                    (back to FAME / IK / safety)
```

The safety_layer’s split-mode merge takes motor_cmd[0:12] from FAME and
motor_cmd[12:27] from the FrameController — exactly the contract documented in
[`h12_safety_layer/docs/architecture.md`](../h12_safety_layer/docs/architecture.md).

---

## Prerequisites

1. The three sibling repos must be cloned at `src/h12_adaptive_policy`,
   `src/h12_safety_layer`, and `src/h12_ros2_controller`, with their
   submodules initialised:

   ```bash
   for d in h12_adaptive_policy h12_safety_layer h12_ros2_controller; do
     (cd "$d" && git submodule update --init --recursive)
   done
   ```

2. The runner uses the adaptive_policy `uv` environment (mujoco, torch,
   unitree_sdk2py are already pinned there). Create it once:

   ```bash
   cd src/h12_adaptive_policy && uv sync
   ```

3. The colcon overlay (`/home/humanoid/ws_ctrl/install/…` here) needs to expose
   `h12_safety_layer` and `h12_ros2_controller` to the venv. If a fresh shell
   does not pick them up, `source install/setup.bash` from the workspace root
   once, or add them to `PYTHONPATH`.

4. Make sure the FAME policy weights exist:
   `src/h12_adaptive_policy/data/rma_hand/{policy.pt,encoder_3999.pt}`.

A quick sanity check that everything is importable from the venv:

```bash
src/h12_adaptive_policy/.venv/bin/python - <<'PY'
import mujoco, torch, unitree_sdk2py, h12_safety_layer
from h12_ros2_controller.core.controller.frame_controller import FrameController
print("ok")
PY
```

---

## Headless smoke test (do this first)

Run the entire stack in one shot, no viewer, 20 s duration:

```bash
cd src/h12_fame_ik_runner
./scripts/headless_smoke.sh                            # default: bare H1-2
./scripts/headless_smoke.sh configs/h1_2_fame_ik.yaml  # explicit
```

Equivalent direct invocation:

```bash
../h12_adaptive_policy/.venv/bin/python -m h12_fame_ik_runner.orchestrator \
    --config configs/h1_2_fame_ik.yaml --headless --duration 20
```

What to look for in the output:

* `[safety] {"event": "relay_started", "mode": "split_mode"}` — safety layer
  came up in split mode.
* `[fame] loaded encoder …/encoder_3999.pt` and
  `[fame] low_state=rt/lowstate low_cmd_out=rt/safety/lowcmd_lower_in` —
  FAME is subscribed and publishing.
* `[ik] All joints are locked in the initial position.` and `[ik] goal=home …`
  — FrameController initialised and started commanding the home pose.
* `[bridge] summary:` — printed when the bridge stops. The interesting fields:

  ```
  low_cmd_msgs_received=4844 (first at t+0.00s)   # 500 Hz from safety_layer
  q_des[0:12] = …                                  # FAME-driven leg targets
  q_des[13:20] = [0]*7                             # IK target = "home"
  q_now[13:20] = small values                      # arms tracking toward 0
  ```

  Successful integration looks like: `q_des[0:12]` deviates from the YAML
  defaults (proves FAME is producing non-trivial actions) AND `q_now[13:27]`
  approaches zero (proves IK is overlaying its targets on top of FAME).

> **Note on balance:** the RMA policy was trained with the arms held at
> `default_angles_arms`. When the IK driver pulls the arms to a different
> pose, FAME's input distribution shifts, so the simulated robot may not stay
> upright. The point of the smoke test is to verify the control pipeline
> (FAME → safety → mujoco; IK → safety → mujoco), **not** to demonstrate
> stable walking with arbitrary arm postures.

---

## Live Mujoco viewer

```bash
cd src/h12_fame_ik_runner
./scripts/viewer.sh                            # default: bare H1-2
./scripts/viewer.sh configs/h1_2_fame_ik.yaml  # explicit
```

Equivalent:

```bash
../h12_adaptive_policy/.venv/bin/python -m h12_fame_ik_runner.orchestrator \
    --config configs/h1_2_fame_ik.yaml --viewer
```

The mujoco passive viewer opens on the main thread of the bridge process. Ctrl-C
in the terminal tears down all four processes via SIGINT to each process group.

Tips while the viewer is up:

* `[bridge] summary:` is only printed at shutdown. To watch live, add a print
  inside `_apply_pd_to_ctrl` (the simplest spot is right before `_d.ctrl[: …]`)
  for the joints you care about.
* The viewer respects mouse drag / scroll to orbit and zoom. Press `Esc` (or
  close the window) to stop just the bridge — the orchestrator will then SIGINT
  the other three.

---

## Sending positions interactively (REPL)

When you want to drive the arms manually — same UX as `ros2 run
h12_ros2_controller frame_task_client`, but going through pure DDS — start
the stack **without** the batch IK driver, then run the REPL in its own
terminal.

```bash
# Terminal A — the stack minus IK
cd src/h12_fame_ik_runner
./scripts/viewer.sh configs/h1_2_fame_ik.yaml  # or headless_smoke.sh
# (or, more explicit:)
../h12_adaptive_policy/.venv/bin/python -m h12_fame_ik_runner.orchestrator \
    --config configs/h1_2_fame_ik.yaml --viewer --no-ik

# Terminal B — REPL
./scripts/ik_repl.sh configs/h1_2_fame_ik.yaml
```

You'll get a prompt:

```
ik> help
[ik] commands
  show                              - print current left/right wrist poses
  <named_config>                    - run a NAMED_CONFIGS entry (['home'])
  left  x y z  R P Y                - target left wrist; xyz in m, RPY in deg
  right x y z  R P Y                - target right wrist; xyz in m, RPY in deg
  frame <link_name>  x y z  R P Y   - target an arbitrary link
  timeout <seconds>                 - max time per goal (default 6 s)
  help                              - print this help
  q / quit                          - leave the REPL (stack keeps running)
```

A typical session:

```
ik> show
[ik] left_wrist  xyz=(+0.466,+0.116,+0.301) rpy_deg=(+5.4,-21.0,-14.8)
[ik] right_wrist xyz=(+0.442,-0.040,+0.292) rpy_deg=(+2.2,-19.9,+19.8)
ik> left 0.30 0.25 0.50  0 0 0
[ik] driving frame_task on left_wrist_yaw_link; xyz=(+0.300,+0.250,+0.500) ...
[ik]   linear=0.2942 m  angular=0.4080 rad
[ik]   linear=0.0437 m  angular=0.0345 rad
[ik]   target reached
ik> right 0.30 -0.25 0.50  0 0 0
ik> frame left_wrist_yaw_link  0.40 0.10 0.45  0 30 0
ik> home
ik> q
```

Notes:

- **RPY is in degrees** at the REPL (converted to radians internally), matching
  the `frame_controller_goto.py` example. xyz is in metres, world frame.
- `left` / `right` are short aliases for `left_wrist_yaw_link` /
  `right_wrist_yaw_link`. Use `frame <name> ...` for anything else (e.g.
  `left_elbow_link`).
- Convergence prints linear and angular errors every 0.5 s. The driver stops
  the inner loop early when both errors fall below the thresholds, then runs
  ~50 extra steps to let the publisher latch the final command.
- `q` only exits the REPL — the FAME / safety / bridge processes started in
  Terminal A keep running. Ctrl-C those when you're done.
- Pink's IK can emit a `RuntimeWarning: invalid value encountered in sqrt`
  from `acceleration_limit.py` near singular configurations; it's benign and
  the IK recovers on the next step.

If you'd rather have everything in one terminal, the orchestrator can launch
the IK driver in interactive mode too, but stdin will fight with the prefixed
log streams — running the REPL separately is the cleaner UX.

---

## Magpie gripper variants

Three mujoco scenes ship in `h12_adaptive_policy/h1_2/`:

| scene file | gripper |
| --- | --- |
| `scene.xml`                       | none (default) |
| `h1_2_magpie_fame.xml`            | Magpie |
| `h1_2_magpie_eflesh_fame.xml`     | Magpie + eFlesh pads |

Each has a matching runner config. The orchestrator behaves identically;
just point it at the right YAML:

```bash
# Bare H1-2 (no gripper)
./scripts/viewer.sh configs/h1_2_fame_ik.yaml
./scripts/headless_smoke.sh configs/h1_2_fame_ik.yaml 30

# H1-2 + Magpie gripper
./scripts/viewer.sh configs/h1_2_fame_ik_magpie.yaml
./scripts/headless_smoke.sh configs/h1_2_fame_ik_magpie.yaml 30

# H1-2 + Magpie + eFlesh
./scripts/viewer.sh configs/h1_2_fame_ik_magpie_eflesh.yaml
./scripts/headless_smoke.sh configs/h1_2_fame_ik_magpie_eflesh.yaml 30
```

The bridge always drives the **first 27 actuators** as the H1-2 motors and
holds any **extra actuators** (Magpie grippers, eFlesh) at the midpoint of
their `ctrlrange` so the additional DOFs stay passive while you exercise the
locomotion + IK stack. If you want to actively drive the gripper:

1. Add a publisher for the gripper command topic (the Magpie node already
   exists in [magpie_control](../magpie_control)).
2. Or extend `mujoco_dds_bridge._apply_pd_to_ctrl` to subscribe a gripper
   command topic and write directly to `d.ctrl[27:]`.

Neither is needed for the FAME + IK verification this runner is built for.

---

## Configuration files

Each runner YAML carries four sections — `mujoco`, `robot`, `topics`, `fame`,
`ik`. The fields are documented inline in
[configs/h1_2_fame_ik.yaml](configs/h1_2_fame_ik.yaml). The three runner
configs differ only in `mujoco.xml_path` (and could share a common base if
you add an include mechanism later).

The two **sim-only** safety configs are the load-bearing tweak that lets the
stack survive the FAME warm-up window:

* [`configs/sim_safety_split.yaml`](configs/sim_safety_split.yaml) is loaded
  by the safety_layer. Its `limits.estop.position_offset = -3.0` (i.e. expand
  the URDF range by 3 rad on both sides) plus very high velocity/torque
  ratios make estop a no-op while keeping the split-mode merge active.
* [`configs/sim_split_controller.yaml`](configs/sim_split_controller.yaml) is
  loaded by the IK driver's `FrameController`. Same idea applied to the
  controller's internal upper-body safety thread.

**Both are wrong for the real robot.** Use the original
`default_safety_split.yaml` / `safety_split.yaml` from the sibling repos for
hardware deploys; this runner just picks them at the orchestrator's
`--safety-config` and the YAML's `ik.controller_config` fields.

---

## Running individual processes (for debugging)

You can run each component on its own to localise issues. Each uses the same
runner YAML to keep topics/paths consistent.

```bash
# Terminal 1 — safety layer in split mode
src/h12_adaptive_policy/.venv/bin/python \
  src/h12_safety_layer/h12_safety_layer/script/safety_layer_main.py \
  --config src/h12_fame_ik_runner/configs/sim_safety_split.yaml

# Terminal 2 — mujoco bridge (DDS) [headless or with --viewer]
src/h12_adaptive_policy/.venv/bin/python -m h12_fame_ik_runner.mujoco_dds_bridge \
  --config configs/h1_2_fame_ik.yaml --headless

# Terminal 3 — FAME publisher
src/h12_adaptive_policy/.venv/bin/python -m h12_fame_ik_runner.fame_dds_runner \
  --config configs/h1_2_fame_ik.yaml

# Terminal 4 — IK driver (batch goals from YAML)
src/h12_adaptive_policy/.venv/bin/python -m h12_fame_ik_runner.ik_dds_driver \
  --config configs/h1_2_fame_ik.yaml

# …or interactive (stdin REPL). See "Sending positions interactively" above.
src/h12_adaptive_policy/.venv/bin/python -m h12_fame_ik_runner.ik_dds_driver \
  --config configs/h1_2_fame_ik.yaml --interactive
```

You can also drop the IK or FAME process to bisect — the orchestrator exposes
`--no-fame` / `--no-ik` shortcuts.

```bash
# Bridge + safety + FAME only (legs walking, arms held at warmup defaults)
python -m h12_fame_ik_runner.orchestrator --config configs/h1_2_fame_ik.yaml \
    --headless --duration 20 --no-ik

# Bridge + safety + IK only (arms move to home, legs sag because no FAME)
python -m h12_fame_ik_runner.orchestrator --config configs/h1_2_fame_ik.yaml \
    --headless --duration 20 --no-fame
```

---

## Known limitations

* FAME walks poorly when the arms diverge from `default_angles_arms`. Not a
  bug in this runner — it is a domain-shift in the policy's inputs.
* The bridge's `tau_est` field is the actuator force from Mujoco, not the
  measured-torque-after-friction the real H1-2 reports. Consumers that base
  decisions on `tau_est` should be aware.
* The Magpie gripper (and eFlesh pads) are kept passive at the centre of their
  ctrl range. Active gripper control needs the Magpie node — out of scope here.
* `NAMED_CONFIGS` currently only declares `"home"`; richer pose sequences
  require either extending that dict in the controller repo, or replacing the
  IK driver with one that uses `frame_controller.add_frame_task(...)` for
  6-DOF wrist targets.
* The sim configs disable safety estop. Do **not** copy them onto hardware.

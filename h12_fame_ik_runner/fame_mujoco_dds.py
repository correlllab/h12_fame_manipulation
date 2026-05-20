"""Unified FAME + Mujoco + DDS — the simpler setup.

This is a leaner alternative to running the separate `mujoco_dds_bridge` +
`fame_dds_runner` processes. One Python process does ALL of:

  * loads Mujoco and steps the sim at 500 Hz with a viewer (or headless),
  * runs FAME's RMA encoder + policy at 50 Hz for the legs,
  * applies local leg PD from the policy output to `data.ctrl[0:12]`,
  * applies local arm PD; the arm targets come either from `default_angles_arms`
    (warm-up) or from the merged `rt/lowcmd` once an upper-body publisher
    (`arm_controller_goto.py` or `arm_joint_stream.py`) is online,
  * publishes `rt/lowstate` so the upper-body controller can read state,
  * publishes the 27-motor leg LowCmd to `rt/safety/lowcmd_lower_in` so the
    safety_layer in split mode has something to merge.

What this script does NOT do:
  * It does not subscribe to `rt/lowcmd` for legs — those come from the local
    policy. Reading them back through the safety_layer would just add latency.
  * It does not act as a "passive" bridge that re-applies the merged command
    blindly; in this topology the safety_layer is mostly a recorder/clipper
    for legs.

If you want the strictly DDS-faithful path where Mujoco only applies whatever
`rt/lowcmd` says, use the orchestrator's `mujoco_dds_bridge` instead.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import mujoco

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as LowCmdDefault,
    unitree_hg_msg_dds__LowState_ as LowStateDefault,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

NUM_H12_MOTORS = 27
NUM_LEG_MOTORS = 12
RMA_LATENT_DIM = 8

# Make RMA importable as a top-level package (h12_adaptive_policy ships RMA as
# a namespace under its inner directory; mirror the deploy script's trick).
_SRC_DIR = Path(__file__).resolve().parents[2]
_ADAPTIVE_INNER = _SRC_DIR / "h12_adaptive_policy" / "h12_adaptive_policy"
if _ADAPTIVE_INNER.is_dir() and str(_ADAPTIVE_INNER) not in sys.path:
    sys.path.insert(0, str(_ADAPTIVE_INNER))


def _resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _quat_rotate_inverse(quat_wxyz, v):
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    cx, cy, cz = -x, -y, -z
    a = w * w
    return np.array([
        v[0] * (a + cx * cx - cy * cy - cz * cz)
        + 2.0 * v[1] * (cx * cy - w * cz)
        + 2.0 * v[2] * (cx * cz + w * cy),
        2.0 * v[0] * (cx * cy + w * cz)
        + v[1] * (a - cx * cx + cy * cy - cz * cz)
        + 2.0 * v[2] * (cy * cz - w * cx),
        2.0 * v[0] * (cx * cz - w * cy)
        + 2.0 * v[1] * (cy * cz + w * cx)
        + v[2] * (a - cx * cx - cy * cy + cz * cz),
    ], dtype=np.float32)


def _gravity_orientation(quat_wxyz):
    return _quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))


class FameMujocoDDS:
    def __init__(self, config_path: Path, headless: bool | None, duration: float | None):
        self._config_path = config_path
        with config_path.open("r", encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh)
        cfg_dir = config_path.parent

        # ----- mujoco
        mj_cfg = self._cfg.get("mujoco", {})
        self._xml_path = _resolve_path(mj_cfg["xml_path"], cfg_dir)
        self._sim_dt = float(mj_cfg.get("simulation_dt", 0.002))
        self._headless = headless if headless is not None else bool(mj_cfg.get("headless", False))
        self._duration = float(duration if duration is not None else mj_cfg.get("duration", 0.0))

        self._m = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._d = mujoco.MjData(self._m)
        self._m.opt.timestep = self._sim_dt

        # First 27 actuators correspond to the H1-2 motors (verified against
        # scene.xml). Extra actuators (Magpie etc.) get held at midpoint.
        if self._d.ctrl.shape[0] < NUM_H12_MOTORS:
            raise ValueError(f"Model exposes {self._d.ctrl.shape[0]} actuators; need {NUM_H12_MOTORS}+")
        self._h12_joint_ids = self._m.actuator_trnid[:NUM_H12_MOTORS, 0].astype(np.int32)
        self._h12_qpos_adr = self._m.jnt_qposadr[self._h12_joint_ids].astype(np.int32)
        self._h12_qvel_adr = self._m.jnt_dofadr[self._h12_joint_ids].astype(np.int32)
        self._leg_qpos_adr = self._h12_qpos_adr[:NUM_LEG_MOTORS]
        self._leg_qvel_adr = self._h12_qvel_adr[:NUM_LEG_MOTORS]
        self._arm_qpos_adr = self._h12_qpos_adr[NUM_LEG_MOTORS:]
        self._arm_qvel_adr = self._h12_qvel_adr[NUM_LEG_MOTORS:]
        if self._d.ctrl.shape[0] > NUM_H12_MOTORS:
            self._extra_ctrl_range = self._m.actuator_ctrlrange[NUM_H12_MOTORS:].copy()
        else:
            self._extra_ctrl_range = None

        # ----- robot defaults / gains
        robot = self._cfg.get("robot", {})
        self._default_q = np.asarray(robot["default_q"], dtype=np.float32)
        self._default_kp = np.asarray(robot["default_kp"], dtype=np.float32)
        self._default_kd = np.asarray(robot["default_kd"], dtype=np.float32)

        # Seed the sim at the standing default so the legs do not collapse
        # before the policy warms up.
        for i, adr in enumerate(self._h12_qpos_adr):
            self._d.qpos[adr] = float(self._default_q[i])

        # ----- FAME
        fame = self._cfg.get("fame", {})
        self._policy_path = _resolve_path(fame["policy_path"], cfg_dir)
        encoder_cfg = fame.get("encoder_path")
        self._encoder_path = _resolve_path(encoder_cfg, cfg_dir) if encoder_cfg else None
        self._policy_joints = int(fame.get("policy_num_joints", NUM_H12_MOTORS))
        self._num_actions = int(fame.get("num_actions", NUM_LEG_MOTORS))
        self._single_obs_dim = int(fame.get("single_obs_dim", 76))
        self._num_obs = int(fame.get("num_obs", 252))
        self._obs_history_len = int(fame.get("obs_history_len", 3))
        self._ang_vel_scale = float(fame.get("ang_vel_scale", 0.25))
        self._dof_pos_scale = float(fame.get("dof_pos_scale", 1.0))
        self._dof_vel_scale = float(fame.get("dof_vel_scale", 0.05))
        self._action_scale = float(fame.get("action_scale", 0.25))
        self._cmd_scale = np.asarray(fame.get("cmd_scale", [2.0, 2.0, 0.25]), dtype=np.float32)
        self._cmd = np.asarray(fame.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._height_cmd = float(fame.get("height_cmd", 0.75))
        self._control_decimation = int(fame.get("control_decimation", 10))
        self._publish_hz = float(fame.get("publish_hz", 500.0))
        self._no_encode = bool(fame.get("no_encode", False))
        self._default_legs = np.asarray(fame["default_angles_legs"], dtype=np.float32)
        self._default_arms = np.asarray(fame["default_angles_arms"], dtype=np.float32)
        self._legs_kp = np.asarray(fame["legs_kp"], dtype=np.float32)
        self._legs_kd = np.asarray(fame["legs_kd"], dtype=np.float32)
        self._left_force = np.asarray(fame.get("left_hand_force", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._right_force = np.asarray(fame.get("right_hand_force", [0.0, 0.0, 0.0]), dtype=np.float32)

        self._policy = torch.jit.load(str(self._policy_path))
        self._policy.eval()
        self._encoder = None
        if self._encoder_path is not None and self._encoder_path.is_file():
            from RMA.rma_modules.env_factor_encoder import (  # type: ignore
                EnvFactorEncoder, EnvFactorEncoderCfg,
            )
            self._encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
            self._encoder.load_state_dict(
                torch.load(str(self._encoder_path), map_location="cpu", weights_only=True)
            )
            self._encoder.eval()
            print(f"[fame_mj] loaded encoder {self._encoder_path}", flush=True)
        else:
            print("[fame_mj] no encoder; z_t will be zeros", flush=True)

        # ----- DDS
        topics = self._cfg.get("topics", {})
        net = self._cfg.get("network", {})
        self._low_state_topic = topics.get("low_state", "rt/lowstate")
        # We pub to the safety layer split-mode lower topic. Mujoco's local
        # PD ignores the safety_layer's merged output for legs — that's the
        # whole point of running them in-process.
        self._lower_in_topic = topics.get("low_cmd_lower_in", "rt/safety/lowcmd_lower_in")
        # Arms are pulled from the merged rt/lowcmd so safety_layer clipping
        # is exercised even though only upper-body cmds reach us.
        self._lowcmd_topic = topics.get("low_cmd_out", "rt/lowcmd")
        self._domain_id = int(net.get("domain_id", 0))
        interface = net.get("interface", "")
        self._interface = interface if interface else None

        if self._interface:
            ChannelFactoryInitialize(self._domain_id, self._interface)
        else:
            ChannelFactoryInitialize(self._domain_id)
        self._crc = CRC()
        self._state_pub = ChannelPublisher(self._low_state_topic, LowState_)
        self._state_pub.Init()
        self._lower_pub = ChannelPublisher(self._lower_in_topic, LowCmd_)
        self._lower_pub.Init()
        self._lower_cmd = LowCmdDefault()
        self._lower_cmd.mode_machine = 6
        for i in range(NUM_LEG_MOTORS):
            mc = self._lower_cmd.motor_cmd[i]
            mc.mode = 1
            mc.kp = float(self._legs_kp[i])
            mc.kd = float(self._legs_kd[i])
            mc.q = float(self._default_legs[i])
        # Upper-body slots (12..26) stay at mode=0; safety_layer drops them
        # in split mode.

        self._low_state = LowStateDefault()
        self._low_state.mode_machine = 6

        # Latched arm targets, fed from rt/lowcmd subscriber.
        self._arm_lock = threading.Lock()
        self._arm_q_des = self._default_arms.copy()
        self._arm_dq_des = np.zeros(NUM_H12_MOTORS - NUM_LEG_MOTORS, dtype=np.float32)
        self._arm_tau_ff = np.zeros(NUM_H12_MOTORS - NUM_LEG_MOTORS, dtype=np.float32)
        self._arm_kp = self._default_kp[NUM_LEG_MOTORS:].copy()
        self._arm_kd = self._default_kd[NUM_LEG_MOTORS:].copy()
        self._first_arm_cmd_time = 0.0

        self._lowcmd_sub = ChannelSubscriber(self._lowcmd_topic, LowCmd_)
        self._lowcmd_sub.Init(self._on_low_cmd, 10)

        # External hand forces (RMA encoder input + applied in Mujoco)
        self._left_wrist_id = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
        self._right_wrist_id = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        self._apply_forces = self._left_wrist_id >= 0 and self._right_wrist_id >= 0

        # Policy state
        self._action = np.zeros(self._num_actions, dtype=np.float32)
        self._target_leg_q = self._default_legs.copy()
        self._obs_history = collections.deque(
            [np.zeros(self._single_obs_dim, dtype=np.float32) for _ in range(self._obs_history_len)],
            maxlen=self._obs_history_len,
        )
        self._z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
        self._counter = 0
        self._tick = 0
        self._running = False

        # State and lower-cmd publishes both happen inside the main step loop
        # (right after `mj_step`) so MjData is only touched from one thread —
        # MuJoCo's d is NOT safe for concurrent access. The rates below are
        # gated by step counters; at the default sim_dt=2ms the state pub
        # naturally lands at 500 Hz with decim==1.
        self._state_pub_decim = max(1, int(round(1.0 / (500.0 * self._sim_dt))))
        self._lower_pub_decim = max(1, int(round(1.0 / (self._publish_hz * self._sim_dt))))

    # ------------------------------------------------------------ DDS plumbing
    def _on_low_cmd(self, msg: LowCmd_) -> None:
        any_active = False
        with self._arm_lock:
            for j in range(NUM_LEG_MOTORS, NUM_H12_MOTORS):
                mc = msg.motor_cmd[j]
                if int(mc.mode) != 1:
                    continue
                any_active = True
                k = j - NUM_LEG_MOTORS
                self._arm_q_des[k] = float(mc.q)
                self._arm_dq_des[k] = float(mc.dq)
                self._arm_tau_ff[k] = float(mc.tau)
                self._arm_kp[k] = float(mc.kp)
                self._arm_kd[k] = float(mc.kd)
            if any_active and self._first_arm_cmd_time == 0.0:
                self._first_arm_cmd_time = time.time()

    def _fill_and_publish_state(self) -> None:
        q = self._d.qpos[self._h12_qpos_adr]
        dq = self._d.qvel[self._h12_qvel_adr]
        quat = self._d.qpos[3:7]
        omega_body = _quat_rotate_inverse(quat, self._d.qvel[3:6])
        accel_body = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -9.81], dtype=np.float32))

        msg = self._low_state
        self._tick = (self._tick + 1) & 0xFFFFFFFF
        msg.tick = self._tick
        msg.imu_state.quaternion = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        msg.imu_state.gyroscope = [float(omega_body[0]), float(omega_body[1]), float(omega_body[2])]
        msg.imu_state.accelerometer = [float(accel_body[0]), float(accel_body[1]), float(accel_body[2])]
        for i in range(NUM_H12_MOTORS):
            ms = msg.motor_state[i]
            ms.mode = 1
            ms.q = float(q[i])
            ms.dq = float(dq[i])
            ms.ddq = 0.0
            ms.tau_est = float(self._d.actuator_force[i]) if self._d.actuator_force.shape[0] > i else 0.0
        msg.crc = self._crc.Crc(msg)
        self._state_pub.Write(msg)

    def _publish_lower_cmd(self) -> None:
        for i in range(NUM_LEG_MOTORS):
            mc = self._lower_cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = float(self._target_leg_q[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self._legs_kp[i])
            mc.kd = float(self._legs_kd[i])
        self._lower_cmd.crc = self._crc.Crc(self._lower_cmd)
        self._lower_pub.Write(self._lower_cmd)

    # ------------------------------------------------------------ control / sim
    def _pd(self, q_des, q, kp, dq_des, dq, kd, tau_ff=None):
        tau = kp * (q_des - q) + kd * (dq_des - dq)
        if tau_ff is not None:
            tau = tau + tau_ff
        return np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_obs(self):
        qj_h12 = self._d.qpos[self._h12_qpos_adr].copy()
        dqj_h12 = self._d.qvel[self._h12_qvel_adr].copy()
        full_default = np.concatenate([self._default_legs, self._default_arms], dtype=np.float32)
        qj_obs = (qj_h12[: self._policy_joints] - full_default[: self._policy_joints]) * self._dof_pos_scale
        dqj_obs = dqj_h12[: self._policy_joints] * self._dof_vel_scale
        quat = self._d.qpos[3:7]
        ang_vel = self._d.qvel[3:6] * self._ang_vel_scale
        grav = _gravity_orientation(quat)

        n = self._policy_joints
        obs = np.zeros(self._single_obs_dim, dtype=np.float32)
        i = 0
        obs[i:i + 3] = self._cmd * self._cmd_scale; i += 3
        obs[i] = self._height_cmd; i += 1
        obs[i:i + 3] = ang_vel; i += 3
        obs[i:i + 3] = grav; i += 3
        obs[i:i + n] = qj_obs; i += n
        obs[i:i + n] = dqj_obs; i += n
        obs[i:i + self._num_actions] = self._action
        return obs, qj_h12

    def _policy_step(self):
        single_obs, qj_h12 = self._compute_obs()
        self._obs_history.append(single_obs)
        upper_q = qj_h12[NUM_LEG_MOTORS:self._policy_joints]
        if self._no_encode:
            e_t = np.concatenate([upper_q, np.zeros(6, dtype=np.float32)], dtype=np.float32)
        else:
            e_t = np.concatenate([upper_q, self._left_force, self._right_force], dtype=np.float32)
        if self._encoder is not None:
            with torch.no_grad():
                z_t = self._encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
        else:
            z_t = np.zeros(RMA_LATENT_DIM, dtype=np.float32)
        self._z_history[1:, :] = self._z_history[:-1, :].copy()
        self._z_history[0, :] = z_t
        z_flat = np.flip(self._z_history, axis=0).flatten().astype(np.float32)

        proprio = np.concatenate(list(self._obs_history), axis=0).astype(np.float32)
        actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
        with torch.no_grad():
            action = self._policy(torch.from_numpy(actor_obs).unsqueeze(0)).cpu().numpy().squeeze()
        self._action = np.asarray(action, dtype=np.float32).reshape(self._num_actions)
        self._target_leg_q = self._action * self._action_scale + self._default_legs

    def _step_once(self):
        # apply hand forces (RMA-style external disturbance)
        self._d.xfrc_applied[:] = 0
        if self._apply_forces:
            self._d.xfrc_applied[self._left_wrist_id, :3] = self._left_force
            self._d.xfrc_applied[self._right_wrist_id, :3] = self._right_force

        # leg PD
        leg_tau = self._pd(
            self._target_leg_q, self._d.qpos[self._leg_qpos_adr], self._legs_kp,
            np.zeros_like(self._legs_kp), self._d.qvel[self._leg_qvel_adr], self._legs_kd,
        )
        leg_tau = np.clip(leg_tau, -200.0, 200.0)
        self._d.ctrl[:NUM_LEG_MOTORS] = leg_tau

        # arm PD (target from DDS or default)
        with self._arm_lock:
            aq, adq, atau, akp, akd = (
                self._arm_q_des.copy(), self._arm_dq_des.copy(),
                self._arm_tau_ff.copy(), self._arm_kp.copy(), self._arm_kd.copy(),
            )
        arm_tau = self._pd(
            aq, self._d.qpos[self._arm_qpos_adr], akp,
            adq, self._d.qvel[self._arm_qvel_adr], akd, tau_ff=atau,
        )
        arm_tau = np.clip(arm_tau, -300.0, 300.0)
        self._d.ctrl[NUM_LEG_MOTORS:NUM_H12_MOTORS] = arm_tau

        # extra actuators (Magpie etc.) held mid-range
        if self._extra_ctrl_range is not None:
            self._d.ctrl[NUM_H12_MOTORS:] = 0.5 * (
                self._extra_ctrl_range[:, 0] + self._extra_ctrl_range[:, 1]
            )

        mujoco.mj_step(self._m, self._d)
        self._counter += 1
        if self._counter % self._control_decimation == 0:
            self._policy_step()

        # Publish state + leg cmd from the main thread so MjData (which is
        # NOT thread-safe) is only ever touched here. The DDS subscriber for
        # rt/lowcmd lives in unitree_sdk2py's own callback thread but only
        # writes to numpy arrays guarded by _arm_lock — never to self._d.
        if self._counter % self._state_pub_decim == 0:
            self._fill_and_publish_state()
        if self._counter % self._lower_pub_decim == 0:
            self._publish_lower_cmd()

    # ---------------------------------------------------------------- runner
    def run(self):
        self._running = True
        print(
            f"[fame_mj] scene={self._xml_path.name} headless={self._headless}\n"
            f"          pub lowstate -> {self._low_state_topic}\n"
            f"          pub legs     -> {self._lower_in_topic}\n"
            f"          sub arms     <- {self._lowcmd_topic} (merged via safety_layer)",
            flush=True,
        )

        start = time.time()
        try:
            if self._headless:
                self._run_headless(start)
            else:
                self._run_with_viewer(start)
        finally:
            self._print_summary(start)
            self.shutdown()

    def _should_stop(self, start):
        return self._duration > 0.0 and (time.time() - start) >= self._duration

    def _run_headless(self, start):
        while not self._should_stop(start):
            t0 = time.time()
            self._step_once()
            slp = self._sim_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)

    def _run_with_viewer(self, start):
        import mujoco.viewer
        with mujoco.viewer.launch_passive(self._m, self._d) as viewer:
            while viewer.is_running() and not self._should_stop(start):
                t0 = time.time()
                self._step_once()
                viewer.sync()
                slp = self._sim_dt - (time.time() - t0)
                if slp > 0:
                    time.sleep(slp)

    def _print_summary(self, start):
        with self._arm_lock:
            aq = self._arm_q_des.copy()
            t_first = self._first_arm_cmd_time - start if self._first_arm_cmd_time else -1.0
        q_now = self._d.qpos[self._h12_qpos_adr].copy()
        print(
            "[fame_mj] summary:\n"
            f"  elapsed_s={time.time() - start:.2f}\n"
            f"  first_upper_cmd_at=t+{t_first:.2f}s\n"
            f"  base_xyz={self._d.qpos[:3].tolist()}\n"
            f"  target_legs={self._target_leg_q.tolist()}\n"
            f"  q_now_legs={q_now[:NUM_LEG_MOTORS].tolist()}\n"
            f"  arm_q_des={aq.tolist()}\n"
            f"  q_now_arms={q_now[NUM_LEG_MOTORS:].tolist()}",
            flush=True,
        )

    def shutdown(self):
        self._running = False
        for closeable in (self._lowcmd_sub, self._lower_pub, self._state_pub):
            try:
                closeable.Close()
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description="Unified FAME + Mujoco + DDS (simple setup)")
    p.add_argument("--config", required=True, type=str, help="Runner YAML")
    p.add_argument("--headless", action="store_true", default=None,
                   help="Run sim without viewer (overrides YAML)")
    p.add_argument("--viewer", dest="headless", action="store_false",
                   help="Force viewer (overrides YAML)")
    p.add_argument("--duration", type=float, default=None,
                   help="Stop after N seconds (overrides YAML; 0 = run forever)")
    args = p.parse_args()
    runner = FameMujocoDDS(Path(args.config).resolve(), headless=args.headless, duration=args.duration)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[fame_mj] interrupted", flush=True)


if __name__ == "__main__":
    main()

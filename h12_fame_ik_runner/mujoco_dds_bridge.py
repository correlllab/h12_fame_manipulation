"""Mujoco simulation that bridges to the H1-2 Unitree DDS topology.

Publishes a 27-motor LowState_ on rt/lowstate at 500 Hz and subscribes to the
merged LowCmd_ stream on rt/lowcmd (typically produced by h12_safety_layer in
split mode). On each mj_step the bridge computes
    tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
for each of the first 27 actuators and writes the result to data.ctrl. Extra
actuators (e.g. Magpie gripper) are held at the center of their ctrl range.

To keep the robot upright while FAME / FrameController are still booting, the
bridge latches a per-motor default (q_des / kp / kd) at startup and only
overrides a motor when a LowCmd_ arrives with that motor's mode==1. mode==0
samples are ignored so a transient safety-layer estop publish during warmup
does not collapse the sim.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

import mujoco

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowState_ as LowState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


NUM_H12_MOTORS = 27


def _resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return cfg


class MujocoDDSBridge:
    def __init__(self, config_path: Path, headless: bool | None, duration: float | None):
        self._config_path = config_path
        self._cfg = _load_config(config_path)
        cfg_dir = config_path.parent

        mj_cfg = self._cfg.get("mujoco", {})
        self._xml_path = _resolve_path(mj_cfg["xml_path"], cfg_dir)
        self._sim_dt = float(mj_cfg.get("simulation_dt", 0.002))
        self._headless = headless if headless is not None else bool(mj_cfg.get("headless", False))
        self._duration = float(duration if duration is not None else mj_cfg.get("duration", 0.0))
        # Publish lowstate but freeze physics for this long at startup, so FAME
        # (which takes ~2-4 s to import torch + load the encoder/policy in a
        # sibling process) has time to come online and start publishing real
        # leg targets before gravity is allowed to act.
        self._warmup_seconds = float(mj_cfg.get("warmup_seconds", 0.0))

        robot_cfg = self._cfg.get("robot", {})
        self._num_h12 = int(robot_cfg.get("num_h12_motors", NUM_H12_MOTORS))
        if self._num_h12 != NUM_H12_MOTORS:
            raise ValueError(f"num_h12_motors must be {NUM_H12_MOTORS}")

        self._default_q = np.asarray(robot_cfg["default_q"], dtype=np.float32)
        self._default_kp = np.asarray(robot_cfg["default_kp"], dtype=np.float32)
        self._default_kd = np.asarray(robot_cfg["default_kd"], dtype=np.float32)
        if (
            self._default_q.shape != (self._num_h12,)
            or self._default_kp.shape != (self._num_h12,)
            or self._default_kd.shape != (self._num_h12,)
        ):
            raise ValueError("robot defaults must each have length 27")

        topics = self._cfg.get("topics", {})
        self._low_state_topic = topics.get("low_state", "rt/lowstate")
        self._low_cmd_topic = topics.get("low_cmd_out", "rt/lowcmd")

        net_cfg = self._cfg.get("network", {})
        self._domain_id = int(net_cfg.get("domain_id", 0))
        interface = net_cfg.get("interface", "")
        self._interface = interface if interface else None

        # Mujoco model + data
        self._m = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._d = mujoco.MjData(self._m)
        self._m.opt.timestep = self._sim_dt

        # Optional push-block eval hooks. Activated by env vars so the
        # orchestrator can stay unchanged — set PUSH_BLOCK_MASS=<kg> to
        # rescale the block's mass/inertia, and BLOCK_LOG_PATH=<csv> to
        # dump per-step block world pose. Both default to off.
        self._block_qpos_adr = -1
        self._block_log_path = os.environ.get("BLOCK_LOG_PATH", "").strip()
        self._block_log_fh = None
        self._right_wrist_body_id = mujoco.mj_name2id(
            self._m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        try:
            block_bid = mujoco.mj_name2id(
                self._m, mujoco.mjtObj.mjOBJ_BODY, "push_block"
            )
        except Exception:
            block_bid = -1
        if block_bid >= 0:
            mass_env = os.environ.get("PUSH_BLOCK_MASS", "").strip()
            if mass_env:
                try:
                    new_mass = float(mass_env)
                    old_mass = float(self._m.body_mass[block_bid])
                    if new_mass > 0.0 and old_mass > 0.0:
                        scale = new_mass / old_mass
                        self._m.body_mass[block_bid] = new_mass
                        # uniform-box inertia ∝ mass; keep the shape, rescale magnitude
                        self._m.body_inertia[block_bid] *= scale
                        mujoco.mj_setConst(self._m, self._d)
                        print(
                            f"[bridge] PUSH_BLOCK_MASS override: {old_mass:.4f} -> {new_mass:.4f} kg"
                            f" (inertia x{scale:.3f})",
                            flush=True,
                        )
                except ValueError:
                    print(f"[bridge] ignoring invalid PUSH_BLOCK_MASS={mass_env!r}", flush=True)
            # Locate qpos slot for telemetry (first 3 floats = world xyz)
            try:
                jid = mujoco.mj_name2id(
                    self._m, mujoco.mjtObj.mjOBJ_JOINT, "push_block_free"
                )
                if jid >= 0:
                    self._block_qpos_adr = int(self._m.jnt_qposadr[jid])
            except Exception:
                pass

        # First 27 actuators correspond to BODY_JOINTS / JOINT_NAMES (verified
        # against h1_2.xml/scene.xml at runner-creation time).
        if self._d.ctrl.shape[0] < self._num_h12:
            raise ValueError(
                f"Model exposes {self._d.ctrl.shape[0]} actuators; expected at least {self._num_h12}."
            )
        self._h12_joint_ids = self._m.actuator_trnid[: self._num_h12, 0].astype(np.int32)
        self._h12_qpos_adr = self._m.jnt_qposadr[self._h12_joint_ids].astype(np.int32)
        self._h12_qvel_adr = self._m.jnt_dofadr[self._h12_joint_ids].astype(np.int32)
        if self._d.ctrl.shape[0] > self._num_h12:
            self._extra_ctrl_range = self._m.actuator_ctrlrange[self._num_h12:].copy()
        else:
            self._extra_ctrl_range = None

        # IMU site for orientation (free-base quat from qpos[3:7] suffices for
        # FAME, but we still try to find the named site for body-frame data).
        self._imu_site_id = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_SITE, "imu")

        # Leave qpos at the model's qpos0 (the XML's `pos`/joint defaults).
        # For the H1-2 scenes this is "straight legs on the ground" — a stable
        # equilibrium that holds while the bridge waits for FAME to come up.
        # We used to override joints to default_q here, but default_q is the
        # bent operating stance and the XML pelvis is at z=1.03 (sized for
        # straight legs), so overriding produced an off-ground init that dropped
        # the robot onto bent feet on first mj_step — an impulse the default PD
        # could not absorb. mj_resetData (Backspace in the viewer) hit the same
        # stable qpos0 state, which is why pressing reset "fixed" the fall.

        # Latched command state. Pre-populate with warmup PD so the bridge can
        # apply torques immediately, without waiting for the first DDS message.
        self._cmd_lock = threading.Lock()
        self._latched_q = self._default_q.copy()
        self._latched_dq = np.zeros(self._num_h12, dtype=np.float32)
        self._latched_tau = np.zeros(self._num_h12, dtype=np.float32)
        self._latched_kp = self._default_kp.copy()
        self._latched_kd = self._default_kd.copy()
        self._last_cmd_time = 0.0

        # DDS setup
        if self._interface:
            ChannelFactoryInitialize(self._domain_id, self._interface)
        else:
            ChannelFactoryInitialize(self._domain_id)

        self._crc = CRC()
        self._state_pub = ChannelPublisher(self._low_state_topic, LowState_)
        self._state_pub.Init()
        self._cmd_sub = ChannelSubscriber(self._low_cmd_topic, LowCmd_)
        self._cmd_sub.Init(self._on_low_cmd, 10)
        self._low_state = LowState_default()
        # The real H1-2 publishes mode_machine==6. FAME's deploy_real captures
        # that field at startup and echoes it back. Set it explicitly so the
        # FrameController's LowCmdHandler observes a sane value.
        self._low_state.mode_machine = 6

        self._running = False
        # State pub happens from the MAIN thread inside the step loop so MjData
        # is only touched from one thread (it is NOT safe for concurrent
        # access). With sim_dt=2ms the natural state-pub rate is 500 Hz.
        self._state_pub_decim = max(1, int(round(1.0 / (500.0 * self._sim_dt))))
        self._step_counter = 0
        self._tick = 0
        self._cmd_msg_count = 0
        self._first_cmd_time = 0.0
        # Set when _on_low_cmd first sees a mode==1 motor_cmd for a leg motor;
        # used as the "FAME is alive" signal that ends the warmup window early.
        self._first_drive_leg_time = 0.0

    # ----------------------------------------------------------- DDS callbacks

    def _on_low_cmd(self, msg: LowCmd_) -> None:
        with self._cmd_lock:
            now = time.time()
            if self._first_cmd_time == 0.0:
                self._first_cmd_time = now
            self._last_cmd_time = now
            self._cmd_msg_count += 1
            for i in range(self._num_h12):
                mc = msg.motor_cmd[i]
                if int(mc.mode) != 1:
                    # Treat mode==0 as "not driving this motor" and hold the
                    # last known good target; this protects the sim from a
                    # safety-layer estop publish before FAME/IK come online.
                    continue
                self._latched_q[i] = float(mc.q)
                self._latched_dq[i] = float(mc.dq)
                self._latched_tau[i] = float(mc.tau)
                self._latched_kp[i] = float(mc.kp)
                self._latched_kd[i] = float(mc.kd)
                if i < 12 and self._first_drive_leg_time == 0.0:
                    self._first_drive_leg_time = now

    # ----------------------------------------------------------- State publish

    def _fill_and_publish_low_state(self) -> None:
        # Called from the main step thread, right after mj_step. Safe to read
        # MjData here because no other thread touches it.
        q = self._d.qpos[self._h12_qpos_adr]
        dq = self._d.qvel[self._h12_qvel_adr]
        # Body-frame angular velocity for the IMU. d.qvel[3:6] for a free joint
        # is world-frame; rotate into body frame with the base quat (w,x,y,z).
        quat = self._d.qpos[3:7]
        omega_world = self._d.qvel[3:6]
        omega_body = _quat_rotate_inverse(quat, omega_world)
        # Linear acceleration sensor would require sensordata; for now report
        # gravity in body frame as a usable proxy (FAME does not read accel).
        accel_body = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -9.81], dtype=np.float32))

        msg = self._low_state
        msg.tick = (self._tick + 1) & 0xFFFFFFFF
        self._tick = msg.tick
        msg.imu_state.quaternion = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        msg.imu_state.gyroscope = [float(omega_body[0]), float(omega_body[1]), float(omega_body[2])]
        msg.imu_state.accelerometer = [float(accel_body[0]), float(accel_body[1]), float(accel_body[2])]
        # rpy: skip (not used by FAME's observation builder).
        for i in range(self._num_h12):
            ms = msg.motor_state[i]
            ms.mode = 1
            ms.q = float(q[i])
            ms.dq = float(dq[i])
            ms.ddq = 0.0
            # tau_est from actuator force on the joint
            ms.tau_est = float(self._d.actuator_force[i]) if self._d.actuator_force.shape[0] > i else 0.0
        msg.crc = self._crc.Crc(msg)
        self._state_pub.Write(msg)

    # --------------------------------------------------------------- Sim step

    def _apply_pd_to_ctrl(self) -> None:
        with self._cmd_lock:
            q_des = self._latched_q.copy()
            dq_des = self._latched_dq.copy()
            tau_ff = self._latched_tau.copy()
            kp = self._latched_kp.copy()
            kd = self._latched_kd.copy()

        q = self._d.qpos[self._h12_qpos_adr]
        dq = self._d.qvel[self._h12_qvel_adr]
        tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        tau = np.clip(tau, -300.0, 300.0)
        self._d.ctrl[: self._num_h12] = tau

        if self._extra_ctrl_range is not None:
            mid = 0.5 * (self._extra_ctrl_range[:, 0] + self._extra_ctrl_range[:, 1])
            self._d.ctrl[self._num_h12:] = mid

    def _step_once(self) -> None:
        self._apply_pd_to_ctrl()
        mujoco.mj_step(self._m, self._d)
        self._step_counter += 1
        if self._step_counter % self._state_pub_decim == 0:
            self._fill_and_publish_low_state()
            self._log_block_pose()

    def _log_block_pose(self) -> None:
        if not self._block_log_path or self._block_qpos_adr < 0:
            return
        if self._block_log_fh is None:
            try:
                self._block_log_fh = open(self._block_log_path, "w", buffering=1)
                # leg torque columns: tau_l0..tau_l11 (12 leg actuators
                # in BODY_JOINTS order: 6 left + 6 right hip/knee/ankle).
                leg_tau_cols = ",".join(f"tau_l{i}" for i in range(12))
                self._block_log_fh.write(
                    "t,block_x,block_y,block_z,base_x,base_y,base_z,"
                    "base_qw,base_qx,base_qy,base_qz,"
                    "wrist_x,wrist_y,wrist_z,"
                    "wrist_qw,wrist_qx,wrist_qy,wrist_qz,"
                    "rsp_des,rsr_des,rsy_des,rel_des,"
                    "rsp_now,rsr_now,rsy_now,rel_now,"
                    f"{leg_tau_cols}\n"
                )
            except OSError as exc:
                print(f"[bridge] could not open BLOCK_LOG_PATH={self._block_log_path}: {exc}",
                      flush=True)
                self._block_log_path = ""
                return
        a = self._block_qpos_adr
        bx, by, bz = float(self._d.qpos[a]), float(self._d.qpos[a + 1]), float(self._d.qpos[a + 2])
        px, py, pz = float(self._d.qpos[0]), float(self._d.qpos[1]), float(self._d.qpos[2])
        qw, qx, qy, qz = (float(self._d.qpos[3]), float(self._d.qpos[4]),
                          float(self._d.qpos[5]), float(self._d.qpos[6]))
        if self._right_wrist_body_id >= 0:
            wp = self._d.xpos[self._right_wrist_body_id]
            wq = self._d.xquat[self._right_wrist_body_id]
            wx, wy, wz = float(wp[0]), float(wp[1]), float(wp[2])
            wqw, wqx, wqy, wqz = float(wq[0]), float(wq[1]), float(wq[2]), float(wq[3])
        else:
            wx = wy = wz = 0.0
            wqw, wqx, wqy, wqz = 1.0, 0.0, 0.0, 0.0
        # right-arm joint des/now (indices 20-23 in the 27-motor list)
        with self._cmd_lock:
            rsp_d, rsr_d, rsy_d, rel_d = (
                float(self._latched_q[20]), float(self._latched_q[21]),
                float(self._latched_q[22]), float(self._latched_q[23]),
            )
        rsp_n = float(self._d.qpos[self._h12_qpos_adr[20]])
        rsr_n = float(self._d.qpos[self._h12_qpos_adr[21]])
        rsy_n = float(self._d.qpos[self._h12_qpos_adr[22]])
        rel_n = float(self._d.qpos[self._h12_qpos_adr[23]])
        t = float(self._d.time)
        # Leg actuator forces (12 leg joints, indices 0..11 in BODY_JOINTS).
        # These are the torques FAME's policy is asking the legs to apply
        # this step — direct measure of "how hard the lower body is working".
        leg_tau = ",".join(
            f"{float(self._d.actuator_force[i]):.3f}" for i in range(12)
        )
        self._block_log_fh.write(
            f"{t:.4f},{bx:.5f},{by:.5f},{bz:.5f},{px:.5f},{py:.5f},{pz:.5f},"
            f"{qw:.5f},{qx:.5f},{qy:.5f},{qz:.5f},"
            f"{wx:.5f},{wy:.5f},{wz:.5f},"
            f"{wqw:.5f},{wqx:.5f},{wqy:.5f},{wqz:.5f},"
            f"{rsp_d:.4f},{rsr_d:.4f},{rsy_d:.4f},{rel_d:.4f},"
            f"{rsp_n:.4f},{rsr_n:.4f},{rsy_n:.4f},{rel_n:.4f},"
            f"{leg_tau}\n"
        )

    # ------------------------------------------------------------- Entrypoint

    def run(self) -> None:
        self._running = True
        print(
            f"[bridge] scene={self._xml_path.name} headless={self._headless} "
            f"low_state={self._low_state_topic} low_cmd_in={self._low_cmd_topic}",
            flush=True,
        )

        start = time.time()
        try:
            if self._headless:
                self._run_headless(start)
            else:
                self._run_with_viewer(start)
        finally:
            self._print_run_summary(start)
            self.shutdown()

    def _print_run_summary(self, start: float) -> None:
        elapsed = time.time() - start
        with self._cmd_lock:
            n_cmds = self._cmd_msg_count
            first_dt = (self._first_cmd_time - start) if self._first_cmd_time else -1.0
            q_des = self._latched_q.copy()
        q_now = self._d.qpos[self._h12_qpos_adr].copy()
        base_pos = self._d.qpos[:3].copy()
        # leg vs arm sample positions
        print(
            "[bridge] summary:\n"
            f"  elapsed_s={elapsed:.2f}\n"
            f"  low_cmd_msgs_received={n_cmds} (first at t+{first_dt:.2f}s)\n"
            f"  base_xyz={base_pos.tolist()}\n"
            f"  q_des[0:12]={q_des[:12].tolist()}\n"
            f"  q_now[0:12]={q_now[:12].tolist()}\n"
            f"  q_des[13:20]={q_des[13:20].tolist()}  # left arm\n"
            f"  q_now[13:20]={q_now[13:20].tolist()}\n"
            f"  q_des[20:27]={q_des[20:27].tolist()}  # right arm\n"
            f"  q_now[20:27]={q_now[20:27].tolist()}",
            flush=True,
        )

    def _should_stop(self, start: float) -> bool:
        return self._duration > 0.0 and (time.time() - start) >= self._duration

    def _warmup_done(self, start: float) -> bool:
        if self._warmup_seconds <= 0.0:
            return True
        if (time.time() - start) >= self._warmup_seconds:
            return True
        with self._cmd_lock:
            return self._first_drive_leg_time > 0.0

    def _warmup_tick(self) -> None:
        """Publish state at sim rate but do NOT step physics.

        Keeps the robot frozen at its init pose (joints = default_q, base from
        XML) while FAME loads torch and starts publishing. Once a real leg cmd
        lands the latched targets are already live, and the main loop can step
        without the policy fighting an already-falling robot.
        """
        self._step_counter += 1
        if self._step_counter % self._state_pub_decim == 0:
            self._fill_and_publish_low_state()

    def _run_headless(self, start: float) -> None:
        if self._warmup_seconds > 0.0:
            print(f"[bridge] warmup: holding sim frozen up to {self._warmup_seconds:.2f}s "
                  "or until first FAME leg cmd lands", flush=True)
            while not self._warmup_done(start):
                tick_start = time.time()
                self._warmup_tick()
                sleep_for = self._sim_dt - (time.time() - tick_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            with self._cmd_lock:
                dt_drive = self._first_drive_leg_time - start if self._first_drive_leg_time else -1.0
            print(f"[bridge] warmup done (first leg cmd at t+{dt_drive:.2f}s); stepping physics now",
                  flush=True)
        # Step in real time so that DDS rates remain reasonable. A tighter sim
        # rate is possible but matching wall-clock keeps FAME's policy stable.
        while not self._should_stop(start):
            step_start = time.time()
            self._step_once()
            sleep_for = self._sim_dt - (time.time() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _run_with_viewer(self, start: float) -> None:
        import mujoco.viewer  # imported lazily so headless works without a display
        with mujoco.viewer.launch_passive(self._m, self._d) as viewer:
            if self._warmup_seconds > 0.0:
                print(f"[bridge] warmup: holding sim frozen up to {self._warmup_seconds:.2f}s "
                      "or until first FAME leg cmd lands", flush=True)
                while viewer.is_running() and not self._warmup_done(start):
                    tick_start = time.time()
                    self._warmup_tick()
                    viewer.sync()
                    sleep_for = self._sim_dt - (time.time() - tick_start)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                with self._cmd_lock:
                    dt_drive = self._first_drive_leg_time - start if self._first_drive_leg_time else -1.0
                print(f"[bridge] warmup done (first leg cmd at t+{dt_drive:.2f}s); stepping physics now",
                      flush=True)
            while viewer.is_running() and not self._should_stop(start):
                step_start = time.time()
                self._step_once()
                viewer.sync()
                sleep_for = self._sim_dt - (time.time() - step_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)

    def shutdown(self) -> None:
        self._running = False
        try:
            self._cmd_sub.Close()
        except Exception:
            pass
        try:
            self._state_pub.Close()
        except Exception:
            pass
        if self._block_log_fh is not None:
            try:
                self._block_log_fh.close()
            except Exception:
                pass
            self._block_log_fh = None


def _quat_rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by inverse of unit quat (w, x, y, z). Returns float32 length-3."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    # conjugate
    cx, cy, cz = -x, -y, -z
    # Quaternion-vector multiplication: q^-1 * (0,v) * q (matches deploy_real).
    a = w * w
    out = np.array([
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
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mujoco ↔ DDS bridge for H1-2 sim")
    parser.add_argument("--config", required=True, type=str, help="Path to runner YAML")
    parser.add_argument("--headless", action="store_true", default=None,
                        help="Disable viewer (overrides config)")
    parser.add_argument("--viewer", dest="headless", action="store_false",
                        help="Enable viewer (overrides config)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (overrides config; 0 = run forever)")
    args = parser.parse_args()

    bridge = MujocoDDSBridge(
        config_path=Path(args.config).resolve(),
        headless=args.headless,
        duration=args.duration,
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("[bridge] interrupted, shutting down", flush=True)


if __name__ == "__main__":
    main()

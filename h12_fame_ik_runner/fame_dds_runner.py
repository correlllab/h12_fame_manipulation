"""FAME RMA policy that drives the H1-2 legs via the safety_layer split topology.

Subscribes to rt/lowstate, runs the RMA encoder + base policy at the configured
control rate, and publishes a 27-motor LowCmd_ to rt/safety/lowcmd_lower_in. We
only populate motor_cmd[0..11] (legs) with mode==1; motors 12..26 are left at
mode==0 because they are dropped in split mode by the safety_layer merge step.
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

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as LowCmd_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


NUM_H12_MOTORS = 27
NUM_LEG_MOTORS = 12
RMA_LATENT_DIM = 8

# Make RMA modules importable. h12_adaptive_policy's RMA tree is a namespace
# package (no __init__.py at the top level); mirror the deploy script's trick
# of adding the inner package directory to sys.path so `import RMA.rma_modules`
# resolves.
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


class FameDDSRunner:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as fh:
            top = yaml.safe_load(fh)
        cfg_dir = config_path.parent
        fame = top.get("fame", {})

        self._policy_joints = int(fame.get("policy_num_joints", NUM_H12_MOTORS))
        self._num_actions = int(fame.get("num_actions", NUM_LEG_MOTORS))
        self._num_obs = int(fame.get("num_obs", 252))
        self._single_obs_dim = int(fame.get("single_obs_dim", 76))
        self._obs_history_len = int(fame.get("obs_history_len", 3))
        self._ang_vel_scale = float(fame.get("ang_vel_scale", 0.25))
        self._dof_pos_scale = float(fame.get("dof_pos_scale", 1.0))
        self._dof_vel_scale = float(fame.get("dof_vel_scale", 0.05))
        self._action_scale = float(fame.get("action_scale", 0.25))
        self._cmd_scale = np.asarray(fame.get("cmd_scale", [2.0, 2.0, 0.25]), dtype=np.float32)
        self._cmd = np.asarray(fame.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._height_cmd = float(fame.get("height_cmd", 1.0))
        self._control_decimation = int(fame.get("control_decimation", 10))
        self._publish_hz = float(fame.get("publish_hz", 500.0))
        self._no_encode = bool(fame.get("no_encode", False))

        self._default_legs = np.asarray(fame["default_angles_legs"], dtype=np.float32)
        self._default_arms = np.asarray(fame["default_angles_arms"], dtype=np.float32)
        self._legs_kp = np.asarray(fame["legs_kp"], dtype=np.float32)
        self._legs_kd = np.asarray(fame["legs_kd"], dtype=np.float32)
        self._left_force = np.asarray(fame.get("left_hand_force", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._right_force = np.asarray(fame.get("right_hand_force", [0.0, 0.0, 0.0]), dtype=np.float32)

        self._policy_path = _resolve_path(fame["policy_path"], cfg_dir)
        encoder_path_cfg = fame.get("encoder_path")
        self._encoder_path = _resolve_path(encoder_path_cfg, cfg_dir) if encoder_path_cfg else None

        topics = top.get("topics", {})
        net = top.get("network", {})
        self._low_state_topic = topics.get("low_state", "rt/lowstate")
        self._low_cmd_topic = topics.get("low_cmd_lower_in", "rt/safety/lowcmd_lower_in")
        self._domain_id = int(net.get("domain_id", 0))
        interface = net.get("interface", "")
        self._interface = interface if interface else None

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
            print(f"[fame] loaded encoder {self._encoder_path}", flush=True)
        else:
            print("[fame] no encoder available; z_t will be zeros", flush=True)

        # Latest robot state cache.
        self._state_lock = threading.Lock()
        self._q = np.zeros(NUM_H12_MOTORS, dtype=np.float32)
        self._dq = np.zeros(NUM_H12_MOTORS, dtype=np.float32)
        self._tau_est = np.zeros(NUM_H12_MOTORS, dtype=np.float32)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._ang_vel = np.zeros(3, dtype=np.float32)
        self._got_state = False

        # Action/observation history.
        self._action = np.zeros(self._num_actions, dtype=np.float32)
        self._target_dof_pos = self._default_legs.copy()
        self._obs_history = collections.deque(
            [np.zeros(self._single_obs_dim, dtype=np.float32) for _ in range(self._obs_history_len)],
            maxlen=self._obs_history_len,
        )
        self._z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
        self._counter = 0

        # DDS setup
        if self._interface:
            ChannelFactoryInitialize(self._domain_id, self._interface)
        else:
            ChannelFactoryInitialize(self._domain_id)
        self._crc = CRC()
        self._cmd_pub = ChannelPublisher(self._low_cmd_topic, LowCmd_)
        self._cmd_pub.Init()
        self._low_cmd = LowCmd_default()
        self._low_cmd.mode_pr = 0
        self._low_cmd.mode_machine = 6
        # Pre-fill leg motor mode/kp/kd; q starts at default.
        for i in range(NUM_LEG_MOTORS):
            mc = self._low_cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = float(self._default_legs[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self._legs_kp[i])
            mc.kd = float(self._legs_kd[i])
        # Upper motors stay at mode=0 (dropped by safety_layer split).

        self._state_sub = ChannelSubscriber(self._low_state_topic, LowState_)
        self._state_sub.Init(self._on_low_state, 10)

        self._running = False
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop, name="fame_publisher", daemon=True
        )

    # ------------------------------------------------------------ subscribers
    def _on_low_state(self, msg: LowState_) -> None:
        with self._state_lock:
            self._quat = np.asarray(msg.imu_state.quaternion, dtype=np.float32)
            self._ang_vel = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)
            for i in range(NUM_H12_MOTORS):
                self._q[i] = float(msg.motor_state[i].q)
                self._dq[i] = float(msg.motor_state[i].dq)
                self._tau_est[i] = float(msg.motor_state[i].tau_est)
            self._got_state = True

    # ----------------------------------------------------------- observation
    def _compute_single_obs(self) -> np.ndarray:
        with self._state_lock:
            qj = self._q.copy()
            dqj = self._dq.copy()
            quat = self._quat.copy()
            ang_vel = self._ang_vel.copy()
        full_default = np.concatenate([self._default_legs, self._default_arms], dtype=np.float32)
        qj_obs = (qj[: self._policy_joints] - full_default[: self._policy_joints]) * self._dof_pos_scale
        dqj_obs = dqj[: self._policy_joints] * self._dof_vel_scale
        omega = ang_vel * self._ang_vel_scale
        grav = _gravity_orientation(quat)

        n = self._policy_joints
        obs = np.zeros(self._single_obs_dim, dtype=np.float32)
        i = 0
        obs[i:i + 3] = self._cmd * self._cmd_scale; i += 3
        obs[i] = self._height_cmd; i += 1
        obs[i:i + 3] = omega; i += 3
        obs[i:i + 3] = grav; i += 3
        obs[i:i + n] = qj_obs; i += n
        obs[i:i + n] = dqj_obs; i += n
        obs[i:i + self._num_actions] = self._action
        return obs

    def _build_et(self, qj_upper: np.ndarray) -> np.ndarray:
        if self._no_encode:
            return np.concatenate([qj_upper, np.zeros(6, dtype=np.float32)], dtype=np.float32)
        return np.concatenate([qj_upper, self._left_force, self._right_force], dtype=np.float32)

    # ---------------------------------------------------------------- step
    def _control_step(self) -> None:
        single_obs = self._compute_single_obs()
        self._obs_history.append(single_obs)
        with self._state_lock:
            qj_upper = self._q[NUM_LEG_MOTORS:self._policy_joints].copy()
        e_t = self._build_et(qj_upper)
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
        if actor_obs.shape[0] != self._num_obs:
            # Tolerate small mismatches by padding/truncating; this only fires
            # if the YAML obs sizes are wrong and we want a useful error.
            raise ValueError(f"actor_obs has shape {actor_obs.shape}, expected {self._num_obs}")

        with torch.no_grad():
            action = self._policy(torch.from_numpy(actor_obs).unsqueeze(0)).cpu().numpy().squeeze()
        self._action = np.asarray(action, dtype=np.float32).reshape(self._num_actions)
        self._target_dof_pos = self._action * self._action_scale + self._default_legs

    # ---------------------------------------------------------- publisher
    def _publisher_loop(self) -> None:
        dt = 1.0 / self._publish_hz
        ctrl_dt = self._control_decimation * dt
        next_ctrl_time = time.time()
        while self._running:
            t0 = time.time()
            if self._got_state and time.time() >= next_ctrl_time:
                try:
                    self._control_step()
                except Exception as exc:
                    print(f"[fame] control_step error: {exc}", flush=True)
                next_ctrl_time = t0 + ctrl_dt
            self._publish_low_cmd()
            sleep_for = dt - (time.time() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _publish_low_cmd(self) -> None:
        for i in range(NUM_LEG_MOTORS):
            mc = self._low_cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = float(self._target_dof_pos[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self._legs_kp[i])
            mc.kd = float(self._legs_kd[i])
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._cmd_pub.Write(self._low_cmd)

    # ---------------------------------------------------------- entrypoint
    def run(self) -> None:
        self._running = True
        self._publisher_thread.start()
        print(
            f"[fame] low_state={self._low_state_topic} low_cmd_out={self._low_cmd_topic} "
            f"decim={self._control_decimation} pub_hz={self._publish_hz}",
            flush=True,
        )
        try:
            while self._running:
                time.sleep(0.5)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._running = False
        try:
            self._publisher_thread.join(timeout=1.0)
        except RuntimeError:
            pass
        try:
            self._cmd_sub_close = self._state_sub.Close()
        except Exception:
            pass
        try:
            self._cmd_pub.Close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="FAME RMA policy DDS runner")
    parser.add_argument("--config", required=True, type=str, help="Path to runner YAML")
    args = parser.parse_args()
    runner = FameDDSRunner(Path(args.config).resolve())
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[fame] interrupted, shutting down", flush=True)


if __name__ == "__main__":
    main()

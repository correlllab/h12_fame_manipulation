#!/usr/bin/env bash
# Run the example arm_controller_goto.py from h12_ros2_controller with the
# sim-relaxed split controller config so it publishes to
# rt/safety/lowcmd_upper_in (matching the safety_layer the simpler setup runs
# under). Pass EE poses as "x y z roll pitch yaw" (xyz in m, RPY in degrees).
set -euo pipefail

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV_PY="$ROOT/../h12_adaptive_policy/.venv/bin/python"
H12_CTL="$ROOT/../h12_ros2_controller"
SIM_CFG="$ROOT/configs/sim_split_controller.yaml"

if [[ ! -x "$VENV_PY" ]]; then
  echo "expected venv python at $VENV_PY" >&2
  exit 1
fi
if [[ ! -d "$H12_CTL" ]]; then
  echo "expected h12_ros2_controller at $H12_CTL" >&2
  exit 1
fi

# arm_controller_goto.py uses relative paths like assets/h1_2/... so it must
# run from inside the h12_ros2_controller folder.
cd "$H12_CTL"
exec "$VENV_PY" -u h12_ros2_controller/example/arm_controller_goto.py --config "$SIM_CFG"

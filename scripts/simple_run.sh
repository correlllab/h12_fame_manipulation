#!/usr/bin/env bash
# Yutong's recommended setup: one safety_layer + one unified FAME-Mujoco-DDS
# process, then run any upper-body publisher you like (interactive REPL, joint
# streamer, or arm_controller_goto.py from h12_ros2_controller) in another
# terminal.
#
# Usage:
#   ./scripts/simple_run.sh                         # bare H1-2, viewer
#   ./scripts/simple_run.sh configs/h1_2_fame_ik_magpie.yaml
#   ./scripts/simple_run.sh configs/h1_2_fame_ik.yaml --headless
#
# When this script is up, in a second terminal run ONE of:
#   ./scripts/arm_stream.sh             # raw joint streaming REPL (this repo)
#   ./scripts/ik_repl.sh                # frame-task REPL                (this repo)
#   ./scripts/arm_goto.sh      # arm_controller_goto.py        (h12_ros2_controller)
#
# Ctrl-C in THIS terminal tears down the safety_layer + fame_mj pair.
set -euo pipefail

CFG="${1:-configs/h1_2_fame_ik.yaml}"
shift || true
EXTRA_ARGS=("$@")

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV_PY="$ROOT/../h12_adaptive_policy/.venv/bin/python"
SAFETY_MAIN="$ROOT/../h12_safety_layer/h12_safety_layer/script/safety_layer_main.py"
SAFETY_CFG="$ROOT/configs/sim_safety_split.yaml"

if [[ ! -x "$VENV_PY" ]]; then
  echo "expected venv python at $VENV_PY" >&2
  exit 1
fi

cd "$ROOT"

cleanup() {
  local rc=$?
  echo "[simple_run] shutting down (rc=$rc)"
  for pid in "${SL_PID:-}" "${FM_PID:-}"; do
    [[ -n "$pid" ]] && kill -INT "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[simple_run] launching safety_layer (split mode) ..."
"$VENV_PY" -u "$SAFETY_MAIN" --config "$SAFETY_CFG" 2>&1 | sed 's/^/[safety] /' &
SL_PID=$!
sleep 0.5

echo "[simple_run] launching unified FAME + Mujoco + DDS ..."
"$VENV_PY" -u -m h12_fame_ik_runner.fame_mujoco_dds --config "$CFG" "${EXTRA_ARGS[@]}" 2>&1 | sed 's/^/[fame_mj] /' &
FM_PID=$!

cat <<EOF

================================================================================
Simple stack is up.

  safety_layer pid = $SL_PID  (config: $SAFETY_CFG)
  fame_mujoco_dds pid = $FM_PID (config: $CFG)

Open a second terminal and run ONE of:

  cd "$ROOT" && ./scripts/arm_stream.sh    # raw joint streaming REPL
  cd "$ROOT" && ./scripts/ik_repl.sh              # frame-task REPL
  cd "$ROOT" && ./scripts/arm_goto.sh      # arm_controller_goto.py (h12_ros2_controller)

Ctrl-C in THIS terminal to stop everything.
================================================================================

EOF

# Wait on the FAME process. If either dies, the trap will tear the other down.
wait $FM_PID

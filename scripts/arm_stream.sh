#!/usr/bin/env bash
# Direct joint-command streamer. Publishes 27-motor LowCmd to
# rt/safety/lowcmd_upper_in at 500 Hz (mode=1 on motors 12..26, mode=0 on legs).
# Type `help` at the prompt for the command list.
#
# Use this when you want bare joint control with no IK in the loop.
set -euo pipefail

CFG="${1:-configs/h1_2_fame_ik.yaml}"

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV_PY="$ROOT/../h12_adaptive_policy/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "expected venv python at $VENV_PY" >&2
  exit 1
fi
cd "$ROOT"
exec "$VENV_PY" -u -m h12_fame_ik_runner.arm_joint_stream --config "$CFG"

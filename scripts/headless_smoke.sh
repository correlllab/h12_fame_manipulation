#!/usr/bin/env bash
# Headless smoke test: spin up the full FAME + IK + safety + mujoco stack with
# no viewer, run for 20s, and print a bridge summary so you can confirm
# commands are flowing.
set -euo pipefail

CFG="${1:-configs/h1_2_fame_ik.yaml}"
DURATION="${2:-20}"

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV_PY="$ROOT/../h12_adaptive_policy/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "expected venv python at $VENV_PY" >&2
  echo "run 'uv sync' inside src/h12_adaptive_policy first" >&2
  exit 1
fi

cd "$ROOT"
exec "$VENV_PY" -u -m h12_fame_ik_runner.orchestrator \
  --config "$CFG" \
  --headless \
  --duration "$DURATION"

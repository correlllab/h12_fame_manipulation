#!/usr/bin/env bash
# Live-viewer run: same stack as headless_smoke but with mujoco's passive
# viewer attached so you can watch the FAME-driven legs + IK-driven arms in
# real time. Ctrl-C to stop.
set -euo pipefail

CFG="${1:-configs/h1_2_fame_ik.yaml}"

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
  --viewer

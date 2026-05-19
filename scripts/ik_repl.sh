#!/usr/bin/env bash
# Interactive IK REPL. Run this in a SEPARATE terminal after starting the rest
# of the stack via:
#
#     ./scripts/viewer.sh   configs/h1_2_fame_ik.yaml &     # or headless_smoke
#     # then in another terminal:
#     ./scripts/ik_repl.sh  configs/h1_2_fame_ik.yaml
#
# Or, more robustly, start the stack with --no-ik and then run this:
#
#     python -m h12_fame_ik_runner.orchestrator --config configs/h1_2_fame_ik.yaml \
#         --viewer --no-ik
#     ./scripts/ik_repl.sh
#
# Commands at the prompt are documented by typing `help`.
set -euo pipefail

CFG="${1:-configs/h1_2_fame_ik.yaml}"

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV_PY="$ROOT/../h12_adaptive_policy/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "expected venv python at $VENV_PY" >&2
  exit 1
fi

cd "$ROOT"
exec "$VENV_PY" -u -m h12_fame_ik_runner.ik_dds_driver \
  --config "$CFG" \
  --interactive

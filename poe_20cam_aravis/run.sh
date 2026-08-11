#!/usr/bin/env bash
# Launch the 20-camera Aravis monitor from a source checkout.
#
#   ./run.sh                 # normal run
#   ./run.sh --fake          # Aravis fake-camera interface
#   ./run.sh --log-level DEBUG
#
# Nothing here needs installing: PYTHONPATH points at src/ so the package
# imports straight from the working tree.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"

PY="${PYTHON:-python3}"

exec "${PY}" -m poe_multi_aravis.app "$@"

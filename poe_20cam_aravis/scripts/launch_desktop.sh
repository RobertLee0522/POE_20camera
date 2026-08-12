#!/usr/bin/env bash
# Desktop-icon launcher for the POE 20-Camera Monitor (Aravis edition).
#
# Activates the pre-built `poe` conda environment (PyQt5 + OpenCV + PyGObject
# + Aravis bindings all live there — see MIGRATION.md) and starts the app.
# Meant to be invoked by poe-20cam-aravis.desktop, not run interactively.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
LOG_DIR="$PROJECT_DIR/data/logs"
LOG_FILE="$LOG_DIR/desktop_launch.log"
mkdir -p "$LOG_DIR"

{
    echo "───── $(date '+%Y-%m-%d %H:%M:%S') launch ─────"

    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate poe
    else
        echo "warning: $CONDA_BASE/etc/profile.d/conda.sh not found; using system python3" >&2
    fi

    cd "$PROJECT_DIR"
    export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
    exec python3 -m poe_multi_aravis.app "$@"
} >>"$LOG_FILE" 2>&1

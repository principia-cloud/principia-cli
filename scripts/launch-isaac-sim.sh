#!/usr/bin/env bash
# Launch Isaac Sim with the MCP extension enabled.
#
# Usage:
#   ./scripts/launch-isaac-sim.sh [EXT_FOLDER]
#
# Arguments:
#   EXT_FOLDER  Directory containing the isaac.sim.mcp_extension folder.
#               If omitted, searches common locations automatically.
#
# Environment:
#   ISAAC_SIM_DIR   Path to Isaac Sim install (default: /isaac-sim)
#   SLURM_JOB_ID   If set, MCP port is derived from this via hash (default port: 11323)
#   POLL_TIMEOUT    Seconds to wait for MCP socket (default: 120)
#
# You can also enable the extension manually when launching Isaac Sim:
#   isaac-sim.sh --ext-folder <EXT_FOLDER> --enable isaac.sim.mcp_extension

set -euo pipefail

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-/isaac-sim}"
EXT_NAME="isaac.sim.mcp_extension"
EXT_DIR_NAME="isaac.sim.mcp_extension"
LOG_DIR="${HOME}/.principia/log"
LOG_FILE="${LOG_DIR}/isaac-sim.log"
POLL_PORT=11323
POLL_TIMEOUT="${POLL_TIMEOUT:-120}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Resolve extension folder
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Bundled location: <repo>/servers/isaacsim/isaac.sim.mcp_extension
BUNDLED_EXT="${REPO_DIR}/servers/isaacsim"

EXT_FOLDER="${1:-}"
if [ -z "$EXT_FOLDER" ]; then
    if [ -d "${BUNDLED_EXT}/${EXT_DIR_NAME}" ]; then
        EXT_FOLDER="$BUNDLED_EXT"
    elif [ -d "${PRINCIPIA_HOME:-${HOME}/.principia}/source/servers/isaacsim/${EXT_DIR_NAME}" ]; then
        EXT_FOLDER="${PRINCIPIA_HOME:-${HOME}/.principia}/source/servers/isaacsim"
    else
        error "Could not find ${EXT_DIR_NAME}. Expected at:
  ${BUNDLED_EXT}/${EXT_DIR_NAME}

Pass the path explicitly:
  $0 /path/to/folder/containing/${EXT_DIR_NAME}

Or launch Isaac Sim manually with:
  ${ISAAC_SIM_DIR}/isaac-sim.sh --ext-folder <EXT_FOLDER> --enable ${EXT_NAME}"
    fi
fi

[ -d "${EXT_FOLDER}/${EXT_DIR_NAME}" ] || \
    error "Extension not found at ${EXT_FOLDER}/${EXT_DIR_NAME}"

info "Extension folder: ${EXT_FOLDER}"

# ---------------------------------------------------------------------------
# Compute MCP port (must match the extension's slurm_job_id_to_port)
# ---------------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    POLL_PORT=$(python3 -c "
import hashlib
h = int(hashlib.md5(str('${SLURM_JOB_ID}').encode()).hexdigest(), 16)
print(8080 + h % (40000 - 8080 + 1))
")
    info "SLURM_JOB_ID=${SLURM_JOB_ID} -> MCP port ${POLL_PORT}"
else
    info "No SLURM_JOB_ID -> MCP port defaults to ${POLL_PORT}"
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[ -d "$ISAAC_SIM_DIR" ] || error "Isaac Sim not found at ${ISAAC_SIM_DIR}"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Stop any existing Isaac Sim process
# ---------------------------------------------------------------------------
if pgrep -f "${ISAAC_SIM_DIR}/kit/kit" >/dev/null 2>&1; then
    info "Stopping existing Isaac Sim..."
    pkill -f "${ISAAC_SIM_DIR}/kit/kit" || true
    sleep 3
    # Force-kill if still running
    if pgrep -f "${ISAAC_SIM_DIR}/kit/kit" >/dev/null 2>&1; then
        info "Force-killing remaining Isaac Sim processes..."
        pkill -9 -f "${ISAAC_SIM_DIR}/kit/kit" || true
        sleep 2
    fi
    ok "Isaac Sim stopped"
fi

# ---------------------------------------------------------------------------
# Launch Isaac Sim with MCP extension
# ---------------------------------------------------------------------------
info "Starting Isaac Sim with MCP + VSCode extensions..."
info "Log: ${LOG_FILE}"

SCENE_GEN_DATA_DIR="${SCENE_GEN_DATA_DIR:-${PRINCIPIA_HOME:-${HOME}/.principia}/data/scene-gen}" \
nohup "${ISAAC_SIM_DIR}/isaac-sim.sh" \
    --ext-folder "$EXT_FOLDER" \
    --enable "$EXT_NAME" \
    --enable isaacsim.code_editor.vscode \
    > "$LOG_FILE" 2>&1 &

ISAAC_PID=$!
info "Isaac Sim PID: ${ISAAC_PID}"

# ---------------------------------------------------------------------------
# Wait for MCP socket to be ready
# ---------------------------------------------------------------------------
info "Waiting for MCP extension on port ${POLL_PORT} (timeout ${POLL_TIMEOUT}s)..."

elapsed=0
while [ $elapsed -lt $POLL_TIMEOUT ]; do
    if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
        error "Isaac Sim exited unexpectedly. Check log: ${LOG_FILE}"
    fi

    if (echo >/dev/tcp/localhost/"$POLL_PORT") 2>/dev/null; then
        ok "MCP extension ready on port ${POLL_PORT} (took ${elapsed}s)"
        exit 0
    fi

    sleep 2
    elapsed=$((elapsed + 2))
done

error "Timed out waiting for MCP extension on port ${POLL_PORT}. Check log: ${LOG_FILE}"

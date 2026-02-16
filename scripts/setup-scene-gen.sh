#!/usr/bin/env bash
# Scene-gen MCP server setup.
# Can be sourced from install.sh (reuses its helpers) or run standalone.
#
# Usage (standalone):
#   ./scripts/setup-scene-gen.sh
#
# When sourced from install.sh, call setup_scene_gen after sourcing.

# ---------------------------------------------------------------------------
# Provide helpers if not already defined (standalone mode)
# ---------------------------------------------------------------------------
if ! declare -f info >/dev/null 2>&1; then
    info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*"; }
    ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*"; }
    warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
    error() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
fi

PRINCIPIA_HOME="${PRINCIPIA_HOME:-${HOME}/.principia}"

S3_BASE_URL="https://principia-scene-gen-assets.s3.us-east-1.amazonaws.com"
SCENE_GEN_DATA_DIR="${PRINCIPIA_HOME}/data/scene-gen"

# ---------------------------------------------------------------------------
# Python detection: find python3.12 / 3.11 / 3.10 / 3 >= 3.10
# ---------------------------------------------------------------------------
_find_python() {
    local py
    for py in python3.12 python3.11 python3.10 python3; do
        if command -v "$py" >/dev/null 2>&1; then
            local ver
            ver="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" || continue
            local major minor
            major="${ver%%.*}"
            minor="${ver##*.}"
            if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
                echo "$py"
                return 0
            fi
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# System dependencies (apt)
# ---------------------------------------------------------------------------
_ensure_apt_packages() {
    local needed=()
    local pkg

    # nvidia-cuda-toolkit provides nvcc (needed to build nvdiffrast, pytorch3d)
    if ! command -v nvcc >/dev/null 2>&1; then
        needed+=(nvidia-cuda-toolkit)
    fi

    # OpenGL headers (needed by nvdiffrast / pyrender)
    if ! dpkg -s libegl1-mesa-dev >/dev/null 2>&1; then
        needed+=(libegl1-mesa-dev)
    fi

    if [ ${#needed[@]} -eq 0 ]; then
        ok "System packages already installed"
        return 0
    fi

    info "Installing system packages: ${needed[*]}"
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq "${needed[@]}" || {
            warn "Failed to install apt packages: ${needed[*]}"
            warn "You may need to install them manually for GPU rendering support."
            return 1
        }
        ok "System packages installed"
    else
        warn "sudo not available — cannot install: ${needed[*]}"
        warn "Install them manually for GPU rendering support."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# GPU pip packages (need CUDA toolkit for compilation)
# ---------------------------------------------------------------------------
_ensure_gpu_pip_packages() {
    local venv_dir="$1"
    local pip="${venv_dir}/bin/pip"
    local python="${venv_dir}/bin/python"

    # Check if GPU is available
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "No NVIDIA GPU detected — skipping GPU package installation"
        warn "nvdiffrast and pytorch3d will not be available"
        return 0
    fi

    # nvdiffrast
    if "$python" -c "import nvdiffrast" 2>/dev/null; then
        ok "nvdiffrast already installed"
    else
        info "Installing nvdiffrast (building from source, may take a few minutes)..."
        CUDA_HOME=/usr PIP_USER=0 "$pip" install --no-build-isolation \
            "git+https://github.com/NVlabs/nvdiffrast.git" \
            --cache-dir "${venv_dir}/.pip-cache" --quiet || {
            warn "Failed to install nvdiffrast — GPU rendering may not work"
        }
    fi

}

# ---------------------------------------------------------------------------
# Main setup function
# ---------------------------------------------------------------------------
setup_scene_gen() {
    info "Setting up scene-gen MCP server..."

    local source_dir="${PRINCIPIA_HOME}/source"
    local scene_gen_source="${source_dir}/servers/scene-gen"
    local venv_dir="${SCENE_GEN_DATA_DIR}/venv"
    local matfuse_dir="${SCENE_GEN_DATA_DIR}/matfuse"
    local objathor_dir="${SCENE_GEN_DATA_DIR}/objathor/2023_09_23"
    local results_dir="${SCENE_GEN_DATA_DIR}/results"

    if [ ! -d "$scene_gen_source" ]; then
        warn "scene-gen source not found at ${scene_gen_source} — skipping"
        return
    fi

    # ----- System dependencies -----
    _ensure_apt_packages

    # ----- Python detection -----
    local python_cmd
    python_cmd="$(_find_python)" || {
        warn "Python >= 3.10 not found — skipping scene-gen setup"
        warn "Install Python 3.10+ and re-run: ./scripts/setup-scene-gen.sh"
        return
    }
    info "Using Python: ${python_cmd} ($(${python_cmd} --version 2>&1))"

    # ----- Venv creation with staleness check -----
    local req_file="${scene_gen_source}/requirements.txt"
    local hash_file="${venv_dir}/.requirements_hash"
    local current_hash
    current_hash="$(shasum -a 256 "$req_file" 2>/dev/null | cut -d' ' -f1)" || current_hash=""

    local needs_install=false
    if [ ! -d "$venv_dir" ]; then
        info "Creating Python virtual environment..."
        mkdir -p "$(dirname "$venv_dir")"
        "$python_cmd" -m venv "$venv_dir" || {
            warn "Failed to create venv — skipping scene-gen setup"
            warn "On Debian/Ubuntu you may need: sudo apt install python3-venv"
            return
        }
        needs_install=true
    elif [ ! -f "$hash_file" ] || [ "$(cat "$hash_file" 2>/dev/null)" != "$current_hash" ]; then
        info "Requirements changed — reinstalling dependencies..."
        needs_install=true
    else
        ok "Python venv up to date (requirements unchanged)"
    fi

    # On Debian/Ubuntu, venv may be created without pip if ensurepip is
    # missing (packaged separately as python3.X-venv). Bootstrap it.
    if [ -d "$venv_dir" ] && [ ! -f "${venv_dir}/bin/pip" ]; then
        info "pip not found in venv — bootstrapping via ensurepip / get-pip.py..."
        "${venv_dir}/bin/python" -m ensurepip --upgrade 2>/dev/null || {
            # ensurepip unavailable — fall back to get-pip.py
            # PIP_USER=0 overrides Debian's global pip.conf (user=true)
            curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/_get_pip.py \
                && PIP_USER=0 "${venv_dir}/bin/python" /tmp/_get_pip.py --quiet \
                && rm -f /tmp/_get_pip.py \
                || {
                    warn "Could not install pip into venv — skipping scene-gen setup"
                    warn "On Debian/Ubuntu: sudo apt install python3-venv python3-pip"
                    return
                }
        }
        needs_install=true
    fi

    if $needs_install; then
        info "Installing Python dependencies (this may take a few minutes)..."
        PIP_USER=0 "${venv_dir}/bin/pip" install --upgrade pip --cache-dir "${venv_dir}/.pip-cache" --quiet 2>/dev/null || true
        PIP_USER=0 "${venv_dir}/bin/pip" install -r "$req_file" --cache-dir "${venv_dir}/.pip-cache" --quiet || {
            warn "pip install failed — scene-gen may not work correctly"
            return
        }
        echo "$current_hash" > "$hash_file"
        ok "Python dependencies installed"
    fi

    # ----- GPU packages (nvdiffrast, pytorch3d — need CUDA compilation) -----
    _ensure_gpu_pip_packages "$venv_dir"

    # ----- Download MatFuse checkpoint (4.3GB) -----
    local matfuse_ckpt="${matfuse_dir}/matfuse-full.ckpt"
    if [ -f "$matfuse_ckpt" ]; then
        ok "MatFuse checkpoint already present"
    else
        info "Downloading MatFuse checkpoint (~4.3GB, this will take a while)..."
        mkdir -p "$matfuse_dir"
        local tmp_ckpt="${matfuse_ckpt}.downloading"
        curl -fSL --progress-bar \
            "${S3_BASE_URL}/matfuse/matfuse-full.ckpt" \
            -o "$tmp_ckpt" || {
            rm -f "$tmp_ckpt"
            warn "Failed to download MatFuse checkpoint — scene-gen texture generation will not work"
            warn "Retry: curl -fSL ${S3_BASE_URL}/matfuse/matfuse-full.ckpt -o ${matfuse_ckpt}"
        }
        if [ -f "$tmp_ckpt" ]; then
            mv "$tmp_ckpt" "$matfuse_ckpt"
            ok "MatFuse checkpoint downloaded"
        fi
    fi

    # ----- Download ObjaThor features + annotations (~380MB) -----
    mkdir -p "$objathor_dir/features"
    mkdir -p "$objathor_dir/assets"

    local _objathor_pairs=(
        "annotations.json.gz|${S3_BASE_URL}/objathor/2023_09_23/annotations.json.gz"
        "features/clip_features.pkl|${S3_BASE_URL}/objathor/2023_09_23/features/clip_features.pkl"
        "features/sbert_features.pkl|${S3_BASE_URL}/objathor/2023_09_23/features/sbert_features.pkl"
    )

    local _pair _file _url
    for _pair in "${_objathor_pairs[@]}"; do
        _file="${_pair%%|*}"
        _url="${_pair#*|}"
        local dest="${objathor_dir}/${_file}"
        if [ -f "$dest" ]; then
            ok "ObjaThor ${_file} already present"
        else
            info "Downloading ObjaThor ${_file}..."
            local tmp_dest="${dest}.downloading"
            curl -fSL --progress-bar "$_url" -o "$tmp_dest" || {
                rm -f "$tmp_dest"
                warn "Failed to download ${_file}"
                continue
            }
            mv "$tmp_dest" "$dest"
            ok "Downloaded ${_file}"
        fi
    done

    # ----- Download Holodeck base data (doors, materials, windows — ~77MB) -----
    local holodeck_dir="${SCENE_GEN_DATA_DIR}/objathor/holodeck/2023_09_23"
    if [ -f "${holodeck_dir}/doors/door-database.json" ]; then
        ok "Holodeck base data already present"
    else
        info "Downloading Holodeck base data (doors, materials, windows — ~77MB)..."
        "${venv_dir}/bin/python" -c "
from objathor.dataset.download_holodeck_base_data import DatasetSaveConfig, load_holodeck_base
dsc = DatasetSaveConfig(VERSION='2023_09_23', BASE_PATH='${SCENE_GEN_DATA_DIR}/objathor')
load_holodeck_base(dsc)
" || {
            warn "Failed to download Holodeck base data — doors and materials will not work"
            warn "Install objathor and retry: pip install objathor"
        }
        if [ -f "${holodeck_dir}/doors/door-database.json" ]; then
            ok "Holodeck base data downloaded"
        fi
    fi

    # ----- Create results directory -----
    mkdir -p "$results_dir"

    # ----- Install scene-gen skill -----
    local skills_dir="${PRINCIPIA_HOME}/skills/scene-gen"
    local skill_source="${scene_gen_source}/skill/SKILL.md"
    if [ -f "$skill_source" ]; then
        mkdir -p "$skills_dir"
        ln -sf "$skill_source" "${skills_dir}/SKILL.md"
        ok "scene-gen skill installed → ${skills_dir}/SKILL.md"
    else
        warn "SKILL.md not found at ${skill_source} — skill will not be available"
    fi

    # ----- Register MCP server -----
    local settings_dir="${PRINCIPIA_HOME}/data/settings"
    local mcp_settings_file="${settings_dir}/principia_mcp_settings.json"
    local old_mcp_settings_file="${settings_dir}/cline_mcp_settings.json"

    mkdir -p "$settings_dir"

    # Migrate old settings file if it exists
    if [ -f "$old_mcp_settings_file" ] && [ ! -f "$mcp_settings_file" ]; then
        info "Migrating MCP settings from cline_mcp_settings.json to principia_mcp_settings.json..."
        mv "$old_mcp_settings_file" "$mcp_settings_file"
        ok "MCP settings migrated"
    fi

    info "Registering scene-gen MCP server..."
    node -e "
const fs = require('fs');
const path = '${mcp_settings_file}';
let config = {};
try { config = JSON.parse(fs.readFileSync(path, 'utf8')); } catch {}
if (!config.mcpServers) config.mcpServers = {};
const prev = (config.mcpServers['scene-gen'] || {}).env || {};
config.mcpServers['scene-gen'] = {
    command: '${venv_dir}/bin/python',
    args: ['${scene_gen_source}/server.py'],
    timeout: 600,
    env: {
        MATFUSE_CKPT: '${matfuse_dir}/matfuse-full.ckpt',
        OBJATHOR_ASSETS_BASE_DIR: '${SCENE_GEN_DATA_DIR}/objathor',
        RESULTS_DIR: '${results_dir}',
        QWEN_VL_URL: prev.QWEN_VL_URL || 'https://dashscope-us.aliyuncs.com/compatible-mode/v1',
        QWEN_VL_MODEL: prev.QWEN_VL_MODEL || 'qwen3-vl-30b-a3b-instruct',
        QWEN_VL_API_KEY: prev.QWEN_VL_API_KEY || '',
        TRELLIS_URL: prev.TRELLIS_URL || '',
        FLUX_SERVER_URL: prev.FLUX_SERVER_URL || '',
        PHYSICS_CRITIC_ENABLED: prev.PHYSICS_CRITIC_ENABLED || 'true',
        SEMANTIC_CRITIC_ENABLED: prev.SEMANTIC_CRITIC_ENABLED || 'true',
        ISAAC_SIM_HOST: prev.ISAAC_SIM_HOST || 'localhost'
    },
    autoApprove: [
        'generate_room_layout',
        'get_current_layout',
        'get_room_details',
        'list_rooms',
        'get_layout_from_json',
        'place_objects_in_room',
        'get_layout_save_dir',
        'get_room_information',
        'move_one_object_with_condition_in_room'
    ]
};
fs.writeFileSync(path, JSON.stringify(config, null, 2) + '\n');
" || {
        warn "Failed to register scene-gen MCP server — you may need to configure it manually"
        warn "See: ${scene_gen_source}/mcp_settings_example.json"
    }

    ok "scene-gen MCP server registered"

    # ----- Isaac Sim + MCP extension hint -----
    if [ -d "/isaac-sim" ]; then
        printf '\n'
        info "Isaac Sim detected at /isaac-sim"
        info "The MCP extension must be enabled for build_scene/physics to work."
        info "  Launch helper:  ./scripts/launch-isaac-sim.sh"
        info "  Or manually:    /isaac-sim/isaac-sim.sh --ext-folder ${source_dir}/servers/isaacsim --enable isaac.sim.mcp_extension"
    fi

    printf '\n'
    ok "Scene-gen setup complete!"
    info "Configure API keys in: ${mcp_settings_file}"
    info "  - QWEN_VL_API_KEY: required for object attribute inference"
    info "  - TRELLIS_URL: required for 3D model generation"
}

# ---------------------------------------------------------------------------
# Run standalone if executed directly (not sourced)
# ---------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_scene_gen
fi

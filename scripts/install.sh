#!/usr/bin/env bash
set -euo pipefail

# Principia Agent installer
# Usage: curl -fsSL https://raw.githubusercontent.com/principia-cloud/principia-agent/main/scripts/install.sh | bash

PRINCIPIA_HOME="${HOME}/.principia"
NODE_MIN_MAJOR=20
NODE_VERSION="v22.13.1"
REPO_TARBALL="https://github.com/principia-cloud/principia-agent/archive/refs/heads/main.tar.gz"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || error "Required command not found: $1"
}

# ---------------------------------------------------------------------------
# Detect OS & architecture
# ---------------------------------------------------------------------------

detect_platform() {
    local uname_s uname_m
    uname_s="$(uname -s)"
    uname_m="$(uname -m)"

    case "$uname_s" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="darwin" ;;
        *)       error "Unsupported OS: $uname_s" ;;
    esac

    case "$uname_m" in
        x86_64)  ARCH="x64" ;;
        aarch64) ARCH="arm64" ;;
        arm64)   ARCH="arm64" ;;
        *)       error "Unsupported architecture: $uname_m" ;;
    esac

    info "Detected platform: ${OS}-${ARCH}"
}

# ---------------------------------------------------------------------------
# Node.js
# ---------------------------------------------------------------------------

node_major_version() {
    local ver
    ver="$("$1" --version 2>/dev/null)" || return 1
    ver="${ver#v}"
    echo "${ver%%.*}"
}

ensure_node() {
    # Check system Node.js first
    if command -v node >/dev/null 2>&1; then
        local major
        major="$(node_major_version node)"
        if [ "$major" -ge "$NODE_MIN_MAJOR" ] 2>/dev/null; then
            ok "System Node.js $(node --version) satisfies >=20"
            return
        fi
        warn "System Node.js $(node --version) is too old (need >=20)"
    fi

    # Check previously-installed portable Node.js
    if [ -x "${PRINCIPIA_HOME}/node/bin/node" ]; then
        local major
        major="$(node_major_version "${PRINCIPIA_HOME}/node/bin/node")"
        if [ "$major" -ge "$NODE_MIN_MAJOR" ] 2>/dev/null; then
            export PATH="${PRINCIPIA_HOME}/node/bin:${PATH}"
            ok "Using portable Node.js $(node --version)"
            return
        fi
        warn "Portable Node.js is outdated — re-downloading"
        rm -rf "${PRINCIPIA_HOME}/node"
    fi

    # Download portable Node.js
    info "Downloading Node.js ${NODE_VERSION} for ${OS}-${ARCH}..."

    local node_dir="${PRINCIPIA_HOME}/node"
    local archive_name="node-${NODE_VERSION}-${OS}-${ARCH}"
    local url="https://nodejs.org/dist/${NODE_VERSION}/${archive_name}.tar.gz"

    mkdir -p "${node_dir}"
    curl -fsSL "$url" | tar xz -C "${node_dir}" --strip-components=1 \
        || error "Failed to download Node.js from ${url}"

    export PATH="${node_dir}/bin:${PATH}"
    ok "Installed portable Node.js $(node --version) → ${node_dir}"
}

# ---------------------------------------------------------------------------
# Download & extract source
# ---------------------------------------------------------------------------

download_source() {
    local source_dir="${PRINCIPIA_HOME}/source"

    if [ -d "$source_dir" ]; then
        info "Removing previous installation..."
        rm -rf "$source_dir"
    fi

    mkdir -p "$source_dir"

    info "Downloading Principia Agent source..."
    curl -fsSL "$REPO_TARBALL" | tar xz -C "$source_dir" --strip-components=1 \
        || error "Failed to download source from ${REPO_TARBALL}"

    ok "Source extracted → ${source_dir}"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build() {
    local source_dir="${PRINCIPIA_HOME}/source"

    info "Installing dependencies (npm install)..."
    (cd "$source_dir" && npm install --no-audit --no-fund) \
        || error "npm install failed"

    info "Building CLI (production)..."
    (cd "$source_dir" && npm run cli:build:production) \
        || error "Build failed"

    ok "Build complete"
}

# ---------------------------------------------------------------------------
# Symlink
# ---------------------------------------------------------------------------

create_symlink() {
    local bin_dir="${PRINCIPIA_HOME}/bin"
    local target="${PRINCIPIA_HOME}/source/cli/dist/cli.mjs"

    mkdir -p "$bin_dir"

    if [ ! -f "$target" ]; then
        error "Build artifact not found: ${target}"
    fi

    ln -sf "$target" "${bin_dir}/principia"
    chmod +x "$target"

    ok "Symlink created: ${bin_dir}/principia → ${target}"
}

# ---------------------------------------------------------------------------
# PATH configuration
# ---------------------------------------------------------------------------

configure_path() {
    local bin_dir="${PRINCIPIA_HOME}/bin"
    local path_line="export PATH=\"${bin_dir}:\$PATH\""

    # Already on PATH?
    case ":${PATH}:" in
        *":${bin_dir}:"*) ok "PATH already includes ${bin_dir}"; return ;;
    esac

    local shells_updated=()

    # Bash
    for rc in "${HOME}/.bashrc" "${HOME}/.bash_profile"; do
        if [ -f "$rc" ]; then
            if ! grep -qF "$bin_dir" "$rc" 2>/dev/null; then
                printf '\n# Principia Agent\n%s\n' "$path_line" >> "$rc"
                shells_updated+=("$rc")
            fi
            break
        fi
    done

    # Zsh
    if [ -f "${HOME}/.zshrc" ]; then
        if ! grep -qF "$bin_dir" "${HOME}/.zshrc" 2>/dev/null; then
            printf '\n# Principia Agent\n%s\n' "$path_line" >> "${HOME}/.zshrc"
            shells_updated+=("${HOME}/.zshrc")
        fi
    fi

    # Fish
    local fish_conf="${HOME}/.config/fish/config.fish"
    if [ -f "$fish_conf" ]; then
        if ! grep -qF "$bin_dir" "$fish_conf" 2>/dev/null; then
            printf '\n# Principia Agent\nfish_add_path %s\n' "$bin_dir" >> "$fish_conf"
            shells_updated+=("$fish_conf")
        fi
    fi

    if [ ${#shells_updated[@]} -gt 0 ]; then
        ok "Added ${bin_dir} to PATH in: ${shells_updated[*]}"
    else
        warn "Could not detect shell config — add this to your shell profile manually:"
        warn "  ${path_line}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    printf '\n\033[1m  Principia Agent Installer\033[0m\n\n'

    need_cmd curl
    need_cmd tar

    detect_platform
    ensure_node
    download_source
    build
    create_symlink
    configure_path

    printf '\n\033[1;32m  Installation complete!\033[0m\n\n'
    printf '  Run \033[1mprincipia\033[0m to get started.\n'
    printf '  If the command is not found, restart your shell:\n'
    printf '    exec $SHELL\n\n'
}

main "$@"

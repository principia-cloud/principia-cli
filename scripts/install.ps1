#Requires -Version 5.1
<#
.SYNOPSIS
    Principia Agent installer for Windows.
.DESCRIPTION
    Downloads, builds, and installs the Principia Agent CLI on Windows.
.EXAMPLE
    irm https://raw.githubusercontent.com/principia-cloud/principia-agent/main/scripts/install.ps1 | iex
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PRINCIPIA_HOME = Join-Path $env:USERPROFILE ".principia"
$NODE_MIN_MAJOR = 20
$NODE_VERSION = "v22.13.1"
$REPO_OWNER = "principia-cloud"
$REPO_NAME = "principia-cli"
$REPO_BRANCH = "main"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Info  { param([string]$Msg) Write-Host "[info]  $Msg" -ForegroundColor Blue }
function Write-Ok    { param([string]$Msg) Write-Host "[ok]    $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "[warn]  $Msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host "[error] $Msg" -ForegroundColor Red; throw $Msg }

function Test-Command {
    param([string]$Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

# ---------------------------------------------------------------------------
# Detect architecture
# ---------------------------------------------------------------------------

function Get-Arch {
    switch ($env:PROCESSOR_ARCHITECTURE) {
        "AMD64" { return "x64" }
        "ARM64" { return "arm64" }
        default { Write-Err "Unsupported architecture: $env:PROCESSOR_ARCHITECTURE" }
    }
}

# ---------------------------------------------------------------------------
# Node.js
# ---------------------------------------------------------------------------

function Get-NodeMajorVersion {
    param([string]$NodePath)
    try {
        $ver = & $NodePath --version 2>$null
        if ($ver -match "^v(\d+)") { return [int]$Matches[1] }
    } catch {}
    return 0
}

function Install-Node {
    $arch = Get-Arch

    # Check system Node.js
    if (Test-Command "node") {
        $major = Get-NodeMajorVersion "node"
        if ($major -ge $NODE_MIN_MAJOR) {
            Write-Ok "System Node.js $(node --version) satisfies >=$NODE_MIN_MAJOR"
            return
        }
        Write-Warn "System Node.js $(node --version) is too old (need >=$NODE_MIN_MAJOR)"
    }

    # Check previously-installed portable Node.js
    $portableNode = Join-Path $PRINCIPIA_HOME "node\node.exe"
    if (Test-Path $portableNode) {
        $major = Get-NodeMajorVersion $portableNode
        if ($major -ge $NODE_MIN_MAJOR) {
            $nodeBin = Join-Path $PRINCIPIA_HOME "node"
            $env:PATH = "$nodeBin;$env:PATH"
            Write-Ok "Using portable Node.js $(node --version)"
            return
        }
        Write-Warn "Portable Node.js is outdated - re-downloading"
        Remove-Item (Join-Path $PRINCIPIA_HOME "node") -Recurse -Force
    }

    # Download portable Node.js
    Write-Info "Downloading Node.js $NODE_VERSION for win-$arch..."

    $nodeDir = Join-Path $PRINCIPIA_HOME "node"
    $archiveName = "node-$NODE_VERSION-win-$arch"
    $url = "https://nodejs.org/dist/$NODE_VERSION/$archiveName.zip"
    $tempZip = Join-Path $env:TEMP "$archiveName.zip"

    try {
        Invoke-WebRequest -Uri $url -OutFile $tempZip -UseBasicParsing
    } catch {
        Write-Err "Failed to download Node.js from $url"
    }

    # Extract and flatten (zip contains a top-level folder)
    $tempExtract = Join-Path $env:TEMP "principia-node-extract"
    if (Test-Path $tempExtract) { Remove-Item $tempExtract -Recurse -Force }
    Expand-Archive -Path $tempZip -DestinationPath $tempExtract -Force

    $extractedDir = Get-ChildItem $tempExtract -Directory | Select-Object -First 1
    if (-not $extractedDir) { Write-Err "Failed to extract Node.js archive" }

    New-Item -ItemType Directory -Path $nodeDir -Force | Out-Null
    Copy-Item -Path (Join-Path $extractedDir.FullName "*") -Destination $nodeDir -Recurse -Force

    Remove-Item $tempZip -Force -ErrorAction SilentlyContinue
    Remove-Item $tempExtract -Recurse -Force -ErrorAction SilentlyContinue

    $env:PATH = "$nodeDir;$env:PATH"
    Write-Ok "Installed portable Node.js $(node --version) -> $nodeDir"
}

# ---------------------------------------------------------------------------
# Download & extract source
# ---------------------------------------------------------------------------

function Get-GitHubToken {
    if ($env:GITHUB_TOKEN) { return $env:GITHUB_TOKEN }
    if (Test-Command "gh") {
        try {
            $token = & gh auth token 2>$null
            if ($LASTEXITCODE -eq 0 -and $token) { return $token }
        } catch {}
    }
    return $null
}

function Install-Source {
    $sourceDir = Join-Path $PRINCIPIA_HOME "source"

    if (Test-Path $sourceDir) {
        Write-Info "Removing previous installation..."
        Remove-Item $sourceDir -Recurse -Force
    }

    New-Item -ItemType Directory -Path $sourceDir -Force | Out-Null
    Write-Info "Downloading Principia Agent source..."

    $tempTar = Join-Path $env:TEMP "principia-source.tar.gz"

    $token = Get-GitHubToken
    if ($token) {
        $headers = @{
            "Authorization" = "token $token"
            "Accept" = "application/vnd.github+json"
        }
        $url = "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/tarball/$REPO_BRANCH"
        try {
            Invoke-WebRequest -Uri $url -OutFile $tempTar -Headers $headers -UseBasicParsing
        } catch {
            Write-Err "Failed to download source (authenticated)"
        }
    } else {
        $url = "https://github.com/$REPO_OWNER/$REPO_NAME/archive/refs/heads/$REPO_BRANCH.tar.gz"
        try {
            Invoke-WebRequest -Uri $url -OutFile $tempTar -UseBasicParsing
        } catch {
            Write-Err "Failed to download source from $url"
        }
    }

    # Extract with tar (available on Windows 10+)
    & tar xzf $tempTar -C $sourceDir --strip-components=1
    if ($LASTEXITCODE -ne 0) { Write-Err "Failed to extract source archive" }

    Remove-Item $tempTar -Force -ErrorAction SilentlyContinue
    Write-Ok "Source extracted -> $sourceDir"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

function Build-CLI {
    $sourceDir = Join-Path $PRINCIPIA_HOME "source"

    Write-Info "Installing dependencies (npm install)..."
    Push-Location $sourceDir
    try {
        & npm install --no-audit --no-fund --loglevel=error
        if ($LASTEXITCODE -ne 0) { Write-Err "npm install failed" }

        Write-Info "Building CLI (production)..."
        & npm run cli:build:production
        if ($LASTEXITCODE -ne 0) { Write-Err "Build failed" }
    } finally {
        Pop-Location
    }

    Write-Ok "Build complete"
}

# ---------------------------------------------------------------------------
# Knowledge base & embedding model
# ---------------------------------------------------------------------------

$KB_API_URL = "https://2knihjo39k.execute-api.us-east-1.amazonaws.com/dev/kb/presigned-url"
$KB_SIMULATOR = "isaac_sim"

function Install-KnowledgeBase {
    $kbDir = Join-Path $PRINCIPIA_HOME "data\knowledge-base"
    $sourceDir = Join-Path $PRINCIPIA_HOME "source"

    # Skip if already downloaded
    if (Test-Path (Join-Path $kbDir "simulator_kb.lance")) {
        Write-Ok "Knowledge base already exists at $kbDir"
    } else {
        Write-Info "Downloading knowledge base for $KB_SIMULATOR..."
        New-Item -ItemType Directory -Path $kbDir -Force | Out-Null

        # Get presigned URL from API
        try {
            $body = @{ simulator = $KB_SIMULATOR } | ConvertTo-Json
            $response = Invoke-RestMethod -Uri $KB_API_URL -Method Post -Body $body -ContentType "application/json"
            $downloadUrl = $response.url
        } catch {
            Write-Err "Failed to get KB download URL"
        }

        # Download and extract
        $tempKb = Join-Path $env:TEMP "principia-kb.tar.gz"
        try {
            Invoke-WebRequest -Uri $downloadUrl -OutFile $tempKb -UseBasicParsing
        } catch {
            Write-Err "Failed to download knowledge base"
        }

        & tar xzf $tempKb -C $kbDir --exclude='._*'
        if ($LASTEXITCODE -ne 0) { Write-Err "Failed to extract knowledge base" }

        Remove-Item $tempKb -Force -ErrorAction SilentlyContinue
        Write-Ok "Knowledge base downloaded -> $kbDir"
    }

    # Pre-download embedding model
    Write-Info "Warming up embedding model..."
    Push-Location $sourceDir
    try {
        & node scripts/warm-embedding-model.mjs
        if ($LASTEXITCODE -ne 0) { throw "non-zero exit" }
    } catch {
        Write-Warn "Failed to pre-download embedding model (will download on first use)"
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# Create wrapper script
# ---------------------------------------------------------------------------

function New-Wrapper {
    $binDir = Join-Path $PRINCIPIA_HOME "bin"
    $target = Join-Path $PRINCIPIA_HOME "source\cli\dist\cli.mjs"

    New-Item -ItemType Directory -Path $binDir -Force | Out-Null

    if (-not (Test-Path $target)) {
        Write-Err "Build artifact not found: $target"
    }

    # Create a .cmd wrapper (Windows equivalent of a symlink to a node script)
    $wrapperPath = Join-Path $binDir "principia.cmd"
    $wrapperContent = "@echo off`r`nnode `"$target`" %*"
    Set-Content -Path $wrapperPath -Value $wrapperContent -Encoding ASCII

    Write-Ok "Wrapper created: $wrapperPath"
}

# ---------------------------------------------------------------------------
# PATH configuration
# ---------------------------------------------------------------------------

function Set-PrincipiaPath {
    $binDir = Join-Path $PRINCIPIA_HOME "bin"

    # Already on PATH?
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -and $currentPath.Split(";") -contains $binDir) {
        Write-Ok "PATH already includes $binDir"
        return
    }

    # Add to persistent user PATH
    $newPath = if ($currentPath) { "$binDir;$currentPath" } else { $binDir }
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")

    # Also update current session
    $env:PATH = "$binDir;$env:PATH"

    Write-Ok "Added $binDir to user PATH (restart your terminal for changes to take effect)"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function Main {
    Write-Host ""
    Write-Host "  ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗██╗██████╗ ██╗ █████╗ " -ForegroundColor Cyan
    Write-Host "  ██╔══██╗██╔══██╗██║████╗  ██║██╔════╝██║██╔══██╗██║██╔══██╗" -ForegroundColor Cyan
    Write-Host "  ██████╔╝██████╔╝██║██╔██╗ ██║██║     ██║██████╔╝██║███████║" -ForegroundColor Blue
    Write-Host "  ██╔═══╝ ██╔══██╗██║██║╚██╗██║██║     ██║██╔═══╝ ██║██╔══██║" -ForegroundColor Blue
    Write-Host "  ██║     ██║  ██║██║██║ ╚████║╚██████╗██║██║     ██║██║  ██║" -ForegroundColor Magenta
    Write-Host "  ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "                        Agent Installer" -ForegroundColor Cyan
    Write-Host ""

    # Verify tar is available (Windows 10 1803+)
    if (-not (Test-Command "tar")) {
        Write-Err "tar command not found. Windows 10 version 1803 or later is required."
    }

    Install-Node
    Install-Source
    Build-CLI
    Install-KnowledgeBase
    New-Wrapper
    Set-PrincipiaPath

    Write-Host ""
    Write-Host "  Installation complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Run " -NoNewline
    Write-Host "principia" -ForegroundColor Green -NoNewline
    Write-Host " to get started."
    Write-Host "  If the command is not found, restart your terminal."
    Write-Host ""
}

Main

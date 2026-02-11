```
    ____       _            _       _
   / __ \_____(_)___  _____(_)___  (_)___ _
  / /_/ / ___/ / __ \/ ___/ / __ \/ / __ `/
 / ____/ /  / / / / / /__/ / /_/ / / /_/ /
/_/   /_/  /_/_/ /_/\___/_/ .___/_/\__,_/
                         /_/
```

# Principia — AI Agent for Robotics Simulation

Principia is an open-source CLI agent that builds robotics simulations from natural language. Describe what you want — a robotic arm picking up objects, a quadruped navigating terrain, a warehouse fleet — and Principia writes the code, configures the physics, and runs the simulation. Think of it as Cursor for robotics.

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.com/invite/ZrvJpUVK56)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-principia.cloud-orange)](https://principia.cloud/agent)

</div>

---

## What can Principia do?

- **Generate simulation code** from plain English descriptions
- **Read, write, and execute code** across your robotics project with your permission
- **Configure physics and environments** for NVIDIA Isaac Sim and other simulators
- **Iterate interactively** — refine your simulation through conversation
- **Plan before acting** — use Plan mode to discuss architecture, then switch to Act mode to execute
- **Pipe and script** — integrate into CI/CD pipelines with JSON output and yolo mode

## Quick Install

```bash
curl -fsSL https://principia.cloud/install.sh | bash
```

Or install via npm:

```bash
npm i -g principia
```

## CLI Usage

### Interactive Mode (Default)

```bash
# Launch interactive mode
principia

# Run a task directly
principia "Create a pick-and-place simulation for a UR5 arm"

# With verbose output and extended thinking
principia -v --thinking "Analyze this simulation codebase"
```

### Commands

#### `task` (alias: `t`)

Run a new task with a prompt.

```bash
principia task "Set up a quadruped robot walking on uneven terrain"
principia t "Add collision detection to the gripper"
```

| Option | Description |
|--------|-------------|
| `-a, --act` | Run in act mode (default) |
| `-p, --plan` | Run in plan mode |
| `-y, --yolo` | Auto-approve all actions (plain text output) |
| `-m, --model <model>` | Model to use for the task |
| `-i, --images <paths...>` | Image file paths to include |
| `-v, --verbose` | Show verbose output including reasoning |
| `-c, --cwd <path>` | Working directory for the task |
| `--config <path>` | Path to configuration directory |
| `--thinking [tokens]` | Enable extended thinking (default: 1024) |
| `--json` | Output messages as JSON |
| `-T, --taskId <id>` | Resume an existing task by ID |

#### `history` (alias: `h`)

List task history with pagination.

```bash
principia history
principia history -n 20 -p 2
```

#### `config`

Show current configuration.

```bash
principia config
```

#### `auth`

Authenticate a provider and configure which model to use.

```bash
# Interactive
principia auth

# Quick setup
principia auth -p anthropic -k sk-ant-xxxxx -m claude-sonnet-4-5-20250929
```

#### `update`

Check for updates and install if available.

```bash
principia update
```

### Piped Input

```bash
cat scene.py | principia "Add a second robot arm to this scene"
git diff | principia "Review these simulation changes"
```

### Scripting & Automation

```bash
# JSON output for parsing
principia --json "List all robot joints" | jq '.text'

# Yolo mode for CI/CD
principia -y "Run the test suite and fix failures"
```

### Resuming Tasks

```bash
# Get task IDs from history
principia history

# Resume a task
principia -T abc123def

# Resume with a follow-up
principia -T abc123def "Now add unit tests for the changes"
```

## Configuration

Principia stores its data in `~/.principia/data/` by default:

```
~/.principia/
├── data/
│   ├── globalState.json     # Global settings and state
│   ├── secrets.json         # API keys and secrets
│   ├── workspace/           # Workspace-specific state
│   └── tasks/               # Task history and conversation data
└── log/                     # Log files
```

Override with `--config <path>` or the `PRINCIPIA_DIR` environment variable.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PRINCIPIA_DIR` | Override the default configuration directory |
| `PRINCIPIA_COMMAND_PERMISSIONS` | JSON config restricting which shell commands Principia can execute |

## Architecture

The CLI directly imports and reuses the core TypeScript codebase. This means feature parity is easy to maintain — when core gets updated, the CLI automatically benefits.

```
┌─────────────────────────────────────────────────────────┐
│                     CLI (cli/)                          │
│  - React Ink terminal UI                                │
│  - Command parsing (commander)                          │
│  - Terminal-specific adapters                           │
└─────────────────────────────────────────────────────────┘
                          │
                          │ direct imports
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Core (src/core/)                       │
│  - Controller: task lifecycle, state management         │
│  - Task: AI API calls, tool execution                   │
│  - StateManager: persistent storage                     │
│  - Proto types: message definitions                     │
└─────────────────────────────────────────────────────────┘
```

The CLI runs everything in a single Node.js process. The "host bridge" pattern provides terminal-appropriate implementations for things the VS Code extension would handle differently (clipboard, file dialogs, etc.).

## Attribution

Built on top of [Cline](https://github.com/cline/cline) (Apache-2.0).

## Links

- [Website](https://principia.cloud/agent)
- [Discord](https://discord.com/invite/ZrvJpUVK56)
- [GitHub Issues](https://github.com/principia-cloud/principia-agent/issues)

## License

[Apache 2.0](LICENSE) &copy; 2026 Principia Cloud

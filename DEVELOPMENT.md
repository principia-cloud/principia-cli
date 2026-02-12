# Development Guide

## Project Structure

```
principia-cli/
├── cli/              # CLI package (React Ink terminal UI, Commander.js)
│   ├── src/          # CLI source code
│   ├── dist/         # Built output
│   └── package.json
├── src/
│   ├── core/         # Controller, Task, StateManager
│   ├── shared/       # Shared types and utilities
│   ├── services/     # Auth, telemetry, MCP, error handling
│   ├── integrations/ # Editor, terminal, external tool integrations
│   └── generated/    # Proto-generated TypeScript types
├── proto/            # Protocol buffer definitions
└── scripts/          # Build and utility scripts
```

## Quick Start

1. **Clone the repository**:

   ```bash
   git clone https://github.com/principia-cloud/principia-cli.git
   cd principia-cli
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Generate proto types and build**:

   ```bash
   npm run cli:build
   ```

4. **Link for local development**:

   ```bash
   cd cli && npm run link
   ```

5. **Verify installation**:

   ```bash
   principia --help
   ```

## Development Workflow

### Scripts

Run these from the repository root:

| Script | Description |
|--------|-------------|
| `npm run cli:build` | Generate protos and build CLI |
| `npm run cli:build:production` | Production build (minified) |
| `npm run cli:watch` | Watch mode only (no initial build) |
| `npm run cli:dev` | Link + watch mode for development |
| `npm run cli:run` | Run CLI from dist directly |

### Development Loop

1. Run `npm run cli:dev` — links the CLI globally and starts watch mode
2. Make changes to files in `cli/src/`
3. The build automatically rebuilds on save
4. Test your changes by running `principia` in another terminal
5. When done, run `cd cli && npm run unlink` to clean up

<details>
<summary>Proto generation</summary>

The CLI uses proto-generated types for message passing. If you modify any `.proto` files:

```bash
npm run protos
```

This generates TypeScript types in `src/generated/` that both the CLI and core use.

</details>

## CLI Reference

### Interactive Mode

```bash
principia                                    # Launch interactive mode
principia "Create a simulation for a UR5"    # Run a task directly
principia -v --thinking "Analyze this code"  # Verbose with extended thinking
```

### Commands

<details>
<summary><code>task</code> (alias: <code>t</code>) — Run a task with a prompt</summary>

```bash
principia task "Set up a quadruped robot walking on uneven terrain"
principia t "Add collision detection to the gripper"
```

| Option | Description |
|--------|-------------|
| `-a, --act` | Run in act mode (default) |
| `-p, --plan` | Run in plan mode |
| `-y, --yolo` | Auto-approve all actions |
| `-m, --model <model>` | Model to use |
| `-i, --images <paths...>` | Image file paths to include |
| `-v, --verbose` | Show verbose output |
| `-c, --cwd <path>` | Working directory |
| `--config <path>` | Configuration directory |
| `--thinking [tokens]` | Enable extended thinking (default: 1024) |
| `--json` | Output as JSON |
| `-T, --taskId <id>` | Resume an existing task |

</details>

<details>
<summary><code>history</code> (alias: <code>h</code>) — List task history</summary>

```bash
principia history
principia history -n 20 -p 2
```

</details>

<details>
<summary><code>config</code> — Show current configuration</summary>

```bash
principia config
```

</details>

<details>
<summary><code>auth</code> — Authenticate a provider</summary>

```bash
principia auth                                                         # Interactive
principia auth -p anthropic -k sk-ant-xxxxx -m claude-sonnet-4-5-20250929  # Quick setup
```

</details>

<details>
<summary><code>update</code> — Check for and install updates</summary>

```bash
principia update
```

</details>

### Piped Input

```bash
cat scene.py | principia "Add a second robot arm to this scene"
git diff | principia "Review these simulation changes"
```

### Scripting & Automation

```bash
principia --json "List all robot joints" | jq '.text'    # JSON output
principia -y "Run the test suite and fix failures"        # Auto-approve for CI/CD
```

### Resuming Tasks

```bash
principia history                                         # Get task IDs
principia -T abc123def                                    # Resume a task
principia -T abc123def "Now add unit tests"               # Resume with follow-up
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

The CLI directly imports and reuses the core TypeScript codebase. When core gets updated, the CLI automatically benefits.

```
┌─────────────────────────────────────────────────────────┐
│                     CLI (cli/)                          │
│  - React Ink terminal UI                                │
│  - Command parsing (Commander.js)                       │
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

The CLI runs everything in a single Node.js process. The "host bridge" pattern provides terminal-appropriate implementations for things the VS Code extension handles differently (clipboard, file dialogs, etc.).

### Key Files

| File | Purpose |
|------|---------|
| `cli/src/index.ts` | Entry point, command definitions |
| `cli/src/components/App.tsx` | Main React Ink app |
| `cli/src/components/ChatView.tsx` | Task conversation UI |
| `cli/src/controllers/CliWebviewProvider.ts` | Bridges core messages to terminal output |
| `cli/src/vscode-context.ts` | Mock VS Code extension context |
| `cli/src/vscode-shim.ts` | Shims for VS Code APIs that core depends on |

<details>
<summary>React Ink details</summary>

The CLI uses [React Ink](https://github.com/vadimdemedes/ink) for its terminal UI. This lets us build the interface with React components that render to the terminal.

- Components in `cli/src/components/` render terminal UI
- Hooks in `cli/src/hooks/` manage terminal-specific state (size, scrolling)
- The `useStateSubscriber` hook subscribes to core state changes

</details>

## Code Formatting

This project uses [Biome](https://biomejs.dev/) for formatting and linting. Biome runs automatically during builds — no manual formatting needed.

## Publishing

### 1. Publish to npm

```bash
npm publish
```

### 2. Update the Homebrew formula

```bash
npm run update-brew-formula
```

<details>
<summary>Test Homebrew formula locally</summary>

```bash
# Create a local tap
brew tap-new principia/local
cp ./cli/principia.rb "$(brew --repository)/Library/Taps/principia/homebrew-local/Formula/principia.rb"

# Build from source
brew install --build-from-source principia/local/principia

# Clean up when done
brew untap principia/local
```

</details>

## Troubleshooting

<details>
<summary>Build errors</summary>

```bash
# Make sure all deps are installed
npm install

# Regenerate proto types
npm run protos

# Then rebuild
npm run cli:build
```

</details>

<details>
<summary>"command not found: principia"</summary>

The CLI isn't linked globally. Run:

```bash
cd cli && npm run link
```

</details>

<details>
<summary>Changes not reflected</summary>

1. Make sure watch mode is running (`npm run cli:dev`)
2. Check for TypeScript errors in the watch output
3. Try unlinking and relinking: `cd cli && npm run unlink && npm run link`

</details>

<details>
<summary>Import errors from core</summary>

The CLI imports from `@core/`, `@shared/`, etc. These paths are defined in the root `tsconfig.json`. If you see import errors, make sure you're building from the repo root, not from inside `cli/`.

</details>

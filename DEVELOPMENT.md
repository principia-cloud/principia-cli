# Principia CLI — Development Guide

The official CLI for Principia. Run Principia tasks directly from the terminal with the same underlying functionality as the core engine.

## Features

- **Reuses Core Codebase**: Shares the same Controller, Task, and API handling as the core engine
- **Terminal Output**: Displays Principia messages directly in your terminal with colored output
- **Task History**: Access your task history from the command line
- **Configurable**: Use custom configuration directories and working directories
- **Image Support**: Attach images to your prompts using file paths or inline references

## Prerequisites

- Node.js 20.x or later
- npm or yarn
- The parent Principia project dependencies installed

## Installation

From the repository root:

```bash
# Install all dependencies first
npm install

# Ensure protos are generated
npm run protos

# Build the CLI
npm run cli:build
```

## Development Workflow

### Quick Start

```bash
# 1. Install all dependencies
npm install

# 2. Build and link globally so you can run `principia` from anywhere
cd cli && npm run link

# 3. Test it
principia --help
```

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

1. Run `npm run cli:dev` — this links the CLI globally and starts watch mode
2. Make changes to files in `cli/src/`
3. The build automatically rebuilds on save
4. Test your changes by running `principia` in another terminal
5. When done, run `cd cli && npm run unlink` to clean up

### Proto Generation

The CLI uses proto-generated types for message passing. If you modify any `.proto` files, run:

```bash
npm run protos
```

This generates TypeScript types in `src/generated/` that both the CLI and core use.

## Publish

#### 1. Publish to npm
```bash
npm publish
```

#### 2. Update the Homebrew formula
```bash
npm run update-brew-formula
```

#### 3. Test the formula locally
```bash
# Create a local tap
brew tap-new principia/local
cp ./cli/principia.rb "$(brew --repository)/Library/Taps/principia/homebrew-local/Formula/principia.rb"

# Build from source
brew install --build-from-source principia/local/principia

# Clean up when done
brew untap principia/local
```

## Architecture

### How It Works

The CLI directly imports and reuses the core Principia TypeScript codebase. This means feature parity is easy to maintain — when core gets updated, the CLI automatically benefits.

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

Unlike a client-server architecture, the CLI runs everything in a single Node.js process. The "host bridge" pattern provides terminal-appropriate implementations for things like clipboard, file dialogs, etc.

### Key Files

| File | Purpose |
|------|---------|
| `cli/src/index.ts` | Entry point, command definitions |
| `cli/src/components/App.tsx` | Main React Ink app |
| `cli/src/components/ChatView.tsx` | Task conversation UI |
| `cli/src/controllers/CliWebviewProvider.ts` | Bridges core messages to terminal output |
| `cli/src/vscode-context.ts` | Mock VS Code extension context for core compatibility |
| `cli/src/vscode-shim.ts` | Shims for VS Code APIs that core depends on |
| `cli/src/constants/colors.ts` | Terminal color definitions |

### React Ink

The CLI uses [React Ink](https://github.com/vadimdemedes/ink) for its terminal UI. This lets us build the interface with React components that render to the terminal. Key patterns:

- Components in `cli/src/components/` render terminal UI
- Hooks in `cli/src/hooks/` manage terminal-specific state (size, scrolling)
- The `useStateSubscriber` hook subscribes to core state changes

## Configuration

The CLI stores its data in `~/.principia/data/` by default:

- `globalState.json`: Global settings and state
- `secrets.json`: API keys and secrets
- `workspace/`: Workspace-specific state
- `tasks/`: Task history and conversation data

Override with the `--config` option or `PRINCIPIA_DIR` environment variable.

## Troubleshooting

### Build Errors

If you encounter build errors:

```bash
# Make sure all deps are installed
npm install

# Regenerate proto types
npm run protos

# Then rebuild
npm run cli:build
```

### "command not found: principia"

The CLI isn't linked globally. Run:

```bash
cd cli && npm run link
```

### Changes Not Reflected

If your code changes aren't showing up:

1. Make sure watch mode is running (`npm run cli:dev`)
2. Check for TypeScript errors in the watch output
3. Try unlinking and relinking: `cd cli && npm run unlink && npm run link`

### Import Errors from Core

The CLI imports from `@core/`, `@shared/`, etc. These paths are defined in the root `tsconfig.json`. If you see import errors, make sure you're building from the repo root, not from inside `cli/`.

# Contributing to Principia

Thanks for your interest in contributing to Principia! This guide will help you get started.

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/principia-cloud/principia-agent/issues) with:

- A clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Node version, Principia version)

## Suggesting Features

Open a feature request on [GitHub Issues](https://github.com/principia-cloud/principia-agent/issues). Describe the use case and why it would be valuable.

## Development Setup

### Prerequisites

- Node.js 20.x or later
- npm

### Getting Started

```bash
# Clone the repo
git clone https://github.com/principia-cloud/principia-agent.git
cd principia-agent

# Install dependencies
npm install

# Generate proto types
npm run protos

# Build the CLI
npm run cli:build

# Link for local development
cd cli && npm run link
```

### Development Workflow

```bash
# Start watch mode (rebuilds on save)
npm run cli:dev

# Run tests
cd cli && npm test

# Type-check
cd cli && npm run typecheck
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full development guide.

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure the build passes: `npm run cli:build`
4. Ensure type-checking passes: `cd cli && npm run typecheck`
5. Open a PR with a clear description of what and why

### Code Style

This project uses [Biome](https://biomejs.dev/) for formatting and linting. The formatter runs automatically during the build. No manual formatting needed.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

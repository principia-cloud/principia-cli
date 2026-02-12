# Contributing to Principia

We appreciate your interest in contributing to Principia! Whether you're reporting bugs, suggesting features, improving docs, or submitting code, your contributions help improve the project for everyone.

## Reporting Bugs

1. **Check existing issues**: Search [GitHub Issues](https://github.com/principia-cloud/principia-cli/issues) to see if the bug has already been reported.
2. **Create a new issue**: If it hasn't been reported, open a new issue with:
   - A clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Node version, `principia --version`)
   - Relevant error messages or terminal output
3. **Label your issue**: Use the `bug` label.

## Suggesting Enhancements

1. **Check existing issues**: See if someone has already suggested something similar.
2. **Create a new issue**: Describe the use case, how the enhancement would work, and why it benefits Principia users.

## Code Formatting

This project uses [Biome](https://biomejs.dev/) for formatting and linting. The formatter runs automatically during builds. See the [Development Guide](DEVELOPMENT.md#code-formatting) for details.

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure the build passes: `npm run cli:build`
4. Ensure type-checking passes: `cd cli && npm run typecheck`
5. Open a PR with a clear description of what and why

## Documentation

Documentation improvements are always welcome. Fix typos, clarify explanations, add examples.

For detailed development setup and workflow, see the [Development Guide](DEVELOPMENT.md).

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

Join our [Discord](https://discord.com/invite/ZrvJpUVK56) to discuss ideas or get help with contributions.

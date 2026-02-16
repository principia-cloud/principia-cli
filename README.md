<div align="center">
  <a href="https://principia.cloud/agent" target="_blank" rel="noopener noreferrer">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="img/logo_white.png" width="420">
      <source media="(prefers-color-scheme: light)" srcset="img/logo.png" width="420">
      <img alt="Principia" width="420" src="img/logo.png">
    </picture>
  </a>

  <p>Build robotics simulations from natural language</p>

  <p>
    <a href="https://principia.cloud/agent"><img src="https://img.shields.io/badge/principia.cloud-orange" alt="Website"></a>
    <a href="https://discord.com/invite/ZrvJpUVK56"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
    <a href="https://x.com/PrincipiaSim"><img src="https://img.shields.io/twitter/follow/PrincipiaSim?style=social" alt="X (Twitter)"></a>
    <a href="https://github.com/principia-cloud/principia-cli"><img src="https://img.shields.io/github/stars/principia-cloud/principia-cli?style=social" alt="GitHub Stars"></a>
    <a href="https://github.com/principia-cloud/principia-cli/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

---

## What Principia Does

<table>
  <tr>
    <td align="center" valign="top" width="20%">
      <h4>Prompt to Sim</h4>
      <sub>Describe a simulation in plain English — Principia writes the code and runs it.</sub>
    </td>
    <td align="center" valign="top" width="20%">
      <h4>Code Execution</h4>
      <sub>Read, write, and execute code across your entire robotics project with permission.</sub>
    </td>
    <td align="center" valign="top" width="20%">
      <h4>Physics Config</h4>
      <sub>Configure environments and physics for Isaac Sim and other simulators.</sub>
    </td>
    <td align="center" valign="top" width="20%">
      <h4>Interactive</h4>
      <sub>Refine simulations through conversation — iterate without restarting.</sub>
    </td>
    <td align="center" valign="top" width="20%">
      <h4>Plan & Act</h4>
      <sub>Discuss architecture in Plan mode, then execute changes in Act mode.</sub>
    </td>
  </tr>
</table>

## See It in Action

<table>
  <tr>
    <td align="center" width="50%">
      <img src="img/scene-1.gif" alt="Natural Language to Sim" width="360">
      <br>
      <b>Natural Language to Sim</b>
      <br>
      <sub>Type a prompt — the agent launches Isaac Sim and spawns a Unitree H1 with full physics.</sub>
    </td>
    <td align="center" width="50%">
      <img src="img/scene-2.gif" alt="Live Scene Interaction" width="360">
      <br>
      <b>Live Scene Interaction</b>
      <br>
      <sub>"Add colorful boxes" — the agent generates and injects Python into the active sim.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="img/scene-3.gif" alt="Virtual Sensor Integration" width="360">
      <br>
      <b>Virtual Sensor Integration</b>
      <br>
      <sub>Attach a camera — the agent mounts an Intel RealSense D435 and starts a ROS 2 stream.</sub>
    </td>
    <td align="center" width="50%">
      <img src="img/scene-4.gif" alt="Agentic Debugging" width="360">
      <br>
      <b>Agentic Debugging</b>
      <br>
      <sub>"The orientation looks wrong" — the agent finds the transform error and corrects it live.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="img/scene-5.gif" alt="SOTA Model Integration via MCP" width="360">
      <br>
      <b>SOTA Model Integration via MCP</b>
      <br>
      <sub>Provide a 2D image — the agent reconstructs it as a 3D mesh and imports it with UsdPhysics.</sub>
    </td>
  </tr>
</table>

### Full Demo

https://github.com/user-attachments/assets/d7521dd5-2653-4c3a-8601-0634c4ae476e

## Quick Install

**macOS / Linux**

```bash
curl -fsSL https://principia.cloud/install.sh | bash
```

**Windows**

```powershell
irm https://principia.cloud/install.ps1 | iex
```

**Or [build from source](DEVELOPMENT.md#quick-start)**

## Set Up

Principia supports your favorite LLMs, model providers and API keys.

Principia works best with **Claude 4.6 Opus**; we recommend accessing this model via your **Claude Code** subscription.

**You don't have a Claude Code subscription?** Get a free Claude Code subscription on us for testing Principia! -> [fill out this form](https://docs.google.com/forms/d/e/1FAIpQLScet06zVcxfU-PhEiyHUrTBWha5_MjXd7n0-uj5VVLbIteyHw/viewform).

If you would like to try Principia with a free LLM (Worse performance), you can do so by selecting "Login with Cline" and choose one of the FREE models available (e.g. KIMI K2)

<img width="644" height="513" alt="Screenshot 2026-02-13 at 7 25 21 PM" src="https://github.com/user-attachments/assets/c3b89011-9822-4e2a-9014-49b06ea3dbfd" />


## Usage 

### Interactive Mode

```bash
principia
```

### Run with a Prompt

```bash
principia "Create a pick-and-place simulation for a UR5 arm"
```

See the [Development Guide](DEVELOPMENT.md#cli-reference) for the full CLI reference.

## Resources

- [Website](https://principia.cloud/agent) — Product page and demos
- [Discord](https://discord.com/invite/ZrvJpUVK56) — Community support and discussions
- [GitHub Issues](https://github.com/principia-cloud/principia-cli/issues) — Bug reports and feature requests
- [Development Guide](DEVELOPMENT.md) — Architecture, CLI reference, and setup

## Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Attribution

Built on [Cline](https://github.com/cline/cline) (Apache-2.0).

## License

[Apache 2.0](LICENSE) &copy; 2026 Principia Cloud

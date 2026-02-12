<div align="center">
  <a href="https://principia.cloud/agent" target="_blank" rel="noopener noreferrer">
    <img alt="Principia" width="200" src="img/logo.png">
  </a>

  <p>Build robotics simulations from natural language</p>

  <p>
    <a href="https://principia.cloud/agent"><img src="https://img.shields.io/badge/principia.cloud-orange" alt="Website"></a>
    <a href="https://discord.com/invite/ZrvJpUVK56"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
    <a href="https://github.com/principia-cloud/principia-cli"><img src="https://img.shields.io/github/stars/principia-cloud/principia-cli?style=social" alt="GitHub Stars"></a>
    <a href="https://github.com/principia-cloud/principia-cli/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

<div align="center">
  <video src="https://github.com/principia-cloud/principia-cli/raw/main/img/demo.mp4" width="720" controls></video>
</div>

---

## What Principia Does

- **Generate simulation code** from plain English descriptions
- **Read, write, and execute code** across your robotics project
- **Configure physics and environments** for NVIDIA Isaac Sim and other simulators
- **Iterate interactively** — refine simulations through conversation
- **Plan before acting** — discuss architecture in Plan mode, execute in Act mode

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

## Usage

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

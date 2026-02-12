import { execFile } from "node:child_process"
import osModule from "node:os"
import { promisify } from "node:util"
import { getShell } from "@utils/shell"
import osName from "os-name"
import { getWorkspacePaths } from "@/hosts/vscode/hostbridge/workspace/getWorkspacePaths"
import { SystemPromptSection } from "../templates/placeholders"
import { TemplateEngine } from "../templates/TemplateEngine"
import type { PromptVariant, SystemPromptContext } from "../types"

const execFileAsync = promisify(execFile)

const SYSTEM_INFO_TEMPLATE_TEXT = `SYSTEM INFORMATION

Operating System: {{os}}
IDE: {{ide}}
Default Shell: {{shell}}
Home Directory: {{homeDir}}
{{WORKSPACE_TITLE}}: {{workingDir}}`

/**
 * Get the shell that will actually be used for command execution.
 * When using background exec mode, commands run in the system default shell
 * (cmd.exe on Windows, /bin/bash on Unix), not the VS Code configured shell.
 */
function getEffectiveShell(context: SystemPromptContext): string {
	if (context.terminalExecutionMode === "backgroundExec") {
		// Background exec uses the system default shell, not VS Code config
		if (process.platform === "win32") {
			return process.env.COMSPEC || "cmd.exe"
		} else {
			return process.env.SHELL || "/bin/bash"
		}
	}
	// VS Code terminal mode (or undefined) uses the VS Code configured shell
	return getShell()
}

// Module-level cache for simulator environment detection
let cachedSimEnv: string | null = null

async function runQuiet(cmd: string, args: string[], timeoutMs = 3000): Promise<string> {
	const { stdout } = await execFileAsync(cmd, args, { timeout: timeoutMs })
	return stdout.trim()
}

async function detectPythonImport(module: string, timeoutMs = 5000): Promise<string | undefined> {
	try {
		return await runQuiet("python3", ["-c", `import ${module}; print(${module}.__version__)`], timeoutMs)
	} catch {
		return undefined
	}
}

/**
 * Detect simulator environment details (Python, Isaac Sim, MuJoCo, Genesis, ROS 2).
 * Results are cached after the first call.
 */
export async function detectSimulatorEnv(isTesting = false): Promise<string> {
	if (isTesting) {
		return ""
	}

	if (cachedSimEnv !== null) {
		return cachedSimEnv
	}

	const lines: string[] = []

	// Python version and path
	try {
		const [version, path] = await Promise.all([
			runQuiet("python3", ["--version"]),
			runQuiet("which", ["python3"]),
		])
		const ver = version.replace(/^Python\s+/, "")
		lines.push(`  Python: ${ver} (${path})`)
	} catch {
		// python3 not available
	}

	// Conda / venv environment
	const condaEnv = process.env.CONDA_DEFAULT_ENV
	const virtualEnv = process.env.VIRTUAL_ENV
	if (condaEnv) {
		lines.push(`  Conda Environment: ${condaEnv}`)
	} else if (virtualEnv) {
		const envName = virtualEnv.split("/").pop() || virtualEnv
		lines.push(`  Virtual Environment: ${envName} (${virtualEnv})`)
	}

	// Isaac Sim
	try {
		let isaacPath = process.env.ISAAC_SIM_PATH
		if (!isaacPath) {
			// Check common install locations
			const home = osModule.homedir()
			const candidates = [
				`${home}/.local/share/ov/pkg`,
				"/isaac-sim",
			]
			for (const dir of candidates) {
				try {
					const result = await runQuiet("ls", ["-d", `${dir}/isaac-sim-*`])
					if (result) {
						// Take the last match (highest version)
						const matches = result.split("\n").filter(Boolean)
						isaacPath = matches[matches.length - 1]
						break
					}
				} catch {
					// directory or glob didn't match
				}
			}
			// Also check if /isaac-sim itself exists (container installs)
			if (!isaacPath) {
				try {
					await runQuiet("test", ["-d", "/isaac-sim"])
					isaacPath = "/isaac-sim"
				} catch {
					// not found
				}
			}
		}
		if (isaacPath) {
			let ver = "detected"
			// 1. Try reading the VERSION file (most reliable)
			try {
				ver = await runQuiet("cat", [`${isaacPath}/VERSION`])
			} catch {
				// 2. Fall back to extracting version from path name (isaac-sim-X.Y.Z)
				const versionMatch = isaacPath.match(/isaac-sim[- ]?([\d.]+)/)
				if (versionMatch) {
					ver = versionMatch[1]
				}
			}
			lines.push(`  Isaac Sim: ${ver} (${isaacPath})`)
		}
	} catch {
		// Isaac Sim detection failed
	}

	// MuJoCo, Genesis — run in parallel
	const [mujocoVer, genesisVer] = await Promise.all([
		detectPythonImport("mujoco"),
		detectPythonImport("genesis"),
	])
	if (mujocoVer) {
		lines.push(`  MuJoCo: ${mujocoVer}`)
	}
	if (genesisVer) {
		lines.push(`  Genesis: ${genesisVer}`)
	}

	// ROS 2
	const rosDistro = process.env.ROS_DISTRO
	if (rosDistro) {
		lines.push(`  ROS 2: ${rosDistro}`)
	}

	cachedSimEnv =
		lines.length > 0
			? `\n\nSimulator Environment:\n${lines.join("\n")}`
			: `\n\nSimulator Environment:\n  No simulators were automatically detected. If the user's request involves a simulator (e.g., Isaac Sim, MuJoCo, Genesis), ask them which simulator they are using, its version, and its install path before proceeding.`
	return cachedSimEnv
}

/** Reset the cached simulator environment (for testing). */
export function resetSimulatorEnvCache(): void {
	cachedSimEnv = null
}

export async function getSystemEnv(context: SystemPromptContext, isTesting = false) {
	const currentWorkDir = context.cwd || process.cwd()
	const workspaces = (await getWorkspacePaths({}))?.paths || [currentWorkDir]
	return isTesting
		? {
			os: "macOS",
			ide: "TestIde",
			shell: "/bin/zsh",
			homeDir: "/Users/tester",
			workingDir: "/Users/tester/dev/project",
			// Multi-root workspace example: ["/Users/tester/dev/project", "/Users/tester/dev/foo", "/Users/tester/bar"],
			workspaces: ["/Users/tester/dev/project"],
		}
		: {
			os: osName(),
			ide: context.ide,
			shell: getEffectiveShell(context),
			homeDir: osModule.homedir(),
			workingDir: currentWorkDir,
			workspaces: workspaces,
		}
}

export async function getSystemInfo(variant: PromptVariant, context: SystemPromptContext): Promise<string> {
	const testMode = !!process?.env?.CI || !!process?.env?.IS_TEST || context.isTesting || false
	const info = await getSystemEnv(context, testMode)

	// Check if multi-root is enabled and we have workspace roots
	const isMultiRoot = context.isMultiRootEnabled && context.workspaceRoots && context.workspaceRoots.length > 1

	let WORKSPACE_TITLE: string
	let workingDirInfo: string

	if (isMultiRoot && context.workspaceRoots) {
		// Multi-root workspace with feature flag enabled
		WORKSPACE_TITLE = "Workspace Roots"
		const rootsInfo = context.workspaceRoots
			.map((root) => {
				const vcsInfo = root.vcs ? ` (${root.vcs})` : ""
				return `\n  - ${root.name}: ${root.path}${vcsInfo}`
			})
			.join("")
		workingDirInfo = rootsInfo + `\n\nPrimary Working Directory: ${context.cwd}`
	} else {
		// Single workspace
		WORKSPACE_TITLE = "Current Working Directory"
		workingDirInfo = info.workingDir
	}

	const template = variant.componentOverrides?.[SystemPromptSection.SYSTEM_INFO]?.template || SYSTEM_INFO_TEMPLATE_TEXT

	const baseInfo = new TemplateEngine().resolve(template, context, {
		os: info.os,
		ide: info.ide,
		shell: info.shell,
		homeDir: info.homeDir,
		WORKSPACE_TITLE,
		workingDir: workingDirInfo,
	})

	const simEnv = await detectSimulatorEnv(testMode)
	return baseInfo + simEnv
}

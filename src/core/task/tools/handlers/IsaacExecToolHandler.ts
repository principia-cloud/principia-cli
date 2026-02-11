import * as net from "net"
import * as os from "os"
import * as fs from "fs"
import * as path from "path"
import { spawn } from "child_process"
import { ClineAsk, ClineSayTool } from "@shared/ExtensionMessage"
import { ClineDefaultTool } from "@shared/tools"
import type { ToolUse } from "../../../assistant-message"
import { formatResponse } from "../../../prompts/responses"
import type { ToolResponse } from "../.."
import type { IFullyManagedTool } from "../ToolExecutorCoordinator"
import type { TaskConfig } from "../types/TaskConfig"
import type { StronglyTypedUIHelpers } from "../types/UIHelpers"

const DEFAULT_HOST = "127.0.0.1"
const DEFAULT_PORT = 8226
const CONNECT_TIMEOUT_MS = 5_000
const RESPONSE_TIMEOUT_MS = 30_000
const PORT_PROBE_TIMEOUT_MS = 1_000
const STARTUP_POLL_INTERVAL_MS = 3_000
const STARTUP_TIMEOUT_MS = 120_000

export class IsaacExecToolHandler implements IFullyManagedTool {
	readonly name = ClineDefaultTool.ISAAC_EXEC

	getDescription(block: ToolUse): string {
		const code = block.params.code || ""
		const preview = code.length > 60 ? code.slice(0, 57) + "..." : code
		return `[${block.name}: ${preview}]`
	}

	async handlePartialBlock(block: ToolUse, uiHelpers: StronglyTypedUIHelpers): Promise<void> {
		const code = block.params.code || ""
		const sharedMessageProps: ClineSayTool = {
			tool: "isaacExec",
			path: uiHelpers.removeClosingTag(block, "code", code),
			content: `Executing Python in Isaac Sim...`,
			operationIsLocatedInWorkspace: false,
		} satisfies ClineSayTool

		const partialMessage = JSON.stringify(sharedMessageProps)

		await uiHelpers.removeLastPartialMessageIfExistsWithType("say", "tool")
		await uiHelpers.ask("tool" as ClineAsk, partialMessage, block.partial).catch(() => {})
	}

	async execute(config: TaskConfig, block: ToolUse): Promise<ToolResponse> {
		const code: string | undefined = block.params.code
		const host: string | undefined = block.params.host
		const port: string | undefined = block.params.port
		const headless: string | undefined = block.params.headless

		if (!code) {
			config.taskState.consecutiveMistakeCount++
			return await config.callbacks.sayAndCreateMissingParamError(this.name, "code")
		}
		config.taskState.consecutiveMistakeCount = 0

		const resolvedHost = host || process.env.ISAAC_SIM_HOST || DEFAULT_HOST
		const resolvedPort = parseInt(port || process.env.ISAAC_SIM_PORT || String(DEFAULT_PORT), 10)

		// Emit tool-use message
		const sharedMessageProps: ClineSayTool = {
			tool: "isaacExec",
			path: code,
			content: `Executing Python in Isaac Sim (${resolvedHost}:${resolvedPort})...`,
			operationIsLocatedInWorkspace: false,
		}
		const completeMessage = JSON.stringify(sharedMessageProps)
		await config.callbacks.removeLastPartialMessageIfExistsWithType("ask", "tool")
		await config.callbacks.say("tool", completeMessage, undefined, undefined, false)

		// Check if Isaac Sim is already listening
		const alreadyRunning = await this.isPortReachable(resolvedHost, resolvedPort)
		if (!alreadyRunning) {
			// Attempt to auto-launch Isaac Sim
			const isaacPath = await this.findIsaacSimPath()
			if (!isaacPath) {
				return formatResponse.toolError(
					`Isaac Sim is not running on ${resolvedHost}:${resolvedPort} and could not be found on this system.\n\n` +
						`To fix this, either:\n` +
						`1. Start Isaac Sim manually with: isaac-sim.sh --enable isaacsim.code_editor.vscode\n` +
						`2. Set the ISAAC_SIM_PATH environment variable to your Isaac Sim installation directory\n` +
						`3. Install Isaac Sim via the Omniverse Launcher or as a container`,
				)
			}

			const useHeadless = headless === "true"
			try {
				this.launchIsaacSim(isaacPath, useHeadless)
			} catch (error) {
				return formatResponse.toolError(
					`Failed to launch Isaac Sim from ${isaacPath}: ${(error as Error).message}`,
				)
			}

			const mode = useHeadless ? "headless" : "headed"
			await config.callbacks.say(
				"tool",
				JSON.stringify({
					tool: "isaacExec",
					path: "",
					content: `Starting Isaac Sim (${mode}) from ${isaacPath}... This may take up to 2 minutes.`,
					operationIsLocatedInWorkspace: false,
				} satisfies ClineSayTool),
				undefined,
				undefined,
				false,
			)

			const ready = await this.waitForPort(resolvedHost, resolvedPort, STARTUP_TIMEOUT_MS)
			if (!ready) {
				return formatResponse.toolError(
					`Isaac Sim was launched from ${isaacPath} but did not become ready within ${STARTUP_TIMEOUT_MS / 1000}s. ` +
						`Check the Isaac Sim console/logs for errors. The process may still be starting — try again in a moment.`,
				)
			}
		}

		try {
			const result = await this.executeViaTcp(code, resolvedHost, resolvedPort)
			return formatResponse.toolResult(result)
		} catch (error) {
			return formatResponse.toolError((error as Error).message)
		}
	}

	/**
	 * Quick TCP probe to check if a port is reachable.
	 */
	private isPortReachable(host: string, port: number): Promise<boolean> {
		return new Promise((resolve) => {
			const socket = new net.Socket()
			const timer = setTimeout(() => {
				socket.destroy()
				resolve(false)
			}, PORT_PROBE_TIMEOUT_MS)

			socket.connect(port, host, () => {
				clearTimeout(timer)
				socket.destroy()
				resolve(true)
			})

			socket.on("error", () => {
				clearTimeout(timer)
				socket.destroy()
				resolve(false)
			})
		})
	}

	/**
	 * Detect Isaac Sim installation path using the same logic as system_info.ts.
	 */
	private async findIsaacSimPath(): Promise<string | undefined> {
		// 1. Explicit env var override
		const envPath = process.env.ISAAC_SIM_PATH
		if (envPath && this.isValidIsaacSimDir(envPath)) {
			return envPath
		}

		// 2. Omniverse Launcher installs: ~/.local/share/ov/pkg/isaac-sim-*
		const ovPkgDir = path.join(os.homedir(), ".local", "share", "ov", "pkg")
		try {
			const entries = fs.readdirSync(ovPkgDir, { withFileTypes: true })
			const isaacDirs = entries
				.filter((e) => e.isDirectory() && e.name.startsWith("isaac-sim-"))
				.map((e) => path.join(ovPkgDir, e.name))
				.sort()
			// Take the last (highest version)
			for (let i = isaacDirs.length - 1; i >= 0; i--) {
				if (this.isValidIsaacSimDir(isaacDirs[i])) {
					return isaacDirs[i]
				}
			}
		} catch {
			// directory doesn't exist
		}

		// 3. Container installs: /isaac-sim
		if (this.isValidIsaacSimDir("/isaac-sim")) {
			return "/isaac-sim"
		}

		return undefined
	}

	/**
	 * Check if a directory looks like a valid Isaac Sim installation.
	 */
	private isValidIsaacSimDir(dirPath: string): boolean {
		try {
			return fs.existsSync(path.join(dirPath, "isaac-sim.sh"))
		} catch {
			return false
		}
	}

	/**
	 * Launch Isaac Sim as a detached background process.
	 */
	private launchIsaacSim(isaacPath: string, headless: boolean): void {
		const script = headless ? "isaac-sim.streaming.sh" : "isaac-sim.sh"
		const scriptPath = path.join(isaacPath, script)

		if (!fs.existsSync(scriptPath)) {
			throw new Error(`Launch script not found: ${scriptPath}`)
		}

		const child = spawn(scriptPath, ["--enable", "isaacsim.code_editor.vscode"], {
			detached: true,
			stdio: "ignore",
			cwd: isaacPath,
		})
		child.unref()
	}

	/**
	 * Poll until a TCP port becomes reachable, or timeout.
	 */
	private async waitForPort(host: string, port: number, timeoutMs: number): Promise<boolean> {
		const deadline = Date.now() + timeoutMs
		while (Date.now() < deadline) {
			if (await this.isPortReachable(host, port)) {
				return true
			}
			await new Promise((resolve) => setTimeout(resolve, STARTUP_POLL_INTERVAL_MS))
		}
		return false
	}

	private executeViaTcp(code: string, host: string, port: number): Promise<string> {
		return new Promise((resolve, reject) => {
			const chunks: Buffer[] = []
			const socket = new net.Socket()

			const connectTimer = setTimeout(() => {
				socket.destroy()
				reject(new Error(`Connection to Isaac Sim timed out after ${CONNECT_TIMEOUT_MS / 1000}s (${host}:${port}). Is Isaac Sim running with the isaacsim.code_editor.vscode extension enabled?`))
			}, CONNECT_TIMEOUT_MS)

			let responseTimer: ReturnType<typeof setTimeout> | undefined

			socket.connect(port, host, () => {
				clearTimeout(connectTimer)
				socket.write(code)

				responseTimer = setTimeout(() => {
					socket.destroy()
					reject(new Error(`Isaac Sim did not respond within ${RESPONSE_TIMEOUT_MS / 1000}s. The code may be long-running — try again or check the Isaac Sim console.`))
				}, RESPONSE_TIMEOUT_MS)
			})

			socket.on("data", (chunk) => {
				chunks.push(chunk)
			})

			socket.on("close", () => {
				if (responseTimer) {
					clearTimeout(responseTimer)
				}
				const raw = Buffer.concat(chunks).toString("utf-8")
				if (!raw) {
					resolve("(no output)")
					return
				}

				try {
					const json = JSON.parse(raw)
					if (json.status === "ok") {
						resolve(json.output || "(executed successfully, no output)")
					} else {
						let msg = json.output || json.evalue || "Unknown error"
						if (json.traceback && json.traceback.length > 0) {
							msg += "\n\nTraceback:\n" + json.traceback.join("\n")
						}
						if (json.ename) {
							msg = `${json.ename}: ${msg}`
						}
						reject(new Error(msg))
					}
				} catch {
					// Not valid JSON — return raw output
					resolve(raw)
				}
			})

			socket.on("error", (err: NodeJS.ErrnoException) => {
				clearTimeout(connectTimer)
				if (responseTimer) {
					clearTimeout(responseTimer)
				}

				if (err.code === "ECONNREFUSED") {
					reject(new Error(`Cannot connect to Isaac Sim at ${host}:${port} — the simulator is not running or the isaacsim.code_editor.vscode extension is not enabled. Start Isaac Sim with: --enable isaacsim.code_editor.vscode`))
				} else if (err.code === "ETIMEDOUT") {
					reject(new Error(`Connection to Isaac Sim timed out (${host}:${port}). Check that the simulator is reachable and try again.`))
				} else {
					reject(new Error(`TCP error communicating with Isaac Sim: ${err.message}`))
				}
			})
		})
	}
}

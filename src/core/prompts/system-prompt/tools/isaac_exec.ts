import { ModelFamily } from "@/shared/prompts"
import { ClineDefaultTool } from "@/shared/tools"
import type { ClineToolSpec } from "../spec"

const GENERIC: ClineToolSpec = {
	variant: ModelFamily.GENERIC,
	id: ClineDefaultTool.ISAAC_EXEC,
	name: "isaac_exec",
	description: `Execute Python code inside an Isaac Sim instance via its TCP code-execution socket.
- If Isaac Sim is not already running, it will be started automatically (requires a local installation)
- The code runs in Isaac Sim's embedded Python interpreter with full access to the USD stage, physics, and all Omniverse APIs
- Use this for live scene manipulation, spawning prims, controlling physics, querying state, and running simulations
- Each invocation opens a fresh TCP connection (the server closes after responding)
- Default endpoint: 127.0.0.1:8226 (override with host/port parameters or ISAAC_SIM_HOST/ISAAC_SIM_PORT env vars)`,
	parameters: [
		{
			name: "code",
			required: true,
			instruction:
				"Python code to execute in Isaac Sim's Python scope. Has access to omni.usd, omni.isaac, pxr, and all loaded extensions.",
			usage: 'import omni.usd\nstage = omni.usd.get_context().get_stage()\nfor prim in stage.Traverse():\n    print(prim.GetPath())',
		},
		{
			name: "host",
			required: false,
			instruction: "TCP host of the Isaac Sim code-execution server (default: 127.0.0.1 or ISAAC_SIM_HOST env var)",
			usage: "127.0.0.1",
		},
		{
			name: "port",
			required: false,
			instruction: "TCP port of the Isaac Sim code-execution server (default: 8226 or ISAAC_SIM_PORT env var)",
			usage: "8226",
		},
		{
			name: "headless",
			required: false,
			instruction:
				"Set to 'true' to launch Isaac Sim in headless mode (no GUI window). Default: false (headed with GUI). Only used when Isaac Sim is not already running.",
			usage: "true",
		},
	],
}

const NATIVE_NEXT_GEN: ClineToolSpec = {
	variant: ModelFamily.NATIVE_NEXT_GEN,
	id: ClineDefaultTool.ISAAC_EXEC,
	name: "isaac_exec",
	description:
		"Execute Python code inside an Isaac Sim instance via TCP. Full access to USD stage, physics, and Omniverse APIs. Auto-launches Isaac Sim if not already running.",
	parameters: [
		{
			name: "code",
			required: true,
			instruction: "Python code to execute in Isaac Sim's Python scope",
		},
		{
			name: "host",
			required: false,
			instruction: "TCP host override (default: 127.0.0.1)",
		},
		{
			name: "port",
			required: false,
			instruction: "TCP port override (default: 8226)",
		},
		{
			name: "headless",
			required: false,
			instruction: "Set to 'true' to launch headless (no GUI). Default: false",
		},
	],
}

const NATIVE_GPT_5: ClineToolSpec = {
	...NATIVE_NEXT_GEN,
	variant: ModelFamily.NATIVE_GPT_5,
}

export const isaac_exec_variants = [GENERIC, NATIVE_NEXT_GEN, NATIVE_GPT_5]

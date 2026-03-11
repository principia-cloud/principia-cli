/**
 * Pattern to match simplified Principia CLI syntax: cline/principia "prompt" or cline/principia 'prompt'
 * with optional additional flags after the closing quote
 */
const CLINE_COMMAND_PATTERN = /^(cline|principia)\s+(['"])(.+?)\2(\s+.*)?$/

/**
 * Detects if a command is a Principia CLI subagent command.
 *
 * Matches the simplified syntax: cline/principia "prompt" or cline/principia 'prompt'
 * This allows the system to apply subagent-specific settings like autonomous execution.
 *
 * @param command - The command string to check
 * @returns True if the command is a Principia CLI subagent command, false otherwise
 */
export function isSubagentCommand(command: string): boolean {
	// Match simplified syntaxes
	// cline "prompt" / principia "prompt"
	// cline 'prompt' / principia 'prompt'
	return CLINE_COMMAND_PATTERN.test(command)
}

/**
 * Transforms simplified Principia CLI command syntax with subagent settings.
 *
 * Converts: cline/principia "prompt" or cline/principia 'prompt'
 * To: cline/principia "prompt" --json -y
 *
 * Preserves additional flags like --cwd:
 * principia "prompt" --cwd ./path → principia "prompt" --json -y --cwd ./path
 *
 * This enables autonomous subagent execution with proper CLI flags for automation.
 *
 * @param command - The command string to potentially transform
 * @returns The transformed command if it matches the pattern, otherwise the original command
 */
export function transformClineCommand(command: string): string {
	if (!isSubagentCommand(command)) {
		return command
	}

	// Inject subagent-specific command structure and settings
	const commandWithSettings = injectSubagentSettings(command)

	return commandWithSettings
}

/**
 * Injects subagent-specific command structure and settings into Principia CLI commands.
 *
 * @param command - The Principia CLI command (simplified or full syntax)
 * @returns The command with injected flags and settings
 */
function injectSubagentSettings(command: string): string {
	// No pre-prompt flags needed - use standard "cline 'prompt'" syntax
	const prePromptFlags: string[] = []

	// Flags/settings to insert after the prompt
	const postPromptFlags = ["--json", "-y"]

	const match = command.match(CLINE_COMMAND_PATTERN)

	if (match) {
		const cmd = match[1]
		const quote = match[2]
		const prompt = match[3]
		const additionalFlags = match[4] || ""
		const prePromptPart = prePromptFlags.length > 0 ? prePromptFlags.join(" ") + " " : ""
		return `${cmd} ${prePromptPart}${quote}${prompt}${quote} ${postPromptFlags.join(" ")}${additionalFlags}`
	}

	// Already full format: just inject settings after prompt
	const parts = command.split(" ")
	const promptEndIndex = parts.findIndex((p) => p.endsWith('"') || p.endsWith("'"))
	if (promptEndIndex !== -1) {
		parts.splice(promptEndIndex + 1, 0, ...postPromptFlags)
	}
	return parts.join(" ")
}

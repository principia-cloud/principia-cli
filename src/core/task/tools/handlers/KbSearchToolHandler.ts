import * as fs from "fs"
import * as os from "os"
import * as path from "path"
import { ClineAsk, ClineSayTool } from "@shared/ExtensionMessage"
import { ClineDefaultTool } from "@shared/tools"
import { KnowledgeBaseService } from "@services/knowledge-base"
import type { ToolUse } from "../../../assistant-message"
import { formatResponse } from "../../../prompts/responses"
import type { ToolResponse } from "../.."
import type { IFullyManagedTool } from "../ToolExecutorCoordinator"
import type { TaskConfig } from "../types/TaskConfig"
import type { StronglyTypedUIHelpers } from "../types/UIHelpers"

const DEFAULT_SIMULATOR = "isaac_sim"

export class KbSearchToolHandler implements IFullyManagedTool {
	readonly name = ClineDefaultTool.KB_SEARCH

	getDescription(block: ToolUse): string {
		return `[${block.name} for '${block.params.query}' (${DEFAULT_SIMULATOR} ${block.params.version})]`
	}

	async handlePartialBlock(block: ToolUse, uiHelpers: StronglyTypedUIHelpers): Promise<void> {
		const query = block.params.query || ""
		const version = block.params.version || ""
		const sharedMessageProps: ClineSayTool = {
			tool: "kbSearch",
			path: uiHelpers.removeClosingTag(block, "query", query),
			content: `Searching KB for: "${uiHelpers.removeClosingTag(block, "query", query)}" (${DEFAULT_SIMULATOR} ${version})`,
			operationIsLocatedInWorkspace: false,
		} satisfies ClineSayTool

		const partialMessage = JSON.stringify(sharedMessageProps)

		await uiHelpers.removeLastPartialMessageIfExistsWithType("say", "tool")
		await uiHelpers.ask("tool" as ClineAsk, partialMessage, block.partial).catch(() => {})
	}

	async execute(config: TaskConfig, block: ToolUse): Promise<ToolResponse> {
		const version: string | undefined = block.params.version
		const query: string | undefined = block.params.query
		const topK: string | undefined = block.params.top_k
		const sourceType: string | undefined = block.params.source_type

		if (!version) {
			config.taskState.consecutiveMistakeCount++
			return await config.callbacks.sayAndCreateMissingParamError(this.name, "version")
		}
		if (!query) {
			config.taskState.consecutiveMistakeCount++
			return await config.callbacks.sayAndCreateMissingParamError(this.name, "query")
		}
		config.taskState.consecutiveMistakeCount = 0

		// Emit tool-use message so the CLI displays what we're doing
		const sharedMessageProps: ClineSayTool = {
			tool: "kbSearch",
			path: query,
			content: `Searching KB for: "${query}" (${DEFAULT_SIMULATOR} ${version})`,
			operationIsLocatedInWorkspace: false,
		}
		const completeMessage = JSON.stringify(sharedMessageProps)
		await config.callbacks.removeLastPartialMessageIfExistsWithType("ask", "tool")
		await config.callbacks.say("tool", completeMessage, undefined, undefined, false)

		const lancedbPath = process.env.LANCEDB_PATH || path.join(os.homedir(), ".principia", "data", "knowledge-base")

		if (!fs.existsSync(lancedbPath)) {
			return formatResponse.toolError(
				`KB search database not found at: ${lancedbPath}\n` +
					"Re-run the installer to download it, or set LANCEDB_PATH to point to an existing database.",
			)
		}

		const kbService = new KnowledgeBaseService(lancedbPath)
		try {
			const results = await kbService.searchDocuments({
				simulator: DEFAULT_SIMULATOR,
				version,
				query,
				topK: topK ? parseInt(topK) : undefined,
				sourceType: sourceType || undefined,
			})

			if (results.length === 0) {
				await kbService.dispose()
				return formatResponse.toolResult(
					`No results found for query "${query}" in ${DEFAULT_SIMULATOR} ${version} knowledge base.`,
				)
			}

			let resultText = `Found ${results.length} relevant document(s) for "${query}":\n\n`
			for (const [i, result] of results.entries()) {
				resultText += `--- Result ${i + 1} ---\n`
				resultText += `Title: ${result.title}\n`
				resultText += `URL: ${result.source_url}\n`
				resultText += `Category: ${result.category}\n`
				resultText += `Type: ${result.source_type}\n`
				resultText += `Breadcrumbs: ${result.breadcrumbs}\n`
				resultText += `Matching chunks: ${result.matching_chunks}\n`
				resultText += `Relevance distance: ${result.best_distance.toFixed(4)}\n\n`
				resultText += `Content:\n${result.full_text}\n\n`
			}

			await kbService.dispose()
			return formatResponse.toolResult(resultText)
		} catch (error) {
			await kbService.dispose().catch(() => {})
			return formatResponse.toolError(`KB search failed: ${(error as Error).message}`)
		}
	}
}

import { ModelFamily } from "@/shared/prompts"
import { ClineDefaultTool } from "@/shared/tools"
import type { ClineToolSpec } from "../spec"

const GENERIC: ClineToolSpec = {
	variant: ModelFamily.GENERIC,
	id: ClineDefaultTool.KB_SEARCH,
	name: "kb_search",
	description: `Search the Isaac Sim knowledge base for relevant documentation, API references, and tutorials.
- Uses vector similarity search over pre-indexed Isaac Sim documentation
- Returns the most relevant document chunks for your query
- Use this tool when you need information about Isaac Sim APIs, configuration, or features
- Results include document titles, URLs, and content snippets`,
	parameters: [
		{
			name: "version",
			required: true,
			instruction: "The Isaac Sim version to search (e.g., '4.5.0', '5.0.0', '5.1.0', '6.0.0')",
			usage: "4.5.0",
		},
		{
			name: "query",
			required: true,
			instruction:
				"Natural language search query describing what you're looking for",
			usage: "How to create a rigid body with physics properties",
		},
		{
			name: "top_k",
			required: false,
			instruction: "Number of results to return (default: 5)",
			usage: "3",
		},
		{
			name: "source_type",
			required: false,
			instruction:
				"Filter by source type: 'docs' for documentation, 'api_reference' for API docs",
			usage: "docs",
		},
	],
}

const NATIVE_NEXT_GEN: ClineToolSpec = {
	variant: ModelFamily.NATIVE_NEXT_GEN,
	id: ClineDefaultTool.KB_SEARCH,
	name: "kb_search",
	description:
		"Search the Isaac Sim knowledge base for documentation, API references, and tutorials using vector similarity.",
	parameters: [
		{
			name: "version",
			required: true,
			instruction: "Isaac Sim version (e.g., '4.5.0', '5.1.0', '6.0.0')",
		},
		{
			name: "query",
			required: true,
			instruction: "Natural language search query",
		},
		{
			name: "top_k",
			required: false,
			instruction: "Number of results (default: 5)",
		},
		{
			name: "source_type",
			required: false,
			instruction: "Filter: 'docs' or 'api_reference'",
		},
	],
}

const NATIVE_GPT_5: ClineToolSpec = {
	...NATIVE_NEXT_GEN,
	variant: ModelFamily.NATIVE_GPT_5,
}

export const kb_search_variants = [GENERIC, NATIVE_NEXT_GEN, NATIVE_GPT_5]

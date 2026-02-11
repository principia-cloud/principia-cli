import { Embedder } from "./Embedder"
import type { KbSearchOptions, KbSearchResult } from "./types"

const TABLE_NAME = "simulator_kb"
const DOCUMENTS_TABLE_NAME = "simulator_kb_documents"

export class KnowledgeBaseService {
	private db: any | null = null
	private embedder = new Embedder()

	constructor(private dbPath: string) {}

	async connect(): Promise<void> {
		const lancedb = await import("@lancedb/lancedb")
		this.db = await lancedb.connect(this.dbPath)
	}

	async searchDocuments(options: KbSearchOptions): Promise<KbSearchResult[]> {
		if (!this.db) await this.connect()
		const table = await this.db!.openTable(TABLE_NAME)
		const queryVec = await this.embedder.embed(options.query)

		// Build filter — use parameterized-style escaping for safety
		const sim = options.simulator.replace(/'/g, "''")
		const ver = options.version.replace(/'/g, "''")
		let where = `simulator = '${sim}' AND version = '${ver}'`
		if (options.sourceType) {
			const st = options.sourceType.replace(/'/g, "''")
			where += ` AND source_type = '${st}'`
		}

		// Fetch more chunks than topK so we can group by document
		const topKChunks = (options.topK ?? 5) * 4
		const chunks = await table.vectorSearch(queryVec).where(where).limit(topKChunks).toArray()

		// Group by source_url — keep best chunk per document
		const docs = new Map<string, KbSearchResult>()
		for (const chunk of chunks) {
			const url = chunk.source_url as string
			const dist = (chunk._distance as number) ?? Infinity
			const existing = docs.get(url)

			if (!existing || dist < existing.best_distance) {
				docs.set(url, {
					title: chunk.title as string,
					source_url: url,
					breadcrumbs: chunk.breadcrumbs as string,
					category: chunk.category as string,
					source_type: chunk.source_type as string,
					best_chunk_text: chunk.text as string,
					full_text: chunk.text as string, // will be replaced by enrichWithFullText
					best_distance: dist,
					matching_chunks: existing ? existing.matching_chunks + 1 : 1,
					simulator: chunk.simulator as string,
					version: chunk.version as string,
				})
			} else {
				existing.matching_chunks++
			}
		}

		// Sort by distance, return top_k
		const topResults = [...docs.values()].sort((a, b) => a.best_distance - b.best_distance).slice(0, options.topK ?? 5)

		// Look up full document text from the documents table
		await this.enrichWithFullText(topResults, options.simulator, options.version)

		return topResults
	}

	private async enrichWithFullText(results: KbSearchResult[], simulator: string, version: string): Promise<void> {
		if (results.length === 0) return

		const tableNames = await this.db!.tableNames()
		if (!tableNames.includes(DOCUMENTS_TABLE_NAME)) {
			throw new Error(
				`Documents table "${DOCUMENTS_TABLE_NAME}" not found in knowledge base. ` +
					"The KB was built without full document text. Re-build the knowledge base to include it.",
			)
		}

		const docsTable = await this.db!.openTable(DOCUMENTS_TABLE_NAME)
		const sim = simulator.replace(/'/g, "''")
		const ver = version.replace(/'/g, "''")

		const urls = results.map((r) => `'${r.source_url.replace(/'/g, "''")}'`)
		const where = `simulator = '${sim}' AND version = '${ver}' AND source_url IN (${urls.join(", ")})`

		const rows = await docsTable.query().where(where).toArray()
		const fullTextByUrl = new Map<string, string>()
		for (const row of rows) {
			fullTextByUrl.set(row.source_url as string, row.full_text as string)
		}

		const missing: string[] = []
		for (const r of results) {
			const fullText = fullTextByUrl.get(r.source_url)
			if (!fullText) {
				missing.push(r.source_url)
			}
			r.full_text = fullText!
		}

		if (missing.length > 0) {
			throw new Error(
				`Full text not found in documents table for ${missing.length} result(s):\n` +
					missing.map((u) => `  - ${u}`).join("\n"),
			)
		}
	}

	async dispose(): Promise<void> {
		await this.embedder.dispose()
		this.db = null
	}
}

export interface KbSearchResult {
	title: string
	source_url: string
	breadcrumbs: string
	category: string
	source_type: string
	best_chunk_text: string
	full_text: string
	best_distance: number
	matching_chunks: number
	simulator: string
	version: string
}

export interface KbSearchOptions {
	simulator: string
	version: string
	query: string
	topK?: number
	sourceType?: string
}

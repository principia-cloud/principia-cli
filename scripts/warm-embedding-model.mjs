#!/usr/bin/env node
// Pre-downloads the HuggingFace embedding model so kb_search works offline.
// Run after `npm install` to populate the HuggingFace cache.

import { pipeline } from "@huggingface/transformers"

console.log("Downloading embedding model (Xenova/all-MiniLM-L6-v2)...")
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "fp32" })
// Run a dummy embed to ensure everything is cached
await extractor("warmup", { pooling: "mean", normalize: true })
await extractor.dispose()
console.log("Embedding model cached successfully.")

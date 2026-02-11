import type { FeatureExtractionPipeline } from "@huggingface/transformers"

export class Embedder {
	private extractor: FeatureExtractionPipeline | null = null
	private static exitOverrideInstalled = false

	async initialize(): Promise<void> {
		// Dynamic import + cast to avoid TS2590 (union type too complex)
		const { pipeline } = await import("@huggingface/transformers")
		this.extractor = (await (pipeline as Function)(
			"feature-extraction",
			"Xenova/all-MiniLM-L6-v2",
			{ dtype: "fp32" },
		)) as FeatureExtractionPipeline

		// onnxruntime-node's C++ Environment has a static destructor that crashes
		// (mutex lock failed) when process.exit() triggers C exit(). Workaround:
		// close stderr fd before calling exit to suppress the ugly crash message.
		// Exit code will be 134 (SIGABRT) instead of 0 — this is a known
		// onnxruntime-node issue (https://github.com/microsoft/onnxruntime/issues).
		if (!Embedder.exitOverrideInstalled) {
			Embedder.exitOverrideInstalled = true
			const fs = await import("fs")
			const originalExit = process.exit
			process.exit = ((code?: number) => {
				try {
					fs.closeSync(2) // close stderr fd to suppress C++ crash message
				} catch {}
				originalExit.call(process, code ?? 0)
			}) as never
		}
	}

	async embed(text: string): Promise<number[]> {
		if (!this.extractor) await this.initialize()
		const output = await this.extractor!(text, { pooling: "mean", normalize: true })
		return Array.from(output.data as Float32Array)
	}

	async dispose(): Promise<void> {
		if (this.extractor) {
			await (this.extractor as any).dispose()
			this.extractor = null
		}
	}
}

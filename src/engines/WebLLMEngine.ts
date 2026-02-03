import * as webllm from "@mlc-ai/web-llm";
import type { IEngine, Token, EngineMetrics, ModelManifest } from "../core/types";

export interface LoadProgress {
  stage: "downloading" | "loading" | "ready";
  progress: number; // 0-100
  text: string;
}

/**
 * WebLLMEngine wraps the MLC WebLLM library to provide inference
 * through our standard IEngine interface.
 *
 * WebLLM handles:
 * - Model downloading with caching
 * - Tokenization
 * - The full transformer forward pass on WebGPU
 * - KV cache management
 */
export class WebLLMEngine implements IEngine {
  readonly name = "webllm";

  private engine: webllm.MLCEngineInterface | null = null;
  private modelId: string;
  private onProgress?: (progress: LoadProgress) => void;

  // Metrics tracking
  private _ttft = 0;
  private _tps = 0;
  private _tokenCount = 0;
  private _genStartTime = 0;

  constructor(modelId: string, onProgress?: (progress: LoadProgress) => void) {
    this.modelId = modelId;
    this.onProgress = onProgress;
  }

  async init(_device: GPUDevice, _manifest: ModelManifest): Promise<void> {
    // WebLLM manages its own GPU device, so we ignore the passed device.
    // This is a limitation — in a more integrated system we'd want to share the device.

    this.onProgress?.({ stage: "downloading", progress: 0, text: "Starting..." });

    this.engine = await webllm.CreateMLCEngine(this.modelId, {
      initProgressCallback: (report) => {
        // report.progress is 0-1, report.text describes current action
        const progress = Math.round(report.progress * 100);
        const stage = report.progress < 1 ? "downloading" : "loading";
        this.onProgress?.({ stage, progress, text: report.text });
      },
    });

    this.onProgress?.({ stage: "ready", progress: 100, text: "Model loaded" });
  }

  async *generate(prompt: string, maxTokens: number): AsyncGenerator<Token> {
    if (!this.engine) {
      throw new Error("Engine not initialized");
    }

    this._tokenCount = 0;
    this._genStartTime = performance.now();
    this._ttft = 0;

    // Use streaming chat completion
    const stream = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      stream: true,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) {
        this._tokenCount++;

        if (this._tokenCount === 1) {
          this._ttft = performance.now() - this._genStartTime;
        }

        const elapsed = (performance.now() - this._genStartTime) / 1000;
        this._tps = elapsed > 0 ? this._tokenCount / elapsed : 0;

        yield {
          id: this._tokenCount,
          text: delta,
          timestampMs: performance.now() - this._genStartTime,
        };
      }
    }
  }

  getMetrics(): EngineMetrics {
    return {
      ttft: this._ttft,
      tps: this._tps,
      memoryUsed: 0, // WebLLM doesn't expose this directly
    };
  }

  dispose(): void {
    this.engine?.unload();
    this.engine = null;
  }

  /** Get list of available models from WebLLM */
  static getAvailableModels(): { id: string; name: string; size: string }[] {
    // These are popular small models that work well on M1
    return [
      { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", name: "SmolLM2 360M", size: "~250MB" },
      { id: "SmolLM2-1.7B-Instruct-q4f16_1-MLC", name: "SmolLM2 1.7B", size: "~1GB" },
      { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", name: "Llama 3.2 1B", size: "~700MB" },
      { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", name: "Llama 3.2 3B", size: "~1.8GB" },
      { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", name: "Qwen 2.5 1.5B", size: "~1GB" },
    ];
  }
}

import type { IEngine, Token, EngineMetrics, ModelManifest } from "../core/types";
import { MemoryOrchestrator } from "../core/MemoryOrchestrator";
import { ModelLoader } from "../core/ModelLoader";

/** Default memory budget: 4GB (conservative for 8GB machines, plenty for 16GB) */
const DEFAULT_BUDGET = 4 * 1024 * 1024 * 1024;

/**
 * BaseEngine provides shared boilerplate for all engine implementations.
 * Subclasses override the abstract methods to implement their specific strategy.
 */
export abstract class BaseEngine implements IEngine {
  abstract readonly name: string;

  protected device!: GPUDevice;
  protected memory!: MemoryOrchestrator;
  protected loader!: ModelLoader;
  protected manifest!: ModelManifest;

  private _ttft = 0;
  private _tps = 0;
  private _tokenCount = 0;
  private _genStartTime = 0;

  async init(device: GPUDevice, manifest: ModelManifest): Promise<void> {
    this.device = device;
    this.manifest = manifest;
    this.memory = new MemoryOrchestrator(device, DEFAULT_BUDGET);
    this.loader = new ModelLoader(this.memory);

    await this.onInit();
  }

  /** Subclass hook: load model weights, create pipelines, etc. */
  protected abstract onInit(): Promise<void>;

  async *generate(prompt: string, maxTokens: number): AsyncGenerator<Token> {
    this._tokenCount = 0;
    this._genStartTime = performance.now();
    this._ttft = 0;

    for await (const token of this.onGenerate(prompt, maxTokens)) {
      this._tokenCount++;
      if (this._tokenCount === 1) {
        this._ttft = performance.now() - this._genStartTime;
      }
      const elapsed = (performance.now() - this._genStartTime) / 1000;
      this._tps = elapsed > 0 ? this._tokenCount / elapsed : 0;
      yield token;
    }
  }

  /** Subclass hook: actual inference logic. Yield tokens as they're produced. */
  protected abstract onGenerate(prompt: string, maxTokens: number): AsyncGenerator<Token>;

  getMetrics(): EngineMetrics {
    return {
      ttft: this._ttft,
      tps: this._tps,
      memoryUsed: this.memory.usedBytes,
    };
  }

  dispose(): void {
    this.loader.abort();
    this.memory.dispose();
  }
}

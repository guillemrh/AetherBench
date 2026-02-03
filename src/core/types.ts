/** A single generated token */
export interface Token {
  id: number;
  text: string;
  /** Time since generation started, in ms */
  timestampMs: number;
}

/** Metrics collected during and after generation */
export interface EngineMetrics {
  /** Time to First Token in ms */
  ttft: number;
  /** Tokens per second (throughput) */
  tps: number;
  /** GPU memory currently used by this engine, in bytes */
  memoryUsed: number;
  /** Bytes saved vs loading the full model (relevant for SSP/NWC) */
  egressSaved?: number;
}

/** Describes a model's weight files and structure */
export interface ModelManifest {
  /** Human-readable model name */
  name: string;
  /** Total model size in bytes (all shards) */
  totalBytes: number;
  /** Individual weight file shards */
  shards: ShardInfo[];
  /** Number of transformer layers */
  numLayers: number;
  /** Hidden dimension size */
  hiddenSize: number;
  /** Vocabulary size */
  vocabSize: number;
}

export interface ShardInfo {
  /** URL to fetch this shard from */
  url: string;
  /** Byte offset within the full model */
  offset: number;
  /** Size of this shard in bytes */
  size: number;
  /** Which layer(s) this shard contains */
  layers: number[];
}

/** Standard interface every engine must implement */
export interface IEngine {
  readonly name: string;

  /** Initialize the engine with a GPU device and model info */
  init(device: GPUDevice, manifest: ModelManifest): Promise<void>;

  /**
   * Generate tokens from a prompt.
   * Yields tokens one at a time as they're produced.
   */
  generate(prompt: string, maxTokens: number): AsyncGenerator<Token>;

  /** Get current metrics snapshot */
  getMetrics(): EngineMetrics;

  /** Release all GPU resources */
  dispose(): void;
}

/** Identifies which engine implementation to use */
export type EngineType = "nwc" | "streaming" | "pruning" | "ssp";

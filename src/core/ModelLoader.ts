import type { ShardInfo } from "./types";
import type { MemoryOrchestrator, BufferHandle } from "./MemoryOrchestrator";

export interface DownloadProgress {
  shardIndex: number;
  totalShards: number;
  bytesLoaded: number;
  totalBytes: number;
}

/**
 * ModelLoader fetches model weight shards over HTTP and uploads them to the GPU
 * via the MemoryOrchestrator.
 *
 * Flow: HTTP fetch → ReadableStream → ArrayBuffer chunks → MemoryOrchestrator.upload → GPUBuffer
 */
export class ModelLoader {
  private memory: MemoryOrchestrator;
  private abortController: AbortController | null = null;

  constructor(memory: MemoryOrchestrator) {
    this.memory = memory;
  }

  /**
   * Download a single shard and upload it to a GPU buffer.
   * Streams the download so we can report progress.
   */
  async loadShard(
    shard: ShardInfo,
    onProgress?: (loaded: number, total: number) => void
  ): Promise<BufferHandle> {
    this.abortController = new AbortController();

    const response = await fetch(shard.url, {
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch shard: ${response.status} ${response.statusText}`);
    }

    const contentLength = Number(response.headers.get("content-length")) || shard.size;
    const reader = response.body?.getReader();

    if (!reader) {
      throw new Error("Response body is not readable");
    }

    // Read the stream into a single ArrayBuffer
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.(loaded, contentLength);
    }

    // Combine chunks into one ArrayBuffer
    const combined = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.byteLength;
    }

    // Upload to GPU
    const label = `shard_layers_${shard.layers.join("_")}`;
    return this.memory.allocateAndUpload(combined, label);
  }

  /**
   * Download all shards sequentially, reporting overall progress.
   * Returns handles in shard order.
   */
  async loadAll(
    shards: ShardInfo[],
    onProgress?: (progress: DownloadProgress) => void
  ): Promise<BufferHandle[]> {
    const handles: BufferHandle[] = [];
    const totalBytes = shards.reduce((sum, s) => sum + s.size, 0);
    let bytesLoaded = 0;

    for (let i = 0; i < shards.length; i++) {
      const handle = await this.loadShard(shards[i], (shardLoaded) => {
        onProgress?.({
          shardIndex: i,
          totalShards: shards.length,
          bytesLoaded: bytesLoaded + shardLoaded,
          totalBytes,
        });
      });
      bytesLoaded += shards[i].size;
      handles.push(handle);
    }

    return handles;
  }

  /** Cancel any in-progress downloads */
  abort(): void {
    this.abortController?.abort();
    this.abortController = null;
  }
}

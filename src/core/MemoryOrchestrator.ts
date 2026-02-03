/**
 * MemoryOrchestrator manages all GPU buffer allocations.
 *
 * Responsibilities:
 * - Allocate and track GPUBuffers
 * - Upload data from CPU (ArrayBuffer) to GPU (GPUBuffer) via single-copy staging
 * - Track total memory usage
 * - Free buffers and handle allocation failures gracefully
 */

export interface BufferHandle {
  id: string;
  buffer: GPUBuffer;
  size: number;
  label: string;
}

export class MemoryOrchestrator {
  private device: GPUDevice;
  private buffers: Map<string, BufferHandle> = new Map();
  private nextId = 0;

  /** Maximum bytes we'll allow ourselves to allocate. */
  private budgetBytes: number;

  constructor(device: GPUDevice, budgetBytes: number) {
    this.device = device;
    this.budgetBytes = budgetBytes;
  }

  /** How many bytes are currently allocated */
  get usedBytes(): number {
    let total = 0;
    for (const handle of this.buffers.values()) {
      total += handle.size;
    }
    return total;
  }

  get availableBytes(): number {
    return this.budgetBytes - this.usedBytes;
  }

  /**
   * Allocate a GPU buffer for storage (model weights, activations, etc).
   * Returns a handle, or throws if we'd exceed the memory budget.
   */
  allocate(
    size: number,
    label: string,
    usage?: GPUBufferUsageFlags
  ): BufferHandle {
    const resolvedUsage = usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    if (size > this.availableBytes) {
      throw new Error(
        `MemoryOrchestrator: cannot allocate ${formatBytes(size)} for "${label}" — ` +
        `only ${formatBytes(this.availableBytes)} of ${formatBytes(this.budgetBytes)} available`
      );
    }

    const buffer = this.device.createBuffer({ size, usage: resolvedUsage, label });
    const id = `buf_${this.nextId++}`;
    const handle: BufferHandle = { id, buffer, size, label };
    this.buffers.set(id, handle);
    return handle;
  }

  /**
   * Upload CPU data (ArrayBuffer / TypedArray) into an existing GPU buffer.
   * This is the "single-copy" path: JS ArrayBuffer → device.queue.writeBuffer → GPUBuffer.
   */
  upload(handle: BufferHandle, data: ArrayBuffer | ArrayBufferView, offset = 0): void {
    if (data instanceof ArrayBuffer) {
      this.device.queue.writeBuffer(handle.buffer, offset, data);
    } else {
      this.device.queue.writeBuffer(handle.buffer, offset, data.buffer, data.byteOffset, data.byteLength);
    }
  }

  /**
   * Allocate a buffer AND upload data into it in one step.
   * Convenience method for the common "download weights → GPU" flow.
   */
  allocateAndUpload(
    data: ArrayBuffer | ArrayBufferView,
    label: string,
    usage?: GPUBufferUsageFlags
  ): BufferHandle {
    const byteLength = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
    const handle = this.allocate(byteLength, label, usage);
    this.upload(handle, data);
    return handle;
  }

  /** Free a specific buffer */
  free(handle: BufferHandle): void {
    handle.buffer.destroy();
    this.buffers.delete(handle.id);
  }

  /** Free all buffers */
  dispose(): void {
    for (const handle of this.buffers.values()) {
      handle.buffer.destroy();
    }
    this.buffers.clear();
  }

  /** Get a snapshot of all current allocations (for debugging / dashboard) */
  getSnapshot(): { id: string; label: string; size: number }[] {
    return Array.from(this.buffers.values()).map((h) => ({
      id: h.id,
      label: h.label,
      size: h.size,
    }));
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryOrchestrator } from "./MemoryOrchestrator";

/** Minimal mock of GPUBuffer */
function mockGPUBuffer(): GPUBuffer {
  return {
    destroy: vi.fn(),
    size: 0,
    usage: 0,
    mapState: "unmapped",
    label: "",
    getMappedRange: vi.fn(),
    mapAsync: vi.fn(),
    unmap: vi.fn(),
  } as unknown as GPUBuffer;
}

/** Minimal mock of GPUDevice */
function mockDevice(): GPUDevice {
  return {
    createBuffer: vi.fn(() => mockGPUBuffer()),
    queue: {
      writeBuffer: vi.fn(),
    },
  } as unknown as GPUDevice;
}

describe("MemoryOrchestrator", () => {
  let device: GPUDevice;
  let mem: MemoryOrchestrator;
  const BUDGET = 1024 * 1024; // 1MB

  beforeEach(() => {
    device = mockDevice();
    mem = new MemoryOrchestrator(device, BUDGET);
  });

  it("starts with zero used bytes", () => {
    expect(mem.usedBytes).toBe(0);
    expect(mem.availableBytes).toBe(BUDGET);
  });

  it("tracks allocation size", () => {
    mem.allocate(256, "test-buf", 0);
    expect(mem.usedBytes).toBe(256);
    expect(mem.availableBytes).toBe(BUDGET - 256);
  });

  it("tracks multiple allocations", () => {
    mem.allocate(256, "buf-a", 0);
    mem.allocate(512, "buf-b", 0);
    expect(mem.usedBytes).toBe(768);
  });

  it("throws when allocation exceeds budget", () => {
    expect(() => mem.allocate(BUDGET + 1, "too-big", 0)).toThrow(/cannot allocate/);
  });

  it("throws when cumulative allocations exceed budget", () => {
    mem.allocate(BUDGET - 100, "first", 0);
    expect(() => mem.allocate(200, "second", 0)).toThrow(/cannot allocate/);
  });

  it("frees buffer and reclaims space", () => {
    const handle = mem.allocate(512, "temp", 0);
    expect(mem.usedBytes).toBe(512);
    mem.free(handle);
    expect(mem.usedBytes).toBe(0);
    expect(handle.buffer.destroy).toHaveBeenCalled();
  });

  it("dispose frees all buffers", () => {
    const h1 = mem.allocate(256, "a", 0);
    const h2 = mem.allocate(256, "b", 0);
    mem.dispose();
    expect(mem.usedBytes).toBe(0);
    expect(h1.buffer.destroy).toHaveBeenCalled();
    expect(h2.buffer.destroy).toHaveBeenCalled();
  });

  it("upload calls device.queue.writeBuffer", () => {
    const handle = mem.allocate(16, "data", 0);
    const data = new Float32Array([1, 2, 3, 4]);
    mem.upload(handle, data);
    expect(device.queue.writeBuffer).toHaveBeenCalled();
  });

  it("allocateAndUpload creates buffer and uploads in one step", () => {
    const data = new Uint8Array([10, 20, 30]);
    const handle = mem.allocateAndUpload(data, "combo", 0);
    expect(mem.usedBytes).toBe(3);
    expect(device.queue.writeBuffer).toHaveBeenCalled();
    expect(handle.label).toBe("combo");
  });

  it("getSnapshot returns current allocations", () => {
    mem.allocate(100, "alpha", 0);
    mem.allocate(200, "beta", 0);
    const snap = mem.getSnapshot();
    expect(snap).toHaveLength(2);
    expect(snap.map((s) => s.label)).toEqual(["alpha", "beta"]);
  });
});

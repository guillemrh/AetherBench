import { initWebGPU } from "../core/webgpu-init";
import { runMatmul } from "../core/matmul";
import { MemoryOrchestrator } from "../core/MemoryOrchestrator";
import type { WorkerRequest, WorkerResponse, MemoryTestResult } from "./messages";

let device: GPUDevice | null = null;

function respond(msg: WorkerResponse) {
  self.postMessage(msg);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const req = e.data;

  switch (req.type) {
    case "init": {
      try {
        const result = await initWebGPU();
        device = result.device;
        respond({ type: "init-ok", info: result.info });
      } catch (err) {
        respond({
          type: "init-error",
          error: err instanceof Error ? err.message : String(err),
        });
      }
      break;
    }

    case "matmul": {
      if (!device) {
        respond({ type: "matmul-error", error: "GPU not initialized" });
        return;
      }
      try {
        const { gpuTimeMs, sample } = await runMatmul(device, req.size);
        respond({
          type: "matmul-ok",
          result: { size: req.size, gpuTimeMs, sample },
        });
      } catch (err) {
        respond({
          type: "matmul-error",
          error: err instanceof Error ? err.message : String(err),
        });
      }
      break;
    }

    case "test-memory": {
      if (!device) {
        respond({ type: "test-memory-error", error: "GPU not initialized" });
        return;
      }
      try {
        const result = await runMemoryTest(device);
        respond({ type: "test-memory-ok", result });
      } catch (err) {
        respond({
          type: "test-memory-error",
          error: err instanceof Error ? err.message : String(err),
        });
      }
      break;
    }
  }
};

/**
 * Integration test: allocate a buffer, upload known data, read it back, verify it matches.
 * This tests the full CPU → GPU → CPU round-trip on real hardware.
 */
async function runMemoryTest(device: GPUDevice): Promise<MemoryTestResult> {
  const mem = new MemoryOrchestrator(device, 64 * 1024 * 1024); // 64MB budget

  // 1. Allocate a buffer with COPY_SRC so we can read it back
  const testData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
  const byteSize = testData.byteLength; // 32 bytes
  const handle = mem.allocate(
    byteSize,
    "test-roundtrip",
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  );
  const allocated = true;

  // 2. Upload data
  mem.upload(handle, testData);
  const uploaded = true;
  const usedBytes = mem.usedBytes;

  // 3. Read back: copy GPU buffer → readback buffer → map → compare
  const readbackBuffer = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(handle.buffer, 0, readbackBuffer, 0, byteSize);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();
  readbackBuffer.destroy();

  const readback = true;

  // 4. Verify data matches
  let dataMatches = true;
  for (let i = 0; i < testData.length; i++) {
    if (testData[i] !== result[i]) {
      dataMatches = false;
      break;
    }
  }

  // 5. Free and verify
  mem.free(handle);
  const freedOk = mem.usedBytes === 0;

  return { allocated, uploaded, readback, dataMatches, usedBytes, freedOk };
}

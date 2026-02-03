import shaderSource from "../shaders/matmul.wgsl?raw";

const TILE_SIZE = 16;

/**
 * Run a GPU matrix multiplication: C = A × B
 * where A is [size × size] and B is [size × size].
 * Returns the result matrix and GPU execution time.
 */
export async function runMatmul(
  device: GPUDevice,
  size: number
): Promise<{ gpuTimeMs: number; sample: number[] }> {
  const M = size;
  const N = size;
  const K = size;
  const floatCount = size * size;
  const byteSize = floatCount * 4;

  // Create input matrices with known values for verification.
  // A[i][j] = (i + j) % 7, B[i][j] = (i * j + 1) % 5
  const aData = new Float32Array(floatCount);
  const bData = new Float32Array(floatCount);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      aData[i * size + j] = (i + j) % 7;
      bData[i * size + j] = ((i * j + 1) % 5);
    }
  }

  // Create GPU buffers
  const bufferA = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferB = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bufferC = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uniformBuffer = device.createBuffer({
    size: 16, // vec4<u32>
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Upload data
  device.queue.writeBuffer(bufferA, 0, aData);
  device.queue.writeBuffer(bufferB, 0, bData);
  device.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([M, N, K, 0])
  );

  // Create pipeline
  const shaderModule = device.createShaderModule({ code: shaderSource });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferC } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  // Dispatch
  const workgroupsX = Math.ceil(N / TILE_SIZE);
  const workgroupsY = Math.ceil(M / TILE_SIZE);

  const t0 = performance.now();

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);
  pass.end();

  // Copy result to readback buffer
  encoder.copyBufferToBuffer(bufferC, 0, readbackBuffer, 0, byteSize);
  device.queue.submit([encoder.finish()]);

  // Wait for GPU to finish and read back
  await device.queue.onSubmittedWorkDone();
  const gpuTimeMs = performance.now() - t0;

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultData = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();

  // Sample first 4 values for verification
  const sample = Array.from(resultData.slice(0, 4));

  // Cleanup
  bufferA.destroy();
  bufferB.destroy();
  bufferC.destroy();
  uniformBuffer.destroy();
  readbackBuffer.destroy();

  return { gpuTimeMs, sample };
}

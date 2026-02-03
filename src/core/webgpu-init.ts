/**
 * Initialize WebGPU adapter and device.
 * Must be called from a context with GPU access (main thread or worker).
 */
export interface GPUInfo {
  vendor: string;
  architecture: string;
  maxBufferSize: number;
  maxComputeWorkgroupSizeX: number;
}

export async function initWebGPU(): Promise<{
  device: GPUDevice;
  info: GPUInfo;
}> {
  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is not supported in this browser. Use Chrome 113+ or Edge 113+."
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new Error(
      "No WebGPU adapter found. Your GPU may not be supported."
    );
  }

  const adapterInfo = adapter.info;
  const limits = adapter.limits;

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    },
  });

  device.lost.then((info) => {
    console.error("WebGPU device lost:", info.message);
  });

  return {
    device,
    info: {
      vendor: adapterInfo.vendor,
      architecture: adapterInfo.architecture,
      maxBufferSize: Number(limits.maxBufferSize),
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    },
  };
}

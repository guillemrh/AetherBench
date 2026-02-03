/** Messages sent from main thread → GPU worker */
export type WorkerRequest =
  | { type: "init" }
  | { type: "matmul"; size: number }
  | { type: "test-memory" };

/** Messages sent from GPU worker → main thread */
export type WorkerResponse =
  | { type: "init-ok"; info: GPUInfoMsg }
  | { type: "init-error"; error: string }
  | { type: "matmul-ok"; result: MatmulResult }
  | { type: "matmul-error"; error: string }
  | { type: "test-memory-ok"; result: MemoryTestResult }
  | { type: "test-memory-error"; error: string };

export interface GPUInfoMsg {
  vendor: string;
  architecture: string;
  maxBufferSize: number;
  maxComputeWorkgroupSizeX: number;
}

export interface MatmulResult {
  size: number;
  gpuTimeMs: number;
  /** First few values from the result matrix for verification */
  sample: number[];
}

export interface MemoryTestResult {
  allocated: boolean;
  uploaded: boolean;
  readback: boolean;
  dataMatches: boolean;
  usedBytes: number;
  freedOk: boolean;
}

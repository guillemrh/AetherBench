import { useState, useEffect, useRef, useCallback } from "react";
import type { WorkerRequest, WorkerResponse, GPUInfoMsg, MatmulResult, MemoryTestResult } from "../worker/messages";
import InferencePanel from "./InferencePanel";

type Status = "idle" | "initializing" | "ready" | "running" | "error";

export default function App() {
  const workerRef = useRef<Worker | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [gpuInfo, setGpuInfo] = useState<GPUInfoMsg | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [matmulResult, setMatmulResult] = useState<MatmulResult | null>(null);
  const [matrixSize, setMatrixSize] = useState(1024);
  const [memTest, setMemTest] = useState<MemoryTestResult | null>(null);

  const send = useCallback((msg: WorkerRequest) => {
    workerRef.current?.postMessage(msg);
  }, []);

  useEffect(() => {
    const worker = new Worker(
      new URL("../worker/gpu-worker.ts", import.meta.url),
      { type: "module" }
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      switch (msg.type) {
        case "init-ok":
          setGpuInfo(msg.info);
          setStatus("ready");
          break;
        case "init-error":
          setError(msg.error);
          setStatus("error");
          break;
        case "matmul-ok":
          setMatmulResult(msg.result);
          setStatus("ready");
          break;
        case "matmul-error":
          setError(msg.error);
          setStatus("error");
          break;
        case "test-memory-ok":
          setMemTest(msg.result);
          setStatus("ready");
          break;
        case "test-memory-error":
          setError(msg.error);
          setStatus("error");
          break;
      }
    };

    setStatus("initializing");
    worker.postMessage({ type: "init" } satisfies WorkerRequest);

    return () => worker.terminate();
  }, []);

  const runBenchmark = () => {
    setStatus("running");
    setError(null);
    setMatmulResult(null);
    send({ type: "matmul", size: matrixSize });
  };

  return (
    <div style={{ fontFamily: "monospace", padding: 32, maxWidth: 720, background: "#0a0a0f", color: "#e0e0e0", minHeight: "100vh" }}>
      <h1 style={{ color: "#f0f0f0" }}>AetherBench</h1>
      <p style={{ color: "#666" }}>WebGPU Benchmarking Suite — Phase 3</p>

      <section style={{ marginTop: 24 }}>
        <h2>GPU Status</h2>
        {status === "initializing" && <p>Initializing WebGPU...</p>}
        {status === "error" && (
          <p style={{ color: "#e55" }}>Error: {error}</p>
        )}
        {gpuInfo && (
          <table style={{ borderCollapse: "collapse" }}>
            <tbody>
              {Object.entries(gpuInfo).map(([k, v]) => (
                <tr key={k}>
                  <td style={{ padding: "4px 16px 4px 0", color: "#888" }}>{k}</td>
                  <td>{typeof v === "number" ? v.toLocaleString() : v || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {status !== "initializing" && status !== "error" && (
        <section style={{ marginTop: 24 }}>
          <h2>Matrix Multiply</h2>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <label>
              Size:&nbsp;
              <select
                value={matrixSize}
                onChange={(e) => setMatrixSize(Number(e.target.value))}
                disabled={status === "running"}
                style={{ background: "#1a1a2e", color: "#e0e0e0", border: "1px solid #333", padding: "4px 8px", borderRadius: 4 }}
              >
                {[256, 512, 1024, 2048, 4096].map((s) => (
                  <option key={s} value={s}>
                    {s}x{s}
                  </option>
                ))}
              </select>
            </label>
            <button
              onClick={runBenchmark}
              disabled={status === "running"}
              style={{ background: "#2a2a4a", color: "#e0e0e0", border: "1px solid #444", padding: "4px 16px", borderRadius: 4, cursor: status === "running" ? "wait" : "pointer" }}
            >
              {status === "running" ? "Running..." : "Run"}
            </button>
          </div>

          {matmulResult && (
            <div style={{ marginTop: 16 }}>
              <p>
                <strong>{matmulResult.size}x{matmulResult.size}</strong> FP32 matmul
                in <strong>{matmulResult.gpuTimeMs.toFixed(2)} ms</strong>
              </p>
              <p style={{ color: "#888" }}>
                GFLOPS: {(
                  (2 * matmulResult.size ** 3) /
                  (matmulResult.gpuTimeMs / 1000) /
                  1e9
                ).toFixed(1)}
              </p>
              <p style={{ color: "#888", fontSize: 12 }}>
                Sample C[0..3]: [{matmulResult.sample.map((v) => v.toFixed(1)).join(", ")}]
              </p>
            </div>
          )}
        </section>
      )}

      {status !== "initializing" && status !== "error" && (
        <section style={{ marginTop: 24 }}>
          <h2>Memory Orchestrator Test</h2>
          <button
            onClick={() => { setStatus("running"); setMemTest(null); send({ type: "test-memory" }); }}
            disabled={status === "running"}
            style={{ background: "#2a2a4a", color: "#e0e0e0", border: "1px solid #444", padding: "4px 16px", borderRadius: 4, cursor: status === "running" ? "wait" : "pointer" }}
          >
            {status === "running" ? "Testing..." : "Run Memory Test"}
          </button>
          {memTest && (
            <table style={{ marginTop: 12, borderCollapse: "collapse" }}>
              <tbody>
                {Object.entries(memTest).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ padding: "4px 16px 4px 0", color: "#888" }}>{k}</td>
                    <td style={{ color: v === true ? "#5e5" : v === false ? "#e55" : "#e0e0e0" }}>
                      {typeof v === "boolean" ? (v ? "PASS" : "FAIL") : typeof v === "number" ? `${v} bytes` : String(v)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}

      {status !== "initializing" && status !== "error" && <InferencePanel />}
    </div>
  );
}

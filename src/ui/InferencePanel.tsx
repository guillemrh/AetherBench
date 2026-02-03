import { useState, useRef } from "react";
import { WebLLMEngine, type LoadProgress } from "../engines/WebLLMEngine";
import type { EngineMetrics } from "../core/types";

const MODELS = WebLLMEngine.getAvailableModels();

const buttonStyle = {
  background: "#2a2a4a",
  color: "#e0e0e0",
  border: "1px solid #444",
  padding: "4px 16px",
  borderRadius: 4,
  cursor: "pointer",
};

const disabledButtonStyle = {
  ...buttonStyle,
  cursor: "wait",
  opacity: 0.6,
};

export default function InferencePanel() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [loadTimeMs, setLoadTimeMs] = useState<number | null>(null);
  const [prompt, setPrompt] = useState("Explain what WebGPU is in 2 sentences.");
  const [output, setOutput] = useState("");
  const [metrics, setMetrics] = useState<EngineMetrics | null>(null);

  const engineRef = useRef<WebLLMEngine | null>(null);

  const loadModel = async () => {
    // Dispose previous engine if any
    engineRef.current?.dispose();
    setLoadTimeMs(null);

    const engine = new WebLLMEngine(selectedModel, setLoadProgress);
    engineRef.current = engine;

    const loadStart = performance.now();
    try {
      // @ts-expect-error - we're not using manifest for WebLLM
      await engine.init(null, null);
      setLoadTimeMs(performance.now() - loadStart);
      setIsLoaded(true);
    } catch (err) {
      console.error("Failed to load model:", err);
      setLoadProgress({
        stage: "downloading",
        progress: 0,
        text: `Error: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
  };

  const generate = async () => {
    if (!engineRef.current) return;

    setIsGenerating(true);
    setOutput("");
    setMetrics(null);

    try {
      let fullOutput = "";
      for await (const token of engineRef.current.generate(prompt, 256)) {
        fullOutput += token.text;
        setOutput(fullOutput);
        setMetrics(engineRef.current.getMetrics());
      }
    } catch (err) {
      console.error("Generation error:", err);
      setOutput(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsGenerating(false);
      setMetrics(engineRef.current?.getMetrics() ?? null);
    }
  };

  return (
    <section style={{ marginTop: 24 }}>
      <h2>Inference Engine</h2>

      {/* Model Selection */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        <label>
          Model:&nbsp;
          <select
            value={selectedModel}
            onChange={(e) => {
              setSelectedModel(e.target.value);
              setIsLoaded(false);
              setLoadProgress(null);
            }}
            disabled={isGenerating}
            style={{
              background: "#1a1a2e",
              color: "#e0e0e0",
              border: "1px solid #333",
              padding: "4px 8px",
              borderRadius: 4,
            }}
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.size})
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={loadModel}
          disabled={isGenerating || (loadProgress !== null && loadProgress.stage !== "ready")}
          style={
            isGenerating || (loadProgress !== null && loadProgress.stage !== "ready")
              ? disabledButtonStyle
              : buttonStyle
          }
        >
          {loadProgress?.stage === "downloading" || loadProgress?.stage === "loading"
            ? `Loading... ${loadProgress.progress}%`
            : isLoaded
            ? "Reload"
            : "Load Model"}
        </button>
      </div>

      {/* Load Time Result */}
      {loadTimeMs !== null && (() => {
        // Estimate: cached load is ~2-3 MB/s from IndexedDB, network is ~0.5 MB/s typical
        // If load was faster than 10s for the expected size, likely cached
        const model = MODELS.find(m => m.id === selectedModel);
        const sizeMatch = model?.size.match(/~?([\d.]+)\s*(MB|GB)/i);
        const sizeMB = sizeMatch
          ? parseFloat(sizeMatch[1]) * (sizeMatch[2].toUpperCase() === 'GB' ? 1024 : 1)
          : 500;
        const expectedNetworkSec = sizeMB / 5; // ~5 MB/s generous network estimate
        const isCached = loadTimeMs < expectedNetworkSec * 1000 * 0.5; // loaded in less than half the expected network time
        return (
          <p style={{ color: "#888", fontSize: 13, marginBottom: 12 }}>
            Model loaded in <strong style={{ color: "#e0e0e0" }}>{(loadTimeMs / 1000).toFixed(2)}s</strong>
            {isCached && <span style={{ color: "#5e5" }}> (cached)</span>}
          </p>
        );
      })()}

      {/* Loading Progress */}
      {loadProgress && loadProgress.stage !== "ready" && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              width: "100%",
              height: 8,
              background: "#1a1a2e",
              borderRadius: 4,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${loadProgress.progress}%`,
                height: "100%",
                background: "#4a9eff",
                transition: "width 0.3s",
              }}
            />
          </div>
          <p style={{ color: "#888", fontSize: 12, marginTop: 4 }}>{loadProgress.text}</p>
        </div>
      )}

      {/* Prompt Input */}
      {isLoaded && (
        <>
          <div style={{ marginBottom: 12 }}>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isGenerating}
              placeholder="Enter your prompt..."
              style={{
                width: "100%",
                minHeight: 80,
                background: "#1a1a2e",
                color: "#e0e0e0",
                border: "1px solid #333",
                borderRadius: 4,
                padding: 8,
                fontFamily: "monospace",
                resize: "vertical",
              }}
            />
          </div>
          <button
            onClick={generate}
            disabled={isGenerating || !prompt.trim()}
            style={isGenerating || !prompt.trim() ? disabledButtonStyle : buttonStyle}
          >
            {isGenerating ? "Generating..." : "Generate"}
          </button>
        </>
      )}

      {/* Output */}
      {output && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            background: "#1a1a2e",
            borderRadius: 4,
            whiteSpace: "pre-wrap",
            fontFamily: "monospace",
            fontSize: 14,
            lineHeight: 1.5,
          }}
        >
          {output}
          {isGenerating && <span style={{ opacity: 0.5 }}>▋</span>}
        </div>
      )}

      {/* Metrics */}
      {metrics && (
        <div style={{ marginTop: 12, color: "#888", fontSize: 13 }}>
          <span style={{ marginRight: 24 }}>
            TTFT: <strong style={{ color: "#e0e0e0" }}>{metrics.ttft.toFixed(0)} ms</strong>
          </span>
          <span>
            TPS: <strong style={{ color: "#e0e0e0" }}>{metrics.tps.toFixed(1)}</strong>
          </span>
        </div>
      )}
    </section>
  );
}

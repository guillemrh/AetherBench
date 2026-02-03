# AetherBench

WebGPU-powered benchmarking suite for comparing AI inference optimization strategies in the browser.

## What is this?

AetherBench runs Large Language Models (LLMs) directly in your browser using WebGPU, and measures performance across different optimization techniques:

- **Neural Weight Compression (NWC)** — Extreme quantization (Q4) with on-GPU dequantization
- **Weight Streaming** — Progressive layer-by-layer loading to minimize time-to-first-token
- **Pruning (DarwinLM)** — Sparse model inference with zero-weight skipping
- **Semantic Speculative Prefetching (SSP)** — Intent-aware LoRA adapter loading

## Key Metrics

| Metric | What it measures |
|--------|------------------|
| **TTFT** | Time to First Token — how quickly the model starts responding |
| **TPS** | Tokens Per Second — generation throughput |
| **Load Time** | Model download + GPU initialization time |
| **Memory** | GPU buffer utilization |

## Requirements

- **Browser:** Chrome 113+ or Edge 113+ (WebGPU required)
- **Hardware:** Apple Silicon (M1/M2/M3) recommended, any WebGPU-capable GPU works
- **Memory:** 8GB+ RAM (16GB recommended for larger models)

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Open http://localhost:5173 in Chrome
```

## Available Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| SmolLM2 360M | ~250MB | Basic | Fast |
| SmolLM2 1.7B | ~1GB | Good | Medium |
| Llama 3.2 1B | ~700MB | Good | Medium |
| Llama 3.2 3B | ~1.8GB | Best | Slower |
| Qwen 2.5 1.5B | ~1GB | Good | Medium |

Models are cached in your browser after first download.

## Architecture

```
┌─────────────────────┐       postMessage        ┌─────────────────────┐
│    Main Thread      │ ◄──────────────────────► │    GPU Worker       │
│                     │                           │                     │
│  React Dashboard    │   commands ──────────►   │  WebGPU Device      │
│  - Metrics display  │                           │  - Model weights    │
│  - Model selector   │   ◄────────── tokens,    │  - Inference engine │
│  - Prompt input     │              metrics      │  - WGSL shaders     │
└─────────────────────┘                           └─────────────────────┘
```

## Project Structure

```
src/
├── core/
│   ├── MemoryOrchestrator.ts  # GPU buffer management
│   ├── ModelLoader.ts         # Streaming model downloads
│   ├── matmul.ts              # Matrix multiply benchmark
│   └── types.ts               # Shared interfaces
├── engines/
│   ├── BaseEngine.ts          # Abstract engine class
│   └── WebLLMEngine.ts        # WebLLM integration
├── shaders/
│   └── matmul.wgsl            # Compute shader
├── ui/
│   ├── App.tsx                # Main dashboard
│   └── InferencePanel.tsx     # LLM interface
└── worker/
    ├── gpu-worker.ts          # WebGPU worker
    └── messages.ts            # Worker protocol
```

## Development

```bash
# Type check
npm run typecheck

# Run tests
npm run test

# Production build
npm run build
```

## References

- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
- [WebLLM](https://github.com/mlc-ai/web-llm)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B)

## License

MIT

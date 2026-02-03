# Project AetherBench: WebGPU Next-Gen AI Benchmarking Suite

---

## ⚠️ Instructions for Claude Code

**READ THIS FIRST — Critical Collaboration Guidelines**

This project involves complex architectural decisions with significant trade-offs. Claude, you are expected to:

1. **Question everything.** If a requirement seems underspecified, contradictory, or technically unsound — say so. Do not assume I've thought through all implications.

2. **Push back on bad ideas.** If I propose something that conflicts with WebGPU best practices, memory constraints, or performance goals — challenge me directly. I would rather have a 10-minute debate than a 10-hour refactor.

3. **Ask before implementing.** For any non-trivial decision (buffer allocation strategy, worker architecture, API design), ask clarifying questions first. Do not fill in gaps with assumptions.

4. **Flag unknowns explicitly.** If you're uncertain about M1-specific behavior, WebGPU edge cases, or WGSL limitations — tell me. We'll research together or I'll test locally.

5. **Disagree with me.** If I say "just do X" and you think Y is better — make the case for Y. I hired you to think, not to comply.

**Example interactions I want:**

> ❌ Bad: "Sure, I'll implement zero-copy transfers using mapped buffers."
>
> ✅ Good: "You mentioned zero-copy transfers, but WebGPU's `mapAsync` requires the buffer to be unmapped before GPU operations. Are you okay with a double-buffering approach instead, or do you want to explore `GPUExternalTexture` patterns? Also — what's your expected model file size? That affects whether we should chunk the transfers."

---

## 1. Vision

AetherBench is a production-grade testing environment designed to benchmark and compare state-of-the-art Local AI acceleration techniques on WebGPU. The core mission is to validate **Semantic Speculative Prefetching (SSP)** against established 2024-2025 SOTA methods (NWC, Weight Streaming, and DarwinLM).

### Open Questions (Claude should probe these)

- [ ] What constitutes a "fair" benchmark comparison between fundamentally different optimization strategies?
- [ ] How do we isolate SSP benefits from network variability?
- [ ] Is TTFT the right primary metric, or should we weight TPS differently for different use cases?

---

## 2. Target Hardware

| Spec | Value |
|------|-------|
| Primary | Apple Silicon (M1/M2/M3) Unified Memory Architecture |
| Environment | Chrome/Edge (WebGPU enabled), High-performance mode |
| Fallback | None specified — **Claude: should we have a WebGL2 fallback or hard-fail?** |

### Constraints Claude Should Verify

- M1 base has 8GB unified memory. M1 Air often thermal-throttles under sustained load.
- WebGPU buffer size limits vary by implementation — **what's the practical max on Chrome/M1?**
- "High-performance mode" — **is this a Chrome flag, macOS setting, or just aspirational?**

---

## 3. Technical Architecture (The 4 Engines)

### Engine A: Neural Weight Compression (NWC)

**Goal:** Minimize storage/bandwidth via extreme quantization.

**Proposed Implementation:**
- Custom WGSL dequantization kernel
- 2.5-bit weight unpacking during FMA loop
- Reference: Sub-3-bit neural codecs

**Questions Claude Must Ask Before Implementing:**

1. Do we have a specific 2.5-bit quantization scheme in mind, or should I research options (GPTQ, AWQ, QuIP#)?
2. "Directly into GPU registers" — WGSL doesn't expose registers. Do you mean threadgroup shared memory, or are you describing the conceptual goal?
3. What's the expected accuracy degradation we'll accept? This affects kernel design.
4. Do we have pre-quantized model weights, or is quantization in-scope for this project?

---

### Engine B: Weight Streaming (WS)

**Goal:** Minimize Time-to-First-Token (TTFT) via progressive loading.

**Proposed Implementation:**
- Layer-by-layer sequential loading
- Overlap fetch with compute (pipeline `Layer_N+1` while executing `Layer_N`)
- Fetch API streaming into GPU buffers

**Questions Claude Must Ask Before Implementing:**

1. What's the layer granularity? Per-transformer-block, or finer (per-attention-head)?
2. How do we handle layer dependencies? Later layers need outputs from earlier layers — are we pre-allocating activation buffers?
3. "Fetch API" — are we assuming HTTP/2 multiplexing, or sequential range requests? Server support matters here.
4. What happens on network interruption mid-stream? Retry strategy?

---

### Engine C: DarwinLM (Pruning)

**Goal:** Maximize Tokens/Sec (TPS) via structural sparsity.

**Proposed Implementation:**
- Load pre-pruned sparse model
- Optimized SpMV kernels in WGSL
- Skip zero-weight computation

**Questions Claude Must Ask Before Implementing:**

1. What sparsity pattern? Unstructured, 2:4 structured, block-sparse? This fundamentally changes the kernel design.
2. Do we have a DarwinLM-pruned model, or are we creating one? If creating — what's the pruning budget?
3. "Avoid computing zero-value weights" — for unstructured sparsity, the indexing overhead often exceeds the compute savings. Have we validated this is actually faster for our target sparsity level?
4. How does this interact with NWC? Are we running pruned + quantized simultaneously?

---

### Engine D: Semantic Speculative Prefetching (SSP) — The Innovation

**Goal:** Context-aware predictive weight loading based on user intent.

**Proposed Pipeline:**

```
User Input → Sentry Model (SmolLM-135M) → Intent Classification
                                              ↓
                                    Semantic Byte-Range Mapping
                                              ↓
                                    HTTP Range Requests (Expert Blocks)
                                              ↓
                                    JIT GPU Memory Stitching
```

**This is the most speculative component. Claude MUST challenge these assumptions:**

1. **Sentry model overhead:** SmolLM-135M inference takes ~50-200ms. If TTFT goal is <500ms, we've burned 10-40% of budget on classification. Is this acceptable? Would a simpler heuristic (keyword matching, embedding similarity) suffice?

2. **Expert Block hypothesis:** This assumes model weights have semantic locality ("coding weights" cluster together). Is this validated for Llama-3.2's architecture? Transformer layers are generally not semantically organized.

3. **"Just-In-Time Stitching":** What does this mean concretely? WebGPU doesn't support sparse buffer binding or virtual memory remapping. Are we copying shards into a pre-allocated contiguous buffer? That's not "stitching" — it's copying.

4. **Range request latency:** Multiple small range requests often have worse total latency than one large request due to RTT overhead. Have we modeled this?

5. **Cache coherency:** If we prefetch "coding blocks" but the user pivots to "creative writing" — do we evict and re-fetch? What's the eviction policy?

**Alternative approaches Claude should propose if this seems unworkable:**

- Tiered model loading (load base capability fast, enhance on-demand)
- Speculative full-model loading with priority queuing
- LoRA-style adapter loading based on intent (smaller delta to fetch)

---

## 4. System Components

### `src/core/MemoryOrchestrator.ts`

**Responsibilities:**
- Manage WebGPU `GPUBuffer` lifecycle
- Implement streaming transfers from `ReadableStream` → `GPUBuffer`
- Respect M1 memory pressure limits
- Handle allocation failures gracefully

**Implementation Questions:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Before writing any code, Claude must answer:                    │
├─────────────────────────────────────────────────────────────────┤
│ 1. What's the memory budget? 2GB? 4GB? Dynamic based on device? │
│ 2. "Zero-copy" from Fetch → GPU is impossible in WebGPU today.  │
│    What's the acceptable copy strategy? (ArrayBuffer staging?)  │
│ 3. How do we detect memory pressure before OOM? Is there an API?│
│ 4. Should this be a singleton, or per-engine instances?         │
│ 5. Error recovery: retry, degrade, or crash?                    │
└─────────────────────────────────────────────────────────────────┘
```

---

### `src/core/SSPManager.ts`

**Responsibilities:**
- Analyze user intent (Sentry model or heuristic)
- Map intent → byte ranges via manifest
- Coordinate HTTP range requests
- Track prefetch accuracy metrics

**Implementation Questions:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Before writing any code, Claude must answer:                    │
├─────────────────────────────────────────────────────────────────┤
│ 1. What's the manifest format? JSON? Binary? Where is it hosted?│
│ 2. How many "semantic categories" are we supporting? 3? 10? 50? │
│ 3. What if classification confidence is low? Fetch everything?  │
│ 4. How do we measure "prefetch hit rate" for benchmarking?      │
│ 5. Is the Sentry model loaded persistently, or on-demand?       │
└─────────────────────────────────────────────────────────────────┘
```

---

### `src/ui/Dashboard.tsx`

**Telemetry Display:**

| Metric | Description | Update Frequency |
|--------|-------------|------------------|
| TTFT | Time to First Token (ms) | Per-generation |
| TPS | Tokens Per Second (throughput) | Real-time (100ms) |
| Egress Savings | % data avoided via SSP/NWC | Per-session |
| Thermal State | Estimated M1 thermal headroom | **Unclear — how?** |
| Memory Pressure | Current GPU buffer utilization | Real-time |

**Implementation Questions:**

1. "Thermal State" — there's no web API for this. Are we estimating from performance degradation, or is this aspirational?
2. What charting library? Recharts, D3, custom Canvas?
3. Real-time updates at 100ms — is this WebSocket, polling, or `postMessage` from worker?
4. Do we need historical data persistence, or session-only?

---

## 5. Development Milestones

| Phase | Description | Acceptance Criteria | Claude's Pre-Work |
|-------|-------------|---------------------|-------------------|
| 1 | WebGPU Foundation | Matrix multiply runs on M1 GPU | **Ask:** What matrix sizes? FP32 or FP16? |
| 2 | Multi-Engine Wrapper | Standardized `IEngine` interface | **Ask:** What methods? How do we handle engine-specific config? |
| 3 | SSP Implementation | Semantic router + range requests work | **Ask:** Need manifest format spec first |
| 4 | Visualization | Dashboard shows all metrics live | **Ask:** Design mockup? |

---

## 6. Implementation Notes

### Memory

```typescript
// Proposed pattern — Claude should critique this
const buffer = device.createBuffer({
  size: modelWeights.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: false, // Why not true for initial upload?
});
```

**Questions for Claude:**
- `COPY_DST` allows `writeBuffer`, but should we use `mappedAtCreation` + `getMappedRange` instead for large uploads?
- Should `STORAGE` be combined with `COPY_SRC` for debugging/readback?

### Parallelism

- Use `Web Workers` to keep UI at 60fps
- WebGPU device is not transferable — **how do we handle this?** (OffscreenCanvas? MessagePort?)

### Models

- Default: Llama-3.2-3B (Quantized)
- **What quantization?** GGUF Q4_K_M? ONNX INT8? SafeTensors with external scales?
- **Where hosted?** HuggingFace? Self-hosted? CDN with range request support?

---

## 7. Getting Started with Claude Code

### Initial Prompt Template

```
Claude, read this README carefully — especially the "Instructions for Claude Code" section.

Before we write any code, I need you to:
1. List the top 3 technical risks you see in this architecture
2. Identify any contradictions or underspecified requirements
3. Ask me at least 5 clarifying questions

Then, and only then, let's discuss which component to build first.
```

### Expected Claude Behavior

**When I say:** "Let's build the MemoryOrchestrator"

**Claude should respond with:**
> Before I start, I have some questions:
>
> 1. You mentioned "zero-copy transfers" but WebGPU requires explicit copies from JS ArrayBuffer to GPUBuffer. Did you mean we should minimize copies (single staging buffer) or is there a specific technique you have in mind?
>
> 2. What's the memory ceiling we're targeting? M1 Air has 8GB shared between CPU/GPU — should I assume 4GB max for model weights?
>
> 3. For error handling — if we hit memory pressure, should the orchestrator: (a) throw and let the caller handle it, (b) automatically evict least-recently-used buffers, or (c) queue the allocation and retry?
>
> 4. I notice the SSPManager will also need buffer access. Should MemoryOrchestrator be a singleton that SSPManager consumes, or should each have independent pools?

**Do NOT just start writing code.** This project is too complex for implementation without clarification.

---

## 8. Technical Debt & Known Unknowns

| Item | Status | Owner |
|------|--------|-------|
| True "zero-copy" is impossible in WebGPU | Needs alternative design | Claude to propose |
| Thermal monitoring has no web API | Needs feasibility check | TBD |
| SSP "Expert Blocks" hypothesis is unvalidated | Needs research | Claude to investigate |
| WGSL SpMV kernel performance is uncertain | Needs benchmarking | Phase 2 |
| Worker ↔ WebGPU device transfer strategy | Needs design | Phase 1 |

---

## 9. Reference Links

- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
- [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B)

---

## 10. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-XX-XX | Initial README created | Human |
| 2025-XX-XX | Enhanced for Claude Code collaboration | Claude |

---

**Remember: Claude's job is to make this project succeed, not to make me feel good about my ideas. Challenge, question, and improve.**
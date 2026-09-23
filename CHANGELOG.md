# Changelog

## v0.1.0-beta — 2026-09-24

First public release. A single-GPU LLM serving engine measured on two architectures
(Tesla T4 / sm_75, RTX 4060 / sm_89) with one model family (Qwen3-0.6B and 1.7B).
Beta because the evidence covers exactly that: two GPUs, one family, one node.

### Engine

- Continuous batching over a paged KV cache: per-layer page pools, per-request block
  tables, FCFS admission bounded by KV capacity, LIFO preemption with recompute
  accounting, cancellation from every request state.
- Chunked prefill scheduled against decode under a per-step token budget.
- **Fused step**: a step's decode tokens and prefill chunk tokens run as one packed
  forward, so the weights are read once for both. Measured −17% prefill-carrying step
  and −18% ITL p50 on the T4.
- CUDA graphs for every forward shape (decode buckets, chunked prefill, fused), captured
  during warmup; captures that happen while serving are counted and reported, because
  two tail-latency "regressions" turned out to be exactly that.
- Prefix cache (radix + exact entries, copy-on-write tails), off by default: unresolved
  on the measured workloads.

### Kernels

- Triton: paged decode attention (per-head; GQA-shared and split-K variants registered),
  paged K/V writes, fused RMSNorm, fused Q/K RoPE, fused SwiGLU with optional gate/up
  projection fusion, INT8 paged KV.
- Chunked-prefill attention: SDPA over gathered pages (default on both measured GPUs at
  16-token pages), a per-token Triton kernel (INT8-capable), and a tiled `tl.dot` kernel
  for sm_80+.
- FlashAttention-2 over the engine's own pages for both phases, where a wheel and a
  compatible page size exist.

### Portability hooks

- **Backends**: attention implementations are registered, not hard-coded. Each declares
  why it cannot run on a given device and geometry; `"auto"` selects by priority, and a
  *named* backend that cannot run raises with the reason instead of falling back.
- **Device policy**: per-architecture measured defaults with the evidence that chose
  them (sm_75 and sm_89 entries); anything else gets capability-led defaults explicitly
  labelled unmeasured.
- **Model families**: fused kernels find their modules structurally and RoPE is patched
  in the model's own modeling module, so Llama-style checkpoints need no new code.
  Geometry the paged kernels cannot serve (head_dim > 128, non-divisible GQA,
  sliding-window attention, MoE, MLA) is refused at load with the reason.

### Serving

- OpenAI-compatible `/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming
  for both, `usage`, logprobs, stop strings, the model's chat template. Parameters the
  engine does not honour (`n > 1`, `echo`, `best_of`, `logit_bias`) are refused, not
  ignored.
- Per-request sampling: temperature, top-p, top-k, min-p, repetition/presence/frequency
  penalties, per-request seeds, stop token ids, `ignore_eos`. An all-greedy batch takes
  the same argmax path every benchmark was measured with.
- Prometheus `/metrics`: TTFT, inter-token, end-to-end and queue-time histograms, token
  counters, KV utilization, decode batch, in-service graph captures.
- Health, readiness, graceful drain, native `/generate` and SSE endpoints.

### Measured (see `docs/optimization-journal.md`, `docs/rtx4060-final-evaluation.md`)

Tesla T4, Qwen3-0.6B fp16, chat profile: prefill-carrying step 98 → 20.5 ms; ITL p50
~25 → 19.8 ms; ITL p99 295 → 33 ms; TTFT p50 2.8 s → ~0.4 s; decode-only step 10.2 ms
against a ~5 ms weight-read floor.

RTX 4060, fp16, 256-token pages, long profile: direct FlashAttention prefill −7.8%
prefill step and −22.6% ITL p99 (0.6B), −13.7% TTFT and −5.3% expected gap (1.7B).

### Recorded negative results

Tiled Triton prefill on sm_75 (`tl.dot` emits no `mma.sync`, 128 register spills);
GQA-shared decode reads (0.95–1.06x — L2 already serves the second read); FlashAttention
decode on sm_89 (+113.9% ITL against graphed Triton: its kvcache path is not
stream-capture safe on the tested build); dense-gather Flash prefill at 16-token pages
(+38.5% prefill step); INT8 KV; prefix caching on random-prompt workloads.

### Known limitations

- Performance is validated for Qwen3 only. Other Llama-style families load and run
  through the same structural hooks but have not been measured or token-gated.
- INT8 KV is disabled: a paged-KV recompute/preemption path drifts on Ada and is marked
  as an expected failure in the CUDA suite.
- No speculative decoding in the batched path, no quantized weights, no tensor or
  pipeline parallelism, no structured output.
- Qwen3-4B does not fit in fp16 on 8 GB with a useful KV pool; it needs the planned
  W4A16 weight path.
- CI runs the CPU suite only. GPU results are reproduced by hand and recorded in `docs/`
  and `results/`.

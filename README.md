# full-inference-engine

A single-GPU LLM serving engine built from the model down: continuous batching over a
paged KV cache, chunked prefill fused with decode into one forward per step, CUDA graphs
for every forward shape, custom Triton attention/KV kernels, an FCFS scheduler with
preemption, and an HTTP/SSE server. Hugging Face `transformers` supplies the model
definition and weights; everything that runs a request is here.

Every optimization in the repo was either measured to win on a Tesla T4 with interleaved
A/B runs and a stock-Transformers token-identity gate, or is recorded as a negative result
with the numbers. `docs/optimization-journal.md` has all of them.

## Status (2026-09-21, Tesla T4, Qwen3-0.6B fp16, chat workload ~656-token prompts)

| | first T4 measurement (Sep 20) | now |
|---|---|---|
| prefill-carrying step p50 | 98 ms | **20.5 ms** |
| decode-only step p50 (batch ~5) | 9.7 ms | 10.2 ms (weight-read floor ~5 ms + attention ~4 ms) |
| ITL p50 / p99 | ~25 / 295 ms | **19.8 / 33 ms** |
| TTFT p50 | 2.8 s | ~0.3-0.4 s |

Defaults set by measurement: SDPA-over-pages chunked prefill, graphs on decode and
prefill, fused step, warmup at startup, per-head paged decode kernel. Retired with
evidence: tiled Triton prefill on sm_75 (no tensor cores from `tl.dot`), GQA-shared
decode reads (L2 already dedups), prefix cache and INT8 KV on these workloads.

Not production-ready: greedy sampling only, no OpenAI-compatible API, no Prometheus
metrics, speculative decoding not integrated. See "Roadmap".

## Quick start

```bash
pip install -e ".[dev,server]"     # torch, transformers, triton must match your CUDA
python -m pytest -q                # CPU tests
python -m pytest -q -m cuda        # GPU correctness gates (downloads Qwen/Qwen3-0.6B)
python scripts/check_hooks.py      # warmup captures every graph; step phases report
uvicorn engine.server.api:create_app --factory --port 8000
curl -N localhost:8000/generate/stream -d '{"prompt":"Explain KV caching.","max_new_tokens":64}'
```

Engine in a script:

```python
from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.model import load_model
loaded = load_model("Qwen/Qwen3-0.6B")
engine = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, loaded.device,
                                  max_active=8, num_blocks=1024, cuda_graph_batch_sizes=(2, 4, 8))
engine.warmup()
print(engine.generate(["The capital of France is"], max_new_tokens=16))
```

## How it works

`docs/architecture.md` is the full walkthrough: libraries, every file, the request
lifecycle, one `step()` in detail, graph capture rules, and each Triton kernel explained.
The short version:

```
HTTP/SSE (FastAPI) → ContinuousBatchingService (one worker thread owns the engine)
  → FCFSScheduler: admit by KV capacity, plan prefill chunks under a token budget
  → ContinuousBatchingEngine.step():
        decode rows (1 token each) + prefill chunk rows (≤128 tokens each)
        → ONE packed forward, replayed from a CUDA graph keyed by shape
        → custom attention fn cuts the packed row: paged decode kernel / SDPA over gathered pages
        → argmax, one device→host copy, commit tokens, finish/preempt/admit
```

KV lives in per-layer pools `[num_blocks, 16, kv_heads, head_dim]`; each request owns a
block table. Triton kernels write K/V into pages and attend over them; RMSNorm, RoPE and
SwiGLU are fused Triton kernels installed onto the Qwen3 modules.

## Measuring

```bash
python -m benchmarks.reliability.ab --setting fused_step --prompt-profile chat --cuda-graphs --repeats 5 --duration 30
python -m benchmarks.kernels.paged_decode_regime_sweep --kernel both
python -m benchmarks.kernels.prefill_attention_ab --ptx-only      # does tl.dot reach the tensor cores on this GPU?
```

`benchmarks/reliability/ab.py` runs interleaved arms on warmed engines, checks greedy
tokens against stock Transformers first (tie-aware), and reports medians with run-to-run
spread; anything inside the spread is "unresolved". `scripts/t4_phase0_phase1.ipynb` is
the two-GPU Kaggle runner used for every recorded result.

## Repository

```
engine/      batching (the engine), scheduler, runtime (request state machine), cache (paging,
             prefix cache), graphs (capture), kernels (Triton), server (FastAPI), model (loader)
benchmarks/  reliability (soak, A/B, sweep), kernels, batching, server, quantization, understanding
tests/       CPU tests + `-m cuda` gates mirroring engine/
docs/        architecture.md · optimization-journal.md · checkpoint.md · t4-reevaluation-plan.md ·
             design-decisions.md · understanding-journal.md · rtx4060-plan.md
results/t4/  transcribed T4 measurements
scripts/     check_hooks, token_margins, Kaggle/Colab setup, the T4 notebook
```

## Roadmap

Tier 1 (serving): batched sampling (temperature/top-p/top-k, stop, logprobs),
OpenAI-compatible `/v1/chat/completions`, Prometheus metrics.
Tier 2 (performance, next GPU is an RTX 4060 - `docs/rtx4060-plan.md`): tiled prefill
behind the PTX gate, FlashAttention-2, split-K decode attention, weight-only INT8/INT4,
batched speculative decoding.
Tier 3: second model family, CPU/GPU step overlap, structured output.

## Correctness policy

Every engine path must reproduce stock Transformers' greedy tokens on the identity
prompts except at logit ties (top-2 margin < 0.02, one fp16 ulp); kernel swaps may
drift later than the first 8 tokens and are gated on the stock reference. Invariants
checked by the soak: no leaked KV pages, every request reaches a terminal state,
cancellation from every state. Negative results are recorded, never deleted.

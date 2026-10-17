# full-inference-engine

A single-GPU LLM serving engine built from the model down: continuous batching over a
paged KV cache, chunked prefill fused with decode into one forward per step, CUDA graphs
for eligible forward shapes, custom Triton attention/KV kernels, an FCFS scheduler with
preemption, per-request sampling, and an OpenAI-compatible server. Hugging Face
`transformers` supplies the model definition and weights; everything that runs a request
is here.

Attention kernels are **pluggable backends** chosen per GPU and per checkpoint, not
constants: the engine ships Triton paged decode (per-head, GQA-shared, split-K),
chunked-prefill attention (SDPA over gathered pages, per-token, tiled `tl.dot`), and
FlashAttention-2 over the same pages where a wheel exists. Fusions find their modules
structurally, so Llama-style checkpoints work without new code.

Every optimization here was either measured to win on a Tesla T4 or RTX 4060 with
interleaved A/B runs and a stock-Transformers token-identity gate, or is recorded as a
negative result with the numbers. The complete Ada investigation—including INT8 KV,
W8A16, 4B-AWQ, accepted/rejected kernels and raw-evidence links—is in
`docs/rtx4060-ada-sm89-complete-report.md`.

## RTX 4060 results (Ada sm_89, Qwen3)

The strongest complete serving comparison is Qwen3-1.7B at the same **5 GiB
weights-plus-KV budget**.
The measured stack uses W8A16 MLP and vocabulary projections, calibrated INT8 KV,
tiled INT8 prefill and packed fused rows. The FP16 arm uses the measured page-16 tiled
prefill path. These are medians of three interleaved end-to-end runs with Poisson arrivals
([chat](results/rtx4060/stacked_vs_fp16_chat_1.7b.json),
[long context](results/rtx4060/stacked_vs_fp16_long_c8_1.7b.json)). The stack is lossy
and does not include the later experimental W8A16 attention projections.

| 1.7B service metric | FP16 | measured stack | change |
|---|---:|---:|---:|
| chat throughput (32 requests, 4/s, up to 8 active) | 318 | 439 output tok/s | **+38.1%** |
| chat TTFT p50 | 1.890 s | 0.852 s | **-54.9%** |
| chat ITL p50 / p99 | 17.40 / 43.09 ms | 10.84 / 26.08 ms | **-37.7% / -39.5%** |
| long-context throughput (3–4k-token prompts, up to 8 active) | 83.7 | 128.5 output tok/s | **+53.6%** |
| long-context TTFT p50 | 6.235 s | 3.497 s | **-43.9%** |
| KV capacity | 16,800 | 45,872 tokens | **2.73×** |
| long-context preemptions | 9 | 0 | eliminated |

Long-context ITL p50 rises from 21.06 to 27.41 ms because the larger KV pool keeps more
requests active; ITL p99 still improves from 37.78 to 35.34 ms. Teacher-forced top-1
agreement with FP16 is **97.3% at 1k context** and **97.7% at 4k** (KL 0.0041/0.0038).
The FP16 profile remains available when quantization drift is unacceptable.
The `optimized` 1.7B profile also enables W8A16 attention; its decode step improves,
while its extra loaded-service benefit remains [unresolved](results/rtx4060/ab_w8a16_attention_chat_c6_bucket8_qwen3_1.7b.json).

Other measured outcomes on this card:

| workload | result | evidence |
|---|---|---|
| 0.6B, 3–4k prompts, same 3 GiB weights-plus-KV budget | smoothed INT8 KV: **166.5 → 190.2 output tok/s** (+14.2%), TTFT p50 **1.383 → 0.998 s** (-27.9%), KV capacity **17,680 → 34,832 tokens** (1.97×) | [long-context A/B](results/rtx4060/optimized_vs_fp16_long_c8_0.6b.json) |
| 4B-AWQ, one 2k-token prompt | hybrid INT4 prefill: TTFT **2,364 → 620 ms** (**3.81×**) versus tinygemm-only, with 32 new tokens | [AWQ prefill A/B](results/rtx4060/awq_prefill_path_ab_4b.json) |
| 1.7B, one 128-token prompt and 128 new tokens | tuned single-request harness: ITL p50 **8.92 ms**, **112 output tok/s**, peak allocation **2.68 GiB**; explicit naive decoder: **17.14 ms**, **58 output tok/s**, **3.26 GiB** | [optimized](results/rtx4060/single_request/Qwen3-1.7B_current_optimized.json) / [naive](results/rtx4060/single_request/Qwen3-1.7B_naive.json) |

The single-request harness has its own optimized settings (W8A16 MLP and vocabulary head,
INT8 KV, without W8A16 attention). Its result is separate from the loaded-service A/B.
Qwen3-4B-AWQ retains FP16 KV; the 3.81× figure is a prefill comparison, not an
end-to-end serving throughput claim. The final validation on this machine passed **316
non-CUDA tests and 265 CUDA tests**, with one expected 8 GB speculative-test skip.

The chronological A/B record is in [the RTX 4060 R&D journal](docs/rtx4060-rnd-priorities.md);
the complete results, limits and rejected paths are in [the Ada report](docs/rtx4060-ada-sm89-complete-report.md).

Serving surface: OpenAI-compatible `/v1/completions` and `/v1/chat/completions` (with
streaming, `usage`, logprobs, stop strings), per-request sampling (temperature, top-p,
top-k, min-p, penalties, seeds, stop tokens), Prometheus `/metrics`, health/readiness,
graceful drain. Measured opt-in profiles now include W8A16 weights, calibrated INT8 KV and
a Qwen3-4B-AWQ INT4 path. Not yet: speculative decoding in the batched path, multi-GPU or
structured output. See "Roadmap".

On the measured RTX 4060, direct FlashAttention prefill wins for long-context FP16 with
256-token pages; Flash decode and dense-gather Flash over 16-token pages lose end to end.
The Ada policy therefore keeps graphed Triton `per_head` decode, uses tiled prefill for
normal 16-token pages, and selects Flash prefill only when compatible page geometry is
explicitly chosen. See the complete report for the 0.6B/1.7B/4B numbers and every rejected
arm.

### Historical T4 baseline (2026-09-21, Qwen3-0.6B FP16 chat)

These numbers describe the earlier Tesla T4 run with approximately 656-token prompts;
they are not RTX 4060 measurements.

| metric | first T4 measurement (Sep 20) | later T4 measurement |
|---|---:|---:|
| prefill-carrying step p50 | 98 ms | 20.5 ms |
| decode-only step p50 (batch ~5) | 9.7 ms | 10.2 ms |
| ITL p50 / p99 | ~25 / 295 ms | 19.8 / 33 ms |
| TTFT p50 | 2.8 s | ~0.3–0.4 s |

The T4 policy used SDPA prefill and graphed per-head decode. The RTX 4060 instead selects
tiled prefill on normal 16-token pages and supports the measured quantized profiles.

## Portability

Three seams instead of three hard-coded assumptions (`docs/architecture.md` §9):

| | hook | what happens |
|---|---|---|
| **another model** | `engine/model/adapters.py` | Fused SwiGLU/RMSNorm/RoPE find their targets structurally and patch the model's own modeling module, so Llama, Mistral and Qwen need no per-family code. Geometry the paged kernels cannot serve (head_dim > 128, non-divisible GQA, sliding-window, MoE, MLA) is **refused at load with the reason**. |
| **another GPU** | `engine/backends/policy.py` | `MEASURED` maps an architecture to the settings an A/B *on that architecture* chose, with the journal entry that chose them. Anything else gets capability-led defaults (bf16 from sm_80, highest-priority runnable backend) labelled `unmeasured`, plus the command that would settle it. |
| **another kernel** | `engine/backends/registry.py` | `register(Backend(name=..., phase=..., run=..., available=...))`. `"auto"` picks by priority; a *named* backend that cannot run raises **with the reason** rather than falling back silently. |

```bash
python scripts/check_hooks.py --backends-only   # seconds, no warmup
```

prints the checkpoint's geometry and what got fused, then every backend with `ok`/`no`
and why, then the defaults this device will serve with. For example, a T4 reports:

```
  ok decode   per_head   p50
  no decode   flash      p80  <- flash_attn wheels require sm_80+; this device is sm_75
  no prefill  tiled      p60  <- tl.dot does not reach the tensor cores on sm_75
defaults: {'decode_attention': 'per_head', 'prefill_attention': 'sdpa', 'dtype': 'float16'}
```

## Quick start

```bash
pip install -e ".[dev,server]"     # torch, transformers, triton must match your CUDA
python -m pytest -q -m 'not cuda'  # non-CUDA tests
python -m pytest -q -m cuda        # GPU correctness gates (downloads Qwen/Qwen3-0.6B)
python scripts/check_hooks.py      # warmup captures every graph; step phases report
uvicorn engine.server.api:create_app --factory --port 8000
# Measured RTX 4060 long-context configuration:
uvicorn engine.server.api:create_rtx4060_flash_app --factory --port 8000
# Measured general RTX 4060 optimized profile (Qwen3-1.7B by default):
uvicorn 'engine.server.api:create_rtx4060_optimized_app' --factory --port 8000
curl -N localhost:8000/generate/stream -d '{"prompt":"Explain KV caching.","max_new_tokens":64}'

# or the OpenAI-compatible surface, with any OpenAI client pointed at /v1
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-1.7B",
  "messages": [{"role": "user", "content": "Explain KV caching in one sentence."}],
  "temperature": 0.7, "top_p": 0.95, "max_tokens": 64, "stream": true}'
curl localhost:8000/metrics
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
        → custom attention fn cuts the packed row: paged decode / selected prefill backend
        → argmax, one device→host copy, commit tokens, finish/preempt/admit
```

KV lives in per-layer pools `[num_blocks, 16, kv_heads, head_dim]`; each request owns a
block table. Triton kernels write K/V into pages and attend over them; RMSNorm, RoPE and
SwiGLU are fused Triton kernels installed onto the model's own modules.

Attention backends, selected per device (`decode_attention=` / `prefill_attention=`,
default `"auto"`):

| phase | backends |
|---|---|
| decode | `per_head` (Triton, the measured baseline) · `split_k` (FlashDecoding structure, for narrow grids) · `gqa` (shared group read; measured neutral on T4) · `flash` (sm_80+) |
| prefill | `sdpa` (gathered pages + torch SDPA, T4 default) · `per_token` (Triton, INT8-capable) · `tiled` (`tl.dot`, sm_80+) · `flash` (sm_80+) |

## Measuring

```bash
python -m benchmarks.reliability.ab --setting fused_step --prompt-profile chat --cuda-graphs --repeats 5 --duration 30
python -m benchmarks.reliability.ab --setting flash_attention --cuda-graphs   # sm_80+
python -m benchmarks.reliability.ab --setting decode_split_k --cuda-graphs
python -m benchmarks.kernels.paged_decode_regime_sweep --kernel both
python -m benchmarks.kernels.prefill_attention_ab --ptx-only      # does tl.dot reach the tensor cores on this GPU?
```

`benchmarks/reliability/ab.py` runs interleaved arms on warmed engines, checks greedy
tokens against stock Transformers first (tie-aware), and reports medians with run-to-run
spread; anything inside the spread is "unresolved". `scripts/t4_phase0_phase1.ipynb` is
the two-GPU Kaggle runner used for every recorded result.

## Repository

```
engine/      batching (the engine, sampler), backends (kernel registry + per-device policy),
             scheduler, runtime (request state, sampling params), cache (paging, prefix cache),
             graphs (capture), kernels (Triton, flash adapter), server (FastAPI + OpenAI),
             model (loader, family adapters), metrics (Prometheus)
benchmarks/  reliability (soak, A/B, sweep), kernels, batching, server, quantization, understanding
tests/       CPU tests + `-m cuda` gates mirroring engine/
docs/        architecture.md · optimization-journal.md · checkpoint.md · t4-reevaluation-plan.md ·
             design-decisions.md · understanding-journal.md · rtx4060-plan.md ·
             rtx4060-final-evaluation.md · rtx4060-ada-sm89-complete-report.md
results/t4/  transcribed T4 measurements
results/rtx4060/  raw Ada A/B runs, kernel searches, quality gates and single-request results
scripts/     check_hooks, token_margins, Kaggle/Colab setup, the T4 notebook
```

## Roadmap

Tier 1 (serving): **done** - batched sampling, OpenAI-compatible routes, Prometheus metrics.
Tier 2 (portability): **done** - backend registry, per-device policy, model-family
discovery, split-K decode and FlashAttention-2 backends. The sm_89 policy is measured;
unrecognized devices remain explicitly labelled unmeasured (`docs/rtx4060-plan.md`).
Tier 3 (performance): **Ada pass substantially complete** - Nsight-guided paged kernels,
tiled/Flash prefill policy, smoothed INT8 KV, W8A16 and 4B-AWQ are measured. N-gram
speculative decoding has a positive batch-1 probe but is not integrated.
Tier 4: second model family end to end, CPU/GPU step overlap, structured output.

## Correctness policy

Every engine path must reproduce stock Transformers' greedy tokens on the identity
prompts except at logit ties (top-2 margin < 0.02, one fp16 ulp); kernel swaps may
drift later than the first 8 tokens and are gated on the stock reference. Invariants
checked by the soak: no leaked KV pages, every request reaches a terminal state,
cancellation from every state. Negative results are recorded, never deleted.

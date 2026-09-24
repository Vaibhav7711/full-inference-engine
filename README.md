# full-inference-engine

**v0.1.0-beta** · Apache-2.0 · CPU test suite runs in CI; GPU results are reproduced by
hand and recorded in [`docs/`](docs/) and [`results/`](results/).

A single-GPU LLM serving engine built from the model down: continuous batching over a
paged KV cache, chunked prefill fused with decode into one forward per step, CUDA graphs
for every forward shape, custom Triton attention/KV kernels, an FCFS scheduler with
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
negative result with the numbers. The RTX 4060 release evidence is in
`docs/rtx4060-final-evaluation.md`.

## Measured status

**Tesla T4 (sm_75), Qwen3-0.6B fp16, chat profile ~656-token prompts** — the full
optimization arc, first measurement to now:

| | first measurement | now |
|---|---|---|
| prefill-carrying step p50 | 98 ms | **20.5 ms** |
| decode-only step p50 (batch ~5) | 9.7 ms | 10.2 ms (weight-read floor ~5 ms + attention ~4 ms) |
| ITL p50 / p99 | ~25 / 295 ms | **19.8 / 33 ms** |
| TTFT p50 | 2.8 s | ~0.3-0.4 s |

**RTX 4060 (sm_89), fp16, long profile** — FlashAttention evaluated phase by phase
([full evidence](docs/rtx4060-final-evaluation.md)):

| | change vs the Triton/SDPA default | |
|---|---|---|
| Flash **prefill**, 256-token pages, 0.6B | prefill step **−7.8%**, ITL p99 **−22.6%** | accepted |
| Flash **prefill**, 256-token pages, 1.7B | TTFT **−13.7%**, expected gap **−5.3%** | accepted |
| Flash **decode** | ITL p50 **+113.9%** | rejected — not stream-capture safe, so it costs graph replay |
| Dense-gather Flash prefill, 16-token pages | prefill step **+38.5%** | rejected |

Defaults are set per architecture by measurement, with the evidence recorded next to
them in `engine/backends/policy.py`. Also retired with numbers: tiled Triton prefill on
sm_75 (`tl.dot` emits no `mma.sync`), GQA-shared decode reads (L2 already serves the
second read), INT8 KV, prefix caching on random-prompt workloads.

### What this is, and is not

It is a working single-GPU serving engine with an OpenAI-compatible surface, validated
end to end on two GPU architectures with one model family, and a measurement record that
includes everything that failed.

It is **not** a vLLM replacement. Greedy speculative decoding now has an experimental,
default-off paged-engine path awaiting Kaggle T4 x2 validation; there is no speculative
sampling, quantized weights, tensor or pipeline parallelism, or structured output. Performance
is validated for Qwen3 only — other Llama-style families load through the same
structural hooks but have not been measured. INT8 KV is disabled (a known Ada
preemption drift, marked as an expected failure in the CUDA suite). See
[CHANGELOG.md](CHANGELOG.md) for the full limitation list.

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
and why, then the defaults this device will serve with:

```
  ok decode   per_head   p50
  no decode   flash      p80  <- flash_attn wheels require sm_80+; this device is sm_75x
  no prefill  tiled      p60  <- tl.dot does not reach the tensor cores on sm_75
defaults: {'decode_attention': 'per_head', 'prefill_attention': 'sdpa', 'dtype': 'float16'}
```

## Quick start

```bash
pip install -e ".[dev,server]"     # torch, transformers, triton must match your CUDA
python -m pytest -q                # CPU suite: 296 tests, no GPU or Triton needed
python -m pytest -q -m cuda        # GPU correctness gates (downloads Qwen/Qwen3-0.6B)
python scripts/check_hooks.py      # warmup captures every graph; step phases report
uvicorn engine.server.api:create_app --factory --port 8000
# Measured RTX 4060 long-context configuration:
uvicorn engine.server.api:create_rtx4060_flash_app --factory --port 8000
curl -N localhost:8000/generate/stream -d '{"prompt":"Explain KV caching.","max_new_tokens":64}'

# or the OpenAI-compatible surface, with any OpenAI client pointed at /v1
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
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
        → custom attention fn cuts the packed row: paged decode kernel / SDPA over gathered pages
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
             rtx4060-final-evaluation.md
results/      t4/ (transcribed) and rtx4060/ (JSON artifacts cited by the evaluation)
scripts/     check_hooks, token_margins, Kaggle/Colab setup, the T4 notebook
```

## Roadmap

Tier 1 (serving): **done** - batched sampling, OpenAI-compatible routes, Prometheus metrics.
Tier 2 (portability): **done** - backend registry, per-device policy, model-family
discovery, split-K decode and FlashAttention-2 backends. The sm_89 policy is measured;
unrecognized devices remain explicitly labelled unmeasured (`docs/rtx4060-plan.md`).
Tier 3 (performance): **Flash prefill measured on RTX 4060**; greedy speculative decoding
is integrated experimentally and has a reproducible Kaggle T4 x2 A/B gate. Next, validate
that result, use Nsight on the live decode step, and integrate weight-only INT8/INT4.
Tier 4: second model family end to end, CPU/GPU step overlap, structured output.

## Correctness policy

Every engine path must reproduce stock Transformers' greedy tokens on the identity
prompts except at logit ties (top-2 margin < 0.02, one fp16 ulp); kernel swaps may
drift later than the first 8 tokens and are gated on the stock reference. Invariants
checked by the soak: no leaked KV pages, every request reaches a terminal state,
cancellation from every state. Negative results are recorded, never deleted.

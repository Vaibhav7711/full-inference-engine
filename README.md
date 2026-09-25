# full-inference-engine

**v0.1.0-beta** · Apache-2.0 · validated on NVIDIA Tesla T4 (`sm_75`) with Qwen3 FP16

A single-GPU LLM serving engine built from the model downward: continuous batching over a
paged KV cache, chunked prefill fused with decode into **one forward per scheduler step**,
CUDA graphs for every forward shape, custom Triton attention and KV kernels, a preemptive
FCFS scheduler, per-request sampling, and an OpenAI-compatible HTTP surface. Hugging Face
`transformers` supplies the model definition and weights; everything that runs a request is
here.

On a Tesla T4, at matched KV capacity and concurrency limits, it sustains **1.06–1.67×
vLLM's batched output throughput** for Qwen3-0.6B/1.7B/4B at batch ≥ 4.

Every optimization in this repository was either measured to win under a documented
protocol, or is recorded as a negative result with its numbers. Seven were built, measured,
and **rejected** — three of them standard practice elsewhere. Two published claims were
retracted after re-examining their own evidence. That record is the point:
[**engineering report**](docs/engineering-report.md) · [**user guide**](docs/user-guide.md)

---

## Measured results

### Against vLLM — Tesla T4, matched KV budget, pre-tokenized prompts to both

256-token prompts, 128 generated tokens, greedy, `ignore_eos`, 2 warmup + 5 timed runs,
NVML memory for both engines, CUDA graphs enabled in both.

| model | batch | this engine | vLLM | ratio |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1 | 131 tok/s | 129 | 0.99 |
| Qwen3-0.6B | 4 | 453 | 362 | **1.25×** |
| Qwen3-0.6B | 8 | 750 | 513 | **1.46×** |
| Qwen3-0.6B | 16 | **1,069** | 639 | **1.67×** |
| Qwen3-1.7B | 16 | **514** | 412 | **1.25×** |
| Qwen3-4B | 16 | **222** | 209 | **1.06×** |

The advantage grows with batch and shrinks with model size — the signature of a per-step
overhead advantage, since larger models spend proportionally more time in GPU work. vLLM
wins single-request TTFT (19.7 ms vs 47.8 ms at a 32-token prompt) and uses less GPU memory.
Full protocol and caveats: [engineering report §5](docs/engineering-report.md#5-head-to-head-against-vllm).

### The optimization arc — Tesla T4, Qwen3-0.6B FP16, chat profile

Five interleaved 30-second closed-loop runs per arm at concurrency 8, each arm token-gated
against stock Transformers.

| | first measurement | now |
|---|---:|---:|
| prefill-carrying step p50 | 98 ms | **20.5 ms** |
| ITL p50 | ~25 ms | **19.8 ms** |
| ITL p99 | 295 ms | **33.0 ms** |
| TTFT p50 | 2.8 s | **0.32–0.41 s** |
| decode-only step p50 (batch ~5) | 9.7 ms | 10.2 ms — at its bandwidth floor |
| host staging per step | 3.4 ms | **0.26 ms** |

| what each change bought | effect |
|---|---|
| continuous batching vs sequential | **6.5×** throughput (24 → 154 tok/s at concurrency 16) |
| CUDA graphs on decode | 34.2 → 9.7 ms/step (**−72%**) |
| SDPA-over-pages chunked prefill | prefill step **−40.6%** chat, **−65.6%** long |
| CUDA graphs on prefill | prefill step **−56.7%**, ITL p50 **−57.4%** |
| fused decode+prefill step | prefill step **−17.6%**, ITL p50 **−18%** |
| warmup before serving | ITL p99 **−71%**, p50 unchanged |

### Rejected with evidence

| | measured | why |
|---|---|---|
| Tiled `tl.dot` prefill on Turing | **3× slower** | PTX: `mma_sync = 0`, 2052 FMA, 128 register spills — Triton emits MMA only from sm_80 |
| GQA-shared K/V decode reads | 0.95–1.06× | L2 already serves the duplicate read |
| FlashAttention-2 decode (Ada) | **+113.9% ITL** | not stream-capture safe, so it costs graph replay; and ties at the roofline anyway |
| Dense-gather Flash prefill at 16-token pages | +38.5% | the gather dominates the better kernel |
| INT8 KV cache | neutral on T4, drifts on Ada | disabled |
| Prefix caching on random-prompt workloads | unresolved | the workload shares no prefixes |

---

## Install and first run

```bash
pip install -e ".[dev,server]"     # torch must match your CUDA; don't reinstall it on Colab
python -m pytest -q                # 314 CPU tests — no GPU or Triton needed
python -m pytest -q -m cuda        # 193 GPU correctness gates
```

**As a library:**

```python
from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.model import load_model

loaded = load_model("Qwen/Qwen3-0.6B", dtype="float16")
engine = ContinuousBatchingEngine(
    loaded.model, loaded.tokenizer, loaded.device,
    max_active=8, num_blocks=1024, block_size=16,
    cuda_graph_batch_sizes=(1, 2, 4, 8),
)
engine.warmup()
print(engine.generate(["Explain KV caching in one sentence."], max_new_tokens=64))
```

**As a server**, with any OpenAI client:

```bash
uvicorn engine.server.api:create_app --factory --port 8000
until curl -sf localhost:8000/ready >/dev/null; do sleep 2; done   # warmup runs first

curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "Explain paged attention."}],
  "temperature": 0.7, "max_tokens": 120, "stream": true}'
curl localhost:8000/metrics
```

Worked examples: [`examples/library_usage.py`](examples/library_usage.py) (capacity
arithmetic, per-request sampling, staggered arrivals) and
[`scripts/live_smoke.py`](scripts/live_smoke.py) (an outside-process check of the HTTP
surface, including a concurrency speedup assertion).

**Sizing matters more than any other setting.** The KV pool, not the weights, bounds
concurrency: a Qwen3-0.6B/1.7B token costs 112 KiB of KV, so `num_blocks × block_size`
decides how many requests fit. See [user guide §3](docs/user-guide.md#3-sizing-the-kv-pool).

---

## How it works

```
HTTP / SSE (FastAPI) · OpenAI-compatible /v1
      │
ContinuousBatchingService — one worker thread owns the engine
      │
FCFSScheduler — admission bounded by KV capacity, LIFO preemption, prefill planning
      │
engine.step():  decode rows (1 token each) + prefill chunk rows (≤128 tokens each)
      └── ONE packed forward, replayed from a CUDA graph keyed by shape
            ├── paged decode attention    (Triton, per row × query head)
            ├── chunked prefill attention (SDPA over gathered pages · Triton · FA2)
            ├── fused RMSNorm / RoPE / SwiGLU (Triton, installed onto the model)
            └── one argmax, one device→host copy for the whole batch
```

Attention is a **registry of backends**, not a constant. Each declares why it cannot run on
a given GPU and geometry; `"auto"` picks the highest-priority available one, and a *named*
backend that cannot run raises **with the reason** rather than falling back silently.

| phase | backends |
|---|---|
| decode | `per_head` (default) · `split_k` · `gqa` · `flash` (sm_80+) |
| prefill | `sdpa` (default) · `per_token` (INT8-capable) · `tiled` (sm_80+) · `flash` (sm_80+) |

Per-architecture defaults live in `engine/backends/policy.py` **with the A/B that chose
them**. On an architecture nobody has measured, every reason string says `unmeasured` and
names the command that would settle it:

```bash
python scripts/check_hooks.py --backends-only   # seconds; what this GPU can run, and why not
```

Fused kernels find their modules structurally and RoPE is patched in the model's own
modeling module, so Llama-style checkpoints load with no new code. Geometry the paged
kernels cannot serve — `head_dim > 128`, non-divisible GQA, sliding-window attention, MoE,
MLA — is refused **at load, with the reason**.

---

## Documentation

| | |
|---|---|
| [engineering-report.md](docs/engineering-report.md) | design, method, results, negative results, roofline analysis, limitations |
| [user-guide.md](docs/user-guide.md) | install, serving, configuration reference, tuning, troubleshooting |
| [architecture.md](docs/architecture.md) | every file, all execution flows, each Triton kernel explained |
| [optimization-journal.md](docs/optimization-journal.md) | every result and retraction, dated, with commits |
| [checkpoint.md](docs/checkpoint.md) | current validated claims — and retracted ones |
| [rtx4060-final-evaluation.md](docs/rtx4060-final-evaluation.md) | the Ada FlashAttention campaign in full |
| [t4-speculative-decoding-plan.md](docs/t4-speculative-decoding-plan.md) | the pending speculative evaluation |
| [results/](results/) | measurement artifacts behind every number above |

## Repository

```
engine/      batching (engine, sampler) · backends (kernel registry + device policy) ·
             scheduler · runtime (request state, sampling) · cache (paging, prefix cache) ·
             graphs (capture) · kernels (Triton, FA2 adapter) · server (FastAPI + OpenAI) ·
             model (loader, family adapters) · metrics (Prometheus) · speculative (experimental)
benchmarks/  reliability (soak, interleaved A/B, sweeps) · kernels · batching · server ·
             quantization · speculative · understanding (trace scripts)
tests/       314 CPU tests + 193 `-m cuda` gates mirroring engine/
examples/    library_usage.py — the engine driven as an imported package
scripts/     check_hooks · verify_hooks · live_smoke · token_margins · notebooks
results/     t4/ · rtx4060/ — the JSON behind every number in the docs
```

## Measuring

```bash
python -m benchmarks.reliability.ab --setting fused_step --prompt-profile chat \
    --cuda-graphs --repeats 5 --duration 30
python -m benchmarks.kernels.roofline                         # measured bandwidth + decode floor
python -m benchmarks.kernels.prefill_attention_ab --ptx-only   # does tl.dot reach the tensor cores here?
```

Read every A/B with two rules. A change smaller than its reported `spread` is
**unresolved** and is not a result. A tail percentile counts only when
`lazy_graph_captures` is 0 in both arms — an in-window CUDA graph capture is 100–200 ms and
lands squarely in p99.

## Roadmap

1. **Weight-only quantization on the serving path** (W8A16 → W4A16). The decode step is at
   its weight-read floor, so fewer weight *bytes* is the only lever that moves it.
2. **Rewrite the kernel timing harnesses** and re-derive the decode tile/warp regime per
   architecture — the current regime is a T4 result reused unchanged elsewhere.
3. **Re-run the Ada results at the T4 protocol**, and A/B the `tiled` prefill kernel on
   sm_89, where its PTX gate passes but it has never been measured inside the engine.
4. **Split-K decode for batch 1–2**, after fixing graph capture to carry a per-graph
   context bound.
5. **Finish or remove speculative decoding** — the T4×2 evaluation is written but unrun.
6. **One non-Qwen checkpoint end to end**, converting a structural claim into evidence.

## Correctness policy

Every engine path must reproduce stock Transformers' greedy tokens on fixed prompts, except
at logit ties — a first difference at a position whose stock top-2 margin is under 0.02 (one
fp16 ulp) is a tie, not a divergence, and that threshold was set by measuring the margins
rather than by assumption. Invariants checked by a randomized soak: no leaked KV pages,
every request reaches a terminal state, cancellation works from every state. Negative
results are recorded, never deleted.

## What this is not

Not a vLLM replacement. No tensor or pipeline parallelism, no quantized weights on the
serving path, no structured output, no LoRA, no multimodal input, no `n > 1` sampling. INT8
KV is implemented but disabled. Speculative decoding has an experimental, **default-off**
greedy path whose speedup is not yet validated. Performance is validated for **Qwen3 on
Tesla T4** only; the RTX 4060 FlashAttention results in
[rtx4060-final-evaluation.md](docs/rtx4060-final-evaluation.md) were produced under a
weaker protocol (1–3 repeats vs 5 × 30 s) and are labelled preliminary. Other model
families load through structural hooks but have not been measured.

## License

Apache-2.0. See [LICENSE](LICENSE) and [CHANGELOG.md](CHANGELOG.md).

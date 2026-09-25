<div align="center">

# full-inference-engine

**A single-GPU LLM serving engine, built from the model down.**

Continuous batching over a paged KV cache · chunked prefill fused with decode into one
forward per step · CUDA graphs for every forward shape · custom Triton attention kernels ·
preemptive scheduling · OpenAI-compatible API

[![version](https://img.shields.io/badge/version-0.1.0--beta-blue)](CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![tests](https://img.shields.io/badge/CPU%20tests-314%20passing-brightgreen)](tests/)
[![gpu](https://img.shields.io/badge/GPU%20gates-193%20passing-brightgreen)](tests/)
[![platform](https://img.shields.io/badge/validated-Tesla%20T4%20%C2%B7%20Qwen3-orange)](docs/engineering-report.md)

[Engineering report](docs/engineering-report.md) ·
[User guide](docs/user-guide.md) ·
[Architecture](docs/architecture.md) ·
[Benchmarks](#benchmarks)

</div>

---

## What it does

Hugging Face `transformers` supplies the model definition and weights. **Everything that
runs a request is here**: the scheduler that owns KV memory, the paged cache, the attention
kernels that read it, the graph capture that replays forwards, the sampler, and the HTTP
surface.

```
prefill-carrying step   98 ms  →  20.5 ms        ITL p99   295 ms  →  33 ms
TTFT p50                 2.8 s →  0.35 s         throughput  24    →  1,069 tok/s
```

<sub>Tesla T4, Qwen3-0.6B FP16 · five interleaved 30 s closed-loop runs per arm at
concurrency 8 · every arm token-gated against stock Transformers</sub>

---

## The optimizations

Each row is a measured change, in the order it landed. Every number comes from an
interleaved A/B whose artifact is in [`results/`](results/).

| # | optimization | what it does | measured gain |
|---|---|---|---|
| 1 | **Continuous batching** | N sequences' next tokens in one forward, so the 1.19 GiB weight read is amortised across N instead of 1 | **6.5×** throughput<br><sub>24 → 154 tok/s @ concurrency 16</sub> |
| 2 | **Paged KV cache** | fixed-size pages + per-request page tables, so sequences of any length share one pool with no copying or fragmentation | enables 1 · sets the concurrency ceiling |
| 3 | **CUDA graphs on decode** | ~1,100 kernel launches per step recorded once per shape and replayed | **−72%** step<br><sub>34.2 → 9.7 ms</sub> |
| 4 | **Chunked prefill** | a new 700-token prompt is admitted 128 tokens at a time *alongside* decode instead of stalling it | decoder ITL **−69%** under load |
| 5 | **SDPA over gathered pages** | prefill attention that reaches the tensor cores by folding GQA into the query axis, with the mask built once per step instead of per layer | prefill step **−40.6%** chat<br>**−65.6%** long |
| 6 | **CUDA graphs on prefill** | the chunk forward captured per (rows × context × kernel) and replayed | prefill step **−56.7%**<br>ITL p50 **−57.4%** |
| 7 | **Fused decode + prefill step** | decode tokens and chunk tokens packed into *one* forward, so weights are read once for both and there is one sync per step | prefill step **−17.6%**<br>ITL p50 **−18%** |
| 8 | **Slice-copy metadata staging** | persistent pinned buffers and batched copies instead of per-element writes | host cost **3.4 → 0.26 ms** |
| 9 | **Fused Triton RMSNorm / RoPE / SwiGLU** | one kernel each, installed onto the model's own modules; FP32 accumulation preserves stock tokens | launch count per layer |
| 10 | **Warmup before readiness** | Triton JIT and every graph capture paid at startup, never inside a request | ITL p99 **−71%**<br>p999 **−63%** |
| 11 | **Preemption with recompute accounting** | LIFO victim selection under KV pressure; the oldest request always completes | survives 3× oversubscription |
| 12 | **Reachable-shape graph capture** | captures skipped for shapes the KV pool cannot physically hold | 255 → 215 graphs |

---

## Benchmarks

### Controlled comparison against vLLM

Tesla T4, **matched KV budget** (16,384 tokens for both), matched concurrency cap,
pre-tokenized prompt ids handed to both engines, greedy with `ignore_eos`, prefix caching
off in both, CUDA graphs on in both, 2 warmup + 5 timed runs, medians.

| model | batch | this engine | vLLM | |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 4 | **453** tok/s | 362 | **1.25×** |
| Qwen3-0.6B | 8 | **750** | 513 | **1.46×** |
| Qwen3-0.6B | 16 | **1,069** | 639 | **1.67×** |
| Qwen3-1.7B | 16 | **514** | 412 | **1.25×** |
| Qwen3-4B | 16 | **222** | 209 | **1.06×** |

The advantage grows with batch and shrinks with model size — 1.67× at 0.6B down to 1.06× at
4B — which is the signature of a per-step *overhead* advantage: the larger the model, the
more each step is dominated by GPU work. This is a controlled micro-benchmark at equal KV
budget on one GPU, **not a production bake-off**; vLLM at its own best configuration, with
tensor parallelism and quantization, is a different comparison. Full protocol, controls and
where vLLM wins: [engineering report §5](docs/engineering-report.md#5-head-to-head-against-vllm).

### Latency, Tesla T4 · Qwen3-0.6B FP16

| | chat profile (~656 tok) | long profile (~1.8k tok) |
|---|---:|---:|
| prefill-carrying step p50 | **20.5 ms** | 23.1 ms |
| decode-only step p50 | 10.2 ms <sub>(bandwidth floor ≈ 4.6 ms)</sub> | 10.4 ms |
| ITL p50 / p99 / p999 | **19.8 / 33.0 / 51.4 ms** | 22.4 / 43.9 / 60.2 ms |
| TTFT p50 | **0.32–0.41 s** | 2.9–3.7 s |
| host staging per step | 0.26 ms | 0.27 ms |

---

## Quick start

```bash
pip install -e ".[dev,server]"     # torch must match your CUDA; don't reinstall it on Colab
python -m pytest -q                # 314 CPU tests — no GPU or Triton required
```

### As a library

```python
from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.model import load_model

loaded = load_model("Qwen/Qwen3-0.6B", dtype="float16")
engine = ContinuousBatchingEngine(
    loaded.model, loaded.tokenizer, loaded.device,
    max_active=8,                       # concurrent sequences
    num_blocks=1024, block_size=16,     # KV pool = 16,384 tokens
    cuda_graph_batch_sizes=(1, 2, 4, 8),
)
engine.warmup()                          # Triton JIT + graph capture, once

print(engine.generate(["Explain KV caching in one sentence."], max_new_tokens=64))
```

### As a server, with any OpenAI client

```bash
uvicorn engine.server.api:create_app --factory --port 8000
until curl -sf localhost:8000/ready >/dev/null; do sleep 2; done   # readiness waits for warmup
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
for chunk in client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "Explain paged attention."}],
        stream=True, temperature=0.7, max_tokens=120):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Worked example with capacity arithmetic, per-request sampling and staggered arrivals:
[`examples/library_usage.py`](examples/library_usage.py).

> **Sizing tip.** The KV pool — not the weights — bounds concurrency. A Qwen3-0.6B/1.7B
> token costs 112 KiB of KV, so `num_blocks × block_size` decides how many requests fit.
> [User guide §3](docs/user-guide.md#3-sizing-the-kv-pool) has the arithmetic.

---

## How it works

```
HTTP / SSE (FastAPI)  ·  OpenAI-compatible /v1  ·  Prometheus /metrics
        │
ContinuousBatchingService ──── one worker thread owns the engine
        │
FCFSScheduler ──────────────── admission bounded by KV capacity
        │                      LIFO preemption · prefill planning
        ▼
engine.step()
   decode rows (1 token each)  +  prefill chunk rows (≤128 tokens each)
        │
        └─► ONE packed forward, replayed from a CUDA graph keyed by shape
                ├── paged decode attention      Triton, per (row × query head)
                ├── chunked prefill attention   SDPA over gathered pages · Triton · FA2
                ├── fused RMSNorm / RoPE / SwiGLU
                └── one argmax · one device→host copy for the whole batch
```

**Attention is a registry, not a constant.** Each backend declares why it cannot run on a
given GPU and geometry. `"auto"` picks the highest-priority available one; a *named* backend
that cannot run raises **with the reason** instead of falling back silently.

| phase | backends |
|---|---|
| decode | `per_head` ✓default · `split_k` · `gqa` · `flash` <sub>sm_80+</sub> |
| prefill | `sdpa` ✓default · `per_token` <sub>INT8-capable</sub> · `tiled` <sub>sm_80+</sub> · `flash` <sub>sm_80+</sub> |

Per-architecture defaults live in [`engine/backends/policy.py`](engine/backends/policy.py)
**with the A/B that chose them**. On an architecture nobody has measured, every reason
string says `unmeasured` and names the command that would settle it:

```bash
python scripts/check_hooks.py --backends-only    # seconds: what this GPU can run, and why not
```

---

## Features

| | |
|---|---|
| **Serving** | OpenAI `/v1/completions` + `/v1/chat/completions` with streaming, `usage`, logprobs, stop strings, chat templates · native `/generate` + SSE · health, readiness, graceful drain |
| **Sampling** | per-request temperature, top-p, top-k, min-p, repetition/presence/frequency penalties, reproducible seeds, stop tokens — applied as whole-batch GPU ops, with an all-greedy batch short-circuiting to a single argmax |
| **Scheduling** | FCFS admission under a KV budget, LIFO preemption with recompute accounting, chunked prefill under a token budget, cancellation from every request state |
| **Observability** | Prometheus histograms for TTFT, inter-token, end-to-end and queue latency · KV utilization, batch width, in-service graph captures · per-request latency reports |
| **Portability** | fused kernels find their modules structurally and RoPE patches the model's own module, so Llama-style checkpoints load with no new code; unsupported geometry is refused **at load with the reason** |
| **Correctness** | every path reproduces stock Transformers' greedy tokens except at measured logit ties · randomized soak asserts no leaked KV pages and a terminal state for every request |

---

## Documentation

| | |
|---|---|
| [**Engineering report**](docs/engineering-report.md) | the thesis: problem, design, method, results, roofline analysis, limitations |
| [**User guide**](docs/user-guide.md) | install, serving, KV sizing, configuration reference, tuning, troubleshooting |
| [Architecture](docs/architecture.md) | every file, all execution flows, each Triton kernel explained |
| [Optimization journal](docs/optimization-journal.md) | every result, dated, with commits |
| [Checkpoint](docs/checkpoint.md) | current validated claims |
| [results/](results/) | the JSON artifacts behind every number above |

## Scope

Validated on **Tesla T4 (`sm_75`) with Qwen3-0.6B / 1.7B / 4B in FP16**.

Runs on other CUDA GPUs and other Llama-style checkpoints — `head_dim ≤ 128`, divisible
GQA, dense feed-forward, no sliding window — but those combinations are not
performance-tuned; the decode tile regime is a T4 result. Mixture-of-experts,
multi-head-latent-attention and `head_dim > 256` checkpoints are refused at load.
[User guide §8](docs/user-guide.md#8-running-on-other-hardware) lists which families work.

Not included: tensor or pipeline parallelism, quantized weights on the serving path,
structured output, LoRA, multimodal input, `n > 1` sampling. Speculative decoding has an
experimental, default-off greedy path.

## Roadmap

1. **Weight-only quantization** (W8A16 → W4A16) — the decode step sits at its weight-read
   floor, so fewer weight *bytes* is the only lever that moves it
2. Re-derive the decode tile regime per architecture, with rewritten kernel timing harnesses
3. A/B the tiled Triton prefill kernel on `sm_89`, where its PTX gate passes
4. Split-K decode for batch 1–2, the single-user interactive case
5. Finish the speculative decoding evaluation
6. One non-Qwen checkpoint end to end

---

<div align="center">
<sub>Apache-2.0 · <a href="CHANGELOG.md">Changelog</a> · built and measured on Tesla T4</sub>
</div>

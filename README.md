# full-inference-engine

The first milestone is deliberately narrow: execute a real Hugging Face decoder-only
model on CUDA with an explicit prefill/decode loop.  `model.generate()` is never part
of the engine path.

## Stage 1: reference runtime

```
text → tokenizer → prefill (all prompt tokens) → first token
     → decode (one token + past KV) → next token → …
```

The runner currently supports a single greedy request. This is the trusted baseline
for later work on request state, batching, and cache allocation.

## Run on Google Colab

In Colab, select **Runtime → Change runtime type → T4 GPU** (or a stronger GPU),
then run these cells after cloning your GitHub repository:

```bash
!git clone https://github.com/<your-user>/full-inference-engine.git
%cd full-inference-engine
!bash scripts/setup_colab.sh
!python scripts/colab_preflight.py
```

Run the correctness test and baseline benchmark:

```bash
!python -m pytest -m cuda
!python -m benchmarks.inference.stage1 --prompt 'The capital of France is' --max-new-tokens 32 --warmup-runs 2 --runs 5
!cat results/stage1.json
```

`scripts/setup_colab.sh` intentionally does **not** reinstall PyTorch: Colab ships a
CUDA-matched PyTorch build, and replacing it can create an incompatible CUDA runtime.

The benchmark writes JSON records containing hardware, package versions, model,
generation configuration, latency breakdown, and memory measurements. Results from
different GPUs must not be compared directly.

## Stage 4: KV-cache accounting

Before introducing an allocator, inspect the cache the reference runtime actually
creates. This reports the GQA-aware analytic formula and physical cache-tensor bytes:

```bash
!python scripts/inspect_kv_cache.py --prompt 'Explain KV-cache memory.' --budget-gib 4
```

For BF16/FP16, the formula is:

```text
bytes/token = 2 (K and V) × layers × KV heads × head dimension × 2 bytes
```

This is intentionally separate from model-weight memory. The next cache stage will
use this evidence to establish a naïve allocation baseline before paging is considered.

## Stages 5, 7, and 8: baseline capacity, request state, and FCFS admission

The next runtime baseline combines a first-fit contiguous logical KV arena with an
explicit request state machine (`WAITING → PREFILLING → DECODING → FINISHED`) and a
strict FCFS scheduler. Each request reserves `prompt_tokens + max_new_tokens` in one
contiguous range. This is deliberately inefficient: it makes allocation failure and
external fragmentation observable before block-based paging is introduced.

These are deterministic CPU-only unit tests, so they can be run in Colab without
loading the model:

```bash
!python -m pytest tests/cache tests/runtime tests/scheduler -v
```

## Stage 6: streaming API

The server emits Server-Sent Events (SSE) one token at a time. It intentionally uses
the reference single-request runner; scheduling-aware streaming comes later.

```bash
!uvicorn 'engine.server.api:create_app' --factory --host 0.0.0.0 --port 8000
```

From a second terminal/session, send `POST /generate` for a complete response or
`POST /generate/stream` for `text/event-stream` events. The CUDA correctness suite
also verifies streamed token IDs equal normal greedy generation.

## Stage 9: static batching

Static batching keeps every row in the batch until every request finishes. It gives us
the throughput baseline and exposes wasted work from completed rows before continuous
batching is introduced.

```bash
!python -m benchmarks.batching.static \
  --prompt 'Explain KV caching in one sentence.' \
  --max-new-tokens 32 --batch-sizes 1,2,4,8 --warmup-runs 1 --runs 3 \
  --output results/static_batching.json
```

## Stage 10: continuous batching control plane

`ContinuousBatcher` replaces fixed membership with an explicit per-iteration plan:
new arrivals prefill, existing requests decode, and completed requests release their KV
reservation so the next waiting request can enter. The current component is tested at
the scheduler/control-plane level; it does not pretend to provide physical batched KV
compaction before the cache-execution layer exists.

```bash
!python -m pytest tests/batching/test_continuous.py -v
```

## Stages 11–12: block tables and paged access

The contiguous allocator is now accompanied by fixed-size physical blocks, request
block tables, capacity accounting, and a reference paged gather path. This proves the
logical-to-physical mapping before attempting model-specific paged attention. It lets
us quantify the tradeoff directly: external fragmentation falls, while each active
request can waste up to `block_size - 1` tokens internally.

```bash
!python -m pytest tests/cache/test_paging.py -v
```

Measure allocator behavior under the same deterministic mixed-length arrival stream:

```bash
!python -m benchmarks.cache.allocator_workload \
  --capacity-tokens 8192 --steps 1000 --arrival-probability 0.7 \
  --block-sizes 8,16,32,64 --output results/allocator_workload.json
```

## Stages 13–14: GQA reporting and INT8 weight-only reference

KV-cache inspection now reports query heads, KV heads, GQA group size, and the KV
memory avoided versus standard multi-head attention. Qwen3-0.6B's 32 query heads and
8 KV heads yield a 4× GQA ratio, so its KV cache is 75% smaller than an otherwise
identical MHA layout.

The included INT8 experiment uses per-output-channel weights and explicitly reports
storage/accuracy. It is a **reference** path that dequantizes before `F.linear`; it
tests the memory-quality tradeoff but must not be benchmarked as an optimized INT8
kernel.

```bash
!python -m pytest tests/quantization -v
```

```bash
!python -m benchmarks.quantization.int8_weight_only \
  --prompt 'Explain KV caching in one sentence.' --max-new-tokens 16 \
  --output results/int8_weight_only.json
```

## Why this exists

The naive path calls `model.generate()`, which hides prefill, decode, cache lifetime,
and timing. This baseline owns each of those operations explicitly, so each later
optimization can be tested against the same token sequence and measured in isolation.

### Current limits

* CUDA only; CPU execution is rejected intentionally.
* Batch size one and greedy sampling only.
* Hugging Face owns the physical KV tensors for this stage; a separate allocator comes
  only after the cache-memory baseline is measured.
* No server or scheduler yet.

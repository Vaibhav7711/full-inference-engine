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

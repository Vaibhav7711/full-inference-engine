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
!python -m benchmarks.inference.stage1 --prompt 'The capital of France is' --max-new-tokens 32
!cat results/stage1.json
```

`scripts/setup_colab.sh` intentionally does **not** reinstall PyTorch: Colab ships a
CUDA-matched PyTorch build, and replacing it can create an incompatible CUDA runtime.

The benchmark writes JSON records containing hardware, package versions, model,
generation configuration, latency breakdown, and memory measurements. Results from
different GPUs must not be compared directly.

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

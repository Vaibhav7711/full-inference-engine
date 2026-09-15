# full-inference-engine

A correctness-first, single-GPU LLM inference-runtime project for Google Colab. It
uses real Hugging Face Qwen checkpoints and owns the actual runtime loop:

```text
tokenize → prefill → KV cache → one-token decode → cache update → repeat
```

Hugging Face provides transformer layers and tokenizers. This repository owns the
inference loop, request lifecycle, cache experiments, scheduling, batching,
streaming, instrumentation, benchmarks, and correctness boundaries. `model.generate()`
is only used as a trusted test reference, never as the engine execution path.

## Run in Google Colab

Choose **Runtime → Change runtime type → GPU**, then run:

```python
!git clone https://github.com/Vaibhav7711/full-inference-engine.git
%cd full-inference-engine
!bash scripts/setup_colab.sh
!python scripts/colab_preflight.py
!python -m pytest -v
```

`requirements-colab.txt` intentionally does not reinstall PyTorch. Colab's supplied
build is CUDA matched; replacing it can create a CUDA/PyTorch incompatibility.

## Architecture

```text
Client → FastAPI/SSE → GenerationRequest → FCFS Scheduler
                                      └→ ContinuousBatcher control plane
                                             │
                     Explicit prefill/decode model runner
                                             │
              KV accounting / contiguous allocator / block tables
                                             │
                                        GPU execution
```

The custom page/block layer is currently a verified allocation and logical addressing
substrate. The physical attention path still uses Hugging Face KV tensors; this repo
does not claim paged attention is complete before it really is.

## Repository map

```text
engine/model/         CUDA model loader and explicit reference runner
engine/metrics/       CUDA-event timing and percentile summaries
engine/cache/         KV geometry, contiguous allocator, blocks, page tables/gather
engine/runtime/       request state machine
engine/scheduler/     FCFS admission control
engine/batching/      static runner and continuous-batching control plane
engine/server/        FastAPI complete and SSE streaming APIs
engine/graphs/        CUDA-Graph eligibility and static-shape capture
engine/quantization/  reference per-channel INT8 weight-only modules
engine/speculative/   vanilla draft proposal, verification, rollback, commit
benchmarks/           reproducible inference/cache/batching/quant/spec experiments
tests/                subsystem correctness tests
```

## How to use it

### Explicit one-request generation

```python
!python -m engine.cli \
  --prompt "Explain KV caching in one sentence." --max-new-tokens 32
```

`ExplicitDecodeRunner.prefill()` processes the full prompt and exposes both logits and
the model cache. `decode_one()` consumes one token with that cache. `generate()` stops
at EOS or `max_new_tokens`.

### Stage 1 baseline

```python
!python -m benchmarks.inference.stage1 \
  --prompt "Explain KV caching in one sentence." --max-new-tokens 32 \
  --warmup-runs 2 --runs 5 --output results/stage1_repeated.json
```

The JSON records GPU, memory, CUDA/PyTorch/Transformers versions, model/dtype,
workload, TTFT, prefill, per-token decode, p50/p95/p99, throughput, and peak memory.

### Streaming API

```bash
uvicorn 'engine.server.api:create_app' --factory --host 0.0.0.0 --port 8000
```

`POST /generate` and `POST /generate/stream` submit work to one background
`ContinuousBatchingEngine` worker. The worker owns all GPU/scheduler state, batches
requests arriving from concurrent HTTP handlers, and exposes paged KV, decode-first
scheduling, and the recommended padded CUDA-Graph buckets (`2,4,8,16`) through the
serving API. SSE emits token ID, text, index, and EOS/LENGTH termination.

The service has a bounded ingress/scheduler queue and returns HTTP 429 when it is full.
It enforces a prompt-token limit and request deadline. SSE disconnects and timeouts are
routed through the GPU worker's cancellation queue so KV state is released without an
HTTP handler mutating scheduler state.

### Static batching

```python
!python -m benchmarks.batching.static \
  --prompt "Explain KV caching in one sentence." --max-new-tokens 32 \
  --batch-sizes 1,2,4,8 --warmup-runs 1 --runs 3 \
  --output results/static_batching.json
```

The static runner left-pads unequal prompts and supplies per-row position IDs. Finished
requests deliberately remain as EOS rows, exposing the wasted work that continuous
batching should eliminate.

### KV cache and allocator experiments

```python
!python scripts/inspect_kv_cache.py --prompt "Explain KV-cache memory." --budget-gib 4

!python -m benchmarks.cache.allocator_workload \
  --capacity-tokens 8192 --steps 1000 --arrival-probability 0.7 \
  --block-sizes 8,16,32,64 --output results/allocator_workload.json
```

### CUDA Graphs, INT8 reference, and speculation

```python
!python -m benchmarks.inference.cuda_graph_decode \
  --prompt "Explain KV caching in one sentence." --decode-steps 32 \
  --output results/cuda_graph_decode.json

!python -m benchmarks.quantization.int8_weight_only \
  --prompt "Explain KV caching in one sentence." --max-new-tokens 16 \
  --output results/int8_weight_only.json

!python -m benchmarks.speculative.vanilla \
  --prompt "Explain KV caching in one sentence." --max-new-tokens 32 \
  --speculation-depths 1,2,4,6,8 --output results/vanilla_depth_sweep.json
```

The INT8 implementation is a memory/quality reference: it dequantizes before
`F.linear`, so it is not an optimized INT8 throughput claim. The speculative benchmark
is retained as an experiment, but disabled as a serving default (see results).

## Stages and implementation status

| Stage | Status | What exists |
| --- | --- | --- |
| 1. Explicit inference | Complete | Real Qwen3 CUDA prefill/decode, cache ownership, greedy stopping. |
| 2. Correctness | Complete | HF greedy token comparisons for prompt/output lengths, EOS, limits. |
| 3. Instrumentation | Complete baseline | CUDA Events; repeated p50/p95/p99 benchmark results. |
| 4. KV memory model | Complete | Analytic geometry validated against observed HF cache bytes. |
| 5. Contiguous allocator | Complete | First-fit ranges, release/merge, external fragmentation. |
| 6. Streaming | Complete reference | FastAPI complete/SSE API plus output-equivalence test. |
| 7. Request state | Complete | WAITING→PREFILLING→DECODING→FINISHED plus cancelled/failed/rejected. |
| 8. FCFS scheduler | Complete control plane | Admission, capacity rejection, explicit waiting/active sets. |
| 9. Static batching | Complete | Explicit mixed-length batched prefill/decode and sweep. |
| 10. Continuous batching | Complete control plane | Dynamic entry/exit plans; no physical batch compaction yet. |
| 11. Block allocator | Complete substrate | Fixed blocks, reuse, dynamic growth, internal waste accounting. |
| 12. Paged execution | Partial | Correct block tables and paged gather; no paged attention kernel. |
| 13. GQA | Complete reporting | Query/KV heads, group ratio, MHA-equivalent memory saving. |
| 14. INT8 | Partial reference | Per-channel INT8 memory/quality path; no optimized kernel or INT4. |
| 15. CUDA Graphs | Complete restricted experiment | Correct fixed-shape/static-cache batch-1 graph. |
| 16. Vanilla speculation | Complete experiment, disabled | Explicit propose/verify/rollback; negative T4 performance result. |
| 17–19 | Deferred | Custom cache speculation and adaptation not justified after vanilla loss. |
| 20. Triton/CUDA | Deferred | Must come from profiler evidence, not resume-driven implementation. |
| 21. Deep profiling | Partial | CUDA-event layer exists; PyTorch profiler/Nsight workflow remains. |
| 22. Load testing | Deferred | Needs physical continuous batched execution and workload driver. |
| 23. Failure matrix | Partial | State/allocator errors covered; OOM/disconnect/batch-failure tests remain. |

## Key design rationale

### Prefill/decode separation

The naive path hides the two fundamentally different operations in `model.generate()`.
Prefill is prompt-length work; decode is a cache-backed one-token recurrence. Exposing
them makes TTFT, per-token latency, cache growth, and future batching measurable.

### KV-cache formula and GQA

```text
KV bytes/token = 2 × layers × KV heads × head dimension × dtype bytes
```

For Qwen3-0.6B in BF16: 28 layers, 8 KV heads, head dimension 128:

```text
2 × 28 × 8 × 128 × 2 = 114,688 bytes/token = 112 KiB/token
```

Qwen has 32 query heads and 8 KV heads: a 4:1 GQA ratio, yielding 75% less KV memory
than an equal model with 32 KV heads. Weight memory and KV memory are tracked as
separate optimization dimensions.

### Contiguous versus paged allocation

Contiguous reservation is simple but suffers external fragmentation: total free memory
may be sufficient while no single range fits a request. Paging removes that constraint,
but incurs internal waste in each request's final partly used block and eventual
mapping/gather overhead. The project therefore benchmarks allocator behavior before
claiming a paging win.

### CUDA Graph limitations

Graphs require stable tensor addresses, fixed batch/sequence shape, static cache, and
stable control flow. They cannot be blindly applied to dynamic continuous batching.

### Why not EAGLE or Medusa now?

EAGLE and Medusa require specialized trained auxiliary heads; stock Qwen checkpoints
are not drop-in candidates. Vanilla draft-model speculation is the proper baseline.

## Measured Colab Tesla T4 evidence

Hardware/software: Tesla T4 (14.56 GiB), CUDA 12.8, PyTorch 2.11.0+cu128,
Transformers 5.16.1.

### Stage 1 baseline

Qwen3-0.6B BF16, batch 1, 8-token prompt, 32 greedy output tokens:

| Metric | Result |
| --- | ---: |
| Prefill | 348.97 ms |
| TTFT | 349.70 ms |
| Mean decode | 44.70 ms/token |
| Steady decode rate | 22.37 tokens/s |
| End-to-end output rate | 18.41 tokens/s |
| Peak allocated memory | 1.12 GiB |
| Peak reserved memory | 1.14 GiB |

Use the repeated benchmark command above for distributions; this table is the initial
recorded run, not a cross-GPU comparison.

### KV validation

For a six-token prompt, analytic and observed HF cache values both measured 688,128
bytes (ratio 1.0), validating the geometry model.

### Allocator workload

8,192-token capacity, 1,000 steps, 0.7 arrival probability, seed 7:

| Policy | Admissions | Rejections/exhaustions | Fragmentation |
| --- | ---: | ---: | --- |
| Contiguous | 554 | 171 rejected | mean external 0.737 |
| Paged 8-token | 673 | 52 + 44 | mean internal 147 tokens |
| Paged 16-token | 672 | 53 + 42 | mean internal 310 tokens |
| Paged 32-token | 659 | 66 + 51 | mean internal 622 tokens |
| Paged 64-token | 635 | 90 + 69 | mean internal 1,180 tokens |

Eight-token blocks are the current candidate, pending actual paged-attention overhead
measurement.

### CUDA Graph experiment

Qwen3-0.6B, static cache, batch 1, eight prompt tokens, 32 decode steps; capture cost
excluded:

| Path | Decode latency |
| --- | ---: |
| Normal static-cache | 49.59 ms/token |
| CUDA Graph replay | 11.53 ms/token |

Speedup: **4.30×**. Graph and static-cache paths both matched the original dynamic
cache reference token sequence. Retain this only for fixed-shape execution.

### Vanilla speculative decoding: deliberate non-win

Target Qwen3-1.7B + draft Qwen3-0.6B on the same T4, depth 4: exact greedy output,
14 target rounds instead of 32, but 5,010 ms versus 1,886 ms baseline: **0.376×**
speedup (2.66× slower) with 34.5% acceptance. Depth 1 was also correct and slower.

On one GPU, target and draft execute serially; draft work, low acceptance, and cache
rollback cost dominate. Vanilla speculation is therefore disabled by default. This is
a documented failed optimization, not hidden work.

## Correctness policy

| Component | Required comparison |
| --- | --- |
| Explicit decode | Hugging Face greedy token IDs |
| Static batching | Individual explicit runs |
| Streaming | Non-streaming explicit output |
| Paged gather | Equivalent contiguous logical gather |
| CUDA Graph | Static normal and dynamic-cache reference output |
| INT8 | Numerical linear error plus benchmarked output quality |
| Speculation | Ordinary target greedy output |

Run all tests after changes:

```bash
python -m pytest -v
```

## Remaining work

The project is substantial but not “finished” as a full vLLM-class runtime. The
remaining work should be justified by measurement:

1. Profile baseline/static batch with `torch.profiler`, then use Nsight only where
   timeline evidence warrants it.
2. Bind page tables to a real paged-attention execution path and compare memory benefit
   against gather/attention overhead.
3. Add server load generation: steady, bursty, randomized arrivals; TTFT p50/p95/p99,
   queue/service time, occupancy, throughput, and GPU utilization.
4. Add failure tests for OOM, cache exhaustion, cancellation, disconnect, malformed
   requests, model errors, and partial batch failures.
5. Choose a Triton/CUDA target only after profiling establishes a bottleneck.

Honest measured failures and clear limits are part of the engineering evidence.

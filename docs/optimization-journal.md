# Inference Engine Optimization Journal

This journal is the durable record of the single-T4 optimization program. Colab is
the experimental scratchpad; this file records only reproducible evidence and final
engineering decisions.

## Working protocol

Every optimization must record:

1. The bottleneck or hypothesis.
2. The reference implementation and correctness oracle.
3. The isolated code change and commit.
4. Exact Colab commands and environment.
5. Correctness, latency, throughput, memory, and relevant profiler results.
6. A `KEEP`, `REVISE`, or `REVERT` decision.
7. The follow-up action and any superseded files removed.

Rules:

- Correctness is a hard gate.
- Performance is measured after warmup on the same workload.
- Custom kernels are compared with the best available PyTorch/CUDA baseline.
- A slower optimization is reverted, but its result stays documented.
- Reference code is removed only after its replacement passes all gates.
- Profiler latency is not compared directly with unprofiled latency.
- Separate Colab runs can expose regressions, but causal attribution requires a
  same-session, preferably interleaved A/B experiment.

## Target environment

- GPU: NVIDIA Tesla T4, 14.56 GiB
- Compute capability: 7.5 (Turing)
- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Transformers: 5.16.1
- Triton: 3.6.0
- Model: Qwen/Qwen3-0.6B
- Observed model revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Primary runtime dtype: FP16

PyTorch reports that BF16 operations are supported, but the T4 has no native BF16
Tensor Core path. Runtime telemetry therefore reports framework support separately
from native hardware acceleration.

## Phase 0 — Reproducible T4 baseline

Status: `COMPLETE`

### Repository and measurement infrastructure

Commit: `370a11a` — `Establish reproducible T4 dtype baseline`

Changes:

- Added hardware-aware `auto` dtype selection.
- Selected FP16 on Turing/T4.
- Recorded requested and resolved model revisions.
- Added common environment and deterministic-token benchmark helpers.
- Added isolated FP16/BF16 benchmark subprocesses.
- Added dtype-policy tests.
- Removed the redundant walkthrough and empty test package files.

Commit: `10c28cf` — `Add T4 operator profiling baseline`

Changes:

- Added explicit-reference operator profiling.
- Added optional Chrome trace export.
- Distinguished framework BF16 support from native BF16 Tensor Cores.

### Correctness

- Dtype-policy tests: passed.
- Explicit decode versus Hugging Face greedy generation: passed.
- All five repetitions were deterministic for each dtype.
- FP16 and BF16 emitted identical token IDs for the measured prompt.

### Dtype baseline

Workload: 10 prompt tokens, 32 greedy output tokens, two warmups, five runs.

| Metric | FP16 | BF16 | Decision |
| --- | ---: | ---: | --- |
| TTFT p50 | 43.571 ms | 59.026 ms | FP16 26.2% lower |
| Decode p50 | 40.004 ms/token | 39.650 ms/token | effectively tied |
| End-to-end p50 | 1324.107 ms | 1317.921 ms | effectively tied/noisy |
| Aggregate throughput | 23.741 tok/s | 24.323 tok/s | effectively tied/noisy |
| Peak allocated | 1,206,714,368 B | 1,206,714,368 B | equal |

Decision: `KEEP` FP16 as the T4 default because it materially improves prompt prefill
and uses the T4's native FP16 Tensor Core path. Do not claim a decode-speed advantage
from this short workload.

### Operator trace

The profiler approximately doubled end-to-end latency, so its timings are diagnostic
only. Important counts from 31 decode forwards over 28 transformer layers:

- 4,340 GEMV kernel calls.
- 3,647 `aten::cat` calls.
- 3,534 `aten::copy_` calls.
- 3,616 reduction-kernel calls.
- Large groups of 1,736 and 868 matrix operations.

Interpretation: batch-one decode is dominated by many small GEMV and elementwise
launches, while the dynamic cache repeatedly concatenates K/V tensors. Continuous
batching and direct paged KV access are higher-value targets than Python-loop tuning.

## Phase 1 — Reference-loop synchronization experiment

Status: `COMPLETE — EXPERIMENT REVERTED`

### Hypothesis

The runner synchronized a CUDA timing event and then immediately called `.item()` on
the selected token. Since `.item()` also waits for the GPU, the explicit event wait
appeared redundant.

### Change

Commit: `1c74f0c` — `Remove redundant per-token CUDA synchronization`

The experiment deferred event-duration reads until the end of generation. Preserving
per-token timings required one distinct start/end event pair per decode step.

### Result

| Metric | Reference | Experiment | Change |
| --- | ---: | ---: | ---: |
| TTFT p50 | 43.571 ms | 61.837 ms | 41.92% worse |
| End-to-end p50 | 1324.107 ms | 1431.373 ms | 8.10% worse |
| Decode p50 | 40.004 ms/token | 42.797 ms/token | 6.98% worse |
| Throughput | 23.741 tok/s | 21.220 tok/s | 10.6% worse |

Interpretation:

- The actual dependency remained: Python still needed `.item()` once per token.
- Creating and retaining many CUDA events introduced additional bookkeeping.
- The TTFT regression cannot be caused by decode-event allocation, which happens
  after TTFT. It demonstrates meaningful run-to-run Colab/GPU variation.
- Because this was not an interleaved same-session A/B measurement, the complete
  regression cannot be causally assigned to the code change.
- There was no demonstrated performance benefit, so keeping the change was unjustified.

Decision: `REVERT`.

Commit: `f1e11a3` — `Revert "Remove redundant per-token CUDA synchronization"`

The original faster reference runner is active on `main`.

## Cache baseline — DynamicCache versus reference PagedCache

Commit: `79d8671` — `Make paged cache benchmark apples to apples`

The earlier benchmark compared different generation loops. The replacement runs
DynamicCache and PagedCache through the same manual greedy loop and verifies exact
token equality.

Correctness: all paged-cache tests passed; all variants emitted the reference tokens.

Workload: 41 final sequence tokens, FP16, two warmups, five measured runs.

| Cache | Median | Throughput | Change | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| DynamicCache | 1379.997 ms | 23.19 tok/s | baseline | 1151.10 MiB |
| Paged, block 8 | 1426.099 ms | 22.44 tok/s | 3.34% slower | 1153.62 MiB |
| Paged, block 16 | 1395.510 ms | 22.93 tok/s | 1.12% slower | 1155.29 MiB |
| Paged, block 32 | 1404.465 ms | 22.78 tok/s | 1.77% slower | 1162.29 MiB |

Decision: `KEEP` block size 16 as the current baseline candidate.

- Block 8 grew every layer from four to eight blocks and paid growth/copy overhead.
- Block 16 needed four blocks per layer with no growth.
- Block 32 reserved 128 tokens per layer for a 41-token sequence, doubling unused
  capacity relative to block 16.
- The single-request paged path is already close enough to DynamicCache; its important
  remaining work is shared-pool continuous execution, not further gather micro-tuning.

## Phase 2 — Unified scheduling and persistent batch metadata

Status: `IN PROGRESS`

Entry gate:

- Validate the K4 batched paged-decode kernel on the active Colab stack.
- Validate full continuous generation against independent reference generations.
- Record throughput at active widths 1, 2, 4, 8, and 16.

Planned change:

- Replace duplicate `GenerationRequest` and `SeqState` state.
- Replace the scheduler, control-plane batcher, and engine-local queues with one
  scheduler.
- Replace request-owned Python block lists and parallel logical paging abstractions
  with one production KV block manager.
- Preallocate and persist GPU batch metadata rather than rebuilding it each token.

Acceptance gate:

- Exact deterministic token correctness.
- No allocation leaks across completion, cancellation, or failure.
- Throughput within 5% of the pre-refactor K4 baseline before further optimization.

### Pre-refactor K4 throughput baseline

Environment: target T4 environment above. Workload: 16 requests, 32 output tokens per
request, block size 16, 1,024 physical blocks, with a compilation/cache warmup.

| Maximum active sequences | Elapsed | Throughput | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 22.056 s | 23.2 tok/s | 1.00x |
| 2 | 12.116 s | 42.3 tok/s | 1.82x |
| 4 | 6.732 s | 76.1 tok/s | 3.28x |
| 8 | 4.949 s | 103.5 tok/s | 4.46x |
| 16 | 3.006 s | 170.3 tok/s | 7.34x |

Interpretation:

- Physical batching is working and materially amortizes model-weight reads.
- Width 16 is the best measured point and processes the workload in one active wave.
- Scaling is sublinear because prefill remains sequential and decode still has Python
  metadata construction and per-layer launch overhead.
- The Phase 2 refactor must retain at least 161.8 tok/s at width 16 (a 5% tolerance)
  before any further optimization claim is accepted.

### Block-ownership unification — accepted

Commit: `acc719d` — `Unify continuous KV block ownership`

Change:

- Replaced the reference-only `PagedKVCacheManager` with `KVBlockManager`.
- `KVBlockManager` is now the sole owner of each live request's physical block IDs,
  committed sequence length, capacity growth, release, and fragmentation accounting.
- Continuous sequence state references its manager-owned allocation instead of copying
  a second mutable Python block list.
- Decode now guarantees capacity first and constructs the GPU block table once. The
  previous path constructed it before growth and discarded/rebuilt it after growth.
- The K4 Triton kernel and its tensor contract are unchanged.

Local validation:

- All project Python files parse successfully.
- KV ownership/growth/release smoke test passed.
- Synthetic paged allocator workload passed.

T4 result:

- All allocator, paging, K4 kernel, and continuous-generation tests passed.
- Sequential throughput improved from 23.2 to 23.7 tok/s.
- Width-16 throughput improved from 170.3 to 171.2 tok/s (+0.5%).
- Width-16 speedup was 7.22x versus the refactored sequential path.
- The result is above the 161.8 tok/s acceptance floor.

Decision: `KEEP`.

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

### Direct pool-backed prefill KV write — accepted

Commit: `9eda221` — `Write prefill KV directly with Triton`

Hypothesis:

Prefill previously created a temporary `PagedCache`, grew and populated its per-layer
pages, then copied every token of every layer again into the continuous engine's shared
KV pool through nested Python loops. Writing the model-produced K/V directly into the
authoritative shared pool should remove temporary KV ownership, repeated allocations,
and thousands of Python-indexed CUDA assignments.

Change:

- Added a `DynamicCache`-compatible prefill adapter that returns the current K/V to
  stock SDPA while making the shared pool the only persistent owner.
- Added one Triton program that writes K and V together from `[1, H, S, D]` tensors to
  arbitrary physical blocks in `[blocks, block_size, H, D]` storage.
- Removed the scratch `PagedCache` allocation and the nested layer/token scatter from
  the production continuous-prefill path.
- Added boundary and non-contiguous-block correctness tests for the kernel. The existing
  D1 integration test independently compares every stored layer/token against the old
  paged-cache reference, and D2/D3 verify token-level decode and generation equivalence.

Risk and gate:

- The adapter relies on the verified Transformers cache-update contract and on K/V being
  passed after RoPE, as established by the existing integration path.
- Kernel compilation can make a first un-warmed run slower, so the throughput benchmark
  must retain its warmup.
- Accept only if all kernel and staged continuous tests pass and width-16 throughput is
  at least 164.0 tok/s (within 5% of the immediate 172.6 tok/s baseline). Record all
  concurrency points; do not claim a speedup from normal run-to-run noise.

T4 result:

- All focused kernel and staged continuous-generation tests passed.
- Sequential throughput was 23.3 tok/s.
- Width-8 throughput was 109.3 tok/s, up 4.0% from 105.1 tok/s.
- Width-16 throughput was 178.3 tok/s, up 3.3% from 172.6 tok/s.
- Width-16 speedup was 7.65x over the same run's sequential path.

Decision: `KEEP`.

### Batched decode KV write — accepted

Commit: `a8469ab` — `Fuse batched decode KV writes`

Hypothesis:

The decode attention callback still loops over active requests in Python for every
transformer layer. Each request performs two `.item()` reads from CUDA metadata before
two indexed K/V assignments. Replacing this with one request-by-head Triton grid should
remove host synchronization and issue one fused K/V write kernel per layer.

Change:

- Added a batched single-token Triton writer for `[N, H, 1, D]` decode K/V.
- Sequence positions and physical-block lookup remain on the GPU.
- Replaced the per-request Python loop in the physical decode callback.
- Added correctness coverage across block boundaries, distinct sequence lengths, and
  non-contiguous physical block tables.

Gate:

- All kernel and continuous-generation tests must remain token-exact.
- Width-16 throughput must remain at least 169.4 tok/s (within 5% of the immediate
  178.3 tok/s baseline).
- Because this modifies every decode layer and token, a same-session repeated benchmark
  will be required if the observed difference is small or unexpectedly negative.

T4 result:

- All focused KV-write and staged continuous-generation tests passed.
- Sequential throughput was 23.5 tok/s.
- Width-8 throughput was 154.8 tok/s, up 41.6% from 109.3 tok/s.
- Width-16 throughput was 247.8 tok/s, up 39.0% from 178.3 tok/s.
- Width-16 speedup reached 10.55x over the same run's sequential path.
- The scaling-specific improvement strongly matches removal of work proportional to
  active requests: the old callback performed 2 CUDA `.item()` synchronizations per
  request per layer, or 896 synchronization points per decode step at width 16 across
  Qwen3-0.6B's 28 layers.

Decision: `KEEP`.

### In-kernel decode length offset — accepted

Commit: `291ccf8` — `Move decode length offset into attention kernel`

Hypothesis:

After the KV-write synchronization bottleneck was removed, the attention callback still
formed `seq_lens + 1` independently in every transformer layer. Applying the constant
offset when each Triton attention program loads its sequence length removes 28 temporary
tensors and elementwise kernel launches per decode step.

Change:

- Added a compile-time `length_offset` to the batched paged-attention kernel.
- Continuous decode now passes the persistent pre-write length buffer directly and uses
  an in-kernel offset of one.
- Other callers retain the default zero offset and unchanged semantics.
- Added direct equality coverage between materialized incremented lengths and the
  in-kernel offset path.

Gate:

- All paged-attention, KV-write, and continuous-generation tests must pass.
- Width-16 must remain at least 235.4 tok/s (within 5% of 247.8 tok/s).
- Expect a smaller improvement than the prior change; accept a neutral result because
  it also removes repeated allocations, but revert a repeatable regression.

T4 result:

- All 21 paged-attention, KV-write, and continuous-generation tests passed.
- Sequential throughput was 23.9 tok/s.
- Width-4 throughput was 86.3 tok/s.
- Width-8 throughput was 142.6 tok/s; this individual point was below the preceding
  154.8 tok/s run and is treated as cross-run variance rather than a separate claim.
- Width-16 throughput was 263.5 tok/s, up 6.3% from 247.8 tok/s.
- Width-16 scaling reached 11.01x over the same run's sequential path.

Decision: `KEEP`.

### Post-synchronization decode profile — complete

Commit: `c9bcf82` — `Add continuous decode profiler`

Purpose:

The major known host-synchronization and temporary-length launches have now been
removed. The next optimization must be selected from the new measured profile rather
than from the obsolete Phase 0 operator trace.

Change:

- Added a continuous-engine profiler that warms the model and Triton kernels first.
- Prefill occurs outside the profiling window, isolating batched decode.
- Reports separate top operators by self GPU and self CPU time, including call counts.
- Saves structured environment, configuration, and operator data for later comparison.

T4 evidence (concurrency 16, 31 decode steps, decode only):

- Matrix multiplication remained the largest GPU category: 6,107 `aten::mm` calls and
  237.7 ms self GPU time. Most are expected model projections and cannot simply be
  removed.
- K4 paged attention used 35.8 ms across 868 calls, exactly 28 layers × 31 steps. It is
  material but no longer the dominant end-to-end bottleneck.
- The model issued 3,503 reduction chains: `mean` used 35.7 ms, with corresponding
  `pow`, `rsqrt`, `add`, and multiply kernels. This identifies fused Triton RMSNorm as
  the next substantial GPU elementwise target; Qwen's extra Q/K norms explain why the
  count exceeds two normalization sites per transformer layer.
- There were 40,548 `cudaLaunchKernel` calls and 283.5 ms CPU launch time, confirming
  that fusion/launch reduction now matters alongside GEMM performance.
- There were 496 `cudaStreamSynchronize` calls, exactly 16 requests × 31 decode steps.
  These originate from reading each sampled token separately with CUDA `.item()`.
  Batching the sampled-token device-to-host transfer can reduce this to one necessary
  synchronization per decode step before larger RMSNorm work begins.
- `aten::cat` still appeared 1,767 times and should be localized with a stack/shape
  trace if it remains prominent after the synchronization and norm passes.

Next order:

1. Batch sampled-token materialization to remove per-request stream synchronizations.
2. Implement and validate model-compatible Triton RMSNorm, including Q/K norm shapes.
3. Re-profile before considering SwiGLU or projection-level fusion.

Decision: `DIAGNOSTIC COMPLETE`; no performance claim is made from profiler timings.

### Batched sampled-token materialization — rejected and reverted

Commit: `f74c6ce` — `Batch decode token transfer to host`

Hypothesis:

The optimized width-16 decode profile reported 496 stream synchronizations, exactly one
per request per decode step (16 × 31). The scheduler needs token IDs on the CPU, but it
needs only one device/host dependency boundary for the complete batch.

Change:

- Added a persistent pinned-host token buffer sized to `max_active`.
- Copy the complete GPU argmax result asynchronously into that buffer.
- Synchronize the current CUDA stream once, then perform request lifecycle and EOS work
  from host-resident values.
- Removed per-request CUDA `.item()` calls from the decode loop. The prefill path remains
  single-request and therefore unchanged.

Gate:

- All continuous-generation tests must remain token-exact, including mixed lengths and
  early completion/reclamation.
- Width-16 throughput must remain at least 250.3 tok/s (within 5% of 263.5 tok/s).
- Re-run the decode profiler after the throughput gate. Expected synchronization count
  is approximately 31 rather than 496 for this fixed 16 × 31 workload.

T4 result:

- All continuous-generation correctness tests passed.
- Synchronization calls fell exactly as predicted, from 496 to 31 for 16 requests and
  31 decode steps.
- The first throughput run reached only 227.2 tok/s at width 16. A repeat after the
  runtime was warm produced approximately the same result, below both the 250.3 tok/s
  acceptance floor and the 263.5 tok/s immediate baseline.
- The first `.item()` already waits for outstanding decode work. Later per-request
  synchronization API calls observe an almost idle stream, so their count overstated
  their critical-path cost. The replacement introduced a pinned D2H copy plus an
  explicit wait and did not improve end-to-end execution.

Decision: `REVERT`. Preserve this result so synchronization call count is not mistaken
for latency saved in future profiling.

### Fused Triton RMSNorm — accepted

Commit: `5282517` — `Fuse Qwen RMSNorm with Triton`

Hypothesis:

The decode-only profile reported 3,503 RMSNorm reduction chains over 31 steps: 113 norm
sites per forward. Each stock norm separately launches power, mean reduction, epsilon
addition, reciprocal square root, and multiplication operations. A single row-wise
Triton kernel should reduce GPU work and, more importantly at these small decode shapes,
remove thousands of CPU-launched CUDA operations.

Change:

- Added a row-wise Triton RMSNorm supporting both `[N, hidden_size]` hidden states and
  `[N, heads, 1, head_dim]` Q/K norm tensors.
- Accumulates variance and normalization in FP32, then follows Qwen's reference ordering
  by casting to the input dtype before applying the same-dtype weight.
- Added a narrowly scoped installer for recognized RMSNorm modules with reversible
  restoration support; the physical continuous engine enables it before warmup.
- Added numerical tests at widths 128 and 1024 and an integration test covering all 113
  RMSNorm modules in Qwen3-0.6B against their original forward implementations.
- Existing staged generation tests remain the token-exact end-to-end gate.

Gate:

- Numerical kernel/module comparisons and all continuous-generation tests must pass.
- Width-16 throughput must remain at least 250.3 tok/s (within 5% of the restored 263.5
  tok/s accepted baseline).
- If accepted, re-profile: `mean`, `pow`, and `rsqrt` counts attributable to RMSNorm
  should disappear, while the fused RMSNorm kernel should appear 3,503 times.

T4 result:

- All RMSNorm numerical/module and continuous-generation tests passed.
- Width-16 throughput was 253.1 tok/s, 4.0% below the 263.5 tok/s immediate baseline
  and above the 250.3 tok/s acceptance floor.
- Width-8 throughput was 161.1 tok/s, above the prior 142.6 tok/s run, while lower
  widths continued to show normal Colab run-to-run variance.
- The decode profile confirmed 3,503 fused RMSNorm calls and removed stock `mean`,
  `pow`, and `rsqrt` operators from the top results.
- `cudaLaunchKernel` calls fell from 40,548 before fusion to 12,524 afterward.

Decision: `KEEP`.

### Decode `aten::cat` localization — awaiting T4 evidence

Commit: `1c465fd` — `Add shape-aware continuous decode profiling`

Purpose:

The RMSNorm profile still contains 1,767 `aten::cat` calls, or 57 per decode step. This
is now a meaningful allocation/copy target, but its origin must be identified by actual
input shape before attempting a model-specific replacement.

Change:

- Added an optional shape-recording profiler mode that groups events by input shape and
  prints the `aten::cat` groups separately.

T4 result:

- The `aten::cat` tensor-list signature reported only `[[], []]`, which does not expose
  its member tensors. This is a profiler representation limitation, not evidence that
  the cat is harmless or shape-free.
- Added an opt-in stack-grouped mode to identify the source call site directly.

Commit: `6996116` — `Add stack grouped cat profiling`

Decision: `DIAGNOSTIC PENDING`; shape profiling is not a throughput measurement.

### Combined Triton RoPE and SwiGLU fusion — accepted

Commit: `35a08f6` — `Fuse Qwen RoPE and SwiGLU with Triton`

Hypothesis:

Source inspection localized the remaining repeated operations without another isolated
experiment. Qwen applies `rotate_half` separately to Q and K in every layer, and each
call constructs its result with `torch.cat`; 2 × 28 layers × 31 steps explains 1,736 of
the observed 1,767 cats exactly. Every layer also evaluates SiLU and the gate/up product
as separate elementwise operations.

Combined change:

- Added one Triton RoPE launch over batch, token, and head that rotates and writes both
  Q and K, supporting Qwen's different query and KV head counts.
- Removed the materialized `rotate_half`, `neg`, two multiplies, add, and cat chain from
  the Q/K RoPE path.
- Added a Triton SwiGLU kernel for `silu(gate) * up` and patched all 28 Qwen MLP modules.
- Kept gate/up/down projections on PyTorch's tuned CUTLASS GEMM path.
- Added separate numerical tests for decode and prefill RoPE shapes, SwiGLU shapes, and
  installer coverage, while retaining one end-to-end token-equivalence suite.

Single Colab gate:

- Run the fused-op numerical tests, RMSNorm tests, and staged continuous-generation
  tests together.
- Run one throughput sweep. Width-16 must remain at least 240.4 tok/s (within 5% of the
  immediate 253.1 tok/s RMSNorm baseline).
- Run one decode profile only after correctness. Expected changes: nearly all 1,767
  `aten::cat` calls disappear, `aten::silu` disappears, and `_rope_qk_kernel` plus
  `_swiglu_kernel` each appear 868 times.

Decision: `PENDING T4 MEASUREMENT`.

First T4 gate attempt:

- All three fused Q/K RoPE numerical cases passed.
- SwiGLU compilation failed before numerical comparison because Triton 3.6 restricts
  `tl.sigmoid` to FP32/FP64 and the kernel supplied FP16.
- The four continuous-generation failures had the identical downstream compiler error;
  no benchmark or profile ran because the combined gate stopped at correctness.
- Repair: promote gate and up values to FP32 for fused SiLU/product evaluation and let
  the output store cast back to the model's FP16 dtype.

Final T4 result:

- The fail-fast combined gate reached benchmarking, confirming fused-op, RMSNorm, and
  staged continuous-generation test processes all passed.
- Sequential throughput was 24.3 tok/s.
- Width-8 throughput was 165.4 tok/s.
- Width-16 throughput was 261.8 tok/s, up 3.4% from the immediate 253.1 tok/s RMSNorm
  run and close to the earlier 263.5 tok/s peak.
- `_rope_qk_kernel` and `_swiglu_kernel` each ran exactly 868 times (28 × 31).
- `aten::cat`, `aten::neg`, and `aten::silu` disappeared from the top GPU operators.
- The remaining 1,736 `aten::add` calls correspond to two residual additions per layer
  per step, not the eliminated RoPE chain.
- Summed launch API call counts fell materially: the main `cudaLaunchKernel` category
  dropped from 12,524 after RMSNorm to 2,108, while 6,975 launches moved through
  `cuLaunchKernelEx` for the fused/custom path.

Decision: `KEEP`. End the elementwise-fusion pass here and return to architectural work;
the next phase is true batched/chunked prefill.

### Mixed-length batched prefill — awaiting T4 gate

Commit: `b728915` — `Batch mixed-length prompt prefill`

Architecture change:

- Newly admitted requests now share one padded and attention-masked model prefill
  instead of executing one complete model forward per request.
- A new batched Triton K/V writer uses per-request sequence lengths and block tables to
  write only real prompt tokens into the authoritative shared pool; padding is never
  committed to KV storage.
- First-token logits are selected at each request's real final prompt position rather
  than the padded batch boundary.
- The existing `prefill(request)` API delegates to the batch path with width one.
- Removed the superseded single-request pool cache adapter, Triton writer, and duplicate
  tests so production retains one authoritative prefill implementation.
- Added mixed-length first-token equivalence, padding isolation, physical block mapping,
  and full generation coverage.
- Changed reversible RMSNorm/SwiGLU installers to retain unbound original functions,
  avoiding self-referential bound-method cycles that delayed GPU model reclamation in
  multi-test Colab processes.

Measurement:

- Added an alternating same-engine A/B benchmark comparing sequential width-one prefill
  against one batched prefill for the same prompts and token outputs.
- The combined Colab gate will run correctness, isolated prefill A/B, and the existing
  end-to-end continuous throughput sweep once.

Gate:

- All K/V writer, fused-kernel, and staged continuous-generation tests must pass.
- Batched and sequential prefill must produce identical first tokens.
- Batched prefill must improve the isolated width-16 prompt-token throughput.
- End-to-end width-16 throughput must remain at least 248.7 tok/s (within 5% of the
  immediate 261.8 tok/s accepted baseline).

Decision: `PENDING T4 MEASUREMENT`.

### Persistent decode metadata — accepted

Commit: `5126ade` — `Persist continuous decode metadata buffers`

Hypothesis:

The engine rebuilt input IDs, position IDs, sequence lengths, and block tables as new
GPU tensors every decode step, including individual Python-driven GPU writes for every
block-table entry. Persistent buffers should reduce allocation and small-copy overhead.

Change:

- Preallocate pinned-host staging tensors and GPU tensors for all decode metadata.
- Populate scalar metadata on the host and issue four batched asynchronous copies.
- Reuse the same storage for every decode iteration.
- Keep GPU block-table rows at their complete fixed width so the view remains
  contiguous. The K4 wrapper therefore does not materialize another contiguous table
  once per transformer layer.
- The K4 kernel math and KV pool layout remain unchanged.

Risk:

Copying the full fixed-width block-table rows can cost more than rebuilding narrow
tables for very short contexts. This is an empirical tradeoff and must pass the same
161.8 tok/s width-16 gate. A later version may use stable fixed request slots and copy
only dirty block-table rows.

T4 result:

- All affected tests passed.
- Sequential throughput was 23.2 tok/s.
- Width-8 throughput was 105.1 tok/s.
- Width-16 throughput was 172.6 tok/s.
- Width-16 improved 3.0% over the unified-scheduler run (167.5 tok/s) and 0.8%
  over the earlier block-manager run (171.2 tok/s).
- Width-16 speedup was 7.43x over the same run's sequential path.

Decision: `KEEP`.

### Request and scheduler unification — accepted

Commit: `90485ba` — `Unify request scheduling with continuous execution`

Change:

- `GenerationRequest` now carries prompt token IDs, the next decode token, generated
  output, lifecycle state, and its manager-owned KV allocation.
- The FCFS scheduler now admits directly against `KVBlockManager` and reserves prompt
  blocks lazily instead of reserving a request's maximum possible output up front.
- The physical continuous engine now submits, admits, prefills, decodes, finishes, and
  releases the same request object through that scheduler.
- Removed the duplicate `SeqState` type.
- Removed the obsolete planning-only `ContinuousBatcher` and `BatchPlan` implementation.
- Removed its duplicate control-plane test module; lifecycle and refill behavior now
  belong to the scheduler and physical-engine tests.
- The K4 Triton kernel remains unchanged.

Code-size effect: 250 lines removed and 143 lines added across the refactor, for a net
reduction of 107 lines while joining the previously disconnected layers.

T4 result:

- All runtime, scheduler, allocator, paging, K4, and full-generation tests passed.
- Width-16 throughput was 167.5 tok/s, 2.2% below the 171.2 tok/s immediate baseline
  and above the 161.8 tok/s acceptance floor.
- The width-16 speedup was 7.49x relative to that run's 22.4 tok/s sequential path.
- The batch-one result varied from 23.7 to 22.4 tok/s across separate Colab runs,
  reinforcing that small comparisons need same-session A/B measurement.

Decision: `KEEP`.

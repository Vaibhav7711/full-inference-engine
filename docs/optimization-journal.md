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

Status: `COMPLETE — KEEP`

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

### Mixed-length batched prefill — accepted

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

First batched-prefill T4 gate attempt:

- All 12 focused KV-write and fused-kernel tests passed.
- Width-one prefill and decode tests passed.
- Mixed-width prefill stopped before benchmarking because Transformers represents
  shared batch positions as broadcastable `[1,S,D]` cos/sin tables, while the fused
  RoPE wrapper accepted only materialized `[B,S,D]` tables.
- Repair: accept either leading dimension and expand `[1,S,D]` as a zero-batch-stride
  view. This preserves broadcasting without allocating or copying the tables.

Final T4 result:

- The affected fused-op and all five continuous-generation tests passed; the fail-fast
  gate also completed the isolated prefill A/B without a token mismatch.
- Sequential throughput was 24.7 tok/s.
- Width-4 throughput reached 106.2 tok/s.
- Width-8 throughput reached 214.2 tok/s, up 29.5% from 165.4 tok/s.
- Width-16 throughput reached 413.8 tok/s, up 58.1% from 261.8 tok/s.
- Width-16 end-to-end scaling reached 16.74x over the same run's sequential path.
- The gain grows with concurrency because the previous engine executed every admitted
  prompt as a separate full-model forward before entering batched decode. The new path
  amortizes model-weight reads across prompt rows and removes that serial prefill region.

Decision: `KEEP`. This is an architectural throughput improvement, not a profiler-only
or micro-kernel claim.

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

## Phase 3 — Latency-aware online scheduling

Status: `COMPLETE — KEEP`

### Chunked paged prefill and production lifecycle controls

Commit: `24e7621` — `Add chunked prefill scheduling and attention`

Problem:

- A long prompt previously ran as one indivisible forward and could delay every active
  decoder for the full prefill duration.
- Admission allocated the full prompt KV capacity immediately, reducing useful
  concurrency even though most admitted prompt tokens had not executed yet.
- The engine had no online `submit`/`step` surface, no partial-prefill lifecycle, no
  bounded waiting queue, and no graceful per-request response to KV growth failure.

Architecture:

- Requests now track committed prefill progress separately from prompt length and cannot
  enter decode until the complete prompt is present in paged KV.
- Admission allocates one initial KV block. Prefill and decode grow the physical block
  table just before the corresponding GPU work, while sequence length commits only
  after that work succeeds.
- The scheduler owns a rotating prefill queue. Each iteration plans at most one chunk
  per request under both a per-request chunk limit and a global prompt-token budget.
- Every online step runs existing decoders first, then admission, then one fair prefill
  plan. This preserves decoder priority while bounding the following scheduling gap.
- Optional waiting-queue capacity rejects overload explicitly with `QUEUE_FULL`.
- Cancellation works for waiting, partially-prefilled, and decoding requests and
  immediately returns their blocks. KV growth failure marks only the affected request
  `FAILED/KV_POOL_EXHAUSTED`; unrelated requests continue.
- Requests record queue time, TTFT, and per-token timestamps for latency analysis.

Kernel path:

- Generalized the Triton prefill K/V writer to accept a per-row absolute start position,
  so later chunks write directly into their final physical paged locations.
- Added a causal Triton paged-prefill attention kernel. Each valid query token reads its
  request's previously committed prefix plus the current chunk through the block table,
  using online FP32 softmax and GQA head mapping without materializing contiguous KV.
- Current-chunk K/V is written before attention on the same CUDA stream. Later model
  layers therefore receive correct chunk hidden states while historical tokens are not
  recomputed.
- Complete fresh short prompts retain the accepted padded SDPA prefill fast path. This
  prevents the 413.8 tok/s short-prompt workload from paying for resumability it does
  not need; only prompts exceeding the configured chunk/budget use paged chunk attention.

Correctness and measurement:

- Added physical-location tests for K/V chunks beginning inside and across page
  boundaries.
- Added a materialized causal-attention oracle covering mixed prefix lengths, mixed
  chunk lengths, GQA, and padded queries.
- Added full-model token-equivalence coverage for a prompt spanning multiple chunks and
  cancellation coverage after a partial prefill.
- Added scheduler tests for prefill transition invariants, round-robin budget fairness,
  queue backpressure, cancellation cleanup, and failure cleanup.
- Added an interleaved latency benchmark comparing one full long prefill against bounded
  chunks while a previously admitted request decodes. It reports decoder ITL p50/p95/max
  and the long request's TTFT, making the throughput/latency tradeoff explicit.
- Local CPU gate: 95 passed, 93 CUDA tests skipped; compilation and diff checks passed.
  The shared CUDA fixture now skips cleanly when CUDA is unavailable.

T4 acceptance gate:

- All focused kernel, runtime, scheduler, and continuous-generation tests pass.
- Chunked full-model generation is token-identical to stock greedy generation.
- The chunked latency run reduces the long-prefill-induced decoder ITL maximum or p95;
  long-request TTFT is reported as the expected tradeoff, not hidden.
- The established 16-request, 32-token short-prompt throughput remains at least
  393.1 tok/s (within 5% of the accepted 413.8 tok/s baseline).

First T4 gate:

- All 15 lifecycle/scheduler/allocator tests, all four focused Triton tests, and all
  seven full-model continuous/chunked tests passed.
- Width-16 short-prompt throughput was 407.4 tok/s, only 1.5% below the accepted
  413.8 tok/s baseline and above the 393.1 tok/s floor. The fast path is retained.
- The first latency output reported unchunked/chunked worst ITL of 313.92/513.15 ms.
  This cannot support the latency claim.
- Audit found the benchmark warmed only a short prompt, which selected SDPA and never
  invoked the new paged-prefill kernel. The first measured chunk therefore included
  Triton compilation even though the benchmark comment claimed both paths were warm.
- The benchmark now explicitly executes and synchronizes one real partial chunk before
  either measured scenario and prints a conditional result instead of always claiming
  that chunking helped. Only the latency benchmark must be rerun; correctness and the
  short-prompt throughput gate are already accepted.

Final warmed T4 latency result at chunk size 64 with a 1,001-token prompt:

| Mode | ITL p50 | ITL p95 | ITL max | Long-prompt TTFT |
| --- | ---: | ---: | ---: | ---: |
| Unchunked | 34.90 ms | 39.55 ms | 308.19 ms | 360.63 ms |
| Chunked (64) | 70.92 ms | 91.68 ms | 96.64 ms | 1,229.79 ms |

Interpretation:

- Chunking reduced worst-case decoder ITL by 68.6%, from 308.19 to 96.64 ms.
- It increased median ITL by 103.2% and p95 ITL by 131.8%, because a prompt chunk now
  executes between successive decode iterations instead of causing one isolated stall.
- Long-prompt TTFT increased by 241.0%, from 360.63 to 1,229.79 ms, because the prompt
  was deliberately spread over multiple scheduling iterations.
- This is latency isolation, not a throughput optimization. Chunk size remains a
  workload policy: smaller chunks favor decoder responsiveness, while larger chunks
  favor prompt TTFT and prefill efficiency.
- Alongside the already accepted 407.4 tok/s short-prompt result and complete correctness
  gate, the implementation satisfies its stated purpose without regressing the fast path.

Decision: `KEEP` chunked prefill and decode-first scheduling. Do not claim lower typical
ITL or better long-prompt TTFT; claim a measured 68.6% reduction in the worst decoder
stall for this concurrent 1,001-token-prompt workload.

## Phase 4 — Refcounted paged prefix caching

Status: `COMPLETE — KEEP`

### Block-radix reuse, copy-on-write continuation, and pressure eviction

Commit: `6428f81` — `Add refcounted paged prefix caching`

Problem:

Repeated system prompts and shared conversation prefixes were recomputed in full. The
allocator also assumed exclusive block ownership, so merely copying block-table IDs
would have caused use-after-free or allowed one request to overwrite another's KV.

Ownership architecture:

- `BlockAllocator` now refcounts physical blocks and tracks logical owners separately.
  Allocation creates a reference, prefix attachment increments it, extension adds private
  blocks, and a physical block returns to the free list only after its last owner exits.
- `KVBlockManager.attach_prefix` accepts only block-aligned immutable prefixes. Shared
  partial blocks are forbidden, avoiding copy-on-write inside a physical page.
- Cache nodes retain their own allocator reference independently of active requests.
  Finishing or cancelling a request therefore cannot invalidate a cached prefix, while
  evicting a cache node cannot invalidate blocks still referenced by active requests.

Lookup and eviction:

- Added a block-radix tree keyed by `(parent, complete token block)`. Lookup walks the
  longest matching token path and returns the existing physical block IDs directly.
- A general shared-prefix lookup leaves at least one prompt token uncached. KV alone
  does not contain final-token logits, so residual prefill is required for a prefix that
  belongs to a different prompt.
- Exact prompt entries additionally retain every prompt block and the first-token
  decision produced by that KV state. Exact hits can therefore bypass residual prefill.
  A shared partial tail is copied to a private physical block before decode writes into
  it, preserving both the cached entry and token-identical repeated generation.
- Residual tokens use the Phase 3 paged-prefill kernel with absolute positions, naturally
  continuing from shared blocks into newly allocated private blocks.
- Cache capacity is expressed in physical blocks. Eviction removes least-recently-used
  radix leaves first, preserving valid parent paths and never breaking descendants.
- Admission and KV growth ask the cache to release LRU ownership under allocator
  pressure before failing a request. If active references prevent reclamation, the
  existing per-request `KV_POOL_EXHAUSTED` behavior remains authoritative.
- Cache metrics expose lookups, request-level hits, reused tokens, hit rate, cached
  blocks, and evictions. Allocator metrics expose unique used blocks, shared blocks, and
  total logical references.

Execution integration:

- Scheduler admission attaches the longest available prefix and initializes committed
  prefill progress from the reused token count.
- Both full SDPA prefill and paged chunk completion publish complete prompt blocks before
  decode begins. Concurrent identical misses remain correct: the first publication wins
  the radix edge and duplicate request blocks are reclaimed normally on release.
- The existing short-prompt batched fast path remains unchanged. Prefix caching is
  configured by `prefix_cache_blocks` and is cleared with the engine allocator on reset.

Validation and measurement:

- Added unit coverage for reference lifetime, final-owner reclamation, block alignment,
  residual-token preservation, longest-prefix lookup, LRU leaf eviction, scheduler
  attachment, and cancellation after a cache hit.
- Added full-model repeated-prompt token-equivalence and hit-accounting coverage.
- Updated leak assertions to distinguish intentional cache residency from live request
  ownership.
- Added a same-engine TTFT benchmark. It warms the long SDPA shape, records one true
  miss, excludes the first cache lookup, then measures five steady-state exact hits
  while requiring identical greedy output tokens.
- Local gate after exact-hit repair: 102 passed, 94 CUDA tests skipped; compilation and
  diff checks passed.

T4 acceptance gate:

- All prefix ownership, scheduler, paged-kernel, and continuous-generation tests pass.
- Repeated-prompt output is token-identical and the benchmark reports nonzero reused
  tokens with stable cache residency.
- Warm cache-hit median TTFT improves over the same-engine warmed miss; report the exact
  speedup without generalizing beyond the measured shared-prefix workload.
- The established 16-request short-prompt throughput remains at least 393.1 tok/s,
  confirming that miss-only workloads retain the accepted fast path.

First T4 gate and root-cause repair:

- All 20 ownership/scheduler tests and all four residual-prefill kernel tests passed.
- Seven existing full-model tests passed. The new repeated-prompt test produced the same
  first token but diverged on later greedy tokens, so benchmarking correctly stopped.
- The original design reused complete blocks but recomputed the uncached prompt suffix
  with a different kernel and GEMM shape. Its numerically valid FP16 differences were
  sufficient to change later greedy choices for the selected prompt. Refcounting and
  physical addressing were not the failing invariants.
- The repair adds exact-prompt entries containing all prompt block IDs and the cached
  first-token decision. Exact hits now attach the original KV state without residual
  recomputation. If the last prompt block is partial, decode performs allocator-level
  copy-on-write and copies that K/V page across every layer before modifying it.
- Added direct tests for exact lookup metadata, partial-tail ownership replacement, and
  preservation of the original shared mapping. The full T4 integration test remains the
  authoritative token-equivalence gate.

Final repaired T4 gate:

- The focused exact-prefix full-model test passed, including token-identical eight-token
  greedy generation through copy-on-write decode.
- The repeated prompt contained 871 tokens and occupied 55 physical cache blocks.
- Same-engine warmed cache-miss TTFT was 343.70 ms.
- Exact cache-hit median TTFT was 3.11 ms, a measured 110.42x improvement.
- All 871 prompt tokens were reused. This exact-hit path also reuses the cached first
  token decision; it is intentionally stronger than a general shared system prefix
  followed by a new suffix, which still requires residual paged prefill.
- Width-16 miss-only throughput was 406.0 tok/s with 16.65x scaling over the same run's
  24.4 tok/s sequential path. This is 1.9% below the 413.8 tok/s pre-cache baseline and
  above the 393.1 tok/s acceptance floor.

Decision: `KEEP`. Claim 110.42x TTFT only for the measured warmed exact-prompt hit. Do
not generalize it to partial-prefix hits, cold cache operation, multi-token decode
throughput, or arbitrary production traffic.

## Phase 5 — T4 paged-decode attention regimes

Status: `COMPLETE — KEEP`

### Measured tile selection

The Phase 2 batched decode kernel originally used one fixed `BLOCK_N=64`, four-warp
configuration for every context length. A T4 calibration sweep of the live kernel at
batch widths 1, 8, and 16 found a stable regime boundary:

- Below 128 context tokens, retain `64x4`. At 64 tokens it is the lowest-latency
  conservative setting across the measured widths.
- At 128 tokens and above, use `128x4`. At 256--2048 tokens it reduced isolated
  attention-kernel median latency by roughly 20--42% versus `64x4`, depending on
  width and context length.

The scheduler selects this compile-time kernel configuration from the longest active
sequence before the model call and passes it through the decode attention context. This
keeps a mixed batch on one valid kernel variant and avoids any scalar device read or
synchronization inside the hot kernel wrapper. The public kernel wrapper retains its
old `64x4` default for direct callers.

The policy remains deliberately conservative: a `16x2` result was slightly faster for
the narrow 64-token, width-16 microbenchmark, but not across the rest of the measured
space. It is not a safe engine-wide default. Acceptance still requires CUDA correctness
for both live tile variants plus an end-to-end short-prompt regression gate.

T4 acceptance gate:

- The full CUDA paged-decode correctness suite passed, including both `64x4` and
  `128x4` tile variants, plus the continuous-batching integration suite.
- The 16-request short-prompt regression run reached 483.1 tok/s at width 16, with
  16.90x scaling over its 28.6 tok/s sequential measurement. This is above the Phase 4
  393.1 tok/s acceptance floor.

Decision: `KEEP`. Attribute the 20--42% improvement only to the measured isolated
long-context attention-kernel cases. Do not attribute the 483.1 tok/s short-prompt
result to this policy: those prompts select the unchanged `64x4` regime and Colab
end-to-end runs vary with runtime warm state.

## Phase 6 — Kernel-native INT8 paged KV

Status: `COMPLETE — KEEP AS OPT-IN MEMORY MODE`

The existing INT8 KV cache is a memory-and-quality reference only: it reconstructs the
entire cache as FP16 before attention, so it cannot be used to claim serving throughput.
Phase 6 starts with a separate decode-only Triton path. It quantizes each incoming K/V
head vector directly into INT8 paged storage with one FP16 symmetric scale, then loads
and dequantizes each vector inside the online paged-attention kernel. No full FP16 KV
materialization is permitted in this path.

The first gate covers writer agreement with the PyTorch per-vector quantization
reference and attention error against the existing FP16 paged kernel at 64, 256, and
1,024-token contexts. Integration with chunked prefill, copy-on-write prefix tails,
and the online scheduler is explicitly deferred until this isolated CUDA gate and a
long-context latency A/B measurement pass.

### Phase 6A acceptance — isolated decode kernel

The CUDA writer-reference test and the 64/256/1,024-token attention-error tests passed.
A paired, interleaved T4 benchmark was used for the latency decision; it alternates FP16
and INT8 launches every sample and reports independent round medians, preventing clock
or thermal drift from favoring either variant.

| Context | Batch | INT8 / FP16 latency | Interpretation |
| ---: | ---: | ---: | --- |
| 256 | 1 | 0.92x | Do not use INT8 for a single short stream. |
| 256 | 16 | 1.05x | Marginal; not an integration trigger. |
| 1,024 | 1 | 0.99x | Neutral. |
| 1,024 | 16 | 1.22x | Accepted long-context saturated regime. |
| 2,048 | 1 | 1.05x | Small and not a serving claim. |
| 2,048 | 16 | 1.37x | Accepted long-context saturated regime. |

Across all six cases, relative attention-output error was 0.87--0.96% and INT8 pages
plus FP16 scales reduced K/V storage by 49.2%. The 1,024-token width-16 paired-round
speedup range was 1.21--1.36x; the 2,048-token width-16 range was 1.36--1.38x.

Decision: `KEEP Phase 6A`. The result justifies an opt-in INT8 KV storage mode for
long-context saturated batches, not a global replacement for FP16. Phase 6B must add
INT8 chunk-prefill writes/reads, prefix-tail copy-on-write for both data and scales, and
an end-to-end long-context continuous-batching gate before it is exposed by the engine.

### Phase 6B acceptance — engine integration

`ContinuousBatchingEngine(kv_cache_dtype="int8")` now stores complete prefills and
resumable paged-prefill chunks as INT8 K/V plus FP16 per-vector scales, dispatches fused
INT8 paged decode attention, and copies scales with K/V bytes during prefix-tail
copy-on-write. FP16 remains the default and has no changed dispatch path.

All six isolated INT8 CUDA kernel tests passed, followed by the full-engine chunked
prefill/decode test. The end-to-end 16-request, 1,024-token-prompt, 32-token-decode A/B
reported 100% greedy token agreement between FP16 and INT8. Its end-to-end throughput
was effectively the same, despite the isolated attention gain. This is expected: at this
model size, the attention read is only one part of a complete decode step, which is also
dominated by projection/MLP GEMMs, norms, and kernel-launch overhead.

Decision: `KEEP` the INT8 mode for its 49.2% KV-storage reduction and validated
long-context kernel benefit. Do not claim an end-to-end throughput gain on this T4
workload. Use it as an explicit capacity/long-context option; keep FP16 as the default
latency-oriented mode.

## Phase 7 — Fixed-width paged CUDA-Graph decode buckets

Status: `COMPLETE — KEEP AS SATURATED-BATCH FAST PATH`

CUDA Graphs require stable tensor addresses and launch shapes, but paged continuous
decode already owns persistent device buffers for input IDs, positions, sequence lengths,
and full-width block tables. A capture therefore remains valid when the *contents* of
per-request page tables and lengths differ; only the active batch width and selected
kernel regime must stay fixed.

Implementation:

- Added a graph capture helper around the real paged decode model forward, including
  direct K/V writes and paged attention kernels.
- `ContinuousBatchingEngine(cuda_graph_batch_size=16)` captures lazily on the first
  full width-16 decode step and replays only that fixed bucket. Partial batches and all
  other widths keep the normal dynamic model-forward path.
- Capture writes the pending K/V slot once, then the immediate replay overwrites that
  same slot before normal request-state advancement. Subsequent replays use fresh device
  metadata copied by the existing scheduler path.

Validation and measurement:

- A fixed-width real-paged-forward microbenchmark measured 37.16 ms ordinary versus
  9.65 ms graph replay (3.85x), with identical logits.
- A deliberately mixed-length width-16 bucket measured 40.56 ms ordinary versus
  9.46 ms replay (4.29x), again with identical logits. Variable per-row lengths are
  graph-safe because they are device data; variable tensor addresses/shapes are not.
- The end-to-end graph-bucket token-equivalence test passed.
- On the 16-request, 32-token continuous-throughput gate, width 16 reached 962.4 tok/s
  in 0.532 s, a 32.85x speedup over the same run's 29.3 tok/s sequential path.

Decision: `KEEP`. Claim this result only for a stable saturated width-16 decode bucket
on the measured T4 setup. Do not generalize it to partial batches, changing batch widths,
capture construction cost, or arbitrary arrival/departure patterns; those continue on
the ordinary dynamic path.

## Phase 8 — Padded power-of-two CUDA-Graph buckets

Status: `COMPLETE — KEEP`

Phase 7 replayed only exact-width batches. Phase 8 adds safe padding so a live occupancy
can use the smallest configured graph bucket that contains it, for example three real
requests in a width-four graph. The engine permanently reserves one physical KV block
per potential dummy row under a non-customer allocator owner. Padded rows use those
private blocks at length zero, so their K/V writes cannot alias a real request or a
prefix-cache page. Only logits for real rows are sampled and advanced.

Validation and measurement:

- The CUDA correctness test compared three real requests replayed in a width-four graph
  against ordinary greedy generation and passed token-identically.
- The end-to-end occupancy A/B warmed/captured each bucket before measurement and failed
  on any output difference. All rows were token-identical.

| Live requests | Ordinary tok/s | Padded graph tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 30.0 | 103.9 | 3.46x |
| 2 | 62.0 | 252.6 | 4.08x |
| 3 | 92.5 | 376.3 | 4.07x |
| 4 | 122.3 | 504.8 | 4.13x |
| 8 | 243.5 | 966.9 | 3.97x |
| 16 | 480.2 | 1,627.4 | 3.39x |

Decision: `KEEP`. Configured buckets now provide flexible graph replay across the tested
occupancies while preserving safety and token equivalence. Capture construction remains
excluded from these steady-state measurements, and capacity consumed by dummy blocks is
an explicit small reservation. Test width 32 separately before enabling a 32-row bucket
on the constrained T4 runtime.

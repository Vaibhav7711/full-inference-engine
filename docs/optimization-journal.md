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
an explicit small reservation. Width 32 is registered as an optional benchmark bucket;
it remains unvalidated on the constrained T4 runtime until separately measured with
`max_active=32` and enough KV blocks.

## Phase 9 — Fused Qwen MLP gate/up projection experiment

Status: `COMPLETE — REJECT AS DEFAULT; RETAIN AS EXPERIMENT`

Each Qwen MLP normally launches separate gate and up FP16 projections before SwiGLU.
The experiment concatenated their output weights into one projection, split the result,
then used the existing fused Triton SwiGLU kernel. CUDA correctness passed and the
fusion remains available as an explicit experiment.

One width-16 A/B run measured separate projections at 961.5 tok/s and the fused path at
915.5 tok/s. A later paired, interleaved seven-round graph-enabled A/B warmed and
captured both variants before timing. It reported median end-to-end times of 318.079 ms
unfused and 310.904 ms fused (1.023x), but the individual-round range was 0.957--1.053x
and five of seven rounds were neutral or slower for fusion. Greedy output tokens were
identical throughout.

Decision: the 2.3% median difference is below the observed Colab runtime variation, so
fusion is not a defensible default. Keep the existing Triton SwiGLU elementwise fusion;
keep gate/up projection fusion available only as an explicit experiment. Separate
PyTorch/CUTLASS projections are the engine default.

## Phase 10 — Weight-only and W8A8 decode-linear experiments

Status: `COMPLETE — REJECT FOR T4 DECODE; DO NOT INTEGRATE`

This phase tested whether reducing linear-layer weight bandwidth could improve the
Qwen3-0.6B decode path. The result is negative for this model, GPU, and batch regime.

First, the fused-scale W8A16 Triton linear kernel was compared against CUDA FP16 GEMM:

| Shape | Batch | FP16 ms | W8A16 ms | FP16 / W8A16 |
| --- | ---: | ---: | ---: | ---: |
| attention 1024 | 1 | 0.0410 | 0.2252 | 0.18x |
| attention 1024 | 16 | 0.0432 | 0.2228 | 0.19x |
| MLP 3072 | 1 | 0.0618 | 0.5025 | 0.12x |
| MLP 3072 | 16 | 0.0712 | 0.5059 | 0.14x |

W8A16 dequantizes values in the custom kernel before a floating-point dot product. It
does not use the T4's INT8 tensor cores, while the FP16 baseline dispatches optimized
CUTLASS tensor-core GEMMs. It is therefore structurally the wrong optimization here.

Second, a true W8A8 path was attempted. Triton 3.x could not lower signed INT8 MMA for
the Colab T4's `sm75` target, so the final experiment used CUDA's native INT8xINT8 to
INT32 GEMM through `torch._int_mm`. It prepacked the transposed INT8 weight once, then
performed dynamic per-row activation quantization, the CUDA INT8 GEMM, and scale
dequantization. For decode widths at or below 16, PyTorch's `torch._int_mm` wrapper
also requires padding to 17 internal rows; that cost was included in the measurement.

| Shape | Batch | FP16 ms | W8A8 ms | FP16 / W8A8 | Relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention 1024 | 1 | 0.0395 | 0.3463 | 0.11x | 1.22% |
| attention 1024 | 16 | 0.0466 | 0.3886 | 0.12x | 1.11% |
| MLP 3072 | 1 | 0.0589 | 0.3762 | 0.16x | 1.11% |
| MLP 3072 | 16 | 0.0676 | 0.3858 | 0.18x | 1.11% |

The first W8A8 implementation incorrectly repacked weights per invocation; correcting
that reduced latency from 0.47--0.67 ms to 0.35--0.39 ms, but did not change the
decision. The remaining cost is activation reduction/quantization, temporary INT8
materialization, short-batch padding, dequantization, and extra launches. At these
small decode GEMMs, optimized resident FP16 weights are substantially faster.

Decision: retain the code only as an isolated benchmark/reference, with no model or
engine integration. FP16 stays the decode-linear default. INT8 KV remains separately
justified as an opt-in long-context capacity mode; it should not be conflated with
weight quantization.

## Phase 11 — Mixed-arrival scheduler prefill budget

Status: `COMPLETE — KEEP 128-TOKEN DEFAULT`

Earlier scheduler work established a decode-first loop with chunked prefill, but its
default prefill iteration budget was 512 tokens. The mixed-arrival workload introduces
short, medium, and long requests over scheduler iterations and measures queue time,
TTFT, and ITL under the real graph-bucket path. Its initial 256-token run showed typical
ITL near 10 ms but approximately 125 ms p95 stalls while prefill work shares the loop.

An interleaved five-round sweep kept the 64-token chunk size fixed and rotated the
64/128/256-token budget order every round. Reported values are per-policy medians:

| Prefill budget | Throughput | Short p95 ITL | Medium p95 ITL | Long p50 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 295.2 tok/s | 50.44 ms | 45.78 ms | 898.13 ms |
| 128 | 430.0 tok/s | 63.39 ms | 63.39 ms | 530.70 ms |
| 256 | 424.0 tok/s | 62.59 ms | 62.56 ms | 543.79 ms |

Decision: set `max_prefill_tokens_per_iteration=128` as the engine default. It is the
highest-throughput and lowest-long-TTFT setting in the interleaved workload. The roughly
0.8 ms p95 ITL difference versus 256 is not enough to offset 256's lower throughput and
higher long TTFT. A 64-token budget is available for an explicit latency-priority mode,
but is not a reasonable general default because it increases long TTFT by about 69%.

## Phase 12 — Mixed-arrival padded CUDA-Graph serving path

Status: `COMPLETE — KEEP AS RECOMMENDED SERVER CONFIGURATION`

The mixed-arrival workload compared ordinary dynamic decode against padded graph buckets
`(2, 4, 8, 16)` while using the selected 128-token prefill budget. Both paths generated
identical greedy token sequences.

| Path | Throughput | Elapsed | Relative throughput |
| --- | ---: | ---: | ---: |
| Ordinary | 150.2 tok/s | 2.557 s | 1.00x |
| Padded graphs | 279.6 tok/s | 1.374 s | 1.86x |

Graph replay improved medium p95 TTFT from 208.71 to 107.36 ms and long p95 TTFT from
913.46 to 817.44 ms. It also improved long p95 ITL from 100.68 to 89.06 ms. Short p95
ITL regressed from 123.25 to 156.91 ms, a known interaction between padded graph work
and the shared prefill/decode scheduling loop.

Decision: recommend graph buckets `(2, 4, 8, 16)` for throughput-oriented serving, but
do not make graphs an unconditional base-engine default. The service layer enables them
by default; callers requiring the best short-request tail latency can explicitly choose
the dynamic path.

## Phase 13 — HTTP API integration with continuous batching

Status: `COMPLETE — KEEP`

The previous FastAPI surface called `ExplicitDecodeRunner` directly, which serialized
each HTTP request and bypassed the scheduler, paged KV, prefix cache, and graph work.
The new service layer introduces a single GPU-owning worker thread with a bounded ingress
queue. HTTP and SSE handlers submit tokenized requests and only observe lifecycle state;
the worker alone drains submissions, calls continuous scheduler steps, and publishes
completion. This removes concurrent handler access to mutable GPU/scheduler state.

The CPU lifecycle test passes. The CUDA FastAPI smoke test loaded one service instance,
completed four concurrent `/generate` requests through its shared worker, and then
verified that `/generate/stream` emitted the identical greedy token sequence for the
same request. The service enables recommended graph buckets `(2, 4, 8, 16)` by default.

Decision: `KEEP`. The serving API now exercises the production continuous engine rather
than the obsolete one-request reference runner. Request cancellation on disconnect,
timeout policy, and load-generator percentile testing remain the next service hardening
work; they are not required to claim continuous HTTP batching correctness.

## Phase 14 — Server cancellation and burst-load observability

Status: `COMPLETE — KEEP CANCELLATION; THROUGHPUT BASELINE ONLY`

The HTTP worker owns all GPU and scheduler state, so an async SSE handler must never
call `engine.cancel()` directly on client disconnect. The service now routes cancellation
through a control queue consumed by that same worker. It handles both requests already
admitted to the scheduler and requests still waiting in the bounded ingress queue. The
worker alone performs the state transition and KV release, then signals the caller's
completion handle. A CPU lifecycle test verifies this ownership boundary.

The phase also adds a concurrent SSE burst-load benchmark. It warms the service, starts
a configurable burst of streaming requests at once, and records aggregate output
throughput plus client-observed TTFT and completion p50/p95. This is intentionally a
service-level measure: it includes ingress, token streaming, scheduler queueing, graph
bucket selection, and decode work rather than only model-forward time.

Acceptance results:

- The worker-routed cancellation lifecycle test passed (2 tests).
- A 16-request, 32-token CUDA burst completed all 512 requested tokens without server
  errors at 944.2 tok/s.
- The FastAPI `TestClient` first-event and completion p50/p95 values were both roughly
  529/540 ms. This equality exposes a measurement limitation: TestClient buffers SSE
  bodies, so those values are **not** wire-level TTFT and must not guide a latency,
  timeout, or ingress policy.

Decision: `KEEP` worker-routed cancellation and retain 944.2 tok/s as a same-process
service-throughput baseline. Relabel the benchmark's first-event field accordingly. A
real `uvicorn` plus network HTTP client load harness is required for valid streamed TTFT
percentiles; defer that transport-level measurement rather than publishing false TTFT.

## Post-Phase 14 audit — prioritized remaining work

This audit distinguishes a measured T4 bottleneck from production-runtime gaps. Items
are ordered by value and risk; rejected W8A16/W8A8 and inconclusive MLP-projection fusion
are intentionally excluded from the active backlog.

| Priority | Work item | Why it remains | Acceptance evidence required |
| --- | --- | --- | --- |
| P0 | Batched decode-token transfer | Decode state advancement reads `next_tokens[i].item()` once per active row. Replace it with one batched host transfer and verify token identity. | CUDA profile: fewer DtoH transfers; no end-to-end regression. |
| P0 | Real transport load harness | TestClient buffers SSE, so current TTFT is invalid as a network metric. | `uvicorn` + real HTTP client burst/steady tests with valid TTFT p50/p95/p99. |
| P1 | Admission limits, timeout, and disconnect tests | Cancellation is worker-routed, but the server has no request deadline, prompt-token limit, or real disconnect test. | KV release, no leaked handles, correct 429/timeout behavior. |
| P1 | Adaptive graph QoS policy | Graph buckets give 1.86x mixed-arrival throughput but regress short p95 ITL. | Policy A/B improves selected latency objective without sacrificing token identity. |
| P1 | Long-prompt prefill coalescing | Current 128-token budget is a good default on one synthetic workload, not a universal policy. | Bursty long-prompt workload with throughput/TTFT/ITL improvement. |
| P2 | Remove process-global attention contexts | `_BATCH_CTX` and `_PREFILL_CTX` make one process safe only for one active engine forward. The server enforces this deliberately. | Two independent engines/models cannot cross-contaminate state. |
| P2 | GPU-resident sampling/request state | A deeper follow-on to batched transfer; removes most host decode control but requires redesign. | Profile-backed gain that exceeds added complexity. |
| P2 | Paged-attention long-context retuning | Existing 64x4/128x4 regimes are measured; further tuning needs a new long-context profiler signal. | Isolated + end-to-end long-context A/B. |
| P3 | Multi-model/model-family and distributed support | Current patches are Qwen/Transformers-version specific and single-GPU by design. | Adapter contract and model-specific correctness suite. |

Documentation is also stale in places: the README still describes several pre-paged,
pre-continuous milestones. Reconcile it with the journal after the P0/P1 runtime work so
the public architecture description does not understate or contradict the implemented
engine.

## Phase 15 — Final P0/P1 optimization closure

Status: `COMPLETE — KEEP; OPTIMIZATION FREEZE`

This is the final optimization phase before the project switches to broad correctness,
stress, and failure testing. Its scope is deliberately limited to both P0 audit items
and the first P1 item.

Decode state advancement now transfers the complete argmax token vector to the host with
one `.tolist()` operation. The previous loop called `.item()` once per active row,
creating one Python-visible CUDA synchronization per request. Request lifecycle updates
remain on the CPU, so this removes repeated scalar transfers without redesigning the
scheduler. Acceptance requires token-identical CUDA correctness plus the isolated
width-1/2/4/8/16 transfer A/B.

The FastAPI server now enforces three bounded-admission controls: a finite ingress and
scheduler waiting limit, a maximum prompt-token count, and a request deadline. Complete
and streaming timeouts enqueue cancellation through the worker. SSE generation uses a
`finally` guard because an ASGI server may cancel the response generator immediately on
socket loss before `Request.is_disconnected()` returns true.

A real transport harness launches `uvicorn`, opens loopback HTTP connections, reads SSE
incrementally from the socket, and reports TTFT and completion p50/p95/p99. It also drops
one stream after its first token and requires the worker's cancellation counter to
advance before running the measured burst. Unlike TestClient, these TTFT values are
wire-visible and are valid for local transport comparisons.

Acceptance gates:

- server lifecycle, cancellation, and queue-backpressure tests pass;
- continuous-batching CUDA correctness remains token-identical;
- batched token transfer reduces width-16 host-transfer latency;
- real `uvicorn` disconnect cancellation passes; and
- all 16 burst requests return their requested token count with valid wire-level TTFT.

Acceptance results on the Colab T4:

- All three server lifecycle/cancellation/backpressure tests passed.
- All 11 integrated continuous-batching CUDA tests passed, covering prefill, batched
  decode, mixed/staggered generation, FP16 and INT8 KV, exact and padded graphs,
  chunked-prefill cancellation, block release, and prefix reuse.
- Batched host transfer remained neutral at width 1 and improved monotonically with
  width: 1.55x at 2, 2.72x at 4, 5.04x at 8, and 9.35x at 16. Width-16 latency fell
  from 0.20600 ms for repeated scalar reads to 0.02203 ms for one batched transfer.
- The real loopback `uvicorn` burst completed 512 tokens across 16 requests at
  666.1 tok/s. Wire-level TTFT was 210.44/270.15/277.91 ms p50/p95/p99; completion
  was 754.62/759.95/762.41 ms p50/p95/p99.
- Deliberately closing an SSE socket after its first token advanced the worker's
  cancellation counter, proving disconnect cleanup reached the scheduler owner.

Decision: `KEEP` every Phase 15 change. Freeze performance optimization at this commit.
Subsequent work moves to comprehensive correctness, stress, failure-recovery, and API
contract testing. Performance changes after this point require a failing test or a new
profile-backed bottleneck, not speculative tuning.

## Gate 1 — Survivability (delivery plan, gate 1 of 7)

Status: `COMPLETE — KEEP; FP16 SERVING PATH ACCEPTED`

Goal: the server survives its first bad request. Request-level failures end in a
terminal state with the right HTTP status; only engine-level failures stop the worker,
and they do so observably.

Changes:

- KV exhaustion no longer fails requests. `ContinuousBatchingEngine._acquire_capacity`
  preempts the newest active request (FCFS priority by arrival) and retries; the victim
  keeps its generated tokens, returns to the queue head, and is recomputed later
  (`GenerationRequest.preempt`, `FCFSScheduler.preempt`). A request is failed with
  `KV_POOL_EXHAUSTED` only when it is the sole active request or has yielded
  `MAX_PREEMPTIONS_PER_REQUEST` times. A request whose declared prompt plus maximum
  generation length cannot fit an otherwise empty pool is instead rejected at admission
  with `KV_CAPACITY_EXCEEDED`.
- Resumed requests prefill `prompt + generated[:-1]`, re-attach any published prefix
  blocks, and continue from the pending token; the recompute's own prediction is
  discarded (`_complete_prefill`).
- Prefix eviction under pressure (`PrefixCache.evict_until_free`) skips entries whose
  blocks are pinned by active requests instead of wiping the cache for no gain.
- Both KV write kernels bounds-check the block-table lookup and the destination block;
  `_prepare_decode_metadata` asserts host-side that every row has a slot for its token.
- Service: `alive` (liveness) vs `ready` (readiness), typed `SubmitError` with a status
  hint, `drain()` on shutdown, every handle completed on stop, admission event so
  streaming responses can send real status codes, asyncio completion callbacks instead
  of one executor thread per waiting request.
- API: `/health` returns 503 when the worker is dead; new `/ready`; 400 for empty or
  over-context prompts, 413 for oversized/unfittable, 429 for full queues, 503 for
  pool drops and shutdown, 504 for deadlines; `x-request-id` on every response;
  tokenization off the event loop; `engine_factory` injection for GPU-free tests.
- Tests: `tests/server/test_chaos.py` (12 cases, CPU, scripted engine incl. a real ASGI
  disconnect), preemption tests for request/scheduler/prefix layers, and D6 engine tests
  (`test_d6_*`) that force pool pressure on the T4 and require token-identical output.
- Reference retired: `_reference_greedy` now runs a second, unpatched checkpoint under
  `stock_rope()`; the engine's Triton RMSNorm/SwiGLU/RoPE are no longer on both sides
  of the token-identical comparison. The global RoPE patch also delegates to the stock
  implementation for non-CUDA tensors.

GPU gate to run (Kaggle T4x2 or Colab T4):

    python -m pytest tests/batching/test_continuous_batching.py -v -m cuda
    python -m pytest tests/kernels/test_kv_write.py tests/kernels/test_paged_decode_batched.py -v
    python -m pytest tests/server tests/scheduler tests/runtime tests/cache -q

Known risk to watch in D6: recompute preemption rebuilds KV with batched prefill rather
than the original incremental decode; fp16 rounding could in principle flip a greedy
token. D3/D4 already require prefill-built KV to match the stock reference, so a flip
here is a real finding, not noise — record it if it happens.

Not in this gate: INT8 KV write kernels are not bounds-checked yet; OpenAI-compatible
routes, detokenizer, and sampling are Gate 3.

Acceptance results:

- Local CPU reliability gate: 100 passed, 8 CUDA-marked tests skipped; full local suite:
  126 passed, 116 skipped.
- On the Colab T4, all three D6 pressure tests passed in 31.22 seconds. The engine
  preempted and recomputed under a ten-block pool without changing greedy tokens,
  rejected an individually impossible request before admission, and reattached a
  published prefix while resuming.
- The first D6 GPU run exposed an actual ordering bug: resumption cleared its expanded
  prefill accounting before the guarded `PREFILLING -> DECODING` transition. The fix
  keeps `resuming=True` through that transition, then calls
  `GenerationRequest.complete_resumption()` to restore ordinary prompt accounting. A
  CPU regression test now covers the invariant.

Decision: `KEEP` the FP16 Gate 1 survivability path, including bounded recompute
preemption. Maintain the explicit boundary that INT8 K/V writer bounds checks are not yet
part of this gate.

## Gate 1B — Recompute preemption: fairness policy, limits, and cost

Status: `ACCEPTED at 0b20313` — 5/5 Gate 1B interaction tests and 19/19 CUDA tests
pass on Colab T4 (Qwen3-0.6B, FP16), plus 19/19 kernel tests and the CPU policy suite.

Gate 1 proved recompute preemption is *correct* (token-identical under pressure). Gate 1B
defines the policy it runs under, removes the arbitrary limit, and makes its cost visible.

### Policy: strict-LIFO recompute with progress-gated readmission

Stated in full on `FCFSScheduler`. Six clauses, each pinned by a test:

1. Victims are chosen strictly by arrival, newest first. The oldest active request is
   never displaced, so it always completes and always releases blocks.
2. A request that needs capacity and is itself the newest active request yields itself.
   If it is the *only* active request, nothing can free memory for it and it fails with
   `KV_POOL_EXHAUSTED` rather than looping.
3. A yielded request is not re-admitted until `progress_epoch` advances past the epoch at
   which it yielded. The epoch advances only on a terminal state (finish/fail/cancel),
   i.e. when blocks are permanently released. The gate is skipped when nothing is active,
   so the GPU never idles behind it.
4. `MAX_PREEMPTIONS_PER_REQUEST` is **removed**. A count was the wrong criterion: a
   healthy request queued behind several long generations yields once per iteration
   through no fault of its own, so any fixed bound (8, then 64) eventually fails valid
   work under load. Termination is now structural — clauses 1-3 guarantee the oldest
   request completes, every completion advances the epoch, and every yield strictly
   shrinks the active set.
5. Admission checks *effective* capacity: pool minus permanently reserved blocks. CUDA
   graph dummy rows are reserved out of the pool, so counting them made admission
   optimistic and pushed a solvable rejection into a late `KV_POOL_EXHAUSTED` failure.
   A request that passes admission can now be served once it is alone.
6. A yielded request re-enters the queue in arrival order, not at the head, so it cannot
   overtake an older request still waiting for its first admission.

The previous readmission behaviour — requeue at head, retry immediately — spent a full
recompute prefill per retry with no memory freed in between. Clause 3 makes each retry
follow an actual release.

### Cost accounting (measure, don't just survive)

Per request: `preempted_count`, `recomputed_token_count`, `recompute_ms`,
`preempted_wait_ms`, and `recompute_tokens_per_output_token` via
`GenerationRequest.recompute_overhead()`. Recompute tokens are counted on every replayed
prefill token, including a request preempted before it emitted anything.

Engine-wide: `ContinuousBatchingEngine.recompute_report()` sums banked (terminal) and
in-flight cost, so a soak can sample it at any moment. Scheduler snapshot adds
`progress_epoch`, `effective_capacity_tokens`, `recomputed_tokens_total`,
`recompute_ms_total`.

### Subsystem interaction tests (each isolated)

- `test_g1b_mixed_length_pressure_stays_token_identical` — uneven prompts.
- `test_g1b_preemption_under_cuda_graphs_stays_token_identical` — yielding changes the
  live row count, so replay must cross graph buckets; also asserts clause 5's arithmetic.
- `test_g1b_int8_kv_preemption_matches_int8_without_preemption` — INT8 rebuilds per-block
  scales on recompute. INT8 is not bit-identical to fp16, so the reference is the same
  engine with a pool large enough that nothing yields.
- `test_g1b_cancelling_a_yielded_request_releases_everything` — cancellation reaches a
  request parked in the queue; no customer page survives.
- `test_g1b_recompute_cost_is_measured_not_just_survived` — prints the cost report.
- D6-3 strengthened: prefix-cache reuse between serialized requests no longer satisfies
  it; a request that actually yielded must have come back through the prefix path.

### Known limitation, deliberately not fixed here

Preempting a partially-prefilled request discards its completed chunks: `preempt()`
resets `prefilled_token_count` to zero. Retaining partial prefill needs the blocks it
already filled, which is exactly what the yield is releasing. Revisit only if soak data
shows chunked prefills being yielded often.


### Gate 1B GPU gate, first run: three pressure tests never created pressure

`test_g1b_preemption_under_cuda_graphs_stays_token_identical` and
`test_g1b_cancelling_a_yielded_request_releases_everything` passed, which is the result
that matters: yielding across CUDA-graph buckets and cancelling a request parked in the
queue both work. The other three failed on `preemption_count == 0` — the pool was sized
by checking that each request fits *alone*, never that the four together exceed it. At 16
tokens per block the four prompts at `max_new=32` need 3 blocks each, exactly the 12-block
pool, so nothing ever yielded. The graph test passed only because its 3 reserved dummy
rows left 11 usable; the cancellation test only because `max_new=64` needed 20 blocks.

Fixed by deriving the pool from the real tokenization (`_pressure_blocks`) rather than
hard-coding it: 60% of the blocks all requests need at peak, floored at the largest single
request so admission cannot reject it, with an assertion that the workload is squeezable
at all. A hard-coded pool that merely looks tight silently stops testing anything the next
time a prompt or the tokenizer changes, which is precisely what happened here.


### Gate 1B measured recompute cost (Colab T4, Qwen3-0.6B FP16)

Workload: four prompts of 5-14 tokens, `max_new=32`, `block_size=16`, `max_active=4`, pool
squeezed to 7 pages against the 12 needed at peak (58%). Four requests, two yields.

| request | preempted | tokens rebuilt | recompute ms | queued after yield ms | rebuilt/output |
|---|---|---|---|---|---|
| d6-0 | 0 | 0 | 0 | 0 | 0 |
| d6-1 | 0 | 0 | 0 | 0 | 0 |
| d6-2 | 1 | 32 | 37.53 | 130.75 | 1.00 |
| d6-3 | 1 | 22 | 37.53 | 663.99 | 0.69 |

Engine totals: `preemptions=2, progress_epoch=4, recomputed_tokens=54, recompute_ms=75.06`.
Per-request sums reconcile exactly with the engine report (32+22=54, 37.53+37.53=75.06).

**Queue wait dominates rebuild cost by 3.5x to 17x.** The expensive part of a preemption is
not rebuilding KV, it is waiting for the progress epoch to advance: 131 ms and 664 ms of
queueing against 37.5 ms of recompute. That is the epoch gate working as designed. Under
the previous requeue-at-head behaviour those same waits would have been filled with repeated
rebuild attempts at ~37.5 ms each, so the gate converted roughly 17 wasted rebuilds on d6-3
into idle queue time. It trades GPU waste for latency, deliberately, which is why per-request
deadlines (Gate 3) must treat preemption as a latency event.

**Rebuild time is independent of token count.** 32 tokens and 22 tokens both took 37.53 ms -
0.003 ms apart. Prefill at this size is bound by per-call fixed overhead, not by the tokens
in it, and 37.5 ms sits right next to the ~39 ms flat decode step measured earlier. Same
overhead, same cause. It also means the rebuilt-tokens-per-output-token ratio (1.00 and 0.69
here) understates nothing at short prompts but will dominate at long ones, where rebuild
becomes genuinely compute-bound. That is the condition under which copy-out preemption
(rejected in DD-028) should be reconsidered.

**`recompute_ms` includes first-use Triton JIT and must be read as steady-state only.** The
same workload reported 814.96 ms in the first test of the session and 75.06 ms in the last,
a 10.9x difference with identical code and configuration; the INT8 engine reported 1464.86 ms
on its first pressured run because its own kernels compile separately. Only the steady-state
numbers above are meaningful. This is direct evidence for the Gate 5 requirement to warm every
Triton variant and capture every graph bucket before a server reports ready - otherwise the
first real request pays a full compile.

**Determinism check.** The FP16 and CUDA-graph runs report identical `preemptions=2` and
`recomputed_tokens=54` despite different pool sizes (7 pages, and 10 pages of which 3 are
graph dummy rows), because clause 5 makes both 7 *usable* pages. Equal effective capacity
produced an identical schedule.

### Gate 1B GPU gate, second run: the patch had not been pushed

The full `-m cuda` suite was re-run before the pressure-sizing fix reached `origin/main`, so
the same three tests failed identically on the old tree. The traceback was the giveaway: it
showed the hard-coded `num_blocks=12` that the fix replaces. Worth recording because the
run was otherwise informative - 16/19 passing, including D6-3 with the strengthened
assertion that a request which actually yielded came back through the prefix path.
Verification step added to the workflow: `grep -c "_pressure_blocks"` before spending GPU time.


### Metrics accounting fix before the soak harness

Gate 1B's own counters were verified by the T4 run - per-request sums reconciled exactly
with the engine report. The pre-existing latency metrics were not, and Gate 1 had broken
them: `admitted_ns` now records only the first admission, so `queue_ms` stopped counting a
preempted request's second wait, while `generation_ms` silently absorbed it. On the measured
run d6-3's 664 ms park was missing from one number and buried inside the other, with no way
to separate them.

Fixed before the soak rather than after, because the soak's whole output is percentiles over
these fields. Requests now report `queue_ms` and `total_queue_ms`, `generation_ms` and
`decode_ms`, plus `stall_ms` and a mean ITL that excludes the preemption gap. Engine stats
(`kv_utilization`, queue depth, preemptions, progress epoch, prefix hit counts) are published
by the worker every 100 ms and served from `/health` and `/ready`.

Still unmeasured, deliberately: end-to-end latency under load. That is the soak harness's job
and should not be built twice. Also outstanding is the decode-step re-baseline against the
~39 ms flat measurement, which now has two named suspects rather than a general suspicion -
Gate 1B added a per-step `sorted(active, ...)` to `decode_step`, and Gate 1 added a per-row
capacity check to metadata staging plus two mask computations in each write kernel. All are
small, but the step is host-overhead-bound, which is exactly where small per-step Python
costs show up.


## Item 1 — Mixed-arrival reliability soak

Status: `CPU HARNESS VERIFIED — PENDING GPU GATE`

`benchmarks/reliability/soak.py` drives the engine with Poisson arrivals, prompts built
from a small pool of shared prefixes plus random tails (so the prefix cache is genuinely
used and genuinely evicted - hit rates of 0.35 to 0.83 across configurations), and
cancellations aimed at whichever lifecycle state has not been covered yet. It ends with an
accounting audit rather than a completion check.

The audit earned its place on the first run by finding a real defect: `finish`, `fail` and
`cancel` released a request's KV pages but left `request.allocation` populated. `preempt`
had always cleared it; the terminal paths had not. Nothing read a terminal request's
allocation today, so no test caught it, but `block_table` on a finished request returned
block ids that had already been reissued. Fixed by clearing the handle on any terminal
transition.

Two further findings were the audit being wrong rather than the engine:

- **Epoch accounting.** The first invariant counted admitted terminals, which is one too
  many: a request cancelled while parked after a yield was admitted, but `preempt` had
  already returned its pages. Advancing the epoch for it would release parked peers on an
  event that freed no memory. The scheduler now marks `held_pages_at_exit` where the epoch
  advances, and the invariant counts page-releasing exits.
- **Coverage over unreachable states.** `PREFILLING` is only observable when a prompt
  exceeds the prefill token budget, and `PREEMPTED` only under pressure. The soak now
  records observed states and demands a cancellation only from states actually entered.

Three CPU configurations (roomy, pressure, oversized) run clean against a fake engine that
uses the real scheduler, block manager and prefix cache. Peak KV utilisation reaches 1.0
under pressure, the oversized configuration rejects 286 of 294 requests at admission with
`KV_CAPACITY_EXCEEDED` and zero `KV_POOL_EXHAUSTED`, confirming clause 5 rejects before any
work rather than failing after it.

`tests/reliability/test_soak.py` holds six seeded short soaks for the T4, including the
long-generation configuration that the removed preemption-count limit would have failed,
and one test that deliberately strands an allocation to prove the audit can fail.


### Item 1 GPU gate: invariants held, first conclusions retracted

Six soaks pass on Colab T4, zero invariant violations, with peak KV utilisation reaching
1.00 in four of five configurations. The accounting identity - every page in use belongs to
an engine reservation or the prefix cache - held under real pressure, real cancellation and
real preemption. That is item 1's pass criterion and it is met.

Three conclusions drawn from that first run are **withdrawn**, for reasons that are mostly
not about sample size:

- **"CUDA graphs cut inter-token latency 2.2-4.6x."** Unsupported. The graph-enabled
  configurations also differ in `max_active` (4 vs 8), pool size, and - decisively - in what
  survived: the `oversized` arm rejected or cancelled 112 of 114 requests, so its sequences
  were short and its attention read less KV per token. The 7.6 ms there is partly short
  sequences, not graphs. Three variables moved at once.
- **"Recompute costs 2.4-3.4 rebuilt tokens per delivered token."** The denominator was
  modelled, not measured: finished requests multiplied by a guessed average `max_new_tokens`,
  plus an assumed half-budget for cancelled ones. The engine knows the real figure and was
  never asked. `SoakResult.delivered_tokens` now sums actual output lengths.
- **Single run per configuration on shared hardware.** Repeating one fixed configuration
  four times shows ITL p50 varying by 11% while the waste ratio varies by 220% (0.88 to
  5.20, a 6x swing on seed alone). One preemption of one long prompt dominates that ratio.
  Latency and waste need entirely different sample sizes before either supports a claim.

Two real gaps the run exposed:

- **Open loop above service rate measures the queue, not the engine.** Arrivals at 25-30/s
  against an engine retiring roughly 4-15/s produced TTFT medians of 2.6-25.6 s, essentially
  all of it queueing (`total_queue_p50` tracks `ttft_p50` almost exactly). It also distorted
  everything downstream: the cancellation injector hit `WAITING` 170 times out of 204 in the
  pressure arm, so that soak largely measured cancelling queued requests. `SoakConfig`
  now has a `concurrency` setting for closed-loop operation, which is what any run whose
  numbers will be compared against another engine must use.
- **Backpressure was never exercised.** Every soak engine left `max_waiting_requests`
  unset, so the queue was unbounded and `QUEUE_FULL` never fired once. Now covered by
  `test_soak_bounded_queue_applies_backpressure_instead_of_growing_without_limit`.

`benchmarks/reliability/ab.py` replaces the accidental comparison with a controlled one:
one setting varied, pool, concurrency, workload, seeds and model object held identical,
each arm repeated, and a verdict that returns "unresolved" whenever the median change is
within run-to-run spread. The CUDA-graph question is worth answering properly - it is just
not answered yet.


### First supported performance claim: CUDA graphs cut median ITL 2.8x

`benchmarks/reliability/ab.py --setting cuda_graphs --repeats 5 --duration 8 --concurrency 8`,
Colab T4, Qwen3-0.6B FP16, closed-loop at concurrency 8, `num_blocks=256`, `max_active=8`,
`max_waiting_requests=64`, `cancel_probability=0.02`. Only `cuda_graph_batch_sizes` differs
between arms; pool, workload, seeds and model object are identical.

| metric | graphs off | graphs on | change | run-to-run spread | verdict |
|---|---|---|---|---|---|
| ITL p50 | 43.71 ms | 15.50 ms | -64.5% | 12.7% | real, 5x the noise |
| TTFT p50 | 70.70 ms | 43.62 ms | -38.3% | 10.4% | real |
| ITL p99 | - | - | -32.4% | 89.8% | **unresolved** |

Supported claim: CUDA graphs cut *median* inter-token latency by 2.8x. The earlier
"2.2-4.6x" is withdrawn - it compared configurations differing in three settings at once.
The p99 effect is genuinely unresolved at five repeats; tail latency needs more samples or
a longer run before anything is said about it.

The TTFT improvement is almost certainly a consequence rather than a cause: graphs do not
touch prefill, but at fixed concurrency a faster decode retires requests sooner, so a new
request waits less. Not worth claiming as a prefill effect.

Context: 15.50 ms still sits about 3.9x above the ~4 ms T4 bandwidth floor for this model,
so graphs removed a large share of the per-step host overhead measured earlier but not all
of it.

### Coverage is not correctness: a self-inflicted false alarm

The first version of the soak filed "states reached but never cancelled from" as an
invariant violation, alongside "you leaked a KV page". It is not the same kind of statement.
Whether a random injector at `cancel_probability=0.02` happens to catch a request in
`PREFILLING` during a four-second run is a fact about the workload, not about the engine.
The result was three tests failing on a correct engine, and the A/B printing
`INVARIANT VIOLATIONS` on clean runs - the precise way a suite teaches people to ignore it.

`SoakResult` now carries `violations` (correctness, always a failure) and `coverage_gaps`
(this run did not exercise something) separately; `ok` depends only on the former.
Coverage gaps now also record "no preemption occurred" and "no admission rejection
occurred", which are useful signals that a configuration is not testing what it intended.
One dedicated test asserts full cancellation coverage, with a workload built to make every
state reachable: long prompts to keep requests in `PREFILLING` across steps, a tight pool
to park `PREEMPTED` requests, oversubscription to fill `WAITING`, and a high cancel
probability to give the injector enough attempts.


## A measured optimisation target, not a quoted one

`benchmarks/kernels/roofline.py` derives the decode-step floor from two things measured on
the GPU in front of us, because both had been asserted rather than checked:

- **Achieved bandwidth**, from probes on the device: a STREAM-style copy, a read-only
  sweep, and an fp16 matrix-vector product. The GEMV probe is the one the floor is built
  on, because it *is* the decode pattern - read an enormous matrix, touch a tiny vector,
  write almost nothing. Buffers are 512 MB, far past the T4's 4 MB L2, so nothing is
  served from cache; a small-matrix microbenchmark would have flattered the number.
- **Bytes per step**, from the loaded checkpoint's own parameters. Qwen3-0.6B: ~881 MB of
  transformer layers plus ~311 MB of `lm_head` = ~1192 MB. The embedding table is excluded
  because a decode step gathers one row per sequence, but `lm_head` is included in full -
  and in this model it is *tied* to that same table, so the memory is read one row at a
  time for input embeddings and swept entirely for logits.

Two corrections to what had been said earlier in this project:

- **Spec bandwidth was quoted as if achieved.** The T4's 320 GB/s sticker figure produced
  a 3.72 ms floor. Real memory-bound kernels reach some fraction of the sticker number, and
  which fraction is exactly what was being hand-waved. Earlier in the project 256 GB/s was
  used for the speculative-decoding estimate and 320 GB/s later for the roofline, without
  the switch being flagged. The probe settles it per-device.
- **KV traffic was omitted, and it is not small.** At batch 8 with 512 tokens of context
  the KV read is ~470 MB against ~1192 MB of weights - 40% more traffic, and a materially
  higher floor. At batch 16 with 2048 tokens it is ~3758 MB, three times the weights, so a
  long-context decode step is KV-bound rather than weight-bound. That reframes the
  known kernel inefficiency (the batched decode kernel reads each GQA group once per query
  head, twice over for this model) from a minor waste into the dominant cost at long
  context.

A third correction, caught in the script before it ran: inter-token latency must be
compared against **step time**, not step time divided by batch. One step advances every
active sequence by exactly one token, so the gap a caller sees between their tokens is one
step regardless of how many sequences share it. Dividing by batch answers a throughput
question instead, and would have understated the floor eightfold at concurrency 8.

Projected answer, pending the run: with weights + KV at batch 8 / 512 context, the floor
lands between 5.5 and 7.6 ms depending on achieved bandwidth, putting the measured 15.50 ms
(CUDA graphs on) at **2.1x to 2.8x the floor** rather than the 3.9x claimed earlier. The
script reports the real figure and the absolute headroom in ms, which is what an
optimisation target has to be.


### Validated target (Tesla T4, Qwen3-0.6B FP16)

Measured, not quoted:

| probe | achieved |
|---|---|
| read-only sweep | 271.4 GB/s |
| fp16 GEMV (decode pattern) | **258.8 GB/s** |
| STREAM-style copy | 240.6 GB/s |

258.8 GB/s is 81% of the T4's 320 GB/s spec figure, inside the usual 70-85% band for
memory-bound kernels, and the probes order as they should (a copy pays write-allocate, a
read-only sweep does not). Bytes read per decode step: 880.9 MB of transformer layers plus
311.2 MB of tied `lm_head` = **1192.1 MB**, independent of batch.

**Weight-only floor: 4.61 ms/step.** Adding KV traffic:

| decode batch | 128 ctx | 512 ctx | 2048 ctx |
|---|---|---|---|
| 1 | 4.66 | 4.83 | 5.51 |
| 4 | 4.83 | 5.51 | 8.24 |
| 8 | 5.06 | 6.42 | 11.87 |
| 16 | 5.51 | 8.24 | 19.13 |

The headline ratio from the first run is **not trustworthy**, because the reference cell was
guessed. The script was told batch 8 / 512 context; the A/B workload built prompts of 20-224
tokens plus up to 64 generated, so real context was closer to 100-250, and closed-loop
concurrency 8 does not mean decode batch 8 - requests in prefill are not decoding. The
plausible cells put 15.50 ms at roughly **3x the floor with about 10 ms/token of headroom**,
rather than the 2.4x and 9.08 ms printed.

Rather than guess again, the operating point is now measured: `stats_snapshot` reports
`decode_batch` and `decode_mean_context`, the soak averages them across sampled steps, and
the A/B prints the exact `roofline.py` command line for each arm's real operating point.

**What the headroom is likely made of**, as the agenda for item 6 rather than a conclusion:
per-step Python in `_prepare_decode_metadata` (a scalar-write loop over block tables), the
64 KB host-to-device block-table copy per step, work outside the captured graphs, and the
batched decode kernel reading each GQA group once per query head - twice over for this
model. That last one is now quantifiable: at batch 16 / 2048 context the ideal KV read is
3758 MB against 1192 MB of weights, so doubling it adds roughly 14 ms to a 19 ms floor. At
short context it is a rounding error; at long context it is the single largest cost in the
step.


### Validated: CUDA graphs remove 72% of per-step overhead; 10.2 ms/token remains

Second independent A/B, five runs per arm, now with the operating point measured rather
than assumed. The effect replicates almost exactly: -64.5% in the first experiment,
-63.8% in the second.

| arm | decode batch | context | floor | measured ITL p50 | multiple | overhead above floor |
|---|---|---|---|---|---|---|
| graphs off | 6.9 | 133 | 5.01 ms | 41.93 ms | 8.4x | 36.92 ms |
| graphs on | 7.5 | 125 | 5.02 ms | 15.19 ms | **3.0x** | **10.17 ms** |

CUDA graphs removed 26.75 ms of 36.92 ms of overhead - 72% of everything above the memory
floor. The remaining 10.17 ms/token is the validated optimisation target.

The earlier "2.4x, 9.08 ms headroom" is corrected: it used a guessed 512-token context,
which inflated the floor by 28%. The conclusion is robust to the remaining uncertainty -
context varied 35-55% between runs, but that moves the floor only from 5.02 to 5.24 ms,
because at ~130 tokens KV traffic is under 10% of the weight read.

TTFT fell 38.0% with spread of only 1.4-3.2%, the tightest measurement in the set. It is
still a consequence rather than a cause: graphs do not touch prefill, but at fixed
concurrency a faster decode retires requests sooner.

### Why itl_p99 never resolved: it was not a tail

Both A/B runs reported `itl_p99` as unresolved, with run-to-run spread of 70-90% against a
consistent -33% median change. That spread was an artefact of how the percentile was
computed: over *per-request mean* inter-token latency, not over individual token gaps.

Averaging inside each request first destroys exactly what a tail measures. One 200 ms
hiccup in a 60-token response moves that request's mean by 3 ms and vanishes. With roughly
twenty requests per run, a "p99" over twenty means is the slowest request's average - a
single sample, which is why it swung wildly while p50 held at 6-9%.

Percentiles are now taken over every individual token gap, which is hundreds to thousands
of samples per run instead of tens. Gaps from preempted requests are excluded from the
headline tail, because one of their gaps contains the whole queue wait and would turn every
tail number into a preemption detector; that stall is already reported separately as
`stall_p99`, and `itl_p99_including_preempted` keeps the combined view. `itl_samples` is
reported so the sample count behind a percentile is visible, and `itl_request_mean_p50`
retains continuity with earlier runs.


## Revised target: the decode step is 1.64x the memory floor

Third A/B, with `itl_p50` now taken over individual token gaps rather than per-request
means. The headline number moved a long way, and in the right direction.

| arm | decode batch | context | floor | ITL p50 | multiple | overhead |
|---|---|---|---|---|---|---|
| graphs off | 6.9 | 137 | 5.02 ms | 32.98 ms | 6.6x | 27.96 ms |
| graphs on | 7.5 | 124 | 5.02 ms | **8.24 ms** | **1.64x** | **3.22 ms** |

CUDA graphs remove 24.74 of 27.96 ms of overhead above the floor: **88%**, not the 72%
computed from the old estimator. A pure decode step now costs only 64% more than the bytes
it must move.

Every earlier figure in this section is superseded. The chain of corrections, kept because
each was a different kind of error:

1. **3.9x above a 4 ms floor** - spec bandwidth quoted as achieved, KV traffic omitted.
2. **2.4x, 9.08 ms headroom** - measured bandwidth, but a guessed 512-token context that
   inflated the floor 28%.
3. **3.0x, 10.17 ms headroom** - real operating point, but ITL percentiled over
   per-request means, which inflated the measurement 84%.
4. **1.64x, 3.22 ms headroom** - percentiles over individual token gaps.

Only the last is defensible. The engine's decode path is in far better shape than any
earlier number implied.

### The percentile gradient points at prefill

| percentile | graphs on vs off | spread | verdict |
|---|---|---|---|
| p50 | -75.0% | 6.9% | real |
| p99 | -35.0% | 25.7% | real |
| p999 | -2.4% | 64.2% | unresolved |

Graphs help the median enormously, the tail moderately, and the extreme tail not at all.
That is exactly the signature of a tail composed of work graphs do not capture: the decode
path is graphed, prefill is not. So the deeper into the tail, the more of what is being
measured is a prefill forward interrupting decode, and the less a decode-path optimisation
can do about it.

That also explains the 84% gap between the median gap (8.24 ms) and the median
per-request mean (15.19 ms). Most gaps are pure decode steps; a minority ride a step that
also prefills, and those drag every average up.

**This is inference from a pattern, not a measurement**, so it is now instrumented rather
than asserted. `ContinuousBatchingEngine` counts `prefill_steps` and `decode_only_steps`
and exposes `last_step_prefill_tokens`; the soak times every `step()` call and reports
`decode_step_p50_ms`, `prefill_step_p50_ms`, the fraction of steps that carry prefill, and
`prefill_penalty_p50_ms` - how much longer a sequence waits for its next token when the
step it is riding also carries someone else's prompt. If the pattern holds, the penalty is
the number that justifies the prefill work in items 2 and 4; if it does not, something else
is producing the tail and that is worth knowing before optimising the wrong thing.


## Measured: half of inter-token latency is one caller waiting on another caller's prompt

Step-kind instrumentation, five runs per arm, CUDA-graph A/B.

| | graphs off | graphs on | change |
|---|---|---|---|
| decode-only step p50 | 35.999 ms | **8.202 ms** | -77.2% |
| prefill-carrying step p50 | 71.896 ms | **44.658 ms** | -37.9% |
| prefill step fraction | 21.7% | 22.6% | - |

**My earlier bracket was wrong by a factor of 20 to 200.** Reasoning from the percentile
gradient, I put prefill-carrying steps between 0.1% and 1% of all steps. They are 22.6%.
The inference assumed prefill occupied the extreme tail; at 22.6% it sits above roughly
p77, so p99 is prefill-dominated rather than "partly prefill".

With the counters, the percentiles line up exactly. A prefill-carrying step runs the decode
for every active sequence *and then* the prefill forward, so graphs accelerate its decode
half and it improves 37.9% - which is the p99 improvement of 37.4% to within noise. p50 is
decode-only steps at -77.2%, matching `itl_p50` at -77.3%. Nothing about the distribution
is mysterious once steps are labelled.

**Cost.** A prefill-carrying step costs 36.5 ms more than a decode-only step, 5.4x. At a
22.6% share the frequency-weighted average gap is
`0.774 x 8.202 + 0.226 x 44.658 = 16.44 ms`, which reconstructs the independently measured
per-request mean of 15.19 ms. Removing the decode work inside those steps leaves the pure
interruption at `0.226 x 36.456 = 8.24 ms/token`: **half of all user-visible inter-token
latency**.

**This reorders the plan.** The decode path has 3.22 ms/token of headroom above the memory
floor after CUDA graphs. Prefill interruption costs 8.24 ms/token. Prefill is the larger
target by 2.6x, and it is item 2 in the delivery order rather than item 6, so the ordering
already had it right for reasons that are now measured rather than assumed.

Two things to establish before optimising it:

- **Is 44.7 ms reasonable for this prefill?** Prompts average ~124 tokens against a 128
  chunk budget, so one chunk. A rough floor: the same 1192 MB weight read (4.61 ms) plus
  ~154 GFLOP of prompt compute (~2.4 ms at the T4's fp16 rate), plus the 8.2 ms decode the
  step also performs, is about 15 ms. Measured 44.7 ms is roughly 3x that. The code review
  flagged the chunked-prefill kernel as a per-token GEMV that reloads the whole K/V prefix
  per query token on CUDA cores; whether this workload takes that path or the SDPA batched
  path needs checking before blaming it.
- **Does smaller chunking help?** Shrinking the chunk makes each interruption shorter but
  spreads a prompt over more steps, delaying its first token. That is the ITL-versus-TTFT
  trade a QoS policy (item 4) has to pick a point on. `ab.py` gains `prefill_chunk`
  (128 vs 32) and `prefill_chunk_small` (128 vs 16) settings, and a `--cuda-graphs` flag,
  because studying prefill against an ungraphed decode path would drown the effect.

The soak now also reports the decomposition directly - `expected_gap_ms`,
`expected_gap_from_decode_ms`, `expected_gap_from_prefill_ms`, `prefill_share_of_gap` -
with a test asserting it reconstructs the observed per-request mean. If those three
measured quantities stop reconciling, one of them is wrong.


## Measured: prefill step cost is fixed per invocation, not proportional to work

`ab.py --setting prefill_chunk --repeats 5 --duration 8 --concurrency 8 --cuda-graphs`.

| | chunk 128 | chunk 32 | change |
|---|---|---|---|
| prefill step p50 | 42.541 ms | 41.502 ms | -2.4%, **unresolved** |
| prefill step fraction | 22.7% | 46.9% | +106.5% |
| decode step p50 | 8.046 ms | 8.043 ms | unchanged |
| ITL p50 | 8.149 ms | 8.462 ms | +3.8%, unresolved |
| TTFT p50 | 43.05 ms | 125.85 ms | **+192.3%** |
| expected gap (computed) | 15.88 ms | 23.73 ms | **+49.5%** |

**Four times less prefill work per step made the step 2.4% cheaper, inside the noise.** The
~34 ms a prefill-carrying step costs above a decode-only step is therefore fixed
per-invocation overhead, not work proportional to the tokens processed.

Consequences:

- **Smaller chunks are strictly worse here.** Same cost per interruption, twice as many
  interruptions, 49% worse average gap, nearly 3x the TTFT. The ITL-versus-TTFT trade the
  experiment was designed to map does not exist at these sizes, so this is not a QoS
  question (item 6) - there is no curve to choose a point on.
- **No scheduling change can fix it.** Chunk budgets, decode-first ordering and admission
  policy all move *when* prefill runs, never what an invocation costs. The target is the
  prefill path itself.

**A median cannot see this.** `itl_p50` moved +3.8% and was correctly called unresolved,
because with 46.9% prefill steps the median gap is still a decode step - the statistic is
blind to a doubling in how often the expensive step occurs. Only the frequency-weighted
`expected_gap_ms` shows the 49.5% regression. It was computed in the soak but missing from
the A/B's comparison list; it now leads that list, because it is the only latency metric
sensitive to a change in the *mix* of step kinds rather than the cost of one kind.

**Why the cost is probably launch overhead.** A prefill step's excess over a decode step is
36.5 ms with CUDA graphs on and 35.9 ms with them off - identical, confirming prefill is
entirely ungraphed and untouched by graphs. That excess is close to the 27.8 ms graphs
removed from the decode step (35.999 -> 8.202 ms), which was eager per-layer kernel-launch
cost. The hypothesis is that prefill is paying the same bill decode used to.

If that holds, the fix follows from it: a fixed chunk size produces fixed tensor shapes,
which are capturable. Padding every prefill to a small set of bucketed shapes would make
the prefill path graphable exactly as decode is. That is a real design change, not a tuning
knob, and it should not be started before the hypothesis is tested.

**Next measurement, not next optimisation:** `--setting prefill_chunk_large` (128 vs 512).
If a prefill step still costs ~42 ms at 512 tokens, the cost is per-invocation and the
launch-overhead hypothesis stands. If it rises roughly fourfold, there is a real compute
component and the per-token-GEMV chunk kernel flagged in the code review is implicated
instead. The two lead to different work, so the measurement comes first.


### chunk 512 vs 128 was a non-binding experiment; the workload is too short

Every metric came back unresolved, and it had to. The soak workload builds prompts of
20-224 tokens (mean ~122), so **both arms fit every prompt in a single chunk** - 128 and
512 are identical treatments here. The prefill fraction is the proof: 0.220 vs 0.205,
unresolved. Had prompts exceeded 128, chunk 512 would have collapsed several prefill steps
into one and the fraction would have fallen sharply.

A simple model reproduces all three arms. With ~122-token prompts, ~32 output tokens and
8 concurrent requests, each completed request needs *P* prefill steps and contributes about
4 shared decode steps:

| chunk | chunks per prompt | predicted fraction | measured |
|---|---|---|---|
| 32 | 4 | 0.50 | 0.469 |
| 128 | 1 | 0.20 | 0.220 |
| 512 | 1 | 0.20 | 0.205 |

So the chunk 32 finding stands - that treatment was real, and fixed cost dominates in the
32-128 range. Above 128 remains untested.

**This is the second experiment run with a treatment that could not take effect**, after
the p999 CUDA-graph comparison where both arms ran identical un-graphed prefill. A null
result from a non-binding treatment is indistinguishable from a null result from a real
one, which is what makes the mistake expensive. `ab.py` now runs a pre-flight
`binding_check` that computes chunks-per-prompt for each arm and refuses to start when they
are equal, naming the profile to switch to. It covers chunk-size binding only; it cannot
detect every null-by-construction design.

**The larger problem: the workload is not representative.** Everything measured about
prefill - 22.6% of steps, 8.24 ms/token of interruption - was taken at ~122-token prompts.
Real chat traffic carries a system prompt and history, typically 500-4000 tokens. Two
consequences:

- At ~122 tokens the prompt compute is roughly 146 GFLOP, about 2.25 ms on a T4, so compute
  is ~5% of a 42 ms prefill step and fixed overhead dominates. At 2048 tokens it is about
  2.5 TFLOP, roughly 38 ms, and would dominate instead. **The crossover is somewhere near
  1000-2000 tokens and every measurement so far sits well below it.**
- At a 128 budget a 2048-token prompt needs 16 chunks. If step cost is fixed at ~42 ms that
  is ~670 ms of prefill per request, and the interruption to concurrent decoders scales with
  it. The prefill problem is likely far worse at realistic lengths than measured.

`ab.py` gains `--prompt-profile short|chat|long` (mean ~122, ~656, ~1824 tokens). Every
prefill conclusion should be re-established on `chat` before any of it guides design.


## Prefill investigation: planned as one sweep instead of a chain of A/Bs

Four GPU sessions went into the prefill question and three produced nothing usable: a
CUDA-graph comparison whose p999 arm ran identical un-graphed prefill on both sides, a
chunk 512-vs-128 comparison where both arms fit every prompt in a single chunk, and a first
chunk A/B whose 49% regression was invisible to the statistic being read. Each fault was
visible before the run. The cause was procedural rather than technical: each session was
designed from the previous session's result, so no design ever got reviewed against the
whole question.

`docs/experiment-plan-prefill.md` pre-registers the rest of the investigation - the model
being fitted, three sweeps, predictions written before the data, and a decision rule fixed
in advance. `benchmarks/reliability/sweep.py` runs the whole Phase B matrix in one session
against one model load.

**The model.** A prefill-carrying step is assumed to cost `a + b * chunk_tokens`, where `a`
is per-invocation cost (kernel launches over 28 layers, metadata staging, the decode the
step also performs) and `b` is marginal cost per prefill token. Fitting a line across four
chunk sizes reads off both, instead of asking a binary question that can return null for
uninteresting reasons. Reference: the dense compute floor is ~0.018 ms/token at the T4's
~65 TFLOPS, so `b` near that means the kernel is near roofline and `b` several times that
means the kernel is the problem.

**Why it binds.** B1 sweeps chunk 64/128/256/512 on the `long` profile (~1824-token
prompts), which needs 29/15/8/4 chunks respectively - every arm is a distinct treatment,
checked before the run rather than discovered after.

**Decision rule, fixed in advance.** `a` > 20 ms with `b` < 0.05 means launch-bound: pad
prefill to bucketed shapes and capture it in CUDA graphs, as decode already is. `b` > 0.1
ms/token means the chunk kernel is off the roofline: replace the per-token GEMV with a
tiled causal prefill. Both means do the graph work first, being smaller and with a gain
estimable from the decode precedent. Neither means profile before designing anything.

Also instrumented: `prefill_sdpa_calls` / `prefill_chunked_calls` and their token counts,
because the engine keeps an SDPA fast path for batches of complete fresh prompts and a
resumable chunk path for everything else. A measured prefill cost cannot be attributed to
an implementation without knowing which one ran, and until now it was assumed.


## Phase B result: the prefill kernel is 22.5x off the compute floor

One 6.6-minute session, three sweeps, `results/prefill_sweep.json`. Full analysis and the
prediction scorecard in `docs/experiment-plan-prefill.md` (Phase C).

`step_ms = 21.13 + 0.40519 * chunk_tokens`, R^2 = 0.998. The fixed per-invocation cost is
21.13 ms; the marginal cost is **0.405 ms per prefill token, 22.5x the 0.018 ms/token dense
compute floor**. Compute is 55% of a chunk-64 step, 71% at chunk 128 and 91% at chunk 512.

**The prediction that mattered was wrong.** `b` was predicted at 0.02-0.10 ms/token and
measured at 0.405. That single coefficient decides which of two very different fixes gets
built, and the pre-registered tie-break ("both thresholds met, do the graph work first")
rested on the assumption that `b` was small. Graphing prefill removes at most `a`, capping
the gain at 28.9% at chunk 128; a tiled kernel reaching `b` = 0.05 would cut the step 62%.
The override is recorded in the plan with its reasoning, because silently revising a
pre-registered rule is the failure mode pre-registration is meant to prevent.

**A free win the script misreported.** B3's verdict asked whether packing four chunks into
one step makes that step cheaper. It cannot - the step does four times the work. Against
four separate steps: 55.37 ms versus 186.32 ms, **3.4x cheaper per unit work**. So the
fixed cost amortises across chunks from different requests, while a larger chunk from one
request costs linearly more. Two different knobs that had been conflated:
`prefill_chunk_size` bounds one request's slice, `max_prefill_tokens_per_iteration` bounds
the step. Raising only the latter already shows prefill share 61.0% → 45.4% and TTFT
496 → 472 ms at `chat`.

**Every earlier prefill number understated the problem.** At ~122-token prompts the
expected gap is 17.17 ms; at `chat` (~656) it is 36.01 ms and at `long` (~1824) 63.32 ms.
TTFT at `long` is **12.5 seconds** - the headline number for realistic prompts, invisible
until prompt length became a swept variable.

**Decode is not the problem, and is better at long context than short.** At batch 7 with
1824 tokens of context the floor is 10.26 ms against 13.36 ms measured - **1.30x**, versus
1.64x at ~128 tokens. The residual fixed overhead in the decode step is amortised by the
larger KV read. The remaining 3.22 ms/token of decode headroom identified earlier is real
but small beside a prefill step running 22.5x off its own floor.


## D2: tiled causal prefill kernel

The kernel being replaced runs on grid `(batch, q_heads, query_len)` — one program per
query token per head — and every program walks the KV prefix from zero. A 128-token chunk
with 16 heads is 2,048 programs re-reading the same keys and values. Across 28 layers at a
mean chunk start of 912 that is **28.67 GB of KV traffic per prefill step**, where a tiled
kernel with BLOCK_M=64 needs 0.462 GB: **62x less**. Scores were also computed with
`tl.sum(q * k, axis=-1)` on CUDA cores, so the tensor cores never ran.

The attention arithmetic is 26.8 GFLOP, about 0.41 ms on tensor cores. The work was never
the problem.

`engine/kernels/tiled_paged_prefill.py` is FlashAttention-2 structure over paged KV: one
program per tile of BLOCK_M queries, `tl.dot` for both matmuls, online softmax with an fp32
accumulator, and a per-tile causal bound so early tiles stream far less KV than late ones.
The page gather is the one real deviation from contiguous FlashAttention.

Recorded because it is easy to get wrong: masked scores use a finite `-1e30` sentinel
rather than `-inf`. A fully padded query row would otherwise make the softmax rescale
evaluate `(-inf) - (-inf)` and produce NaN; with a finite sentinel the rescale is `exp(0)`
and masked probabilities are zeroed explicitly.

Seven correctness tests run before any speed claim: a dense fp32 reference over five
(start, chunk) shapes, agreement with the kernel being replaced, padded rows exactly zero
and finite, five tile shapes agreeing, the fp32-PV path agreeing with the tensor-core path,
and multi-query grouping at both geometry edges.

Measured two ways. `benchmarks/kernels/prefill_attention_ab.py` times the kernel alone and
fits `b` on the Phase B axis — necessary because a 2x kernel win is a fraction of a step and
disappears into step-level noise. `ab.py --setting prefill_kernel` measures the in-engine
effect with the usual repeats and spread check. Target: `b` <= 0.05 ms/token against the old
kernel's 0.405.


### D2 first GPU run: the kernel did not compile

Seventeen of eighteen tests failed with one error repeated: Triton cannot read a plain
module-level Python global from inside a `@triton.jit` function. The masked-score sentinel
was declared as a module constant and had to be a kernel-local (or a `tl.constexpr`
instance). Nothing to do with the algorithm - the kernel never built.

Reviewing the rest of the file for anything else unverifiable without a device turned up a
second fault that would have failed immediately afterwards: the optional fp32 PV
accumulation used a broadcast-and-reduce, `tl.sum(probs[:, :, None] * values[None, :, :])`,
which materialises a `[BLOCK_M, BLOCK_N, HEAD_DIM]` intermediate - 2 MB per program at
64x64x128. It is now a `tl.dot` on fp32 operands like the fast path.

The process lesson is the useful part. There is no CUDA device in the authoring
environment, so a Triton kernel ships compile-unchecked and the first GPU run *is* its
compile gate. That is acceptable, but the failure mode is not: one compilation error
produced seventeen identical tracebacks and buried the numerical questions the suite was
written to answer. Every kernel test file now opens with a sub-second smoke test that
builds and runs the kernel on a tiny input, so a build failure costs one test and the
expensive correctness work is never reached on a kernel that cannot compile.


### D2 second GPU run: the kernel is numerically correct; one store-mask bug

The kernel compiles and every numerical test passes - both compile smoke tests, all five
dense fp32 reference shapes (including ragged chunks, unaligned prefixes and mixed
batches), and all four agreement checks against the kernel it replaces. The algorithm is
right.

One real bug, caught by exactly the test written for it. The store was masked by
`row_valid`, the *logical* bound - which rows carry a real token this chunk - so rows that
are padding were never written at all, and the output comes from `torch.empty_like`. The
uniform `1.5318e-05` values in the failure dump were allocator leftovers, not computation.
The kernel this replaces stored unconditionally with an already-zeroed result, so it
guaranteed zeros; that guarantee was lost.

The fix is not simply to drop the mask. Two different bounds had been conflated:

- `row_valid = offs_m < chunk_len` - logical, for the causal masking and the computation
- `in_tensor = offs_m < query_len` - physical, for the store

Dropping the mask entirely would write past the end of the output whenever `query_len` is
not a multiple of `BLOCK_M`, which is every ragged chunk. Both bounds are needed and they
are not the same number. The store now uses the physical bound, with `result` already
zeroed by the logical one.

Two tests added: one that poisons the output shape and asserts every row is finite
afterwards across four (query_len, BLOCK_M) combinations chosen so the last tile overhangs,
and one with a three-row batch of widely different chunk lengths asserting each row's
padding is zero and its neighbours are untouched.


### D2 third GPU run: 23/25, both failures are shared memory, not correctness

Every numerical test passes - the dense fp32 reference across all five shapes, agreement
with the kernel being replaced across all four, padded rows, tile overhang, multi-query
grouping, and the working tile shapes. The two failures are `OutOfResources: shared memory`
from test *parameters* the T4 cannot hold.

`BLOCK_M=128, BLOCK_N=64` needs 98,304 bytes against a 65,536 limit. That figure is exactly
Q(128x128xfp16 = 32 KB) + K(64x128xfp16 x 2 stages = 32 KB) + V(same = 32 KB), which says
Q, K and V all pass through shared memory - `mma.sync` reads operands from registers filled
by `ldmatrix` out of smem, so nothing avoids it - and K/V are double-buffered for the
pipelined loop. Doubling `BLOCK_M` doubles the Q tile alone.

`pv_in_fp32` at the default 64x64 needs 81,920 bytes: converting V to fp32 doubles that
operand and both `tl.dot` operands must be resident together.

**A correction.** Earlier in this conversation these were attributed to register pressure
from the fp32 accumulator. They are shared memory, and the arithmetic above identifies the
Q tile. The register explanation was wrong.

**And a second correction, about method.** The obvious next move was to encode a
shared-memory model so infeasible shapes could be rejected up front. That model reproduces
the 128x64 figure exactly and then mispredicts the fp32 case, and it claims 64x64 fp16
should not fit when it demonstrably does. Triton's allocation depends on pipeliner
decisions that vary by shape, so it is not a function of the tile dimensions. The runtime
already reports the exact requirement, so the code now catches rather than predicts: the
wrapper re-raises with the reported numbers plus what to reduce, and tests call
`_run_or_skip`, which attempts the shape and skips on the runtime's verdict. That keeps a
hardware limit from being reported as a numerical failure, and keeps the suite portable to
the 4060, where a ~100 KB SM will hold shapes a 64 KB Turing one will not.

The fp32 PV cross-check now runs at 32x32, since the comparison is about numerics rather
than tile size.

**Usable tile space on sm_75 at HEAD_DIM=128:** `BLOCK_M <= 64`, `BLOCK_N <= 64`. The
default 64x64 sits at the edge of what fits.


## D2 result: the tiled kernel is 3x SLOWER, and it is broken rather than badly suited

Colab T4, batch 4, 16 q heads, 8 kv heads, prefix 896, one layer:

| chunk | per-token kernel | tiled kernel | speedup |
|---|---|---|---|
| 64 | 5.429 ms | 19.104 ms | **0.3x** |
| 128 | 11.423 ms | 37.905 ms | 0.3x |
| 256 | 25.576 ms | 79.081 ms | 0.3x |
| 512 | 57.905 ms | 177.397 ms | 0.3x |

Fitted marginal cost went the wrong way: `b` = 0.83 ms/token for the old kernel and **2.49
for the new one**, 138x the compute floor against 46x. The Phase D2 target of `b <= 0.05`
is missed by fifty times.

Both kernels do the same total work, so putting each against its own hardware peak settles
what kind of failure this is:

| | FLOPs | time | achieved | share of that hardware's peak |
|---|---|---|---|---|
| per-token, CUDA cores | 1.88 GFLOP | 5.43 ms | 0.35 TFLOPS | **4.3%** of ~8.1 TFLOPS |
| tiled, tensor cores | 2.01 GFLOP | 19.10 ms | 0.11 TFLOPS | **0.2%** of ~65 TFLOPS |

The tiled kernel uses hardware 8x faster and gets 3.5x less done - roughly 28x off. A
correct FlashAttention-style kernel beats naive attention on every GPU it has been measured
on, so this is an implementation defect, not evidence that tiling is unsuited here. An
earlier draft of this entry blamed L2 residency; that was rationalisation and is withdrawn.

### The largest cause was computable before the kernel was written

The grid is `(batch, heads, ceil(query_len / BLOCK_M))`. At the benchmark's chunk of 64
with `BLOCK_M=64` that is `4 x 16 x 1 = 64` blocks on 40 SMs: most SMs get one block of 128
threads against 1024 of capacity, roughly 12% occupancy and nothing to hide memory latency
behind. The per-token kernel launches 4,096 blocks, 102 per SM.

This is structural to *chunked* prefill and is exactly what FlashAttention's design assumes
away. FA takes its parallelism from long query sequences; a prefill chunk is 64-512 tokens,
so a large `BLOCK_M` can collapse the grid to a single tile per (row, head). **Tiling the
query dimension trades away the parallelism that was paying for the redundant reads.** The
design was imported without checking that its central assumption held.

It is not the whole story - at chunk 512 the grid is 512 blocks, 12.8 per SM, and the
kernel is still 3x slower - so the remainder is now measured rather than guessed.

### One suspect eliminated with no run

The benchmark builds page tables with `torch.arange`, so pages are already sequential and
gather addresses contiguous. Coalescing of the page lookup is not implicated.

### Three defaults corrected, all off-GPU arithmetic

- **`num_warps` scales with the accumulator.** `acc[BLOCK_M, head_dim]` fp32 plus the score
  tile is 48 KB at 64x128; across 4 warps that is ~96 registers per thread before any
  temporaries. 8 warps halves it.
- **`BLOCK_M` scales with query length**, 32 below 256 tokens, so the grid keeps enough
  tiles to fill the device.
- **The K-load layout is a swept flag, not a hunch.** KV is stored `[page, slot, head, dim]`
  with `dim` contiguous, so a `[BLOCK_N, HEAD_DIM]` load is coalesced and a
  `[HEAD_DIM, BLOCK_N]` one is strided. Contiguous FlashAttention pre-transposes its
  pointer block for free; a paged gather cannot, so the transpose is paid on one side or
  the other and which is cheaper is empirical.

### Instrumentation, so the next run decides rather than another argument

A dense SDPA reference is added as an upper bound - the same attention on pages gathered
into contiguous tensors, gather outside the timed region. Any Triton kernel far below that
is losing to its own implementation rather than to the problem. The sweep reports Triton's
`n_regs`, `n_spills` and shared bytes per variant, covers `num_stages` and the K layout,
prints the grid size and blocks per SM, and states plainly when no configuration beats the
kernel it replaces - in which case the honest outcome is to delete it rather than tune it.

## Structural overhead removed ahead of the T4 re-measurement

Status: `APPLIED — UNMEASURED`. Earlier results are being re-run on Kaggle T4 (multiple
runs each) because some were malformed. Only changes that are faster by construction -
same bytes, same results, fewer launches or less host work - were made; anything that
needs a measurement to justify (the tiled prefill default, the prefill budget, SDPA
grouping) is left for the re-run to decide.

- **Decode metadata staging** (`_prepare_decode_metadata`): block tables, token ids,
  positions and lengths were written into the pinned host buffers one tensor
  `__setitem__` at a time. Timed off-GPU: **3.4 ms per step** at batch 16 x 64 blocks per
  row, vs 0.18 ms staging each row with one slice copy. That cost was serialised ahead of
  every graph replay and grew with context length. Now one slice copy per buffer/row.
- **Copy-on-write tail**: 2 x num_layers separate `copy_` launches (56 for Qwen3-0.6B) on
  a request's first decode step, since the exact-hit entry shares its partial tail block.
  Now one `torch._foreach_copy_`. The exact-entry design itself is unchanged.
- **Fused SwiGLU on strided halves**: `triton_swiglu` called `.contiguous()` on both
  inputs, so the gate/up projection fusion (Phase 9) paid two full activation copies per
  layer for the one GEMM launch it saved. The kernel now takes a row stride and reads the
  `split()` views in place. Phase 9's "neutral" result measured those copies, not the
  fusion; it should be re-run.
- **Prefix-cache eviction bookkeeping**: the block-reference map and cached-block set
  were rebuilt from every node and exact entry per evicted block. With a full cache each
  publish evicted several blocks at O(N) each, on the worker thread inside a step. The map
  is now maintained incrementally; a churn test checks it against a rebuild.
- **Warm-up before serving** (`ContinuousBatchingEngine.warmup`, called by the server's
  engine factory): synthetic requests through the ordinary step loop capture every graph
  bucket in both decode regimes and compile both prefill paths, then reset engine state.
  Previously each capture (two eager forwards plus a device sync) and Triton JIT landed on
  the first live requests to reach that bucket - the likeliest source of the unexplained
  p999. Benchmarks that construct the engine directly should call `warmup()` before
  timing, or keep their own warm rounds.
- `reset()` now passes `reserved_blocks` to the rebuilt scheduler, as `__init__` does.

Left as found, pending measurement or a design decision: `tiled_prefill=True` default
(D2 measured 0.3x before the tile defaults were corrected), SDPA fast path being
all-or-nothing across a plan batch, budget remainders planned as sub-chunk slivers, SSE
handlers polling every 5 ms per stream on the event loop.

## The tiled prefill kernel never used the tensor cores

Status: `DIAGNOSED — default flipped to the per-token kernel`. Kaggle T4, Triton compiled
kernels read directly (`benchmarks/kernels/prefill_attention_ab.py --ptx-only`, batch 4,
16/8 heads, prefix 896, chunk 64):

| | per_token | tiled |
|---|---|---|
| `mma.sync` in PTX | 0 | **0** |
| `fma.rn.f32` in PTX | 65 | **2052** |
| registers / spill bytes | 72 / 0 | **255 / 128** |
| shared memory | 2 KB | **49 KB** (T4: 64 KB per SM → one block per SM) |
| K/V global loads | mostly 16-byte vector | scalar |

`tl.dot` did not lower to an MMA instruction on this Triton/sm_75 combination. Both dots
are FMA loops, but they still run through the shared-memory layout conversions that exist
to feed tensor cores - that is the 49 KB - so the kernel pays for tensor-core plumbing and
gets CUDA-core arithmetic, at one block per SM, spilling. This is why a kernel moving 60x
fewer bytes than the naive one ran 3x slower (D2), and why tuning its tile shape could
not help: the instruction the design is built on is absent on the device.

Two earlier readings are corrected by this:

- D2's "0.2% of tensor-core peak" was measured against a peak the kernel could not reach.
  Against the CUDA-core peak it is ~1.4%, with occupancy and spills accounting for the rest.
- The per-token kernel's "4.3% of peak" is a lean kernel (72 registers, no spills, many
  blocks per SM) that is bandwidth-bound on redundant reads. That is the right shape to
  fix, not the wrong design.

Also found on the same pass: the engine A/B token-identity gate refused `tiled` versus
`per_token` - the two produce different greedy tokens on ~50-token continuations of
long prompts, though both pass the short-prompt reference tests. fp16 P·V accumulation
against fp32 is the likely cause.

Decision: `tiled_prefill` defaults to `False`. The engine had run the tiled kernel on every
chunked prefill since the D2 commit; the first Kaggle pass measured that path at 0.77
ms/token on chat prompts (98 ms per 128-token step) against the 0.405 ms/token fitted for
the per-token kernel in Phase B. The `prefill_chunk` and `kv_dtype` A/Bs from that pass are
confounded by it (their prefill "wins" are the slow kernel doing less work per step, and
the INT8 arm running a per-token-structured kernel) and are to be re-run.

Next for prefill, in order:

1. **Gather + SDPA for chunked prefill.** The benchmark's "upper bound" is a valid
   implementation: gather the prefix pages into a dense tensor, call
   `F.scaled_dot_product_attention`. PyTorch's mem-efficient kernel does use the T4's
   tensor cores. Gather cost at a 896 prefix is ~3.7 MB per row per layer, ~1.6 ms per
   28-layer chunk step at the measured 258 GB/s, against 37 ms per layer for the tiled
   kernel. It is also the same kernel the fresh-prompt path uses, so it should match that
   path's tokens rather than drift.
2. If a Triton kernel is still wanted: a register-tiled CUDA-core kernel - one K/V tile
   loaded per program, an unrolled loop over BLOCK_M query rows using the per-token
   kernel's `tl.sum(q * k)` body. The byte saving without `tl.dot` and without the
   shared-memory shuffles. Only worth building if (1) leaves something on the table.

### SDPA over gathered pages as a chunked-prefill implementation; the tiled kernel on later GPUs

`engine/kernels/sdpa_prefill.py` adds the third chunked-prefill attention path, selected by
`ContinuousBatchingEngine(prefill_attention="per_token" | "sdpa" | "tiled")` (the old
`tiled_prefill` boolean still works). Per layer it gathers each row's prefix pages into a
dense `[B, H, T, D]` tensor (`index_select` on the pool, no Python loop), builds one boolean
causal mask for the padded chunk batch, and calls `torch.nn.functional.scaled_dot_product_attention`
with `enable_gqa`. The longest `start + chunk` is passed from the planner so no layer reads
it back from the device. Unit-tested on CPU against a dense reference and on CUDA against
the per-token kernel; `test_d4` now runs the engine's chunked path with both kernels
against stock Transformers. `ab.py --setting prefill_kernel` is a three-arm run.

**The tiled kernel is kept, for the GPUs this engine will also serve on.** The T4 result is a
property of Triton on sm_75, not of the kernel's design: on Ada (RTX 40, sm_89) and
Blackwell (RTX 50, sm_120) `tl.dot` lowers to `mma.sync`, `cp.async` makes `num_stages > 2`
real, and the register and shared-memory budgets are larger. The same
FlashAttention-over-pages structure that loses 3x here is expected to win there. The
device profile in `engine/kernels/device.py` already distinguishes these cases
(`supports_async_copy`, per-device tile defaults). Rule for enabling it on a new device,
in order:

1. `benchmarks/kernels/prefill_attention_ab.py --ptx-only` must show `mma_sync > 0`,
   no spills, and vector K/V loads for the tiled kernel. Without that, do not run timings.
2. `--sweep-tiles` on that device picks the tile shape; the SDPA column is the bound to beat.
3. `ab.py --setting prefill_kernel --prompt-profile chat` in the engine, with the
   first-divergence-from-stock line recorded for each arm. A kernel that diverges earlier
   than `sdpa` does not become the default on speed alone.

Until then the default stays `per_token`, with `sdpa` the candidate to replace it on the T4
pending the Phase 1b measurement.

## Phase 1b on Kaggle T4: the corrected default, measured

Kaggle T4, commit `e709d4e`, `chat` profile (mean prompt 656 tokens), concurrency 8,
5 interleaved runs of 30 s per arm, warmed engines, graphs on. `results/t4/20260921_e709d4e/`.

**Flipping chunked prefill from `tiled` to `per_token`** (compared with the previous day's
tiled-default pass on the same workload): prefill step 98 → 47.9 ms, expected gap 63 → 38 ms,
TTFT p50 2.83 s → 0.89 s, ITL p99 295 → 82 ms. The `tiled` arm in the same run: prefill step
+110.6%, prefill GPU +134.2%, ITL p99 +275.0%, all far outside spread. Decode unchanged
(+0.7%, unresolved), as it must be.

**`prefill_chunk` on the real kernel**: chunk 32 vs 128 gives prefill step −13.5%, expected gap
unresolved (+4.7% in 13.3% spread), TTFT +343.6%. This is the original "cost is fixed per
invocation" finding, now on the right kernel; yesterday's −62% was the tiled kernel's
marginal cost. Chunk 128 stays.

**`prefix_cache`**: everything unresolved except prefill step +6.8% (spread 3.9%) with the
cache on. On a workload with three shared prefixes the publish/copy-on-write cost is
visible and the hit benefit is not. Not enabled by default until a workload with real
sharing shows it paying.

**`sdpa` arm**: +30.0% prefill step, +36.7% prefill GPU versus `per_token`, and it diverged
from stock earlier (tokens 10 and 18 on two prompts, `per_token` never). Cause found in the
implementation, not the idea: `enable_gqa=True` is honoured only by torch's math backend
on this GPU (no flash kernel on sm_75; the memory-efficient kernel rejects the flag), so
the arm ran fp32 materialised attention. Fixed by folding the query heads that share a KV
head into the query axis (`[B, kv_heads, repeat*Q, D]`), which the memory-efficient kernel
accepts; a CUDA test now forces that backend so a regression to math raises. Re-measured
as Phase 1c.

**`kv_dtype`**: the gate refused the INT8 arm for diverging from stock at token 0 on one
prompt. The identity prompts are submitted together, so that prompt fell into a mixed plan
and took the chunked path, where a prompt's own K/V are quantized before its attention
runs; a near-tie flipping there is quantization, which the INT8 kernels' own reference
tests bound. The 8-token rule is right for fp16 kernel swaps and wrong for a dtype change.
Re-run with drift allowed and the divergence recorded. Note from the first pass that INT8
did not move decode time at 1.8k-token contexts (±5%, unresolved), so at present it is an
accuracy cost without a speed win on this model.

Stock-divergence line, for the record: `per_token` matches stock on three of four identity
prompts and diverges at token 13 on the fourth, as do every fp16 arm and the fresh-prompt
path - that one is the engine's fp16 decode numerics, not any prefill kernel.

## Phase 2a — CUDA graphs for chunked prefill (built, unmeasured)

The decode step lost 72% of its time to graph capture. The chunked prefill forward is the
same un-graphed Hugging Face forward, launch-bound in the same way (Phase B fit: ~21 ms
fixed per invocation), and had no graph. It now has one:

- Chunk batches are staged into persistent pinned/device buffers `[max_active, chunk]`
  (`_prepare_prefill_metadata`), as decode batches are. Rows past the batch and columns
  past a chunk are inert - chunk length 0, block table -1 - so the write kernels store
  nothing and the attention kernels visit no keys for them.
- One graph per `(row bucket, attention kind, context bucket)`; the row buckets are the
  decode buckets, the context bucket is a power of two of the gathered prefix length and
  applies to the SDPA kind only. All graphs share one memory pool. Captured on inert rows at
  warm-up and lazily otherwise, so capture never touches a request's KV.
- The vocabulary projection now runs on each row's last chunk token only, in both the
  fresh-prompt path and the chunked path. Projecting the whole padded chunk was ~20% of a
  prefill step's FLOPs for 1/128 of the output.
- A kind whose forward refuses capture falls back to eager on the same buffers, and the
  reason is reported (`warmup()` summary, `scripts/check_hooks.py`), so a benchmark cannot
  silently measure the eager path as the graphed one.
- `prefill_cuda_graphs=False` keeps decode graphs and disables prefill capture: the A/B
  arm (`ab.py --setting prefill_graphs`), token-identical by construction.

Expected: prefill step 48 → ~25-30 ms on the chat profile if the fixed cost behaves as it
did for decode. Cost: a chunk shorter than the chunk size runs the full width under a
graph (a 2-token tail pays a 128-token forward); under the fixed-cost regime that is the
same time, and the planner change that avoids sliver chunks is still the right fix for it.

### Kernel-level truth vs engine-level result: both alternatives are fast in isolation

`prefill_attention_ab.py --chunks 64 128 256` on the T4, prefix 896, batch 4, one layer,
ms per call:

| chunk | per_token | tiled (8 warps) | dense SDPA bound | engine `sdpa_paged_prefill` |
|---|---|---|---|---|
| 64 | 8.85 | 4.27 | 1.07 | **0.89** |
| 128 | 11.93 | 6.14 | 1.39 | **1.15** |
| 256 | 26.50 | 11.78 | 2.06 | **1.75** |

The engine's SDPA path is 10x faster than the per-token kernel here and beats the dense
bound; the tiled kernel is 2x faster than per-token. The engine A/Bs said +18% and +110%
slower respectively. Two causes, one per kernel, both in the engine rather than the kernels:

- **Tiled ran at 4 warps in the engine, 8 in the benchmark.** `prefill_tile_defaults`
  chose 4 at `BLOCK_M=32`; at 255 registers that doubles per-thread demand and the spills
  with it. Default is now 8 warps at every tile, from the measured column.
- **SDPA rebuilt its mask and page indices every layer**: ~15 small launches x 28 layers
  in a forward that is launch-bound, which is exactly the fixed-cost regime the Phase B fit
  describes. Roughly the +8 ms the A/B saw. They now build once per step on the first layer
  and are reused (`_PrefillContext.sdpa_cache`), and the graphed prefill (Phase 2a) makes
  the remaining launches free.

INT8 KV on the long profile with drift allowed: every metric unresolved (decode -9% inside
23% spread). No speed win at 1.8k context on this model; stays off.

## SDPA under graphs: chunked prefill 2-3x faster and token-identical to stock — default flipped

Kaggle T4, commit `9f1f828` (SDPA on the memory-efficient backend, prefill graphs on, before
the per-step mask cache), `ab.py --setting prefill_sdpa`, 5 interleaved 30 s runs per arm,
decode graphs on in both arms. `results/t4/20260921_e709d4e/ab_prefill_sdpa_graphed_*.json`.

| | per_token | sdpa | change (spread) |
|---|---|---|---|
| chat: prefill step p50 | 45.3 ms | **27.0 ms** | −40.6% (9.9%) |
| chat: prefill GPU p50 | 35.7 ms | **16.5 ms** | −53.8% (20.4%) |
| chat: TTFT p50 / ITL p99 | 701 ms / 82.5 ms | **414 ms / 41.2 ms** | ITL p99 −50.0% (20.1%); TTFT unresolved |
| long: prefill step p50 | 87.9 ms | **30.3 ms** | −65.6% (31.6%) |
| long: prefill GPU p50 | 80.8 ms | **20.5 ms** | −74.6% (32.2%) |
| long: TTFT p50 / ITL p99 | 9.8 s / 211 ms | **3.7 s / 76 ms** | unresolved (spreads ~100%) |
| decode step | 10.3 ms | 10.4 ms | +1.1%, unresolved — as it must be |
| first divergence from stock, 4 prompts | 4, 18, none, 13 | **none, none, none, none** | |

The long profile gains more because attention's share of the step grows with prefix
length, which is the signature of a kernel win rather than noise. `sdpa` is also the first
chunked path to reproduce stock Transformers token-for-token on every identity prompt: it
is torch's own kernel, the same one the fresh-prompt path runs.

**Decision:** `prefill_attention` defaults to `"sdpa"` in the engine, the server factory and
the A/B `full` configuration. `per_token` stays as the measured baseline; `tiled` stays for
sm_80+ devices under the rule recorded above.

Open observation, not chased: `per_token`'s divergence from stock moved from
`[none, none, none, 13]` at `e709d4e` to `[4, 18, none, 13]` at `9f1f828`. The changes in
between (graphed, full-width chunk batches; vocabulary projection on last tokens only) are
shared with `sdpa`, which matches stock exactly, so the shared machinery is not the cause.
It is no longer the default, so this is recorded rather than investigated.

What is now established for the prefill path, in order of arrival: the tiled kernel never
reached the tensor cores (PTX); the per-token kernel was 10x off SDPA in isolation
(kernel table); the engine A/Bs contradicted the kernel table because the forward is
launch-bound and the SDPA path added launches (per-layer mask rebuild) - fixed by the
per-step cache and made irrelevant by graphing the forward; and with both in place SDPA
wins by 2-3x on the step. The remaining prefill cost is the un-attention part of the forward
plus the decode it carries, i.e. the same fixed-cost regime decode was in before graphs.
Phase 2a's `prefill_graphs` A/B measures that piece on its own.

## Phase 2b — one forward per step: decode rows and prefill chunks packed together (built, unmeasured)

A prefill-carrying step after Phase 2a still ran two graphed forwards back to back: the
decode batch (10.3 ms p50) and then the chunk batch (27.0 ms p50 for the whole step, so
~17 ms for the prefill forward). Both are the same 28-layer forward over the same weights.
Every weight is read twice, every launch is issued twice, and there are two device-to-host
syncs (one per argmax). The attention kernels are per-request either way; nothing else in
the forward cares which request a token belongs to.

**What was built.** `fused_step=True` (default) runs both as one forward over a single
packed row of tokens, `[1, decode_rows + prefill_rows * chunk]`: decode tokens first, the
chunk batch flattened after them, per-token `position_ids` for each. Norms, projections
and the MLP run once over the union. A third attention function
(`fused_step_attention_forward`) cuts the row at the boundary, hands the decode slice to
the paged decode kernel as `[N, heads, 1, D]` and the chunk slice to the chunk attention
(`sdpa` over gathered pages, or the Triton kernels) as `[B, heads, chunk, D]`, and packs the
two results back. The cuts are contiguous copies of a few hundred KB per layer. The
vocabulary projection runs on the decode rows plus each chunk row's last valid token, one
argmax, one `.tolist()` for both sides.

The forward is captured per (decode bucket, chunk-row bucket, attention kind, context
bucket, decode kernel regime) on inert rows, sharing the prefill graphs' memory pool.
Chunk-row buckets now start at 1 (also for the prefill-only graphs): under the 128-token
budget a chunk batch is nearly always one or two rows, and each padded row costs a whole
chunk of per-token work, so padding a single row to the smallest decode bucket (2) had
been doubling the prefill forward's non-attention cost. Above four chunk rows the fused
step runs eagerly at exact sizes rather than padding. Warmup captures every decode bucket
x {1, 2} chunk rows x {256, 512, 1024} context x both kernel regimes; other shapes are
captured on first use.

Scheduling order changed by one detail: with fusion the decode rows acquire their next KV
slot *before* admission and planning (the separate decode forward did this too, it just
also ran before admission), and the forward runs after. A chunk's capacity acquisition can
still preempt a decode row that was admitted after the prefilling request; such a row is
dropped from the batch before staging, as it would have been dropped from the next step.
A request finishing with EOS now frees its pages after admission rather than before, so
admission in a full pool can lag one step. `fused_step=False` restores the two-forward
step exactly.

**What the arithmetic says.** The fused forward's cost is bounded below by the larger of
the two it replaces, not their sum: the weight read (the decode floor, ~4 ms of the 10 ms
step) is shared, and the 128 chunk tokens' compute is the same. The expected prefill step
p50 is therefore in the region of 17-20 ms against 27 ms, and the decode rows' ITL
penalty on a prefill step drops by the same amount. The `sync_ms` phase halves on those
steps. Nothing else moves; the decode-only step is untouched.

**Correctness.** Same kernels in both arms, but the GEMMs see a different M dimension
(`N + 128` rather than `N` and `128` separately), so cuBLAS may pick a different tiling
and fp16 accumulation order can differ. Bit-identity across the arms is therefore not
guaranteed and `fused_step` is in `TOKEN_DRIFT_EXPECTED`; the stock-reference gate
(`first_divergence_vs_stock`, `--min-identical-tokens`) decides. The CUDA tests compare
the fused engine against both the two-forward engine and stock Transformers on staggered
prompts (eager and graphed), and under KV pressure with preemption.

**How it is measured.** `ab.py --setting fused_step --cuda-graphs` (`separate_forwards`
vs `fused_forward`), chat and long profiles, 5 x 30 s interleaved; notebook Phase 2b.
Decision metric: `prefill_step_p50_ms` and `expected_gap_ms`; `fused_gpu_ms_p50` is the
fused forward on its own. `check_hooks` fails if `fused_graphs` is zero after warmup.
Result to be recorded here when the run is pasted.

**First Kaggle run of Phase 2b failed before measuring anything**, with a device-side
assert on the very first chunked prefill step - and not in the fused path. The per-step
SDPA cache from `ec173b9` (mask and page indices built once per step, reused across
layers) was never run under graphs on a GPU: the graphed SDPA measurement was taken at
`9f1f828`, one commit earlier. Graph capture runs the forward once eagerly (compile,
allocate) and then again under capture, both on the same `_PrefillContext`; the capture
therefore found the cache already filled and recorded reads of the *eager* run's mask and
index tensors, which are freed after capture. Replay gathered pages through garbage
indices. Fix: the capture gets a fresh context after the eager run, in both the prefill
and the fused capture. `test_d4` now runs its graphed variant so this is caught by the
CUDA suite rather than by the benchmark.

## Phase 2b result: fused step −17% ITL p50 on both profiles; tail regression traced to in-window capture

Kaggle T4, commit `ae1eb71`, `ab.py --setting fused_step --cuda-graphs`, 5 interleaved
30 s runs per arm. `results/t4/20260921_ae1eb71/ab_fused_step_{chat,long}_30s.json`.

| | separate_forwards | fused_forward | change (spread) |
|---|---|---|---|
| chat: expected gap | 21.3 ms | **18.1 ms** | −15% |
| chat: ITL p50 / p99 | 24.1 / 38.5 ms | **19.9 / 33.1 ms** | −17.5% / −14% |
| chat: prefill-carrying step p50 | 24.9 ms | **20.7 ms** | −17.0% (6.8%) |
| chat: prefill penalty on decoders | | | −30.1% (9.9%) |
| chat: decode-only step p50 | 10.2 ms | 10.3 ms | +1.2%, unresolved — as it must be |
| chat: ITL p999 | | | **+149%** (spread 64%) |
| long: expected gap | 25.2 ms | **21.9 ms** | −13% |
| long: ITL p50 / p99 | 27.3 / 64.7 ms | **22.6 / 126.9 ms** | −17% / **+96%** |
| long: prefill-carrying step p50 | 26.3 ms | **23.1 ms** | −12.3% (6.3%) |
| long: prefill penalty on decoders | | | −27.3% (13.7%) |
| TTFT p50, both profiles | 406 / 3080 ms | 318 / 2901 ms | unresolved (spreads ~100%) |
| first divergence from stock, 4 prompts | none ×4 | none, 18, none, none | |

Fused wins every one of the ten interleaved pairs on the gap. The shape is what the
arithmetic said: prefill-carrying steps cheaper, decode-only steps untouched. Two
comparisons in the printed table look bad and are accounting: `host_stage_ms` +66-92%
because one phase now stages both buffers, and `sync_ms` +104-117% because the fused
step's single sync waits for the whole forward whereas the separate arm's `sync_ms` only
ever timed the decode wait (the prefill path recorded none). Both sit inside the step
time that fell.

**The tail regression is real and diagnosed.** Warmup pre-captured fused graphs for
{1, 2} chunk rows at contexts up to 1024. Long-profile prompts (~1.8k tokens) live in the
2048 bucket, so every (decode bucket x chunk rows x regime) key at 2048 was captured on
first use, inside the timed window, on every fresh engine: two eager forwards over a
2048-token gather plus syncs, ~150 ms each, about a dozen per run. At decode batch ~2
that is ~25 affected token gaps in a run of ~2,700, which is exactly the p99 sample. Chat
hits the same thing with rarer keys (3-4 chunk rows, the 64-wide kernel regime), so it
shows only at p999. Fix: warmup captures every shape the fused path can replay (all
decode buckets x chunk-row buckets up to the limit x the four context buckets x both
regimes; 72 graphs at three buckets, ~10 s), and the engine counts
`lazy_graph_captures` after warmup; soak/A/B report it and `check_hooks` fails on any.
The p99/p999 columns are to be re-measured with that in place; p50 and the step costs
do not depend on it.

**Token identity.** The fused arm diverges from stock on one prompt at token 18 (the
separate arm on none). That is the GEMM-shape drift predicted when this was built: the
same kernels see `N + 128` rows instead of `N` and `128`, cuBLAS picks a different
tiling, and an fp16 near-tie 18 tokens in flips. It passes the 8-token gate and is of
the same class as `per_token`'s drift. Recorded, not chased.

**Decision:** `fused_step=True` stays the default. Prefill-carrying steps are now 20.7 ms
on chat against 98 ms at the start of the T4 work (tiled, eager, two forwards).

**Re-run with warmup covering the fused shapes** (commit `2dcecfd`, same protocol,
`lazy_graph_captures` now reported per arm):

| | separate_forwards | fused_forward |
|---|---|---|
| chat: ITL p50 / p99 / p999 | 24.2 / 38.4 / 56.0 ms | **19.8 / 33.0 / 51.4 ms** |
| chat: captures inside the window, per run | 0 | 0 |
| long: ITL p50 / p99 / p999 | 27.2 / 64.5 / 114 ms | **22.6 / 44.6** / 151 ms |
| long: captures inside the window, per run | 1-3 | 3-8 |
| prefill-carrying step p50 | 24.9 / 26.3 ms | **20.5 / 23.0 ms** (−17.6% / −12.6%) |

Chat is now clean at every percentile: with zero captures in either arm, fused wins p50 by
18%, p99 by 14%, p999 by 8%. Long wins p99 by 31% and still loses p999, and the counter
says why: both arms capture inside the window there, the baseline 1-3 times, fused 3-8.
The uncovered shape is the 4096 context bucket - a long prompt plus its generation
crosses 2048 tokens, and a request re-prefilling after preemption gathers all of it.
Fused has three times the keys per context (row buckets x regimes), so it pays three
times the captures, which is the whole p999 difference. Warmup now derives the context
buckets from the engine's capacity (pool size, bounded by the model context) instead of a
fixed list ending at 2048. Cost: at 1024 blocks that is seven context buckets and ~126
fused captures, roughly 20 s more warmup, paid once.

The result stands as measured for everything but long-profile p999, which is expected
to follow p99 once the counter reads zero there.

**Long profile, third run, warmup contexts sized to capacity** (`f87416f`; zero
in-window captures in both arms, five runs each):

| long | separate_forwards | fused_forward | change |
|---|---|---|---|
| ITL p50 / p99 / p999 | 27.3 / 49.5 / 68.5 ms | **22.4 / 43.9 / 60.2 ms** | −18% / −11% / −12% |
| prefill-carrying step p50 | | | −13.9% (spread 5.4%) |
| prefill penalty on decoders | | | −27.7% |
| decode-only step | | | +0.6%, unresolved |

The baseline's own p99 fell from 64.5 to 49.5 ms once its 4096-context captures left the
window, so the −31% p99 in the previous run overstated the fused gain; −11% is the
number. Every percentile moves the same way in both profiles and the counter reads zero,
so Phase 2b is closed: **fused step, default on**. Prefill-carrying step p50 on chat:
98 ms (Sep 20, tiled, eager, two forwards) → 27.0 (SDPA, graphed) → 20.5 ms (fused).

Lesson recorded for the method: a tail percentile compared across arms is only a
result when both arms report zero graph captures inside the window. `lazy_graph_captures`
is now in every soak/A/B JSON and `check_hooks` fails on a non-zero count.

## Phase 2c — decode attention with the K/V tile shared across the GQA group (built, unmeasured)

With prefill-carrying steps at 20 ms, decode-only steps (10.2 ms p50, 75-90% of all
steps) are where the time is. The step's floor is the weight read, ~4-5 ms at the T4's
measured 258 GB/s; the rest is attention over paged KV plus per-layer overhead. The
per-head decode kernel runs one program per (row, query head) and each program streams
its KV head's tiles from the pool, so under Qwen3-0.6B's 2:1 GQA every K/V byte is read
twice per layer. At batch 8 and ~700 tokens that is 8 x 700 x 8 heads x 128 x 2 B x 2 (K,V)
= 23 MB per layer, 640 MB per step at the double read, i.e. of the same order as the
1.2 GB weight read. Adjacent head programs may already share through L2 (4 MB on the
T4), which is why this is measured before it is believed.

**What was built.** `paged_decode_gqa`: one program per (row, KV head), carrying the
online-softmax state of all `n_rep` heads (`[REP]` max and sum, `[REP, D]` accumulator);
each K/V tile is loaded once and used for the group. The products broadcast over
`[REP, BLOCK_N, D]` in registers, so the regime policy halves BLOCK_N for this kernel
(64 -> 32, 128 -> 64) to keep the per-head kernel's footprint; the static check that
forbids rank-3 intermediates has a stated exemption for this bounded case. The grid
shrinks by `n_rep`: at batch 8 that is 64 programs on 40 SMs, which is the risk. Engine
flag `decode_attention="gqa"` (default stays `"per_head"`), A/B setting `decode_kernel`,
sweep `paged_decode_regime_sweep.py --kernel both` with a best-config ratio table.

**What decides.** The sweep's `gqa/per_head` ratio at (batch 4-8, 512-1024 tokens) and
the engine A/B's `decode_step_p50_ms`. If the kernel wins in isolation but the step does
not move, the decode step is not attention-bound at this operating point and the next
lever is elsewhere (per-layer launch count inside the graph, or the lm_head GEMM). If
neither moves, the L2 already absorbed the double read and the item is closed.

## Phase 2c result: the rank-3 GQA kernel loses 1.5-2.8x; warmup closes p99 by 71%; the identity gate has a hole

Kaggle T4, commit `c7afabb`, `results/t4/20260921_c7afabb/`.

**Decode kernel, K/V tile shared across the group - first version, negative.** Sweep
(`paged_decode_regime_sweep.py --kernel both`, CUDA events, best config per kernel):

| context | batch | per_head best | gqa best | gqa / per_head |
|---|---|---|---|---|
| 128 | 8 | 0.109 ms (128x4) | 0.165 ms (32x8) | 1.52 |
| 512 | 8 | 0.136 ms | 0.211 ms | 1.56 |
| 1024 | 8 | 0.208 ms | 0.368 ms | 1.77 |
| 2048 | 16 | 0.659 ms | 1.376 ms | 2.09 |
| 2048 | 1 | 0.165 ms | 0.464 ms | 2.82 |

Engine A/B (`decode_kernel`, chat / long): decode step +76.5% / +129.8%, ITL p50 +35% /
+64%. Unambiguous. The ratio does not close at batch 16 x 2048, where the halved grid
(128 programs on 40 SMs) has parallelism to spare, so the grid is not the cost; the
`[REP, BLOCK_N, D]` rank-3 products are. That is the same lesson the static check
`test_no_rank_three_broadcast_intermediate` encodes, this time at a size (64 KB per
program) where the bytes argument does not apply: Triton's 3-D layouts are what cost.
The kernel is rewritten as an explicit two-head unroll over rank-2 tiles - one K load and
one V load per iteration feeding two heads, every product `[BLOCK_N, D]` - which is the
formulation that actually tests the shared read. To be swept once; if it does not beat
per-head in isolation the item is closed and the double read is judged absorbed by L2.

What the sweep gave regardless: per-head decode attention is 0.14-0.21 ms per layer at
(batch 4-8, 512-1024 tokens), i.e. 4-6 ms of the 10 ms decode step across 28 layers.
Attention is about half the decode step at the chat operating point; the other half is
the weight read plus per-layer overhead. That is the decode budget from here on.

**`warmup` (P7): closed.** Cold start vs warmed, chat, 5 x 15 s: ITL p99 −71.4% (spread
7.5%), p999 −63.1% (6.0%), `lazy_graph_captures` −100%; p50, step costs and gap all
unresolved at <1%. Warmup moves nothing but the tail, and moves all of it.

**`prefill_graphs`: refused by the identity gate**, and the refusal is informative.
`prefill_eager` diverges from stock at token 4 (prompt 0) and 13 (prompt 3);
`prefill_graphed` at 18 (prompt 1). Phase 1b's `per_token` arm diverged at 4, 18 and 13 -
the same positions, from a different kernel. Three implementations flipping at the same
three positions is the signature of near-ties in the logits there, not of a wrong kernel,
and the gate's "early divergence = wrong kernel" rule cannot tell the two apart.
`scripts/token_margins.py` prints the stock top-2 logit margin at each generated
position; if the margins at 4/13/18 are below fp16 resolution the gate is re-specified
to skip tied positions, and `prefill_graphs` reruns with drift allowed. Not decided
until measured.

## Phase 2c closed: ties, not kernels; the shared-KV read is already served by L2; graphs on prefill −57%

Kaggle T4, commit `9a9f86d`, `results/t4/20260921_9a9f86d/`.

**The identity gate was refusing on coin flips.** `scripts/token_margins.py`, stock
kernels, top-2 logit margin at the positions where three different chunked-prefill
implementations had all diverged:

| prompt | position | margin | fp16 ulp at that magnitude |
|---|---|---|---|
| 0 | 4 | 0.0078 | 0.0078 |
| 1 | 18 | 0.0156 | 0.0156 |
| 3 | 13 | 0.0156 | 0.0156 |

One ulp each. The next-smallest margins on those prompts are 4-6x larger, and prompt 2,
on which nothing ever diverged, has no margin below 0.14. So "diverges from stock within
8 tokens" was, here, "reproduces a one-ulp tie" - which no implementation that is not
bit-identical to stock's fp16 accumulation order can be asked to do. The gate now takes
the stock margin at every position (`TIE_MARGIN = 0.02`) and does not count a first
difference on a tied position as a divergence; ties are reported and saved separately
(`ties_vs_stock`). The "early divergence = wrong kernel" rule stands for real margins.
In hindsight the Phase 1b note that `per_token`'s divergence "moved from [none, none,
none, 13] to [4, 18, none, 13]" was this: the positions are properties of the prompts.

**GQA-shared decode kernel, rank-2 rewrite: 0.95-1.06x, noise.** Best-config ratio
gqa/per_head across 16 operating points sits between 0.89 and 1.13 with no trend in
batch or context. The kernel does half the K/V loads and runs in the same time, so the
second read was already coming from L2 (4 MB on the T4; a layer's KV for the batch at
the chat operating point is ~25 MB, but the two programs of a group run adjacently and
the tile is reused within microseconds). Closed; `decode_attention` stays `"per_head"`
and the variant is kept as the negative result it is.

What the two sweeps establish about decode attention instead: at (batch 8, 1024) the
per-head kernel moves 33 MB of unique KV per layer in 0.21 ms, ~160 GB/s - half of what
this GPU delivers to a well-shaped kernel. It is not traffic-bound; it is
parallelism-bound: 128 programs on 40 SMs, each walking its tiles serially, and at batch
1 only 16 programs. The lever that fits that diagnosis is split-K - several programs per
(row, head), each over a slice of the context, merged by a small second pass - not
fewer reads. Candidate D19, not built.

**`prefill_graphs` (Phase 2a, chat, drift allowed): graphed −56.7% prefill step (spread
8%), −60.5% prefill GPU, −57.4% ITL p50, −51.3% expected gap, −36.4% ITL p99.** With
`fused_step` on, the eager arm runs every prefill-carrying forward eagerly, so this is
the whole launch-bound cost of the chunk forward, captured. The graphed arm's decode-only
step is +8.6%: it completes more requests per second, so its decode batches are larger -
an operating-point shift, visible in `mean_decode_batch`, not a cost of the graphs. Its
`sync_ms` +1368% is accounting: the eager arm's wait is hidden inside its own launch
loop. Phase 2a closed.

**Gate failures in the CUDA suite** (`test_d4[*-graphed]`, `test_fused_step[graphed]`):
one wrong assertion of mine - graph dummy pages are permanently reserved and count as
used blocks - after every token assertion had passed. Fixed in `59dc024`.

## Serving surface: per-request sampling, OpenAI routes, Prometheus metrics (2026-09-22)

The engine decoded greedily for its whole measurement history, which is what made every
A/B comparable against stock Transformers. That is not a serving engine: real traffic
sets a temperature, sends stop strings, wants `usage` and logprobs, and arrives through
an OpenAI client. Three pieces, none of which changes a measured number.

**`SamplingParams` per request, applied per batch.** `engine/batching/sampler.py` treats
one step's `[N, vocab]` logits as a batch: penalties by one gather-scatter, temperature
by a `[N, 1]` divide, top-k by a per-row threshold from one `topk`, top-p by one sort,
min-p by one max. Two invariants decided the design. (1) A batch where every row is
greedy and asks for no penalties returns `logits.argmax(-1).tolist()` before any of that
machinery runs, so the default path is bit-identical to what every benchmark measured.
(2) A row's token must not depend on its neighbours, which per-row parameters give for
free and a per-request `seed` extends to the draw itself - at the cost of one
`torch.multinomial` per seeded row, since generators cannot be batched. Greedy is not
the same as "nothing to do": penalties change which token the argmax picks, as they do
in `transformers`, so they are applied before the shortcut is taken.

**Graphs now return logits, not tokens.** Sampling is per request and a graph is captured
for a *shape*, so the argmax could no longer live inside the captured forward. The
prefill and fused forwards end by copying their `[rows, vocab]` logits into one shared
buffer (`_write_logits`) whose address every graph records. A private vocabulary-sized
output per captured shape would have been ~5 MB x dozens of shapes; the shared buffer is
9.7 MB total and costs one extra copy per step. The fused step also compacts its live
rows on the device before the transfer, so it still pays exactly one device-to-host copy
per step for decode and prefill together.

**Stop conditions** are per request: the request's own stop token ids, then EOS unless
`ignore_eos`, then the length bound. Stop *strings* live in the API layer, which has the
detokenized text; a request that hits one is cancelled so its KV pages return to the pool
instead of generating to its bound.

**OpenAI routes** (`engine/server/openai.py`): `/v1/models`, `/v1/completions`,
`/v1/chat/completions`, streaming for both, `usage` including
`stream_options.include_usage`, logprobs, the model's own chat template. Parameters the
engine does not honour (`n > 1`, `echo`, `best_of`, `logit_bias`) are refused with 400
rather than ignored - a serving surface that silently drops a parameter is worse than one
that says no. 17 CPU tests cover the contract against a scripted engine.

**Prometheus `/metrics`** (`engine/metrics/prometheus.py`): the numbers the soak already
computed, exported in the text format without a client dependency - TTFT, inter-token
latency, end-to-end latency, queue time, prompt and generation token histograms, and
gauges for running/waiting requests, decode batch, KV utilization and
`graph_captures_in_service` (the counter that caught two false tail regressions on the
T4). Recording happens on the worker thread as requests terminate; `render()` only reads
the exporter's own copies, never live scheduler state.

## Hooks: making the engine portable across models, GPUs and kernels (2026-09-23)

Everything measured so far is one model on one GPU, and the code said so in three places:
the SwiGLU installer matched the class name `Qwen3MLP`, the RoPE installer imported
`transformers.models.qwen3.modeling_qwen3` by path, and the attention implementation was
an `if/elif` over three constants whose defaults were whatever won on a T4. None of those
is wrong; all three are unextendable, and the RTX 4060 is about to make each one bind.

**Model hook** (`engine/model/adapters.py`). Fusion targets are now found structurally -
any module with `gate_proj`/`up_proj`/`down_proj`, any class whose name ends in `RMSNorm`
with a 1-D weight - and RoPE is patched in `type(model).__module__`, the model's own
modeling file. A Llama or Mistral checkpoint therefore gets the same fused kernels with
no new code. What *is* declared is what the kernels cannot serve: `ensure_supported()`
refuses head_dim above 128, non-divisible GQA, sliding-window attention, mixture-of-
experts and MLA at load time, with the reason. Serving a windowed model on kernels that
attend to the whole prefix would have produced a plausible-looking wrong answer.

**Kernel hook** (`engine/backends/`). A backend is a name, a phase, a `run` and an
`available(profile, geometry)` that returns the reason it cannot run here or None.
`resolve("auto")` takes the highest-priority available backend; a named backend that
cannot run raises with the reason rather than falling back, because a silent fallback is
exactly how Phase 2a measured an eager forward for a week and called it a graph result.
`describe()` prints the whole table with reasons, which is what
`check_hooks.py --backends-only` now reports in seconds on an unfamiliar machine.

**Device hook** (`engine/backends/policy.py`). `MEASURED` holds, per architecture, the
settings an A/B on that architecture chose, each with its journal entry - sm_75 is
`per_head` + `sdpa` + fp16 and says why. Anything else gets capability-led defaults
(highest-priority runnable backend, bf16 from sm_80) and every reason string carries the
word `unmeasured` plus the command that would settle it. The engine keeps the whole
decision in `backend_reasons`, so a result can always answer "why this kernel".

**Two new kernels**, both registered rather than wired in:

* `paged_decode_split_k.py` - FlashDecoding's structure. The per-head kernel reaches
  ~160 GB/s at (batch 8, 1024 tokens) with 128 programs on 40 SMs; at batch 1 it has 16.
  Splitting the key range gives each slice its own program and merges the partial
  softmaxes in a second pass. `choose_splits` refuses to split a grid that is already
  wide enough (16 rows x 16 heads returns 1) and the wrapper then calls the single-pass
  kernel, so this can only cost a launch when it is not needed. Unmeasured; the arm is
  `ab.py --setting decode_split_k`.
* `flash_paged.py` - `flash_attn_with_kvcache` over the existing pool for both phases.
  The pool layout is already what flash wants (`[blocks, page, kv_heads, head_dim]` plus
  an int32 block table), so this is a mapping, not a port: no gather for prefill, no mask
  tensor, GQA native, internal splitting for decode. Import-guarded; on sm_75 its absence
  is a reason in the table rather than an error.

Two consequences worth stating. The engine now refuses a named backend it cannot run,
which is a behaviour change from "fall back to the default" - intentional, and the
message names the reason. And `DECODE_ATTENTION_KINDS` / `PREFILL_ATTENTION_KINDS` are no
longer the source of truth; the registry is, and those tuples remain only for callers
that enumerate the built-ins.

Nothing here changes a measured number: the T4's defaults are what `MEASURED[75]`
returns, and the CPU suite (288 tests) plus the CUDA gates cover the new paths.

## RTX 4060 FlashAttention closure (2026-09-24)

The sm_89 measurement changed the provisional hook conclusion. FlashAttention-2 2.8.4
was built from source for the local CUDA/PyTorch combination and its kvcache API executed,
but required pages divisible by 256. That geometry constraint made “Flash versus baseline”
several distinct experiments rather than one switch.

The first combined page-256 arm lost: ITL p50 +11.5%, prefill step +16.1%. Phase isolation
found two independent causes. Flash decode could not enter a CUDA graph because its
kvcache workspace path synchronizes during capture; against graphed `per_head` it produced
+113.9% ITL p50 and was rejected. Flash prefill performed a CUDA-to-CPU length conversion
inside the attention adapter once per transformer layer. Moving grouping to pinned host
staging once per engine step and adding an equal-length direct path changed the end-to-end
result.

For Qwen3-0.6B FP16 long prompts, optimized direct Flash prefill reduced prefill step p50
7.8%, prefill penalty 13.4%, fused GPU p50 8.8%, and ITL p99 22.6%. Qwen3-1.7B confirmed
the serving result: expected gap -5.3%, TTFT p50 -13.7%, ITL p99 -9.9%, prefill GPU p50
-5.3%, and fused GPU p50 -11.8%. The stock leading-token gate passed in FP16.

The compatibility experiment was negative. Gathering normal 16-token pages into dense K/V
before calling Flash made prefill step 38.5% slower and TTFT 52.6% worse. It remains an
explicit low-priority backend only to keep the result reproducible. Forced decode split
counts also lost to FA2 automatic planning.

The accepted Ada deployment is consequently asymmetric: graphed Triton `per_head` decode
plus eager direct Flash prefill on page 256, FP16. Compact pages use SDPA. The external
OpenAI-compatible smoke passed 6/6 after it exposed one final correctness defect: exact
prefix reuse had skipped the first seeded RNG draw. Exact cached token decisions are now
restricted to parameter-free greedy requests; sampled traffic still reuses complete KV.

Full evidence, commands, negative results and publication boundaries are consolidated in
`docs/rtx4060-final-evaluation.md`.

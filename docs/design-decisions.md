# Inference engine design-decision journal

This is the authoritative decision record for the project. It complements the
optimization journal (measurements) and the understanding journal (runtime explanation).
Every material architecture or implementation choice must be recorded here when it is
accepted, rejected, or deliberately deferred.

Each entry states the decision, why it was made, the active code boundary, its tradeoff,
and the evidence used to keep or reject it. New implementation work must add or update a
record before it is considered complete.

## Active architecture decisions

### DD-001 — Target the Colab Tesla T4 and use FP16 by default

- **Decision:** treat the T4 (`sm75`, 14.56 GiB) as the primary target and resolve the
  normal runtime dtype to FP16.
- **Why:** the T4 has native FP16 Tensor Cores but no native BF16 Tensor Cores. FP16 is
  the appropriate throughput and memory baseline for Qwen3-0.6B.
- **Code:** `engine/model/loader.py`.
- **Tradeoff:** results are intentionally hardware-specific; they should not be presented
  as an Ampere/Hopper optimum.
- **Evidence:** Phase 0 environment probe and repeatable Qwen3-0.6B baseline.

### DD-002 — Keep Hugging Face Qwen for transformer structure; replace only cache/attention-critical paths

- **Decision:** retain Hugging Face for model loading, layer order, projections, MLP,
  residuals, logits, and tokenizer integration. Intercept attention dispatch and selected
  pointwise operations instead of rewriting Qwen end-to-end.
- **Why:** this preserves model correctness and lets PyTorch/CUTLASS use mature FP16 GEMM
  implementations, while replacing the inference-specific DynamicCache bottleneck.
- **Code:** `engine/batching/continuous_batching.py`, `engine/model/loader.py`,
  `engine/kernels/rmsnorm.py`, `rope.py`, `swiglu.py`.
- **Tradeoff:** the engine is Qwen/Transformers-version aware rather than model-agnostic.
- **Evidence:** token-identical continuous-batching correctness tests and real Qwen traces.

### DD-003 — Treat the explicit HF runner as the correctness reference, not the serving path

- **Decision:** preserve `ExplicitDecodeRunner` as a transparent, single-stream reference
  implementation; do not serve requests through it.
- **Why:** it exposes the exact prefill/decode recurrence and provides a stable output
  oracle. Its `DynamicCache` grows by `torch.cat`, reallocating cache storage on decode.
- **Code:** `engine/model/runner.py`; serving uses `engine/batching/continuous_batching.py`.
- **Tradeoff:** two paths must remain token-equivalent through tests.
- **Evidence:** 32-token T4 baseline at about 22.8 tok/s and traced DynamicCache storage
  replacement on every decode step.

### DD-004 — Use a shared persistent paged KV pool as the authoritative cache

- **Decision:** allocate K and V as one fixed GPU pool per layer with layout
  `[physical_block, token_offset, kv_head, head_dim]` and keep it for engine lifetime.
- **Why:** paged storage avoids cache reallocation/copying, allows noncontiguous request
  histories, supports concurrent requests, and enables prefix-page sharing.
- **Code:** `ContinuousBatchingEngine.key_pool/value_pool`, `engine/cache/paging.py`.
- **Tradeoff:** every attention kernel must translate logical positions through a block
  table, which makes kernels more complex than contiguous-cache SDPA.
- **Evidence:** real Qwen trace proved unchanged pool pointers across a 28-layer decode.

### DD-005 — Use 16-token physical pages with refcounted ownership

- **Decision:** use `block_size=16` by default, with `BlockAllocator` free lists and
  per-page reference counts.
- **Why:** this is small enough to limit internal fragmentation and makes page-level
  sharing/copy-on-write practical on the T4 workload.
- **Code:** `engine/cache/allocator.py`, `engine/cache/paging.py`.
- **Tradeoff:** smaller pages increase page-table length and address-translation work;
  every partly filled last page incurs bounded internal fragmentation.
- **Evidence:** allocator and real scheduling traces; all kernel layouts and benchmarks
  use the same geometry.

### DD-006 — Write K/V directly with Triton before paged attention reads it

- **Decision:** route post-RoPE K/V from each Qwen attention layer into its paged pool via
  Triton writers, then execute paged attention against that same pool.
- **Why:** eliminates HF cache concatenation and avoids materializing contiguous history.
- **Code:** `engine/kernels/kv_write.py`, `paged_decode_batched.py`, `paged_prefill.py`.
- **Tradeoff:** custom kernels are constrained to tested head dimensions and require an
  explicit model-integration contract.
- **Evidence:** SDPA comparisons pass; real decode changed exactly the expected 56 K slots
  for two requests across 28 layers.

### DD-007 — Use batched paged decode for continuous batching

- **Decision:** a decode step stacks one token from each live request and launches one
  paged attention grid over `(sequence, query_head)`.
- **Why:** batching amortizes weight reads and kernel launches across active sequences.
- **Code:** `ContinuousBatchingEngine.decode_step`, `paged_decode_batched.py`.
- **Tradeoff:** requests have unequal sequence lengths, so page tables and sequence lengths
  must be staged per row; latency is sensitive to scheduling policy.
- **Evidence:** throughput improved from about 23 tok/s sequentially to hundreds of tok/s
  at saturated concurrency; output tokens remained identical.

### DD-008 — Stage decode metadata in persistent pinned host and device buffers

- **Decision:** allocate fixed `[max_active, ...]` host-pinned and CUDA metadata buffers
  once; update contents and slice views each decode iteration.
- **Why:** avoids per-token allocation and scalar transfer overhead and is required for
  fixed-address CUDA graph replay.
- **Code:** `_prepare_decode_metadata` in `engine/batching/continuous_batching.py`.
- **Tradeoff:** complete fixed-width block-table rows are copied even when only a small
  prefix is live.
- **Evidence:** real two-row trace reused all device pointers and copied 1,064 bytes.

### DD-009 — Commit logical request length only after a successful model forward

- **Decision:** reserve writable KV capacity before forward, write pending K/V during
  forward, then append committed length only after logits return.
- **Why:** keeps CPU lifecycle state consistent with successful inference and gives a
  clear failure boundary.
- **Code:** `_ensure_kv_capacity`, `decode_step`, `prefill_chunks`.
- **Tradeoff:** readers must use `length_offset=1` during decode because the new K/V is
  already written but the committed length is still old.
- **Evidence:** real decode integration trace and continuous correctness tests.

### DD-010 — Schedule decode first, then bounded round-robin prefill

- **Decision:** each engine iteration serves existing `DECODING` work first, admits work,
  then plans at most one prompt chunk per prefill request in rotating order.
- **Why:** protects interactive inter-token latency while allowing long prompts to make
  cooperative progress.
- **Code:** `ContinuousBatchingEngine.step`, `FCFSScheduler.plan_prefill`.
- **Tradeoff:** prompt TTFT and overall throughput depend on budget selection; a small
  budget can underutilize the GPU.
- **Evidence:** real long/short trace showed a seven-token request begin decoding while
  two 125-token prompts were still prefilling.

### DD-011 — Default prefill token budget to 128; retain chunking as separate control

- **Decision:** use `max_prefill_tokens_per_iteration=128` as the T4 default. Keep
  `prefill_chunk_size` separate: it limits one request's contribution in a visit, whereas
  the budget limits combined prompt work in an iteration.
- **Why:** 128 was the best measured mixed-arrival balance; 16 was used only for teaching
  traces, not as a serving recommendation.
- **Code:** `ContinuousBatchingEngine.__init__`, `FCFSScheduler.plan_prefill`.
- **Tradeoff:** a deployment targeting short-request tail latency may select 64; a
  different model/GPU must remeasure rather than inherit this value blindly.
- **Evidence:** mixed-arrival sweep: 64/128/256 budgets gave 295/430/424 tok/s and long
  p50 TTFT 898/531/544 ms respectively.

### DD-012 — Use SDPA for complete fresh prefills; use custom paged prefill only for resumable chunks

- **Decision:** complete fresh prompt batches take the HF SDPA fast path with a pool-backed
  cache adapter. Partial/resumable chunks use custom causal paged prefill.
- **Why:** dense SDPA is faster for short complete prompts; custom prefill is needed only
  when scheduling requires a prefix plus a partial chunk.
- **Code:** `prefill_batch`, `prefill_chunks`, `engine/cache/pool_cache.py`,
  `engine/kernels/paged_prefill.py`.
- **Tradeoff:** two prefill implementations must preserve identical pool semantics.
- **Evidence:** integrated prefill/chunked-prefill correctness tests pass.

### DD-013 — Share prefix pages through a block-aligned radix cache plus exact entries

- **Decision:** cache complete prompt blocks in a radix tree and retain exact entries with
  the first-token decision when the whole prompt can be reused.
- **Why:** longest-prefix reuse saves prompt prefill; exact hits can bypass prefill fully.
- **Code:** `engine/cache/prefix.py`.
- **Tradeoff:** cached pages consume allocator capacity and require LRU eviction; only
  immutable page content may be shared.
- **Evidence:** 871-token exact hit achieved about 110x measured TTFT speedup; real trace
  made zero attention-hook calls on exact admission.

### DD-014 — Copy shared partial tail pages before a decode write

- **Decision:** if an exact prefix hit ends inside a shared page, allocate a private page
  and copy the whole K/V tail across all layers before writing a new token.
- **Why:** the page contains cache-owned valid prefix data, and writing an unused slot in
  place would mutate shared state.
- **Code:** `_ensure_writable_tail`, `KVBlockManager.copy_on_write_tail`.
- **Tradeoff:** Qwen FP16 copies 1.75 MiB across K/V and 28 layers for one 16-token page.
- **Evidence:** real prefix trace verified 28/28 copied K prefixes and correct token output.

### DD-015 — Make INT8 paged KV opt-in, not default

- **Decision:** retain native INT8 K/V writer and reader as a selectable cache mode; keep
  FP16 as default.
- **Why:** INT8 halves KV memory and helps larger, long-context batches, but quantization
  work can lose at short/single-stream decode.
- **Code:** `engine/kernels/int8_paged_kv.py`, `kv_cache_dtype` engine option.
- **Tradeoff:** around 0.9% relative attention error and variable latency benefit.
- **Evidence:** roughly 49.2% KV-memory saving; speedups from sub-1x at batch 1 to about
  1.37x at context 2048/batch 16 with measured run ranges.

### DD-016 — Use fixed-width CUDA-graph buckets with isolated dummy rows

- **Decision:** capture paged decode graphs for configured widths and pad a smaller live
  batch with permanent dummy pages.
- **Why:** CUDA graphs remove repeated CPU launch overhead while preserving variable request
  contents through stable metadata-buffer addresses.
- **Code:** `engine/graphs/paged_decode_graph.py`, graph logic in
  `continuous_batching.py`.
- **Tradeoff:** capture warmup cost, graph-memory footprint, padding work, and possible
  short-request tail-latency regression.
- **Evidence:** fixed-width replay measured about 3.9–4.3x; mixed-arrival graph mode gave
  1.86x throughput but worsened short p95 ITL in the measured workload.

### DD-017 — Enable graph buckets by default in the service, not unconditionally in the base engine

- **Decision:** the FastAPI service defaults to `(2, 4, 8, 16)` graph buckets; callers
  can choose dynamic decode for latency-sensitive use.
- **Why:** the service target favors throughput, but the mixed-arrival result showed a real
  QoS tradeoff.
- **Code:** `engine/server/api.py`.
- **Tradeoff:** graph behavior is configuration-dependent, not universally optimal.
- **Evidence:** Phase 12 mixed-arrival A/B.

### DD-018 — Retain selective Triton fusion; reject unproven linear quantization/fusion paths

- **Decision:** keep RMSNorm, RoPE, SwiGLU, KV writers, and paged attention kernels. Keep
  gate/up projection fusion optional and leave W8A16/W8A8 linear out of serving.
- **Why:** accepted kernels reduced real repeated elementwise/cache work. T4 W8A16/W8A8
  experiments were 0.10–0.19x FP16 performance; paired MLP fusion was only ~1.02x and
  noisy.
- **Code:** accepted `engine/kernels/rmsnorm.py`, `rope.py`, `swiglu.py`; experimental
  `w8a16_linear.py`, `w8a8_linear.py`.
- **Tradeoff:** the engine remains FP16-GEMM dominated, but avoids a slower and more
  complex default path.
- **Evidence:** profiler and A/B results recorded in Phases 9–10.

### DD-019 — Keep one GPU-owning worker thread for API serving

- **Decision:** FastAPI handlers only enqueue immutable request inputs/cancellations;
  `ContinuousBatchingService` owns all engine and scheduler mutations on one worker.
- **Why:** prevents API threads from releasing pages or mutating state while CUDA work is
  executing.
- **Code:** `engine/server/continuous.py`, `engine/server/api.py`.
- **Tradeoff:** one process serves one engine forward at a time; scale-out requires more
  processes/GPUs or a future multi-engine context redesign.
- **Evidence:** lifecycle, queue-backpressure, cancellation, and real disconnect tests.

### DD-020 — Make server admission and cancellation bounded and worker-routed

- **Decision:** enforce ingress limit, scheduler waiting limit, prompt-token limit, and
  request timeout; route timeouts and disconnects through the worker cancellation queue.
- **Why:** protects KV capacity and ensures a terminal request releases blocks through the
  same owner that scheduled it.
- **Code:** `engine/server/api.py`, `engine/server/continuous.py`.
- **Tradeoff:** clients may receive deliberate 429/413/504 responses under pressure.
- **Evidence:** Phase 15 server tests and real uvicorn disconnect harness.

### DD-021 — Measure transport metrics over real sockets, not TestClient buffering

- **Decision:** use the loopback uvicorn harness for published SSE TTFT/completion results.
- **Why:** ASGI TestClient buffers stream behavior and cannot report wire-visible TTFT.
- **Code:** `benchmarks/server/uvicorn_load.py`.
- **Tradeoff:** the harness is slower and more environment-sensitive than in-process tests.
- **Evidence:** 16-request/512-token Colab burst: 666.1 tok/s, wire TTFT p50/p95/p99
  210/270/278 ms, with disconnect cancellation passing.

## Deferred and intentional-scope decisions

### DD-022 — Keep process-global attention contexts as a single-engine constraint

- **Decision:** retain `_BATCH_CTX` and `_PREFILL_CTX` for the current service.
- **Why:** they are a minimal adapter to HF attention dispatch and are safe under the
  single GPU-worker invariant.
- **Tradeoff:** concurrent forwards from independent engines in one process are not a
  supported configuration.
- **Next evidence required:** per-engine/thread-local context design plus isolation tests.

### DD-023 — Do not integrate speculative decoding into the active serving path yet

- **Decision:** keep `engine/speculative/` and `engine/batching/batched_speculative.py`
  as experiments.
- **Why:** they use a separate cache/batching model and have not been reconciled with the
  paged continuous scheduler, prefix cache, graph buckets, or worker service contract.
- **Tradeoff:** the production path does not receive speculative-decoding speedups.
- **Next evidence required:** token-equivalent integration plus end-to-end serving A/B.

### DD-024 — Begin post-optimization work with reliability validation, not new kernels

- **Decision:** performance optimization is frozen after Phase 15. The next build phase
  is stress, cancellation, allocator-leak, failure-recovery, and API-contract validation.
- **Why:** current performance is profile-backed; the highest risk is now correctness under
  prolonged mixed traffic rather than an unmeasured micro-optimization.
- **Tradeoff:** no new headline throughput claim until a failing gate or profiler identifies
  a real bottleneck.
- **Next evidence required:** deterministic mixed-arrival stress harness using the real
  engine, terminal-state accounting, allocator/refcount invariants, and repeatable Colab
  results.

### DD-025 — Isolate request failures from worker failures

- **Decision:** Gate 1 introduces typed submission errors, terminal-status mapping,
  liveness/readiness probes, worker-routed drain/shutdown, and a CPU fake-engine chaos
  suite. A rejected, cancelled, timed-out, or capacity-failed request becomes observable
  to its own caller without taking down unrelated requests or the worker.
- **Why:** a production server must survive malformed input, queue pressure, disconnects,
  and a request-local resource failure. Only an actual engine/worker exception should mark
  the service unavailable.
- **Code:** `engine/server/api.py`, `engine/server/continuous.py`,
  `tests/server/test_chaos.py`.
- **Tradeoff:** the API now has a deliberate status contract and additional lifecycle
  state; callers must distinguish readiness from liveness.
- **Evidence:** local CPU suites pass. Real uvicorn/T4 acceptance remains required.

### DD-026 — Defend paged K/V writes at both host and device boundaries

- **Decision:** Gate 1 validates decode-page capacity before metadata staging and adds
  masked page-table/destination bounds checks to FP16 K/V writers.
- **Why:** a violated page-table invariant must never become an out-of-bounds GPU write.
- **Tradeoff:** the device mask is a last-resort memory-safety guard, not a substitute for
  host-side lifecycle validation. INT8 paths require equivalent coverage before the gate
  can be considered fully complete.
- **Code:** `engine/batching/continuous_batching.py`, `engine/kernels/kv_write.py`.
- **Evidence:** local tests and the FP16 T4 D6 gate pass. INT8 writers remain an explicit
  follow-up because this decision currently protects the active FP16 serving path only.

### DD-027 — Evaluate recompute preemption under real GPU pressure before accepting it

- **Decision:** Gate 1 adds and accepts a bounded newest-active-request preemption mechanism: release
  a victim's KV pages, retain its generated token history, requeue it, and rebuild its KV
  state later. It is provisional until CUDA tests prove token identity and convergence.
- **Why:** temporary KV pressure should not automatically turn into request failure when a
  newer request can yield cache pages and be recomputed later.
- **Tradeoff:** recomputation adds GPU work and latency, can thrash without a bound, and
  complicates prefix/cache/graph interactions. The policy is FCFS-priority preserving,
  newest-active-victim preemption, not a generic fairness guarantee.
- **Code:** `engine/runtime/request.py`, `engine/scheduler/scheduler.py`,
  `engine/batching/continuous_batching.py`, `tests/scheduler/test_preemption.py`.
- **Evidence:** CPU lifecycle tests pass and all three real Qwen/T4 D6 pressure tests
  passed: token-identical recomputation, admission rejection of an impossible request,
  and prefix reattachment during resumption. The original GPU run exposed and corrected
  a resumption-accounting ordering bug, now covered by a CPU regression test.

### DD-028 — Gate recompute readmission on a progress epoch, not a preemption count

- **Decision:** accept *strict-LIFO recompute preemption with progress-gated readmission*
  as the fairness policy. Six clauses, stated in full on `FCFSScheduler`: victims strictly
  by arrival (newest first); a requester that is itself the newest yields itself, and fails
  only when it is the sole active request; a yielded request is not re-admitted until
  `progress_epoch` advances past its yield epoch, with the gate skipped when nothing is
  active; no preemption-count limit; admission checks capacity minus permanently reserved
  blocks; yielded requests requeue in arrival order.
- **Why:** termination becomes structural rather than heuristic. The oldest active request
  is never displaced, so it always completes; every completion advances the epoch; every
  yield strictly shrinks the active set. The epoch gate also removes the dominant waste in
  the previous design, where a yielded request was requeued at the head and rebuilt its KV
  immediately even though nothing had been released in between.
- **Rejected — a preemption-count limit (`MAX_PREEMPTIONS_PER_REQUEST`).** Set to 8 in the
  first Gate 1 patch, raised to 64 when the D6 workload tripped it. A count is the wrong
  criterion: a healthy request queued behind several long generations yields once per
  iteration through no fault of its own, so the bound scales with peers' remaining
  generation length and any fixed value eventually fails valid work under load. Removed.
- **Rejected — copy-out/swap preemption.** Copying a victim's KV to host memory instead of
  discarding it would avoid rebuild work, but Gate 1 proved recompute is token-identical
  and the measured rebuild cost is ~37 ms per yield on the T4, well below the ~131–664 ms
  a yielded request already spends queued. Swap adds PCIe traffic and a second memory pool
  for no measured benefit at this scale. Revisit only when long prompts make the
  rebuilt-tokens-per-output-token ratio large.
- **Code:** `engine/scheduler/scheduler.py`, `engine/runtime/request.py`,
  `engine/batching/continuous_batching.py`.
- **Tradeoff:** the gate converts avoided GPU waste into queue latency. That is the right
  trade here — see the measurement in the optimization journal — but it makes preemption a
  latency event, so per-request deadlines (Gate 3) must account for it.
- **Evidence:** 14 CPU policy tests, one per clause, plus five Qwen/T4 interaction tests
  covering mixed lengths, CUDA-graph buckets, INT8 KV, cancellation of a yielded request,
  and cost measurement. All pass at `0b20313`.

### DD-029 — Size pressure tests from real tokenization, never from hard-coded pools

- **Decision:** pressure tests derive their KV pool from the actual tokenized workload
  (`_pressure_blocks`): 60% of the blocks all requests need at peak, floored at the largest
  single request so admission cannot reject it, with an assertion that the workload is
  squeezable at all.
- **Why:** the first version hard-coded a 12-block pool after checking that each request
  fits *alone*, never that the four together exceed it. At 16 tokens per page the four
  prompts at `max_new=32` need exactly 12 pages, so three tests ran to completion without
  a single preemption and asserted nothing. A pool that merely looks tight silently stops
  testing anything the next time a prompt or the tokenizer changes.
- **Code:** `tests/batching/test_continuous_batching.py`.
- **Tradeoff:** test setup is computed rather than literal, so a reader must run the helper
  to know the pool size; the printed recompute report compensates.
- **Evidence:** identical suite, same engine code — three failures before, five passes
  after. Two tests passed throughout only by accident: the CUDA-graph test because its
  three reserved dummy pages left 11 usable, and the cancellation test because
  `max_new=64` needed 20 pages.


### DD-030 — Report queueing, stalls, and decode time separately once preemption exists

- **Decision:** a request reports `queue_ms` (to first admission) and `total_queue_ms`
  (including post-yield waits); `generation_ms` (wall clock) and `decode_ms` (stalls
  removed); plus `stall_ms` and a mean inter-token latency that excludes the preemption
  gap. `latency_report()` returns the set; the server merges it with
  `recompute_overhead()`.
- **Why:** Gate 1 made `admitted_ns` record only the first admission, so `queue_ms` stopped
  counting a preempted request's second wait, while `generation_ms` silently absorbed it -
  d6-3 spent 664 ms parked, invisible in one metric and hidden inside the other. Any
  percentile computed from those numbers, including the soak harness's, would have been
  wrong in both directions at once.
- **Code:** `engine/runtime/request.py`, `engine/server/api.py`.
- **Tradeoff:** more fields to carry, and two defensible answers to "how long did
  generation take". The wall-clock number stays the default because it is what the client
  experienced; `decode_ms` exists for engine-efficiency comparisons.
- **Evidence:** CPU tests construct a real preempt/rebuild cycle and assert the stall
  appears in `total_queue_ms` and `stall_ms`, is absent from `decode_ms`, and does not
  dominate the reported mean ITL.

### DD-031 — Publish engine stats from the worker; never read live scheduler state cross-thread

- **Decision:** `ContinuousBatchingEngine.stats_snapshot()` returns plain ints and floats
  and is called only by the GPU worker, at most every 100 ms, into a lock-guarded dict that
  `ContinuousBatchingService.snapshot()` merges for `/health` and `/ready`. A failing
  snapshot is swallowed.
- **Why:** the worker mutates `scheduler.waiting`, `scheduler.active`, the block manager
  and the prefix cache on every step. An HTTP handler calling `scheduler.snapshot()`
  directly would iterate those containers from another thread and can raise mid-iteration.
  Rate-limiting matters too: `recompute_report()` walks active plus waiting, which is
  per-step work on a path we know is host-overhead-bound.
- **Rejected — locking the scheduler.** A lock around scheduler mutation would put metrics
  in the decode path's critical section to serve a monitoring endpoint. The publish model
  costs one dict copy per 100 ms instead.
- **Code:** `engine/batching/continuous_batching.py`, `engine/server/continuous.py`.
- **Tradeoff:** reported stats can be up to 100 ms stale, which is irrelevant for scraping
  and preferable to a metrics call that can crash a serving thread.
- **Evidence:** server tests assert stats reach `/health`, and that an exploding
  `stats_snapshot` leaves the worker alive and generation unaffected.


## Recording rule

When a future change affects a kernel, cache layout, scheduler policy, service contract,
default configuration, supported model family, or measured deployment recommendation, add
or update a `DD-*` record in the same commit. A benchmark alone is not a design decision;
the record must state whether the implementation is accepted, optional, rejected, or
deferred.

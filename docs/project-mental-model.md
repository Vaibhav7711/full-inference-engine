# Full Inference Engine: project status and mental model

**Repository examined:** `Vaibhav7711/full-inference-engine` (the configured `origin` remote)  
**Local revision:** `ae1eb719a1d20c16bb6d5dcf0111f5a2c6a38c13` on `t4-phase0`  
**Revision date:** 2026-09-21 12:32 +05:30  
**Assessment date:** 2026-09-21  

## Executive summary

This is a serious, correctness-first research implementation of a **single-GPU,
decoder-only LLM inference runtime**, specialised for Qwen and a Tesla T4.  It is not
a general replacement for vLLM and it does not reimplement a transformer from
scratch.  Hugging Face retains model loading, model topology, projections, MLPs,
tokenization, and logits.  This repository takes ownership of the part that turns a
model into a serving engine: request lifetime, scheduling, cache ownership, batched
prefill/decode, GPU metadata staging, selected Triton kernels, CUDA Graph replay,
streaming transport, measurements, and reliability checks.

The central design has changed materially since the older README wording which says
that physical attention still uses Hugging Face cache tensors.  The current serving
path has persistent, per-layer paged K/V pools, direct Triton K/V writes, and a
batched paged decode kernel.  `docs/design-decisions.md`, the integration tests, and
the current `ContinuousBatchingEngine` agree on that point.  The README remains useful
as an introduction but is not the authoritative status document; the checkpoint and
recent code are.

The project is best understood as two intentionally separate products:

1. **A transparent reference path** (`ExplicitDecodeRunner`) that demonstrates the
   standard Hugging Face cache-backed token recurrence and provides an output oracle.
2. **The serving path** (`ContinuousBatchingEngine` plus `ContinuousBatchingService`)
   which manages many requests in a shared paged-KV pool and batches the GPU work.

The project has meaningful measured results and disciplined retractions.  Its most
important remaining performance problem is long/chunked prefill, not decode.  Its
immediate status is “advanced experimental engine with a working server surface and
accepted reliability gates,” rather than “production-complete serving system.”

## Evidence and repository health

### What was inspected

The local repository contains 189 Git-tracked files and 233 commits since 2026-09-01.
The codebase is concentrated in 8,399 Python lines under `engine/`, with 5,446 test
lines and 7,731 benchmark lines.  The largest implementation unit is
`engine/batching/continuous_batching.py` (1,623 lines); its size is a useful signal
that the engine’s orchestration, rather than an isolated kernel, is the architectural
centre of gravity.

The configured GitHub remote is
`https://github.com/Vaibhav7711/full-inference-engine.git`.  The remote GitHub page
could not be fetched from this environment at assessment time, so public GitHub
metadata such as stars, forks, issues, CI state, and the remote default branch were
not asserted.  The repository’s local Git history and remote configuration were
available and are the basis for this report.

The working tree was otherwise clean except for the untracked archive `fol.zip` that
predated this report.  This document is the only assessment artifact added.

### Test evidence, and an important limitation

The repository includes a broad test suite: lifecycle/scheduler/cache/kernel/server/
reliability/quantization tests, plus CUDA-marked end-to-end tests.  The project’s own
optimization journal records these most recent historical gates:

| Recorded gate | Latest recorded outcome |
| --- | --- |
| CPU reliability gate | 100 passed, 8 CUDA tests skipped |
| Full local suite | 126 passed, 116 skipped |
| Gate 1 input survivability | accepted: 14 CUDA, 19 kernel, CPU suite green |
| Gate 1B recompute preemption | accepted: 5/5 interaction tests, token-identical under pressure |
| Reliability soak | accepted: six soaks, zero invariant violations, peak KV accounting 1.00 |

I attempted a fresh `pytest -q` run.  It stopped during collection because the shell
environment has `pytest` but no `python` command and that `pytest` environment could
not import `engine`.  That is an **environment/packaging verification failure**, not
evidence that project tests fail.  No fresh test-pass claim is made here.  A meaningful
re-run requires activating the intended Python environment (or using `python3 -m
pytest` after installing the project and dependencies) and, for CUDA tests, access to
a supported GPU and the Qwen checkpoint.

### Measured performance: what can currently be claimed

The authoritative checkpoint reports Qwen3-0.6B FP16 on a Kaggle Tesla T4, warmed
engines, concurrency 8, five interleaved 30-second closed-loop runs.  These measurements
are device- and workload-specific, not general hardware claims.

| Metric | Chat prompts (~656 tokens) | Long prompts (~1.8k tokens) |
| --- | ---: | ---: |
| Decode-only step p50, CUDA graphs on | 10.3 ms | 10.4 ms |
| Host metadata staging per step | 0.26 ms | 0.27 ms |
| Paged-SDPA prefill step p50 (default) | 27.0 ms | 30.3 ms |
| Per-token prefill p50 | 45.3 ms | 87.9 ms |
| TTFT p50, default SDPA prefill | 414 ms | 3.7 s |
| ITL p99, default SDPA prefill | 41 ms | 76 ms |

The roofline probe measured 258.8 GB/s of FP16 GEMV bandwidth (81% of the T4’s quoted
320 GB/s) and derives an approximately 5.02 ms decode-floor at batch ~7 and 128-token
context.  Decode at 8.05–8.20 ms in the earlier closed-loop measurement is thus about
1.64x the measured floor.  CUDA graphs reduced decode step time from 34.2 ms to 9.7 ms
in the current re-baseline (about 72%).

The `results/` directory also preserves earlier, simpler measurements.  For a 32-token
generation, the historical DynamicCache baseline was 23.89 tokens/s; an earlier
reference paged-cache path was slower (21.52 tokens/s at 16-token pages), and a
reference page-gather read path was much slower (14.25 tokens/s).  These are useful
historical baselines, not measurements of the current direct-write/decode-kernel
serving path.  The current results file reports 153.7 aggregate tokens/s at concurrency
16 versus 23.6 sequentially for that workload.

### Current capability ledger

| Area | State | Meaning |
| --- | --- | --- |
| Explicit Qwen prefill/decode | Complete | Clear single-request reference recurrence and HF-greedy oracle. |
| Persistent paged FP16 KV | Implemented and tested | Per-layer pools, physical block tables, direct writers, paged batched decode. |
| Dynamic continuous batching | Implemented | Decode-first steps, bounded prefill work, request admission/exit. |
| Prefix reuse | Implemented | Block-aligned radix cache, exact prompt entries, refcounts, LRU-like eviction and copy-on-write tail. |
| CUDA graph buckets | Implemented and measured | Fixed padded batch buckets with isolated dummy KV pages. |
| Qwen pointwise Triton fusion | Implemented | RMSNorm, RoPE, and SwiGLU are switchable A/B arms. |
| Paged SDPA prefill | Current default | Gathers page history then uses PyTorch SDPA; correct and faster than the T4 tiled kernel. |
| INT8 paged KV | Opt-in experimental | Kernel-native implementation; no default e2e win for recorded workloads. |
| Weight-only INT8 / W8A8 linears | Experimental/rejected for T4 | Correctness/reference work exists, but T4 results are substantially slower than FP16 CUTLASS. |
| Speculative decoding | Experimental, disabled | Vanilla draft/verify/rollback implementation exists; negative T4 result. |
| Multi-GPU, model-generic, vLLM parity | Out of scope/not complete | Project target remains one Qwen-oriented GPU engine. |
| Full load test, cold-start/packaging, vLLM/L4 comparison | Not complete | Explicitly listed in checkpoint plan. |

## The conceptual model: autoregressive inference

A decoder-only model emits one token at a time.  Given prompt tokens `x[0..P-1]`, a
prefill forward computes logits for every prompt position and stores each layer’s key
and value (K/V) tensors.  The last prefill logit predicts the first generated token.
Each later decode forward consumes one chosen token, reads all prior K/V, writes one
new K/V position, and predicts the next token.

```text
prompt tokens ──prefill──> prompt K/V + logits ──argmax/sample──> token 1
                                                            │
     token t + all committed K/V ──decode──> new K/V + logits ─┘
```

The K/V cache prevents re-running the whole prompt on every output token.  It does not
avoid the model’s weights: every decode step still traverses every layer and is commonly
weight-bandwidth dominated for this small model.

For this model family, cache memory per sequence token is:

```text
2 (K and V) × layer_count × KV_head_count × head_dimension × bytes_per_element
```

For the documented Qwen3-0.6B BF16 geometry (28 layers, 8 KV heads, 128 dimensions,
two-byte elements), that is 114,688 bytes = 112 KiB/token.  Qwen’s grouped-query
attention maps several query heads to one KV head, reducing K/V memory relative to
full multi-head attention.  The serving engine tracks K/V capacity independently from
model-weight memory.

## Architecture at a glance

```text
HTTP client
  │ POST /generate or /generate/stream (SSE)
  v
FastAPI application ──> ContinuousBatchingService
                            │ thread-safe inbox/cancel queue; one GPU worker
                            v
                      ContinuousBatchingEngine
                        ├─ GenerationRequest state machines
                        ├─ FCFS scheduler + preemption policy
                        ├─ KVBlockManager + PrefixCache (CPU ownership plane)
                        ├─ persistent GPU K/V pools (data plane)
                        ├─ staged metadata buffers and graph caches
                        └─ Qwen model with selected attention/fusion hooks
                                  │
                       Triton K/V writers / paged decode
                       or gathered-page SDPA prefill
                                  │
                         selected tokens → request handles → SSE/JSON
```

There are two planes, and keeping them separate is a key to understanding the code:

* The **control/ownership plane** is mostly Python/CPU state: request states, queues,
  token lists, logical sequence lengths, physical page IDs, refcounts, and timers.
* The **data/execution plane** is GPU state: fixed K/V tensors, persistent metadata
  buffers, model weights, graph-capture buffers, and Triton/PyTorch kernels.

The block table bridges the two.  For logical token position `i`, `i // block_size`
selects a page ID and `i % block_size` selects the token slot inside that physical page.
No request needs a contiguous physical history.

## Component-by-component role map

### Model and reference runtime

`engine/model/loader.py` loads a Hugging Face checkpoint, selects a safe dtype for the
device, chooses CUDA, and ties output embeddings when appropriate.  The target is
Qwen3-0.6B by default, with FP16 chosen for the T4 because it has native FP16 Tensor
Cores rather than native BF16 Tensor Cores.

`engine/model/runner.py` is deliberately small and explicit:

* `prefill(input_ids, attention_mask)` forwards the entire prompt with `use_cache=True`.
* `decode_one(token, state)` forwards a `[batch, 1]` token and the returned cache.
* `generate()` loops: choose the last logit’s greedy token, stop at EOS or a maximum,
  otherwise decode it.
* `stream_generate()` exposes the same recurrence event by event.

This implementation uses Hugging Face `DynamicCache`, which grows tensors by
concatenation.  It is the transparency and correctness boundary, not the high-throughput
server implementation.

### Request state machine

`engine/runtime/request.py` represents one generation using `GenerationRequest` and
`RequestState`.  Its normal lifecycle is:

```text
WAITING → PREFILLING → DECODING → FINISHED
                │          │
                └──────────┴─→ CANCELLED / FAILED / REJECTED
```

The request owns prompt IDs, generated IDs, `next_token_id`, output limit, allocation
pointer, admission/resumption/preemption timestamps, and detailed latency accounting.
It separates queue time, time to first token, generation time, stalls, decode time, and
inter-token latency.  This matters: an end-to-end latency number alone cannot identify
whether a slowdown arose from queueing, prefill, decode, or a preempted recomputation.

### Scheduling and preemption

`engine/scheduler/scheduler.py` supplies FCFS admission.  It owns waiting/active sets
and asks the block manager whether a request’s planned K/V reservation can fit.  A
request’s reservation includes its prompt and maximum output because a cache failure
mid-generation is worse than admitting fewer requests upfront.

The engine schedules **decode first**: currently decoding users are advanced before
new prompt work.  It then takes bounded, round-robin prefill chunks up to
`max_prefill_tokens_per_iteration`.  This prevents long prefills from indefinitely
blocking existing streams, while deliberately accepting that prefill-bearing steps
increase inter-token latency.

Under page pressure it can use recompute preemption: an eligible active request yields,
releases pages, and later rebuilds K/V by re-prefilling its retained tokens.  The policy
is strict LIFO for reclamation and gates readmission on a progress epoch rather than an
arbitrary maximum preemption count.  This structural rule avoids pathological starvation
from a count-based cutoff.  It is a resilience/fairness mechanism, not a free
performance optimization: recorded rebuild time is 37.53 ms, with queue wait often
dominant.

### Paged cache ownership

The cache package has three useful levels that should not be conflated:

1. `kv_cache.py` is the analytical geometry/memory calculator.
2. `allocator.py` contains both a contiguous first-fit allocator for experiments and
   the real fixed-block allocator with free lists and reference counts.
3. `paging.py` is the real logical mapping manager.  `KVBlockManager` reserves,
   extends, attaches, copy-on-writes, commits logical length, and releases mappings.

`ContinuousBatchingEngine` allocates one zero-filled K pool and V pool per transformer
layer.  The FP16 layout is:

```text
[physical_page, offset_within_16-token_page, KV_head, head_dimension]
```

Those tensors live for the engine lifetime.  Each request merely owns an ordered Python
list of physical page IDs plus a committed sequence length.  The sequence length is
only committed after a successful forward; during a decode forward the new K/V can be
written at the pending slot but readers use the previous committed length plus an
offset.  That ordering is the engine’s main failure-consistency boundary.

`paged_cache.py` and `paged_attention.py` are important reference/compatibility
implementations: they make dynamically-grown paged cache layers and gather pages for
verification.  They demonstrate semantics but are not the authoritative persistent-pool
data path used by continuous serving.

### Prefix cache and sharing

`engine/cache/prefix.py` caches immutable, complete K/V pages in a radix tree keyed by
whole token blocks.  A lookup finds the longest block-aligned common prefix, always
leaving a token where required for correct next-token prediction.  It also maintains
exact-prompt entries containing the next token, allowing an exact hit to avoid even the
last prompt forward.  Physical pages are shared by reference count; eviction follows
access recency but only reclaims pages whose remaining owners allow it.

The difficult edge case is a shared partial tail.  Before a request writes into unused
slots of that page, `_ensure_writable_tail()` performs copy-on-write across every
layer’s K/V pools.  Without this, two logically unrelated continuations could overwrite
each other’s prefix cache state.

### GPU execution and custom kernels

The engine patches Qwen attention dispatch through Transformers’ attention-function
registry, selecting one of three contexts:

* **Batched decode:** `paged_decode_batched.py` reads logically noncontiguous K/V
  directly from pools.  Its grid is approximately `(active_sequences, query_heads)`;
  each program maps a query head to the proper GQA KV head, walks positions through the
  block table, and performs online softmax in FP32.
* **Chunked prefill:** `kv_write.py` stores each chunk’s post-RoPE K/V directly to its
  absolute paged locations.  The default attention path (`sdpa_prefill.py`) gathers the
  relevant pages to dense logical K/V and uses PyTorch scaled-dot-product attention with
  a causal mask.  It caches mask/page-index metadata across layers for one step.
* **Fused decode+prefill step:** a packed forward handles decode rows first and prefill
  rows second, so weights are read once rather than in two serial model forwards.

`tiled_paged_prefill.py` is a Triton tiled causal prefill experiment.  It is retained
behind a switch, but retired as default on T4 after PTX inspection showed that the T4
path did not use tensor-core `mma.sync`, had high register pressure, and ran one block
per SM.  This is exemplary project discipline: it preserves a potentially useful
sm_80+ candidate without calling it a win on sm_75.

`rmsnorm.py`, `rope.py`, and `swiglu.py` replace selected Qwen pointwise operations
with Triton kernels.  They are reversible toggles so A/B experiments do not accidentally
inherit model-object patches from an earlier arm.  `w8a16_linear.py` and
`w8a8_linear.py` are experimental linear paths; on T4 they lost decisively to mature
FP16 GEMMs and should not be enabled as an optimization claim.

### CUDA Graphs and persistent metadata

CUDA Graph replay requires stable tensor addresses, shapes, and control flow.  Dynamic
batching violates those conditions unless it is transformed into a small number of
fixed shapes.  The engine therefore uses configured batch buckets (typically powers
of two).  A live batch is padded to a bucket with inert dummy rows; permanent dummy KV
pages make those rows memory-safe and isolate them from customer pages.

Every decode step stages token IDs, positions, sequence lengths, and complete-width
block tables from pinned host tensors into persistent GPU tensors.  It copies an entire
row of block-table capacity even if most entries are logically dead.  This is an
intentional trade: fixed contiguous buffers make DMA and graph capture possible, while
the kernel masks table entries beyond each sequence’s valid length.  The recorded
two-row, 128-block example transfers 1,064 bytes per step.

Prefill and fused steps have analogous stable staging buffers and graph caches keyed by
their bucket/context shape.  Capture can fail for a particular attention implementation;
the engine records the unsupported reason and falls back to eager execution rather
than silently treating eager execution as graph replay.

### Server and transport boundary

`engine/server/api.py` exposes `POST /generate` (complete JSON) and
`POST /generate/stream` (SSE).  It validates prompt and output limits and converts
terminal request states into appropriate HTTP responses.

`engine/server/continuous.py` is intentionally the concurrency firewall.  FastAPI
handlers do not manipulate scheduler or K/V state.  They submit a `RequestHandle` to
a bounded inbox.  Exactly one background worker owns the engine/GPU state, drains
submissions, routes cancellations, repeatedly calls `engine.step()`, publishes
completed tokens, and emits a safe statistics snapshot.  Queue saturation becomes a
429 rather than unbounded memory growth.  Disconnects/timeouts enqueue cancellation
for the same worker, which prevents races between HTTP threads and allocator ownership.

## The important execution flows

### 1. Single-request reference generation

```text
text → tokenizer → input IDs
     → ExplicitDecodeRunner.prefill
     → HF model + DynamicCache → last prompt logit
     → choose token
     → repeated decode_one(token, cache) → next logit/cache
     → EOS or length limit → result
```

Use this flow when learning the mathematical recurrence, debugging model correctness,
or comparing a serving-path output to an oracle.  Do not use it to judge dynamic
batching throughput.

### 2. HTTP request to first token

```text
HTTP handler validates request
 → service.submit() creates handle and queues immutable token IDs
 → GPU worker turns item into GenerationRequest
 → scheduler reserves/attaches K/V blocks and admits it
 → engine plans a bounded prefill chunk (or finds a prefix hit)
 → metadata is staged; Qwen forward writes paged K/V and attends
 → successful forward commits logical prefill length
 → last prefill logit selects first generated token
 → request enters DECODING; worker publishes token to handle/SSE
```

The exact-prefix fast path can provide an already-known next token from cache metadata;
the normal prefix path instead attaches full shared blocks, copy-on-writes if needed,
and pre-fills only the remaining suffix.

### 3. One continuous-batching step

```text
engine.step()
  1. Apply worker-routed cancellation requests.
  2. Scheduler admits capacity-feasible waiting requests.
  3. Collect live decode rows; ensure one writable KV slot per row.
  4. Plan round-robin prefill chunks within the iteration budget.
  5. If both kinds exist and eligible, pack them into one fused forward;
     otherwise run decode and/or prefill forwards.
  6. Each Qwen layer executes hooked attention:
       K/V writer → paged decode kernel OR gathered-page SDPA prefill.
  7. Synchronize only where selected IDs must return to host; choose tokens.
  8. Commit K/V logical lengths only after successful forward.
  9. Transition EOS/limit rows to FINISHED, publish/release their ownership;
     continue surviving rows on the next step.
```

This is the project’s core loop.  “Continuous batching” here means that a request can
join after another starts and leave immediately when finished; it is not just a static
tensor batch.  It is deliberately decode-first, so users already receiving a stream
are not held behind unlimited newly arriving prompt work.

### 4. Capacity pressure, cancellation, and cleanup

```text
not enough free pages
  → evict reclaimable prefix entries first
  → if policy allows, choose latest eligible active request to preempt
  → release its mapping, preserve token history, requeue behind progress gate
  → later re-prefill it to rebuild K/V

terminal request / cancellation / error
  → scheduler marks terminal state
  → block manager releases customer references
  → prefix-owned references remain only where explicitly published
  → worker signals handle completion
```

The reliability subsystem audits the allocator after soaks: each page must be owned,
free, or accounted for by prefix/graph infrastructure exactly as expected.  This is a
stronger invariant than merely checking whether a process avoided an OOM.

## Observability, benchmarks, and how to read claims

`engine/metrics/metrics.py` provides CUDA-event timing and `stats.py` computes
percentiles.  `GenerationRequest` supplies user-facing lifetime metrics.  The engine
also records step decomposition when instrumentation is enabled: host staging, decode
GPU time, prefill GPU time, and sampling synchronization.

The benchmark suite is unusually large and should be treated as part of the project,
not peripheral scripts:

* `benchmarks/inference/`: single-request baseline, trace, and graph experiments.
* `benchmarks/cache/`: geometry, allocator fragmentation, reference paging, memory.
* `benchmarks/batching/`: static, continuous, mixed-arrival, chunk/budget, prefix and
  graph-bucket workloads.
* `benchmarks/kernels/`: roofline and isolated K/V, decode, prefill, and linear kernels.
* `benchmarks/reliability/`: soak, sweeps, pressure, A/B control.
* `benchmarks/server/`: real uvicorn/SSE burst-load checks.
* `benchmarks/understanding/`: executable traces that teach cache/page/graph/lifecycle
  mechanics with real model state.

The methodology recorded in `docs/checkpoint.md` and `docs/optimization-journal.md`
is a major strength: repeat arms, change one variable, warm comparably, use
closed-loop latency loads, report spread, and mark results unresolved inside noise.
The journal explicitly retracts flawed earlier claims rather than averaging them into
a narrative.  When extending the project, use `expected_gap_ms` for scheduling changes
that alter the mix of decode and prefill work; a median token-gap alone can hide that
mix shift.

## What is deliberately not solved

The following boundaries are intentional and should guide expectations:

* One process-global attention context means one serving engine/model per process;
  concurrent independent engines are not a supported topology.
* The custom integration is Qwen/Transformers-version aware, not a universal model ABI.
* Paged prefill still gathers history to dense K/V for the default SDPA path.  It avoids
  DynamicCache growth but is not a fully direct-paged prefill kernel.
* Padded CUDA graph buckets trade some dummy-row/page waste for lower launch overhead.
* INT8 K/V and speculative decoding are experiments, not serving defaults.
* This is not yet hardened for every operational concern: checkpoint comparison against
  vLLM on T4/L4, packaging/cold start, broader long-context/INT8/EOS testing, and full
  load testing remain planned.

## Suggested reading order for a newcomer

1. Read `README.md` for scope and the basic prefill/decode vocabulary, then this file
   for corrected current status.
2. Read `engine/model/runner.py` and `engine/runtime/request.py`.  At this point, the
   recurrence and state machine should be clear.
3. Read `engine/cache/kv_cache.py`, `allocator.py`, and `paging.py`; draw a logical
   token index to `(page, offset)` mapping on paper.
4. Read `engine/scheduler/scheduler.py`, then the `step`, `prefill_chunks`,
   `_decode_rows`, and fused-step sections of `continuous_batching.py`.
5. Read `engine/kernels/kv_write.py`, `paged_decode_batched.py`, and
   `sdpa_prefill.py` to connect control metadata to GPU data access.
6. Read `engine/server/continuous.py` and `api.py` to understand the safe threading
   boundary.
7. Use `benchmarks/understanding/real_request_lifecycle_trace.py`,
   `real_decode_step_trace.py`, and `real_prefix_cache_trace.py` on a configured GPU.
8. Finally read `docs/design-decisions.md`, `docs/checkpoint.md`, and the relevant
   sections of `docs/optimization-journal.md` before changing a default.

## Recommended next work

The code’s own checkpoint sequence is sensible.  Do not jump to a fashionable new
kernel before closing the evidence gaps.

1. Re-establish a reproducible local/CI test environment and record a fresh CPU gate;
   separately run the CUDA suite on the target T4/Kaggle image.
2. Measure the newly built fused decode+prefill step (`fused_step=False` versus true)
   under the stated interleaved, closed-loop Kaggle protocol.  It is on by default but
   explicitly marked “built, unmeasured” in the checkpoint.
3. Sweep prefill budget independently of chunk size.  The known bottleneck is the
   default gathered-page SDPA prefill path; smaller chunks were shown to worsen the
   work mix for prior workloads.
4. Expand correctness/reliability testing across long contexts, EOS, resets, INT8,
   and failure/transport conditions before broadening the serving claim.
5. Complete the declared vLLM T4/L4 comparison and packaging/cold-start work before
   calling the system production-hardened.

## Source-of-truth hierarchy

When project documents disagree, use this order:

1. Current code plus focused tests for actual behaviour.
2. `docs/design-decisions.md` for accepted boundaries and rationale.
3. `docs/checkpoint.md` for current validated/retracted performance claims and next gates.
4. `docs/optimization-journal.md` for experimental provenance and historical detail.
5. `README.md` for onboarding, with awareness that some status prose predates the
   direct paged-KV implementation.

That hierarchy prevents two common misunderstandings: treating reference cache
experiments as the live serving path, and treating an unmeasured/experimental feature
as an established performance result.

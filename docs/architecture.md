# Architecture: how this engine works, file by file

Written for: the project's own author, coming back after the T4 measurement sessions and
needing the full mental model again - what every file does, what every flow looks like,
what the kernels actually compute, and which libraries carry which part. Read it top to
bottom once; afterwards the "Repository map" and "Flows" sections are the lookup tables.

Everything here describes the code at `main` as of 2026-09-21. Where a component is a
baseline, parked, or historical, it says so.

---

## 1. The problem, in one page

A decoder-only transformer generates text one token at a time. For each new token the
model runs a forward pass over the *new* token only, attending to the keys and values
(K, V) of every previous token. Those K/V vectors are cached so they are computed once -
the **KV cache**. Two phases follow from this:

- **Prefill**: the prompt's tokens go through the model in one forward pass (many tokens,
  one step). Compute-bound: `tokens x parameters` FLOPs, and attention is quadratic in
  prompt length. Produces the first output token and fills the KV cache.
- **Decode**: one token per step per sequence. Memory-bound: every step reads *all* the
  model's weights (1.2 GB for Qwen3-0.6B in fp16) and the sequence's whole KV cache, to
  produce one token. On a T4 (~260 GB/s achieved) the weight read alone is ~5 ms, so one
  sequence can never exceed ~200 tokens/s no matter how good the code is.

**Continuous batching** is the answer to the decode floor: run N sequences' single new
tokens through one forward, so the 5 ms weight read is paid once for N tokens. The
sequences have different lengths and start/finish at different times, so the batch
membership changes every step ("continuous") and each row needs its own KV cache of its
own length. That is what **paged KV** is for: the cache is a pool of fixed-size blocks
(16 tokens each) and each sequence owns a list of block ids (its **block table**), so
rows of any length share one pool with no fragmentation and no copying.

The rest of the engine exists to keep that batched decode step cheap and its latency
predictable while new requests keep arriving:

- **Chunked prefill**: a new 700-token prompt is not prefilled in one go (which would
  stall every decoding row for ~100 ms); it is admitted 128 tokens per step alongside
  the decode rows.
- **Fused step**: those 128 prefill tokens and the N decode tokens go through *one*
  forward, so the weights are read once for both.
- **CUDA graphs**: the ~1,100 kernel launches of a forward are recorded once per shape
  and replayed; the per-step CPU cost drops from ~25 ms to ~0.3 ms.
- **Scheduler**: FCFS admission bounded by KV capacity, preemption (LIFO) when the pool
  runs out, cancellation, and a prefill token budget per step.
- **Custom Triton kernels**: attention over paged KV (stock attention needs contiguous
  K/V), K/V writes into pages, and fused RMSNorm / RoPE / SwiGLU.

Measured end state on the T4, chat workload (see `docs/checkpoint.md`): decode step
10.2 ms at batch ~5, prefill-carrying step 20.5 ms, ITL p50 19.8 ms, p99 33 ms.

---

## 2. Libraries and what each one is used for

| library | what we take from it | where |
|---|---|---|
| **PyTorch** (`torch`) | tensors on the GPU; `F.scaled_dot_product_attention` (SDPA - fused attention with several backends: flash, memory-efficient, math); `torch.cuda.CUDAGraph` + `torch.cuda.graph()` for capture/replay; `torch.cuda.Event` for GPU timing; pinned host memory (`pin_memory=True`) and `non_blocking` copies; `torch.inference_mode()` | everywhere; graphs in `engine/graphs/`, SDPA in `engine/kernels/sdpa_prefill.py` and the fresh-prompt path |
| **transformers** | the model definition (`Qwen3ForCausalLM`), tokenizer, weight loading. Two hooks matter: (1) `ALL_ATTENTION_FUNCTIONS[name] = fn` registers a custom attention function, and `model.config._attn_implementation = name` makes every layer call it - this is how our paged attention replaces stock attention **without modifying the model**; (2) the model accepts `position_ids` per token and applies RoPE before calling attention, so our kernels receive already-rotated Q and K | `engine/batching/continuous_batching.py` (`ALL_ATTENTION_FUNCTIONS`), `engine/model/loader.py` |
| **Triton** | a Python DSL for writing GPU kernels. A `@triton.jit` function is compiled per set of `tl.constexpr` arguments. `tl.program_id(axis)` is the block index in a launch grid; each program works on tiles (`tl.arange`) with masked `tl.load`/`tl.store`. We never write CUDA C | `engine/kernels/*.py` |
| **FastAPI + uvicorn** | HTTP and server-sent events (SSE) streaming; async handlers hand work to one engine thread | `engine/server/api.py` |
| **pytest** | CPU tests always; `-m cuda` tests need a GPU and the checkpoint | `tests/` |

Model geometry the kernels assume (Qwen3-0.6B): 28 layers, hidden 1024, 16 query heads,
8 KV heads (GQA 2:1), head_dim 128, vocab 151,936, fp16. `head_dim <= 128` is baked into
several kernels.

---

## 3. Repository map

Status legend: **live** = on the serving path; **baseline** = kept for comparison;
**parked** = works, not integrated; **historical** = evidence for an earlier decision.

### `engine/` - the runtime

| file | status | what it is |
|---|---|---|
| `batching/continuous_batching.py` (1.7k lines) | live | **The engine.** `ContinuousBatchingEngine`: KV pools, persistent staging buffers, the three attention functions (decode / chunked prefill / fused), `step()`, `decode_step()`, `prefill_chunks()`, `prefill_batch()`, graph selection and capture, `warmup()`, stats. Section 5 walks it. |
| `batching/static.py` | baseline | Stage-9 fixed-membership batching (prefill all, then decode all, no admission mid-run). The "no continuous batching" comparison. |
| `batching/batched_speculative.py` | parked | N sequences speculating together in a padded batch; not wired to the scheduler. |
| `scheduler/scheduler.py` | live | `FCFSScheduler`: `waiting` deque, `active` dict, `submit`, `admit_available` (capacity check incl. prefix-cache hits), `plan_prefill` (round-robin chunks under a token budget), `preempt` (LIFO victim, requeue in arrival order), `cancel`, `finish/fail`, progress epoch. |
| `runtime/request.py` | live | `GenerationRequest` and `RequestState` (WAITING → PREFILLING → DECODING → FINISHED / CANCELLED / FAILED / REJECTED; PREFILLING/DECODING → WAITING is preemption). Holds prompt ids, `prefilled_token_count`, `allocation`, `next_token_id`, output ids, timing (`latency_report()`), recompute accounting. Transitions are validated. |
| `cache/paging.py` | live | `KVBlockManager` + `KVBlockAllocation`: block ownership per request (`reserve`, `ensure_capacity`, `append_tokens`, `release`, `attach_prefix`, `copy_on_write_tail`). The allocator underneath is refcounted so prefix-cache blocks can be shared. |
| `cache/prefix.py` | live (off by default) | `PrefixCache`: block-aligned radix tree + exact-match entries over the shared allocator; `lookup` returns reusable blocks, `publish` after prefill, LRU eviction with refcount awareness. |
| `cache/pool_cache.py` | live | `BatchedPoolBackedPrefillCache`: a transformers `Cache` adapter so the **fresh-prompt fast path** (stock SDPA over the whole prompt) writes K/V straight into our pages. |
| `cache/paged_cache.py`, `cache/paged_attention.py`, `cache/kv_cache.py`, `cache/allocator.py`, `cache/block_table.py` | baseline / historical | Single-request paged cache adapter (M1/M2 experiments), KV geometry formulas, contiguous first-fit allocator (the "before paging" baseline), block-table dataclass. Tests still cover them. |
| `graphs/paged_decode_graph.py` | live | Capture one decode forward at a fixed batch width and kernel regime; `replay()` returns logits. |
| `graphs/paged_prefill_graph.py` | live | Capture one chunked prefill forward at (rows, chunk width, gathered-context bucket). |
| `graphs/fused_step_graph.py` | live | Capture one fused forward at (decode rows, chunk rows, context bucket, regime). |
| `graphs/cuda_graphs.py` | historical | Stage-15 eligibility check and a contiguous-cache decode capture. |
| `kernels/paged_decode_batched.py` | live | **Decode attention** over paged KV, one program per (row, query head). Section 6. |
| `kernels/paged_decode_gqa.py` | baseline (negative result) | Same, one program per (row, KV head) sharing tiles across the GQA pair. Measured 0.95-1.06x; L2 already serves the second read. |
| `kernels/paged_decode_config.py` | live | Tile/warp regime by context length (64x4 below 128 tokens, 128x4 above). |
| `kernels/kv_write.py` | live | Two kernels: write one decode token's K/V per row into its page slot; write a padded prefill chunk per row starting at its position. |
| `kernels/sdpa_prefill.py` | live (default chunked prefill attention) | Gather each row's pages into a dense `[B, kv_heads, T, D]`, build the chunk causal mask once per step, fold the GQA pair into the query axis, call torch SDPA (memory-efficient backend, tensor cores). |
| `kernels/paged_prefill.py` | live (fallback, `prefill_attention="per_token"`) | Chunked prefill attention in Triton, one program per (row, head, query token); reads pages in place, no gather. Slower than SDPA on T4. |
| `kernels/tiled_paged_prefill.py` | parked for sm_80+ | FlashAttention-structured chunked prefill with `tl.dot`. On T4 `tl.dot` compiles to FMA (no tensor cores) and it is 3x slower; the PTX gate in `benchmarks/kernels/prefill_attention_ab.py --ptx-only` decides per GPU. |
| `kernels/rmsnorm.py`, `kernels/rope.py`, `kernels/swiglu.py` | live | Fused elementwise kernels plus *installers* that monkey-patch the Qwen3 modules (`install_triton_rmsnorm(model)`, `install_triton_qwen_rope()`, `install_triton_qwen_swiglu(model, fuse_gate_up=)`) and matching uninstallers. |
| `kernels/int8_paged_kv.py` | live (off by default, `kv_cache_dtype="int8"`) | INT8 K/V pages with per-token scales: write kernels, decode and per-token prefill attention that dequantize in-kernel. Negative/unresolved on T4. |
| `kernels/device.py` | live | `DeviceProfile` (SM version, memory, name) used for per-GPU defaults and result provenance. |
| `kernels/w8a16_linear.py`, `kernels/w8a8_linear.py` | parked | Weight-only INT8 and INT8xINT8 GEMMs; benchmarked in isolation, not in the serving path. Roadmap Tier 2. |
| `kernels/triton_attention.py`, `kernels/paged_attention_kernel.py`, `kernels/paged_kernel_attention.py` | historical (K1-K3) | The kernel ladder that led to K4 (`paged_decode_batched`): contiguous attention, then paged reads, then wired into live generation. Tests keep them honest. |
| `model/loader.py` | live | `load_model(name)` → fp16 on CUDA, tokenizer, `tie_output_embeddings` (transformers 5 refuses to tie when both tensors ship; we tie explicitly). |
| `model/runner.py` | baseline | Explicit single-request prefill/decode loop with a contiguous cache - the Stage-1 reference. |
| `server/continuous.py` | live | `ContinuousBatchingService`: one worker **thread** owns the engine; async handlers `submit()` into a bounded queue and get a `RequestHandle` with `on_accept` / `on_complete` callbacks; cancellations go through the worker; drain/stop semantics. |
| `server/api.py` | live | FastAPI app: `POST /generate`, `POST /generate/stream` (SSE), `GET /health`, `GET /ready`; status-code mapping for every terminal state; `default_engine_factory` builds the engine and runs `warmup()` before readiness. |
| `quantization/int8.py`, `quantization/kv_int8.py` | parked / live-off | Reference weight-only INT8 (`Int8Linear`), KV INT8 helpers. |
| `speculative/vanilla.py`, `speculative/optimized.py` | parked | Single-sequence draft-model speculative decoding with greedy acceptance. Not integrated with batching (roadmap Tier 2 rebuilds it inside the batched step). |
| `metrics/` | live | `cuda_timed` context manager, `percentile`, latency summaries used by benchmarks. |

### `benchmarks/` - measurement

| file | what it measures |
|---|---|
| `reliability/soak.py` | Closed-loop soak: random arrivals, cancellations from every state, invariants (no leaked pages, every request terminal), per-request latency, per-step phase split, `RepeatedResult` with median/min/max/spread. **The workload every A/B uses.** |
| `reliability/ab.py` | Interleaved multi-arm A/B over engine kwargs (`SETTINGS`), warmed engines, stock-reference token-identity gate (tie-aware), provenance, verdicts. `--setting`, `--prompt-profile chat|long`, `--cuda-graphs`. |
| `reliability/sweep.py` | Prefill chunk/budget sweeps fitted to `a + b * chunk`. |
| `kernels/paged_decode_regime_sweep.py` | Decode attention kernel in isolation across (context, batch, tile, warps, kernel). |
| `kernels/prefill_attention_ab.py` | Chunked prefill kernels vs SDPA bound; `--ptx-only` prints the compiled kernel's `mma.sync`/spill/register facts. |
| `kernels/roofline.py` | Measured bandwidth and the decode floor at an operating point. |
| `kernels/int8_paged_decode_ab.py`, `kernels/w8a16_linear_ab.py`, `kernels/w8a8_linear_ab.py` | INT8 kernel A/Bs. |
| `batching/continuous_throughput.py` | D4: continuous vs sequential throughput (the batching multiplier). |
| `batching/prefill_throughput.py`, `chunked_prefill_latency.py`, `prefix_cache_ttft.py`, `token_transfer_ab.py`, `mlp_gate_up_fusion_ab.py`, `profile_continuous_decode.py` | Component measurements referenced by the plan. |
| `batching/static.py` | Static-batching baseline runner. |
| `server/burst_load.py`, `uvicorn_load.py`, `continuous_api_smoke.py` | HTTP/SSE load, disconnects, smoke. |
| `quantization/*` | Weight-only INT8 memory/quality. |
| `understanding/*` | Trace scripts that print what one step / one request / one graph capture actually does. Learning aids; pair with `docs/understanding-journal.md`. |
| `common.py` | `git_record()`, `device_clock_record()`, `environment_record()` for provenance. |

### `scripts/`, `tests/`, `docs/`

- `scripts/check_hooks.py` - warmup captures every graph, phases report, zero in-service captures.
- `scripts/token_margins.py` - stock top-2 logit margins (tie diagnosis).
- `scripts/t4_phase0_phase1.ipynb` - the Kaggle two-GPU runner and every phase cell.
- `scripts/setup_kaggle.sh`, `setup_colab.sh`, `colab_preflight.py`, `verify.py`.
- `tests/` - mirrors `engine/`; CPU tests run anywhere (232), CUDA tests (`-m cuda`) run the real model and are the correctness gates.
- `docs/optimization-journal.md` - every result and every retraction, dated, with SHAs. **Source of truth for numbers.**
- `docs/checkpoint.md` - current claims table + retracted claims. `docs/t4-reevaluation-plan.md` - item ledger (D/P/M rows) with status. `docs/design-decisions.md` - why things are shaped as they are. `docs/understanding-journal.md` - the learning log.
- `results/t4/` - transcribed T4 numbers.

---

## 4. Data structures

### The KV pool
```
key_pool[layer]   : [num_blocks, block_size=16, kv_heads=8, head_dim=128]  fp16 (or int8)
value_pool[layer] : same
```
One pair per layer (28 pairs). Token `t` of a sequence lives in physical block
`block_table[t // 16]` at slot `t % 16`. A block is 16 x 8 x 128 x 2 B = 32 KB for K
plus 32 KB for V, per layer; a 1k-token sequence is 64 blocks x 64 KB x 28 layers =
115 MB. Eight such sequences are ~1 GB, nearly the size of the weights: the pool, not the
weights, bounds concurrency on a 16 GB card, and every byte of it is re-read per decode
step by attention.

### Block table and allocation
`KVBlockAllocation`: `physical_block_ids: list[int]`, `sequence_length`, `capacity_tokens`.
The engine stages block tables into a persistent `[max_active, num_blocks]` int32 device
tensor; unused entries are `-1`; kernels only index blocks covered by the row's length.

### `GenerationRequest`
`request_id`, `prompt_token_ids`, `prompt_token_count`, `max_new_tokens`, `state`,
`allocation`, `prefilled_token_count` (chunked prefill progress), `next_token_id` (the
token to feed next step), `output_token_ids`, `created_ns` (FCFS key), prefix-cache
fields (`cached_next_token_id`, reusable blocks), preemption/recompute counters, and
timestamps for TTFT / ITL / queue reports.

### Persistent staging buffers (engine)
Decode: `_host_*` pinned and `_device_*` tensors for `input_ids [max_active,1]`,
`position_ids`, `seq_lens`, `block_tables [max_active, num_blocks]`. Prefill: the same at
`[max_active, chunk]` plus `starts`, `chunk_lens`. **Fixed addresses** are what let a CUDA
graph captured once be replayed on new contents: the graph's kernels read these tensors
by pointer.

### Graph dummy rows
A graph captured at width 4 replayed with 3 live rows needs a fourth row that is harmless:
pad token, position 0, length 0, and a **permanently reserved dummy block** to write into.
`max(bucket) - 1` such blocks are reserved at construction and subtracted from admission
capacity.

---

## 5. Flows

### 5.1 HTTP request to first token
1. `POST /generate/stream` → `submit()`: tokenize (off the event loop), check
   `max_prompt_tokens` and the model context, `service.submit(token_ids, max_new)`.
2. `ContinuousBatchingService.submit` builds a `GenerationRequest`, enqueues it in a
   bounded inbox (429 when full), returns a `RequestHandle`.
3. The worker thread loop (`_worker`): drain inbox → `engine.submit` (scheduler
   `waiting`); apply cancellations; `engine.step()`; publish completions and stats;
   repeat. Nothing else touches the engine.
4. The handler awaits `on_accept` (admission → response headers) then streams tokens
   from `request.output_token_ids` as they grow, ending with a `finish_reason` event.
   Disconnect → `service.cancel(handle)` → scheduler releases KV on the worker.

### 5.2 One `step()` (the heart)
```
decoding = active rows in DECODING
if fused_step and decoding:
    rows = _decode_viable(decoding)      # acquire next KV slot per row, FCFS; may preempt newer rows
else: decode_step(decoding)              # two-forward mode: run decode now
admitted = scheduler.admit_available()   # FCFS from waiting, bounded by free KV (+prefix hits)
plans = scheduler.plan_prefill(chunk=128, budget=128)   # round-robin one chunk per PREFILLING request
if fused: _fused_rows_step(rows, plans)  # one forward for both (falls back if one side is empty)
elif plans: prefill_chunks(plans)
```
`prefill_chunks` has two paths:
- **Fresh-prompt fast path** (`prefill_batch`): every plan is a whole, never-started
  prompt that fits the budget → stock SDPA over the padded prompt batch, K/V written into
  pages by `BatchedPoolBackedPrefillCache`.
- **Chunked path**: stage rows into the prefill buffers, pick a graph by (row bucket,
  attention kind, context bucket) or run eagerly, replay, `argmax` of each row's last
  valid token, one `.tolist()`. Rows whose prompt is now complete move to DECODING with
  that token as `next_token_id`.

`_fused_rows_step`: acquire prefill capacity, re-filter decode rows (a prefill may have
preempted a newer decode row), stage both buffer sets, pick a fused graph by (decode
bucket, chunk-row bucket, kind, context bucket, regime), replay, `.tolist()` once, commit
decode rows (`append_tokens`, EOS/length checks) and prefill rows.

### 5.3 The decode forward, layer by layer
`model(input_ids [N,1], position_ids [N,1])` with `_attn_implementation = "batched_paged_decode"`:
1. Embedding → for each of 28 layers: RMSNorm (Triton) → q/k/v projections (cuBLAS) →
   q_norm/k_norm (Triton RMSNorm) → RoPE (Triton, per-row positions) →
   **`batched_decode_attention_forward`** → o_proj → RMSNorm → gate/up projection →
   SwiGLU (Triton) → down_proj.
2. Inside our attention function, per layer: `write_decode_kv` stores this step's K/V for
   every row at slot `seq_len` of its block table; `paged_decode_batched` attends each
   row's single query to its `seq_len + 1` keys (the `+1` is the `length_offset`, so the
   engine can stage pre-write lengths and never touch them between the two kernels).
3. Final norm → `lm_head` → logits `[N, 1, vocab]` → `argmax` → one `.tolist()`.

The fused variant runs the same forward on one packed row `[1, N + rows*chunk]`; only
`fused_step_attention_forward` knows the boundary and hands each slice to the right
kernel.

### 5.4 Graphs: capture and replay
Capture = run the forward once eagerly (compile Triton kernels, allocate), synchronize,
then run it again inside `torch.cuda.graph(g)` which records every kernel and its
pointers. Replay = `g.replay()`: the same kernels on the same buffers, whatever their
contents now are. Rules the engine obeys:
- shapes are fixed per graph → buckets (decode rows 2/4/8/16; chunk rows 1/2/4; gathered
  context 256..capacity in powers of two; kernel regime);
- nothing inside may synchronize or allocate outside the pool → the SDPA mask and page
  indices are built *inside* the captured forward (a cache filled by the eager pre-run
  once leaked into a capture and replayed freed memory: journal, Phase 2b);
- capture is done on inert rows at `warmup()` for every shape that can occur; a capture
  during serving is counted (`lazy_graph_captures`) and treated as a tail-latency bug.

### 5.5 Capacity pressure
`_capacity_or_fail(request, target_len)`: `ensure_capacity` on the block manager; if
short, evict prefix-cache blocks; if still short, `scheduler.preempt(newest_active)` and
retry; if the requester is itself the newest and not alone, it yields (state WAITING,
requeued in arrival order, re-prefilled later with `resuming=True` so its own generated
tokens are re-prefilled and the pending token preserved). Only a request that is alone in
the pool and still cannot fit is failed (`KV_POOL_EXHAUSTED`). Recompute cost is accounted
per request.

### 5.6 Prefix cache (off by default)
On admission, `lookup(prompt_ids)` finds the longest block-aligned cached prefix (radix)
or an exact match; matched blocks are attached with a refcount, the request starts
prefilling after them (or skips prefill entirely on an exact hit, using the cached next
token). After prefill, `publish` inserts the sequence. A shared partially-filled tail
block is copied on first write (`copy_on_write_tail`, one `_foreach_copy_` across all
pools). Unresolved benefit on the random-prompt soak; needs a shared-prefix workload.

### 5.7 Warmup
Synthetic requests through the real `step()` loop for every bucket in both kernel
regimes and both prefill paths; then explicit capture of every prefill and fused graph
shape; then reset allocator/scheduler/counters. Measured: ITL p99 −71%, p50 unchanged.

---

## 6. The kernels, one by one

Triton vocabulary: a *launch grid* is a tuple of program counts; each **program** is one
thread block (`num_warps` x 32 threads) that runs the kernel body once with its
`tl.program_id`s. Tensors inside the body (`tl.arange(0, BLOCK)`) are distributed across
the block's threads by the compiler; `tl.load(ptrs, mask=, other=)` gathers from global
memory with bounds masking; `tl.sum/max(axis=)` are block-wide reductions. `constexpr`
arguments are compile-time (a new value → a new compiled kernel).

**Online softmax** (used by every attention kernel): to compute
`softmax(q·K^T) V` over keys that arrive tile by tile without storing all scores, keep a
running max `m`, running denominator `l`, and running numerator `acc`. For each tile:
`m_new = max(m, max(scores))`, `alpha = exp(m - m_new)` rescales the old state,
`p = exp(scores - m_new)`, `acc = acc*alpha + p·V_tile`, `l = l*alpha + sum(p)`,
`m = m_new`. Output `acc / l`. Numerically identical to the full softmax.

### `paged_decode_batched` (K4) - live decode attention
- Grid `(S rows, 16 query heads)`; each program: one query vector `[128]`, loops over its
  row's keys in tiles of `BLOCK_N` (64 or 128).
- Paged addressing per tile: logical position `n` → `phys = block_table[n // 16]`,
  slot `n % 16` → pointer into the pool. GQA: `kv_head = q_head // 2`.
- Rank-2 products only: `scores = sum(q[None,:] * k_tile, axis=1)` (`[BLOCK_N]`),
  `acc += sum(p[:,None] * v_tile, axis=0)` (`[128]`).
- Measured 0.08-0.7 ms per layer across (batch 1-16, context 128-2048); ~160 GB/s at
  (8, 1024): parallelism-bound (128 programs on 40 SMs), not traffic-bound. Next step
  would be split-K.

### `paged_decode_gqa` - negative result, kept
Grid `(S, 8 KV heads)`, two query heads unrolled per program sharing each K/V tile. Same
speed as K4 → the L2 was already serving the duplicate read. Its first version used a
`[2, BLOCK_N, 128]` rank-3 product and was 1.5-2.8x *slower*: Triton's 3-D layouts are
expensive; the repo's static test forbids that pattern.

### `write_decode_kv` / `write_prefill_kv_batched`
Grid `(rows, kv_heads)` / `(rows, kv_heads, chunk)`. Each program copies one `[128]` K and
V vector to `pool[phys][slot]`. Bounds-checked: a `-1` or out-of-range block writes
nothing (device-side backstop for the host invariant).

### `sdpa_paged_prefill` - live chunked prefill attention (torch, not Triton)
1. `page_indices`: the first `ceil(T/16)` block ids per row, `-1` clamped to 0 (never
   visible under the causal bound).
2. `gather_pages`: `pool.index_select(0, ids)` → `[B, T, 8, 128]` → permute to
   `[B, 8, T, 128]`. One copy of the prefix per layer per step (~1.6 ms across 28 layers
   at 896 tokens).
3. `chunk_causal_mask`: bool `[B, 1, Q, T]`; row `j` of batch `b` sees keys `<= start_b + j`;
   padding rows see key 0 only (finite softmax, output discarded).
4. GQA fold: `enable_gqa=True` only runs on the slow math backend on sm_75, so the two
   query heads of a group are reshaped into the query axis `[B, 8, 2Q, 128]` (mask
   expanded to match) and the memory-efficient kernel applies. Result reshaped back.
5. Mask and indices are cached per step (`_PrefillContext.sdpa_cache`) so 28 layers build
   them once - and that cache is created fresh for a graph capture.

### `paged_prefill` (per_token) - fallback
Grid `(B, 16, chunk)`: one program per query token; loops the row's keys up to
`start + token + 1` in tiles. No gather, no mask tensor, but 128x the programs of a
tiled kernel and a full re-read of the prefix per token. 2-3x slower than SDPA on T4.

### `tiled_paged_prefill` - parked for sm_80+
FlashAttention-2 structure: `BLOCK_M` query rows per program, `tl.dot` for QK^T and PV,
paged K/V tiles gathered by pointer. On T4 the PTX shows `mma.sync = 0` (Triton emits
MMA only for sm_80+), 255 registers with spills, 49 KB smem → one block per SM. On an
RTX 40-series it should compile to `mma.sync` and is the candidate to beat SDPA-with-gather.

### `rmsnorm`, `rope`, `swiglu`
- RMSNorm: one program per row; fp32 variance, normalize, cast, multiply by weight - the
  exact Qwen ordering so tokens stay identical. Installed on both hidden-state norms and
  the per-head q_norm/k_norm.
- RoPE: grid `(batch, token, head)`, rotates Q and K in one launch from `cos/sin`
  tables (broadcast over batch when the table has batch 1); installed by replacing
  transformers' `apply_rotary_pos_emb` for the Qwen3 module (process-global; `stock_rope()`
  context manager restores it for references).
- SwiGLU: `silu(gate) * up` in one elementwise kernel over strided halves of a fused
  gate/up projection (no `.contiguous()` copies); optional `fuse_gate_up` concatenates the
  two weight matrices into one GEMM.
- Under CUDA graphs these mostly remove launches that graphs also remove; measured
  neutral-to-small. Kept because they are correct, tested, and cost nothing.

### `int8_paged_kv`
INT8 pages plus fp16 per-token scales; write kernels quantize, attention kernels
dequantize per tile. Halves KV bytes; on T4 the decode step did not move and the chunked
prefill path drifts from stock (quantized prompt K/V) - off by default.

---

## 7. How to read a result

Every A/B prints medians over 5 interleaved runs with `spread`; "unresolved" means the
change is inside the spread and is not a result. Token identity vs stock Transformers is
checked before timing; a first difference on a tied logit (`TIE_MARGIN = 0.02`) is a tie,
not a divergence. `lazy_graph_captures` must be 0 in both arms before any p99/p999
comparison counts. The journal records negative results with the same care as wins -
they are what stop the same idea from being retried.

## 8. Reading order for re-learning the code

1. `engine/runtime/request.py` (state machine) → `engine/scheduler/scheduler.py`.
2. `engine/cache/paging.py` (blocks) → `engine/kernels/kv_write.py` → `engine/kernels/paged_decode_batched.py` (with section 6 open).
3. `engine/batching/continuous_batching.py`: read `__init__`, then `step`, then
   `_decode_rows`, `prefill_chunks`, `_fused_rows_step`, then the three attention
   functions at the top, then `warmup`.
4. `engine/graphs/paged_decode_graph.py` → `fused_step_graph.py`.
5. `engine/kernels/sdpa_prefill.py`, then `paged_prefill.py` for contrast.
6. `engine/server/continuous.py` → `api.py`.
7. `benchmarks/reliability/soak.py` → `ab.py`; then run `benchmarks/understanding/real_decode_step_trace.py` on a GPU and watch one step happen.
8. `docs/optimization-journal.md` from "Structural overhead removed" onward, with
   `docs/checkpoint.md` beside it.

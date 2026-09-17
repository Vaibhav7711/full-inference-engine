# Delivery checkpoint

Living status for the single-GPU inference engine. Updated at each gate. The point of this
file is that nothing established here gets re-argued, and nothing retracted here comes back.

**Goal:** a complete, readable, production-hardened single-GPU engine, with a benchmark
table against vLLM on T4 and L4 for Qwen3-0.6B/1.7B/4B. Parity on any axis, built by one
person, beats an unproven claim of superiority. The product wedge, if one is wanted, is
cold start and footprint.

---

## 1. Validated numbers

Measured on Colab T4 with Qwen3-0.6B FP16, reproducible from a committed script. Anything
not here is not established.

### Hardware and model

| quantity | value | source |
|---|---|---|
| achieved bandwidth (fp16 GEMV probe) | **258.8 GB/s** (81% of 320 spec) | `benchmarks/kernels/roofline.py` |
| read-only sweep / STREAM copy | 271.4 / 240.6 GB/s | same |
| weight bytes read per decode step | **1192.1 MB** (881 layers + 311 tied `lm_head`) | same |
| KV bytes per token, all layers | 112 KB | same |
| weight-only floor | **4.61 ms/step** | derived |
| floor at batch ~7, ~128-token context | **~5.02 ms/step** | derived |

A step advances every active sequence by one token, so step time *is* the inter-token
latency a caller experiences. Do not divide by batch; that answers a throughput question.

### Engine (closed-loop, concurrency 8, 5 runs per arm, CUDA graphs on)

| quantity | value |
|---|---|
| decode-only step p50 | **8.05-8.20 ms** — 1.64x the memory floor, 3.22 ms/token headroom |
| prefill-carrying step p50 | **42.5-44.7 ms** |
| prefill share of steps | **22.6%** |
| frequency-weighted average gap | **15.9-16.4 ms** |
| pure prefill interruption | **8.24 ms/token — ~50% of user-visible ITL** |

Cross-validated: the weighted reconstruction (15.88 ms) matches an independently measured
per-request mean (15.19 ms).

**CUDA graphs** remove 88% of overhead above the floor: decode step 35.999 → 8.202 ms
(-77.2%), prefill step 71.896 → 44.658 ms (-37.9%), TTFT -39.4%. Replicated three times at
-64.5%, -63.8%, -77.3% on ITL as the estimator was corrected.

**Prefill cost is fixed per invocation.** 4x less work per step (chunk 128 → 32) changed
step cost -2.4%, inside noise, while doubling prefill frequency: average gap +49.5%, TTFT
+192.3%. Prefill's excess over a decode step is 36.5 ms with graphs and 35.9 ms without —
identical, so prefill is entirely ungraphed. **No scheduling change can reduce it**; the
prefill path itself is the target.

### Preemption (Gate 1B)

Rebuild 37.53 ms, independent of token count (32 and 22 tokens both 37.53 ms). Queue wait
after yielding 131-664 ms, dominating rebuild 3.5-17x. `recompute_ms` includes first-use
Triton JIT: 815 ms first run vs 75 ms steady state, 1465 ms for INT8.

---

## 2. Retracted claims

Listed so they cannot resurface. Each was wrong for a different reason.

| claim | why it was wrong |
|---|---|
| decode is 3.9x above a 4 ms floor | spec bandwidth quoted as achieved; KV traffic omitted; 256 and 320 GB/s used in different places unflagged |
| 2.4x, 9.08 ms headroom | measured bandwidth, but a guessed 512-token context inflated the floor 28% |
| 3.0x, 10.17 ms headroom | real operating point, but ITL percentiled over per-request means, inflating the measurement 84% |
| CUDA graphs give 2.2-4.6x | compared configurations differing in three settings at once, one having rejected 112 of 114 requests so its sequences were short |
| recompute costs 2.4-3.4 rebuilt tokens per delivered token | denominator modelled from a guessed average, not measured |
| prefill rides 0.1-1% of steps | inferred from a percentile gradient; measured at 22.6%, wrong by 20-200x |
| p999 null result shows the tail is prefill | guaranteed by the design: both arms run identical un-graphed prefill |
| smaller chunks trade ITL for TTFT | no such trade at 32-128 tokens; cost is fixed per invocation, so smaller chunks lose on both |
| chunk 512 vs 128 shows cost is fixed above 128 | non-binding: the workload's ~122-token prompts fit one chunk in both arms |
| prefill cost is fixed per invocation | true only for 32-128 tokens; the fitted marginal cost is 0.405 ms/token and dominates above chunk 64 |
| packing chunks does not amortise the fixed cost | the script compared a packed step against a single-chunk step rather than against four separate steps; packing is 3.4x cheaper per unit work |

---

## 3. Rules established

| record | rule |
|---|---|
| DD-028 | Strict-LIFO recompute preemption, readmission gated on a progress epoch, no count limit; copy-out/swap rejected on measured rebuild cost |
| DD-029 | Pressure tests size their pool from real tokenization, never a constant |
| DD-030 | Queueing, stalls and decode time reported separately once preemption exists |
| DD-031 | The worker publishes engine stats; HTTP handlers never read live scheduler state |
| DD-032 | Every soak ends in a KV page accounting audit, and one test proves the audit can fail |
| DD-033 | Cancellation coverage asserted only for states the workload reaches |
| DD-034 | Comparisons are repeated, single-variable, closed-loop, "unresolved" inside noise |
| DD-035 | Latency runs are closed-loop; open-loop runs are reliability tests only |
| DD-036 | Correctness violations and coverage gaps are separate; `ok` depends only on the former |
| DD-037 | The optimisation target is measured on the device, never quoted from a datasheet |
| DD-038 | Tail latency is percentiled over individual token gaps, never over request means |
| DD-039 | Steps are timed by what they did, not just how long they took |

**Statistic selection, learned the hard way:** a median over token gaps cannot see a change
in the *mix* of step kinds. When prefill rose from 22.7% to 46.9% of steps, `itl_p50` moved
+3.8% (unresolved) while the average gap worsened 49.5%. `expected_gap_ms` now leads the
A/B comparison list for that reason.

---

## 4. Gate status

| gate | state |
|---|---|
| Gate 1 — survive bad input | **accepted**: 14 CUDA, 19 kernel, CPU suite green |
| Gate 1B — recompute preemption policy | **accepted**: 5/5 interaction tests, token-identical under pressure |
| Metrics accounting | **accepted** |
| Item 1 — reliability soak | **accepted**: 6 soaks, zero invariant violations, peak KV 1.00 |
| Measured roofline | **accepted** |
| Prefill chunk A/B (128 vs 32) | **accepted**: cost is fixed per invocation |
| Prefill chunk A/B (128 vs 512) | **void**: non-binding, both arms fit a ~122-token prompt in one chunk |
| Prefill investigation (Phase B sweep) | **accepted**: `step_ms = 21.13 + 0.405*chunk`, R²=0.998 |
| Phase D1 — decouple prefill budget from chunk size | **next**, scheduling only |
| Phase D2 — tiled causal prefill kernel | **failed**: 0.3x, 0.2% of tensor-core peak. Grid collapses to 64 blocks on 40 SMs at chunk 64 — tiling the query dimension trades away the parallelism it was meant to buy. Diagnosing with spills, occupancy and an SDPA upper bound. |

---

## 5. Open questions

1. ~~Is prefill launch-bound or compute-bound?~~ **Answered: both, and compute dominates.**
   `a`=21.13 ms fixed, `b`=0.405 ms/token (22.5x floor). See `docs/experiment-plan-prefill.md`.
1b. ~~Is the prefill picture representative?~~ **Answered: no, it understated by 3.7x.**
   Now measured across three prompt profiles.
1c. **What `b` is achievable with a tiled kernel?** Target 0.05 ms/token (2.8x floor).
   Unknown until D2 is built; D3 checks it against that target.
1d. **What budget value?** D1 needs a sweep of `max_prefill_tokens_per_iteration` at fixed
   `prefill_chunk_size`; B3 tested only 1x and 4x.
2. **Which prefill path does this workload take?** The SDPA fast path applies only when
   every request in the batch is fresh *and* its chunk covers the whole prompt
   (`docs/understanding-journal.md`, Layer 7). With ~120-token prompts against a 128 budget
   and several requests admitted together, it is a mix. Unmeasured.
3. **Is 44.7 ms reasonable?** Rough floor ~15 ms: 4.61 ms weight read + ~2.4 ms prompt
   compute + the 8.2 ms decode the step also performs. Measured is ~3x.
4. **Waste ratio needs many repeats.** 220% spread across four runs of one configuration
   (0.88-5.20). No single-run figure means anything.
5. **p999 cause.** Unresolved, low priority. Candidates: graph capture, eager fallback
   outside buckets, host noise.

---

## 6. Plan order

0. Gate 1 GPU gate — **done**
1. Reliability validation — **done** (dependency pin still open)
2. **Prefill path — current**, evidence-backed at 8.24 ms/token and now known to be a path
   problem rather than a scheduling one
3. Correctness breadth (long contexts, INT8, EOS, resets)
4. Serving surface, then hardening
5. Packaging and cold start (the wedge)
6. QoS policy — *narrowed*: the chunk-size knob is not a QoS lever, since cost is fixed
   per invocation
7. Re-baseline and conditional kernel work
8. Consolidation and the vLLM/L4 table — the definition of done
9. Parked: speculative decoding (Qwen3-0.6B → Qwen3-4B), W4A16

---

## 7. Carried debt

Flagged and not yet addressed:

- `transformers` pinned `>=4.51` with no upper bound while depending on private APIs
  (`_attn_implementation_internal`, `ALL_ATTENTION_FUNCTIONS`, `get_mask_sizes`); needs an
  upper bound and import-time assertions so an upgrade fails loudly at startup
- streaming detokenizer decodes one token at a time, breaking multi-byte UTF-8
- no OpenAI-compatible routes, sampling, stop sequences or logprobs — **blocks the vLLM
  table**, since `vllm bench serve` and genai-perf target that surface
- decode attention has no split-K and reads each GQA group once per query head (twice for
  this model); at batch 16 / 2048 context the ideal KV read is 3758 MB against 1192 MB of
  weights, so doubling it adds ~14 ms to a ~19 ms floor
- ~1,900 lines of superseded paths still importable; decide keep-or-delete before writing
  correctness-breadth tests against them
- README still describes the HuggingFace-tensor attention path
- `_BATCH_CTX` / `_PREFILL_CTX` remain process-global; blocks a second engine in-process,
  which speculative decoding would need
- **`results/` holds only three stale files.** Every roofline and A/B run since has been
  reported in chat and journalled but never committed. The evidence behind section 1 should
  be in the repo.

---

## 8. Existing work to reuse, not rebuild

Found while checking for duplication:

- `docs/understanding-journal.md` — seven layers of traced behaviour, including chunked
  prefill and decode-first scheduling (Layer 7), paged KV, metadata staging, graph replay,
  prefix cache copy-on-write
- `benchmarks/understanding/` — eight tracing scripts for the same
- `benchmarks/batching/chunked_prefill_latency.py` — decode latency isolation while a long
  prompt prefills; no committed results
- `benchmarks/batching/mixed_arrival_prefill_budget_ab.py`,
  `mixed_arrival_graph_ab.py`, `padded_graph_occupancy_ab.py` — earlier A/B harnesses
  predating `benchmarks/reliability/ab.py`; consolidate at the cleanup gate

---

## 9. Working agreement

- Patches applied from a file, pushed from the Mac, run in Colab; results pasted back.
- Every gate ends with a journal entry: measured numbers, decision, what was rejected.
- Reference comparisons use a separately loaded, unpatched checkpoint under `stock_rope()`.
- No performance claim without repeats, a single varied variable, and a spread check.
- Choose the statistic before reading it: medians hide changes in population mix.
- **Everything checkable without a GPU is checked without a GPU.** Launch grids, register
  budgets, shared-memory bounds, tile feasibility and whether an experiment's treatment
  binds are arithmetic, not hardware questions.
  `tests/kernels/test_kernel_static_checks.py` is the first stage of `scripts/verify.py`.
- **Triton kernels cannot be validated before they reach the GPU.** There is no CUDA device
  in the authoring environment, so a kernel ships compile-unchecked and the first GPU run
  is its compile gate. Every kernel test file therefore opens with a sub-second smoke test
  that just builds and runs the kernel, so a compilation error costs one failing test
  rather than burying the real numerical questions under N copies of the same traceback.
- **Plan the whole investigation before the first run.** Three GPU sessions were spent on
  questions whose answers could not change what gets built, or on treatments that could not
  take effect. A multi-question investigation gets a written plan with pre-registered
  predictions and a fixed decision rule (`docs/experiment-plan-prefill.md` is the model),
  and runs as one sweep rather than a chain of A/Bs.

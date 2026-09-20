# T4 re-evaluation plan — per-optimization attribution

Purpose: re-measure every optimization recorded in `docs/optimization-journal.md` on one
Colab T4, in one consistent method, and produce a **contribution ledger**: how much each
single change is worth, against which baseline, on which workload, with a verdict that says
`real` / `unresolved` / `regression` rather than a number quoted from a different session.

This plan is pre-registered in the sense of `docs/experiment-plan-prefill.md`: the workloads,
metrics, and decision rules are fixed here before the GPU runs.

## 1. Ground rules

1. **One T4, one process per benchmark, `CUDA_VISIBLE_DEVICES=0`.** Colab gives one card, so
   the two-GPU split in `scripts/t4_phase0_phase1.ipynb` collapses to a single queue.
2. **Same-session interleaved A/B, 5 repeats minimum.** A number from a different Colab
   session is not a baseline. Every comparison in this plan runs both arms in the same process
   (or the same notebook session for the commit ladder), alternating arms per round.
3. **Verdict by spread.** A median change smaller than the run-to-run spread is `unresolved`.
   `benchmarks/reliability/ab.py` already does this; the ladder scripts must print spread too.
4. **Token identity is a gate, not a metric.** Every arm must emit identical greedy tokens
   except `kv_dtype` (INT8 is expected to drift). A speedup that changes tokens is a bug.
5. **Warm before timing.** `ContinuousBatchingEngine.warmup()` (captures every graph bucket,
   compiles both prefill paths) or the benchmark's own warm rounds. `recompute_ms` and
   first-bucket captures were 10x off cold in Gate 1B; never time a cold path.
6. **Record clocks.** Colab does not allow `nvidia-smi -lgc`; `ab.py` records SM MHz
   before/after each run. A run whose clocks moved >10% is re-run, not averaged.
7. **Measure the operating point.** Report `decode_batch` and `decode_mean_context` next to
   every ITL figure; the roofline floor depends on both (journal: "Validated target").
8. **Percentiles over token gaps**, never over per-request means (journal: "Why itl_p99 never
   resolved").

## 2. The two "naive" baselines

| Baseline | What it is | Journal figure (T4, Qwen3-0.6B FP16) | Re-measure with |
| --- | --- | --- | --- |
| **B0 — stock HF greedy** | `DynamicCache`, one request, plain `generate` loop | 23.7 tok/s single stream; TTFT p50 43.6 ms; decode 40.0 ms/token | `benchmarks/inference/t4_baseline.py`, `benchmarks/cache/paged_cache_bench.py` |
| **B1 — pre-refactor K4 engine** | Paged KV + batched decode kernel, Python metadata per step, sequential prefill | 170.3 tok/s @ width 16; 23.2 tok/s @ width 1 (16 req × 32 tok) | `git checkout 79d8671`, `benchmarks/batching/continuous_throughput.py` |
| **B2 — memory floor** | Bytes/step ÷ achieved GEMV bandwidth | 4.61 ms/step weight-only; 5.02 ms at batch 7.5 / 125 ctx (258.8 GB/s achieved) | `benchmarks/kernels/roofline.py` |

B0 is the "naive" reference for everything user-visible. B1 is the reference for the Phase 2
ladder. B2 is the ceiling: contribution is also reported as "ms of overhead above the floor
removed".

## 3. Optimization inventory and how each is isolated

Isolation tiers:

- **T** — runtime toggle exists today (engine kwarg or CLI flag). Measured leave-one-out and
  stacked with `ab.py`.
- **K** — kernel installer exists (`install_*` / `uninstall_*`) but no engine kwarg.
  Needs the prerequisite in §7, then measured like **T**.
- **L** — structural change with no toggle. Measured by a **commit ladder**: checkout
  commit N−1 and N, run the identical benchmark in the same session, alternate 3×.
- **R** — rejected/reverted. Re-run its own benchmark once to confirm the rejection still
  holds on the current stack; not part of the ledger unless it flips.

### 3.1 Decode path

| # | Optimization | Commit | Journal claim (baseline → after) | Tier | Isolate by |
| --- | --- | --- | --- | --- | --- |
| D1 | FP16 over BF16 on Turing | `370a11a` | TTFT 59.0 → 43.6 ms (−26%); decode tied | T | `t4_baseline.py --include-bfloat16` |
| D2 | Paged KV cache, block 16 (vs DynamicCache) | `79d8671` | 1.1% slower single-stream; enables batching | T | `paged_cache_bench.py --block-sizes 8,16,32` |
| D3 | K4 batched paged decode kernel (B1 itself) | pre-`79d8671` | 23.2 → 170.3 tok/s @16 (7.3x) | L | ladder B0 → B1 |
| D4 | Unified KV block ownership | `acc719d` | 170.3 → 171.2 @16 (+0.5%) | L | ladder `79d8671` → `acc719d` |
| D5 | Persistent decode metadata buffers | `5126ade` | 167.5 → 172.6 @16 (+3.0%) | L | ladder `90485ba` → `5126ade` |
| D6 | Scheduler/request unification | `90485ba` | 171.2 → 167.5 @16 (−2.2%, accepted) | L | ladder `acc719d` → `90485ba` |
| D7 | Direct pool-backed prefill KV write (Triton) | `9eda221` | 172.6 → 178.3 @16 (+3.3%) | L | ladder `5126ade` → `9eda221` |
| D8 | Batched decode KV write (removes 896 `.item()`/step) | `a8469ab` | 178.3 → 247.8 @16 (**+39%**) | L | ladder `9eda221` → `a8469ab` |
| D9 | In-kernel decode length offset | `291ccf8` | 247.8 → 263.5 @16 (+6.3%) | L | ladder `a8469ab` → `291ccf8` |
| D10 | Fused Triton RMSNorm (113 sites) | `5282517` | 263.5 → 253.1 @16 (−4%, inside noise); launches 40.5k → 12.5k | K | `triton_rmsnorm=False` |
| D11 | Fused Triton RoPE + SwiGLU | `35a08f6` | 253.1 → 261.8 @16 (+3.4%); cats 1767 → 0 | K | `triton_rope=False`, `triton_swiglu=False` (separately) |
| D12 | Paged-decode tile regimes 64x4 / 128x4 | `ff491ff` | isolated kernel −20–42% at ctx ≥256; e2e short-prompt unchanged | T | `paged_decode_regime_sweep.py --configs`; engine: pin regime via `paged_decode_config` |
| D13 | Fixed-width CUDA-graph bucket (16) | `2687241` | 37.2 → 9.7 ms/step (3.85x); 962 tok/s @16 | T | `cuda_graph_batch_sizes=None` vs `(16,)` |
| D14 | Padded power-of-two graph buckets | `5086c14` | 3.4–4.1x at occupancies 1–16 | T | `cuda_graph_batch_sizes=(16,)` vs `(2,4,8,16)` |
| D15 | Batched token transfer (`.tolist()`) | Phase 15 | host transfer 0.206 → 0.022 ms @16 (9.35x); e2e neutral | L | `token_transfer_ab.py` (isolated); ladder for e2e |
| D16 | Decode metadata staging: slice copy per row | `5a9514c` | 3.4 → 0.18 ms/step host stage @16×64 blocks (off-GPU) | L | ladder `55ebb7a` → `5a9514c` (with D17, P6) |
| D17 | Copy-on-write tail via one `_foreach_copy_` | `5a9514c` | 56 launches → 1 on first decode step after exact hit | L | ladder `55ebb7a` → `5a9514c` (with D16) |

### 3.2 Prefill and scheduling

| # | Optimization | Commit | Journal claim | Tier | Isolate by |
| --- | --- | --- | --- | --- | --- |
| P1 | Mixed-length batched prefill | `b728915` | 261.8 → 413.8 @16 (**+58%**); 165 → 214 @8 | L | ladder `35a08f6` → `b728915`; `prefill_throughput.py` (sequential vs batched A/B) |
| P2 | Chunked paged prefill + decode-first scheduling | `24e7621` | worst decoder ITL 308 → 97 ms (−69%); long TTFT 361 → 1230 ms; short-prompt tp unchanged | T | `chunked_prefill_latency.py --chunk-size`; `ab.py --setting prefill_chunk` |
| P3 | Prefill budget 128 tok/iteration | `b8811e9` | 64/128/256: 295/430/424 tok/s; long TTFT 898/531/544 ms | T | `mixed_arrival_prefill_budget_ab.py --budgets 64,128,256` |
| P4 | Batching chunks across requests in one step | Phase B | 4 chunks in one step 55.4 ms vs 4 steps 186.3 ms (3.4x/unit work) | T | `sweep.py --only B3` |
| P5 | Tiled causal prefill kernel (D2) | `tiled_paged_prefill.py` | **0.3x** (slower) before tile-default fixes; now default `tiled_prefill=True`, unmeasured | T | `ab.py --setting prefill_kernel --prompt-profile chat`; `prefill_attention_ab.py --sweep-tiles` |
| P6 | Prefix-cache eviction bookkeeping incremental | `5a9514c` | O(N) per evicted block → O(1); unmeasured | L | ladder `55ebb7a` → `5a9514c` (with D16) |
| P7 | `warmup()` before serving | `5a9514c` | removes first-request capture/JIT from p999; unmeasured | T | soak with/without `warmup()`; compare p999 and first-request TTFT |

### 3.3 Memory / capacity

| # | Optimization | Commit | Journal claim | Tier | Isolate by |
| --- | --- | --- | --- | --- | --- |
| M1 | Refcounted paged prefix cache | `6428f81` | exact-hit TTFT 343.7 → 3.11 ms (110x); miss-only tp −1.9% | T | `prefix_cache_ttft.py`; `ab.py --setting prefix_cache --prompt-profile chat` |
| M2 | INT8 paged KV (kernel-native) | Phase 6 | KV bytes −49.2%; kernel 1.22x @1024/16, 1.37x @2048/16; e2e neutral at 1024 | T | `int8_paged_decode_ab.py`; `ab.py --setting kv_dtype --prompt-profile long` |
| M3 | Recompute preemption (Gate 1/1B) | `0b20313` | survives pool pressure; recompute 37.5 ms fixed/rebuild; queue wait 3.5–17x rebuild | T | `soak.py` pressure config; `recompute_report()` |

### 3.4 Rejected / reverted (confirm-only)

| # | Experiment | Commit | Journal result | Re-run |
| --- | --- | --- | --- | --- |
| R1 | Remove per-token CUDA sync (reference loop) | `1c74f0c` / `f1e11a3` | −7–11% (non-interleaved; not causal) | skip — reference runner is obsolete |
| R2 | Batched sampled-token materialization (pinned D2H) | `f74c6ce` / `9b4b428` | 263.5 → 227.2 @16; syncs 496 → 31 | ladder `291ccf8` → `f74c6ce` → `9b4b428`, 3 rounds. Cheap, and it's the one reverted result with a clean adjacent baseline |
| R3 | Fused MLP gate/up projection | `189e29c` / `b5ba645` | 1.023x, 5/7 rounds neutral. **Journal says re-run**: the old measurement paid two `.contiguous()` copies that `5a9514c` removed | `mlp_gate_up_fusion_ab.py --rounds 7 --cuda-graph-batch-size 16` — this one can flip |
| R4 | W8A16 Triton linear | `d6fef48` | 0.12–0.19x vs FP16 CUTLASS | `w8a16_linear_ab.py`, 1 run |
| R5 | W8A8 via `torch._int_mm` | `6e1fc86` | 0.11–0.18x | `w8a8_linear_ab.py`, 1 run |
| R6 | Chunk 32 vs 128 | — | expected gap +49.5%, TTFT +192% | covered by P2 |

## 4. Fixed workloads

Every optimization is scored on the workload(s) it targets. Nothing is scored on a workload
where the treatment cannot bind (`ab.py`'s `binding_check` enforces this for chunk sizes).

| ID | Workload | Script | Used for |
| --- | --- | --- | --- |
| W1 | 16 requests × 32 output tokens, short prompts, widths 1/2/4/8/16 | `continuous_throughput.py --concurrencies 1,2,4,8,16 --warmup` | ladder (D3–D9, P1, R2), D13/D14 |
| W2 | closed-loop soak, concurrency 8, `chat` profile (~656-tok prompts), 8 s × 5 repeats | `ab.py --prompt-profile chat --repeats 5 --duration 8` | all **T**/**K** leave-one-out |
| W3 | closed-loop soak, `long` profile (~1824 tok) | `ab.py --prompt-profile long` | M2, P5, D12 |
| W4 | isolated kernel microbenchmarks (CUDA events) | `paged_decode_regime_sweep.py`, `int8_paged_decode_ab.py`, `prefill_attention_ab.py`, `token_transfer_ab.py` | D12, D15, M2, P5 |
| W5 | 871-token repeated prompt, 5 warm hits | `prefix_cache_ttft.py` | M1 |
| W6 | mixed-arrival short/medium/long | `mixed_arrival_*_ab.py` | P3, D14 |
| W7 | 16-stream SSE burst over real `uvicorn` | `uvicorn_load.py --requests 16` | end-to-end sanity, not attribution |

## 5. Metrics per optimization

| Metric | Definition | Attributed to |
| --- | --- | --- |
| `tp16` | aggregate tok/s at width 16, W1 | D3–D9, D13, D14, P1 |
| `decode_step_p50_ms` | decode-only step time, W2 (`instrument=True`) | D10, D11, D13–D16 |
| `itl_p50_ms` | median individual token gap, W2 | everything in §3.1 |
| `expected_gap_ms` | frequency-weighted step cost (decode share + prefill share) | P2, P3, P4, P5 — the only latency metric that sees a change in step *mix* |
| `prefill_step_p50_ms`, `b` (ms/token slope) | W2/W3 and `sweep.py` fit `a + b·chunk` | P4, P5 |
| `ttft_p50_ms` | W2, W5 | M1, P2, P3 |
| `overhead_above_floor_ms` | `decode_step_p50 − roofline floor at measured (batch, ctx)` | D13 (headline: 27.96 → 3.22 ms) |
| `kv_bytes` | pool bytes at fixed blocks | M2 |
| `host_stage_ms` | `last_step_timing["host_stage_ms"]` | D16 |
| kernel `ms` | CUDA-event median, isolated | D12, D15, M2, P5, R4, R5 |

## 6. Attribution method

Two views are produced for every **T**/**K** item, because they answer different questions:

1. **Leave-one-out (marginal in the final engine).** All optimizations on = `full`. Each
   arm turns exactly one off. Contribution = `metric(full) − metric(full − X)`. This is what
   the optimization is worth *today*, with everything else present.
2. **Stacked ladder (historical).** Start from B1, add optimizations in the order the journal
   applied them, one per arm. Contribution = `metric(step N) − metric(step N−1)`. This is
   what the journal's numbers claimed, re-measured in one session.

The two views disagree where optimizations overlap, and that disagreement is itself a
finding. Known overlaps to expect:

- **Fusions vs CUDA graphs.** D10/D11 mostly removed *launch* cost (40.5k → 2.1k launches).
  Graphs remove launch cost too, so leave-one-out of a fusion with graphs on should be near
  zero; measure fusions with graphs **off and on** and report both.
- **Persistent metadata (D5) vs graphs (D13).** Graphs require D5's stable addresses; D5 cannot
  be turned off with graphs on. Ladder only.
- **INT8 KV (M2) vs context.** Neutral at ~128 ctx, 1.37x kernel at 2048/16. Score on W3 only.
- **Chunked prefill (P2) vs prefill budget (P3).** `prefill_chunk_size` bounds one request,
  `max_prefill_tokens_per_iteration` bounds the step; vary one at a time.

For **L** items the ladder is the only view. Run each adjacent pair interleaved 3×
(`N−1, N, N−1, N, N−1, N`), each checkout in a fresh process, same model load path, same
`--warmup`. Report median and spread per commit.

Decision rule per item, fixed in advance:

- `|Δ| > 2 × spread` → `real`, sign gives contribution/regression.
- otherwise → `unresolved`; the ledger records "≤ spread", not the point estimate.
- token identity false (outside `kv_dtype`) → `INVALID`, stop, file a bug.

## 7. Prerequisites — done

1. **Kernel toggles as engine kwargs**: `ContinuousBatchingEngine(..., triton_rmsnorm=,
   triton_rope=, triton_swiglu=)`, default `True`. A `False` flag actively *uninstalls* the
   fusion from the model object, because the A/B builds every arm on one shared checkpoint
   and a previous arm may have patched it. CUDA test:
   `tests/batching/test_continuous_batching.py::test_d3_each_fusion_toggle_off_matches_ref`.
2. **`ab.py` settings**: `triton_rmsnorm`, `triton_rope`, `triton_swiglu`, `mlp_gate_up`,
   `graph_buckets_padded` (exact width vs powers of two), `warmup` (cold vs warmed), and the
   leave-one-out family `loo_graphs`, `loo_prefix_cache`, `loo_tiled_prefill`, `loo_rmsnorm`,
   `loo_rope`, `loo_swiglu`, plus `loo_all` (all seven arms interleaved against one `full`).
   The harness now compares every arm against the first and writes `comparisons[label]`.
3. **Ladder runner**: `scripts/commit_ladder.py` — one worktree per rung, every rung every
   round, `continuous_throughput.py` on the 16×32 workload, per-rung median ± spread and a
   verdict against the previous rung. The JSON schema is identical back to `79d8671`.
4. **Preflight** writes `--out env.json` with torch/triton/transformers versions.
5. **Notebook**: `scripts/t4_reevaluation.ipynb` — gate, Session A/B/C, ledger, save.
   Jobs whose output already exists are skipped, so a dead session resumes.

## 8. Colab session plan

Run `scripts/t4_reevaluation.ipynb` top to bottom. Colab T4 sessions are killed on idle and
hard-capped, so each session cell is resumable and every job writes its own JSON. Times are
estimates (model load ~30 s; an `ab.py` arm costs ~8 s timed + ~25 s warmup per repeat).

Gate (~10 min): CUDA test suites, including the new fusion-toggle test.

Session A — baselines and toggles (~80 min)

| Order | Job | Est. | Attributes |
| --- | --- | --- | --- |
| A2 | `roofline.py` (B2) | 2 min | floor |
| A3 | `t4_baseline.py --include-bfloat16` (B0, D1) | 4 min | D1 |
| A4 | `paged_cache_bench.py --block-sizes 8,16,32` | 4 min | D2 |
| A5 | `ab.py --setting loo_all --prompt-profile chat` | 25 min | D10, D11, D13, M1, P5 leave-one-out with graphs on |
| A6 | `ab.py --setting graph_buckets_padded` | 5 min | D14 |
| A7 | `ab.py --setting mlp_gate_up --cuda-graphs` | 5 min | R3 re-run |
| A8–A10 | `ab.py --setting triton_{rmsnorm,rope,swiglu}` (graphs off) | 12 min | D10, D11 before graphs |
| A11 | `ab.py --setting prefill_chunk --cuda-graphs` | 5 min | P2 |
| A12 | `prefix_cache_ttft.py` | 3 min | M1 exact-hit TTFT |
| A13 | `ab.py --setting warmup --cuda-graphs` | 5 min | P7 |
| A14 | `token_transfer_ab.py --widths 1,2,4,8,16` | 1 min | D15 |

Session B — commit ladder (~60–70 min, W1 only)

`scripts/commit_ladder.py --rounds 3 --widths 1,8,16`. Rungs in the journal's measurement
order: `79d8671 → acc719d → 90485ba → 5126ade → 9eda221 → a8469ab → 291ccf8 → f74c6ce →
9b4b428 → 5282517 → 35a08f6 → b728915 → 24e7621 → 6428f81 → HEAD → HEAD+graph16`.
(`90485ba` and `5126ade` were measured before `9eda221`: 171.2 → 167.5 → 172.6 → 178.3.)
Every rung runs every round, so drift is spread across all rungs; add widths 2/4 if time
remains.

Session C — long context, kernels, mixed arrival (~50 min)

| Order | Job | Attributes |
| --- | --- | --- |
| C1 | `ab.py --setting kv_dtype --prompt-profile long --cuda-graphs --num-blocks 512` | M2 e2e |
| C2 | `int8_paged_decode_ab.py --seq-lens 256,1024,2048 --batches 1,16 --rounds 5` | M2 kernel |
| C3 | `paged_decode_regime_sweep.py --seq-lens 64…2048 --batches 1,8,16 --configs 64x4,128x4` | D12 |
| C4 | `sweep.py --only B1 B3` | P4, prefill slope `b` |
| C5 | `prefill_attention_ab.py --sweep-tiles --chunks 64 128 256 512` | P5 kernel |
| C6 | `mixed_arrival_prefill_budget_ab.py --budgets 64,128,256 --rounds 5` | P3 |
| C7 | `mixed_arrival_graph_ab.py` | D14 on mixed arrivals |
| C8 | `padded_graph_occupancy_ab.py` | D14 per occupancy |
| C9–C10 | `w8a16_linear_ab.py`, `w8a8_linear_ab.py` | R4, R5 |
| C11 | `uvicorn_load.py --requests 16` | W7 sanity |

If a session dies, re-run the session cell: `done()` skips jobs whose JSON exists.

## 9. Output: the contribution ledger

One row per optimization, filled from the JSON files, committed to
`results/t4/<date>_<sha>/ledger.md`. Template:

| # | Optimization | Workload | Metric | Baseline | With | Δ | Spread | Verdict | Journal claim | Agrees? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D8 | Batched decode KV write | W1 | tp16 | | | | | | +39% | |
| D13 | CUDA graphs | W2 chat | itl_p50 | | | | | | −75% | |
| D13 | CUDA graphs | W2 chat | overhead above floor | | | | | | 27.96 → 3.22 ms | |
| P1 | Batched prefill | W1 | tp16 | | | | | | +58% | |
| M1 | Prefix cache | W5 | ttft (exact hit) | | | | | | 110x | |
| … | | | | | | | | | | |

Plus two summary views:

- **Stacked**: a table `B0 → B1 → … → HEAD` of `tp16` and `itl_p50`, so the reader sees where
  the 23.7 → ~1600 tok/s came from step by step.
- **Leave-one-out**: `full` minus each toggle, sorted by Δ, with graphs-off and graphs-on
  columns for the fusions.

`Agrees?` is `yes` (same sign, within 2× spread of the journal's magnitude), `smaller`,
`larger`, `flipped`, or `n/a` (journal figure was profiler-only or non-interleaved).

## 10. What this plan deliberately does not do

- It does not re-run Phase 1 (R1): the reference runner it measured no longer exists.
- It does not score the fusions on W1 alone; at short prompts with graphs on their marginal
  contribution is expected to be inside spread, and reporting that as "worthless" would be
  the same mistake as reporting the journal's +3.4% as "real".
- It does not compare across Colab sessions. Sessions A/B/C each contain their own baselines.
- It does not fix `tiled_prefill`. P5 decides it; whichever arm wins becomes the default in a
  separate commit that cites the ledger row.

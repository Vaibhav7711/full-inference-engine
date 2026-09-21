# T4 results (Kaggle, 2026-09-21) — transcribed from session output

The Kaggle notebook wrote every run's JSON and log to `results/t4/<date>_<sha>/` on its
own disk; a git push from that session was never completed (GitHub returned 403 to a
token that `GET /repos/...` reported as `push: true`; not resolved). The numbers below
were pasted from the notebook's stdout into the working session as each run finished
and are transcribed here verbatim. They are the same numbers the journal
(`docs/optimization-journal.md`) cites. Anything not pasted is not here.

Protocol for every engine A/B unless stated: `benchmarks/reliability/ab.py`, 5 interleaved
30 s closed-loop runs per arm at concurrency 8, warmed engines, `--cuda-graphs`, medians
across runs with `spread = (max - min) / median`; a verdict is "unresolved" when the
median change is within the spread. Kernel sweeps: CUDA events, median of 30.

## Phase 2b — `fused_step` (decode rows + chunk rows in one forward)

### Run 1, commit `ae1eb71` (warmup covered chunk rows {1,2}, contexts ≤ 1024)

| profile | arm | gap | ITL p50 | ITL p99 | prefill step p50 | fused GPU p50 | TTFT p50 |
|---|---|---|---|---|---|---|---|
| chat | separate_forwards | 21.32 | 24.10 | 38.48 | 24.88 | — | 406 |
| chat | fused_forward | 18.06 | 19.88 | 33.13 | 20.65 | 20.15 | 318 |
| long | separate_forwards | 25.23 | 27.28 | 64.70 | 26.29 | — | 3080 |
| long | fused_forward | 21.89 | 22.55 | 126.85 | 23.05 | 27.14 | 2901 |

Verdicts (fused vs separate): chat prefill step −17.0% (spread 6.8%), prefill penalty
−30.1% (9.9%), decode step +1.2% unresolved, ITL p999 +148.9% (63.8%); long prefill step
−12.3% (6.3%), penalty −27.3% (13.7%), decode step +3.3% unresolved. host_stage +66/+92%
and sync +104/+117% are accounting (one phase stages both buffers; one sync waits for
the whole forward). First divergence vs stock: separate [None×4], fused [None, 18, None, None].
Tail regression traced to graph capture inside the timed window (2048/4096 contexts).

### Run 2, commit `2dcecfd` (warmup covered all fused shapes to 2048; `lazy_graph_captures` reported)

| profile | arm | ITL p50 | p99 | p999 [min,max] | captures in window / run |
|---|---|---|---|---|---|
| chat | separate_forwards | 24.16 | 38.4 | 56.0 [51,56] | 0 |
| chat | fused_forward | 19.83 | 33.0 | 51.4 [50,52] | 0 |
| long | separate_forwards | 27.19 | 64.5 | 114.2 [68,153] | 1–3 |
| long | fused_forward | 22.64 | 44.6 | 150.9 [63,241] | 3–8 |

Verdicts: chat prefill step −17.6% (5.8%), penalty −30.8% (8.4%); long prefill step
−12.6% (6.7%), penalty −26.2% (15.2%); decode step +1.0% / +3.3% unresolved.

### Run 3, commit `f87416f`, long profile only (warmup contexts sized to pool capacity)

| arm | ITL p50 | p99 | p999 [min,max] | captures in window |
|---|---|---|---|---|
| separate_forwards | 27.28 | 49.5 | 68.5 [67,80] | 0 |
| fused_forward | 22.38 | 43.9 | 60.2 [58,103] | 0 |

Verdicts: prefill step −13.9% (5.4%), penalty −27.7% (16.0%), decode step +0.6%
unresolved, TTFT −16.5% unresolved (74% spread). `check_hooks`: problems [].

**Decision: `fused_step=True` default.**

## Phase 2c — decode kernel, warmup, prefill graphs (commit `c7afabb`)

### `paged_decode_regime_sweep.py --kernel both` — rank-3 `[REP, BLOCK_N, D]` GQA kernel

| context | batch | per_head best (ms, cfg) | gqa best (ms, cfg) | gqa/per_head |
|---|---|---|---|---|
| 128 | 1 | 0.0805 128x4 | 0.1357 64x8 | 1.69 |
| 128 | 4 | 0.0840 128x4 | 0.1323 64x8 | 1.57 |
| 128 | 8 | 0.1085 128x4 | 0.1646 32x8 | 1.52 |
| 128 | 16 | 0.1189 128x4 | 0.1869 32x4 | 1.57 |
| 256 | 1 | 0.0887 128x4 | 0.2007 64x8 | 2.26 |
| 256 | 4 | 0.1043 128x4 | 0.2022 32x8 | 1.94 |
| 256 | 8 | 0.1325 128x4 | 0.2583 32x8 | 1.95 |
| 256 | 16 | 0.1495 128x4 | 0.2485 32x4 | 1.66 |
| 512 | 1 | 0.0897 128x4 | 0.1669 32x8 | 1.86 |
| 512 | 4 | 0.1132 128x4 | 0.1747 64x8 | 1.54 |
| 512 | 8 | 0.1357 128x4 | 0.2114 32x8 | 1.56 |
| 512 | 16 | 0.2129 128x4 | 0.3340 16x4 | 1.57 |
| 1024 | 1 | 0.1024 128x4 | 0.2612 32x8 | 2.55 |
| 1024 | 4 | 0.1411 128x4 | 0.2704 64x8 | 1.92 |
| 1024 | 8 | 0.2081 128x4 | 0.3676 32x8 | 1.77 |
| 1024 | 16 | 0.3641 128x4 | 0.6675 16x4 | 1.83 |
| 2048 | 1 | 0.1647 128x4 | 0.4644 64x8 | 2.82 |
| 2048 | 4 | 0.2150 128x4 | 0.4863 32x8 | 2.26 |
| 2048 | 8 | 0.3496 128x4 | 0.7268 32x8 | 2.08 |
| 2048 | 16 | 0.6590 128x4 | 1.3761 16x4 | 2.09 |

### `ab.py --setting decode_kernel` (gqa_shared vs per_head, rank-3 kernel)

| metric | chat | long |
|---|---|---|
| expected gap | +40.8% (6.7%) | +66.6% (8.6%) |
| ITL p50 | +35.2% (3.9%) | +63.6% (31.7%) |
| decode step p50 | +76.5% (3.8%) | +129.8% (15.0%) |
| decode GPU p50 | +79.8% (4.1%) | +135.4% (15.2%) |
| prefill step p50 | +34.9% (3.2%) | +65.4% (10.2%) |
| operating point | batch ~4.7, ~680 tokens; ITL 19.92 → 26.94 ms | batch ~2, ~1880 tokens; ITL 23.28 → 38.08 ms |

### `ab.py --setting warmup` (warmed vs cold_start, chat, 5 x 15 s)

ITL p99 −71.4% (7.5%), p999 −63.1% (6.0%), `lazy_graph_captures` −100%; gap +1.0%,
ITL p50 −0.2%, decode step +0.1%, prefill step +0.4% — all unresolved. Operating point
batch ~5, ~670 tokens, ITL p50 19.8 ms both arms.

### `ab.py --setting prefill_graphs` — refused by the identity gate

first divergence vs stock: prefill_eager [4, None, None, 13]; prefill_graphed [None, 18, None, None].

## Phase 2c follow-up (commit `9a9f86d`)

### `scripts/token_margins.py` — stock top-2 logit margins

| prompt | position | margin | note |
|---|---|---|---|
| 0 | 4 | 0.0078 | divergence position; one fp16 ulp |
| 0 | 10 | 0.0000 | exact tie (no arm happened to flip it) |
| 1 | 18 | 0.0156 | divergence position |
| 3 | 13 | 0.0156 | divergence position |
| 2 | (smallest) | 0.1406 | never diverged |

### `paged_decode_regime_sweep.py --kernel both` — rank-2 two-head unroll

| context | batch | per_head best | gqa best | gqa/per_head |
|---|---|---|---|---|
| 128 | 1 | 0.0800 128x4 | 0.0850 128x4 | 1.06 |
| 128 | 4 | 0.0855 | 0.0834 | 0.97 |
| 128 | 8 | 0.0999 | 0.0959 | 0.96 |
| 128 | 16 | 0.1373 | 0.1311 | 0.95 |
| 512 | 1 | 0.1137 | 0.1270 | 1.12 |
| 512 | 4 | 0.1498 | 0.1513 | 1.01 |
| 512 | 8 | 0.2214 128x8 | 0.1966 | 0.89 |
| 512 | 16 | 0.2638 | 0.2593 | 0.98 |
| 1024 | 1 | 0.1341 | 0.1521 | 1.13 |
| 1024 | 4 | 0.1429 | 0.1412 | 0.99 |
| 1024 | 8 | 0.2066 | 0.2066 | 1.00 |
| 1024 | 16 | 0.3622 | 0.3552 | 0.98 |
| 2048 | 1 | 0.2295 128x8 | 0.2321 | 1.01 |
| 2048 | 4 | 0.2952 | 0.2876 | 0.97 |
| 2048 | 8 | 0.4225 | 0.4183 | 0.99 |
| 2048 | 16 | 0.7208 | 0.7057 | 0.98 |

**Decision: `decode_attention="per_head"` stays; the second read is served by L2.**

### `ab.py --setting prefill_graphs --allow-token-drift` (prefill_graphed vs prefill_eager, chat)

expected gap −51.3% (11.6%), ITL p50 −57.4% (7.9%), ITL p99 −36.4% (9.6%), p999 −21.8%
unresolved, prefill step −56.7% (8.0%), prefill GPU −60.5% (6.9%), fused GPU −57.2%
(8.3%), prefill penalty −73.0% (10.7%), decode step +8.6% (5.8%; operating-point shift),
sync +1368% (accounting), host stage −1.5% unresolved.

## CUDA test gates

`c7afabb`: kernels/cache/correctness/server 138 passed; batching/reliability 41 passed,
3 failed on a test-side page-accounting assertion (fixed `59dc024`).
`b64af08`: batching/reliability 44 passed.

## Earlier phases (1, 1b, 1c, 2a hooks)

Transcribed into the journal entries "Phase 1b on Kaggle T4", "Kernel-level truth vs
engine-level result", and "SDPA under graphs" at the time; the run folders
(`20260921_e709d4e`, `20260921_9f1f828`, and earlier) are on the Kaggle disk only.

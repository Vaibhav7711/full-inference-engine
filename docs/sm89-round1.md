# sm_89 round 1 — what was built blind, and the gate order for first boot

Everything in this round was written on a machine with no CUDA and no Triton. Nothing
here has been launched. The design response to that is not caution in the code but
*order in the gates*: every kernel has a pure-torch reference pinned by CPU tests, every
Triton kernel is structurally a copy of one already proven on a card, and the first
command on the 4060 is a test run, not a benchmark.

Round 2 — CUDA C++ kernels and the 25 MB L2 — starts only after every gate below is green.

## The card, measured (`results/rtx4060/roofline.json`)

| | T4 | RTX 4060 |
|---|---:|---:|
| gemv bandwidth | 258 GB/s | **257 GB/s** |
| weight-only decode floor, Qwen3-0.6B | 4.6 ms | 4.63 ms |
| L2 | 4 MB | **25.2 MB** |
| `tl.dot` → `mma.sync` | no | **yes** |
| FP8 | no | **yes** |
| SMs | 40 | 24 |

Decode cannot get faster here at the same byte count. The three levers on this path are
fewer bytes, tensor-core prefill, and L2-resident working sets. Round 1 builds the first
two; round 2 is the third.

## What round 1 adds

| item | where | status |
|---|---|---|
| `verify_attention` — speculative verification on an in-place paged kernel instead of the prefill path's prefix gather; `tiled` (16-row tile) from sm_80, `per_token` below, `"prefill"` as the A/B baseline | `policy._verify_default`, engine `__init__`, `_set_prefill_context(backend=…)`, `_speculative_rows` | wired, CPU-tested, **not launched** |
| `kv_cache_dtype="fp8"` — E4M3 storage with per-(token, head) fp16 scales, dequantising decode/prefill kernels, sm_89 gate | `engine/kernels/fp8_format.py` (Triton-free contract), `fp8_paged_kv.py` (kernels) | reference pinned on CPU; kernels **not launched** |
| architecture tiers by gate, not brand | `engine/backends/arch.py`, `results/sm75/`, `results/sm89/` | done |
| A/B settings: `verify_kernel`, `kv_dtype_fp8`, `kv_dtype_all`; the harness skips an arm the registry refuses | `benchmarks/reliability/ab.py` | done |
| W4A16 linear — INT4 weights packed split-half per 128-group, fp16 activations, one scale per (channel, group); two `tl.dot`s per group, dequant in registers | `engine/kernels/w4a16_format.py` (Triton-free contract), `w4a16_linear.py` (kernel) | contract pinned on CPU; kernel **not launched**; not yet installed on model modules — standalone, like `w8a16_linear` |

Not in round 1, deliberately: the fused-step mechanism (needs a profile on the card),
graph replay counters (needs the card), any CUDA C++ (round 2).

## Gate order on first boot

Run `scripts/sm89_round1.sh`. It stops at the first failure. In order:

1. **Static checks and CPU suite.** `pytest -q -m "not cuda"`. Catches anything the
   blind write got structurally wrong before a kernel compiles.
2. **CUDA suite.** `pytest -q -m cuda`. The FP8 kernels are compared against the fp16
   kernels over the *dequantised* pages — a failure here is kernel arithmetic or
   addressing, not quantisation error. The verify path is exercised by the speculative
   tests.
3. **Backend table.** `scripts/check_hooks.py --backends-only`. Confirms `tiled` and
   `flash` are available, FP8 is available, and every refusal names its reason.
4. **Live token gate.** `check_hooks.py` with the sm_89 defaults. Zero problems, or stop.
5. **Then, and only then, the A/Bs** — each a separate `ab.py` invocation, each gated
   against stock Transformers before timing:
   - `--setting prefill_kernel` (long profile): the oldest open question — `tiled` on
     tensor cores against `sdpa` and `per_token`.
   - `--setting verify_kernel` (chat, with the n-gram proposer live): the three
     verification kernels.
   - `--setting kv_dtype_all` (chat and long): fp16 vs INT8 vs FP8 storage.
   - `--setting decode_kernel`, `--setting decode_split_k`: re-measure the T4
     rejections on a card with 6x the L2.
6. **Roofline**, for the record: `benchmarks.kernels.roofline` with the measured ITL.

## Reading the results

Expect, and write down whichever way it goes:

- `tiled` wins prefill on this card or it does not. Either answer closes a question that
  has been open since the T4 PTX dump.
- FP8 KV changes the decode step by close to nothing *at small context* — the weight read
  dominates — and starts to matter as context grows. Report the crossover, not a single
  number.
- `verify_kernel`: the gather path should lose by an amount that grows with context. If
  it does not, the SDPA cache is hiding the gather and the test needs longer prefixes.
- Every `unresolved` is a result. Every token-gate refusal is a bug, not noise.

## Results so far (ci/rtx, RTX 4060, torch 2.14+cu130, Triton 3.8, Python 3.14)

**Gates** — `0001` failed on four tests (dummy-slot reservation one short for a capture
with no live rows; two tests leaking 1.88 GB pools; FP8 writer tests demanding bit-exact
casts). Fixed from the logs in `9b9090d`; `0003` then passed every gate: 361 CPU, 205 CUDA,
backend table, live token gate with 90 fused graphs captured and `prefill_graph_unsupported`
empty. FlashAttention is not importable in this venv, so the measured `flash` default fell
back to `sdpa` with the reason recorded.

**`prefill_kernel`, long profile (~1,824-token prompts), concurrency 8, 3 × 30 s** — all
three arms token-identical to stock:

| arm | expected gap | prefill step p50 | prefill GPU p50 | ITL p99 |
|---|---:|---:|---:|---:|
| `per_token` | 31.22 ms | 33.46 ms | 34.82 ms | 60.2 ms |
| `sdpa` | 17.28 ms | 17.75 ms | 14.12 ms | 26.9 ms |
| **`tiled`** | **15.78 ms** | **16.20 ms** | **12.44 ms** | **20.9 ms** |

`tiled` vs `sdpa`: prefill GPU **−12%**, prefill step **−8.7%**, gap **−8.7%**, ITL p99
**−22%**, spreads 0.4–1.5%. The kernel that ran 3× *slower* on the T4 because `tl.dot`
lowered to scalar FMA wins on sm_89 — the architecture-scoped conclusion the PTX dump
predicted, now measured in the engine. `MEASURED[89].prefill_attention` is `tiled`, at
16-token pages. FA2 prefill remains unmeasured against it on this box.

**`verify_kernel`, chat profile, concurrency 8, n-gram proposer live** — unresolved on
every end-to-end metric at ~620-token contexts (`per_token` decode step −12.7% at a 12.1%
spread is the only resolved cell). Two things to carry forward: the gather-vs-in-place cost
should scale with context, so this needs the long profile to show; and **ITL is not a valid
metric under speculation** — p50 medians of 0.001 ms with 213% spread, because accepted
drafts emit several tokens per step and the gap series records zeros. The harness needs
per-emitted-token accounting before any speculative A/B is quoted.

**`kv_dtype_all`** — crashed 15 min in: `engine/cache/pool_cache.py` dispatched on
"is there a scale pool" and assumed INT8, handing the FP8 pool to the INT8 writer. Fixed to
dispatch on storage dtype, pinned by a Triton-free test; re-run queued as `0004`.

## Memory on an 8 GB card

The T4 attribution configuration (16,384 KV tokens, 7-bucket warm-up) does not fit:
weights 1.2 GB + KV 1.75 GB + graph pool ~4 GB (pre-fix). Size `--num-blocks` to what
`check_hooks` reports free after warm-up, and record the pool size in every result. FP8
KV halves the second term; that is the point.

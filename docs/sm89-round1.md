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
| W4A16 linear (INT4 weights, fp16 activations, group scales) | `engine/kernels/w4a16_linear.py` | see commit log |

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

## Memory on an 8 GB card

The T4 attribution configuration (16,384 KV tokens, 7-bucket warm-up) does not fit:
weights 1.2 GB + KV 1.75 GB + graph pool ~4 GB (pre-fix). Size `--num-blocks` to what
`check_hooks` reports free after warm-up, and record the pool size in every result. FP8
KV halves the second term; that is the point.

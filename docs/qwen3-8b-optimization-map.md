# Qwen3-8B on a Tesla T4: every stage, every lever, with the arithmetic

The 0.6B work established the method and the measured facts about this card. This
document applies both to Qwen3-8B — stage by stage through one forward, with the bytes and
FLOPs each stage moves, what the repository already has for it, what is new, and a
**prediction written down before the run**. The run is `scripts/kaggle_qwen3_8b_t4x2.ipynb`
on a Kaggle T4 x2: GPU 0 serves at W4A16 (`load_model(quantize="w4a16")`, no fp16 copy ever
assembled), GPU 1 is reserved for the reference and the draft model. The identity gate's
reference is the same packed weights through torch (`reference_mode`), so it tests kernels.

## 1. The model and the card

Qwen3-8B (`config.json`): 36 layers, hidden 4096, intermediate 12288, 32 query heads, 8 KV
heads, head_dim 128, vocab 151,936, **untied `lm_head`**, native bf16 (the T4 has no bf16
tensor cores, so fp16), per-head `q_norm`/`k_norm` (RMSNorm over head_dim, before RoPE).

| per layer | params | fp16 bytes |
|---|---:|---:|
| `q_proj` 4096→4096, `k_proj`/`v_proj` 4096→1024, `o_proj` 4096→4096 | 41.9 M | 83.9 MB |
| `gate_proj`/`up_proj` 4096→12288, `down_proj` 12288→4096 | 151.0 M | 302.0 MB |
| norms, QK-norm | ~0 | — |
| **layer** | **192.9 M** | **385.9 MB** |
| × 36 layers | 6.945 B | 13.89 GB |
| embedding table (gathered, not swept) | 622 M | 1.245 GB |
| `lm_head` (read in full every step) | 622 M | 1.245 GB |
| **total** | **8.19 B** | **16.4 GB** |

Tesla T4, as measured in this repository: **258 GB/s** achieved (gemv probe), 40 SMs, 4 MB L2,
~15.1 GB usable, sm_75 — `tl.dot` lowers to scalar FMA, no bf16, no FP8; INT8 tensor cores
exist (130 TOPS) but Triton cannot lower INT8 MMA on sm_75, so INT8 GEMM means `torch._int_mm`.

### What a decode step costs, before any kernel is written

Bytes swept per decode step in fp16: 13.89 GB (layers) + 1.245 GB (`lm_head`) = **15.1 GB →
58.6 ms at 258 GB/s**. That is the floor at fp16, and nothing in the kernel repertoire moves it.
The shares:

| stage | bytes/step (fp16) | share |
|---|---:|---:|
| FFN (`gate`/`up`/`down`) | 10.87 GB | **71.9%** |
| attention projections (`q`/`k`/`v`/`o`) | 3.02 GB | 20.0% |
| `lm_head` | 1.25 GB | 8.2% |
| KV read, batch 8 × 2,048 ctx | 2.4 GB | (+16%) |

KV per token: 2 × 36 × 8 × 128 × 2 B = **144 KiB** (INT8: 73 KiB). A 60,000-token pool is
8.4 GB.

Prefill FLOPs: 2 × 6.945 B = **13.9 GFLOP per token** through the layers, plus attention at
0.59 MFLOP × context per token (8% of the total at 2k context, 26% at 8k, 58% at 32k). A
512-token prompt is 7.1 TFLOP — **~110 ms at the T4's 65 TFLOPS peak, ~200 ms realistically**.
At 0.6B the same prompt was ~15 ms of GEMM. Prefill is compute-bound here in a way it never
was at 0.6B.

### The two consequences everything below follows from

1. **Weight bytes are 14× larger; everything else is 1.3–4× larger.** KV/token 1.3×, hidden
   4×, launches 1.3×. An optimization that saves *weight reads* scales 14×; one that saves
   launches or activation bytes does not. This reorders the whole table from the 0.6B run.
2. **It does not fit in fp16.** 16.4 GB of weights on a 15 GB card. Nothing runs until the
   weights are quantised *in the model*, not in a standalone benchmark.

## 2. Memory budget, one T4

| | GB |
|---|---:|
| CUDA context + Triton/cuBLAS workspaces | 0.3 |
| weights, W4A16 (0.516 B/param with fp16 group-128 scales): layers 3.58 + `lm_head` 0.32 | 3.9 |
| embedding table, fp16 (gathered; could be INT8) | 1.25 |
| CUDA graphs, lean set (~50 × ~10 MiB) + private pool | 0.8 |
| activations, prefill workspace, SDPA gather at 8k context | 0.3 |
| **KV pool** (what is left) | **~8.5 → ~60,000 tokens fp16, ~118,000 INT8** |

Sixteen concurrent 3,700-token requests, or 32 at 1,800. W8A16 instead of W4 costs 3.5 GB of
that pool — 24,000 tokens. The per-graph executable grows with layers: 7.7 MiB at 28 →
**~10 MiB at 36**, so the 0.6B's 215-graph set would be 2.1 GB here; the lean set is required.

## 3. Stage by stage

Shares are of the decode step unless stated. "Have" is what the tree ships today.

### 3.0 Load

**Have:** `from_pretrained(...).to(cuda)` — materialises 16.4 GB in host RAM and again on the
device. Fails on Kaggle's host memory before the GPU is involved.
**Need:** safetensors mmap, per-layer quantise-on-load straight to device, or a pre-quantised
checkpoint (Qwen publishes AWQ/GPTQ-Int4 variants; loader for their packed format). No fp16
copy of the model ever exists.
**Prediction:** gate, not a speedup. Without it nothing else runs.

### 3.1 Embedding gather

622 M params, one row per token. Negligible bytes, one launch.
**Have:** stock. **Need:** nothing. Keep fp16; INT8 table would recover 0.6 GB of pool if memory
binds before compute does.

### 3.2 Input RMSNorm (+ residual)

Activation 4096 wide × tokens. At decode batch 16: 128 KB. Nothing.
**Have:** Triton RMSNorm, fp32 reduction, installed structurally (covers `q_norm`/`k_norm`
too — they are `Qwen3RMSNorm` with a 1-D weight). **Need:** nothing.
**Prediction:** its 0.6B share shrinks ~3× (activations ×4, step ×14). Residual-add fusion:
not worth a kernel.

### 3.3 Attention projections — 20% of bytes

`q/k/v` are three GEMVs at decode (M = batch ≤ 16, N = 4096/1024/1024, K = 4096), `o` one more.
**Have:** stock `nn.Linear` → cuBLAS. **Need:**
- **W4A16** on all four (`install_w4a16`, structural). Saves 2.3 GB/step → **9 ms**.
- **Fused QKV** as one `[4096 → 6144]` GEMV: no byte saving, one launch instead of three, and
  one weight sweep with a wider N — cuBLAS GEMV at N=1024 is launch-bound at M ≤ 16. Under
  graphs the launch is free; the sweep efficiency is the question.
**Prediction:** W4 is the row; QKV fusion within noise under graphs, measurable without.

### 3.4 QK-norm + RoPE

Two RMSNorms over `[B, H, T, 128]` and the rotation. Tiny, three launches per layer × 36.
**Have:** Triton RMSNorm (covers both norms) and fused Q/K RoPE. **Need:** a single
norm+norm+rope kernel would remove ~72 launches/step — under graphs, nothing.
**Prediction:** unresolved under graphs, as at 0.6B.

### 3.5 KV write

Batch × 144 KiB per step. **Have:** paged write kernels fp16 and INT8. **Need:** nothing new;
INT8 KV needs the stock-gate question from `ci/rtx` 0006 answered (quantisation near-tie or
bug) before it is admissible.
**Prediction:** INT8 KV halves the KV read — at batch 8 × 2k that is **−4.6 ms on a ~30 ms
step**, and doubles pool capacity. At 0.6B it was measured where KV was 9% of bytes; here it
is 16% and rising with context. First place its benefit should resolve.

### 3.6 Decode attention

GQA group is **4** (32/8), not 2. `per_head` launches one program per (row, query head) and
reads each KV tile four times.
**Have:** `per_head` (at 98–101% of roofline on both cards at 0.6B), `split_k`, `gqa` — which is
gated to "group of exactly 2" and must be generalised. **Need:** `gqa` for group 4 (rank-2
four-head unroll; register pressure is the risk).
**Prediction:** at 0.6B the duplicate read was served by L2 — not by capacity (the per-layer
working set already exceeded 4 MB) but by *launch adjacency*: programs for the same KV head
are neighbours in the grid and hit L2 within microseconds. That holds at group 4. Prior:
`gqa` stays ~1.0×. If it does not, the KV term is up to 4× larger than the table above and
this becomes the biggest attention item on the card. Worth one sweep.
`split_k`: batch 1 at long context, as before.

### 3.7 Prefill attention

Compute-bound: 0.59 MFLOP × context per token, 8% of prefill at 2k, 58% at 32k.
**Have:** SDPA over gathered pages (tensor cores on sm_75; the T4's best), `per_token` (INT8
pool), `tiled` (refused on sm_75, correctly). **Need:** nothing for ≤ 8k contexts. The SDPA
gather — one copy of the prefix K/V per layer per chunk — scales with context × 36 layers;
at 8k context and 4 chunk rows that is 128 MB per layer per step, ~0.5 ms, which is fine.
**Prediction:** SDPA stays; attention share of prefill small below 8k.

### 3.8 FFN — 72% of bytes, the whole game

Three GEMVs: `gate`/`up` `[4096 → 12288]` each, `down` `[12288 → 4096]`. 302 MB per layer, 10.9 GB
per step at fp16. **Have:** stock cuBLAS; Triton SwiGLU (saves nothing under graphs);
standalone `w8a16_linear` (rejected at 0.6B on a harness with L2-resident weights and an
8-CTA grid — not admissible) and `w4a16_linear` (never launched). **Need:**
- **W4A16 installed** on all three. 10.9 → 2.8 GB. **−31 ms per step.** The single largest
  item in this document by a factor of three.
- **Fused `gate`+`up`** as one `[4096 → 24576]` sweep: one launch, one pass; no bytes saved.
- **W8A8 for prefill only** via `torch._int_mm` (INT8 tensor cores, 130 TOPS vs 65): at
  M = 128–512 it is a real GEMM, not the padded M ≤ 16 case Phase 10 measured. Up to 2× on the
  prefill GEMMs, which are ~85% of prefill. Needs a second weight copy (W8) or W8A8 everywhere
  (7.6 GB instead of 3.9), plus per-token activation quantisation and an accuracy gate.
**Prediction:** W4A16 FFN is 3.5× the next item. The 0.6B microbenchmark that rejected W8A16
becomes honest here by itself — a 100 MB weight per GEMM cannot be L2-resident, so the
measurement finally sees the bytes it is supposed to save.

**The T4 caveat, handled in code:** `tl.dot` lowers to scalar FMA on sm_75, so the Triton
dequant-GEMV is bandwidth-bound only at small M. At M = 16 the FFN is ~220 GFLOP per step,
~27 ms on FMA (~8 TFLOPS) against the 15 ms read floor; at prefill widths it is hopeless.
`W4A16Linear` sends any call with M ≥ 32 to cuBLAS over a dequantised fp16 tile instead —
tensor cores, same read bytes, one 100 MB transient per layer. The decode A/Bs at
concurrency 8 sit below the threshold and measure the kernel; prefill measures cuBLAS. A
sweep of `dense_threshold` on this card is a cheap follow-up: the crossover is a
measurement and 32 is an estimate.

### 3.9 `lm_head` — 8% of bytes, untied

`[4096 → 151,936]`, 1.25 GB fp16, read in full every decode step. **Have:** only last-position
rows at prefill (done; HF computes all positions — watch GPU 1's memory). **Need:** W4 or W8
on the head. **Prediction:** −3.6 ms/step at W4; small accuracy risk on the output
distribution, measure perplexity.

### 3.10 Sampling

argmax over 151,936 × batch. **Have:** batched sampler with greedy short-circuit. Nothing.

### 3.11 The step: scheduler-level optimizations

| optimization | 0.6B verdict | 8B prediction | why |
|---|---|---|---|
| **fused decode+prefill step** | regression (eager copies outweighed a 4.6 ms saving) | **large win** | saves one 15 GB weight read (59 ms, or 15–19 ms at W4) per prefill-carrying step; its copies grow 4×, its saving 14× |
| **chunked prefill**, chunk size | ±2× SLO dial | **sharper by 14×** | a 128-token chunk is ~50 ms of compute here; every decoder waits through it |
| **prefix cache** | unresolved (no shared prefixes) | **grows** | reused tokens cost 13.9 GFLOP each to recompute |
| **preemption / recompute** | survives 3× oversubscription | **more binding** | 144 KiB/token, 8.5 GB pool: the ceiling arrives sooner |
| **CUDA graphs** | 3.5× on the decode step | **~1.3×** | ~30 ms of launches over a 59 ms (fp16) or ~20 ms (W4) step — your vLLM-decay curve, one model further. Still required: without them the step is 30 + 20 |
| **warm-up** | moves captures out of the window | required | 36-layer captures ×2 forwards each: ~30 s for the lean set |
| **speculative decoding**, 0.6B draft | never produced a number | **grows** | 13:1 size ratio; a verify step at W4 costs ~20 ms of weights for `depth+1` tokens; draft on GPU 1 |

### 3.12 Cross-cutting

- **Graph count**: lean set only (`capped` by served max context, fused rows ≤ 2, prefill rows
  ≤ 4, replay counters). 215 graphs is 2.1 GB here.
- **Warm-up time**: scales with layers; the lean set keeps it under a minute.
- **Measurement**: the identity gate's reference is **HF running the dequantised W4 weights**
  on GPU 1, so the gate stays bit-exact and tests kernels, not quantisation. Quantisation
  error is measured separately: perplexity on a held-out set against the fp16 model, which
  must run on GPU 1 in layers (it does not fit either).

## 4. Projected step, after the work

W4A16 everywhere, INT8 KV, lean graphs, one T4:

| operating point | weights | KV | ≈ step | ≈ tok/s |
|---|---:|---:|---:|---:|
| batch 1, short | 15 ms | ~0 | ~18 ms | ~55 single-stream |
| batch 8 × 2k ctx | 15 ms | 4.7 ms | ~24 ms | ~330 |
| batch 16 × 2k ctx | 15 ms | 9.3 ms | ~29 ms | ~550 |
| prefill, 512 tokens | — | — | ~200 ms | — |

Against vLLM on the same card: vLLM's AWQ path on sm_75 uses its older kernels (Marlin needs
sm_80), so the comparison is fair in both directions and worth running at matched KV.

## 5. Order of work

1. `install_w4a16(model)` + pre-quantised checkpoint loader + mmap load — **the gate**.
2. KV pool sized from free memory; lean graph set in the engine; `max_model_len` honest at
   admission.
3. Gates on 8B: CPU suite, CUDA suite, `check_hooks`, identity vs dequantised reference.
4. Memory attribution by stage (the part-4 notebook, pointed at 8B).
5. A/Bs in the order of the predictions: `fused_step`, `prefill_chunk`, `kv_dtype` (INT8),
   `cuda_graphs`, `prefix_cache`, the three Triton fusions, `decode_kernel` with `gqa` at
   group 4, `decode_split_k`.
6. Speculative: 0.6B draft on GPU 1, batch 1 → 8 crossover.
7. Stretch: W8A8 prefill GEMMs via `_int_mm`; INT8 embedding table.

Every prediction above is falsifiable by one setting in `benchmarks/reliability/ab.py`.
Write the result beside the prediction, whichever way it goes.

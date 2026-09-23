# RTX 4060 plan: what changes when the GPU is Ada and it is yours

Target machine: RTX 4060, 8 GB VRAM (8188 MiB reported by the driver), 16 GB host RAM,
local. Replaces the Kaggle T4 as the measurement platform from 2026-09-22.

## What is different, and what it enables

| | Tesla T4 (sm_75, Turing) | RTX 4060 (sm_89, Ada) | consequence |
|---|---|---|---|
| memory bandwidth | 320 GB/s peak, ~260 achieved | ~272 GB/s (128-bit GDDR6) | **the decode floor does not move**: ~5 ms per step to read 1.2 GB of weights, same as the T4 |
| fp16 tensor throughput | ~65 TFLOPs | ~120+ TFLOPs dense, plus bf16 and FP8 (e4m3) | prefill and the fused step get cheaper; the compute half of a prefill-carrying step shrinks |
| L2 cache | 4 MB | 24 MB | KV tiles for a whole batch fit; the GQA double-read question is even more settled; attention becomes latency-bound sooner |
| `tl.dot` in Triton | lowers to FMA (no `mma.sync`) | lowers to `mma.sync` | **the tiled prefill kernel is finally a candidate**; the PTX gate decides |
| FlashAttention | not supported | FlashAttention-2 imports and executes on sm_89, but its paged-KV API requires 256-token pages | a measured candidate, not a serving default: this engine normally uses 16-token pages |
| torch SDPA backends | memory-efficient, math | flash (no custom mask), memory-efficient, math | our chunk mask still forces memory-efficient; flash only via `flash_attn` directly |
| VRAM | 16 GB | **8 GB** | Qwen3-0.6B fp16 (1.2 GB) + KV is comfortable; Qwen3-1.7B fp16 (3.4 GB) fits with a ~3 GB KV pool; 4B needs INT4 weights; graph pools count |
| session | 12 h, remote, no profiler | unlimited, local, **Nsight Systems / Nsight Compute** | kernel-level truth (achieved bandwidth, occupancy, stall reasons) instead of inference from timings |

Two things to check on day one before believing anything: the card's actual bandwidth
(`benchmarks/kernels/roofline.py`) and whether the OS is Linux/WSL2 - Triton and
flash-attn are Linux-only in practice; native Windows is not a supported path.

## Model choice

Keep **Qwen3-0.6B** as the continuity model: every T4 number exists for it, so the
first RTX run is a like-for-like re-baseline and every later A/B has a T4 twin.

Add **Qwen3-1.7B** as the primary serving target once the re-baseline is in: same
architecture family (the fusion installers and kernels need no change; 28 layers, 8 KV
heads, head_dim 128 → identical per-token KV size), 2.8x the weights so the decode step
is ~3x longer and per-step overheads stop dominating - closer to real serving. Memory
budget on this 8 GB card: 3.4 GB weights + ~0.3 GB activations/graph pools + KV. At
114 KB/token, a ~3 GB KV pool is ~25.6k tokens: start at `num_blocks=1600`,
`max_active=8` with ~3.2k-token contexts. This leaves room for fragmentation but still
requires the soak's KV-pressure tests to set the final value.

**Qwen3-4B** only as the quantization target (Tier 2 item 6): fp16 does not fit; W4A16
(2.5 GB) does. Do not start there.

## Phases (each is one local session; A/B protocol unchanged)

### Phase R0 - environment and gates (evening 1)
- Linux/WSL2, CUDA 12.x driver, `torch` cu12 wheel, `triton`, `flash-attn` wheel for
  sm_89, `nsight-systems`/`nsight-compute`.
- `python -m pytest -q` (CPU) and `python -m pytest -q -m cuda` (all suites; the speculative
  suite too - memory is the only reason it was excluded on Kaggle and 8 GB may still exclude it
  again).
- `python scripts/check_hooks.py --out results/rtx4060/check_hooks.json` - warmup, graphs,
  phases, zero lazy captures.
- `benchmarks/kernels/roofline.py` - measured bandwidth; this number replaces 258 GB/s in
  every floor calculation.
- `benchmarks/kernels/prefill_attention_ab.py --ptx-only` - **expect `mma_sync > 0`**,
  no spills. If not, the tiled kernel's tile defaults (`engine/kernels/device.py`) need
  re-deriving before anything else.

#### R0 result — 2026-09-23

- Observed hardware: RTX 4060 (sm_89), 8188 MiB reported by the driver; PyTorch reports
  24 SMs and 25 MB L2. Linux, CUDA toolkit 13.1, PyTorch 2.14.0+cu130 and Triton 3.8.0
  are working together. Nsight Systems and Nsight Compute are installed.
- Test gates: the non-CUDA suite passes (290); CUDA passes 192 tests with one expected
  8 GB skip (the two-model FP32 speculative parity test) and one strict expected failure.
  The expected failure is INT8 paged-KV recompute/preemption changing a greedy token on
  sm_89; keep INT8 KV disabled until its Ada correctness repair and A/B are complete.
- Hook gate passes: six decode, 28 prefill and 126 fused-step graphs were captured during
  warm-up, with zero lazy captures and no reported problems. The unmeasured safe defaults
  are `per_head` decode and SDPA prefill; tiled prefill is deliberately opt-in until R1.
- Roofline: decode-like FP16 GEMV reached **257.4 GB/s**, yielding a Qwen3-0.6B
  weight-only floor of **4.63 ms/token**. See `results/rtx4060/roofline.json`.
- PTX gate: tiled prefill emits 32 `mma.sync` instructions with zero spills. Its K/V loads
  are still scalar, so it is structurally viable but must be tuned and measured in R1.
- FlashAttention probe completed. The PyPI 2.8.3 source build is incompatible with PyTorch
  2.14 because it forces C++17; current upstream 2.8.4 source compiled as an sm_89-only,
  Qwen head-dimension-128 inference build after a private CUDA 13.1/glibc compatibility
  overlay. It imports and `flash_attn_with_kvcache` executes. This installation is a local
  measurement artifact, not a project dependency lock.
- Crucially, the installed API rejects the engine's 16-token KV pages: page size must be
  divisible by **256**. The backend geometry gate now reports that reason and retains the
  safe `per_head`/SDPA defaults for ordinary 16-token serving. Under a 256-token-page test
  layout, both Flash phases are deliberately eager-only because their variable-length
  grouping / split-KV workspace setup are not CUDA-graph safe on this build. See
  `results/rtx4060/check_hooks_flash_block256_full.json`.
- The original combined short-profile A/B changed decode and prefill together and was
  rejected: median ITL was **+11.5%**. Phase isolation below explains why and identifies
  the long-context prefill-only configuration that does win.

### Phase R1 - re-baseline, 0.6B, same settings as T4 (evening 1-2)
Run the notebook's Phase 1/1c/2b/2c A/Bs as plain shell commands (no two-GPU runner
needed): `cuda_graphs`, `prefill_kernel` (three arms - tiled now competes), `prefill_graphs`,
`fused_step`, `decode_kernel`, `warmup`; decode regime sweep `--kernel both`. Output to
`results/rtx4060/<date>_<sha>/`. Record a T4-vs-4060 table in the journal: which wins
transfer, which do not. Expected: graphs and fused step transfer; SDPA-vs-per_token
margin narrows (compute is cheaper); tiled kernel becomes competitive or wins; GQA kernel
still neutral.

### Phase R2 - FlashAttention-2 over paged KV — completed, prefill-only win

`engine/kernels/flash_paged.py` maps both phases onto
`flash_attn_with_kvcache`. The implementation is now geometry-gated: its paged-KV API
requires `block_size % 256 == 0`, so normal 16-token pages never select it and do not crash.
For valid 256-token pages it runs direct over `[num_blocks, page_size, kv_heads, head_dim]`,
with `cache_seqlens = start + chunk` for prefill. Padded staged chunks are grouped by their
actual query length before the FA call; otherwise the API regards padding as real tokens and
misaligns the causal mask.

The follow-up isolated phases, exposed FA's `num_splits`, swept attention kernels over batch
1/4/8/16 and context 128/512/2048/4096, and removed a host synchronization that had run once
per transformer layer. Row groups now come from the engine's pinned host staging once per
step; equal-length chunks take a direct zero-gather adapter path.

Results:

- Decode remains rejected. FA is only 15% faster at the attention-kernel level for the narrow
  batch-1/context-4096 cell, but its kvcache entry point is not CUDA-graph safe on this build.
  Against graphed `per_head`, short-profile ITL regressed **114%**. Forced split counts did not
  improve the useful cells; upstream auto was retained. See
  `flash_paged_decode_sweep.json` and `ab_flash_decode_block256_short.json`.
- Optimized prefill is accepted for long-context, 256-page FP16 configurations. On Qwen3-0.6B,
  prefill steps improved **7.8%**, prefill penalty **13.4%**, and ITL p99 **22.6%**. On
  Qwen3-1.7B, median TTFT improved **13.7%**, expected gap **5.3%**, fused GPU time **11.8%**,
  and ITL p99 **9.9%**. Both passed the leading-token stock gate in FP16. See
  `ab_flash_prefill_block256_long_graphs_fp16.json` and
  `ab_flash_prefill_qwen3_1.7b_long_graphs_fp16.json`.
- A gathered dense-FA variant permits normal 16-token pages, but lost badly end to end
  (**+38.5%** prefill step, **+52.6%** TTFT); it remains an explicit experimental backend.

Therefore the measured Ada policy is:

- always retain graphed `per_head` decode;
- use SDPA prefill with normal 16-token pages;
- for long-context FP16 deployments that deliberately select 256-token pages, use optimized
  Flash prefill only; do not use BF16 for this path because it failed the early token gate;
- keep dense-gather Flash and Flash decode opt-in until a newer backend changes their results.

The production-shaped configuration is exposed as `create_rtx4060_flash_app`: FP16,
256-token pages, graphed `per_head` decode and eager optimized Flash prefill. External HTTP
verification passes completion, streaming chat, stop/seed determinism, input refusal,
eight-way concurrency and Prometheus metrics (`live_smoke_optimized_flash.json`: 6/6). The
verification also caught and fixed an exact-prefix-cache edge case: sampled requests now
reuse prompt KV without replaying a cached sampled token or skipping an RNG draw.

### Phase R3 - Nsight on the decode step
`nsys profile` one graphed decode step at (batch 8, 1024) and `ncu` on the attention
kernel: achieved DRAM bandwidth, achieved occupancy, warp stall reasons. This replaces
the "~160 GB/s, parallelism-bound" inference with a measurement, and it tells whether
the remaining ~5 ms above the weight floor is attention, the lm_head GEMM, or the ~1,100
small kernels' fixed cost. The next kernel decision comes from this profile, not from
another A/B.

### Phase R4 - Qwen3-1.7B
Re-run R1's A/Bs on 1.7B with the memory budget above; record the per-step split. Tune
`num_blocks`/`max_active` from the soak's KV-pressure results. The chat workload now has
a real decode step (~25-30 ms); check that prefill chunk 128 is still the right point
(`prefill_chunk` A/B) because the compute/bandwidth balance moved.

### Phase R5 - weight-only quantization
Decode is weight-read-bound; the only way under the floor is fewer weight bytes.
- W8A16 first (`engine/kernels/w8a16_linear.py` exists; `benchmarks/kernels/w8a16_linear_ab.py`
  measures it) → install on all projections, gate tokens, A/B on 1.7B.
- W4A16 (GPTQ/AWQ checkpoints exist for Qwen3; a Marlin-style kernel needs sm_80+, which
  we now have) → makes 4B fit. Quality is checked by the identity gate and a small
  perplexity run, not assumed.
- FP8 (Ada has FP8 tensor cores; `torch._scaled_mm`) for the GEMMs if W8A16 shows the
  bandwidth win but leaves compute on the table for prefill.

### Phase R6 - batched speculative decoding
At batch 1-4 on a local card this is the largest latency lever left: verifying 4 draft
tokens costs about one decode step. n-gram draft first (no second model, no VRAM), inside
the batched step (the chunk path already handles "k tokens for one row"); then a
0.6B draft for the 1.7B target if VRAM allows. Acceptance rate and tokens/s per
concurrency level in the journal.

### In parallel, no GPU needed - Tier 1 serving
Batched sampler (temperature/top-k/top-p/penalties/stop/logprobs) inside the graph,
OpenAI-compatible `/v1/chat/completions` with the Qwen chat template and streaming
deltas, Prometheus `/metrics` from `stats_snapshot()`. These make the engine usable
by anything that speaks to vLLM today.

## What to carry over unchanged
The A/B harness, the soak, the identity gate, `lazy_graph_captures`, `check_hooks`,
the journal discipline. Add `results/rtx4060/` beside `results/t4/` and a
`DeviceProfile`-keyed defaults table in `engine/kernels/device.py` so the two GPUs'
regimes coexist (tile sizes, SDPA-vs-tiled-vs-flash default, decode kernel default).

## What not to redo
The T4 negatives (tiled kernel on sm_75, GQA-shared reads, INT8 KV on this workload) are
GPU-specific findings; the tiled kernel and INT8 KV get one fresh A/B on Ada because the
hardware reasons changed, the GQA-shared read does not (a bigger L2 only strengthens it).

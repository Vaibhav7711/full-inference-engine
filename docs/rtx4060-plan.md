# RTX 4060 plan: what changes when the GPU is Ada and it is yours

Target machine: RTX 4060, 6 GB VRAM, 16 GB host RAM, local. Replaces the Kaggle T4 as
the measurement platform from 2026-09-22.

## What is different, and what it enables

| | Tesla T4 (sm_75, Turing) | RTX 4060 (sm_89, Ada) | consequence |
|---|---|---|---|
| memory bandwidth | 320 GB/s peak, ~260 achieved | ~272 GB/s (128-bit GDDR6) | **the decode floor does not move**: ~5 ms per step to read 1.2 GB of weights, same as the T4 |
| fp16 tensor throughput | ~65 TFLOPs | ~120+ TFLOPs dense, plus bf16 and FP8 (e4m3) | prefill and the fused step get cheaper; the compute half of a prefill-carrying step shrinks |
| L2 cache | 4 MB | 24 MB | KV tiles for a whole batch fit; the GQA double-read question is even more settled; attention becomes latency-bound sooner |
| `tl.dot` in Triton | lowers to FMA (no `mma.sync`) | lowers to `mma.sync` | **the tiled prefill kernel is finally a candidate**; the PTX gate decides |
| FlashAttention | not supported | FlashAttention-2 supported, incl. `flash_attn_with_kvcache` over **paged** KV with block tables (decode and chunked prefill in one API) | a production attention backend for both phases, split-K built in |
| torch SDPA backends | memory-efficient, math | flash (no custom mask), memory-efficient, math | our chunk mask still forces memory-efficient; flash only via `flash_attn` directly |
| VRAM | 16 GB | **6 GB** | Qwen3-0.6B fp16 (1.2 GB) + KV is comfortable; Qwen3-1.7B fp16 (3.4 GB) fits with ~1-1.5 GB of KV; 4B needs INT4 weights; graph pools count |
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
budget on 6 GB: 3.4 GB weights + ~0.3 GB activations/graph pools + KV. At 114 KB/token,
1.5 GB of KV is ~13k tokens: `num_blocks=800`, `max_active=8` with ~1.6k-token
contexts. Tight but workable; the soak's KV-pressure tests become realistic rather than
synthetic.

**Qwen3-4B** only as the quantization target (Tier 2 item 6): fp16 does not fit; W4A16
(2.5 GB) does. Do not start there.

## Phases (each is one local session; A/B protocol unchanged)

### Phase R0 - environment and gates (evening 1)
- Linux/WSL2, CUDA 12.x driver, `torch` cu12 wheel, `triton`, `flash-attn` wheel for
  sm_89, `nsight-systems`/`nsight-compute`.
- `python -m pytest -q` (CPU) and `python -m pytest -q -m cuda` (all suites; the speculative
  suite too - memory is the only reason it was excluded on Kaggle and 6 GB may exclude it
  again).
- `python scripts/check_hooks.py --out results/rtx4060/check_hooks.json` - warmup, graphs,
  phases, zero lazy captures.
- `benchmarks/kernels/roofline.py` - measured bandwidth; this number replaces 258 GB/s in
  every floor calculation.
- `benchmarks/kernels/prefill_attention_ab.py --ptx-only` - **expect `mma_sync > 0`**,
  no spills. If not, the tiled kernel's tile defaults (`engine/kernels/device.py`) need
  re-deriving before anything else.

### Phase R1 - re-baseline, 0.6B, same settings as T4 (evening 1-2)
Run the notebook's Phase 1/1c/2b/2c A/Bs as plain shell commands (no two-GPU runner
needed): `cuda_graphs`, `prefill_kernel` (three arms - tiled now competes), `prefill_graphs`,
`fused_step`, `decode_kernel`, `warmup`; decode regime sweep `--kernel both`. Output to
`results/rtx4060/<date>_<sha>/`. Record a T4-vs-4060 table in the journal: which wins
transfer, which do not. Expected: graphs and fused step transfer; SDPA-vs-per_token
margin narrows (compute is cheaper); tiled kernel becomes competitive or wins; GQA kernel
still neutral.

### Phase R2 - FlashAttention-2 over paged KV (the big one)
`flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=..., block_table=...,
causal=True)` accepts our pool layout `[num_blocks, block_size, kv_heads, head_dim]`
directly (FA2's paged KV requires the page size to be a multiple of 16 tokens - ours is
16; verify against the installed version's docstring on day one), handles GQA natively, does split-K for decode,
and takes q of length 1 (decode) or `chunk` (prefill, with `cache_seqlens` = start and
causal within the chunk). One kernel for both phases, no gather, no mask tensor.
- Add `decode_attention="flash"` and `prefill_attention="flash"` dispatch in the three
  attention functions (`_decode_attention`, `_prefill_attention`); the K/V write kernels
  stay ours (FA2 can also write K/V via `k=`/`v=` args - try both).
- Graph-capture safety: FA2 kernels are capture-safe; `cache_seqlens` must be a device
  tensor (it is: `_device_seq_lens`).
- A/B: `decode_kernel` (per_head vs flash) and `prefill_kernel` (sdpa vs tiled vs flash),
  chat and long. Decision rule as always: outside the spread, tokens gated.
- This is also the answer to D19 (split-K): FA2's decode path already splits.

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

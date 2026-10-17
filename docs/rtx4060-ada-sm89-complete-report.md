# RTX 4060 / Ada sm_89 inference-engine R&D: complete report

Date range: 2026-09-22 through 2026-10-17

Hardware target: NVIDIA GeForce RTX 4060, Ada `sm_89`, 8 GB

Models: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B-AWQ

Branch: `rtx/ada-sm89`

This is the canonical synthesis of the RTX 4060 work. It records what was built, how it
was measured, every important positive and negative result, the bugs found during the
experiments, the resulting serving policies, and the remaining publication boundary. Raw
measurements are committed under `results/rtx4060/`; the chronological laboratory journal
is `docs/rtx4060-rnd-priorities.md`.

## 1. Executive result

The engine is a usable single-GPU continuous-batching inference service with paged KV
ownership, CUDA-graph replay, prefix caching, request-level sampling, preemption, streaming,
metrics and OpenAI-compatible HTTP endpoints. The RTX 4060 pass replaced hardware guesses
with measured policy for Ada and added three deployable model classes:

- Qwen3-0.6B and Qwen3-1.7B in a lossless FP16 profile;
- Qwen3-0.6B and Qwen3-1.7B in a faster, higher-capacity, lossy profile using W8A16 weights
  and calibrated INT8 KV;
- Qwen3-4B-AWQ using native packed INT4 weights, graphed decode and a three-tier prefill
  dispatch.

The normal page-16 FP16 policy is graphed Triton paged decode plus tiled paged prefill.
Direct paged FlashAttention prefill remains useful only for deliberately selected 256-token
pages and long prompts. Flash decode is rejected. All lossy paths remain explicit profile
choices rather than silently replacing FP16.

The most important end-to-end result is Qwen3-1.7B at an equal 5 GiB weights-plus-KV budget:

| metric | optimized versus FP16, chat | optimized versus FP16, long/context pressure |
|---|---:|---:|
| throughput | **+38.1%** | **+53.6%** |
| TTFT p50 / p99 | **-54.9% / -54.4%** | **-43.9% / -56.1%** |
| ITL p50 / p99 | **-37.7% / -39.5%** | +30.2% / **-6.5%** |
| decode step p50 | **-38.1%** | **-42.9%** |
| KV token capacity | **2.7x** | **2.7x** |
| preemptions | 0 to 0 | **9 to 0** |

The long-profile median ITL increase is an operating-point effect: the optimized pool holds
all eight long requests concurrently, while the FP16 arm queues or preempts requests. Its
tail ITL, TTFT, throughput and useful concurrent work all improve.

Teacher-forced quality for the stacked 1.7B profile is 97.3% top-1 agreement at 1k context
and 97.7% at 4k, with KL 0.0041/0.0038. This is a measured lossy mode, not an exact FP16
replacement.

## 2. Machine and software facts

| item | measured value |
|---|---|
| GPU | GeForce RTX 4060, Ada `sm_89` |
| SM count | 24 |
| VRAM | 8188 MiB reported by the driver |
| L2 | about 24-25 MB |
| memory bus | 128-bit GDDR6 |
| measured FP16 decode-like bandwidth | 257.4 GB/s |
| Qwen3-0.6B weight-read floor | 4.63 ms/token |
| PyTorch | 2.14.0+cu130 |
| Triton | 3.8.0 |
| FlashAttention | locally built 2.8.4, sm89 inference build |
| host link observed | PCIe Gen4 x1; about 1.39 GB/s D2H and 1.07 GB/s H2D pinned |

The 24 MB L2 changes several T4 conclusions: smaller KV working sets can remain cache-resident,
but 0.6B/1.7B weights still stream from DRAM and decode remains primarily a weight-bandwidth
problem. Ada tensor cores make Triton tiled prefill practical; the generated PTX contains
`mma.sync` and no spills for the accepted path.

The host PCIe link is unusually narrow. Host-spill conclusions are valid for this machine
but must not be generalized to a normally trained x8 link without remeasurement.

## 3. Measurement discipline

Serving decisions were not made from a single kernel benchmark. The acceptance sequence was:

1. establish the physical bound with a roofline or byte model;
2. validate kernel numerics against a stock or explicitly dequantized reference;
3. warm every graph bucket before measurement and require zero unexpected lazy captures;
4. run interleaved repeated end-to-end A/B arms;
5. compare the median change with run-to-run spread;
6. run a stock-Transformers early-token gate and record near-ties separately;
7. for lossy modes, run teacher-forced top-1, KL and perplexity checks;
8. validate allocator/page invariants after pressure, cancellation and preemption;
9. retain rejected implementations and their result JSON so conclusions remain reproducible.

`expected_gap_ms` is the central phase-weighted service metric. ITL, TTFT, throughput,
prefill/decode GPU time, queue delay, preemptions and delivered tokens are also reported.
A change smaller than observed spread is labelled unresolved, even when its median is
favourable. Kernel-only wins never override an end-to-end loss.

One methodology error was explicitly discarded: repeatedly timing a 12.5 MB W8 projection
left it hot in the 24 MB L2 and implied an impossible 540 GB/s. Searches now rotate through
at least eight weight copies so decode weights stream from DRAM.

## 4. Engine architecture delivered

### 4.1 Scheduling and memory

- FCFS continuous batching with bounded admission and cancellation.
- Physical paged K/V pools with per-request block tables.
- Prefix-cache radix ownership, exact greedy decisions and copy-on-write partial tails.
- Correct separation of request owners, cache owners and permanent graph dummy pages.
- Preemption with recompute for ordinary FP16/prefill requests.
- Exact host spill/restore of authoritative generated INT8 pages.
- Metrics for allocations, prefix hits, preemption, recomputed tokens, spill bytes and
  transfer time.

### 4.2 Execution

- Independent registered decode and prefill backends with geometry/device capability gates.
- Decode, prefill and mixed fused-step CUDA graphs, with batch/context/width-aware keys.
- Batched sampling with request-owned RNG state, penalties, stop handling and logprobs.
- Packed fused decode+prefill rows so a mixed step reads weights once.
- Packed-width graph buckets, preventing a small planned chunk from running through a
  full-budget graph.
- Idle whole-prompt fast path when exactly one prompt is waiting and nothing decodes.
- OpenAI-compatible completions/chat, streaming deltas, health/model discovery and
  Prometheus metrics.

### 4.3 Device policy

Backend selection is explicit. A named unavailable backend raises with a reason; it does
not silently fall through and accidentally benchmark something else. The measured sm_89
page-16 fallback is tiled prefill. Flash prefill is selected only for compatible page-256
geometry. Decode remains graph-safe Triton unless explicitly overridden.

## 5. Decode-attention investigation

### 5.1 Live profile

For Qwen3-0.6B, batch 8/context 1024, four warmed graph replays contained 112 paged-attention
launches taking 24.05 ms and 788 CUTLASS GEMMs taking 27.27 ms in the profiled trace. Argmax
was about 0.1% of summed kernel time. Nsight Systems instrumentation changes absolute time,
so the trace was used only to allocate work.

Nsight Compute measured 29.44% achieved occupancy against a register-limited 33.33%
theoretical ceiling. Long-scoreboard stalls dominated. At batch 8/context 256, paired GQA
halved L2 read sectors and won the isolated kernel by 1.14x despite lower occupancy. This
demonstrated why occupancy alone is not an optimization target.

### 5.2 Split-K

At batch 1/context 4096, eight-way split-K reduced isolated attention-layer time from
0.0932 to 0.0647 ms, 1.44x. It lost at batch 1/context 1024 and batch 8/context 1024.

The first full-engine run was invalid: graph capture omitted the split setting and fixed
the context bound at 128, so the supposedly split arm replayed per-head attention. The graph
key now carries split configuration and a context bucket. A trace then confirmed split and
merge kernels on every layer.

Corrected full-engine batch-1/4k results:

- decode GPU: 11.519 to 11.126 ms, **-3.4%**;
- ITL: 12.619 to 12.222 ms, **-3.1%**;
- identical tokens and both arms passed the stock gate;
- one lazy graph capture remained per run.

Eight-way split raised occupancy from 8.33% to 21.81%, but still had no eligible warp on
76.58% of scheduler cycles. A 32x2 tile cut registers/thread from 153 to 96 without improving
latency. Sixteen splits narrowly won the isolated 4k cell but not full-model latency. Moving
the V load ahead of QK/softmax was neutral/slower. Therefore split-K remains a narrow
experimental path, not the default.

### 5.3 Ada/GQA variants

Register-pressure, tile, split-scaling, prefetch and paired-GQA variants are all preserved
in the result set. None demonstrated a broad end-to-end win beyond the existing policy.
The remaining attention issue is the dependent page-table/K/V load chain, not simply too
few resident warps.

## 6. FlashAttention investigation

The installed FA2 paged-KV API requires a page size divisible by 256. The engine's normal
16-token pages cannot use it directly.

### Accepted: direct paged Flash prefill at page 256

Qwen3-0.6B long prompts:

- prefill step **-7.8%**;
- prefill penalty **-13.4%**;
- fused GPU **-8.8%**;
- ITL p99 **-22.6%**.

Qwen3-1.7B long prompts:

- expected gap **-5.3%**;
- TTFT p50 **-13.7%**;
- ITL p99/p999 **-9.9%/-9.0%**;
- prefill GPU **-5.3%**;
- fused GPU **-11.8%**.

The adapter uses physical paged pools directly, passes existing-context plus query length,
and groups unequal query lengths. An early implementation copied lengths CUDA-to-host once
per transformer layer; removing those 28 synchronizations was necessary for the win.

### Rejected: Flash decode

- Against graphed Triton, short-profile ITL regressed **113.9%** and expected gap **110%**.
- Against eager Triton, Flash still regressed ITL **11.9%**.
- Its isolated kernel was about 15% faster only at batch 1/context 4096.
- The installed kvcache entry point performs workspace setup/synchronization that is not
  CUDA-graph capture safe.
- Forced split counts generally lost to FA's automatic choice.

### Rejected: dense-gather Flash at page 16

Materializing dense K/V erased the attention win: prefill step **+38.5%**, TTFT **+52.6%**,
expected gap **+32.0%**. BF16 also failed the early-token gate for the published path.

## 7. FP16 tiled prefill and graph policy

Tiled page-16 prefill is the measured Ada fallback and current normal default. Across the
0.6B/1.7B chat and long profiles, it reduced prefill steps roughly 3-12.5% and improved
tails, including up to about 43% at ITL p99 in the strongest 0.6B long case. Padding-skip
logic removed K/V allocation and compute for inert graph tiles. SDPA graph shapes that could
exceed the physical pool were corrected.

Kernel-share profiling explains the remaining boundary:

| 1.7B prefill workload | GEMM | SwiGLU | RMSNorm | RoPE | attention/other |
|---|---:|---:|---:|---:|---:|
| whole prompt, 1024 | 91.61% | 2.89% | 1.90% | 1.13% | 2.47% |
| fused 128-token chunk + 3 decode rows | 87.32% | 0.85% | 1.32% | 0.70% | 9.81% |

Further elementwise fusion is not a leading lever; GEMM/weight representation dominates.

## 8. Packed fused rows, graph widths and prefill budget

Packed fused rows remove per-request chunk padding. They improve tail latency with negligible
host staging cost and are an opt-in building block. A second defect was then found: a packed
graph captured at the entire budget width still ran, for example, a 40-token plan through
256 rows. Width buckets now capture powers of two and replay the smallest fitting width.

At budget 128, bucketing is neutral because most fused steps already carry 128 tokens. At
chunk 128/budget 256 it reduces fused GPU **36.3%**, prefill step **35.8%**, expected gap
**25.0%**, and TTFT about 22%.

After correcting this confound:

- 128/128 remains the throughput/TTFT default;
- 64/64 is an ITL-SLO mode, cutting ITL p99 about 24% while making closed-loop TTFT about
  2.65-2.7x larger;
- a budget above chunk size does not improve throughput/TTFT at measured load and raises
  tail ITL when real 256-token steps occur;
- under overload, 64/64 costs only 6-9% TTFT because queueing dominates.

## 9. INT8 KV cache

### 9.1 Correctness repair

Ada preemption originally recomputed an already-generated prefix with different GEMM shapes.
Small FP16 reduction differences were amplified by per-token INT8 quantization and flipped a
later greedy token. Generated requests now spill the authoritative quantized K/V bytes and
FP16 scales; restore checks page counts. Requests preempted before generation still recompute.

A later mixed-step bug spilled a page reserved for the pending decode token even though it
was not committed. Restore allocated only committed pages. Spill now covers committed pages
only, and a regression test retains the failure case.

### 9.2 Key-quality investigation

Plain per-token K quantization is rejected. Natural-text prompts diverged within eight tokens
on 3/10 cases, including a token-0 flip with a 0.56 logit margin. Fake-quantized stock runs
localized the problem to K, not V. Post-normalization/RoPE key channels had outliers 24-81x
the median.

Static per-layer/per-KV-head/per-channel key smoothing, pair-equalized for RoPE, divides K
before quantization and multiplies Q by the same factor. Storage is unchanged. Four 2048-token
calibration samples take about 1.2 seconds.

| model/context | old top-1 / KL | smoothed top-1 / KL |
|---|---:|---:|
| 0.6B, 1k | 91.0% / 0.0414 | **97.8% / 0.0024** |
| 0.6B, 4k tail | 92.7% / 0.0354 | **97.7% / 0.0026** |
| 1.7B, 1k | 94.5% / 0.0207 | **98.3% / 0.0019** |

FP16 K plus INT8 V reached 99.5% top-1, confirming K as the sensitive component. Group-32
K scales alone reached only 94.3%; smoothing plus group-32 added little and complicated the
scale layout.

### 9.3 Tiled INT8 prefill

The original per-token INT8 prefill kernel cost 1.32 ms/layer at chunk 128/context 1024,
versus 0.17 ms for FP16 SDPA. A 12-configuration x 60-cell search produced a tensor-core
tiled kernel 7.6x faster geometrically and near FP16 tiled performance. Dispatch is
16x128/w4/s2 for chunks <=32 and 32x128/w4/s2 above.

### 9.4 Decode tuning

The INT8 decode search covered batch 1/4/8/16 and context 512-8192 while rotating at least
64 MB of pools to defeat L2 reuse. All candidates matched the reference within 2e-3.

- batch 1 is latency-bound: 256x4 improves isolated time 1.07-1.24x;
- batch 4 is already near the 235-240 GB/s roofline;
- batch 8 showed 1.11-1.17x isolated wins that did not transfer end to end;
- batch 16 paired GQA gained only 1.04-1.06x at the roofline.

The batch-1 rule reduced full-engine decode 2.8% and is accepted inside the opt-in INT8 path.
The batch-8 rule regressed decode 2.7% end to end and is disabled. Paired GQA remains an
unselected research implementation.

### 9.5 End-to-end INT8 results

Qwen3-0.6B, equal approximately 3.25 GB peak memory, long concurrency 8:

| metric | FP16 | optimized INT8 KV | change |
|---|---:|---:|---:|
| throughput | 166.54 tok/s | 190.16 tok/s | **+14.2%** |
| ITL p50 / p99 | 13.76 / 22.72 ms | 11.68 / 21.75 ms | **-15.1% / -4.3%** |
| TTFT p50 | 1383 ms | 998 ms | **-27.9%** |
| decode / prefill step | 9.57 / 16.81 ms | 8.68 / 15.50 ms | **-9.3% / -7.8%** |
| KV capacity | 17,680 tokens | 34,832 tokens | **1.97x** |
| preemptions | median 1 | 0 | eliminated |

The chat profile is capacity-neutral at this size: throughput +0.5%, ITL p50/p99
-16.5%/-13.1%, TTFT about -13%, decode -6.4%, prefill -7.1%.

Under deliberately constrained equal-byte pressure, INT8 removed 55 preemptions and changed
throughput 280 to 459 tok/s, TTFT 3144 to 661 ms and maximum ITL 1199 to 32 ms. In the
decode-heavy spill workload, equal-byte INT8 avoided all spills and reached 243.8 tok/s;
equal-block INT8 spilled twice but still cut the FP16-pressure maximum stall from 8.4 to
2.7 seconds.

Pinned spill was rejected: fresh pinned allocation raised writes from about 108 to 245 ms.
The correct operational conclusion is to size the INT8 pool to avoid spill, not to optimize
a path constrained by this machine's x1 link.

## 10. W8A16 and projection work

### 10.1 Rejected formats

- Dynamic W8A8 lost at every real Qwen3-1.7B projection shape.
- FP8 `_scaled_mm` returned `CUBLAS_STATUS_NOT_SUPPORTED` on this Ada/software stack.
- Early W4A16 projection prototypes did not provide a general 1.7B serving path.
- The first W8A16 timing method was rejected because weights were L2-hot.

### 10.2 MLP W8A16

Quantizing gate/up/down in every 1.7B layer replaces 84 projections and reduces unique
weight storage from about 3.20 to 2.22 GiB. The search covered BM 16-128, BN 32-256,
BK 64-256, 2/4/8 warps, 2-4 stages, grouping 1/8 and scale placement. Thirty-two
out-of-resource configurations are retained in the JSON.

Static dispatch after realistic fused-row remeasurement:

- M <= 16: 16x32x128, 4 warps, 3 stages;
- M <= 48: 32x128x128, 8 warps, 3 stages;
- larger: 32x64x64, 4 warps, 3 stages, group 8, scale after accumulation.

The first dispatch used a 64-row tile above M=96 and regressed the fused step 5.6%; realistic
M=129-136 and 260-264 shapes exposed last-tile waste and corrected the table.

For 1.7B chat/long workloads, W8 MLP reduces decode about 25-27%, expected gap 13-15%,
fused step 6.5-7%, and ITL p99 4-7%. Teacher-forced top-1 is 98.1-98.3%, KL
0.0018-0.0022, with unchanged perplexity. It is accepted as an explicit lossy path.

### 10.3 Vocabulary projection

After MLP quantization, the tied FP16 vocabulary projection is the largest remaining read:
151936 x 2048, about 622 MB and 2.4 ms/step. The W8 copy runs near 1.21 ms but the FP16
embedding must remain, adding about 311 MB to the W8-MLP model. End-to-end 1.7B results:

- batch 1 decode **-11.8%**, expected gap **-11.6%**, ITL p99 **-11.4%**;
- chat concurrency 6 decode **-9.3%**, expected gap **-6.9%**;
- teacher-forced quality is essentially unchanged from W8 MLP alone.

Accepted as opt-in.

### 10.4 Attention projections

The searched kernel changed the old conclusion for Qwen3-1.7B. Quantizing q/k/v/o gives:

- batch 1: decode/ITL about **-15.2%**, expected gap **-14.5%**, TTFT **-3.3%**;
- chat concurrency 6 with graph bucket 8: decode **-13.0%**, ITL p99 **-4.7%**;
- loaded expected gap is only -1.9%, unresolved inside 9.1% spread;
- prefill penalty rises 14% and TTFT is unresolved.

It passes the early-token gate but is lossy. The optimized 1.7B profile currently enables
it because the decode win is real; this setting should be described as experimental until a
lower-variance loaded A/B resolves total service benefit.

Qwen3-0.6B does not enable W8 attention because it produced an early non-tie token change.

### 10.5 Qwen3-0.6B MLP plus lm_head

The model retains tied FP16 embeddings, so storage changes only 1,192,100,352 to
1,084,146,944 bytes, about 9.1%. Teacher-forced top-1 is 97.9%/98.3% at 1k/4k with KL
0.00226/0.00166.

- batch 1: decode/ITL about **-29.5%**, expected gap **-26.6%**, delivered tokens +36.6%;
  TTFT/prefill regress about 7.5%;
- chat concurrency 8: decode **-20.4%**, expected gap **-6.1%**, delivered tokens +6.3%,
  ITL about -5.4% and TTFT about -5.5%.

This is useful as a latency profile, not primarily a storage optimization.

## 11. Qwen3-4B-AWQ / W4A16

The Transformers AWQ loader required an unavailable external runtime, so the checkpoint was
repacked one projection at a time into PyTorch's native `int4pack` layout without ever
materializing a 4B FP16 model. It loads in about 6.8 seconds, occupies 2.54 GiB before KV,
and generates correctly.

Native tinygemm is excellent for decode—3.64x faster than dequantized FP16 at batch 1—but
3.67x/4.45x slower at 128/256 prefill rows. Decode graphs improve ITL 15.5% and decode GPU
20%; native tinygemm prefill graphs regress TTFT 75% and are disabled.

The packed layout was derived and tested directly. The v1 scalar gather was rejected. The
v2 coalesced unpack kernel was accepted at >=64 rows, reducing layer projection time about
1.7-2.0x and end-to-end prefill/fused steps about 34-47%. It also avoids tinygemm's BF16
activation rounding.

The final v3 dispatch adds a third tier:

- tinygemm below 64 rows;
- fused Triton INT4 GEMM from 64 through 191 rows;
- exact Triton dequantization to reusable FP16 scratch followed by cuBLAS from 192 rows.

Whole-prompt TTFT versus tinygemm-only:

| prompt rows | tinygemm | fused INT4 | hybrid dequant+cuBLAS | hybrid speedup |
|---:|---:|---:|---:|---:|
| 128 | 147.5 ms | 71.8 ms | 71.8 ms | **2.05x** |
| 512 | 580.8 ms | 256.6 ms | 176.1 ms | **3.30x** |
| 1024 | 1169.8 ms | 507.4 ms | 322.6 ms | **3.63x** |
| 2048 | 2363.9 ms | 1038.3 ms | 620.2 ms | **3.81x** |
| 4096 | 4836.7 ms | 2168.5 ms | 1286.2 ms | **3.76x** |

AWQ prefill graphs are enabled only with all three proven conditions: Triton prefill,
packed rows and packed-width buckets. In that combination ITL p999 improves 34%, prefill
GPU about 1%, and TTFT is lower in every paired run. Plain or padded AWQ prefill graphs are
rejected.

## 12. Single-request results versus a naive decoder

The explicit baseline uses stock FP16 DynamicCache and one-token decode; it is not continuous
batching. With a 19-token prompt and 64 generated tokens:

| model | arm | ITL p50 | end-to-end p50 | throughput | peak allocated |
|---|---|---:|---:|---:|---:|
| 0.6B | naive FP16 | 23.69 ms | 1543 ms | 42.16 tok/s | 1.210 GB |
| 0.6B | paged FP16 | 11.77 ms | 835 ms | 80.77 tok/s | 1.270 GB |
| 1.7B | naive FP16 | 39.55 ms | 2575 ms | 24.87 tok/s | 3.460 GB |
| 1.7B | paged FP16 | 33.28 ms | 2174 ms | 29.45 tok/s | 3.520 GB |
| 1.7B | paged W8 MLP | 26.08 ms | 1685 ms | 38.03 tok/s | 2.464 GB |

The exact-output FP16 speedups are 2.01x ITL/1.85x end-to-end for 0.6B and 1.19x/1.18x
for 1.7B. Paged attention does not inherently save batch-1 memory in this short workload;
its storage advantage is bounded allocation, reuse, fragmentation control and multi-request
capacity rather than making the model weights disappear.

## 13. Idle whole-prompt policy

Chunking a lone 2k prompt into sixteen 128-token steps made engine TTFT slower than the
untouched model. `idle_prefill_tokens=4096` now sends one fresh prompt through one forward
when nothing is decoding. Loaded traffic retains normal interleaving.

- 0.6B long, one client: TTFT **-32%**, ITL neutral;
- 1.7B long, one client: TTFT **-15%**, ITL neutral;
- 1.7B chat, four closed-loop clients: neutral;
- Poisson chat: TTFT neutral, ITL p99 **-29%**.

Taking every fresh prompt whole, or batching fresh prompts whole with 25% padding, was
rejected because it delayed other prompts. Only the idle single-prompt rule is default.

## 14. Fixed overhead and speculative decoding

Host staging is about 0.06-0.08 ms at batch 1-4. Wall-minus-pure-graph replay is normally
below 2% and at most about 5%. GPU-resident metadata was therefore rejected as non-binding.

N-gram speculative decoding was explored because it yields more tokens per weight read with
no draft model or VRAM. At 1.7B batch 1:

- copy-heavy repository/code context: **1.81x aggregate**, 1.96 accepted tokens/step;
- free chat: **1.04x**, range 0.96-1.11x;
- two late differences occurred at near-tie stock margins;
- pure-Python lookup costs 284 microseconds and can erase the free-chat gain.

Integration is intentionally unfinished. A production version needs a per-request greedy
opt-in, incremental suffix index, k-token packed verify rows, device/host acceptance,
capacity reservation, correct multi-token commit/finish handling, verify graph buckets and
an adaptive cutoff when rolling acceptance is below about 0.1 token/step. It also needs a
batch 1/2/4/8 crossover A/B.

## 15. Accepted, rejected and unresolved ledger

### Defaults

- graphed Triton paged decode;
- tiled FP16 page-16 prefill;
- 128-token chunk and 128-token prefill budget;
- idle whole-prompt prefill up to 4096 tokens;
- all capability/geometry checks and allocator correctness fixes.

### Accepted opt-ins

- direct Flash prefill for long FP16 workloads with 256-token pages;
- packed fused rows and packed-width buckets;
- calibrated smoothed INT8 KV, tiled INT8 prefill and batch-1 tuned INT8 decode;
- searched W8A16 MLP and W8A16 lm_head;
- W8A16 attention for the experimental optimized 1.7B profile;
- Qwen3-4B-AWQ three-tier prefill dispatch;
- AWQ prefill graphs only with Triton + packed rows + width buckets;
- 64/64 ITL-SLO scheduling mode.

### Rejected

- Flash decode and forced Flash split counts;
- dense-gather Flash prefill on 16-token pages;
- BF16 published Flash path;
- plain per-token INT8 keys;
- pinned per-spill allocations;
- INT8 batch-8 decode rule;
- paired GQA as a serving selection;
- split-K as a general default;
- register reduction or occupancy maximization without latency improvement;
- early V prefetch schedule;
- W8A8 projections;
- FP8 on the installed runtime;
- direct-from-AWQ Triton GEMV;
- INT4 v1 scalar-gather prefill;
- tinygemm/padded AWQ prefill graphs;
- larger prefill budgets as a general throughput policy;
- GPU-resident metadata work;
- elementwise/SwiGLU/RoPE tuning as a leading priority.

### Unresolved or future research

- lower-variance loaded A/B for W8A16 attention projections;
- paired-GQA INT8 end-to-end value;
- a structurally different paged-load/pipeline design for long-scoreboard stalls;
- n-gram speculative integration and batch crossover;
- validation of Mistral/Llama families;
- normal-x8 PCIe spill behaviour;
- multi-GPU, distributed scheduling and production fleet claims.

## 16. Bugs found and fixed during R&D

- split-K graphs silently captured the wrong attention function/context bound;
- max-active 1 graph capture reserved no dummy page;
- tiled FP16 and INT8 prefill allocated/worked on padded K/V tiles;
- SDPA context buckets could exceed the physical pool;
- generated INT8 recompute was numerically non-authoritative;
- mixed decode+prefill preemption spilled an uncommitted pending page;
- graph-width padding made larger budgets look inherently slow;
- per-layer Flash query-length synchronization erased the kernel win;
- sampled prefix hits could replay another request's sampled token or skip an RNG draw;
- test imports pinned duplicate CUDA models and caused misleading OOM failures;
- module-scoped model tests retained a selected paged-attention config between cases;
- profile tests confused intentional exact-prefix retention with a leaked request block.

## 17. Final profile definitions

`engine.profiles.build_engine(model_name, profile=...)` provides one measured switch.

`lossless`:

- FP16 weights and KV;
- graph-replayed decode;
- tiled page-16 prefill;
- packed fused rows;
- idle whole-prompt fast path.

`optimized` for 0.6B:

- searched W8A16 MLP plus W8A16 lm_head;
- smoothed INT8 KV;
- tiled INT8 prefill;
- batch-1 tuned INT8 decode;
- no W8 attention projections.

`optimized` for 1.7B adds W8 q/k/v/o attention projections. Its full loaded advantage is
less certain than its decode-step win, so deployment documentation must preserve that caveat.

AWQ models retain FP16 KV because INT8 KV has not been validated on the 4B checkpoint.

## 18. Verification snapshot

On 2026-10-17 before publication cleanup:

- every Python file compiled;
- `git diff --check` passed;
- non-CUDA suite: 316 passed;
- initial CUDA suite: 261 passed, 1 expected memory skip, 4 failures.

The four CUDA failures were audited rather than hidden:

1. a shared module-scoped model retained the preceding engine attention selector;
2. the lossless profile test counted an intentional exact-prefix cache owner as a leak;
3. the optimized profile checked the FP16 backend field instead of the separate INT8
   prefill selector;
4. the W8 attention installer and its stated idempotence contract disagreed.

The publication branch resets stock attention in the independent reference test, accounts
for prefix-cache owners, checks the correct profile field and makes repeat W8 installation
return an empty mapping while retaining the error for models with no attention projections.
After those repairs the complete gates passed: **316 non-CUDA tests**, **265 CUDA tests**,
and **1 expected skip** for the two-model speculative test that exceeds the practical 8 GB
budget. Compilation and `git diff --check` also passed.

## 19. Reproduction

Run tests:

```bash
.venv/bin/python -m compileall -q engine benchmarks tests
.venv/bin/pytest -q -m 'not cuda'
.venv/bin/pytest -q -m cuda
git diff --check
```

Create a measured service directly:

```python
from engine.profiles import build_engine

engine = build_engine("Qwen/Qwen3-1.7B", profile="optimized", max_active=8)
```

Or run the application factory:

```bash
.venv/bin/uvicorn 'engine.server.api:create_rtx4060_optimized_app' \
  --factory --host 127.0.0.1 --port 8000
```

The older `create_rtx4060_flash_app` remains the reproducible page-256 FP16 Flash-prefill
configuration rather than the general page-16 profile.

## 20. Evidence map

| subject | implementation / harness | primary evidence |
|---|---|---|
| roofline | `benchmarks/kernels/roofline.py` | `roofline.json` |
| occupancy/memory | `profile_paged_decode_memory.py`, Nsight | `paged_decode_memory_profile.md`, `paged_decode_occupancy_ab.json` |
| split-K and Ada decode | `paged_decode_split_k.py`, `paged_decode_ada.py` | `ab_decode_split8_*`, `paged_decode_*_ab.json` |
| Flash decode/prefill | `flash_paged.py`, reliability A/B | `flash_paged_*_sweep.json`, `ab_flash_*.json` |
| tiled FP16 prefill | `tiled_paged_prefill.py` | `ab_prefill_tiled_*.json`, `fp16_prefill_tile_search.json` |
| INT8 KV | `int8_paged_kv.py`, `kv_smoothing.py` | `int8_kv_*`, `int8_prefill_search.json` |
| tuned INT8 decode | `int8_decode_tuned.py` | `int8_decode_search.json`, `int8_decode_tuned_ab_*.json` |
| W8A16 | `w8a16_linear.py`, `w8a16_tuned.py` | `w8a16_*`, `ab_w8a16_*.json` |
| AWQ/INT4 | `awq_tinygemm.py`, `int4pack_gemm.py` | `awq_*`, `int4pack_prefill_search_*.json` |
| packing/budgets | `continuous_batching.py` | `ab_packed_*`, `ab_prefill_budget_*.json`, `open_loop_budget_*.json` |
| complete profiles | `profiles.py`, stacked A/B | `stacked_*`, `optimized_vs_fp16_*.json` |
| single request | `single_request_naive_ab.py` | `single_request_naive_ab_*.json` |
| speculative probe | `ngram_*` benchmarks | `spec_ngram_*.json` |
| kernel shares | `decode_kernel_share.py` | `decode_kernel_share_*.json`, `prefill_kernel_share_1.7b.json` |
| serving smoke | server factory / external client | `live_smoke_optimized_flash.json` |

Every raw result includes its run configuration, environment, arm settings and summary where
the corresponding harness supports them. Stock-reference sidecars are retained for lossy
and pressure A/Bs. The repository intentionally does not include downloaded model weights;
model identifiers and the local AWQ conversion path provide reproduction metadata.

## 21. Publication boundary

Supported claim: an experimentally validated, single-RTX-4060 Qwen3 inference engine with
continuous batching, paged KV, graph replay, a lossless FP16 mode, measured lossy memory and
latency profiles, and a practical 4B-AWQ path.

Unsupported claims: universal Flash superiority, bit-identical quantized inference,
performance validation for every transformer family, production-grade distributed serving,
or multi-GPU scale. Negative results in this report are part of the deliverable and should
not be deleted when future implementations supersede them.

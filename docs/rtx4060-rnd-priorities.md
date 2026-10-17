# RTX 4060 memory and latency R&D queue

This queue follows the measured FP16 serving baseline in
`rtx4060-final-evaluation.md`. A T4 rejection is a hypothesis to retest on Ada,
not evidence of an Ada win. Each candidate needs token parity and an interleaved
end-to-end A/B before it changes a serving default.

| Priority | Investigation | Main objective | First decision gate |
|---|---|---|---|
| 0 | Profile real graph-replayed decode and prefill | Locate time in weights, attention, metadata, launches, and host synchronization | Nsight Systems phase trace; Nsight Compute DRAM/L2, sectors, stalls, occupancy when counters are enabled |
| 1 | Ada decode attention layout and offline tile search | Reduce attention latency across batch/context regimes | Profile page-table and K/V loads; search tile/warp/GQA/split-K; verify graph buckets and end-to-end gain |
| 2 | Weight-only W4A16/AWQ on 1.7B and 4B | Cut weight bytes, decode ITL, and 8 GB VRAM use | Measure model load size, parity, batch 1/4/8/16 ITL and throughput; include prefill cost |
| 3 | W8A16, W8A8 and FP8 projection paths | Test the lower-precision speed/capacity frontier | Include quantize/dequantize overhead, all projections, 1.7B model parity and full-step A/B |
| 4 | INT8 KV repair and long-context evaluation | Increase KV capacity and possibly reduce long-context traffic | Fix Ada recompute/preemption parity; compare FP16/INT8 at contexts beyond L2 residency |
| 5 | Ada paged prefill kernel and page geometry | Reduce long-prompt TTFT without 256-token page overhead | Compare tiled paged, SDPA, direct Flash, fragmentation and prefix sharing under one workload |
| 6 | Graph bucket and prefill-budget policy | Improve throughput while controlling ITL tails and TTFT | Mixed arrivals with short/long prompts, queue depth and p95/p99 latency |
| 7 | GPU-resident metadata and low-batch speculative decode | Reduce fixed step overhead and batch 1–4 latency | Nsight-confirmed host/launch cost; full scheduler correctness and A/B |

## First experiment

`benchmarks/batching/trace_live_decode.py` captures only warmed decode graph replays,
with NVTX ranges per step. Run the batch-8/context-1024 baseline and inspect
`nsys stats` plus the timeline. Repeat at batch 1/context 4096 and batch 16/context
256 to separate weight bandwidth, context scaling, and launch overhead. Record the
model, GPU, graph key, page size and kernel names with each trace.

The first run (RTX 4060, Qwen3-0.6B FP16, batch 8, 1024-token prompt, four graph
replays) found 112 paged-attention launches taking 24.05 ms total, 788 CUTLASS GEMM
launches taking 27.27 ms, and 4 argmax launches taking 0.06 ms. These are about 46%,
52%, and 0.1% of summed kernel time in this trace. Four NVTX decode-step ranges took
11.3–17.1 ms each under profiling. The graph key was `(8, 128, 4)` and no graph was
captured in the trace. Nsight Systems node tracing adds overhead, so these numbers
locate work; they are not an unprofiled ITL benchmark.

Nsight Compute initially reported `ERR_NVGPUCTRPERM`: the driver has
`RmProfilingAdminOnly: 1`. Desktop administrator authentication allowed direct
counter collection without changing driver configuration. Results and caveats are
recorded in `results/rtx4060/paged_decode_memory_profile.md`. The batch-8/context-1024
kernel achieved 29.44% occupancy against a register-limited 33.33% theoretical
ceiling, with long-scoreboard stalls dominant. At batch 8/context 256, paired
GQA halved L2 read sectors and won the isolated timing by 1.14x despite lower
occupancy. This puts an Ada memory/occupancy tradeoff first in the queue.

Reproduce the counter profile with:

```bash
pkexec --keep-cwd /usr/local/cuda-13.1/bin/ncu \
  --section MemoryWorkloadAnalysis --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --cache-control none --nvtx --nvtx-include 'profile_target]' \
  --launch-count 1 -o /tmp/paged_decode_b8_ctx1024 \
  .venv/bin/python -m benchmarks.kernels.profile_paged_decode_memory \
  --batch 8 --context 1024
```

Inspect the report with `/usr/local/cuda-13.1/bin/ncu --import /tmp/paged_decode_b8_ctx1024.ncu-rep`.
The target's KV pool is about 32 MB at this point, slightly larger than the card's
L2. Repeat with batch 8/context 256 (about 8 MB) and batch 1/context 4096
(about 16 MB) to test residency and grid-size effects. Inspect DRAM bytes, L1/L2
hit rates, memory sectors per request, achieved occupancy and dominant warp stalls.
Kernel replay may alter cache behavior even with flushing disabled, so compare these
counters with the full-step Systems trace before changing the engine policy.

Repeat the trace at batch 1/context 4096 and batch 16/context 256 before selecting an
attention change. Exact-hit prefix caching, fused QKV/gate-up and Flash decode remain
workload-specific experiments until their full-step evidence wins.

## Split-K follow-up: batch 1, long context

The interleaved isolated-kernel sweep in
`results/rtx4060/paged_decode_occupancy_ab.json` found that at batch 1/context
4096, eight-way split-K (`64x2`) reduced attention-layer latency from 0.0932 to
0.0647 ms (1.44x). It was slower at batch 1/context 1024 and batch 8/context
1024; this is a low-batch, long-context candidate, not a general replacement.

The first full-engine A/B was invalid for this candidate: the split-K setting
captured a per-head graph because the graph context omitted `decode_num_splits`
and used a fixed 128-token bound. The capture path now passes the split count
and buckets the maximum context length in the graph key. A new Nsight Systems
trace confirmed 112 `_split_decode_kernel` and 112 `_merge_splits_kernel`
launches over four graph-replayed decode steps, with no per-head attention
launches.

The corrected three-repeat A/B (`results/rtx4060/ab_decode_split8_long4k_b1_fp16.json`)
used Qwen3-0.6B FP16, RTX 4060, CUDA graphs, batch 1 and mean context 4126.
Median decode GPU time was 11.519 ms per-head versus 11.126 ms split-K (-3.4%);
median inter-token latency was 12.619 versus 12.222 ms (-3.1%). Each arm
matched the stock-model token gate and both arms emitted identical greedy
tokens. TTFT was unresolved within run-to-run spread. The split-K arm recorded
one lazy graph capture per run; the p50 decode result is nevertheless stable,
but this capture cost must be eliminated or charged in a deployment decision.
Keep per-head as the default until the long-context policy is tested across
more batch/context shapes and capture overhead is controlled.

Follow-up Nsight Compute counters for split-K and the merge are in
`results/rtx4060/paged_decode_memory_profile.md`. Eight-way `64x2` split-K at
batch 1/context 4096 raised achieved occupancy from 8.33% to 21.81%, but
still had no eligible warp on 76.58% of scheduler cycles. A `32x2` tile cut
registers/thread from 153 to 96 and reduced that stall fraction, yet did not
improve isolated latency. Sixteen splits won the isolated 4k cell narrowly but
did not improve full-model ITL or decode GPU time over eight splits. The next
kernel experiment should therefore target the dependent page-table/K/V load
chain (for example software pipelining or prefetch), with the same isolated
and full-model gates; adding occupancy alone has not been sufficient.

## Priority 1 close-out and priority 2 W4A16

Moving the V load ahead of QK/softmax in split-K did not improve the selected
batch-1/context-4096 cell: 0.0655 ms versus 0.0653 ms for the established
schedule. It tied or lost at the other tested cells in
`paged_decode_prefetch_v_ab.json`; keep the original schedule. Together with
the rejected 32x2 tile and 16-way split count, this closes the first round of
decode-attention scheduling experiments. The remaining long-scoreboard issue
needs a structurally different load/paging design, not another default change.

The local Qwen3-4B-AWQ checkpoint uses 4-bit, group-128 GEMM-format weights.
The environment's Transformers AWQ loader requires an unavailable `gptqmodel`
runtime, so the checkpoint was repacked losslessly into PyTorch's native INT4
layout one projection at a time. This avoids ever materializing the 4B FP16
model. The resulting model loads in 6.8 seconds and occupies 2.54 GiB before
KV-cache allocation; a real forward generated `Paris` for `The capital of
France is`.

Across all seven layer projections using the real packed checkpoint, native
INT4 was 3.64x faster than dequantized FP16 at batch 1, 2.30x at batch 8 and
1.45x at batch 16. It was 3.67x slower at 128 rows and 4.45x slower at 256
rows, so W4A16 is a decode/capacity win with a serious prefill tradeoff. The
experimental direct-from-AWQ Triton GEMV was rejected: even after split-K it
was only 1.02x faster across a batch-1 layer and roughly 3x slower at batch 4.
The native repack path is retained for the full-engine gate.

The first full-engine Qwen3-4B-AWQ short-prompt run used 3.12 GiB including a
256-block KV pool and successfully captured a decode graph. A corrected isolated
graph A/B kept prefill eager in both arms: decode graphs reduced median ITL from
23.51 to 19.87 ms (-15.5%) and decode GPU time from 23.19 to 18.55 ms (-20.0%),
while TTFT was unchanged within run-to-run spread. A separate prefill-graph A/B
regressed median TTFT from 127 to 223 ms (+75%). The engine now automatically
disables prefill graphs for the native tinygemm INT4 backend while retaining
decode graphs.

## Priority 3: selective W8A16

Real Qwen3-1.7B layer-0 projection measurements rejected W8A8 at every tested
shape and batch. FP8 dynamic GEMM was also rejected on this RTX 4060 software
stack: every `torch._scaled_mm` attempt returned `CUBLAS_STATUS_NOT_SUPPORTED`.
W8A16 lost on Q/K/V projections but won strongly on the wide gate/up and down
MLP matrices, so the end-to-end candidate quantizes only those three projections
in each layer.

The selective implementation replaced 84 projections and reduced unique model
storage from 3.20 to 2.22 GiB. Across two interleaved full-engine repeats with
decode graphs and eager prefill, median results were:

| Active batch | FP16 ITL | W8A16 ITL | ITL change | Decode GPU change | Prefill GPU change |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.83 ms | 16.34 ms | -25.1% | -27.3% | -4.0% |
| 4 | 23.76 ms | 17.94 ms | -24.5% | -28.9% | +3.3% |
| 8 | 24.16 ms | 18.38 ms | -23.9% | -28.7% | +1.0% |
| 16 | 25.70 ms | 19.84 ms | -22.8% | -27.9% | +3.8% |

The four identity prompts passed the eight-token correctness gate; two first
diverged later at positions 13 and 20, as expected from INT8 reduction drift.
Selective MLP W8A16 is therefore retained as an opt-in 1.7B decode and capacity
path. Attention W8A16, W8A8, and FP8 are rejected for this device/runtime.

### Single-request comparison against untouched Transformers

An identical 19-token prompt / 64-token generation benchmark compared stock FP16
`ExplicitDecodeRunner` with DynamicCache against the one-active-request paged engine.
Each arm used two warmups and five measured runs.

| Model | Arm | ITL p50 | End-to-end p50 | Throughput | Peak allocated |
|---|---|---:|---:|---:|---:|
| 0.6B | naive FP16 | 23.69 ms | 1543.2 ms | 42.16 tok/s | 1.210 GB |
| 0.6B | engine FP16 | 11.77 ms | 834.9 ms | 80.77 tok/s | 1.270 GB |
| 0.6B | engine W8A16 MLP | 12.05 ms | 807.3 ms | 79.22 tok/s | 1.007 GB |
| 1.7B | naive FP16 | 39.55 ms | 2575.1 ms | 24.87 tok/s | 3.460 GB |
| 1.7B | engine FP16 | 33.28 ms | 2174.3 ms | 29.45 tok/s | 3.520 GB |
| 1.7B | engine W8A16 MLP | 26.08 ms | 1684.8 ms | 38.03 tok/s | 2.464 GB |

The correctness result changes the W8 decision. FP16 engine output was identical to
naive for both models. W8 first diverged at token 13 on 0.6B and token 1 on 1.7B.
The 1.7B stock top-two margin at that position was 0.265625, far above the 0.02
near-tie threshold, so this is a real quantization-quality failure rather than greedy
tie noise. Selective per-channel MLP W8A16 is therefore **rejected as a serving path**
despite its 1.52x ITL speedup and 28.8% peak-memory reduction against naive on this
workload. The accepted exact-output single-request gains are FP16: 2.01x ITL / 1.85x
end-to-end for 0.6B, and 1.19x ITL / 1.18x end-to-end for 1.7B.

## Priority 4: INT8 KV correctness and long context

The Ada preemption failure was caused by recomputing an already-generated prefix
with different GEMM batching/reduction shapes. Small hidden-state differences were
then amplified by per-token INT8 cache quantization and flipped a later tied greedy
token. Requests that have begun generation now spill their authoritative INT8 K/V
pages and FP16 scales to host on the rare preemption path and restore those exact
bytes on readmission. The formerly expected-failure pressure test now passes, as do
the neighboring mixed-length, CUDA-graph, and cancellation pressure tests. Requests
preempted before generation still use ordinary recompute.

The pressure test spilled two requests (3.73 MB total): measured D2H spill time was
11.8 ms and restore time was 13.8 ms total. These counters are now exposed alongside
live spill requests/bytes in the engine snapshot, so a mixed-arrival run can attribute
the tail directly instead of hiding it inside a generic recompute interval.

At a 4096-token context, the fixed-work Qwen3-0.6B decode A/B retained 100% greedy
token agreement. Batch 1 improved from 933.2 to 897.1 ms for 64 decode steps
(1.04x); batch 4 improved from 702.3 to 553.8 ms for 32 steps (1.27x). For the
batch-4 run's 1152-block pool, FP16 KV occupied 1.969 GiB versus 1.000 GiB for
INT8 including scale storage, a 1.97x capacity gain. INT8 KV remains explicit
opt-in until its host-spill tail latency and mixed-arrival behavior are measured;
the long-context bandwidth and memory gates are now positive.

## Priority 4 deployment gate: mixed arrivals, quality and spill tails

`benchmarks/batching/int8_kv_pressure_ab.py` drives seeded Poisson arrivals with natural-text
prompts (committed docs/engine sources) through interleaved arms whose pools are sized
against peak token demand: `*_roomy` (no preemption), `fp16_pressure` (pool = 0.5-0.6x
peak), INT8 at the same block count (`equal_blocks`) and at the same KV bytes
(`equal_bytes`, 1.97x blocks). Qwen3-0.6B FP16 weights, page 16, graphs (1/2/4/8 buckets),
per-head decode, prefill chunk/budget 128, prefix cache off, three repeats. Every arm drains
to zero used pages (excluding graph dummy pages), with no spill payload left and every request
finished at its length bound.

**Quality: per-token INT8 keys fail; static key smoothing repairs it.** The earlier 100%
greedy agreement used a repetitive prompt. On natural text the plain INT8 cache diverged
from stock within eight tokens on 3/10 prompts, including at token 0 with a 0.56 logit margin,
and with no preemption. A fake-quantized stock model reproduces this: per-token K is the cause
(first-token logit error 1.1-3.0), per-token V is benign (<= 0.15). Qwen3's post-`k_norm`/RoPE
key channels reach 24-81x the median. The fix divides keys by a static per-(layer, kv head,
channel) factor, pair-equalized for RoPE, before per-token quantization, and multiplies the
query by the same factor (`engine/quantization/kv_smoothing.py`, `kv_key_smoothing=`). Storage
is unchanged. Calibration is 4x2048 held-out tokens (tests/benchmarks/scripts) and takes 1.2 s.
Teacher-forced quality versus FP16 KV (`int8_kv_teacher_forced_quality_*.json`):

| model / context | scheme | top-1 agreement | mean KL | PPL change |
|---|---|---:|---:|---:|
| 0.6B / 1024 | per-token K (old) | 91.0% | 0.0414 | +2.9% |
| 0.6B / 1024 | smoothed K | **97.8%** | **0.0024** | +0.16% |
| 0.6B / 4096 (last 512 scored) | per-token K / smoothed K | 92.7% / **97.7%** | 0.0354 / **0.0026** | +3.7% / +0.12% |
| 1.7B / 1024 | per-token K / smoothed K | 94.5% / **98.3%** | 0.0207 / **0.0019** | +0.9% / +0.6% |
| 0.6B / 1024 | FP16 K + INT8 V (reference) | 99.5% | 0.0001 | +0.03% |

Group-32 key scales alone (no smoothing) reached only 94.3% top-1. Smoothing plus
group-32 adds about 0.1-0.7 points but changes the scale layout, so it is not pursued. With
smoothing, greedy runs still flip near-tie positions (stock margin 0.03-0.33) at the rate a
2.2% per-token flip probability predicts. INT8 KV is therefore a measured lossy mode and
cannot pass a bit-identity gate. Plain per-token INT8 KV is rejected.

**INT8 prefill attention was the hidden regression.** With no memory pressure, INT8 KV lost
to FP16: ITL p99 44.3 versus 26.1 ms, TTFT p50 +32%. Isolated timing showed the per-token
INT8 prefill kernel at 8x the cost of FP16 SDPA (1.32 versus 0.17 ms per layer at chunk
128/context 1024). A tensor-core tiled INT8 prefill (`tiled_paged_prefill_int8`,
`int8_prefill_attention="tiled"`) was searched offline over 12 tile configurations x 60
cells (`int8_prefill_search.json`; no compile or numeric failures). It is 7.6x faster
than the per-token kernel (geomean) and at parity with FP16 tiled. Dispatch is static:
16x128/w4/s2 for chunks up to 32 and 32x128/w4/s2 above. Chat profile, 32 requests at
4 req/s, 3 repeats (`int8_kv_pressure_chat_0.6b_tiled.json`):

| metric (vs fp16_roomy) | int8 smoothed + tiled, roomy | int8 smoothed + tiled, equal bytes to fp16_pressure |
|---|---:|---:|
| throughput | **+14.8%** (spread 1.8%) | **+14.7%** |
| ITL p50 / p95 / p99 | -12.0% / -20.5% / -18.4% (p99 unresolved, 31% spread) | -11.9% / -19.9% / **-18.2%** |
| TTFT p50 / p99 | **-35.8% / -37.9%** | **-34.6% / -38.1%** |
| prefill step p50 / decode step p50 | -20.0% / -10.3% | -20.6% / -10.2% |
| KV pool bytes | 0.51x | 0.49x |

Against `fp16_pressure` (same KV bytes), equal-bytes INT8 removed all 55 preemptions:
throughput 280 to 459 tok/s, TTFT p50 3144 to 661 ms, ITL max 1199 to 32 ms.

**Long context beyond L2.** In the long profile (3-4k prompts, max_active 4; FP16 decode at
2+ rows reads more than 24 MB per layer), equal-bytes INT8 versus fp16_pressure gave throughput
69.7 to 116.4 tok/s and TTFT p50 3846 to 1332 ms. Versus fp16_roomy at twice the bytes:
+21.6% throughput and TTFT p50 -37%. All 7-8 preemptions there hit requests still prefilling,
which recompute, so no spill occurred. A decode-heavy profile (`long_decode`: 1.5-3k prompts,
384-768 new tokens, burst arrivals, pressure 0.6) forces spills of generating requests
(`int8_kv_pressure_long_decode_0.6b.json`):

| arm | throughput | ITL p99 / max | TTFT p50 | stall max | preemptions / spills |
|---|---:|---:|---:|---:|---:|
| fp16_roomy | 208.6 | 32.9 / 47 ms | 10.0 s | 0 | 0 / 0 |
| fp16_pressure (recompute) | 153.8 | 32.7 / **8447** ms | 9.4 s | **8.4 s** | 8 / 0 |
| int8 equal blocks (spill) | 172.6 | 23.9 / **2855** ms | 9.7 s | **2.7 s** | 8 / 2 (278 MB) |
| int8 equal bytes | **243.8** | 24.9 / 50 ms | **8.1 s** | 0 | 0 / 0 |

**Spill tail.** Per event, the write took 108-115 ms and the restore 147-163 ms for ~135 MB.
The single steps carrying them took up to 128-161 ms and 172-225 ms; every co-running request
pays that step. This host's GPU link is trained at **PCIe Gen4 x1** (`nvidia-smi`: width 1 of
max 8). Pinned D2H reaches only 1.39 GB/s and H2D 1.07 GB/s, so the spill is link-bound.
A pinned, per-layer async spill was A/B-rejected: transfer was only ~10% faster, and a fresh
pinned allocation per spill raised write time from 108 to 245 ms/event
(`int8_kv_spill_pinned_ab_long_decode.json`). The mixed-arrival run also exposed a latent
spill/restore defect. A decode row that had already taken the page for its next token, and was
then preempted by a later prefill in the same step, spilled one page more than restore
allocated. Spills now cover only committed pages, restore checks the page count, and a
regression test reproduces the original failure.

**Decision.** INT8 KV with calibrated key smoothing and tiled INT8 prefill is the
preferred opt-in for memory-bound and long-context deployments on this card. At equal bytes
it beats FP16 on every measured latency and throughput metric, and its preemption tail is
3x smaller than FP16 recompute. Host spill is acceptable only as a rare fallback. On this
x1-link host a ~135 MB spill/restore costs ~250 ms of blocking work, so deployments should
capacity-plan INT8 to avoid preemption rather than rely on it. Defaults stay FP16 because INT8
KV is lossy (top-1 ~98%). The engine defaults also stay per-token INT8 prefill and no
smoothing until a serving profile opts in; the A/B evidence favours enabling both whenever
INT8 KV is selected.

## Priority 5: Ada paged prefill at 16-token pages

**Kernel search.** `benchmarks/kernels/int8_prefill_attention_ab.py --family fp16` timed
12 tile configurations of the tensor-core tiled kernel against SDPA as the engine runs it
(mask and page index cached across layers), over rows 1/2/4, chunks 16-128 and contexts
0-4096 (`fp16_prefill_tile_search.json`, no failures). The default tiled launch is 2.41x
faster than cached SDPA (geomean, minimum 1.41x). A searched rule (16x64/w4 for chunks up
to 32, 32x32/w4 above) is only 6% faster than the default at kernel level, so no engine
change was made for it.

**Padded tiles.** A query tile containing only padding rows still streamed the whole KV
prefix; `tile_last` was clamped to the chunk length rather than to zero. A graph-padded
128-wide chunk with 16 real tokens at context 2048 cost 247 µs, more than a full chunk
(193 µs). Padding-only tiles now visit no keys: 93 µs FP16 and 127 µs INT8. Before this
fix the long-profile A/B showed tiled ITL p999 +16.5%; afterwards it tied (50.7 versus
50.7 ms).

**Two engine defects confounded the first SDPA A/Bs.**

1. SDPA context buckets were powers of two that could exceed the pool. With 1536 blocks
   (24,576 tokens), warmup tried a 32,768-token gather graph. That capture failed, and one
   failed capture disables graphs for the kind, so every SDPA prefill and fused step ran
   eagerly. Buckets are now clamped to pool capacity.
2. SDPA graphs hold gather buffers per context bucket. On 8 GB, fused captures run out of
   memory: Qwen3-1.7B at 768-1024 blocks and Qwen3-0.6B at 1536 blocks. The engine then
   silently runs fused steps eagerly. Tiled needs 24 fused graphs with no gather buffers
   instead of 144-168.

The soak warmup now prints any failed capture. The initial 1.7B "tiled ITL p99 +20.5%"
compared graphed tiled against eager SDPA, so it is void. With SDPA fully graphed it is
actually a win; see the fair results below.

**End-to-end, every graph captured in both arms** (`ab_prefill_tiled_*_fullgraphs.json`;
Qwen3 FP16, page 16, per-head decode, three interleaved repeats, stock token gate passed,
zero lazy captures):

| model / profile | expected gap | ITL p50 | ITL p99 | ITL p999 | TTFT p50 | prefill step p50 | fused GPU p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B short (1024 blocks, c8) | **-2.4%** | -2.2% (unres.) | -3.9% (unres.) | unres. | **-7.7%** | **-7.2%** | **-7.4%** |
| 0.6B chat (1024 blocks, c8) | **-7.2%** | **-7.6%** | **-15.8%** | **-25.8%** | -12.7% (unres.) | **-6.3%** | **-11.5%** |
| 0.6B long (1024 blocks, c6) | **-11.9%** | **-9.8%** | **-42.9%** | **-39.0%** | -27.3% (unres.) | **-12.5%** | **-17.6%** |
| 1.7B chat (512 blocks, c6) | -1.8% (unres.) | +1.7% (unres.) | **-8.9%** | +30.4%* | -1.1% (unres.) | **-3.1%** | **-3.7%** |
| 1.7B long (512 blocks, c3) | **-4.9%** | +0.7% (unres.) | -23.7% (unres.) | **-28.2%** | -14.8% (unres.) | **-5.2%** | **-9.1%** |

\* Instrumenting the same soak traced the 99 ms tiled p999 to a single fused step carrying
three prefill rows. That step is replayed at the 4-row graph bucket (4x128 tokens), and the
SDPA run happened not to produce one. Per step class, tiled is cheaper: 2-row fused 63-68
versus 68-73 ms, 1-row fused 40.9 versus 42.5 ms, and the Poisson-arrival probe maximum is
66-68 versus 70-75 ms. The tail is the fused-graph padding policy (k rows x full chunk
width for a 128-token budget), which applies to both kernels; it is a Priority-6 target.
The page-16 tiled hook gate passed (`check_hooks_tiled_prefill_block16_fp16.json`).

**Decision.** The RTX 4060 policy now records tiled as the measured *fallback* when the
measured Flash prefill cannot run, i.e. on 16-token pages
(`engine/backends/policy.py`, `prefill_fallback`). This is the one serving-default change
in this pass. It rests on five interleaved end-to-end A/Bs across two models and three
prompt profiles, the token gate and the hook gate. SDPA remains selectable explicitly,
and 256-page Flash prefill is unchanged. INT8 pools are unaffected: they cannot select
FP16 tiled, and they use `int8_prefill_attention`. Page-size changes were not tested here
because only Flash requires 256-token pages and its comparison is already recorded.

## Priority 6: graph padding and prefill budget

**Measurement note.** A `firmware-updater` snap process holding ~7.4 GB RAM and ~91% of a
CPU exited around 21:00 on 2026-10-15. Engine GPU timers close after the sampling sync and
therefore absorb host wake-up latency. Qwen3-1.7B decode "GPU" p50 fell from ~25 ms to
~17 ms at identical clocks (2715 MHz, no throttle reasons). Every conclusion here comes from
interleaved arms inside one run; absolute numbers from before and after that time are not
comparable.

**Packed fused prefill** (`packed_fused_prefill=True`, opt-in). Fused graphs used to pad
every chunk row to the full chunk width. A 128-token budget split over k rows ran k x 128
tokens through every projection and MLP. On Qwen3-1.7B a 2-row fused step cost 63-68 ms
against 41 ms for one row, and a 3-row step cost 95 ms. These steps are the ITL tail. The
packed layout now carries exactly `budget` chunk tokens after the decode tokens. Attention
alone gathers them into the padded `[rows, chunk]` view and scatters its output back (two
small `index_select`s per layer). Graph shapes no longer depend on how the budget splits.
Interleaved A/Bs: tiled prefill (now the page-16 default), graphs, three repeats, stock
token gate passed in every run, zero lazy captures.

| model / profile | ITL p99 | ITL p999 | expected gap | prefill step p50 | TTFT p50 |
|---|---:|---:|---:|---:|---:|
| 1.7B chat (512 blocks, c6) | **-33.4%** | **-55.1%** | +1.0% (unres.) | +1.4% | -12.0% (unres.) |
| 0.6B chat (1024 blocks, c8) | **-27.9%** | **-46.9%** | +1.2% (unres.) | +0.9% (unres.) | -4.6% (unres.) |
| 0.6B long (1024 blocks, c6) | **-18.2%** | **-37.8%** | +0.8% (unres.) | +0.9% (unres.) | -2.5% (unres.) |
| 1.7B long (512 blocks, c3) | -3.1% (unres.) | **-32.6%** | +0.2% (unres.) | +0.2% (unres.) | -2.5% (unres.) |

The 1.7B chat ITL p50 change (+36.8%) is not a regression signal. With 58% of steps
carrying prefill, the median gap sits between the decode mode (~17 ms) and the fused mode
(~41 ms) and flips between them, which is why its spread is 32.8%. Expected gap is the
central metric, and it is unchanged. The cost of packing is ~0.05 ms of host staging and
<= 1.4% on the prefill step. Tests: `tests/batching/test_packed_fused_prefill.py`
(eager and graphed, real multi-row fused steps, leading tokens match stock). Packing stays
opt-in until it is exercised with INT8 KV and Flash prefill; it does not apply to the
prefill-only graph path.

**Prefill chunk/budget sweep with packed rows** (`ab_prefill_budget_packed_chat_qwen3_1.7b.json`;
Qwen3-1.7B FP16 chat profile, 512 blocks, closed loop at concurrency 6, tiled prefill,
graphs, three interleaved repeats, token gate passed, zero lazy captures). Every arm binds,
because chat prompts are 288-1024 tokens:

| chunk / budget | expected gap | ITL p99 | ITL p999 | TTFT p50 | delivered tokens (12 s) | prefill step p50 |
|---|---:|---:|---:|---:|---:|---:|
| 128 / 128 (default) | 22.8 ms | 29.2 ms | 30.2 ms | 139 ms | 2662 | 27.0 ms |
| 64 / 64 | **-14.3%** | **-23.9%** | **-22.7%** | **+168%** | -13% (spread 11%) | -25.8% |
| 128 / 256 | +30.2% | +56.2% | +52.6% | +28% (unres.) | -18% | +61.3% |
| 256 / 256 | +19.6% | +62.4% | +62.9% | -1.8% (unres.) | -9% (unres.) | +64.8% |

Larger budgets are dominated: they lengthen every prefill-carrying step without improving
TTFT or throughput at this concurrency. The step cost is compute-bound, so a 256-token
fused step costs ~1.6x a 128-token one, while closed-loop arrivals rarely leave a prompt
waiting long enough for the larger budget to pay off. The derived policy keys on the
latency objective rather than a universal constant:

- **default / throughput and TTFT:** chunk 128, budget 128;
- **ITL-SLO mode:** chunk 64, budget 64. It cuts the ITL p99 by a quarter in exchange for
  roughly 2.7x the median TTFT. Use it only when inter-token tail latency is the binding SLO;
- **do not raise the budget above the chunk** on this card at these concurrencies.

Open-loop arrival rates above service capacity, where queued prompts could benefit from a
larger budget, are not covered by this sweep.

## Priority 7: low-batch fixed overhead and speculative decoding

**Host overhead is not material** (`low_batch_fixed_overhead.json`, probe
`benchmarks/batching/low_batch_overhead_probe.py`). The probe compares decode-only steps
(wall clock, engine instrumentation on) with back-to-back replays of the same captured
graph, at FP16, 128-token prompts and batches 1/2/4:

| model | batch | step wall | pure graph replay | host staging | wall - replay |
|---|---:|---:|---:|---:|---:|
| 0.6B | 1 / 2 / 4 | 5.53 / 5.64 / 5.89 ms | 5.52 / 5.54 / 5.60 ms | 0.06 ms | 0.01 / 0.10 / 0.28 ms |
| 1.7B | 1 / 2 / 4 | 14.76 / 14.60 / 14.90 ms | 14.46 / 15.35 / 14.83 ms | 0.07-0.08 ms | 0.31 / -0.76 / 0.07 ms |

Staging, synchronization and launch together are at most ~5% and usually below 2%, inside
the replay-to-replay variance. The batch-1 0.6B step is 1.19x the 4.63 ms weight-read
floor. GPU-resident metadata and launch work are therefore **rejected as non-binding**;
the low-batch levers are fewer weight bytes (W8A16, AWQ) and more tokens per weight read.

**Speculative decoding: n-gram prompt lookup.** This needs no draft model and no extra VRAM.

*Cost model* (`benchmarks/speculative/ngram_cost_model.py`). The verify cost is a
CUDA-graphed stock forward of k tokens over a StaticCache at context 512, relative to k=1.
transformers 5.17's StaticLayer advances an in-place device counter, which is reset inside
the graph. Acceptance is simulated exactly on stock greedy continuations of 16 prompts x
128 tokens:

| prompt set | model | verify cost k=4 / 8 | tokens per verify step (draft 3 / 7) | modelled speedup |
|---|---|---:|---:|---:|
| copy-heavy (repository docs/code context) | 0.6B | 1.090x / 1.101x | 1.68 / 1.83 | 1.54x / 1.66x |
| copy-heavy | 1.7B | 1.056x / 1.060x | 1.87 / 2.11 | 1.77x / 1.99x |
| free generation (16 chat instructions) | 0.6B | 1.093x / 1.093x | 1.11 / 1.11 | 1.01x / 1.01x |
| free generation | 1.7B | 1.056x / 1.061x | 1.09 / 1.10 | 1.04x / 1.04x |

A pure-Python lookup costs 284 µs per step (2048-token history, no match). That is ~5% of a
0.6B step and 2% of a 1.7B step, enough to erase the free-generation gain. An engine
version needs an incremental suffix index.

*Batch-1 end-to-end probe* (`benchmarks/speculative/ngram_batch1_probe.py`,
`spec_ngram_batch1_probe_1.7b.json`). Graphed greedy and graphed width-5 verify run on
the same StaticCache, interleaved per request, three repeats, host lookup included.
Rollback resets the cache counter and positions only.

- Copy-heavy: aggregate **1.81x** (median 1.72x, range 1.33-2.89x, 1.96 tokens/step).
- Free generation: **1.04x** (range 0.96-1.11x).
- Correctness: 14/16 requests are bit-identical to greedy. The other two first differ at
  tokens 68 and 54, where the stock top-2 margin is 0.0156 and 0.031. That is fp16 drift from
  the width-5 versus width-1 GEMM shape at near-ties, not a verification error.

**Decision.** N-gram speculation is a workload-conditional batch-1 win. It pays for
retrieval, summarization and code-editing traffic that copies from its context, and is
neutral for free chat. It is not integrated yet. An engine version should be per-request
and adaptive: disable drafting for a request whose rolling acceptance falls below
~0.1 tokens/step. It should carry k-token verify rows through the packed fused layout,
which already supports variable tokens per row, plus page-table length rollback. Benefit
falls with batch size, because the verify cost stops being free once decode is no longer
weight-bound, so it needs its own batch crossover A/B before any default.

## Priority F.1: W8A16 MLP kernel search (Qwen3-1.7B)

`benchmarks/kernels/w8a16_mlp_search.py` searches `engine/kernels/w8a16_tuned.py` on the
real layer-0 gate/up (N=6144, K=2048) and down (N=2048, K=6144) weights. The search covers
tiles (BM 16-128, BN 32-256, BK 64-256), warps 2/4/8, stages 2-4, program grouping 1/8,
and whether the per-channel scale is applied in the loop or once to the FP32 accumulator.
Every candidate is validated against an FP32 dequantization reference before timing, and
rounds are shuffled. 32 configurations failed with `OutOfResources` (BK=256 at 3-4 stages)
and are recorded.

**Methodology defect caught.** The first run timed a 12.5 MB INT8 projection at 23 µs. That
is 540 GB/s, impossible from DRAM: the weight stayed resident in the 24 MB L2 across
repeated calls. The search now cycles a ring of 8 weight copies per call, so every call
streams from DRAM; the L2-hot run was discarded. After the fix, decode gate/up time sits at
the 257 GB/s roofline (12.6 MB in 51.9 µs), where the existing kernel already was.

**Findings** (`w8a16_mlp_search_1.7b.json`, `w8a16_mlp_search_1.7b_fused_rows.json`; layer
MLP = 2 x gate + down):

| rows M | current | searched | gain | note |
|---|---:|---:|---:|---|
| 1-16 (decode) | 168-171 µs | 156 µs | 1.08-1.09x | all from down_proj: BN=128 gave 16 programs on 24 SMs; BN=32 fills the GPU |
| 32 | 175 µs | 161 µs | 1.09x | |
| 64 / 128 / 256 | 222 / 429 / 851 µs | 195 / 380 / 726 µs | 1.14 / 1.13 / 1.17x | at M >= 128 now at or below cuBLAS FP16 (396 / 693 µs) |
| 129-136 (fused steps) | - | - | 1.00-1.05x | the BM=16 current kernel already wastes little |
| 260-264 | - | - | 1.07-1.10x | |

One configuration per row bucket comes within 2% of separate gate/up and down
configurations, so no split is needed. The first dispatch used a 64-row tile above 96
rows, and the end-to-end chat A/B regressed the fused step by 5.6%. Fused steps run at
M = decode bucket + 128 x rows (129-136), where a 64-row tile wastes a third of its last
tile. The power-of-two search had not covered those M values. The table was re-derived
over realistic M:

- M <= 16: 16x32x128, 4 warps, 3 stages;
- M <= 48: 32x128x128, 8 warps, 3 stages;
- otherwise: 32x64x64, 4 warps, 3 stages, group 8, scale after accumulation (1.00-1.15x at
  every M from 64 to 264).

The table is static: no runtime autotuning, and graphs capture the configuration chosen
from the static row count. First compile costs 0.1-0.5 s per configuration, paid at
warmup.

**End to end** (`ab_w8a16_tuned_*_qwen3_1.7b.json`): W8A16 Qwen3-1.7B, graphs, tiled
prefill, three interleaved repeats, same quantized weights in both arms:

| workload | expected gap | decode step p50 | ITL p99 | TTFT p50 | prefill step p50 |
|---|---:|---:|---:|---:|---:|
| batch 1, short | **-3.8%** (spread 0.2%) | **-3.5%** | **-2.9%** | **-6.9%** | **-7.0%** |
| chat, concurrency 6 | -2.5% (unres.) | **-4.6%** | **-5.8%** | -4.5% (unres.) | -1.6% (unres.) |

The token gate passed in both runs. Prompt 0 first differs at token 16 in either arm
across runs, which is a W8A16 near-tie. Model memory is unchanged. **Accepted:**
`quantize_mlp_w8a16` now installs the searched dispatch by default (`tuned=False` restores
the old kernel); W8A16 itself remains opt-in.

**Engine defect found on the way.** With `max_active=1`, the single graph bucket
reserved no dummy page, so every fused-graph capture failed and fused steps ran eagerly.
The engine now reserves at least one dummy page.

## Priority F.2: INT8 KV decode search

`benchmarks/kernels/int8_decode_search.py` (`int8_decode_search.json`) covered batches
1/4/8/16 and contexts 512-8192, with a ring of >= 64 MB of per-layer pools so K/V stream from
DRAM. It searched the per-head launch space (BLOCK_N 32-256, warps 2/4/8, stages 1-3) and a
new paired-GQA kernel (`engine/kernels/int8_decode_tuned.py`): one program per
(row, KV head) scores both Qwen3 query heads against a single INT8 K/V-plus-scale load.
Every candidate was validated against the current kernel (max abs error <= 2e-3); none failed.

| regime | current | best | gain | interpretation |
|---|---:|---:|---:|---|
| batch 1, 512-8192 | 23.9-174 µs | per-head 256x4 | **1.07-1.24x** | latency-bound: 16 programs on 24 SMs, ~97 GB/s at 4k. Fewer, longer tiles shorten the dependent chain |
| batch 4 | 23.6-285 µs | current | 1.00x | already ~235-240 GB/s, the roofline |
| batch 8 | 42-627 µs | per-head 64x2 or paired 128x4 | 1.11-1.17x (isolated) | |
| batch 16 | 73-1127 µs | paired 32x2 | 1.04-1.06x | at roofline; paired halves the loads but is bandwidth-bound |

At batch 1 the INT8 kernel is no faster than FP16, because the regime is latency-bound,
not byte-bound. At batch >= 4 it runs 1.8-1.9x faster than FP16, tracking its byte ratio.

**Static dispatch** (opt-in `int8_decode_tuned=True`). It is keyed on the launched rows
(graph bucket) and the existing tile regime, both already in the decode graph key, so
replay never changes launch. End to end on Qwen3-0.6B (smoothed INT8 plus tiled INT8
prefill in both arms, three interleaved repeats, `int8_decode_tuned_ab_*.json`):

- batch-1 rule (256x4, 3 stages) on the `b1_long` profile (2-4k prompts, max_active 1):
  decode step **-2.8%** (7.09 to 6.89 ms; spreads 2.6% / 1.9%), ITL p99 -5.5%, TTFT and
  throughput unchanged (arrival-bound). Chat profile: neutral (6.410 versus 6.405 ms).
  **Accepted** as opt-in.
- batch-8 rule (64x2): chat decode step regressed **+2.7%** end to end despite 1.11-1.15x
  isolated. **Rejected**; it can still be reproduced via the module flag. The likely cause
  is mixed real lengths and padded rows in live buckets versus the uniform-length
  microbenchmark; not investigated further.
- The paired-GQA kernel is not selected anywhere. It was at most ~2% better than per-head
  in the isolated search and never tested end to end.

## Priority E: W8A16 on chat and long workloads

`benchmarks/quantization/w8a16_mlp_ab.py` now accepts `--prompt-profile` and
`--prefill-graphs`. It compares independent FP16 and selective-MLP W8A16 (searched
dispatch) Qwen3-1.7B instances under the current engine: tiled page-16 prefill, decode,
prefill and fused graphs, three interleaved repeats, cancellation off
(`w8a16_mlp_full_engine_{chat_c4,long_c2}_tuned.json`):

| metric | chat, concurrency 4 | long, concurrency 2 |
|---|---:|---:|
| decode step p50 | **-26.7%** | **-25.5%** |
| expected gap | **-12.8%** | **-14.8%** |
| fused step GPU p50 | **-6.5%** | **-7.0%** |
| prefill step p50 | **-6.4%** | **-7.2%** |
| prefill-only GPU p50 | +3.2% | +3.5% |
| ITL p99 / p999 | **-4.4%** / -3.9% (unres.) | **-6.7% / -5.8%** |
| TTFT p50 / p99 | -15.4% (unres.) / **-12.9%** | -8.3% / +2.6% (both unres.) |
| delivered tokens | **+13.8%** | +7.7% (unres.) |

Prefill tails do not regress. The fused prefill+decode step, the common prefill path under
load, is faster. Prefill steps with no decode rows are 3% slower: they run large-M
projections, where FP16 cuBLAS is competitive.

**Late drift, measured directly** (`w8a16_teacher_forced_1.7b.json`; teacher forcing on
committed natural text versus the FP16 model):

| context | top-1 agreement | mean KL | perplexity FP16 to W8A16 |
|---|---:|---:|---:|
| 1024 (16 sequences) | 98.1% | 0.0022 | 11.37 to 11.28 |
| 4096 (8 sequences, last 512 scored) | 98.3% | 0.0018 | 7.92 to 7.85 |

The drift does not grow with context. The greedy first divergences at tokens 13 and 20
on the identity prompts are ordinary near-tie flips at a ~1.8% per-token disagreement rate,
the same quality class as smoothed INT8 KV. W8A16 remains an explicit opt-in because it is
lossy, but the open questions (chat/long prefill tails and late drift) are now answered in
its favour.

**Test-suite note.** New CUDA tests initially imported the shared loaders from
`tests/batching/test_continuous_batching.py`. Under pytest's import mode that created a
second module copy that pinned two extra models, and later CUDA tests ran out of memory
(21 spurious failures). The new tests now use module-scoped fixtures that free their model.
Full CUDA suite at that point: 248 passed, 1 skipped (the expected 8 GB speculative parity skip);
CPU suite: 311 passed.

## Priority E: Qwen3-4B-AWQ on chat and long prompts

**AWQ is prefill-bound.** Engine defaults (tiled page-16 prefill; graphed decode; prefill
eager because of the tinygemm exclusion), chat concurrency 4 / long concurrency 2, 512 blocks:

- a 128-token prefill or fused step costs ~158 ms against a 14.2 ms decode step;
- prefill is 80-83% of the expected token gap, and TTFT p50 is 0.87 s (chat) and 2.5 s (long).

Tiled versus SDPA attention is neutral here (-0.1% to -1.3%;
`ab_prefill_tiled_{chat,long}_qwen3_4b_awq.json`), because the INT4 projections dominate.
Per 7-projection layer at 128 rows, tinygemm takes 3.73 ms against 1.05 ms for cuBLAS FP16 on
dequantized weights: a 3.5x gap. No external INT4 runtime (torchao, Marlin, AWQ, GPTQ)
is installed, so a new kernel was written instead.

**Triton W4A16 prefill GEMM over tinygemm's own buffers**
(`engine/kernels/int4pack_gemm.py`; no second weight copy). The undocumented
`int4pack` layout (innerKTiles 8, `[N/8, K/128, 32, 4]` int32) was derived with bit-coded
probes and verified as a translation-invariant bijection per 8x128 tile:

- `lane = 4(n mod 8) + (k mod 8) div 2`
- `word = (k mod 128) div 32`
- `nibble = (k mod 32) div 8 + 4(k mod 2)`

- **v1 (rejected)**, a per-element int32 gather: 3.35 ms versus tinygemm's 0.51 ms at
  16 rows, and only 1.09x at 256 rows (`int4pack_prefill_search_4b_v1_rejected.json`).
- **v2 (accepted)**:
  - each row's 16 packed words load as one coalesced vector and unpack in packed order;
  - the activation tile is gathered at the matching permuted columns, so the dot product
    is unchanged;
  - dequantization factors out of the reduction, `scale x (x . (q-8)) + zero x rowsum(x)`,
    with q-8 exact in FP16.

  Error against exact dequantization is 0.15-0.3%, against 1.6-2.7% for tinygemm,
  which rounds activations to bf16. On the real layer-0 projections, DRAM-streamed, with
  no failed candidates (`int4pack_prefill_search_4b_v2.json`), one launch
  (32x128x128, 4 warps, 2 stages) gives the following:

  | rows | tinygemm layer | v2 layer | speedup |
  |---:|---:|---:|---:|
  | 16 | 0.51 ms | 0.58 ms | 0.88x (keep tinygemm) |
  | 64 | 1.90 ms | 1.11 ms | 1.71x |
  | 128 | 3.73 ms | 2.01 ms | 1.85x |
  | 132 | 3.86 ms | 2.63 ms | 1.46x |
  | 256 | 7.45 ms | 3.66 ms | 2.00x |

**End to end** (`ab_awq_int4pack_prefill_{chat,long}_qwen3_4b.json`). Inputs of >= 64 rows
use v2 and decode keeps graphed tinygemm; three interleaved repeats:

| metric | chat, c4 | long, c2 |
|---|---:|---:|
| prefill GPU p50 | **-46.7%** | **-46.2%** |
| fused step GPU p50 | **-33.9%** | **-33.5%** |
| prefill step p50 | **-33.8%** | **-34.5%** |
| ITL p99 / p999 | -43.2% (unres.) / **-43.8%** | **-33.4% / -33.5%** |
| expected gap | -29.4% (unres.; spread 29.8%) | **-31.2%** |
| TTFT p50, paired by repeat | 503/860/1218 to 346/593/664 ms | 1970/2474/4261 to 898/1665/1693 ms |
| decode step | unchanged | unchanged |

The TTFT medians are formally unresolved because each repeat uses a new soak seed, but
all six paired repeats improve, by 31-60%. Token gate: the Triton arm matches stock on all
four identity prompts; tinygemm first diverges at token 20, because it rounds activations
to bf16. **Accepted:** `load_awq_tinygemm_model(triton_prefill_rows=64)` is now the default
for the opt-in AWQ path; `None` restores tinygemm everywhere.

**Prefill graphs for AWQ.** With the Triton prefill GEMM, prefill/fused graph capture still
raised chat ITL p99 by 42% and p999 by 8%, and TTFT rose in all three pairs; the gain was
only 0.7% on the prefill step (`ab_awq_triton_prefill_graphs_chat_qwen3_4b.json`). The cause
is graph padding of chunk rows (Priority 6). Graphs with packed rows were inconclusive
(`ab_awq_graphs_packed_chat_qwen3_4b.json`): ITL p999 -34% (resolved), but p99 -13%
unresolved, TTFT pairs +17% / +4% / -6%, and step medians under 1%. AWQ prefill stays eager.

## Priority F.5 and a new W8A16 target: where decode kernel time goes

`benchmarks/kernels/decode_kernel_share.py` profiles eager decode steps (Qwen3-1.7B, batch
4, 512-token context) by kernel category (`decode_kernel_share_1.7b*.json`):

| category | FP16 | W8A16 MLP |
|---|---:|---:|
| FP16 GEMM | 90.9% (13.9 ms) | 48.8% (5.4 ms) |
| W8A16 GEMM | - | 38.7% (4.3 ms) |
| attention | 6.9% | 9.6% |
| RMSNorm + RoPE + SwiGLU + KV write + other | **2.1%** | **2.8%** |

**F.5 (elementwise and fused kernels) is rejected as non-binding.** They are under 3% of
kernel time, and graphs already remove their launch gaps.

Under W8A16 the largest remaining FP16 read is the tied vocabulary projection: 151936 x 2048,
622 MB, ~2.4 ms per step. Attention projections were rejected for W8A16 because of their
small N, but `lm_head` is bandwidth-bound like the MLP. With the searched kernel it ran at
1.21 versus 2.41 ms (2.0x) at 1-16 rows. `quantize_lm_head_w8a16` (opt-in,
`engine/kernels/w8a16_tuned.py`) gives the projection its own INT8 copy. The input
embedding keeps the FP16 table, so storage rises by ~311 MB: 2.221 to 2.511 GiB (measured), still
below FP16's 3.20 GB. Quantization runs in row chunks, because a single FP32 pass is a
1.2 GB temporary.

- Quality (`w8a16_teacher_forced_1.7b_lm_head.json`): top-1 98.04% / 98.29% at 1k / 4k
  context, against 98.14% / 98.32% for MLP-only. KL is 0.00234 / 0.00191 against
  0.00218 / 0.00179, and perplexity is unchanged.
- End to end (`ab_w8a16_lm_head_{b1,chat_c6}_qwen3_1.7b.json`; both arms W8A16 MLP, graphs,
  three interleaved repeats):

  | metric | batch 1, short | chat, concurrency 6 |
  |---|---:|---:|
  | decode step p50 | **-11.8%** (spread 0.2%) | **-9.3%** |
  | expected gap | **-11.6%** | **-6.9%** |
  | ITL p99 | **-11.4%** | -1.4% (unres.) |
  | TTFT p50 | **-6.4%** | -2.4% (unres.) |
  | prefill / fused step | **-6.4%** | **-4.5%** |

  The token gate passed; the W8A16 lm_head arm first differs at tokens 16 and 25.
  **Accepted as opt-in.** Combined with MLP W8A16 and the searched kernel, the batch-1
  decode step on Qwen3-1.7B is 8.95 ms, against ~14.5 ms for FP16 in the same session.

## Checkpoint, 2026-10-16

Gates at this checkpoint: CPU suite 314 passed; CUDA suite 259 passed, 1 skipped (the
expected 8 GB two-model speculative parity skip); `git diff --check` clean; every changed
or new Python file compiles. Nothing is committed.

**Accepted.** Each was decided by an interleaved end-to-end A/B plus a correctness gate.

| path | status | key end-to-end result |
|---|---|---|
| INT8 KV + calibrated key smoothing + tiled INT8 prefill | opt-in | equal bytes: chat +14.7% throughput, TTFT -35%; long +67% throughput under pressure; top-1 ~98% |
| tiled FP16 prefill as the Ada page-16 fallback | **default** (policy fallback) | prefill step -3% to -12.5%, ITL p99 -9% to -43% (0.6B/1.7B, all profiles) |
| packed fused prefill rows | opt-in | ITL p99 -18% to -33%, p999 -33% to -55%, no median cost |
| searched W8A16 MLP dispatch | default inside the opt-in W8A16 path | decode -3.5% (batch 1), -4.6% (concurrency 6) |
| W8A16 lm_head | opt-in (+290 MB) | decode -11.8% (batch 1), -9.3% (concurrency 6) |
| tuned INT8 decode, batch-1 rule | opt-in | decode -2.8% at batch 1 / 2-4k context |
| Triton int4pack prefill for AWQ | default inside the opt-in AWQ path | prefill GPU -46%, fused step -34%, ITL p99 -33% |

**Engine defects fixed:** INT8 spill page count (regression test added); SDPA context
buckets exceeding the pool; padded-tile KV waste in both tiled prefill kernels; no graph
dummy page at `max_active=1`. The soak warmup now prints graph-capture failures.

**Rejected:**

- plain per-token INT8 keys (quality);
- pinned spill (slower: 108 to 245 ms per write);
- larger prefill budgets;
- GPU-resident metadata (host overhead <= ~5%);
- elementwise kernel work (< 3% of kernel time);
- INT8 decode batch-8 rule (+2.7% end to end);
- int4pack v1 gather kernel;
- AWQ prefill graphs without packing (ITL p99 +42%);
- the L2-hot W8A16 timing methodology.

**Unresolved:**

- AWQ graphs with packed rows (p999 -34% but TTFT mixed);
- paired-GQA INT8 decode (at most ~2% in isolation, never tested end to end);
- n-gram speculation, integration still needed: 1.81x on copy-heavy text, 1.04x on free chat;
- open-loop arrivals above capacity for the budget policy.

**Environment caveats:**

- The GPU link runs at PCIe Gen4 x1 (~1.4 GB/s host transfers).
- A host process held ~91% of a CPU until ~21:00 on 2026-10-15; compare only within one
  interleaved run.

**Next hypotheses, in order:**

1. The stacked 1.7B configuration (W8A16 MLP + lm_head + smoothed INT8 KV + tiled INT8
   prefill + packed rows) should beat FP16 on both latency and KV capacity at equal VRAM,
   without compounding quality loss beyond each component's KL.
2. Engine n-gram speculation as a per-request opt-in with an adaptive cutoff; gate it with
   a batch crossover A/B.
3. Packed rows for the prefill-only graph path, which still pads rows to the chunk width.

## Stacked Qwen3-1.7B configuration versus FP16 at equal VRAM

Hypothesis 1 from the checkpoint. The stacked arm is W8A16 MLP (searched dispatch), W8A16
lm_head, INT8 KV with calibrated key smoothing, tiled INT8 prefill and packed fused rows.
FP16 is the current engine default (tiled page-16 prefill, graphs). Each arm gets a 5.0 GiB
budget for weights plus KV pool:

- FP16: 3.205 GiB of weights, leaving 1050 blocks (16.8k tokens);
- stacked: 2.511 GiB of weights, leaving 2867 INT8 blocks (45.9k tokens, 2.7x).

Peak allocated memory is identical (5.40 GB) in both arms. Harness:
`benchmarks/batching/stacked_vs_fp16_ab.py`; Poisson arrivals over natural-text prompts;
each model reloaded per run, arm order alternating; three repeats; every run passed the page
audit.

**Quality, teacher-forced** (`stacked_teacher_forced_1.7b.json`): top-1 97.3% / 97.7% at
1k / 4k context, KL 0.0041 / 0.0038, perplexity 11.37 to 11.30 and 7.92 to 7.84. The KL is
about the sum of the W8A16 (~0.0023) and INT8 KV (~0.0019) components: the errors add
rather than compound, and do not grow with context.

| metric (`stacked_vs_fp16_*_1.7b.json`) | chat (32 req, 4/s, max_active 8) | long_c8 (3-4k prompts, max_active 8) |
|---|---:|---:|
| throughput | **+38.1%** (318 to 439 tok/s) | **+53.6%** (83.7 to 128.5 tok/s) |
| ITL p50 | **-37.7%** | +30.2% (operating point; see below) |
| ITL p99 | **-39.5%** | **-6.5%** |
| TTFT p50 / p99 | **-54.9% / -54.4%** | **-43.9% / -56.1%** |
| queue p95 | **-56.0%** | **-68.4%** |
| preemptions / recomputed tokens | 0 / 0 both | **9 / 30,060 to 0 / 0** |
| decode step p50 | **-38.1%** | **-42.9%** |
| prefill step p50 | **-13.6%** | +1.4% |

Every change exceeds run-to-run spread (spreads 0.1-2.5%; queue p95 12%). In long_c8 the
FP16 pool holds only ~4 concurrent 4k requests, so the rest queue or are preempted. The
stacked pool holds all eight, so more of its steps carry prefill chunks and the median
token gap rises. Its p95/p99 gaps, TTFT and throughput all improve.

Token gate: the stacked arm has early non-tie flips at the rate its ~2.7% per-token
disagreement predicts (5/32 chat requests, 2/12 long). It is lossy by construction; FP16
has none.

**Decision.** For Qwen3-1.7B on this 8 GB card, the stacked configuration is the
recommended opt-in serving profile whenever ~97-98% teacher-forced top-1 agreement is
acceptable. At the same VRAM it gives 38-54% more throughput, roughly half the TTFT and
2.7x the KV capacity. The default stays FP16 because the stack is lossy.

**Open-loop overload check** (`open_loop_budget_chat_1.7b_r8.json`;
`benchmarks/batching/open_loop_budget_ab.py`). Qwen3-1.7B FP16, chat prompts, 48 Poisson
arrivals at 8/s against ~350 tok/s of service (queue p95 ~10 s), 1000 blocks, max_active 8,
packed rows, three interleaved repeats. Every change below exceeds spread (<= 1.3%):

| chunk / budget | throughput | ITL p99 | TTFT p50 / p99 | queue p95 |
|---|---:|---:|---:|---:|
| 128 / 128 | 350 tok/s | 29.3 ms | 5.9 / 10.0 s | 9.7 s |
| 128 / 256 | -13.9% | +56.0% | +17.9% / +29.0% | +27.9% |
| 64 / 64 | -3.0% | **-24.6%** | +9.4% / +6.1% | +4.0% |

A larger budget does not drain the queue faster even under overload; it is dominated
again. Under overload, 64 / 64 costs only +6-9% TTFT, because queueing dominates TTFT,
against +168% closed-loop. That makes it a much cheaper ITL-SLO mode than the closed-loop
sweep suggested.

**Confound found.** With packed rows, every fused graph is captured at the full budget
width, so a step that plans 40 chunk tokens still runs `budget` tokens through the
per-token modules. That inflates every arm whose budget exceeds the typical step,
budget 256 most of all. The next experiment buckets the packed width.

**Packed width buckets** (`packed_prefill_width_buckets=True`, opt-in alongside packed rows).
Fused graphs are now also captured at powers of two from 32 up to the budget, and each
step replays the smallest width that holds its planned chunk tokens; the width is part of
the fused graph key and of the inert capture. Tests: `test_packed_fused_prefill.py`, eager
and graphed with buckets on, INT8 composition included, zero lazy captures. Qwen3-1.7B
chat, concurrency 6, three interleaved repeats:

| comparison | fused GPU p50 | prefill step p50 | expected gap | TTFT p50 | ITL p99 |
|---|---:|---:|---:|---:|---:|
| buckets vs budget width, budget 128 (`ab_packed_width_buckets_chat_qwen3_1.7b.json`) | -0.9% (unres.) | -1.0% (unres.) | -0.9% (unres.) | -0.5% (unres.) | +0.8% (unres.) |
| buckets vs budget width, chunk 128 / budget 256 (`ab_packed_width_buckets_b256_chat_qwen3_1.7b.json`) | **-36.3%** | **-35.8%** | **-25.0%** | -22.3% (all 3 pairs lower) | -0.0% (unres.) |

At budget 128 almost every fused step already carries a full 128-token chunk, so
bucketing is neutral. At budget 256 it removes most of the earlier budget penalty.

**Budget sweep repeated with buckets** (`ab_prefill_budget_bucketed_chat_qwen3_1.7b.json`,
token gate passed, zero lazy captures):

- 128 / 256 is now even with 128 / 128 on expected gap (-1.1%), TTFT and delivered tokens,
  all within spread. Previously it was +30% expected gap and -18% tokens.
- 128 / 256 still raises ITL p99 by +56.7%, because the steps that really carry 256 tokens
  form the tail.
- 256 / 256 remains dominated: expected gap +19%, ITL p99 +63%.
- 64 / 64 reproduces the ITL-SLO trade: ITL p99 -23.8%, TTFT +165% closed-loop.

The earlier explanation, that larger budgets lengthen *every* prefill step, is corrected:
that was the fixed packed graph width. The budget policy itself is unchanged. 128 / 128 is
the default, and 64 / 64 is the ITL-SLO mode (cheap under overload). A budget above the chunk
buys no throughput or TTFT at these loads and costs tail ITL. Enable width buckets
whenever a packed engine runs a budget above its chunk size.

**AWQ prefill graphs, retested with width buckets**
(`ab_awq_graphs_packed_buckets_chat_qwen3_4b.json`; Qwen3-4B-AWQ with Triton int4pack
prefill, chat concurrency 4, three interleaved repeats). The arms were eager prefill against
graphed prefill/fused steps with packed rows and width buckets:

- ITL p999 **-34.0%** (spread 1.2%);
- prefill and fused GPU **-0.8% to -1.1%** (beyond spread);
- TTFT lower in every pair (381 to 353, 592 to 587, 663 to 620 ms); ITL p99 -13.6% (unresolved);
- decode unchanged, token gate passed.

The earlier inconclusive graphed+packed result came from full-budget-width packed graphs.
The engine now captures AWQ prefill only for this opt-in combination: packed rows, width
buckets, and every AWQ projection routing full chunks to the Triton GEMM. Plain graphs
(+42% ITL p99) and tinygemm prefill stay eager.

## Checkpoint update, 2026-10-16 (later)

Gates: CPU 314 passed; CUDA 261 passed, 1 skipped (expected); `git diff --check` clean.
Nothing is committed.

**Added since the first checkpoint:**

- **Accepted (opt-in):** packed width buckets (essential whenever budget > chunk), and AWQ
  prefill graphs for the packed + buckets + Triton combination only.
- **Measured:**
  - the stacked 1.7B profile against FP16 at equal VRAM: +38% / +54% throughput, about half
    the TTFT, and 2.7x the KV capacity;
  - open-loop overload: 64 / 64 is a cheap ITL-SLO mode there.
- **Corrected:** the earlier explanation of the large-budget penalty, which was really the
  fixed packed graph width.

**Speculative decoding: integration design, not started.** The batch-1 probe gave 1.81x on
copy-heavy text and 1.04x on free chat. An engine version needs:

1. per-request greedy-only opt-in (no penalties or sampling);
2. verify rows `[next_token] + draft` at positions L..L+k-1 carried as packed chunk rows of
   a DECODING request, with logits at every packed position (`P x vocab`) instead of the
   last position only;
3. acceptance on device or host; commit of the a+1 tokens through `append_token` with a
   finish-reason check per token; `append_tokens(a + 1)` on the allocation, with stale K/V
   past it overwritten later;
4. capacity for L+k reserved up front;
5. a captured verify graph per (rows bucket, packed width), because eager verify loses to
   graphed decode;
6. an adaptive cutoff: stop drafting for a request after a rolling acceptance below ~0.1
   tokens per step;
7. an incremental suffix index replacing the 284 µs Python lookup.

Gate: bit-identity to plain greedy except at stock near-ties, then a batch 1/2/4/8
crossover A/B on copy-heavy and free-chat traffic.

## Single-request prefill: idle whole-prompt policy

`benchmarks/inference/single_request_compare.py` (one active request, 128 new tokens,
EOS ignored, medians of three runs; the committed HEAD engine is exported with `git
archive` and run under its own defaults) found the engine slower than stock explicit
decode to the first token on 2k-token prompts: 157 versus 143 ms on 0.6B and 332 versus
314 ms on 1.7B. The chunked path ran sixteen 128-token steps even with nothing decoding.

`idle_prefill_tokens` (default 4096) now prefills a prompt whole through the existing
single-forward fast path when it is the only pending prompt and nothing is decoding.
Anything else keeps the chunk budget, so ITL under load is untouched. Two variants were
rejected on the way (`ab_idle_prefill_1.7b_chat_c4.json`, four closed-loop clients): taking
every fresh prompt whole, and batching fresh prompts up to 25% padding, both raised TTFT in
all three pairs (105 to 128, 120 to 123, 150 to 184 ms). Interleaved chunks give a burst of
prompts earlier first tokens than serial whole prefills. The single-prompt rule is neutral
there (104 to 104, 121 to 120, 150 to 146 ms).

| workload (3 interleaved repeats) | TTFT p50 | ITL p50 / p99 | tokens |
|---|---:|---:|---|
| 0.6B long, one client (`ab_idle_prefill_0.6b_long_c1.json`) | **-32%** (99/133/122 to 66/86/82 ms) | +0.1% / +1.2% (unres.) | identical |
| 1.7B long, one client (`ab_idle_prefill_1.7b_long_c1.json`) | -15% (220/264/240 to 149/215/204 ms; every pair lower) | +1.3% / +2.6% (unres.) | identical |
| 1.7B chat, four clients | -1.0% (unres.) | +0.5% / +0.3% (unres.) | one exact-tie flip (margin 0.0000) |
| 1.7B chat, Poisson 4/s, max_active 8 (`idle_prefill_poisson_chat_1.7b.json`) | +0.8% (unres.) | -1.0% / **-29%** | gate passed |

The fast path is the same forward the engine already used for prompts up to one chunk, so
this is lossless; it is the one new default in this pass. Tests that exercise the chunked
mechanism at idle pin `idle_prefill_tokens=None`.

# RTX 4060 paged decode: memory and occupancy counters

Measured 2026-10-15 on the local RTX 4060 (`sm_89`) with Nsight Compute 2025.4.1.
The target is one warmed attention-layer kernel on FP16 Qwen3-0.6B geometry:
16 query heads, 8 KV heads, head dimension 128, 16-token pages. Input pools use
shuffled physical block IDs. The script is
`benchmarks/kernels/profile_paged_decode_memory.py`; Nsight Compute was launched
through desktop administrator authentication because `RmProfilingAdminOnly=1`.

| Backend / batch / context | Registers/thread | Theoretical occupancy | Achieved occupancy | Long-scoreboard cycles per issued instruction | Memory throughput |
|---|---:|---:|---:|---:|---:|
| per-head / 8 / 1024 | 128 | 33.33% | 29.44% | 15.24 | 243.20 GB/s |
| per-head / 8 / 256 | 128 | 33.33% | 28.22% | 5.75 | 214.69 GB/s |
| paired GQA / 8 / 256 (`64x2`) | 168 | 25.00% | 10.95% | 0.41 | 229.45 GB/s |
| per-head / 1 / 4096 | 128 | 33.33% | 8.33% | 2.38 | 189.84 GB/s |
| split-K 8-way `64x2` / 1 / 4096 | 153 (160 allocated) | 25.00% | 21.81% | 2.77 | 241.15 GB/s |
| split-K 8-way `32x2` / 1 / 4096 | 96 | 41.67% | 21.89% | 4.72 | — |
| split-K merge / 1 / 4096 | 25 (32 allocated) | 100% | 8.31% | — | 28.10 GB/s |

The per-head kernel has no local-memory spilling. Its 128 registers/thread limit
residency to four 128-thread blocks per SM. At batch 1 there are only 16 blocks
for 24 SMs, explaining the much lower achieved occupancy. At batch 8/context
1024, schedulers report no eligible warp in 77.25% of cycles; the long scoreboard
is the dominant stall. This is direct evidence to study K/V load latency and
register pressure. It does not show that a smaller register count will win by
itself: each alternative needs a timing and token gate.

For the long-context batch-1 split-K investigation, the scheduler had no
eligible warp in 76.58% of cycles for the `64x2` split kernel, versus 76.64%
for per-head. However, the split launch has 128 CTAs instead of 16, raising
achieved occupancy from 8.33% to 21.81%. Its small merge has only 16 CTAs,
so its 8.31% achieved occupancy is expected; it took 2.59 µs in the profiled
launch. Neither split kernel spilled registers.

At `32x2`, the split kernel used only 96 registers/thread and its no-eligible
fraction fell to 53.91%, but achieved occupancy barely changed (21.89%) and
the isolated batch-1/context-4096 median was 0.0635 ms versus 0.0623 ms for
`64x2` in the 50-round interleaved sweep. The extra loop work offsets its
better scheduling. These measurements do not support switching to `32x2`.

Application-replay traffic and instruction counters at batch 1/context 4096:

| Kernel | DRAM read | L2 read sectors | Counted FP32 add + mul + 2×FMA | FP32 FLOPs / DRAM-read byte |
|---|---:|---:|---:|---:|
| per-head `128x4` | 16.83 MB | 1,049,232 | 23.48 million | 1.40 |
| split-K 8-way `64x2` | 16.82 MB | 894,351 | 22.89 million | 1.36 |

These intensity figures count only the selected FP32 SASS instructions and
divide by DRAM **reads**, not reads plus writes. They exclude exponentials,
integer address arithmetic and any FP16 operations, and therefore are not a
complete FLOP count. Algorithmic QK plus weighted-V work is about 33.55
million FLOPs over the same 16.8 MB read, or about 2 FLOPs/DRAM-read byte.
Split-K did not reduce DRAM traffic; its speedup comes from exposing more
parallelism and using the available memory bandwidth better. Nsight's
multi-pass times are diagnostic; use the interleaved timing sweep for latency.

The split-count sweep (`results/rtx4060/paged_decode_split_scaling_ab.json`)
found `64x2` eight-way best at context 8192, while 16-way was 0.0021 ms
faster at context 4096 in the isolated kernel. The three-repeat full-engine
A/B at mean context 4125 (`ab_decode_split8_vs16_long4k_b1_fp16.json`)
did not confirm that advantage: median decode GPU time was 11.106 ms for
eight-way and 11.155 ms for 16-way; median ITL was 12.275 versus 12.315 ms.
Both differences were inside run-to-run spread. Both arms passed the stock
token gate, though later greedy tokens differed at near ties. Retain eight-way
for the opt-in long-context candidate; do not promote 16-way or the 32x2 tile.

For batch 8/context 256, a separate application-replay counter collection gave:

| Backend | L2 read sectors (32 B each) | L2 read-hit sectors | One-pass DRAM read | Isolated median kernel time |
|---|---:|---:|---:|---:|
| per-head `128x4` | 525,552 | 258,276 | 8.46 MB | 0.0410 ms |
| paired GQA `64x2` | 263,268 | 83 | 8.47 MB | 0.0359 ms |

Paired GQA roughly halves L2 read transactions while DRAM bytes stay the same
in this profiled cell. The per-head kernel's approximately 49% L2 hit fraction
is consistent with its second query head reusing each KV head's data. The paired
kernel removes that duplicate read but carries more registers and fewer active
warps. A fresh 100-repeat CUDA-event sweep measured the paired kernel 1.14x faster
at this cell; `max_abs_diff` was 0.000122. This is a narrow kernel result, not a
claim about end-to-end ITL or other shapes.

Nsight Compute used `--cache-control none`. Its multi-pass kernel replay warned
that cache behavior could vary, and one kernel-replay L2 comparison produced
inconsistent sector totals. The L2 sector comparison above uses application
replay, which redoes the same warmup for each metric pass. DRAM reads were
collected in separate single-pass reports. Treat cross-report cache conclusions
as hypotheses to confirm in a live graph trace or with a repeat run; the
occupancy and register limits are directly measured for the selected launch.

The raw reports from this session are under `/tmp/paged_decode_*.ncu-rep`.
To reproduce the first report:

```bash
pkexec --keep-cwd /usr/local/cuda-13.1/bin/ncu \
  --section MemoryWorkloadAnalysis --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --cache-control none --nvtx --nvtx-include 'profile_target]' \
  --launch-count 1 -o /tmp/paged_decode_b8_ctx1024 \
  .venv/bin/python -m benchmarks.kernels.profile_paged_decode_memory \
  --batch 8 --context 1024
```

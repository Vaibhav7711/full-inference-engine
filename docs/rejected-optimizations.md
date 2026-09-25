# Rejected optimizations and retracted claims

Internal engineering record. Everything here was designed, implemented, measured, and then
**not shipped** — or shipped, then withdrawn when its own evidence was re-examined.

It is kept for three reasons. It stops the same idea from being retried on intuition; it
records *why* a standard technique failed on this hardware, which is usually more specific
than "it was slow"; and it is the honest counterweight to the results that did ship.

Every implementation below is still in the tree, tested, and reachable by name — a rejected
optimization is a measured configuration, not dead code. None is a default on any
architecture where it lost.

---

## 1. Tiled FlashAttention-structured prefill on Turing — 3× slower

A FlashAttention-style chunked-prefill kernel using `tl.dot` ran **3× slower** than the
per-token kernel it was meant to replace. Inspecting the compiled PTX explained it
completely:

```
mma_sync = 0      fma = 2052      registers = 255      spill stores = 128      smem = 49 KB
```

One block per SM, and not a single tensor-core instruction. Triton lowers `tl.dot` to MMA
only from `sm_80`; on Turing it emits scalar FMA. The kernel was competing against a
tensor-core implementation while doing its matmuls on CUDA cores — a gap no tile or warp
tuning can close.

**Disposition:** retained as `prefill_attention="tiled"`, gated behind a PTX check
(`benchmarks/kernels/prefill_attention_ab.py --ptx-only`) and only selectable on `sm_80+`.
On Ada its PTX gate passes (32 `mma.sync`, zero spills) but it has not yet been A/B'd inside
the engine, which makes it the highest-value untested candidate in the repository.

## 2. Sharing K/V reads across a GQA group — 0.95–1.06×, i.e. nothing

Qwen3 has 16 query heads and 8 KV heads, so the per-head decode kernel streams each K/V tile
twice. A variant reading each tile once for both heads of a group measured **0.95–1.06×**
across 16 operating points, with no trend in batch or context. The duplicate read was
already being served by L2.

An earlier version of the same kernel carried the group as a leading tensor axis and
broadcast over `[REP, BLOCK_N, D]`. It lost **1.5–2.8×** at every point, *including* cells
with ample parallelism — so the cost was the rank-3 layout, not the halved grid. The lesson
generalised: Triton's 3-D intermediates are expensive even when small in bytes, and a static
check (`tests/kernels/test_kernel_static_checks.py`) now forbids the pattern.

**Disposition:** retained as `decode_attention="gqa"`, never a default.

## 3. FlashAttention-2 for decode — +113.9% ITL, and the premise was wrong

`flash_attn_with_kvcache` is not CUDA-graph capturable on the tested build (2.8.4): its
split-KV workspace setup synchronises, and passing an explicit `num_splits` did not help.
Selecting it therefore forces the whole decode forward to run eager, and end-to-end ITL p50
rose **113.9%**.

The more interesting result is that the kernel did not win anyway. An eager-vs-eager probe
still showed **+11.9%**, and a 96-cell sweep put FA2 behind the Triton kernel in 13 of 16
cells. Its only wins were at batch 1 with long context — and forcing `num_splits=1` there
made FA2 **1.7× slower** than the Triton kernel, so its entire advantage was split-K rather
than kernel quality.

A later re-analysis corrected that table in a way that strengthens the conclusion. The
sweep's timing loop synchronised every iteration, placing ~28 µs of host launch latency
inside every sample. Subtracting that one constant puts **every DRAM-bound cell at 98–101%
of the memory roofline for both kernels** — a fit too good across a 4× span to be
coincidence. At decode the query length is 1, so there is no score matrix for
FlashAttention's tiling to avoid; both implementations are correct memory-bound kernels and
the memory is the limit. FA2 does not lose on kernel quality. It ties, and it cannot be
captured.

**Disposition:** retained as `decode_attention="flash"`, priority below `per_head`, never a
default on a measured architecture.

## 4. Dense-gather FlashAttention prefill at 16-token pages — +38.5%

FA2's paged API requires page sizes divisible by 256. Using the dense entry point over a
gathered KV tensor, so that the engine's normal 16-token pages could stay, cost **+38.5%
prefill step and +52.6% TTFT**: the gather and materialisation dominate whatever the better
attention kernel saves.

**Disposition:** retained as `prefill_attention="flash_dense"` so the experiment stays
reproducible, priority below `sdpa`.

## 5. INT8 KV cache

Halves KV bytes per token, which should raise the concurrency ceiling. It did not move the
decode step on the T4, and on Ada it additionally drifts through the preemption/recompute
path — marked as an expected failure in the CUDA suite rather than hidden.

**Disposition:** implemented end to end (`kv_cache_dtype="int8"`, with dequantising decode
and per-token prefill kernels), disabled by default.

## 6. Prefix caching on random-prompt workloads — unresolved

The prefix cache (block-aligned radix tree plus exact-match entries, copy-on-write tails
over a refcounted allocator) measured **unresolved** on the soak workload. That workload
generates independent random prompts, so nothing shares a prefix and there is nothing for
the cache to hit.

This is a statement about the measurement, not the feature. A chat workload with a shared
system prompt, or multi-turn traffic, is where it would pay — and it has not been measured
there. Note also that the interaction with page size matters: at 256-token pages a shared
700-token prefix reuses 512 tokens against 688 at 16-token pages, because the cache matches
complete blocks only.

**Disposition:** implemented, off by default (`prefix_cache_blocks=0` in the measured
configurations), awaiting a workload that can show it.

---

## Retracted claims

### A tail regression that was warmup coverage

The fused step's first measurement reported ITL p99 **doubling** on the long profile while
p50 fell 17%. The cause was CUDA graph captures happening *inside* the timed window: warmup
covered gathered-context buckets only to 2,048 tokens while the long profile reaches 4,096,
so roughly a dozen 150 ms captures landed in each run — about 25 affected token gaps in a
run of ~2,700, which is exactly the p99 sample.

After widening warmup coverage, p99 fell **11%** and p999 **12%**. The lesson is now enforced
in code rather than remembered: the engine counts `lazy_graph_captures` after warmup, soak
and A/B runs report it per arm, `check_hooks` fails on any non-zero value, and **a tail
percentile is only a result when both arms report zero**.

### A documented claim contradicted by its own artifact

The RTX 4060 evaluation stated that forcing FA2 decode split counts was "generally worse
than automatic". Re-examining the committed sweep cell by cell showed the opposite where it
mattered: at batch 8 with 2,048- and 4,096-token contexts, forced splitting was **10–12%
better** than automatic.

The mechanism: FA2's heuristic stops splitting once its CTA count approaches the SM count, a
threshold tuned for datacenter GPUs that mis-fires on a 24-SM consumer card (it launched 64
CTAs onto 24 SMs and declined to split). The original claim had generalised from the batch-1
cells, where it was true.

### Measurement debt, disclosed

Every kernel microbenchmark in the tree shares two defects: it times one synchronised launch
per sample, placing ~28 µs of host launch latency inside every measurement, and it reuses a
single KV pool across repeats so small cells stay resident in L2. Conclusions drawn from
cells below roughly 150 µs of kernel time on either card are therefore **not supported**, and
the harnesses need rewriting — a batched loop between one event pair, plus rotating
per-layer pools — before the next kernel decision.

Engine-level A/B results are unaffected: they measure wall-clock over 30-second closed-loop
windows, where a 28 µs offset and cache residency are both irrelevant.

---

## What this record cost, and why it exists

Six rejected optimizations and three retractions is not a sign of poor judgment. Three of
the six — tiled FlashAttention prefill, GQA-shared reads, FlashAttention decode — are
standard practice in other inference engines, and were rejected only because they were
measured *on this hardware, in this engine*. Adopting them on reputation would have produced
a slower engine with a more impressive feature list.

The two retractions came from re-reading evidence that had already been published. Both were
found by auditing our own artifacts rather than by an external failure, which is the only
mechanism that scales.

Full chronology, with commits and dates: [`optimization-journal.md`](optimization-journal.md).
Current validated claims and previously retracted ones:
[`checkpoint.md`](checkpoint.md).

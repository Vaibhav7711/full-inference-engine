# Building a single-GPU LLM serving engine: design, measurement, and results

**Version:** v0.1.0-beta · **Primary platform:** NVIDIA Tesla T4 (`sm_75`, 16 GB) ·
**Models:** Qwen3-0.6B / 1.7B / 4B, FP16 · **Date:** 2026-09-25

---

## Abstract

This report describes an LLM inference engine written from the model downward: continuous
batching over a paged KV cache, chunked prefill fused with decode into a single forward
per scheduler step, CUDA graphs for every forward shape, custom Triton attention and
KV-write kernels, a preemptive FCFS scheduler, per-request sampling, and an
OpenAI-compatible HTTP surface. Hugging Face `transformers` supplies the model definition
and weights; every other component that participates in serving a request was built here.

On a Tesla T4 with Qwen3-0.6B the engine reduced the cost of a prefill-carrying scheduler
step from 98 ms to 20.5 ms, inter-token latency p99 from 295 ms to 33 ms, and
time-to-first-token p50 from 2.8 s to roughly 0.35 s. In a controlled head-to-head against
vLLM on the same GPU, at matched KV capacity and concurrency limits, it sustained
**1.06–1.67× vLLM's batched output throughput** for Qwen3-0.6B, 1.7B and 4B at batch ≥ 4,
while vLLM retained lower single-request time-to-first-token and used less GPU memory.

The engine's more unusual contribution is methodological. Roughly as much effort went into
being able to *trust* a number as into producing one, and the record of what failed is
kept with the same care as the record of what worked. Seven optimizations were built,
measured, and rejected — including two that are standard practice elsewhere. Two
published claims were retracted after their own evidence was re-examined. Section 6 is
devoted to those results, because they carry more information than the wins.

---

## 1. The problem

A decoder-only transformer generates text autoregressively. Each new token requires a
forward pass that attends to the keys and values of every preceding token; those K/V
vectors are cached so they are computed once. This produces two phases with opposite
performance characteristics:

**Prefill** processes the whole prompt in one forward pass. Work scales as
`prompt_tokens × parameters`, and attention is quadratic in prompt length. It is
compute-bound, and it produces the first output token.

**Decode** produces one token per sequence per step. Every step reads *all* of the model's
weights and the sequence's entire KV cache to produce a single token. It is
bandwidth-bound. For Qwen3-0.6B in FP16 the weights are 1.19 GiB; at the T4's measured
258 GB/s that is a hard floor of ~4.6 ms per step, so a single sequence cannot exceed
roughly 200 tokens/s regardless of software quality.

That floor is the central fact of inference serving, and it has one escape: pay it once
for many sequences. **Continuous batching** runs N sequences' single new tokens through
one forward pass, so the weight read is amortised across N tokens. The sequences have
different lengths and arrive and finish at different times, so batch membership changes
every step — hence "continuous" rather than "static".

That requirement is what forces every other design decision in this engine:

- Sequences of differing length cannot share a contiguous KV tensor, so the cache becomes
  a pool of fixed-size pages with a per-sequence page table (**paged KV**). Stock
  attention implementations cannot read that layout, so attention must be written.
- A newly arrived 700-token prompt cannot be prefilled in one step without stalling every
  decoding sequence for ~100 ms, so prompts are admitted in chunks alongside decode
  (**chunked prefill**), which requires a second attention implementation for
  "attend a chunk to its paged prefix".
- At a 10 ms step, ~1,100 kernel launches of Python dispatch per step is a large fraction
  of the budget, so forwards are recorded and replayed (**CUDA graphs**), which requires
  every buffer to have a fixed address and every shape to be known in advance.
- Requests must be admitted, and refused, and preempted, under a fixed KV budget, which
  requires a **scheduler** that owns memory rather than merely ordering work.

---

## 2. Design

### 2.1 Structure

```
HTTP / SSE (FastAPI)  ·  OpenAI-compatible /v1 surface
        │
ContinuousBatchingService — one worker thread owns the engine; handlers never touch it
        │
FCFSScheduler — admission bounded by KV capacity, LIFO preemption, prefill planning
        │
ContinuousBatchingEngine.step()
        │   decode rows (1 token each)  +  prefill chunk rows (≤128 tokens each)
        └── ONE packed forward, replayed from a CUDA graph keyed by shape
                │
                ├── paged decode attention   (Triton, one program per row × query head)
                ├── chunked prefill attention (SDPA over gathered pages, or Triton, or FA2)
                ├── fused RMSNorm / RoPE / SwiGLU (Triton, installed onto the model)
                └── one argmax + one device→host copy for the whole batch
```

KV lives in per-layer pools shaped `[num_blocks, page_size, kv_heads, head_dim]`. Each
request owns a list of physical page ids. For Qwen3-0.6B a token costs
`2 × 28 layers × 8 KV heads × 128 dims × 2 B = 112 KiB` of KV, so the pool — not the
weights — bounds concurrency.

### 2.2 The scheduler step

One `step()` does the following, in order:

1. Collect rows in `DECODING` and acquire each one's next KV slot in FCFS priority order.
   A row that cannot grow preempts a *newer* row (strict LIFO victim selection, so the
   oldest request always completes) and is failed only when nothing newer exists to
   evict.
2. Admit waiting requests whose prompt fits the remaining KV capacity.
3. Plan prefill: round-robin one chunk per prefilling request under a per-step token
   budget.
4. Run **one packed forward** over the decode tokens and the chunk tokens together.
5. Sample one token per row, commit, and finish or preempt.

Step 4 is the fused step. Before it existed, a prefill-carrying step ran two forwards —
decode (10.3 ms) then the chunk batch (~17 ms) — reading every weight twice and
synchronising twice. Packing both into one token row means the per-token modules (norms,
projections, MLP) run once over the union; only attention needs to know which token
belongs to which request, and it cuts the packed row apart to hand each slice to the
right kernel.

### 2.3 Preemption and recompute

When the pool cannot satisfy a request, a newer request yields all of its KV pages and
returns to the queue; its generated tokens are preserved, and it is later re-prefilled
over prompt-plus-generated-prefix and resumes exactly where it stopped. Recompute cost is
measured per request rather than assumed: rebuilding is 37.5 ms and is **independent of
token count** (32 and 22 tokens both measured 37.53 ms), because it is dominated by the
forward rather than by the tokens replayed. Queue wait after yielding is 131–664 ms,
dominating rebuild by 3.5–17×, which is why the scheduler's admission policy matters more
than its recompute efficiency.

### 2.4 Pluggable attention backends

Attention implementations are registered, not hard-coded. A backend declares a name, a
phase, a `run`, and an `available(profile, geometry)` that returns *the reason it cannot
run here* or `None`:

| phase | backends |
|---|---|
| decode | `per_head` (Triton, the measured baseline) · `split_k` (FlashDecoding structure) · `gqa` (shared group read) · `flash` (FA2 paged, sm_80+) |
| prefill | `sdpa` (gather pages + torch SDPA) · `per_token` (Triton, INT8-capable) · `tiled` (Triton `tl.dot`, sm_80+) · `flash` (FA2 paged) · `flash_dense` |

`"auto"` selects the highest-priority available backend; a **named** backend that cannot
run raises with the reason rather than falling back silently. That last property is not
fastidiousness — a silent fallback is precisely how one measurement campaign timed an
eager forward for a week and reported it as a CUDA-graph result (§6.6).

Per-architecture defaults live in `engine/backends/policy.py`, each with the A/B that
chose it. An architecture not in that table receives capability-led defaults whose reason
strings all contain the word `unmeasured`, plus the command that would settle them.

### 2.5 Model portability

Fused kernels find their targets structurally — any module exposing
`gate_proj`/`up_proj`/`down_proj`, any class whose name ends in `RMSNorm` with a 1-D
weight — and RoPE is patched in the model's own modeling module, located from
`type(model).__module__`. A Llama-style checkpoint therefore receives the same fused
kernels with no new code. What *is* declared explicitly is what the paged kernels cannot
serve: `head_dim > 128`, non-divisible GQA, sliding-window attention, mixture-of-experts,
and multi-head latent attention are refused **at load, with the reason**. Serving a
windowed model on kernels that attend to the full prefix would produce a plausible wrong
answer; refusing is the only correct behaviour.

---

## 3. Method

The engine's measurements are only as good as the protocol that produced them, and early
in the project several were not. What follows is the protocol the recorded results use.

### 3.1 Interleaved A/B on a closed-loop workload

Arms are compared by alternating them — A B A B A — over five 30-second closed-loop runs
at concurrency 8, on freshly constructed engines, each warmed before its timed window. A
fresh engine per run matters: a reused pool carries prefix-cache entries and fragmentation
from the previous run, which is exactly the state a comparison is trying to hold constant.

### 3.2 Spread, and the word "unresolved"

Every metric is reported as a median across runs with `spread = (max − min) / median`. A
change whose magnitude falls inside the spread is labelled **unresolved** and is not
treated as a result. This single rule retired more claims than any other part of the
method.

It also has a failure mode worth naming: because spread is estimated as the sample *range*,
its expectation scales with the number of runs (the control-chart constant d₂ is 1.128 at
n=2, 1.693 at n=3, 2.326 at n=5). A three-run experiment therefore reports only ~73% of
the spread a five-run experiment would, and the gate becomes correspondingly more
permissive. Short-repeat results are flagged wherever they appear (§5.3).

### 3.3 Token identity against stock Transformers

Before anything is timed, every arm generates 48 tokens greedily on four fixed prompts and
is compared against unpatched Hugging Face Transformers running SDPA with a `DynamicCache`.
A kernel swap that changes output is a bug, not a speedup.

This gate needed one correction. Three different chunked-prefill implementations diverged
from stock at the *same three token positions*, which is the signature of a near-tie rather
than of a wrong kernel. Measuring the stock top-2 logit margin at those positions gave
0.0078, 0.0156 and 0.0156 — **one fp16 ulp each**, while the smallest margin on a prompt
where nothing ever diverged was 0.14. The gate now measures the margin and does not count
a first difference on a tied position as a divergence (`TIE_MARGIN = 0.02`). The rule
"early divergence means a wrong kernel" stands for real margins.

### 3.4 Per-phase step decomposition

A wall-clock timer around `step()` blends host staging, H2D copies, the forward, and the
sampling synchronisation, so it cannot say which of them a change moved. With
instrumentation enabled the engine reports `host_stage_ms`, `decode_gpu_ms`,
`prefill_gpu_ms`, `fused_gpu_ms` and `sync_ms` per step. Several results in §4 are
attributable only because of this split.

### 3.5 In-service graph captures are counted

A CUDA graph capture is two eager forwards plus device synchronisations — 100–200 ms. If
one happens inside a timed window it lands squarely in the tail percentiles. The engine
counts captures occurring after warmup (`lazy_graph_captures`); soak and A/B runs report
it, and `check_hooks` fails on any non-zero value. **A tail percentile compared across arms
is only a result when both arms report zero.** This counter caught two "regressions" that
were entirely warmup-coverage artifacts.

### 3.6 Provenance

Every result JSON records the git commit, working-tree cleanliness, GPU identity, driver
and library versions, and SM/memory clocks before and after the run. A re-measurement on a
drifted stack must say so next to its numbers rather than hide it.

---

## 4. Results on the Tesla T4

Qwen3-0.6B, FP16, chat profile (~656-token prompts), concurrency 8, five 30-second
interleaved runs per arm. Full artifacts: `results/t4/`, narrative in
`docs/optimization-journal.md`.

### 4.1 The arc

| | first measurement | final |
|---|---:|---:|
| prefill-carrying step p50 | 98 ms | **20.5 ms** |
| decode-only step p50 (batch ~5) | 9.7 ms | 10.2 ms |
| ITL p50 | ~25 ms | **19.8 ms** |
| ITL p99 | 295 ms | **33.0 ms** |
| ITL p999 | — | 51.4 ms |
| TTFT p50 | 2.8 s | **0.32–0.41 s** |
| host staging per step | 3.4 ms | **0.26 ms** |

Long profile (~1.8k-token prompts): prefill step 23.1 ms, ITL p50 22.4 ms, p99 43.9 ms,
p999 60.2 ms.

Note what did *not* improve: the decode-only step. §7 explains why that is the correct
outcome rather than a failure.

### 4.2 What each change bought, in isolation

| change | measured effect |
|---|---|
| Continuous batching vs sequential decoding | **6.5×** aggregate throughput (24 → 154 tok/s at concurrency 16) |
| CUDA graphs on the decode forward | 34.2 → 9.7 ms/step (**−72%**) |
| Decode metadata staging by slice-copy | 3.4 → 0.18 ms/step of host time |
| SDPA-over-pages chunked prefill vs per-token Triton | 45.3 → 27.0 ms chat (**−40.6%**), 87.9 → 30.3 ms long (**−65.6%**) |
| CUDA graphs on the prefill forward | prefill step **−56.7%**, ITL p50 **−57.4%** |
| Fused decode+prefill step | prefill step **−17.6%** chat / −13.9% long; ITL p50 **−18%**; p99 −14% / −11% |
| Warmup before serving | ITL p99 **−71%**, p999 −63%, p50 unchanged |

The warmup row deserves comment because it is the shape of a correct result: p50 and every
step cost were unchanged (all within spread), and only the tail moved. Warmup does not make
anything faster; it moves first-use costs — Triton JIT, graph capture — out of the serving
window. That is exactly what the numbers say.

### 4.3 Why the fused step works, and how much

The fused forward's cost is bounded below by the larger of the two forwards it replaces,
not their sum: the weight read is shared and the chunk tokens' compute is unchanged. The
prediction was 27 ms → 17–20 ms; the measurement was 20.5 ms. Two reported metrics
*worsened* and both are accounting rather than regression: `host_stage_ms` rose 66–92%
because one phase now stages both buffer sets, and `sync_ms` rose ~100% because the fused
step's single synchronisation waits for the whole forward whereas the two-forward arm's
`sync_ms` only ever timed the decode wait. Both sit inside the step time that fell.

---

## 5. Head-to-head against vLLM

### 5.1 Protocol

Colab Tesla T4, both engines in separate processes on an otherwise idle GPU. Controls:

- identical **pre-tokenized prompt token ids** handed to both engines, eliminating
  tokenizer differences
- identical KV capacity: 16,384 tokens, set on vLLM with `kv_cache_memory_bytes` and on
  this engine with `num_blocks × block_size`
- identical concurrency cap (`max_num_seqs` = `max_active` = 16), FP16 in both, prefix
  caching disabled in both, greedy with `ignore_eos` in both
- identical black-box measurement: wall time around a full generation call, 2 warmup + 5
  timed runs, median; NVML device-level memory for both
- CUDA graphs left enabled in both

### 5.2 Results

Batched output throughput, 256-token prompts, 128 generated tokens:

| model | batch | this engine | vLLM | ratio |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1 | 131 tok/s | 129 | 0.99 |
| Qwen3-0.6B | 4 | 453 | 362 | **1.25×** |
| Qwen3-0.6B | 8 | 750 | 513 | **1.46×** |
| Qwen3-0.6B | 16 | **1,069** | 639 | **1.67×** |
| Qwen3-1.7B | 16 | **514** | 412 | **1.25×** |
| Qwen3-4B | 16 | **222** | 209 | **1.06×** |

Single-request latency (batch 1, 32-token prompt): vLLM's TTFT is lower — 19.7 ms against
47.8 ms — and decode rate is within noise of parity.

Memory, both at 16,384 KV tokens, Qwen3-0.6B: 7,291 MiB resident for this engine against
vLLM's 3,259 MiB.

### 5.3 Interpretation

The advantage **grows with batch and shrinks with model size**: 1.67× at 0.6B, 1.25× at
1.7B, 1.06× at 4B. That is the signature of a per-step overhead advantage. The larger the
model, the more each step is dominated by GPU work and the less a fully-graphed, fused step
can buy. A measurement artifact would not track model size that cleanly, which is the main
reason to believe the result.

The converse is equally real: vLLM wins single-request TTFT, because its per-request entry
path is cheaper than this engine's, and it uses less memory.

Two caveats belong next to these numbers. First, vLLM was not given
`detokenize=False`, so it pays incremental text decoding inside the timed region while
this engine returns raw token ids — a bias *against* vLLM, small (tens of ms per 2,048
tokens) but real. Second, the comparison is **at equal KV budget**, not at each engine's
best configuration; vLLM normally sizes its cache to ~90% of the card. Equal budget is the
right controlled choice, but it is not the same claim as "faster in production".

### 5.4 The memory result, and what caused it

The 4 GB memory gap was not the KV pool — that was matched, and vLLM preallocates the same
1,792 MiB. Every other preallocation in this engine is trivial (the shared logits buffer is
9.7 MB; staging buffers are under 1 MB). A configuration probe isolated the cause exactly:

| configuration | torch reserved after warmup |
|---|---:|
| graph buckets 1,2,4,8,16 + prefill graphs | **5.09 GiB** |
| same buckets, prefill graphs disabled | **3.06 GiB** |
| buckets 1,4,16 + prefill graphs | 5.04 GiB |
| no graphs at all | 2.91 GiB |

Prefill and fused graph capture cost **2.03 GiB**, and decode buckets were irrelevant
(10 → 6 graphs moved it by 0.05 GiB). The mechanism: warmup enumerated row buckets and
context buckets independently and captured their whole product — including **16 rows at a
16,384-token context**, which requires 262,144 tokens of prefix against a 16,384-token
pool, sixteen times more KV than exists. The SDPA path gathers K and V for that shape
inside the capture, and the graph memory pool is sized by its largest capture.

Captures are now skipped when `rows × context` exceeds the pool. At the benchmarked
configuration the largest single gather falls from **1,024 MiB to 64 MiB** and graph count
from 255 to 215. Separately, padded graph rows reserved one whole KV page each — invisible
at 16-token pages, 10.9% of the pool at the 256-token pages FlashAttention requires — and
now share pages with one slot each, reducing that reservation from 1,792 tokens to 256.

> The end-to-end resident-memory figure after these fixes is not yet in
> `results/`; it needs one re-run of the probe on a GPU to publish, and the table above is
> the pre-fix measurement.

---

## 6. Negative results

This section exists because the project's credibility rests on it. Every item here was
built, measured, and rejected or retracted.

### 6.1 Tiled FlashAttention-structured prefill on Turing — 3× slower, and diagnosable without a benchmark

A FlashAttention-style prefill kernel with `tl.dot` ran 3× *slower* than the kernel it was
meant to replace. Inspecting the compiled PTX explained it completely: **`mma_sync = 0`,
`fma = 2052`, 255 registers, 128 spill stores, 49 KB shared memory** — one block per SM.
Triton lowers `tl.dot` to MMA instructions only from `sm_80`; on Turing it emits scalar
FMA. The kernel was competing against a tensor-core implementation while doing its matmuls
on CUDA cores. No tile or warp tuning can close that gap. The kernel is retained, gated
behind a PTX check, as a candidate for `sm_80+`.

### 6.2 Sharing K/V reads across a GQA group — 0.95–1.06×, i.e. nothing

Qwen3 has 16 query heads and 8 KV heads, so the per-head decode kernel streams each K/V
tile twice. A variant reading each tile once for both heads of a group measured
**0.95–1.06×** across 16 operating points, with no trend in batch or context. The
duplicate read was already being served by L2. Closed.

An earlier version of the same kernel carried the group as a leading tensor axis and
broadcast over `[REP, BLOCK_N, D]`; it lost **1.5–2.8×** at every point, including cells
with ample parallelism. The lesson was about Triton layouts, not about the idea: rank-3
intermediates are expensive even when they are small in bytes. A static check now forbids
the pattern.

### 6.3 FlashAttention-2 for decode on Ada — +113.9% ITL, and the premise was wrong

FA2's `flash_attn_with_kvcache` is not CUDA-graph capturable on the tested build (its
split-KV workspace setup synchronises; passing an explicit `num_splits` did not help), so
selecting it forces the entire decode forward to run eager. End-to-end ITL p50 rose
**113.9%**.

The more interesting finding is that the kernel did not win *anyway*. An eager-vs-eager
probe still showed **+11.9%**, and a 96-cell kernel sweep put FA2 behind the Triton kernel
in 13 of 16 cells. Its only wins were at batch 1 with long context — and forcing
`num_splits=1` there made FA2 **1.7× slower** than the Triton kernel, proving its entire
advantage was split-K rather than kernel quality.

Later analysis corrected this table in an important way. The sweep's timing loop
synchronised every iteration, placing ~28 µs of host launch latency inside every sample.
Subtracting that single constant puts **every DRAM-bound cell at 98–101% of the memory
roofline for both kernels** — the fit is too good across a 4× span to be coincidence. The
honest conclusion is therefore stronger and simpler than the original: at decode, where the
query length is 1 and there is no score matrix to avoid, *both implementations are perfect
memory-bound kernels and the memory is the limit*. FA2 does not lose; it ties, and cannot
be captured.

### 6.4 Dense-gather FlashAttention prefill at small pages — +38.5%

FA2's paged API requires page sizes divisible by 256. Using the dense entry point over a
gathered KV tensor at the engine's normal 16-token pages cost **+38.5% prefill step and
+52.6% TTFT**: the gather and materialisation dominate whatever the better attention kernel
saves.

### 6.5 INT8 KV cache, and prefix caching on random prompts

INT8 KV halves KV bytes and did not move the decode step on the T4; on Ada it additionally
drifts through the preemption/recompute path and is marked as an expected failure. Prefix
caching is implemented (radix plus exact entries, copy-on-write tails) and measured
unresolved on the soak's random-prompt workload — which cannot show it, since nothing
shares a prefix. Both are off by default. The prefix-cache result is a statement about the
workload, not about the feature.

### 6.6 Two retractions

**A tail "regression" that was warmup coverage.** The fused step's first measurement showed
ITL p99 doubling on the long profile. It was graph captures happening inside the timed
window, because warmup covered context buckets only to 2,048 while the long profile reaches
4,096. After widening coverage, p99 fell 11% and p999 12%. The lesson is now enforced in
code: `lazy_graph_captures` must be zero in both arms before any tail percentile counts.

**A documented claim contradicted by its own artifact.** The RTX evaluation stated that
forcing FA2 decode split counts was "generally worse than automatic". Re-examining the
committed sweep cell by cell showed that at batch 8 with 2,048- and 4,096-token contexts,
forced splitting was **10–12% better** than automatic, because FA2's heuristic declines to
split once its CTA count approaches the SM count — a threshold tuned for datacenter GPUs
that mis-fires on a 24-SM consumer card. The claim had generalised from the batch-1 cells.

### 6.7 What the negative results cost, and why they are kept

Seven rejected optimizations is not a sign of poor judgment; three of them (tiled prefill,
GQA-shared reads, FA2 decode) are standard practice in other engines and were rejected only
because they were measured *here*. The alternative — adopting them on reputation — would
have produced a slower engine with a more impressive feature list.

---

## 7. Where the remaining time goes

A quantitative budget for one decode step, Qwen3-0.6B on the T4 at batch ~5:

| component | cost | at its limit? |
|---|---:|---|
| weight read (1.19 GiB at 258 GB/s) | ~4.6 ms | **yes** — this is the floor |
| paged decode attention | ~4 ms (0.15 ms × 28 layers) | **yes** — 98–101% of the memory roofline |
| host staging | 0.26 ms | effectively |
| everything else (sampling sync, per-layer overhead inside the graph) | ~1.3 ms | unmeasured |
| **measured total** | **10.2 ms** | |

Two conclusions follow, and they reorder the roadmap:

**Attention is done.** At batch ≥ 4 the decode attention kernel operates at the DRAM
roofline. Split-K, GQA-shared reads, and further attention kernel work cannot help there;
you cannot beat the bandwidth limit. Split-K's remaining value is confined to batch 1–2,
where the grid (16 programs on 40 SMs) is too narrow to saturate DRAM — the single-user
interactive case, and worth having for that reason alone.

**The lever is fewer weight bytes.** The floor is a *byte count*, not a bandwidth problem.
W8A16 halves it; W4A16 quarters it. Both kernels exist in the repository and are
benchmarked in isolation; neither is on the serving path. That is the highest-value next
piece of work, and it is a larger change than any kernel tuning.

---

## 8. Limitations and threats to validity

**Protocol asymmetry between the two GPUs.** The T4 results use five 30-second runs at
concurrency 8. The RTX 4060 results, which evaluated FlashAttention, used **1–3 repeats of
4–5 seconds at concurrency 4**, and per §3.2 that under-reports spread by roughly 27–52%.
The Ada FlashAttention-prefill conclusion (−7.8% prefill step, −13.7% TTFT on 1.7B) should
be treated as preliminary until re-run at the T4 protocol. This report's headline claims
are T4 claims for that reason.

**One model family.** Performance is validated for Qwen3 only. Llama, Mistral and Gemma
checkpoints load through the same structural hooks and are geometry-checked at load, but
none has been measured or token-gated. "Supports Llama-style models" is a structural claim,
not a performance claim.

**One node, one GPU.** No tensor or pipeline parallelism, no multi-node, no disaggregated
prefill/decode.

**Unvalidated subsystems.** Greedy speculative decoding has an experimental, default-off
engine-native path whose speedup claim is explicitly gated on Kaggle T4×2 results that have
not yet run. INT8 KV is disabled. Quantized weights are benchmarked but unintegrated. The
`split_k`, `tiled` and `flash` backends are registered and correctness-tested but are not
defaults on any measured architecture.

**Measurement debt.** Every kernel microbenchmark in the tree shares the timing pattern
described in §6.3 — one synchronised launch per sample, and a single KV pool reused across
repeats so small cells stay L2-resident. Conclusions drawn from cells below roughly 150 µs
of kernel time on either card are not supported, and the harnesses need rewriting before
the next kernel decision. The engine-level A/B results are unaffected: they measure wall
time over 30-second closed-loop windows.

**CI covers CPU only.** 314 tests run without a GPU on every push; the 193 CUDA tests are
run by hand and their results recorded in `docs/` and `results/`.

---

## 9. Reproducing the results

```bash
pip install -e ".[dev,server]"
python -m pytest -q                       # 314 CPU tests, no GPU or Triton required
python -m pytest -q -m cuda                # 193 GPU correctness gates
python scripts/check_hooks.py --backends-only    # what this GPU can run, and why not
python scripts/verify_hooks.py --dtype float16   # every seam, as a pass/fail matrix
python examples/library_usage.py --model Qwen/Qwen3-1.7B
```

The measurement commands behind §4:

```bash
python -m benchmarks.reliability.ab --setting cuda_graphs   --prompt-profile chat --repeats 5 --duration 30
python -m benchmarks.reliability.ab --setting prefill_sdpa  --prompt-profile chat --cuda-graphs --repeats 5 --duration 30
python -m benchmarks.reliability.ab --setting prefill_graphs --prompt-profile chat --cuda-graphs --repeats 5 --duration 30
python -m benchmarks.reliability.ab --setting fused_step    --prompt-profile long --cuda-graphs --repeats 5 --duration 30
python -m benchmarks.reliability.ab --setting warmup        --prompt-profile chat --cuda-graphs --repeats 5
```

Every run writes a JSON carrying its own git commit, GPU identity, library versions and
clocks. `docs/optimization-journal.md` records each result, dated, including the ones that
were later retracted.

---

## 10. Future work, in priority order

1. **Weight-only quantization on the serving path** (W8A16, then W4A16). The decode floor
   is a byte count; this is the only lever that moves it. Kernels exist and are benchmarked
   in isolation.
2. **Rewrite the kernel timing harnesses** (batched loop between one event pair, rotating
   KV pools) and re-derive the decode tile/warp regime per architecture. The current regime
   is a T4 result reused unchanged on Ada.
3. **Re-run the Ada FlashAttention conclusions at the T4 protocol**, and A/B the `tiled`
   Triton prefill kernel on `sm_89`, where its PTX gate passes and it has never been
   measured inside the engine.
4. **Split-K decode for batch 1–2**, after fixing graph capture to carry a per-graph
   context bound (capture currently forces `splits=1`, so the existing arm compares the
   baseline against itself).
5. **Complete the speculative decoding evaluation** on T4×2, or remove the path.
6. **A second model family end to end** — one non-Qwen checkpoint through
   `verify_hooks.py` converts a structural claim into evidence.
7. **Piecewise CUDA graph capture** (attention outside the graph), which would make any
   attention backend usable under graphs and remove the context dimension from the graph
   key — the structural fix for both the FA2 capture problem and the graph memory cost.

---

## Appendix: document map

| document | contents |
|---|---|
| `README.md` | what it is, install, quick start, measured status |
| `docs/user-guide.md` | configuration reference, serving, tuning, troubleshooting |
| `docs/architecture.md` | every file, all execution flows, each kernel explained |
| `docs/optimization-journal.md` | every result and retraction, dated, with commits |
| `docs/checkpoint.md` | current validated claims, and retracted claims |
| `docs/rtx4060-final-evaluation.md` | the Ada FlashAttention campaign in full |
| `docs/design-decisions.md` | why components are shaped as they are |
| `docs/t4-speculative-decoding-plan.md` | the pending speculative evaluation |
| `results/t4/`, `results/rtx4060/` | measurement artifacts cited above |

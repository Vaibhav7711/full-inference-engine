# Experiment plan — prefill path (delivery item 2)

Pre-registered. Predictions are written before the runs so a result cannot be rationalised
after the fact, and the decision rule is fixed before the data arrives.

**Why this document exists.** The prefill investigation so far spent one GPU session per
question, learning the next question from each result, and three of those sessions produced
nothing usable: a CUDA-graph comparison whose p999 arm ran identical un-graphed prefill on
both sides, a chunk 512-vs-128 comparison where both arms fit every prompt in one chunk,
and a first chunk A/B whose regression was invisible to the statistic being read. Each was
a design fault, visible in advance. This plan fixes the design first and runs the matrix
once.

---

## What is established

| fact | value |
|---|---|
| decode-only step | 8.05-8.20 ms, 1.64x the memory floor |
| prefill-carrying step | 42.4-44.7 ms at ~122-token prompts |
| prefill share of steps | 22.6% at ~122-token prompts |
| prefill interruption | 8.24 ms/token, ~50% of user-visible ITL |
| prefill cost vs chunk size | flat between 32 and 128 tokens (-2.4%, unresolved) |
| prefill vs CUDA graphs | excess over a decode step is 36.5 ms with graphs, 35.9 ms without — prefill is entirely ungraphed |

**Everything above was measured at ~122-token prompts.** Real chat traffic is 500-4000.

---

## The model being tested

A prefill-carrying step is assumed to cost:

```
step_ms  =  a  +  b * chunk_tokens
```

- **`a`** — fixed per-invocation cost: kernel launches across 28 layers, metadata staging,
  the decode work the step also performs, one weight sweep. Independent of chunk size.
  This is what capturing the prefill path in CUDA graphs would attack.
- **`b`** — marginal cost per prefill token: the attention and MLP work over the chunk.
  This is what the kernel determines, and where the per-token-GEMV chunk kernel flagged in
  the code review would show up.

Rather than a binary launch-bound/compute-bound question, **fit the line across four chunk
sizes and read off both coefficients.** The two failure modes of earlier runs — a treatment
that does not bind, and a statistic that cannot see the effect — are both avoided by
sweeping a parameter that provably binds and reading the regression rather than a verdict.

### Reference values for `b`

At the T4's measured 258.8 GB/s and roughly 65 TFLOPS fp16:

| quantity | per prefill token |
|---|---|
| dense compute floor (2 x 0.6e9 FLOP) | ~0.018 ms |
| KV write | negligible |
| attention, quadratic in position | rises with prompt length |

So a `b` near 0.018 ms/token means the prefill kernel is near the compute roofline; a `b`
several times that means the kernel is the problem.

---

## Phase A — instrumentation (no GPU, built before any run)

| item | status |
|---|---|
| A1 prefill path attribution: `prefill_sdpa_calls` / `prefill_chunked_calls` and token counts, so a cost can be attributed to an implementation | done |
| A2 prompt-length profiles `short` / `chat` / `long` (~122 / ~656 / ~1824 tokens) | done |
| A3 pre-flight `binding_check`: refuse an A/B whose arms need the same chunk count | done |
| A4 `expected_gap_ms` leading the comparison, since a median cannot see a change in step-kind mix | done |
| A5 single-session sweep runner that executes the whole matrix and fits `a` and `b` | this patch |

## Phase B — one GPU session, whole matrix

Run with `python -m benchmarks.reliability.sweep`. Estimated 25-35 minutes.

| run | varies | holds fixed | answers |
|---|---|---|---|
| **B1** chunk sweep | chunk = 64, 128, 256, 512 | `long` profile (~1824 tokens), concurrency 8, graphs on | fit `a` and `b` |
| **B2** profile sweep | profile = short, chat, long | chunk 128, concurrency 8, graphs on | is the 22.6% / 8.24 ms picture representative? |
| **B3** budget vs chunk | `max_prefill_tokens_per_iteration` = chunk vs 4x chunk | `chat` profile | does batching several chunks into one step amortise `a`? |

B1 binds by construction: at ~1824 tokens, chunk 64 needs 29 chunks and chunk 512 needs 4.
B3 is the cheapest possible test of the launch-overhead hypothesis — if `a` is fixed per
*step*, packing more chunks into a step should divide it.

## Phase C — decision rule, fixed in advance

Read `a` and `b` from B1 and apply, in order:

1. **`a` > 20 ms and `b` < 0.05 ms/token** → prefill is launch-bound. Fix: pad prefill to a
   small set of bucketed chunk shapes and capture them in CUDA graphs, exactly as decode
   is. A fixed chunk size already produces fixed shapes, so this is tractable. Expected
   gain by analogy with decode: graphs removed 27.8 of 36 ms there.
2. **`b` > 0.1 ms/token (over 5x the compute floor)** → the chunk kernel is the problem.
   Fix: replace the per-token GEMV with a tiled `tl.dot` causal prefill over paged KV.
   This is the item already flagged in the code review.
3. **Both** → do (1) first: it is a smaller change, and its gain is estimable from the
   decode precedent.
4. **Neither** (`a` small and `b` near floor) → the 42 ms is somewhere unexamined, and the
   next step is a profiler, not a design change.

If B3 shows `a` divides when several chunks share a step, that is independent evidence for
(1) and the fix may be partly a scheduling change after all.

## Phase D — verify, then re-establish

- **D1** implement whichever branch C selects.
- **D2** re-run B1 and check against the prediction recorded in Phase C. A gain outside the
  predicted range is reported as such, not quietly accepted.
- **D3** re-run B2 on `chat` to re-establish the prefill share and interruption at a
  realistic prompt length. Every number in "what is established" above carries the
  ~122-token caveat until this is done.
- **D4** journal, update `docs/checkpoint.md`, commit `results/*.json`.

---

## Predictions, recorded before the runs

Written so the result is interpretable either way.

1. **B1 will show `a` between 25 and 35 ms.** The prefill step's excess over a decode step
   is ~36 ms and is identical with and without CUDA graphs, which points to per-invocation
   cost rather than work.
2. **B1 will show `b` between 0.02 and 0.10 ms/token.** Flat cost from 32 to 128 tokens
   bounds `b` below roughly 0.03 ms/token in that range; the prefill kernel is unlikely to
   be at the compute floor, so somewhat above it is expected.
3. **B2 will show the prefill share rising with prompt length** — roughly 0.20 at `short`,
   0.45-0.60 at `chat`, above 0.75 at `long` — because chunks per prompt scale with prompt
   length while decode steps per request do not.
4. **B3 will show `a` partly divisible.** If four chunks share a step, the per-chunk fixed
   cost should fall, though not to a quarter, since some of `a` is per-request metadata.
5. **The interruption will be much worse than 8.24 ms/token on `chat`.** A 2048-token
   prompt needs 16 chunks at budget 128; if `a` holds, that is ~670 ms of prefill per
   request.

Prediction 3 is the one most likely to be wrong, because prefill share also depends on
output length, which the profiles do not vary.

---

## What this plan deliberately does not do

- No optimisation before Phase C. Three sessions were spent on questions whose answers
  could not change what gets built.
- No conclusions from `short` prompts. They sit below the compute crossover, estimated near
  1000-2000 tokens, so they cannot distinguish the two hypotheses.
- No new A/B settings without a binding check.

# Kaggle T4 x2 speculative decoding: correctness-first integration and test plan

- Status: experimental implementation complete; Kaggle GPU gates pending
- Target: one Kaggle T4 x2 notebook (two separate `sm_75` GPUs, 16 GB each), FP16
- Repository baseline: `d543a5e` (`release-v0.1.0`)
- Primary target: `Qwen/Qwen3-4B` on physical GPU 0
- Primary model drafter: `Qwen/Qwen3-0.6B` on physical GPU 1
- Pair-screening challenger: `Qwen/Qwen3-1.7B` on physical GPU 1
- First proposer: prompt n-gram (no second model)

Implementation checkpoint (2026-09-24): the repository now contains the pure acceptance
planner, n-gram proposer, per-request rollback-capable draft proposer, eager paged-KV
target verification, explicit two-device loading, counters, CPU tests, pair screen,
selection gate, live-engine A/B, preflight, and Kaggle notebook. The target uses the full
production engine. The first draft integration deliberately uses Hugging Face
`DynamicCache` on GPU 1; a paged/graph draft runtime, speculative verification graphs,
fault-injection soak, and any speedup claim remain gated on actual Kaggle T4 x2 results.

## 1. Decision and scope

The experiment has two stages, in this order:

1. **Engine-native n-gram speculation.** Add speculative verification to the live
   `ContinuousBatchingEngine`, using prompt/output n-gram lookup for proposals. This is
   the clean control: it exercises the scheduler, paged KV, Triton kernels, chunked
   prefill, continuous batching and CUDA graphs without paying for a second model.
2. **Engine-native draft-model speculation.** Run Qwen3-4B as the target on GPU 0 and
   screen Qwen3-0.6B versus Qwen3-1.7B as drafters on GPU 1. The 0.6B model is the
   expected winner because draft cost matters at every proposed token; the 1.7B model is
   retained as an acceptance-rate challenger. Integrate only the winner, and keep it
   only if it beats ordinary and n-gram decoding in paired Kaggle runs.

The parked implementations in `engine/speculative/vanilla.py`,
`engine/speculative/optimized.py`, and `engine/batching/batched_speculative.py` remain
algorithmic references. They are not the final benchmark path because they use Hugging
Face `DynamicCache` and padded forwards rather than the production scheduler, paged KV,
custom attention, or production CUDA graphs.

The initial product boundary is **greedy decoding only**. Requests using temperature,
top-k/top-p/min-p, penalties, or logprobs take the existing non-speculative path. Exact
sampling needs rejection sampling over target/draft distributions and careful per-request
RNG accounting; silently treating sampled requests as greedy is forbidden.

No plan can make GPU/runtime failure impossible. This plan instead makes every known
failure detectable, recoverable, and non-corrupting: speculative work is not committed
until verified; unsupported requests fall back to the existing path; and the feature
ships disabled unless all correctness, reliability, memory, and performance gates pass.

## 2. Why this order is correct for this engine and GPU

Speculative decoding works when verifying several proposed tokens with the target costs
less than generating those tokens serially, and when proposal cost is small enough. The
original algorithms preserve the target distribution through verification and rejection
sampling [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) and
[Chen et al.](https://arxiv.org/abs/2302.01318).

That premise is not automatically true here:

- A T4 has 16 GB physical GDDR6 and 300 GB/s specified bandwidth
  ([NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf));
  Kaggle documents T4 x2 as two T4s with 16 GB each, not a unified 32 GB device
  ([Kaggle notebook specifications](https://www.kaggle.com/docs/notebooks)). This
  repository already measures T4 decode as memory-bandwidth-bound.
- Qwen3-4B is the largest unquantized Qwen3 target that is practical here. Its checkpoint
  is about 8.06 GB and its supported geometry is 36 layers, 32 query heads, 8 KV heads,
  and head dimension 128
  ([official files/config](https://huggingface.co/Qwen/Qwen3-4B/tree/main)). Qwen3-8B is
  about 16.4 GB before KV, activations, or graphs, so it is excluded until this engine
  has a validated weight-quantized path.
- Qwen3-4B FP16 KV costs `36 * 2 * 8 * 128 * 2 = 147,456 bytes` (144 KiB) per cached
  token. A 1,024-block, 16-token target pool costs 2.25 GiB. This leaves useful graph and
  activation headroom on GPU 0; the default 4,096-block pool does not.
- Qwen3-0.6B and 1.7B are same-family draft candidates with matching vocabulary geometry.
  The 1.7B checkpoint is about 4.08 GB, while 0.6B is much cheaper. Same-family still
  does not guarantee high acceptance: exact token-map and prompt-template fingerprints
  remain construction gates
  ([1.7B files](https://huggingface.co/Qwen/Qwen3-1.7B/tree/main)).
- The current optimized prototype itself records that the 0.6B drafter did not beat the
  target-only baseline on the measured T4. That is a hypothesis to retest properly, not
  a result to hide.
- External measurements are contradictory and hardware/implementation-specific. One
  Qwen3-4B/0.6B RTX 3070 report measured 43% draft acceptance and 1.7x speedup, while a
  vLLM H100 result reported 0.94x for the same ordinary draft pair
  ([RTX 3070 report](https://pypi.org/project/sequential-speculative-decoding/0.1.1/),
  [H100 BFCL report](https://gist.github.com/jmamou/a84399725d810fcbbe18c769a5d5ff62)).
  vLLM's current docs nevertheless use Qwen3-4B with Qwen3-0.6B as a draft-model example
  ([vLLM draft-model documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/draft_model.md)).
  These are priors only; neither substitutes for paired measurements in this engine.
- Modern serving systems apply speculation selectively at low/medium load, and vary or
  disable it as batch work grows. vLLM documents that `batch_size * speculative_tokens`
  eventually hurts TPOT and supports batch-dependent speculation
  ([dynamic speculative decoding](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md)).
- Static lookahead is not generally optimal; dynamic lookahead improved over the best
  fixed setting in DISCO
  ([Mamou et al.](https://arxiv.org/abs/2405.04304)). We measure fixed depths first so
  the adaptive policy has sound inputs.

Therefore n-gram is the mandatory control, Qwen3-4B/0.6B is the primary model pair,
Qwen3-4B/1.7B is a short screening challenger, and model drafting remains conditional.
“Feature works” is distinct from “feature improves this T4 x2 system.” A negative
model-drafter result is valid if the method is correct and the measurement controlled.

### 2.1 Model-pair decision

The final selection is based on measured accepted length and cost, not parameter count or
acceptance rate alone. For a fixed depth `K`, compute:

```text
round_cost = K * draft_step_ms + target_verify_K_ms + coordination_ms
useful_tokens_per_round = 1 + mean_accepted_draft_tokens
predicted_speedup = useful_tokens_per_round * target_decode_step_ms / round_cost
```

The `+1` is the target correction/bonus that always advances a valid round. Use the
measured accepted-length histogram rather than assuming independent per-token acceptance.

| target / proposer | cost/acceptance rationale | disposition |
|---|---|---|
| Qwen3-4B / n-gram | almost no GPU draft cost; acceptance is workload-dependent | mandatory control |
| Qwen3-4B / Qwen3-0.6B | largest feasible cost gap (~6.7x by checkpoint bytes), same family | primary pair |
| Qwen3-4B / Qwen3-1.7B | potentially higher acceptance, but ~2.3x target/draft size ratio and much dearer per proposal | challenger only |
| Qwen3-1.7B / Qwen3-0.6B | easier correctness bring-up, but only ~2.8x parameter gap | correctness/control, not final target |
| Qwen3-8B / any FP16 draft | target weights consume essentially one T4 before production KV/graphs | excluded |
| cross-family small draft | lower cost possible, but token mapping/alignment and acceptance risk | excluded from first implementation |
| Qwen3-4B / DFlash2 or EAGLE head | promising dedicated speculator, but a different algorithm/runtime contract | future phase, not the draft-model pair test |

AWS reports that a 1.7B Qwen3 drafter beat 0.6B for a 32B target because the acceptance
gain offset draft cost
([AWS experiment](https://aws.amazon.com/blogs/machine-learning/accelerating-decode-heavy-llm-inference-with-speculative-decoding-on-aws-trainium-and-vllm/)).
That does not imply the same answer for a 4B target: the 1.7B drafter is 42.5% of the
target by parameters rather than 5.3%. We therefore measure both candidates cheaply, but
the 1.7B drafter advances only if its lower-confidence-bound predicted speedup exceeds
the 0.6B candidate by at least 5%.

The pair-screen runs before engine integration and uses identical prompts, K in
`{2,3,4}`, batch in `{1,2,4}`, warmed per-step timings, and the same greedy token oracle.
It records acceptance by task stratum. A candidate is eliminated if any is true:

- predicted speedup is `<= 1.05x` at batch 1;
- draft time is at least 35% of the equivalent serial target work without a compensating
  accepted-length gain;
- acceptance collapses on two or more representative strata;
- the target/draft tokenizer or chat-template fingerprint differs;
- memory/graph warmup leaves less than 1.5 GiB physical headroom on either GPU.

Only the selected pair receives the full paged-draft integration and long reliability
matrix. This prevents spending most of the Kaggle quota implementing a pair whose cost
model already predicts a loss.

## 3. Existing code audit

### What is reusable

- `greedy_accept` correctly accepts the matching prefix and emits a target correction or
  bonus token.
- Both single-sequence decoders explicitly crop target and draft caches after rejection.
- `OptimizedSpeculativeDecoder` keeps proposal tokens on GPU during drafting.
- `BatchedSpeculativeEngine` exposes the important ragged-acceptance problem and records
  true versus uniformly committed acceptance.
- `warmup_compare.py` uses synchronized timings, warmup, and repeated medians.
- `spec_batch_crossover.py` compares batched greedy and speculative execution on the
  same HF kernel path.

### What must not be used as release evidence

- The single-sequence unit suite tests only `greedy_accept`, not cache alignment, EOS,
  rollback, or end-to-end generation.
- `benchmarks/speculative/vanilla.py` times the first target-only generation without
  warmup and is cold-start biased.
- Tokenizer compatibility is checked only by vocabulary size. Release code must compare
  vocabulary/token-id mapping, special token IDs, and chat-template behavior.
- The optimized prototype performs several Python-visible tensor conversions during
  acceptance and has no production scheduler integration.
- The batched prototype counts proposals for completed/padded rows, uses a minimum-commit
  throttle, and keeps completed rows in the batch. Its throughput and acceptance metrics
  cannot be compared directly with the live engine.
- The batched FP16 correctness test permits a top-two flip with an arbitrary logit-gap
  threshold of 1.0. That is a useful diagnostic, not an exactness proof.
- Neither prototype uses the live paged pool, block tables, prefix sharing, preemption,
  per-request sampling, stop rules, fused prefill/decode, server lifecycle, or production
  CUDA graphs.

Local non-CUDA audit result on the baseline checkout: `2 passed, 3 deselected` for the
speculative test files. This proves only the two CPU `greedy_accept` cases; it is not GPU
or integration evidence.

## 4. Required engine contract

### 4.1 Request invariant

The live engine emits the first target token at prefill. While decoding, a request has:

- paged KV for `prompt + output[:-1]`;
- `next_token_id == output[-1]`, already delivered but not yet in KV.

For speculation depth `K`, a proposer predicts `K` future tokens after that pending
token. Target verification consumes:

```text
[pending_token, proposal_0, ..., proposal_(K-1)]
```

and returns `K + 1` predictions: `K` proposal comparisons plus the fully accepted bonus.
If `a` draft tokens are accepted (`0 <= a <= K`), the
round emits the `a` accepted draft tokens followed by one target correction, except that
a fully accepted round may emit the verified bonus. The target cache commits exactly
`a + 1` newly cached inputs (the old pending token plus the accepted prefix), and the
last emitted token becomes the new pending token. This invariant must hold after every
round, rejection, EOS, cancellation, preemption, and resumption.

### 4.2 Transactional cache rules

Before verification, reserve capacity for the maximum writes but do not advance logical
sequence lengths. The verification forward may write speculative K/V into reserved
slots. After acceptance:

- advance logical length only through the accepted cache prefix;
- leave rejected suffix bytes unreachable; later writes overwrite them;
- never publish speculative pages to the prefix cache;
- copy-on-write a shared partial tail before the first speculative write;
- on exception, leave logical lengths unchanged and run/fail according to the existing
  engine error policy;
- release both target and draft allocations on finish, cancellation, rejection, or
  permanent failure.

An explicit assertion after each debug/test round checks:

```text
allocation.sequence_length == prompt_tokens + emitted_tokens - 1
```

for every decoding request.

### 4.3 Verification path

Add a paged multi-token verification forward rather than calling the HF prototype:

- stage `[batch_bucket, K + 1]` input IDs, positions, sequence lengths, and block tables in
  persistent buffers;
- reuse the chunk-capable paged K/V writer and causal prefill attention semantics;
- return logits for every live verification position, not just the last position;
- project only live `[B * (K + 1)]` hidden states through the LM head;
- use one GPU acceptance kernel (or vectorized torch operation initially) to find the
  first mismatch per row and copy only compact counts/tokens to the host;
- key graphs by `(batch_bucket, K, attention regime/context bucket)` and warm every key
  used by the policy before admitting traffic.

Depth 1 must use the same target input and semantics as ordinary decode. It is the
structural oracle: any depth-1 token, cache, stop-reason, or metric mismatch is a bug.

### 4.4 Proposer interface

Use one narrow interface so policy and verification do not depend on proposer type:

```text
propose(requests, max_depth) -> tokens[B, K], valid_lengths[B], proposer_metrics
commit(requests, accepted_lengths, emitted_tokens)
rollback_or_release(requests)
```

Implementations:

1. `NgramProposer`: CPU lookup over prompt plus emitted token IDs; no proposal means
   `valid_length=0` and immediate ordinary decode for that row.
2. `DraftModelProposer`: the pair-screen winner (expected Qwen3-0.6B) on GPU 1, with its
   own right-sized paged KV pool, staging buffers, graph buckets, lifecycle hooks, and
   tokenizer-compatibility gate.

Rows with different valid proposal lengths must not verify uninitialized/pad proposals.
Initially group rows by `(K, context regime)`; do not use the parked minimum-commit
throttle. Per-row accepted lengths remain independent.

### 4.5 Eligibility and fallback

A request is speculative only when all are true:

- greedy sampling with no penalties and no requested logprobs;
- at least two output slots remain (otherwise ordinary decode);
- proposer has at least one candidate;
- batch/context lies in a measured profitable policy cell;
- no unsupported structured-output or API behavior is active;
- target and proposer state are aligned.

Ineligible rows run ordinary decode. A runtime feature flag disables speculation without
restarting or changing output semantics. A speculative exception before logical commit
may fall back to ordinary decode; after an uncertain commit boundary the request fails
loudly rather than risk cache corruption.

### 4.6 Planned code boundaries

Keep the change reviewable; do not expand `continuous_batching.py` with an entire second
engine implementation.

| path | responsibility |
|---|---|
| `engine/speculative/proposer.py` | proposer protocol, eligibility result, common metrics |
| `engine/speculative/ngram.py` | deterministic prompt/output n-gram proposals |
| `engine/speculative/draft_model.py` | optional paged Qwen draft state and lifecycle |
| `engine/speculative/acceptance.py` | batched greedy acceptance and strict near-tie handling |
| `engine/batching/speculative_step.py` | stage, verify, transactional commit/rollback |
| `engine/graphs/paged_verify_graph.py` | fixed-shape multi-token target graph capture/replay |
| `engine/batching/continuous_batching.py` | policy hook and ordinary-path fallback only |
| `engine/runtime/request.py` | minimal per-request speculative state/metrics |
| `benchmarks/speculative/engine_ab.py` | interleaved ordinary/ngram/model-draft A/B/C |
| `benchmarks/speculative/memory_probe.py` | staged pool/graph memory feasibility |
| `tests/speculative/` | state-machine, property, engine, graph and fault tests |
| `scripts/kaggle_t4x2_speculative.ipynb` | restart-safe Kaggle orchestration and artifact export |

The existing parked prototypes remain unchanged until the engine-native path passes;
their behavior is useful as an independent comparison. Shared logic may be consolidated
only after parity tests prove the refactor did not change either path.

## 5. Implementation sequence and gates

Do not proceed to the next phase when a gate fails.

### Phase 0 — freeze and record the Kaggle environment

1. Select **GPU T4 x2** in Kaggle and start a fresh session. Assert exactly two T4s,
   capability 7.5, 16 GB reported per card, and less than 600 MiB initially used on each.
   Fail rather than silently running on a P100, L4, one visible card, or a contaminated
   session.
2. Checkout `d543a5e` (or record the replacement SHA), run `scripts/setup_kaggle.sh`, and
   extend the preflight record to include both physical GPU UUIDs and peer-access result.
3. Record Python, torch, CUDA, Triton, transformers, driver, model revisions, Git diff,
   GPU topology, per-GPU clocks, power limits, and `nvidia-smi` memory.
4. Pin the resolved package versions in the result manifest. Never compare sessions with
   silently different stacks.
5. Set deterministic seeds and disable gradients. Use FP16; T4 has no native BF16 Tensor
   Core path.
6. Use explicit devices. Never use `device_map="auto"`: it may shard the target across
   both cards, which violates the engine's single-device paged-KV contract. GPU 0 owns
   the complete target; GPU 1 owns the complete draft.

Baseline command skeleton (run from the repository root):

```bash
bash scripts/setup_kaggle.sh
python scripts/kaggle_preflight.py --require-t4-count 2 --out "$RUN/preflight.json"
nvidia-smi -L > "$RUN/nvidia_smi_L.txt"
nvidia-smi topo -m > "$RUN/nvidia_topology.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,clocks.sm,clocks.mem,power.limit \
  --format=csv > "$RUN/nvidia_smi.csv"
python -m pytest -q -p no:cacheprovider
python -m pytest -q -m cuda -p no:cacheprovider \
  --ignore=tests/batching/test_batched_speculative.py
python -m pytest -q -m cuda -p no:cacheprovider \
  tests/batching/test_batched_speculative.py
```

`RUN` must be an explicit run directory created by the notebook. If any command exits
non-zero, preserve its log, stop that phase, and do not run later performance cells.

Gate: preflight passes for two T4s; repository is identifiable; all screening checkpoints
download and their revisions/tokenizers are recorded; no unexplained GPU owner exists.
`scripts/kaggle_preflight.py` is a planned deliverable; the existing single-device
preflight must not be treated as sufficient.

### Phase 1 — preserve the current baselines

Run, each in a fresh process:

1. CPU suite, then all non-spec CUDA tests.
2. Existing single-sequence vanilla and optimized speculative tests/benchmarks, with
   warmup and correctness added around them.
3. Existing batched speculative correctness suite in FP16. Run the FP32 test only if the
   measured peak proves it fits; its decorator's 12 GiB check is necessary but not a
   sufficient OOM guarantee.
4. Live-engine target-only soak on Qwen3-4B/GPU 0 with a 1,024-block target pool and
   graph buckets `(1,2,4)`.
5. Pair-screen Qwen3-0.6B and Qwen3-1.7B on GPU 1 using the Section 2.1 cost equation;
   keep raw per-stratum acceptance and phase timings.

Record current failures as baseline facts; do not patch during the baseline run.

Gate: ordinary engine token/correctness tests pass, the soak ends with zero leaked pages,
and the existing speculative prototypes either pass or have a reproducible classified
failure.

### Phase 2 — build a complete correctness oracle

Add CPU/property tests for:

- mismatch at every position `0..K-1` and full acceptance;
- `K` in `{1,2,3,4,6}` and remaining output budget smaller than K;
- empty/invalid depth rejection;
- single EOS and multiple EOS IDs at every proposal/correction/bonus position;
- stop-token precedence, `ignore_eos`, exact length termination, and no post-terminal
  cache write;
- proposal validity shorter than configured K;
- tokenizer incompatibility and unsupported sampling fallback;
- accepted/proposed counters excluding padding, completed rows, missing proposals, and
  candidates after terminal tokens.

Add a tiny deterministic fake model/cache test that records every input, crop, and cache
length. This catches off-by-one errors without FP16 numerical ambiguity.

Gate: 100% branch coverage of the acceptance/commit state machine and property tests over
random K, mismatch, EOS, and remaining-budget combinations.

### Phase 3 — engine-native depth 1

Implement the verification transaction at K=1 behind `speculative=False` by default.
Run the full ordinary engine suite twice, feature off and forced K=1.

Required exact comparisons:

- token IDs, text, finish reason, output logprobs when applicable to the fallback path;
- per-request output timestamps count (timings need not equal);
- target block table, logical sequence length, allocator refcounts, free-page count;
- scheduler state and terminal cleanup;
- prefix-cache refcounts and copy-on-write behavior;
- preemption/rebuild and cancellation from PREFILLING/DECODING.

Gate: forced K=1 is token-identical to ordinary greedy for every corpus row and leaves
identical logical engine state. Zero allocator audit failures over a 30-minute stress run.

### Phase 4 — n-gram proposer and K > 1

Implement prompt/output lookup with configurable match length and proposal depth. Sweep
ngram match sizes `{1,2,3,4}` and K `{2,3,4,6}` first in eager mode.

Correctness has two modes:

- **Strict release mode:** must reproduce ordinary greedy token IDs. When a parallel
  verification position has a target top-two margin below the empirically established
  numerical guard, recompute that ambiguous decision on the ordinary one-token target
  path before commit. If exactness cannot be achieved without recomputing most rounds,
  do not ship an “exact” claim; leave the feature experimental/off.
- **Diagnostic mode:** permits only a classified FP16 near-tie flip. It records both
  choices, top logits, absolute/relative gap, context, depth, and kernel shape. This mode
  is research evidence, not the release gate.

Gate: strict mode is exactly token-identical on the complete corpus; every injected
rollback/cancellation test passes; no invalid page becomes reachable.

### Phase 5 — CUDA graphs and the live scheduler

Capture only the K/batch cells that eager measurement suggests can win. Warm them before
the timed interval and assert `lazy_graph_captures == 0` during every measured run.

Test mixed steps explicitly:

- speculative decode only;
- speculative decode plus new prefill arrival;
- ordinary and speculative rows together;
- request finishing mid-round while others continue;
- cancellation before verification, between verification and commit (fault injection),
  and after commit;
- memory pressure causing preemption of speculative and ordinary rows;
- prefix exact hit and shared partial-tail copy-on-write.

Gate: full CUDA suite passes, graph/eager outputs agree, no in-window capture occurs, and
a one-hour randomized soak has zero leaks, corruptions, deadlocks, or lost completions.

### Phase 6 — dual-GPU model drafter

Only start after Phase 5 and the Section 2.1 pair screen. GPU 0 owns the complete
Qwen3-4B target and GPU 1 owns the complete winning drafter. The cards are separate
memory domains; their capacities must never be added into a fictional 32 GB pool.

Memory configurations to probe in fresh processes:

| configuration | GPU 0 target blocks / KV | GPU 1 draft blocks / KV | purpose |
|---|---:|---:|---|
| safe bring-up | 512 / 1.125 GiB | 512 / 0.875 GiB | correctness and first graph capture |
| capacity probe | 768 / 1.688 GiB | 768 / 1.313 GiB | moderate contexts/concurrency |
| primary candidate | 1024 / 2.25 GiB | 1024 / 1.75 GiB | planned performance configuration |
| upper probe | 1536 / 3.375 GiB | 1536 / 2.625 GiB | only if both headroom gates pass |

Target figures use 144 KiB/token; the 0.6B draft uses 112 KiB/token; pages hold 16
tokens. If 1.7B wins screening, its KV bytes per token remain 112 KiB but its roughly
4.08 GB checkpoint reduces GPU 1 headroom. For every configuration, measure actual
allocated/reserved peaks after all selected graphs are warm. Calculations are planning
bounds, not permission to allocate.

Start with one process only if all model loading, current-device queries, attention
contexts, Triton launches, CUDA events and graph captures are explicitly device-scoped.
The baseline loader currently defaults to `cuda`, so device plumbing is required. If
global hook/context state cannot be made safely device-scoped, use one process per GPU
and a bounded IPC protocol. Proposal payloads are tiny, but every synchronization is
timed. Peer access is an optimization, never a correctness requirement.

Cross-GPU execution is sequential for a given request: target round `r` cannot verify
until drafting completes, and draft round `r+1` depends on target acceptance. Do not
claim compute overlap unless a trace proves useful overlap across independent request
cohorts.

Draft lifecycle rules:

- prefill/rebuild draft state from exactly the same committed prefix as the target;
- target and draft allocations commit/rollback atomically at the logical level;
- preemption releases both pools and resumption rebuilds both;
- cancellation and terminal cleanup release both;
- draft prefix caching starts disabled; add it only after independent refcount tests;
- do not let draft prefill silently inflate TTFT or second-token latency—record both.

Gate: same strict correctness and soak gates as n-gram; at least 1.5 GiB physical free
headroom on **each** GPU after warmup; no PyTorch allocator retry; no process OOM; no
growing reserved memory over repeated reset/warmup cycles; target and draft CUDA events
are recorded on their owning devices.

### Phase 7 — performance experiment

Use same-session, interleaved A/B(/C) runs:

- A: ordinary live engine;
- B: engine-native n-gram speculation;
- C: engine-native winning draft-model pair (only if Phase 6 passes).
- D: two independent Qwen3-4B target-only replicas, one per GPU, for the hardware-normalized
  aggregate-throughput control.

Each arm gets identical prompt order, seeds, request parameters, concurrency, target
checkpoint, pool capacity available to customer requests, graph warmup, and run duration.
Rotate arm order between repeats. Use at least 2 discarded warmups and 5 measured repeats;
report median, min/max, spread, and bootstrap 95% confidence intervals. A result from a
different Kaggle session is contextual only, never the paired baseline.

#### Fixed matrix

| dimension | values |
|---|---|
| concurrency | 1, 2, 4, 8; 16 only if memory permits |
| K | 1, 2, 3, 4 initially; 6 only if K=4 still improves |
| generated length | 32 for correctness, 128 for steady-state performance |
| prompt profile | short 32-64, chat ~656, long ~1,800 tokens |
| content | natural chat, code, summarization/copy-heavy, repetitive, adversarial low-acceptance |
| execution | eager first, then fully warmed CUDA graphs |
| arrivals | closed-loop for latency/performance; open-loop bursts for stability only |

Do not run the full Cartesian product blindly. First screen all K values on concurrency
1 and 4, retain the best two plus K=1, then run the complete workload matrix.

#### Metrics

Primary:

- delivered output tokens/s (aggregate and per request);
- two-GPU system tokens/s versus two independent target-only replicas, so using twice the
  hardware is not presented as a free speedup;
- TTFT, time to second token, ITL p50/p95/p99/p999, end-to-end latency;
- decode-only and prefill-carrying step latency;
- expected gap weighted by actual step mix;
- request completion rate and queue time.

Speculation-specific:

- proposed and accepted draft tokens over active valid rows only;
- acceptance-rate and accepted-length histogram `0..K`;
- mean emitted tokens per verification round;
- target verification forwards and draft forwards per delivered token;
- proposer time, target verify time, acceptance/commit time, host sync time;
- no-proposal, low-confidence, sampling, batch-policy and memory fallbacks;
- wasted verified tokens and overwritten speculative KV bytes;
- strict-mode near-tie recomputations.

Resources:

- peak allocated/reserved VRAM after graph warmup;
- customer-usable KV blocks after graph/draft reservations;
- SM clock, memory clock, power, GPU utilization;
- lazy graph captures and allocator retries.

Acceptance rate alone is never a performance verdict. The deciding value is delivered
tokens per wall-clock second with unchanged correctness and acceptable tail latency.

### Phase 8 — adaptive policy and final validation

Construct a small lookup policy from Phase 7, keyed by proposer, live batch bucket,
context bucket and remaining output length. Enable a cell only when the lower confidence
bound beats ordinary decode. Add an acceptance-length EMA so n-gram/model speculation
backs off after repeated zero-accept rounds. Do not tune and evaluate on the same prompt
set: use separate calibration and held-out corpora.

Final ship gates:

1. Strict greedy token identity: 100% on held-out and stress corpora.
2. Reliability: zero leaked/refcount-corrupt pages, lost requests, deadlocks, or
   post-terminal emissions in the one-hour soak and all fault-injection cases.
3. Memory: at least 1.5 GiB physical headroom after all selected graphs are warm.
4. Performance: for the latency objective, the lower 95% CI for per-request decode
   throughput/TPOT speedup is at least 1.10x at concurrency 1 and 1.05x in every other
   enabled policy cell. Aggregate two-GPU throughput versus two target-only replicas is
   always reported separately; a latency win must not be relabeled a hardware-efficiency
   win.
5. QoS: TTFT p95 and ITL p99 regress no more than 5% in any enabled cell; otherwise that
   cell is disabled even if mean throughput improves.
6. Operational: feature-off path matches the release baseline; kill switch and fallback
   counters are tested; result manifest is complete.

If no cell passes, retain the correct implementation as experimental and keep it off by
default. Do not average winning and losing cells into a misleading global speedup.

## 6. Failure-mode register

| failure | detection | prevention/recovery |
|---|---|---|
| Wrong cache off-by-one | depth-1 state equality; per-round length assertion | transactional logical commit; fake-cache property tests |
| Rejected K/V becomes visible | poison rejected slots; compare later logits | logical lengths gate all reads; overwrite before reuse |
| Shared prefix mutated | refcount/content checks on sibling request | copy-on-write partial tail before verify; never publish speculation |
| Ragged acceptance corrupts rows | different forced mismatch per row | independent per-row lengths/grouping; no minimum-commit throttle |
| EOS/stop emitted past terminal | EOS at every round position | truncate before commit; no terminal cache decode; per-row removal |
| Length limit exceeded by bonus | remaining-budget property tests | cap proposal length and commit/emission independently |
| FP16 batch-shape top-token flip | ordinary-vs-verify logits and margin log | strict near-tie one-token recompute or experimental-only claim |
| Incompatible tokenizer/model pair | full token-map/special-ID fingerprint | reject model proposer at construction; fallback to n-gram/ordinary |
| Sampling distribution changes | sampled request routing tests | greedy-only eligibility until exact rejection sampler is proven |
| RNG changes because neighbor exits | deterministic per-request seed tests | ordinary fallback initially; request-owned generators later |
| Draft/target cache desynchronizes | length/content fingerprint after every debug round | atomic lifecycle and joint rebuild; fail closed on uncertain commit |
| Model or tensor lands on wrong GPU | UUID/device assertions around every model, pool, event and graph | explicit `cuda:0`/`cuda:1`; forbid `device_map="auto"` |
| Cross-GPU wait erases speedup | per-round draft, transfer, wait and verify events | compact token payloads; bounded IPC; disable cells whose cost equation fails |
| Unsafe process-global attention context crosses devices | concurrent two-device stress and device assertions in hooks | make contexts device-scoped or isolate one process per GPU |
| Claimed dual-GPU speedup merely spends twice the hardware | compare with two independent target-only replicas | label latency and aggregate-throughput results separately |
| Draft prefill destroys TTFT/TBT | TTFT and second-token metrics | n-gram default; async policy is out of scope; disable model draft cell |
| Draft cost exceeds saved target work | phase timing and speedup CI | per-cell policy/backoff; feature disabled at large batches |
| No/low n-gram match | proposal-validity and fallback counters | skip verification and use ordinary decode immediately |
| VRAM OOM during pools/graphs | staged fresh-process probes; actual peak/headroom | 512-block bring-up; cap graph keys; abort before load increase |
| CUDA graph captures during traffic | `lazy_graph_captures` assertion | enumerate/warm selected keys; eager fallback for unknown key |
| Graph replays stale metadata | poison/change inputs across replays | fixed buffers plus replay correctness test per key |
| Graph padding touches real pages | dummy-page canaries/refcount audit | permanent isolated dummy pages per graph width |
| Preemption leaks draft pages | pressure soak and dual-pool audit | one lifecycle owner releases both pools; rebuild from committed output |
| Cancellation races with verify | deterministic fault-injection barriers | single GPU-owning worker; check cancellation before commit |
| New prefill starves/blocks speculation | mixed-arrival step attribution | scheduler policy measured with fused arrivals; QoS cell gate |
| Completed rows inflate metrics | hand-computed miniature scenarios | count only active, valid, pre-terminal proposals |
| Warmup/JIT contaminates result | first-vs-warmed trace; graph counter | discard warmups; fresh process; time only after complete warmup |
| Thermal/clock drift creates false win | clocks per repeat and interleaved arms | rotate A/B/C; flag drift; rerun unstable session |
| Different model/package revision | manifest comparison | pinned revisions and environment; refuse unmatched A/B |
| OOM leaves notebook contaminated | GPU-owner and used-memory check | one benchmark per process; terminate failed process; recheck before next |
| Acceptance overfits easy prompts | held-out content-stratified corpus | report each stratum; adaptive policy calibrated separately |
| Server/API behavior silently changes | endpoint/SSE parity and disconnect tests | speculative core below unchanged API; unsupported features fallback |
| Metrics claim speedup from less work/output | token counts and completion parity | compare delivered tokens and identical requests, not rounds alone |

## 7. Kaggle T4 x2 run layout

Every session writes to:

```text
results/t4/speculative/<UTC-date>_<git-sha>/
  manifest.json
  preflight.json
  logs/
  correctness/
  memory/
  microbench/
  closed_loop/
  open_loop/
  summary.json
  decision.md
```

Each JSON includes command line, Git SHA/diff state, model revisions, environment,
device/clocks, configuration, seed, prompt-set hash, warmup count, repetitions, raw
samples, summary statistics, and gate verdict. Logs are retained even on failure. Before
the Kaggle session ends, create a Kaggle output artifact and also zip/download the run
directory or commit only the result artifacts to a dedicated results branch.

Suggested session split:

- Session A: environment, baseline, prototype tests, depth-1 bring-up.
- Session B: 0.6B-versus-1.7B pair screen plus n-gram eager screen.
- Session C: n-gram correctness, rollback/fault injection, graphs and soak.
- Session D: winning model-draft memory/correctness probe.
- Session E: model-draft performance and held-out adaptive-policy validation, only if D
  passes.

Never combine numbers across these sessions as if paired. Each session that measures a
candidate also re-runs its own ordinary baseline.

The checked-in notebook runs the screen, selection, and live-engine validation as
separate fail-fast processes:

```bash
python -m benchmarks.speculative.pair_screen \
  --draft-model Qwen/Qwen3-0.6B --depths 2,3,4 \
  --output "$RUN/pair_06b.json"
python -m benchmarks.speculative.select_pair \
  "$RUN/pair_06b.json" "$RUN/pair_17b.json" \
  --output "$RUN/pair_selection.json"
python -m benchmarks.speculative.engine_ab \
  --draft-model Qwen/Qwen3-0.6B --depth 3 --concurrencies 1,2,4 \
  --num-blocks 512 --warmup 2 --runs 5 --max-new-tokens 128 \
  --output "$RUN/engine_ab.json"
```

The notebook substitutes the measured winning model/depth into the final command. Each
completed process writes its own JSON artifact, so a Kaggle disconnect does not erase
earlier gates.

## 8. Deliverables

The work is complete only with:

1. engine-native proposer/verify/commit code behind a default-off flag;
2. unit, property, CUDA integration, graph, scheduler, cache, server, and fault tests;
3. a Kaggle T4 x2 notebook/script that runs the gated phases and writes machine-readable data;
4. raw result JSON and logs with full provenance;
5. an update to `docs/design-decisions.md` stating accepted/rejected/experimental status;
6. a final report separating algorithm correctness, engine integration, n-gram result,
   model-drafter result, profitable policy cells, negative cells, and limitations;
7. default enablement only if every Phase 8 ship gate passes.

## 9. Expected result, stated before measurement

The strongest T4 candidate is n-gram speculation at low concurrency on repetitive,
copy-heavy, summarization, and code workloads. It has near-zero model proposal cost and
no second KV pool. It may provide no benefit on open-ended chat with few reusable spans.

Qwen3-4B on GPU 0 plus Qwen3-0.6B on GPU 1 is the primary model pair. It has the largest
feasible target/draft cost gap within the engine's validated FP16 architecture family.
The 1.7B drafter advances only if its measured acceptance-length gain more than pays for
its much higher draft cost. Model drafting may still lose as batching rises because the
target verification becomes compute-bound and K serial draft passes remain. This prior
must not influence the gate: paired Kaggle results decide.

The experiment succeeds even if model drafting is rejected, provided it produces a
reproducible, correctness-gated explanation and leaves the ordinary engine unchanged.

"""Mixed-arrival reliability soak for the continuous-batching engine.

What this exercises that the staged tests do not: requests arriving at random times while
others are mid-generation, prompts that share prefixes so the cache is really used and
really evicted, cancellations aimed at *every* lifecycle state rather than whichever one
happens to come up, and a full accounting audit once the engine has drained.

The audit is the point. A soak that merely finishes proves very little; a soak that
finishes and can account for every KV page proves the lifecycle is closed.

Run:
    python -m benchmarks.reliability.soak --duration 60 --arrival-rate 25
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

from engine.runtime import GenerationRequest, RequestState

_TERMINAL = {
    RequestState.FINISHED, RequestState.CANCELLED,
    RequestState.FAILED, RequestState.REJECTED,
}
# States a request can be cancelled from. WAITING appears twice in a request's life -
# before first admission and after a yield - and they are different code paths, so the
# injector tracks them separately.
_CANCEL_TARGETS = ("WAITING", "PREFILLING", "DECODING", "PREEMPTED")


@dataclass(frozen=True)
class SoakConfig:
    duration_s: float = 30.0
    # Open loop: arrivals are independent of completions. Realistic for public traffic,
    # but if the rate exceeds what the engine can retire, the queue grows without bound
    # and every latency number degenerates into a measure of queue depth.
    arrival_rate_per_s: float = 20.0
    # Closed loop: hold exactly this many requests in flight, submitting a replacement
    # as each one ends. Self-limiting, so latency reflects the engine rather than the
    # backlog. Set this for any run whose numbers will be compared against another engine.
    concurrency: int | None = None
    shared_prefixes: int = 3
    prefix_tokens: tuple[int, int] = (16, 128)
    suffix_tokens: tuple[int, int] = (4, 96)
    max_new_tokens: tuple[int, int] = (1, 64)
    cancel_probability: float = 0.15
    seed: int = 0
    max_steps: int = 500_000
    sample_every_steps: int = 25
    # Ask the engine for its per-phase step split (host staging, decode/prefill GPU time,
    # sampling sync). Costs one extra event sync on prefill steps that complete no
    # request; both arms of an A/B pay it equally.
    instrument: bool = False


@dataclass
class SoakResult:
    config: SoakConfig
    steps: int = 0
    wall_s: float = 0.0
    submitted: int = 0
    mode: str = "open-loop"
    counts: Counter = field(default_factory=Counter)
    finish_reasons: Counter = field(default_factory=Counter)
    cancelled_from: Counter = field(default_factory=Counter)
    states_observed: Counter = field(default_factory=Counter)
    latency: dict[str, float] = field(default_factory=dict)
    recompute: dict[str, float] = field(default_factory=dict)
    delivered_tokens: int = 0
    peak_kv_utilization: float = 0.0
    peak_waiting: int = 0
    peak_active: int = 0
    # Mean decode operating point, weighted by sampled steps. Quoting a latency number
    # without these makes it uncomparable to any floor.
    mean_decode_batch: float = 0.0
    mean_context_tokens: float = 0.0
    # Step-duration distributions split by what the step actually did. A decode-only step
    # is the engine at its best; a step that also prefills is what every decoding sequence
    # pays for someone else's prompt.
    step_timing: dict[str, float] = field(default_factory=dict)
    _decode_step_ms: list[float] = field(default_factory=list, repr=False)
    _prefill_step_ms: list[float] = field(default_factory=list, repr=False)
    # Per-phase samples from the engine's instrumentation, keyed by phase name.
    _phase_ms: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _batch_samples: list[float] = field(default_factory=list, repr=False)
    _context_samples: list[float] = field(default_factory=list, repr=False)
    prefix_cache: dict[str, float] = field(default_factory=dict)
    # Correctness: the engine did something it must never do. Always a failure.
    violations: list[str] = field(default_factory=list)
    # Workload: this run did not exercise something it could have. Says nothing about the
    # engine - a short run with a low cancel probability simply may not reach every state.
    # Kept apart from violations so a clean engine never reports a correctness failure for
    # a statistical accident, which is how people learn to ignore a suite.
    coverage_gaps: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def fully_covered(self) -> bool:
        return not self.coverage_gaps

    @property
    def waste_ratio(self) -> float:
        """Prompt tokens rebuilt per output token actually delivered to a caller.

        Counted from real output lengths across every request, finished or cancelled -
        a cancelled request's tokens were still produced and still cost KV pages.
        """
        delivered = self.delivered_tokens
        return self.recompute.get("recomputed_tokens", 0) / delivered if delivered else 0.0

    def to_dict(self) -> dict:
        payload = {
            "config": self.config.__dict__, "steps": self.steps, "wall_s": self.wall_s,
            "submitted": self.submitted, "mode": self.mode, "counts": dict(self.counts),
            "finish_reasons": dict(self.finish_reasons),
            "cancelled_from": dict(self.cancelled_from),
            "states_observed": dict(self.states_observed), "latency_ms": self.latency,
            "recompute": self.recompute, "delivered_tokens": self.delivered_tokens,
            "recompute_tokens_per_delivered_token": self.waste_ratio,
            "peak_kv_utilization": self.peak_kv_utilization,
            "peak_waiting": self.peak_waiting, "peak_active": self.peak_active,
            "mean_decode_batch": self.mean_decode_batch,
            "mean_context_tokens": self.mean_context_tokens,
            "step_timing": self.step_timing,
            "prefix_cache": self.prefix_cache, "violations": self.violations,
            "coverage_gaps": self.coverage_gaps,
        }
        return payload


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


class _Workload:
    """Prompts built from a small pool of shared prefixes plus a random tail.

    Shared prefixes are what make the prefix cache do real work: several requests hit the
    same pages, refcounts rise above one, and eviction has to distinguish pinned entries
    from free ones. Fully random prompts would never exercise that.
    """

    def __init__(self, config: SoakConfig, rng: random.Random, forbidden: set[int]):
        self.config = config
        self.rng = rng
        low, high = config.prefix_tokens
        self.prefixes = [
            [self._token(rng, forbidden) for _ in range(rng.randint(low, high))]
            for _ in range(config.shared_prefixes)
        ]
        self.forbidden = forbidden

    @staticmethod
    def _token(rng: random.Random, forbidden: set[int]) -> int:
        while True:
            token = rng.randint(1000, 20_000)
            if token not in forbidden:
                return token

    def next_request(self, index: int) -> GenerationRequest:
        prefix = self.rng.choice(self.prefixes)
        low, high = self.config.suffix_tokens
        suffix = [
            self._token(self.rng, self.forbidden)
            for _ in range(self.rng.randint(low, high))
        ]
        prompt = prefix + suffix
        new_low, new_high = self.config.max_new_tokens
        return GenerationRequest(
            request_id=f"soak-{index:05d}", prompt_token_count=len(prompt),
            max_new_tokens=self.rng.randint(new_low, new_high), prompt_token_ids=prompt,
        )


def _classify(request: GenerationRequest) -> str:
    """Lifecycle bucket used by the cancellation injector."""
    if request.state is RequestState.WAITING:
        return "PREEMPTED" if request.preempted_count else "WAITING"
    return request.state.name


def check_invariants(engine, submitted: list[GenerationRequest]) -> list[str]:
    """Audit a drained engine. Every returned string is a violation.

    These are the properties that must hold no matter what the workload did, so they are
    equally valid as a soak assertion and as a post-run check in production.
    """
    violations: list[str] = []
    scheduler = engine.scheduler

    if scheduler.active:
        violations.append(f"{len(scheduler.active)} requests still active after drain")
    if scheduler.waiting:
        violations.append(f"{len(scheduler.waiting)} requests still queued after drain")

    unterminated = [r.request_id for r in submitted if r.state not in _TERMINAL]
    if unterminated:
        violations.append(f"non-terminal requests: {unterminated[:5]}")

    holding = [r.request_id for r in submitted if r.allocation is not None]
    if holding:
        violations.append(f"terminal requests still holding KV: {holding[:5]}")

    # The accounting identity: every page in use belongs either to a permanent engine
    # reservation or to the prefix cache. Anything else is a leaked customer page.
    blocks = engine.block_manager.snapshot()
    used = int(blocks["used_blocks"])
    reserved = len(engine._graph_dummy_blocks)
    cached = int(engine.prefix_cache.snapshot()["cached_blocks"])
    if used != reserved + cached:
        violations.append(
            f"KV page leak: {used} used != {reserved} reserved + {cached} cached"
        )

    # The epoch counts exits that actually returned pages. A request rejected at
    # admission, or cancelled while parked in the queue after a yield, held nothing and
    # must not advance it - otherwise yielded peers would be released from the gate by an
    # event that freed no memory.
    page_releasing = sum(1 for r in submitted if r.held_pages_at_exit)
    if scheduler.progress_epoch != page_releasing:
        violations.append(
            f"progress_epoch {scheduler.progress_epoch} != {page_releasing} page-releasing exits"
        )
    never_admitted_but_released = [
        r.request_id for r in submitted if r.held_pages_at_exit and r.admitted_ns is None
    ]
    if never_admitted_but_released:
        violations.append(
            f"released pages without ever being admitted: {never_admitted_but_released[:5]}"
        )

    for request in submitted:
        produced = len(request.output_token_ids)
        if produced > request.max_new_tokens:
            violations.append(f"{request.request_id} produced {produced} > max_new_tokens")
        if request.state is RequestState.FINISHED and produced == 0:
            violations.append(f"{request.request_id} finished with no output")
        if request.state in _TERMINAL and request.finish_reason is None:
            violations.append(f"{request.request_id} terminal with no finish_reason")
    return violations


def run_soak(engine, config: SoakConfig | None = None) -> SoakResult:
    """Drive the engine with Poisson arrivals, injected cancellations, and an audit."""
    config = config or SoakConfig()
    rng = random.Random(config.seed)
    workload = _Workload(config, rng, forbidden=set(engine.eos_ids))
    result = SoakResult(config=config)
    if hasattr(engine, "instrument"):
        engine.instrument = config.instrument

    submitted: list[GenerationRequest] = []
    in_flight: dict[str, GenerationRequest] = {}
    started = perf_counter()
    next_arrival = started
    index = 0
    accepting = True
    result.mode = "closed-loop" if config.concurrency else "open-loop"

    def _admit_one() -> None:
        nonlocal index
        request = workload.next_request(index)
        index += 1
        submitted.append(request)
        result.submitted += 1
        if engine.submit(request):
            in_flight[request.request_id] = request
        # A rejected submission (queue full, or larger than effective capacity) is a
        # completed interaction, not a lost one: it stays in `submitted` for the audit.

    while True:
        now = perf_counter()
        elapsed = now - started
        if accepting and elapsed >= config.duration_s:
            accepting = False
        if accepting and config.concurrency:
            while len(in_flight) < config.concurrency:
                _admit_one()
        elif accepting:
            while now >= next_arrival:
                _admit_one()
                next_arrival += rng.expovariate(config.arrival_rate_per_s)

        for request in in_flight.values():
            if not request.done:
                result.states_observed[_classify(request)] += 1

        # Aim cancellations at states not yet covered, so "cancel from every state" is a
        # guarantee rather than something the random schedule might or might not produce.
        if in_flight and rng.random() < config.cancel_probability:
            uncovered = [s for s in _CANCEL_TARGETS if not result.cancelled_from[s]]
            wanted = uncovered or list(_CANCEL_TARGETS)
            candidates = [
                r for r in in_flight.values()
                if not r.done and _classify(r) in wanted
            ]
            if candidates:
                victim = rng.choice(candidates)
                state = _classify(victim)
                engine.cancel(victim.request_id, reason="SOAK_CANCEL")
                result.cancelled_from[state] += 1
                in_flight.pop(victim.request_id, None)

        if engine.has_unfinished_requests:
            step_started = perf_counter()
            engine.step()
            elapsed_ms = (perf_counter() - step_started) * 1000
            if getattr(engine, "last_step_prefill_tokens", 0):
                result._prefill_step_ms.append(elapsed_ms)
            elif getattr(engine, "last_step_decode_rows", 0):
                result._decode_step_ms.append(elapsed_ms)
            for phase, value in getattr(engine, "last_step_timing", {}).items():
                result._phase_ms.setdefault(phase, []).append(value)
            result.steps += 1
        elif not accepting:
            break
        if result.steps >= config.max_steps:
            result.violations.append(f"step budget {config.max_steps} exhausted")
            break

        for request_id in [rid for rid, r in in_flight.items() if r.done]:
            in_flight.pop(request_id)

        if result.steps % config.sample_every_steps == 0:
            stats = engine.stats_snapshot()
            result.peak_kv_utilization = max(
                result.peak_kv_utilization, float(stats["kv_utilization"])
            )
            result.peak_waiting = max(result.peak_waiting, int(stats["waiting_requests"]))
            result.peak_active = max(result.peak_active, int(stats["active_requests"]))
            if stats.get("decode_batch"):
                result._batch_samples.append(float(stats["decode_batch"]))
                result._context_samples.append(float(stats["decode_mean_context"]))

    result.wall_s = perf_counter() - started
    if result._batch_samples:
        result.mean_decode_batch = sum(result._batch_samples) / len(result._batch_samples)
        result.mean_context_tokens = sum(result._context_samples) / len(result._context_samples)
    for request in submitted:
        result.counts[request.state.name] += 1
        if request.finish_reason:
            result.finish_reasons[request.finish_reason] += 1

    produced = [r for r in submitted if r.state is RequestState.FINISHED]
    ttfts = [r.time_to_first_token_ms() for r in produced if r.time_to_first_token_ms()]
    # Percentiles must be taken over individual token gaps, not over per-request means.
    # Averaging inside each request first destroys the tail - a single 200 ms hiccup in a
    # 60-token response moves that request's mean by 3 ms and disappears. With ~20
    # requests per run, a "p99" over request means is really just the slowest request's
    # average, which is why it swung 70-90% between runs and never resolved.
    gaps_all: list[float] = []
    gaps_clean: list[float] = []
    for request in produced:
        stamps = request.token_timestamps_ns
        gaps = [(b - a) / 1_000_000 for a, b in zip(stamps, stamps[1:])]
        gaps_all.extend(gaps)
        if not request.preempted_count:
            # A preempted request has one gap containing its whole queue wait. That is a
            # real stall and is reported as stall_p99, but mixing it into the token-gap
            # tail would make every tail number a preemption detector.
            gaps_clean.extend(gaps)
    gaps = gaps_clean or gaps_all
    itls = [
        r.mean_inter_token_latency_ms() for r in produced
        if r.mean_inter_token_latency_ms()
    ]
    queues = [r.total_queue_time_ms() for r in produced if r.total_queue_time_ms()]
    stalls = [r.stall_time_ms() for r in produced]
    result.latency = {
        "ttft_p50": _percentile(ttfts, 0.50), "ttft_p99": _percentile(ttfts, 0.99),
        "itl_p50": _percentile(gaps, 0.50), "itl_p99": _percentile(gaps, 0.99),
        "itl_p999": _percentile(gaps, 0.999),
        "itl_samples": len(gaps),
        "itl_p99_including_preempted": _percentile(gaps_all, 0.99),
        # Retained for continuity with earlier runs: the median request's average gap.
        "itl_request_mean_p50": _percentile(itls, 0.50),
        "total_queue_p50": _percentile(queues, 0.50),
        "total_queue_p99": _percentile(queues, 0.99),
        "stall_p99": _percentile(stalls, 0.99),
        "stall_max": max(stalls) if stalls else 0.0,
    }
    result.delivered_tokens = sum(len(r.output_token_ids) for r in submitted)
    decode_ms, prefill_ms = result._decode_step_ms, result._prefill_step_ms
    total_steps = len(decode_ms) + len(prefill_ms)
    result.step_timing = {
        "decode_only_steps": len(decode_ms),
        "prefill_steps": len(prefill_ms),
        "fused_steps": int(getattr(engine, "fused_steps", 0)),
        "prefill_step_fraction": len(prefill_ms) / total_steps if total_steps else 0.0,
        "decode_step_p50_ms": _percentile(decode_ms, 0.50),
        "decode_step_p99_ms": _percentile(decode_ms, 0.99),
        "prefill_step_p50_ms": _percentile(prefill_ms, 0.50),
        "prefill_step_p99_ms": _percentile(prefill_ms, 0.99),
        # How much longer a sequence waits for its next token when the step it is riding
        # also carries someone else's prefill.
        "prefill_penalty_p50_ms": (
            _percentile(prefill_ms, 0.50) - _percentile(decode_ms, 0.50)
            if decode_ms and prefill_ms else 0.0
        ),
    }
    for phase, samples in sorted(result._phase_ms.items()):
        result.step_timing[f"{phase}_p50"] = _percentile(samples, 0.50)
        result.step_timing[f"{phase}_p99"] = _percentile(samples, 0.99)
    if decode_ms and prefill_ms:
        # Frequency-weighted decomposition of the average gap a caller experiences. The
        # penalty matters in proportion to how often a step carries prefill, so neither
        # the per-step cost nor the fraction means much alone.
        decode_p50 = _percentile(decode_ms, 0.50)
        penalty = _percentile(prefill_ms, 0.50) - decode_p50
        share = len(prefill_ms) / (len(decode_ms) + len(prefill_ms))
        expected = decode_p50 + share * penalty
        result.step_timing.update({
            "expected_gap_ms": expected,
            "expected_gap_from_decode_ms": decode_p50,
            "expected_gap_from_prefill_ms": share * penalty,
            "prefill_share_of_gap": (share * penalty) / expected if expected else 0.0,
        })
    result.recompute = dict(engine.recompute_report())
    result.prefix_cache = {
        k: v for k, v in engine.prefix_cache.snapshot().items()
        if k in {"hit_rate", "hits", "lookups", "evictions", "cached_blocks"}
    }
    result.violations.extend(check_invariants(engine, submitted))
    # A state the workload never entered cannot be cancelled from, and demanding it would
    # make the soak fail for the wrong reason. PREFILLING is only observable when prompts
    # exceed the prefill token budget; PREEMPTED only under real KV pressure.
    uncovered = [
        state for state in _CANCEL_TARGETS
        if result.states_observed[state] and not result.cancelled_from[state]
    ]
    if uncovered:
        result.coverage_gaps.append(
            f"states reached but never cancelled from: {uncovered}"
        )
    if not result.recompute.get("preemptions"):
        result.coverage_gaps.append("no preemption occurred: pool was not under pressure")
    if not result.counts["REJECTED"]:
        result.coverage_gaps.append("no admission rejection occurred")
    return result


@dataclass
class RepeatedResult:
    """Several seeded runs of one configuration, summarised with spread."""

    label: str
    runs: list[SoakResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(run.ok for run in self.runs)

    @property
    def coverage_gaps(self) -> list[str]:
        return sorted({gap for run in self.runs for gap in run.coverage_gaps})

    def series(self, path: str) -> list[float]:
        """Pull one metric from every run. Dotted path, e.g. 'latency.itl_p50'."""
        head, _, tail = path.partition(".")
        values = []
        for run in self.runs:
            source = getattr(run, head)
            value = source[tail] if tail else source
            if value is not None:
                values.append(float(value))
        return values

    def summary(self, path: str) -> dict[str, float]:
        values = self.series(path)
        if not values:
            return {"n": 0}
        median = _percentile(values, 0.5)
        low, high = min(values), max(values)
        return {
            "n": len(values), "median": median, "min": low, "max": high,
            # Relative spread. Above ~0.2 on shared hardware, treat differences between
            # configurations as unresolved rather than real.
            "spread": (high - low) / median if median else 0.0,
        }

    def to_dict(self) -> dict:
        metrics = [
            "latency.itl_p50", "latency.itl_p99", "latency.itl_samples",
            "latency.ttft_p50",
            "latency.total_queue_p50", "waste_ratio", "delivered_tokens",
            "peak_kv_utilization", "mean_decode_batch", "mean_context_tokens",
            "step_timing.decode_step_p50_ms", "step_timing.prefill_step_p50_ms",
            "step_timing.prefill_step_fraction", "step_timing.prefill_penalty_p50_ms",
            "step_timing.expected_gap_ms", "step_timing.prefill_share_of_gap",
            "step_timing.host_stage_ms_p50", "step_timing.decode_gpu_ms_p50",
            "step_timing.prefill_gpu_ms_p50", "step_timing.fused_gpu_ms_p50",
            "step_timing.sync_ms_p50",
        ]
        return {
            "label": self.label, "runs": len(self.runs), "ok": self.ok,
            "summary": {metric: self.summary(metric) for metric in metrics},
            "violations": [v for run in self.runs for v in run.violations],
            "coverage_gaps": self.coverage_gaps,
        }


def run_repeated(make_engine, config: SoakConfig, repeats: int = 5,
                 label: str = "") -> RepeatedResult:
    """Run one configuration several times on fresh engines, varying only the seed.

    A fresh engine per run matters: a reused pool carries prefix-cache entries and
    fragmentation from the previous run, which is exactly the state a comparison is
    trying to hold constant.
    """
    result = RepeatedResult(label=label or f"soak-{config.seed}")
    for offset in range(repeats):
        seeded = replace(config, seed=config.seed + offset)
        result.runs.append(run_soak(_warmed(make_engine()), seeded))
    return result


def _warmed(engine):
    """Pay graph capture and kernel JIT before the timed window, never inside it.

    `skip_benchmark_warmup` is set by an A/B arm that wants to measure exactly that
    first-use cost landing on live requests; everything else warms.
    """
    if hasattr(engine, "warmup") and not getattr(engine, "skip_benchmark_warmup", False):
        engine.warmup()
    return engine


def run_interleaved(makers: dict[str, object], config: SoakConfig, repeats: int = 5,
                    on_run=None) -> dict[str, RepeatedResult]:
    """Run several configurations A B A B ... rather than A A A B B B.

    On a shared, thermally limited GPU the run-to-run drift is monotone in time - clocks
    fall as the card heats. Blocking the arms puts all of that drift on one side of the
    comparison; interleaving them spreads it across both. Each configuration still sees
    the same seed sequence.
    """
    results = {label: RepeatedResult(label=label) for label in makers}
    for offset in range(repeats):
        seeded = replace(config, seed=config.seed + offset)
        for label, make_engine in makers.items():
            run = run_soak(_warmed(make_engine()), seeded)
            results[label].runs.append(run)
            if on_run is not None:
                on_run(label, offset, run)
    return results


def _build_engine(args) -> object:
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    return _warmed(ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device,
        num_blocks=args.num_blocks, block_size=16, max_active=args.max_active,
        prefix_cache_blocks=args.prefix_cache_blocks,
        cuda_graph_batch_sizes=(2, 4, 8, 16) if args.cuda_graphs else None,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--arrival-rate", type=float, default=20.0)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--max-active", type=int, default=16)
    parser.add_argument("--prefix-cache-blocks", type=int, default=64)
    parser.add_argument("--cancel-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--out", default="results/soak.json")
    args = parser.parse_args()

    engine = _build_engine(args)
    config = SoakConfig(
        duration_s=args.duration, arrival_rate_per_s=args.arrival_rate,
        cancel_probability=args.cancel_probability, seed=args.seed,
    )
    result = run_soak(engine, config)

    print(json.dumps(result.to_dict(), indent=2, default=str))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    print(f"\nSaved -> {out}")
    if result.coverage_gaps:
        print("\nCoverage gaps (workload, not correctness):")
        for gap in result.coverage_gaps:
            print(f"  - {gap}")
    if result.ok:
        print("\nSOAK PASS: engine drained with every KV page accounted for.")
        return 0
    print("SOAK FAIL:")
    for violation in result.violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Short, seeded soaks that must hold every reliability invariant.

The long-running version lives in `benchmarks/reliability/soak.py`; these are the same
harness bounded to a few seconds so they can gate a change. They deliberately assert
invariants and coverage, never timings, so they do not become flaky on shared hardware.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.reliability.soak import SoakConfig, check_invariants, run_soak

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MODEL_NAME = "Qwen/Qwen3-0.6B"


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def _engine(**overrides):
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    settings = dict(
        num_blocks=96, block_size=16, max_active=8, prefix_cache_blocks=16,
    )
    settings.update(overrides)
    return ContinuousBatchingEngine(model, tok, "cuda", **settings)


def _report(name, result):
    print(f"\n[{name}] {result.wall_s:.1f}s  steps={result.steps}  "
          f"submitted={result.submitted}  counts={dict(result.counts)}")
    print(f"  cancelled_from={dict(result.cancelled_from)}")
    print(f"  finish_reasons={dict(result.finish_reasons)}")
    print(f"  delivered_tokens={result.delivered_tokens}  waste_ratio={result.waste_ratio:.2f}")
    print(f"  latency_ms={result.latency}")
    print(f"  recompute={result.recompute}")
    print(f"  prefix_cache={result.prefix_cache}  peak_kv={result.peak_kv_utilization:.2f}")
    if result.violations:
        print(f"  VIOLATIONS={result.violations}")
    if result.coverage_gaps:
        print(f"  coverage_gaps={result.coverage_gaps}")


@cuda
@requires_cuda
def test_soak_mixed_arrivals_leaves_no_leaked_pages():
    """Baseline: random arrivals, shared prefixes, cancellations across the lifecycle.

    Asserts correctness only. Whether the random injector reaches every state in one run
    is a workload question, tested separately below.
    """
    result = run_soak(_engine(), SoakConfig(duration_s=6.0, arrival_rate_per_s=25.0, seed=1))
    _report("mixed", result)
    assert result.submitted > 0
    assert result.violations == []
    assert result.counts["FINISHED"] > 0
    assert result.counts["CANCELLED"] > 0
    assert result.prefix_cache["hits"] > 0, "shared prefixes never hit the cache"


@cuda
@requires_cuda
def test_soak_cancels_from_every_lifecycle_state_when_given_the_chance():
    """The coverage test proper: a workload built so every state is reachable.

    Long prompts keep requests in PREFILLING across steps, a tight pool produces parked
    PREEMPTED requests, oversubscription fills WAITING, and a high cancel probability
    gives the injector enough attempts to reach all four.
    """
    result = run_soak(
        _engine(num_blocks=40, max_active=6, prefix_cache_blocks=4),
        SoakConfig(duration_s=12.0, arrival_rate_per_s=20.0, seed=11,
                   cancel_probability=0.6, prefix_tokens=(160, 320),
                   suffix_tokens=(32, 128), max_new_tokens=(32, 96)),
    )
    _report("coverage", result)
    assert result.violations == []
    missed = [state for state in ("WAITING", "PREFILLING", "DECODING", "PREEMPTED")
              if not result.cancelled_from[state]]
    assert not missed, (
        f"never cancelled from {missed}; observed={dict(result.states_observed)}"
    )


@cuda
@requires_cuda
def test_soak_under_kv_pressure_preempts_and_still_accounts_for_every_page():
    """A pool far too small for the offered load: preemption is the normal path here."""
    result = run_soak(
        _engine(num_blocks=24, max_active=8, prefix_cache_blocks=4),
        SoakConfig(duration_s=8.0, arrival_rate_per_s=30.0, seed=2,
                   max_new_tokens=(8, 96)),
    )
    _report("pressure", result)
    assert result.violations == []
    assert result.recompute["preemptions"] > 0, "pool was not tight enough to preempt"
    assert result.counts["FINISHED"] > 0, "pressure starved every request"


@cuda
@requires_cuda
def test_soak_with_cuda_graphs_reserves_pages_without_leaking_them():
    """Graph dummy pages are permanent; they must show up as reserved, never as leaked."""
    engine = _engine(num_blocks=64, max_active=8, cuda_graph_batch_sizes=(2, 4, 8))
    assert engine.scheduler.reserved_blocks == len(engine._graph_dummy_blocks) == 7
    result = run_soak(engine, SoakConfig(duration_s=6.0, arrival_rate_per_s=25.0, seed=3))
    _report("graphs", result)
    assert result.violations == []
    assert engine._decode_graphs, "no graph bucket was ever captured"


@cuda
@requires_cuda
def test_soak_long_generations_never_fail_a_request_for_yielding_too_often():
    """The case the removed preemption-count limit would have failed.

    Long generations behind a tight pool make a queued request yield repeatedly through
    no fault of its own. Under Gate 1B that is survivable; under the old bound it was a
    spurious KV_POOL_EXHAUSTED.
    """
    result = run_soak(
        _engine(num_blocks=28, max_active=8, prefix_cache_blocks=4),
        SoakConfig(duration_s=10.0, arrival_rate_per_s=12.0, seed=4,
                   max_new_tokens=(128, 256), suffix_tokens=(4, 32)),
    )
    _report("long-generations", result)
    assert result.violations == []
    assert result.counts["FAILED"] == 0, (
        f"requests failed under pure pressure: {dict(result.finish_reasons)}"
    )
    peak_preemptions = max(
        (r for r in [result.recompute["preemptions"]]), default=0
    )
    assert peak_preemptions > 0


@cuda
@requires_cuda
def test_soak_rejects_impossible_requests_at_admission_not_after_work():
    """Effective capacity must reject before any KV is built, even with pages reserved."""
    engine = _engine(num_blocks=32, max_active=4, prefix_cache_blocks=4,
                     cuda_graph_batch_sizes=(2, 4))
    result = run_soak(
        engine,
        SoakConfig(duration_s=6.0, arrival_rate_per_s=20.0, seed=5,
                   prefix_tokens=(64, 200), suffix_tokens=(32, 200),
                   max_new_tokens=(64, 256)),
    )
    _report("oversized", result)
    assert result.violations == []
    assert result.counts["REJECTED"] > 0, "workload never exceeded effective capacity"
    assert result.finish_reasons["KV_CAPACITY_EXCEEDED"] > 0
    # Rejection is an admission decision, so no rejected request may have been admitted.
    assert result.finish_reasons.get("KV_POOL_EXHAUSTED", 0) == 0


@cuda
@requires_cuda
def test_invariant_checker_detects_a_deliberately_leaked_page():
    """Guard the guard: the audit must fail when a page really is leaked."""
    from engine.runtime import GenerationRequest

    engine = _engine(num_blocks=32, max_active=4, prefix_cache_blocks=0)
    request = GenerationRequest("leak", prompt_token_count=8, max_new_tokens=4,
                                prompt_token_ids=list(range(1000, 1008)))
    assert engine.submit(request)
    engine.step()
    # Strand the allocation: drop the request from the scheduler without releasing pages.
    engine.scheduler.active.pop(request.request_id, None)
    violations = check_invariants(engine, [request])
    assert any("leak" in v or "non-terminal" in v for v in violations), violations


@cuda
@requires_cuda
def test_soak_closed_loop_holds_every_invariant():
    """Closed loop: a fixed number in flight, so latency reflects the engine not the queue."""
    result = run_soak(
        _engine(num_blocks=128, max_active=8),
        SoakConfig(duration_s=6.0, concurrency=8, seed=6, cancel_probability=0.05),
    )
    _report("closed-loop", result)
    assert result.mode == "closed-loop"
    assert result.violations == []
    assert result.counts["FINISHED"] > 0
    assert result.delivered_tokens > 0
    # Closed loop is self-limiting, so queueing must not dominate as it does open-loop.
    assert result.peak_waiting <= 8


@cuda
@requires_cuda
def test_soak_bounded_queue_applies_backpressure_instead_of_growing_without_limit():
    """Open loop above service rate must reject, not accumulate an unbounded backlog.

    Every earlier soak left `max_waiting_requests` unset, so the queue grew without limit
    and the QUEUE_FULL path was never exercised at all.
    """
    engine = _engine(num_blocks=64, max_active=4, max_waiting_requests=16)
    result = run_soak(
        engine,
        SoakConfig(duration_s=6.0, arrival_rate_per_s=60.0, seed=7,
                   cancel_probability=0.02, max_new_tokens=(16, 64)),
    )
    _report("backpressure", result)
    assert result.violations == []
    assert result.peak_waiting <= 16, "queue exceeded its bound"
    assert result.finish_reasons["QUEUE_FULL"] > 0, "backpressure never engaged"
    # Backpressure is an admission decision: nothing may die after doing work.
    assert result.finish_reasons.get("KV_POOL_EXHAUSTED", 0) == 0


@cuda
@requires_cuda
def test_repeated_runs_report_spread_so_single_numbers_are_not_trusted():
    """Any metric compared between configurations must first be compared against noise."""
    from benchmarks.reliability.soak import run_repeated

    model, tok = _load()

    def make_engine():
        from engine.batching.continuous_batching import ContinuousBatchingEngine
        return ContinuousBatchingEngine(
            model, tok, "cuda", num_blocks=128, block_size=16, max_active=8,
            prefix_cache_blocks=16, max_waiting_requests=64,
        )

    repeated = run_repeated(
        make_engine,
        SoakConfig(duration_s=4.0, concurrency=8, seed=20, cancel_probability=0.02),
        repeats=3, label="spread-check",
    )
    for metric in ("latency.itl_p50", "latency.ttft_p50", "waste_ratio", "delivered_tokens"):
        print(f"  {metric:22s} {repeated.summary(metric)}")
    assert repeated.ok, [v for r in repeated.runs for v in r.violations]
    itl = repeated.summary("latency.itl_p50")
    assert itl["n"] == 3 and itl["median"] > 0
    assert 0.0 <= itl["spread"] < 10.0
    # Spread is the point of this test: a comparison must be able to see its own noise.
    assert "spread" in repeated.summary("delivered_tokens")


@cuda
@requires_cuda
def test_soak_records_the_decode_operating_point_for_roofline_comparison():
    """A latency number is uncomparable to any floor without batch and context alongside it."""
    result = run_soak(
        _engine(num_blocks=160, max_active=8),
        SoakConfig(duration_s=6.0, concurrency=8, seed=30, cancel_probability=0.0,
                   prefix_tokens=(96, 160), suffix_tokens=(16, 64), max_new_tokens=(32, 64)),
    )
    _report("operating-point", result)
    print(f"  decode batch ~{result.mean_decode_batch:.1f}  "
          f"context ~{result.mean_context_tokens:.0f} tokens")
    assert result.violations == []
    assert 0 < result.mean_decode_batch <= 8
    # Prompts are 112-224 tokens before generation, so the mean context must land in a
    # plausible band - a zero or a wild value means the sampler is reading the wrong thing.
    assert 50 < result.mean_context_tokens < 400


@cuda
@requires_cuda
def test_itl_percentiles_are_taken_over_token_gaps_not_request_means():
    """A tail computed from per-request averages is not a tail.

    Averaging inside each request first hides every hiccup: one 200 ms stall in a 60-token
    response moves that request's mean by 3 ms. With a few dozen requests per run, a
    percentile over those means is effectively the slowest request's average.
    """
    result = run_soak(
        _engine(num_blocks=160, max_active=8),
        SoakConfig(duration_s=6.0, concurrency=8, seed=31, cancel_probability=0.0,
                   max_new_tokens=(24, 64)),
    )
    _report("itl-tail", result)
    latency = result.latency
    print(f"  itl p50={latency['itl_p50']:.2f} p99={latency['itl_p99']:.2f} "
          f"p999={latency['itl_p999']:.2f} over {latency['itl_samples']} gaps "
          f"(request-mean p50={latency['itl_request_mean_p50']:.2f})")
    assert result.violations == []
    # Percentiles over gaps need far more samples than there are requests; that is the
    # entire point of the change.
    assert latency["itl_samples"] > result.counts["FINISHED"] * 5
    assert latency["itl_p50"] > 0
    assert latency["itl_p99"] >= latency["itl_p50"]
    assert latency["itl_p999"] >= latency["itl_p99"]


@cuda
@requires_cuda
def test_prefill_steps_are_measured_separately_from_decode_steps():
    """Measure the prefill interruption instead of inferring it from a skewed tail.

    A step that also prefills advances every decoding sequence *and* runs a prefill
    forward, so each of those sequences waits longer for its next token. Percentiles over
    token gaps blend the two; this separates them.
    """
    result = run_soak(
        _engine(num_blocks=160, max_active=8, cuda_graph_batch_sizes=(1, 2, 4, 8)),
        SoakConfig(duration_s=8.0, concurrency=8, seed=40, cancel_probability=0.0,
                   max_new_tokens=(24, 64)),
    )
    _report("step-kinds", result)
    timing = result.step_timing
    print(f"  decode-only steps={timing['decode_only_steps']} "
          f"p50={timing['decode_step_p50_ms']:.2f}ms p99={timing['decode_step_p99_ms']:.2f}ms")
    print(f"  prefill steps={timing['prefill_steps']} "
          f"({timing['prefill_step_fraction']:.1%}) "
          f"p50={timing['prefill_step_p50_ms']:.2f}ms p99={timing['prefill_step_p99_ms']:.2f}ms")
    print(f"  prefill penalty p50={timing['prefill_penalty_p50_ms']:.2f}ms")
    assert result.violations == []
    assert timing["decode_only_steps"] > 0 and timing["prefill_steps"] > 0
    # A prefill-carrying step does strictly more work than a decode-only step.
    assert timing["prefill_step_p50_ms"] > timing["decode_step_p50_ms"]
    # The decode-only p50 is the engine's true per-step cost; it must be close to the
    # token-gap median, which is what a caller sees on an uninterrupted step.
    assert timing["decode_step_p50_ms"] == pytest.approx(result.latency["itl_p50"], rel=0.5)

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
    print(f"  latency_ms={result.latency}")
    print(f"  recompute={result.recompute}")
    print(f"  prefix_cache={result.prefix_cache}  peak_kv={result.peak_kv_utilization:.2f}")
    if result.violations:
        print(f"  VIOLATIONS={result.violations}")


@cuda
@requires_cuda
def test_soak_mixed_arrivals_leaves_no_leaked_pages():
    """Baseline: random arrivals, shared prefixes, cancellations from every state."""
    result = run_soak(_engine(), SoakConfig(duration_s=6.0, arrival_rate_per_s=25.0, seed=1))
    _report("mixed", result)
    assert result.submitted > 0
    assert result.violations == []
    assert result.counts["FINISHED"] > 0
    assert result.counts["CANCELLED"] > 0
    assert result.prefix_cache["hits"] > 0, "shared prefixes never hit the cache"


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

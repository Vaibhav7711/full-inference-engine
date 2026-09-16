import pytest

pytest_approx = pytest.approx

from engine.runtime import GenerationRequest, RequestState


def test_request_lifecycle_and_token_limit() -> None:
    request = GenerationRequest("r1", prompt_token_count=4, max_new_tokens=2)
    assert request.reserved_tokens == 6
    request.transition(RequestState.PREFILLING)
    assert request.queue_time_ms() is not None
    request.advance_prefill(4)
    request.transition(RequestState.DECODING)
    request.append_token(42)
    assert request.time_to_first_token_ms() is not None
    request.append_token(43)
    assert request.generation_time_ms() is not None
    with pytest.raises(RuntimeError, match="max_new_tokens"):
        request.append_token(44)
    request.transition(RequestState.FINISHED, reason="LENGTH")
    assert request.finish_reason == "LENGTH"


def test_request_rejects_invalid_transition() -> None:
    request = GenerationRequest("r1", prompt_token_count=1, max_new_tokens=1)
    with pytest.raises(RuntimeError, match="invalid request transition"):
        request.transition(RequestState.DECODING)


def test_request_tracks_partial_prefill_progress() -> None:
    request = GenerationRequest("chunked", prompt_token_count=10, max_new_tokens=2)
    request.transition(RequestState.PREFILLING)
    request.advance_prefill(4)
    assert request.prefilled_token_count == 4
    assert request.remaining_prefill_tokens == 6
    with pytest.raises(RuntimeError, match="fully prefetched"):
        request.transition(RequestState.DECODING)
    assert request.state is RequestState.PREFILLING
    request.advance_prefill(6)
    request.transition(RequestState.DECODING)


def test_request_exposes_manager_owned_block_table() -> None:
    from engine.cache import KVBlockManager

    request = GenerationRequest(
        "r1", prompt_token_count=3, max_new_tokens=2, prompt_token_ids=[1, 2, 3]
    )
    manager = KVBlockManager(num_blocks=4, block_size_tokens=2)
    request.allocation = manager.reserve("r1", 3, sequence_length=3)
    assert request.block_table == request.allocation.physical_block_ids


def test_preempt_keeps_generated_tokens_and_resumes_from_pending_input() -> None:
    from engine.runtime import GenerationRequest, RequestState

    request = GenerationRequest("r-preempt", prompt_token_count=3, max_new_tokens=8,
                                prompt_token_ids=[1, 2, 3])
    request.transition(RequestState.PREFILLING)
    request.advance_prefill(3)
    request.transition(RequestState.DECODING)
    for token in (10, 11, 12):
        request.append_token(token)
    first_admission = request.admitted_ns

    request.preempt()
    assert request.state is RequestState.WAITING
    assert request.resuming and request.preempted_count == 1
    assert request.allocation is None and request.prefilled_token_count == 0
    # KV must be rebuilt for prompt + generated[:-1]; the last token is the pending input.
    assert request.prefill_token_ids == [1, 2, 3, 10, 11]
    assert request.prefill_token_count == 5 and request.remaining_prefill_tokens == 5
    assert request.output_token_ids == [10, 11, 12]
    assert request.finish_reason is None

    request.transition(RequestState.PREFILLING)
    assert request.admitted_ns == first_admission  # queue metrics keep first admission
    request.advance_prefill(5)
    request.transition(RequestState.DECODING)
    request.complete_resumption()
    assert request.state is RequestState.DECODING
    assert not request.resuming
    assert request.prefilled_token_count == request.prompt_token_count
    assert request.remaining_prefill_tokens == 0


def test_preempt_before_any_output_replays_the_prompt() -> None:
    from engine.runtime import GenerationRequest, RequestState

    request = GenerationRequest("r-early", prompt_token_count=2, max_new_tokens=4,
                                prompt_token_ids=[7, 8])
    request.transition(RequestState.PREFILLING)
    request.advance_prefill(1)
    request.preempt()
    assert not request.resuming
    assert request.prefill_token_ids == [7, 8] and request.remaining_prefill_tokens == 2


def test_preempt_requires_an_active_state() -> None:
    import pytest

    from engine.runtime import GenerationRequest

    request = GenerationRequest("r-waiting", prompt_token_count=1, max_new_tokens=1)
    with pytest.raises(RuntimeError):
        request.preempt()


def _preempted_request():
    """A request that emits a token, yields, is rebuilt, and finishes."""
    import time

    from engine.runtime import GenerationRequest, RequestState

    request = GenerationRequest("r-metrics", prompt_token_count=4, max_new_tokens=8,
                                prompt_token_ids=[1, 2, 3, 4])
    time.sleep(0.01)                      # queueing before first admission
    request.transition(RequestState.PREFILLING)
    request.advance_prefill(4)
    request.transition(RequestState.DECODING)
    request.append_token(10)
    request.append_token(11)
    request.preempt()
    time.sleep(0.02)                      # parked in the queue after yielding
    request.transition(RequestState.PREFILLING)
    time.sleep(0.01)                      # rebuilding KV
    request.advance_prefill(request.remaining_prefill_tokens)
    request.transition(RequestState.DECODING)
    request.complete_resumption()
    request.append_token(12)
    return request


def test_preemption_wait_is_reported_in_total_queue_time_not_hidden() -> None:
    request = _preempted_request()
    first = request.queue_time_ms()
    total = request.total_queue_time_ms()
    # queue_time_ms is time to *first* admission and must not move when a request yields.
    assert first is not None and first >= 10
    assert total >= first + 20
    assert total == pytest_approx(first + request.preempted_wait_time_ms())


def test_generation_time_keeps_the_stall_and_decode_time_removes_it() -> None:
    request = _preempted_request()
    wall = request.generation_time_ms()
    stall = request.stall_time_ms()
    decode = request.decode_time_ms()
    # The stall covers both the parked wait and the rebuild, and both fall between the
    # first and last token, so wall clock must contain them and decode time must not.
    assert stall >= 30
    assert wall >= stall
    assert decode == pytest_approx(wall - stall)
    assert decode < stall  # three tokens of real decoding, ~30ms of stall


def test_mean_inter_token_latency_excludes_the_preemption_gap() -> None:
    request = _preempted_request()
    mean_itl = request.mean_inter_token_latency_ms()
    assert mean_itl is not None
    # Raw gaps contain one ~30ms hole; the reported mean must not be dominated by it.
    raw_gaps = [
        (b - a) / 1_000_000
        for a, b in zip(request.token_timestamps_ns, request.token_timestamps_ns[1:])
    ]
    assert max(raw_gaps) >= 30
    assert mean_itl < max(raw_gaps)
    assert mean_itl == pytest_approx(request.decode_time_ms() / 2)


def test_latency_report_exposes_every_field_a_soak_needs() -> None:
    report = _preempted_request().latency_report()
    assert set(report) == {
        "queue_ms", "total_queue_ms", "ttft_ms", "generation_ms",
        "decode_ms", "stall_ms", "mean_itl_ms", "output_tokens",
    }
    assert report["output_tokens"] == 3
    assert report["total_queue_ms"] > report["queue_ms"]
    assert report["generation_ms"] > report["decode_ms"]


def test_an_unstarted_request_reports_no_latency_rather_than_zero() -> None:
    from engine.runtime import GenerationRequest

    request = GenerationRequest("r-fresh", prompt_token_count=2, max_new_tokens=2,
                                prompt_token_ids=[1, 2])
    report = request.latency_report()
    assert report["queue_ms"] is None and report["total_queue_ms"] is None
    assert report["ttft_ms"] is None and report["decode_ms"] is None
    assert report["stall_ms"] == 0.0
    assert report["mean_itl_ms"] is None


def test_terminal_transition_drops_the_allocation_handle() -> None:
    """A finished request must not keep pointing at pages that now belong elsewhere."""
    from engine.cache import KVBlockManager
    from engine.runtime import GenerationRequest, RequestState

    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    request = GenerationRequest("r-terminal", prompt_token_count=4, max_new_tokens=2,
                                prompt_token_ids=[1, 2, 3, 4])
    request.allocation = manager.reserve("r-terminal", 4)
    request.transition(RequestState.PREFILLING)
    request.advance_prefill(4)
    request.transition(RequestState.DECODING)
    assert request.allocation is not None and request.block_table

    request.transition(RequestState.FINISHED, reason="LENGTH")
    assert request.allocation is None
    assert request.block_table == []


def test_held_pages_at_exit_defaults_false_until_the_scheduler_sets_it() -> None:
    from engine.runtime import GenerationRequest

    request = GenerationRequest("r-flag", prompt_token_count=1, max_new_tokens=1)
    assert request.held_pages_at_exit is False

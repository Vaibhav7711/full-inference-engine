import pytest

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

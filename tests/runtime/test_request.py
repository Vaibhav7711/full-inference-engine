import pytest

from engine.runtime import GenerationRequest, RequestState


def test_request_lifecycle_and_token_limit() -> None:
    request = GenerationRequest("r1", prompt_token_count=4, max_new_tokens=2)
    assert request.reserved_tokens == 6
    request.transition(RequestState.PREFILLING)
    assert request.queue_time_ms() is not None
    request.transition(RequestState.DECODING)
    request.append_token(42)
    request.append_token(43)
    with pytest.raises(RuntimeError, match="max_new_tokens"):
        request.append_token(44)
    request.transition(RequestState.FINISHED, reason="LENGTH")
    assert request.finish_reason == "LENGTH"


def test_request_rejects_invalid_transition() -> None:
    request = GenerationRequest("r1", prompt_token_count=1, max_new_tokens=1)
    with pytest.raises(RuntimeError, match="invalid request transition"):
        request.transition(RequestState.DECODING)


def test_request_exposes_manager_owned_block_table() -> None:
    from engine.cache import KVBlockManager

    request = GenerationRequest(
        "r1", prompt_token_count=3, max_new_tokens=2, prompt_token_ids=[1, 2, 3]
    )
    manager = KVBlockManager(num_blocks=4, block_size_tokens=2)
    request.allocation = manager.reserve("r1", 3, sequence_length=3)
    assert request.block_table == request.allocation.physical_block_ids

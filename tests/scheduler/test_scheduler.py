from engine.cache import KVBlockManager
from engine.runtime import GenerationRequest, RequestState
from engine.scheduler import FCFSScheduler


def make_scheduler(capacity: int = 10) -> FCFSScheduler:
    return FCFSScheduler(KVBlockManager(num_blocks=capacity, block_size_tokens=1))


def test_fcfs_admission_and_release() -> None:
    scheduler = make_scheduler()
    first = GenerationRequest("first", 2, 3)
    second = GenerationRequest("second", 2, 3)
    scheduler.submit(first)
    scheduler.submit(second)
    assert [request.request_id for request in scheduler.admit_available()] == ["first", "second"]
    scheduler.mark_decoding("first")
    scheduler.finish("first")
    assert first.state is RequestState.FINISHED
    # Paging reserves prompt blocks lazily; output capacity grows during decode.
    assert scheduler.block_manager.snapshot()["free_blocks"] == 8


def test_oversized_request_is_rejected_without_blocking_queue() -> None:
    scheduler = make_scheduler()
    oversized = GenerationRequest("oversized", 8, 3)
    valid = GenerationRequest("valid", 2, 2)
    scheduler.submit(oversized)
    scheduler.submit(valid)
    assert [request.request_id for request in scheduler.admit_available()] == ["valid"]
    assert oversized.state is RequestState.REJECTED
    assert oversized.finish_reason == "KV_CAPACITY_EXCEEDED"

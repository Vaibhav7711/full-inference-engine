from engine.cache import KVBlockManager, PrefixCache
from engine.runtime import GenerationRequest, RequestState, SamplingParams
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
    assert first.allocation.sequence_length == 0
    first.advance_prefill(first.prompt_token_count)
    scheduler.mark_decoding("first")
    scheduler.finish("first")
    assert first.state is RequestState.FINISHED
    # Admission owns one block; prompt/output capacity grows only as work executes.
    assert scheduler.block_manager.snapshot()["free_blocks"] == 9


def test_oversized_request_is_rejected_without_blocking_queue() -> None:
    scheduler = make_scheduler()
    oversized = GenerationRequest("oversized", 8, 3)
    valid = GenerationRequest("valid", 2, 2)
    scheduler.submit(oversized)
    scheduler.submit(valid)
    assert [request.request_id for request in scheduler.admit_available()] == ["valid"]
    assert oversized.state is RequestState.REJECTED
    assert oversized.finish_reason == "KV_CAPACITY_EXCEEDED"


def test_active_failure_releases_blocks() -> None:
    scheduler = make_scheduler()
    request = GenerationRequest("failed", 3, 2)
    scheduler.submit(request)
    scheduler.admit_available()
    scheduler.fail("failed", "KV_POOL_EXHAUSTED")
    assert request.state is RequestState.FAILED
    assert request.finish_reason == "KV_POOL_EXHAUSTED"
    assert scheduler.block_manager.snapshot()["free_blocks"] == 10


def test_cancel_partial_prefill_releases_blocks() -> None:
    scheduler = make_scheduler()
    request = GenerationRequest("cancelled", 6, 2)
    scheduler.submit(request)
    scheduler.admit_available()
    assert scheduler.block_manager.append_tokens(request.request_id, 3)
    request.advance_prefill(3)
    scheduler.cancel(request.request_id)
    assert request.state is RequestState.CANCELLED
    assert scheduler.block_manager.snapshot()["free_blocks"] == 10


def test_waiting_queue_applies_backpressure() -> None:
    scheduler = FCFSScheduler(
        KVBlockManager(num_blocks=10, block_size_tokens=1), max_waiting_requests=1
    )
    accepted = GenerationRequest("accepted", 2, 1)
    rejected = GenerationRequest("rejected", 2, 1)
    assert scheduler.submit(accepted)
    assert not scheduler.submit(rejected)
    assert rejected.state is RequestState.REJECTED
    assert rejected.finish_reason == "QUEUE_FULL"


def test_prefill_planner_is_token_bounded_and_round_robin() -> None:
    scheduler = make_scheduler(capacity=30)
    requests = [GenerationRequest(f"r{i}", 6, 1) for i in range(3)]
    for request in requests:
        scheduler.submit(request)
    scheduler.admit_available()

    first = scheduler.plan_prefill(chunk_size=3, token_budget=4)
    second = scheduler.plan_prefill(chunk_size=3, token_budget=4)
    assert [(request.request_id, count) for request, count in first] == [
        ("r0", 3), ("r1", 1)
    ]
    assert [(request.request_id, count) for request, count in second] == [
        ("r2", 3), ("r0", 1)
    ]


def test_admission_attaches_longest_cached_prefix() -> None:
    manager = KVBlockManager(num_blocks=12, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=8)
    source = manager.reserve("source", 8, sequence_length=8)
    assert source is not None
    source_blocks = list(source.physical_block_ids)
    cache.publish(list(range(8)), source)
    manager.release("source")
    scheduler = FCFSScheduler(manager, prefix_cache=cache)
    request = GenerationRequest(
        "hit", 9, 2, prompt_token_ids=list(range(9))
    )
    scheduler.submit(request)
    assert scheduler.admit_available() == [request]
    assert request.prefilled_token_count == 8
    assert request.cached_prefix_tokens == 8
    assert request.block_table == source_blocks
    scheduler.cancel(request.request_id)
    assert manager.snapshot()["used_blocks"] == cache.snapshot()["cached_blocks"]


def test_sampled_admission_bypasses_cached_token_but_reuses_kv_blocks() -> None:
    manager = KVBlockManager(num_blocks=12, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=8)
    source = manager.reserve("source", 6, sequence_length=6)
    assert source is not None
    cache.publish(list(range(6)), source, next_token_id=42)
    source_block = source.physical_block_ids[0]
    manager.release("source")
    scheduler = FCFSScheduler(manager, prefix_cache=cache)
    request = GenerationRequest(
        "sampled", 6, 2, prompt_token_ids=list(range(6)),
        sampling=SamplingParams(temperature=0.8, seed=7),
    )

    scheduler.submit(request)
    assert scheduler.admit_available() == [request]

    assert request.cached_next_token_id is None
    assert request.prefilled_token_count == 4
    assert request.cached_prefix_tokens == 4
    assert request.block_table == [source_block]

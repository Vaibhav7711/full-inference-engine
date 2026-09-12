import torch

from engine.batching import ContinuousBatcher
from engine.cache import ContiguousKVAllocator, KVCacheGeometry
from engine.runtime import GenerationRequest, RequestState
from engine.scheduler import FCFSScheduler


def make_batcher(max_active_requests: int = 2) -> ContinuousBatcher:
    allocator = ContiguousKVAllocator(100, KVCacheGeometry(1, 1, 1, torch.float16))
    return ContinuousBatcher(FCFSScheduler(allocator), max_active_requests)


def test_continuous_batcher_refills_after_request_completes() -> None:
    batcher = make_batcher()
    first = GenerationRequest("first", 2, 2)
    second = GenerationRequest("second", 2, 2)
    third = GenerationRequest("third", 2, 2)
    for request in (first, second, third):
        batcher.submit(request)

    first_plan = batcher.plan_next_iteration()
    assert first_plan.prefill_request_ids == ("first", "second")
    assert first_plan.decode_request_ids == ()
    batcher.mark_prefill_complete("first")
    batcher.mark_prefill_complete("second")

    decode_plan = batcher.plan_next_iteration()
    assert decode_plan.decode_request_ids == ("first", "second")
    batcher.record_decoded_token("first", 1, terminal=True)
    assert first.state is RequestState.FINISHED

    refill_plan = batcher.plan_next_iteration()
    assert refill_plan.prefill_request_ids == ("third",)
    assert refill_plan.decode_request_ids == ("second",)
    assert batcher.snapshot()["mean_batch_occupancy"] == 2.0


def test_prefill_request_does_not_decode_in_the_same_plan() -> None:
    batcher = make_batcher(max_active_requests=1)
    batcher.submit(GenerationRequest("one", 1, 1))
    plan = batcher.plan_next_iteration()
    assert plan.prefill_request_ids == ("one",)
    assert plan.decode_request_ids == ()

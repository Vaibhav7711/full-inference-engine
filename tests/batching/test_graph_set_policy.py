"""The graph-set policy surface: admission by max_model_len, replay accounting, row filter.

These run against the engine class with a fake `self` (the pattern in test_static_utils),
so they need the module importable - Triton present - but no GPU and no model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("triton")

from engine.batching.continuous_batching import ContinuousBatchingEngine  # noqa: E402
from engine.runtime import GenerationRequest  # noqa: E402


class _Scheduler:
    def __init__(self):
        self.submitted = []

    def submit(self, request):
        self.submitted.append(request.request_id)
        return True


class _Fake:
    pass


def _request(prompt: int, new: int, rid: str = "r") -> GenerationRequest:
    return GenerationRequest(request_id=rid, prompt_token_count=prompt, max_new_tokens=new,
                             prompt_token_ids=list(range(prompt)))


def test_submit_refuses_a_request_that_could_outgrow_max_model_len():
    fake = _Fake()
    fake.max_model_len = 1024
    fake.scheduler = _Scheduler()
    assert ContinuousBatchingEngine.submit(fake, _request(900, 100, "fits")) is True
    with pytest.raises(ValueError, match="needs 1025 tokens .* max_model_len is 1024"):
        ContinuousBatchingEngine.submit(fake, _request(900, 125, "over"))
    assert fake.scheduler.submitted == ["fits"]


def test_submit_without_a_cap_admits_anything():
    fake = _Fake()
    fake.max_model_len = None
    fake.scheduler = _Scheduler()
    assert ContinuousBatchingEngine.submit(fake, _request(50_000, 1_000)) is True


def test_replay_report_names_graphs_never_replayed_and_lazy_captures():
    fake = _Fake()
    fake._decode_graphs = {(4, 128, 4): object(), (8, 128, 4): object()}
    fake._prefill_graphs = {(1, "sdpa", 256): object()}
    fake._fused_graphs = {(4, 1, "sdpa", 256, 128, 4): object()}
    fake.graph_replays = {}
    fake.lazy_graph_capture_keys = []
    note = ContinuousBatchingEngine._note_replay
    note(fake, ("decode", 4, 128, 4))
    note(fake, ("decode", 4, 128, 4))
    note(fake, ("prefill", 1, "sdpa", 256))
    fake.lazy_graph_capture_keys.append(("fused", 8, 2, "sdpa", 512, 128, 4))
    report = ContinuousBatchingEngine.graph_replay_report(fake)
    assert report["captured"] == 4
    assert report["replayed_keys"] == 2
    assert report["never_replayed"] == 2
    assert set(report["never_replayed_keys"]) == {("decode", 8, 128, 4),
                                                  ("fused", 4, 1, "sdpa", 256, 128, 4)}
    assert report["replays_total"] == 3
    assert report["lazy_captures"] == 1
    assert report["lazy_capture_keys"] == [("fused", 8, 2, "sdpa", 512, 128, 4)]


def test_prefill_row_bucket_honours_the_graph_row_limit():
    fake = _Fake()
    fake.cuda_graph_batch_sizes = (1, 2, 4, 8, 16)
    fake.max_active = 16
    bucket = ContinuousBatchingEngine._prefill_row_bucket
    assert bucket(fake, 3, 4) == 4
    assert bucket(fake, 5, 4) is None          # wider than the limit: eager, no capture
    assert bucket(fake, 5, None) == 8          # no limit: the next bucket up


def test_release_drops_graphs_pools_and_staged_buffers():
    import torch

    fake = _Fake()
    fake.device = "cpu"
    fake._decode_graphs = {("decode", 1): object()}
    fake._prefill_graphs = {}
    fake._fused_graphs = {("fused", 2): object()}
    fake.key_pool = [torch.zeros(4)]
    fake.value_pool = [torch.zeros(4)]
    fake._device_input_ids = torch.zeros(2, dtype=torch.long)
    fake._prefill_device_starts = torch.zeros(2, dtype=torch.int32)
    fake.block_manager = object()           # untouched: not device memory
    ContinuousBatchingEngine.release(fake)
    assert fake._decode_graphs == {} and fake._fused_graphs == {}
    assert fake.key_pool == [] and fake.value_pool == []
    assert fake._device_input_ids is None and fake._prefill_device_starts is None
    assert fake.block_manager is not None

"""Searched INT8 decode launches and the paired kernel reproduce the current INT8 kernel."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
@pytest.mark.parametrize("batch,context", [(1, 300), (1, 2049), (8, 700), (16, 129)])
@pytest.mark.parametrize("launch", [("per_head", 256, 4, 3), ("per_head", 64, 2, 2), ("paired", 64, 2, 1)])
def test_tuned_int8_decode_matches_current_kernel(batch, context, launch) -> None:
    from engine.kernels.int8_decode_tuned import paged_decode_int8_paired, paged_decode_int8_per_head
    from engine.kernels.int8_paged_kv import paged_decode_batched_int8

    torch.manual_seed(batch * 7 + context)
    blocks_per_row = -(-(context + 1) // 16)
    num_blocks = batch * blocks_per_row + 2
    kp = torch.randint(-127, 128, (num_blocks, 16, 8, 128), device="cuda", dtype=torch.int8)
    vp = torch.randint(-127, 128, kp.shape, device="cuda", dtype=torch.int8)
    ks = (torch.rand(kp.shape[:-1], device="cuda") * 0.02 + 0.002).half()
    vs = (torch.rand(kp.shape[:-1], device="cuda") * 0.02 + 0.002).half()
    tables = torch.randperm(num_blocks, device="cuda")[:batch * blocks_per_row].view(batch, -1).to(torch.int32)
    lengths = torch.randint(max(1, context - 50), context + 1, (batch,), device="cuda", dtype=torch.int32)
    query = torch.randn(batch, 16, 1, 128, device="cuda", dtype=torch.float16)
    smoothing = torch.rand(8, 64, device="cuda") * 3 + 0.5
    smoothing = torch.cat((smoothing, smoothing), -1).contiguous()
    expected = paged_decode_batched_int8(query, kp, vp, ks, vs, tables, lengths, length_offset=1,
                                         key_smoothing=smoothing)
    kind, block_n, warps, stages = launch
    fn = paged_decode_int8_paired if kind == "paired" else paged_decode_int8_per_head
    actual = fn(query, kp, vp, ks, vs, tables, lengths, block_n=block_n, num_warps=warps,
                num_stages=stages, length_offset=1, key_smoothing=smoothing)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)


@cuda
@requires_cuda
def test_engine_int8_decode_tuned_keeps_leading_tokens() -> None:
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.batching.continuous_batching import ContinuousBatchingEngine

    # A local model, freed below: importing the shared loader from another test module
    # pinned extra model copies for the rest of the CUDA session.
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16,
                                                 device_map="cuda").eval()
    prompts = ["Explain how a paged KV cache differs from a contiguous one. " * 4]
    outputs = []
    for tuned in (False, True):
        eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=128, max_active=1,
                                       kv_cache_dtype="int8", int8_decode_tuned=tuned,
                                       cuda_graph_batch_sizes=(1,), prefix_cache_blocks=0)
        eng.warmup()
        outputs.append(eng.generate(prompts, max_new_tokens=24)[0])
        assert eng.lazy_graph_captures == 0
        del eng
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    assert outputs[0][:16] == outputs[1][:16]

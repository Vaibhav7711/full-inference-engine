"""CUDA-Graph capture for a fixed-width replay bucket of paged decode work."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PagedDecodeGraph:
    """Captured model forward over an engine's persistent decode metadata buffers."""

    graph: torch.cuda.CUDAGraph
    batch_size: int
    block_n: int
    num_warps: int
    logits: torch.Tensor

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.logits


def capture_paged_decode_graph(engine, *, batch_size: int, block_n: int, num_warps: int) -> PagedDecodeGraph:
    """Capture one fixed-width paged-decode model forward.

    The engine's decode metadata tensors have fixed addresses for its lifetime.  Callers
    update their contents before replaying, but must use the same batch width and kernel
    regime as capture.  The capture forward writes the pending K/V slot once; replaying
    immediately afterwards overwrites that same slot before request state is advanced.
    """
    if not torch.cuda.is_available() or engine.device != "cuda":
        raise ValueError("paged decode graph capture requires a CUDA engine")
    if not 0 < batch_size <= engine.max_active:
        raise ValueError("batch_size must be within the engine's active-batch limit")
    if block_n not in {64, 128} or num_warps not in {4, 8}:
        raise ValueError("unsupported paged decode graph kernel regime")

    from engine.batching.continuous_batching import _BatchContext, _clear_batch_ctx, _set_batch_ctx

    inputs = engine._device_input_ids[:batch_size]
    positions = engine._device_position_ids[:batch_size]
    tables = engine._device_block_tables[:batch_size]
    lengths = engine._device_seq_lens[:batch_size]
    context = _BatchContext(
        key_pool=engine.key_pool, value_pool=engine.value_pool, block_tables=tables,
        seq_lens=lengths, block_size=engine.block_size, decode_block_n=block_n,
        decode_num_warps=num_warps, key_scale_pool=engine.key_scale_pool,
        value_scale_pool=engine.value_scale_pool,
    )
    engine.model.config._attn_implementation = engine.ATTN_NAME
    if hasattr(engine.model.config, "_attn_implementation_internal"):
        engine.model.config._attn_implementation_internal = engine.ATTN_NAME

    # Compile and initialize allocations outside capture.
    _set_batch_ctx(context)
    try:
        with torch.inference_mode():
            engine.model(input_ids=inputs, position_ids=positions, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            outputs = engine.model(input_ids=inputs, position_ids=positions, use_cache=False, return_dict=True)
            logits = outputs.logits
    finally:
        _clear_batch_ctx()
    return PagedDecodeGraph(graph, batch_size, block_n, num_warps, logits)

"""CUDA-Graph capture for a fixed-shape chunked prefill forward.

Decode graphs removed ~24 ms of launch overhead per step on the T4 (36 -> 8.2 ms). The
chunked prefill forward is the same un-graphed HF forward, launch-bound in the same way -
the Phase B fit put its fixed cost at ~21 ms per invocation - and it has had no graph
until now. A chunk batch is staged into the engine's persistent prefill buffers exactly as
a decode batch is, so the forward can be captured once per (row bucket, attention kind,
gathered-context bucket) and replayed against new contents.

Capture runs on a batch of inert rows (chunk length 0): the write kernels store nothing,
the attention kernels visit no keys, and the vocabulary projection reads position 0. No
request's KV is touched, so capture is safe at any point, including warm-up on an empty
engine and lazily under load.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PagedPrefillGraph:
    graph: torch.cuda.CUDAGraph
    rows: int
    attention: str
    context_len: int
    logits: torch.Tensor    # [rows, vocab], a view of the engine's shared buffer

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.logits


def capture_paged_prefill_graph(
    engine, *, rows: int, context_len: int, pool=None,
) -> PagedPrefillGraph:
    """Capture one chunked prefill forward at `rows` x `engine.prefill_chunk_size`.

    `context_len` is the gathered-prefix length for the SDPA path (0 for the Triton
    kernels, which read their lengths from the device tensors at replay). `pool` is a
    `torch.cuda.graph_pool_handle()` shared by graphs that never run concurrently, so a
    dozen buckets do not each keep private copies of the same activation buffers.
    """
    if not torch.cuda.is_available() or torch.device(engine.device).type != "cuda":
        raise ValueError("prefill graph capture requires a CUDA engine")
    if not 0 < rows <= engine.max_active:
        raise ValueError("rows must be within the engine's active-batch limit")

    from engine.batching.continuous_batching import _clear_prefill_ctx

    width = engine.prefill_chunk_size
    engine.model.config._attn_implementation = engine.PREFILL_ATTN_NAME
    if hasattr(engine.model.config, "_attn_implementation_internal"):
        engine.model.config._attn_implementation_internal = engine.PREFILL_ATTN_NAME
    engine._prepare_prefill_metadata([], rows)
    total_len = context_len if context_len else 1
    engine._set_prefill_context(rows, total_len)
    try:
        with torch.inference_mode():
            engine._prefill_forward(rows, width)   # compile and allocate outside capture
        torch.cuda.synchronize()
        # A fresh context for the capture: the eager run above filled the context's
        # per-step SDPA cache with mask and page-index tensors that live outside the
        # graph and are freed after it. Recording reads of them would replay garbage.
        engine._set_prefill_context(rows, total_len)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=pool):
            logits = engine._prefill_forward(rows, width)
    finally:
        _clear_prefill_ctx()
    torch.cuda.synchronize()
    return PagedPrefillGraph(graph, rows, engine.prefill_attention, context_len, logits)

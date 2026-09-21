"""CUDA-Graph capture for one fused decode + chunked-prefill forward.

A prefill-carrying step used to run two forwards: the decode batch (graphed, ~10 ms on
the T4) and the chunk batch (graphed, ~17 ms). Every weight is read twice and every
launch is paid twice. The fused step packs both into one row of tokens - decode tokens
first, then the chunk batch flattened at the staged chunk width - so the per-token
modules run once over the union and only attention cuts the row apart (see
`fused_step_attention_forward`).

The graph is keyed by every shape it fixes: decode row bucket, chunk row bucket,
attention kind, gathered-context bucket (SDPA) and the decode kernel regime. Capture
runs on inert rows: decode rows at length 0 over dummy pages, chunk rows of length 0.
No request's KV is touched, so capture is safe at warm-up and lazily under load.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FusedStepGraph:
    graph: torch.cuda.CUDAGraph
    decode_rows: int
    prefill_rows: int
    attention: str
    context_len: int
    block_n: int
    num_warps: int
    next_tokens: torch.Tensor    # [decode_rows + prefill_rows]

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.next_tokens


def capture_fused_step_graph(
    engine, *, decode_rows: int, prefill_rows: int, context_len: int,
    block_n: int, num_warps: int, pool=None,
) -> FusedStepGraph:
    """Capture one fused forward at `decode_rows` + `prefill_rows` x chunk tokens."""
    if not torch.cuda.is_available() or torch.device(engine.device).type != "cuda":
        raise ValueError("fused step graph capture requires a CUDA engine")
    if not 0 < decode_rows <= engine.max_active or not 0 < prefill_rows <= engine.max_active:
        raise ValueError("row counts must be within the engine's active-batch limit")
    if block_n not in {16, 32, 64, 128} or num_warps not in {2, 4, 8}:
        raise ValueError("unsupported paged decode graph kernel regime")

    width = engine.prefill_chunk_size
    engine._set_attention(engine.FUSED_ATTN_NAME)
    _, _, block_tables, seq_lens = engine._prepare_decode_metadata([], graph_bucket_size=decode_rows)
    engine._prepare_prefill_metadata([], prefill_rows)

    def set_contexts():
        engine._set_fused_contexts(
            decode_rows=decode_rows, block_tables=block_tables, seq_lens=seq_lens,
            block_n=block_n, num_warps=num_warps, prefill_rows=prefill_rows,
            total_len=context_len if context_len else 1, width=width,
        )

    set_contexts()
    try:
        with torch.inference_mode():
            engine._fused_forward(decode_rows, prefill_rows, width)   # compile and allocate
        torch.cuda.synchronize()
        # Fresh contexts for the capture: the eager run filled the prefill context's
        # per-step SDPA cache with tensors outside the graph (see paged_prefill_graph).
        set_contexts()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=pool):
            next_tokens = engine._fused_forward(decode_rows, prefill_rows, width)
    finally:
        engine._clear_fused_contexts()
    torch.cuda.synchronize()
    return FusedStepGraph(
        graph, decode_rows, prefill_rows, engine.prefill_attention, context_len,
        block_n, num_warps, next_tokens,
    )

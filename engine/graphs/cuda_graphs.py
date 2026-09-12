"""Stage 15 CUDA-Graph eligibility and fixed-shape decode capture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import StaticCache


@dataclass(frozen=True)
class GraphEligibility:
    eligible: bool
    reasons: tuple[str, ...]


def assess_graph_eligibility(*, fixed_batch_size: bool, fixed_sequence_length: bool, static_cache: bool, dynamic_arrivals: bool) -> GraphEligibility:
    reasons: list[str] = []
    if not fixed_batch_size:
        reasons.append("batch size changes")
    if not fixed_sequence_length:
        reasons.append("sequence shape changes")
    if not static_cache:
        reasons.append("KV cache allocation is dynamic")
    if dynamic_arrivals:
        reasons.append("request arrivals/departures change control flow")
    return GraphEligibility(not reasons, tuple(reasons))


@dataclass
class CapturedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    cache: StaticCache
    static_input_ids: torch.Tensor
    static_cache_position: torch.Tensor
    logits: torch.Tensor

    def replay(self, token_id: int, position: int) -> torch.Tensor:
        self.static_input_ids.fill_(token_id)
        self.static_cache_position.fill_(position)
        self.graph.replay()
        return self.logits


def capture_decode_graph(
    model: PreTrainedModel, prompt_ids: torch.Tensor, *, max_cache_len: int
) -> tuple[CapturedDecodeGraph, int]:
    """Prefill a StaticCache and capture one fixed `[1, 1]` decode invocation.

    Capture itself executes the first decode at `prompt_length`; callers should treat
    its logits as the next token and replay at later positions.
    """
    if not torch.cuda.is_available() or prompt_ids.shape[0] != 1:
        raise ValueError("CUDA Graph capture requires a single CUDA batch")
    prompt_length = prompt_ids.shape[1]
    if max_cache_len <= prompt_length:
        raise ValueError("max_cache_len must exceed prompt length")
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    positions = torch.arange(prompt_length, device=prompt_ids.device)
    with torch.no_grad():
        prefill = model(input_ids=prompt_ids, past_key_values=cache, cache_position=positions, use_cache=True, return_dict=True)
    first_token = int(prefill.logits[:, -1, :].argmax(dim=-1).item())
    static_input_ids = torch.tensor([[first_token]], device=prompt_ids.device, dtype=prompt_ids.dtype)
    static_position = torch.tensor([prompt_length], device=prompt_ids.device, dtype=torch.long)

    # Warm up kernels outside capture to avoid recording one-time allocations.
    warm_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    with torch.no_grad():
        model(input_ids=prompt_ids, past_key_values=warm_cache, cache_position=positions, use_cache=True, return_dict=True)
        model(input_ids=static_input_ids, past_key_values=warm_cache, cache_position=static_position, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        outputs = model(
            input_ids=static_input_ids,
            past_key_values=cache,
            cache_position=static_position,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits
    return CapturedDecodeGraph(graph, cache, static_input_ids, static_position, logits), first_token

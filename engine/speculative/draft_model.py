"""Stateful single-row draft-model proposer for a second CUDA device."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch

from .proposer import Proposal


@dataclass
class _DraftState:
    cache: object
    tokens: tuple[int, ...]
    next_token: torch.Tensor


class DraftModelProposer:
    """Generate greedy proposals while retaining one rollback-capable KV per request.

    The target may reject a suffix or append a bonus token. On the next round, this
    proposer crops its Hugging Face cache to the common committed prefix and evaluates
    only the new target-approved suffix. GPU 1 is therefore independent of the target
    engine's paged KV while preserving exact request-level rollback semantics.
    """

    name = "draft_model"

    def __init__(self, model, device: str | torch.device):
        self.model = model
        self.device = torch.device(device)
        if next(model.parameters()).device != self.device:
            raise ValueError("draft model parameters must already be on the proposer device")
        self.model.eval()
        self._states: dict[str, _DraftState] = {}
        self.proposal_ms = 0.0
        self.proposed_tokens = 0
        self.rollback_tokens = 0

    @staticmethod
    def _common_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        length = min(len(left), len(right))
        for index in range(length):
            if left[index] != right[index]:
                return index
        return length

    def _forward_ids(self, input_ids: torch.Tensor, cache=None, prefix_length: int = 0) -> _DraftState:
        attention_mask = torch.ones(
            (1, prefix_length + input_ids.shape[1]), dtype=torch.long, device=self.device,
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return _DraftState(
            cache=outputs.past_key_values,
            tokens=(),
            next_token=outputs.logits[:, -1].argmax(dim=-1, keepdim=True),
        )

    def _forward_tokens(self, tokens: tuple[int, ...], cache=None, prefix_length: int = 0) -> _DraftState:
        if not tokens:
            raise ValueError("draft model cannot evaluate an empty token sequence")
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self._forward_ids(ids, cache, prefix_length)

    def _synchronize(self, request_id: str, history: tuple[int, ...]) -> _DraftState:
        state = self._states.get(request_id)
        if state is None:
            state = self._forward_tokens(history)
            state.tokens = history
            return state

        common = self._common_prefix(state.tokens, history)
        suffix = history[common:]
        # A successful round grows history. Rebuild the rare identical/prefix-only case
        # because the saved logits belong to the longer speculative cache position.
        if not suffix:
            state = self._forward_tokens(history)
            state.tokens = history
            return state
        if not hasattr(state.cache, "crop"):
            raise TypeError("draft speculation requires a transformers cache with crop()")
        self.rollback_tokens += len(state.tokens) - common
        state.cache.crop(common)
        advanced = self._forward_tokens(suffix, state.cache, common)
        advanced.tokens = history
        return advanced

    @torch.inference_mode()
    def propose(
        self, history: Sequence[int], max_tokens: int, *, request_id: str | None = None,
    ) -> Proposal:
        if max_tokens <= 0:
            return Proposal((), self.name)
        if request_id is None:
            raise ValueError("draft-model proposals require a request_id")
        committed = tuple(int(token) for token in history)
        if not committed:
            return Proposal((), self.name)

        started = perf_counter()
        state = self._synchronize(request_id, committed)
        proposal_tensors: list[torch.Tensor] = []
        for step in range(max_tokens):
            token = state.next_token
            proposal_tensors.append(token)
            advanced = self._forward_ids(
                token, state.cache, prefix_length=len(committed) + step,
            )
            state = _DraftState(advanced.cache, (), advanced.next_token)
        proposed = tuple(int(token) for token in torch.cat(proposal_tensors, dim=1)[0].tolist())
        state.tokens = committed + proposed
        self._states[request_id] = state
        self.proposed_tokens += len(proposed)
        self.proposal_ms += (perf_counter() - started) * 1000
        return Proposal(proposed, self.name)

    def forget(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def reset(self) -> None:
        self._states.clear()

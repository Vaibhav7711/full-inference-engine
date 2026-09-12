"""Stage 16 vanilla draft-model speculative decoding for greedy generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class GreedyAcceptance:
    emitted: list[int]
    accepted_draft_tokens: int
    fully_accepted: bool


def greedy_accept(draft_tokens: list[int], target_predictions: list[int], bonus_token: int) -> GreedyAcceptance:
    """Accept a matching draft prefix; emit the target token at the first mismatch."""
    if not draft_tokens or len(draft_tokens) != len(target_predictions):
        raise ValueError("draft_tokens and target_predictions must be non-empty and equal length")
    accepted = 0
    for draft_token, target_token in zip(draft_tokens, target_predictions):
        if draft_token != target_token:
            return GreedyAcceptance(draft_tokens[:accepted] + [target_token], accepted, False)
        accepted += 1
    return GreedyAcceptance(draft_tokens + [bonus_token], accepted, True)


@dataclass
class _ModelState:
    cache: object
    attention_mask: torch.Tensor
    next_token: int


@dataclass
class SpeculativeGenerationResult:
    token_ids: list[int]
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    rounds: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.proposed_draft_tokens if self.proposed_draft_tokens else 0.0


class VanillaSpeculativeDecoder:
    """Greedy vanilla speculation with explicit cache rollback/commit semantics.

    Both models retain dynamic HF KV caches. On a rejected proposal, both caches crop
    back to the accepted prefix before the target-resampled token is appended.
    """

    def __init__(self, target: PreTrainedModel, draft: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.target, self.draft, self.tokenizer, self.device = target.eval(), draft.eval(), tokenizer, device

    @torch.inference_mode()
    def _prefill(self, model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _ModelState:
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        return _ModelState(output.past_key_values, attention_mask, int(output.logits[:, -1, :].argmax(dim=-1).item()))

    @torch.inference_mode()
    def _decode(self, model: PreTrainedModel, state: _ModelState, token: int) -> _ModelState:
        mask = torch.cat([state.attention_mask, torch.ones((1, 1), device=self.device, dtype=state.attention_mask.dtype)], dim=1)
        output = model(input_ids=torch.tensor([[token]], device=self.device), attention_mask=mask, past_key_values=state.cache, use_cache=True, return_dict=True)
        return _ModelState(output.past_key_values, mask, int(output.logits[:, -1, :].argmax(dim=-1).item()))

    @staticmethod
    def _crop(state: _ModelState, length: int) -> _ModelState:
        if not hasattr(state.cache, "crop"):
            raise TypeError("speculative rollback requires an HF cache with crop(max_length)")
        tokens_to_remove = state.attention_mask.shape[1] - length
        if tokens_to_remove < 0:
            raise ValueError("cannot crop a cache to a longer sequence")
        if tokens_to_remove:
            # Transformers 5.18+ defines positive crop values as deprecated. The
            # negative form explicitly means “remove this suffix length.”
            state.cache.crop(-tokens_to_remove)
        return _ModelState(state.cache, state.attention_mask[:, :length], state.next_token)

    @torch.inference_mode()
    def generate(self, prompt: str, *, max_new_tokens: int, speculation_depth: int = 4) -> SpeculativeGenerationResult:
        if not prompt or max_new_tokens < 1 or speculation_depth < 1:
            raise ValueError("prompt, max_new_tokens, and speculation_depth must be positive")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_state = self._prefill(self.target, inputs.input_ids, inputs.attention_mask)
        draft_state = self._prefill(self.draft, inputs.input_ids, inputs.attention_mask)
        eos = self.target.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        emitted: list[int] = []
        accepted_total = proposed_total = rounds = 0

        while len(emitted) < max_new_tokens:
            depth = min(speculation_depth, max_new_tokens - len(emitted))
            context_length = target_state.attention_mask.shape[1]
            proposals: list[int] = []
            for _ in range(depth):
                proposals.append(draft_state.next_token)
                draft_state = self._decode(self.draft, draft_state, proposals[-1])
            proposed_total += len(proposals)

            target_mask = torch.cat([target_state.attention_mask, torch.ones((1, len(proposals)), device=self.device, dtype=target_state.attention_mask.dtype)], dim=1)
            verified = self.target(input_ids=torch.tensor([proposals], device=self.device), attention_mask=target_mask, past_key_values=target_state.cache, use_cache=True, return_dict=True)
            target_predictions = [target_state.next_token] + verified.logits[0, :-1, :].argmax(dim=-1).tolist()
            bonus = int(verified.logits[0, -1, :].argmax(dim=-1).item())
            acceptance = greedy_accept(proposals, target_predictions, bonus)
            rounds += 1
            round_tokens = acceptance.emitted
            eos_index = next((index for index, token in enumerate(round_tokens) if token in eos_ids), None)
            if eos_index is not None:
                round_tokens = round_tokens[: eos_index + 1]
                retained_accepted = min(acceptance.accepted_draft_tokens, eos_index + 1)
            else:
                retained_accepted = acceptance.accepted_draft_tokens
            accepted_total += retained_accepted
            emitted.extend(round_tokens[: max_new_tokens - len(emitted)])
            terminal = eos_index is not None

            target_state = _ModelState(verified.past_key_values, target_mask, bonus)
            accepted_length = context_length + retained_accepted
            target_state = self._crop(target_state, accepted_length)
            draft_state = self._crop(draft_state, accepted_length)
            final_token = round_tokens[-1]
            target_state = self._decode(self.target, target_state, final_token)
            draft_state = self._decode(self.draft, draft_state, final_token)
            if terminal:
                break
        return SpeculativeGenerationResult(emitted, accepted_total, proposed_total, rounds)

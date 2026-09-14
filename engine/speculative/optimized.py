"""Optimized vanilla speculative decoding — minimizes GPU->CPU syncs.

This is an OPTIMIZED variant of engine/speculative/vanilla.py. Same algorithm (greedy
draft/verify), same correctness, but it eliminates the per-token GPU->CPU synchronizations
that stall the pipeline in the original.

=========================================================================
WHAT CHANGED vs the original vanilla.py — your mental-model map:
=========================================================================

ORIGINAL (_decode returned an int via .item()):
    def _decode(self, model, state, token):
        output = model(input_ids=torch.tensor([[token]], device=...), ...)  # rebuild tensor
        return _ModelState(..., int(output.logits[:, -1, :].argmax(-1).item()))  # .item() = SYNC

    Problem: every draft step calls .item() -> GPU stalls, CPU pulls 1 int, rebuilds a CPU
    tensor, ships it back. K syncs per round in the draft loop.

OPTIMIZED (_decode keeps the token as a GPU tensor):
    def _decode_gpu(self, model, state, token_tensor):
        output = model(input_ids=token_tensor, ...)          # token already a GPU tensor
        next_tok = output.logits[:, -1, :].argmax(-1, keepdim=True)  # STAYS on GPU [1,1]
        return _ModelStateGpu(..., next_tok)                 # no .item(), no sync

    The draft loop feeds next_tok (a GPU tensor) directly to the next forward. We only
    pull tokens to CPU ONCE per round, right before the acceptance comparison, using a
    single .tolist() on the whole proposals tensor instead of K separate .item() calls.

Change 2 — build the proposals as a stacked GPU tensor, not a Python list of ints:
    ORIGINAL: proposals = []; proposals.append(int(...)); torch.tensor([proposals])  (CPU->GPU each round)
    OPTIMIZED: collect proposal tensors on GPU, torch.cat once. One sync at verify.

Change 3 — the acceptance comparison still needs ints (it's Python control flow), so we
    do ONE .tolist() on the full proposals + target predictions, not per-token .item().

NET EFFECT: syncs per round drop from ~K (one per draft token) to ~1 (at verify).
On a T4 this removes the ~193ms/round implementation overhead we measured.

HONEST NOTE: even fully optimized, the ~270ms/round MEMORY-BANDWIDTH floor remains
(draft ≈ target cost at batch=1). So this makes the engine clean and the measurement
honest, but does NOT flip speculative decoding to a win on a T4. That's physics, not code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class _ModelStateGpu:
    """Like the original _ModelState, but next_token is a GPU TENSOR [1,1], not an int."""
    cache: object
    attention_mask: torch.Tensor
    next_token: torch.Tensor      # [1, 1] on GPU — the key change


@dataclass
class SpeculativeResult:
    token_ids: list[int]
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    rounds: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.proposed_draft_tokens if self.proposed_draft_tokens else 0.0


class OptimizedSpeculativeDecoder:
    """Greedy vanilla speculation with GPU-resident tokens (minimal syncs)."""

    def __init__(self, target: PreTrainedModel, draft: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.target, self.draft = target.eval(), draft.eval()
        self.tokenizer, self.device = tokenizer, device

    @torch.inference_mode()
    def _prefill(self, model, input_ids, attention_mask) -> _ModelStateGpu:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=True, return_dict=True)
        # next_token stays on GPU as [1,1]
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return _ModelStateGpu(out.past_key_values, attention_mask, next_tok)

    @torch.inference_mode()
    def _decode_gpu(self, model, state: _ModelStateGpu, token_tensor: torch.Tensor) -> _ModelStateGpu:
        """One decode step. token_tensor is [1,1] on GPU — fed directly, no rebuild, no .item()."""
        mask = torch.cat(
            [state.attention_mask, torch.ones((1, 1), device=self.device, dtype=state.attention_mask.dtype)],
            dim=1,
        )
        out = model(input_ids=token_tensor, attention_mask=mask,
                    past_key_values=state.cache, use_cache=True, return_dict=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # STAYS on GPU
        return _ModelStateGpu(out.past_key_values, mask, next_tok)

    @staticmethod
    def _crop(state: _ModelStateGpu, length: int) -> _ModelStateGpu:
        if not hasattr(state.cache, "crop"):
            raise TypeError("speculative rollback requires an HF cache with crop()")
        remove = state.attention_mask.shape[1] - length
        if remove > 0:
            state.cache.crop(-remove)
        return _ModelStateGpu(state.cache, state.attention_mask[:, :length], state.next_token)

    @torch.inference_mode()
    def generate(self, prompt: str, *, max_new_tokens: int, speculation_depth: int = 4) -> SpeculativeResult:
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

            # --- DRAFT: generate `depth` tokens, keeping each on GPU ---
            # OPTIMIZED: proposal_tensors holds GPU tensors; NO .item() in this loop.
            proposal_tensors = []
            for _ in range(depth):
                proposal_tensors.append(draft_state.next_token)          # [1,1] GPU tensor
                draft_state = self._decode_gpu(self.draft, draft_state, draft_state.next_token)
            proposed_total += depth

            # Stack proposals into one [1, depth] GPU tensor (single op, no per-token sync)
            proposals_gpu = torch.cat(proposal_tensors, dim=1)           # [1, depth] on GPU

            # --- TARGET: verify all proposals in ONE forward pass ---
            target_mask = torch.cat(
                [target_state.attention_mask, torch.ones((1, depth), device=self.device,
                 dtype=target_state.attention_mask.dtype)], dim=1)
            verified = self.target(input_ids=proposals_gpu, attention_mask=target_mask,
                                   past_key_values=target_state.cache, use_cache=True, return_dict=True)

            # Target's greedy predictions: prev next_token, then argmax of each position (all on GPU)
            # verified.logits: [1, depth, vocab]. Prediction at pos i is argmax of logits[:, i-1] etc.
            target_preds_gpu = verified.logits[0, :-1, :].argmax(dim=-1)  # [depth-1] on GPU
            bonus_gpu = verified.logits[0, -1, :].argmax(dim=-1)          # scalar GPU

            # --- ONE sync here: pull the tensors we need for Python control flow ---
            # This is the SINGLE .tolist()/.item() per round, replacing K .item() calls.
            proposals = proposals_gpu[0].tolist()                        # [depth] ints
            # target prediction for position 0 is the token BEFORE the round (target_state.next_token)
            target_predictions = [int(target_state.next_token.item())] + target_preds_gpu.tolist()
            bonus = int(bonus_gpu.item())

            # --- Greedy acceptance (Python control flow, needs ints — that's fine, one batch) ---
            accepted = 0
            for dt, tt in zip(proposals, target_predictions):
                if dt != tt:
                    break
                accepted += 1

            if accepted == depth:
                round_tokens = proposals + [bonus]
            else:
                round_tokens = proposals[:accepted] + [target_predictions[accepted]]

            # EOS handling
            eos_index = next((i for i, t in enumerate(round_tokens) if t in eos_ids), None)
            if eos_index is not None:
                round_tokens = round_tokens[:eos_index + 1]
                retained = min(accepted, eos_index + 1)
            else:
                retained = accepted
            accepted_total += retained
            emitted.extend(round_tokens[: max_new_tokens - len(emitted)])
            rounds += 1
            terminal = eos_index is not None

            # --- Commit: crop caches to accepted prefix, re-decode the final token ---
            # final token as a GPU tensor (build once)
            final_token = round_tokens[-1]
            final_tok_gpu = torch.tensor([[final_token]], device=self.device)

            target_state = _ModelStateGpu(verified.past_key_values, target_mask, final_tok_gpu)
            accepted_length = context_length + retained
            target_state = self._crop(target_state, accepted_length)
            draft_state = self._crop(draft_state, accepted_length)
            target_state = self._decode_gpu(self.target, target_state, final_tok_gpu)
            draft_state = self._decode_gpu(self.draft, draft_state, final_tok_gpu)

            if terminal:
                break

        return SpeculativeResult(emitted, accepted_total, proposed_total, rounds)

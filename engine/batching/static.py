"""Stage 9 fixed-membership, explicit prefill/decode batching baseline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from engine.metrics.metrics import cpu_elapsed_ms


@dataclass
class StaticBatchResult:
    token_ids: list[list[int]]
    texts: list[str]
    tokenization_ms: float
    prefill_ms: float
    decode_ms: list[float]
    total_ms: float

    @property
    def output_tokens(self) -> int:
        return sum(len(tokens) for tokens in self.token_ids)


class StaticBatchRunner:
    """Fixed batch-size greedy runner.

    Finished rows remain in the batch and receive EOS input. This intentional wasted
    work is the baseline that continuous batching must beat in Stage 10.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.model.eval()

    def _eos_ids(self, eos_token_id: int | list[int] | None) -> set[int]:
        configured = self.model.generation_config.eos_token_id if eos_token_id is None else eos_token_id
        if configured is None:
            configured = self.tokenizer.eos_token_id
        return {configured} if isinstance(configured, int) else set(configured)

    @staticmethod
    def _prefill_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        positions = attention_mask.long().cumsum(dim=-1) - 1
        return positions.masked_fill(attention_mask == 0, 0)

    @torch.inference_mode()
    def generate(self, prompts: list[str], *, max_new_tokens: int, eos_token_id: int | list[int] | None = None) -> StaticBatchResult:
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("prompts must contain at least one non-empty string")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        total_start = perf_counter_ns()
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            tokenize_start = perf_counter_ns()
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
            tokenization_ms = cpu_elapsed_ms(tokenize_start)
        finally:
            self.tokenizer.padding_side = original_padding_side
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        position_ids = self._prefill_positions(attention_mask).to(self.device)
        eos_ids = self._eos_ids(eos_token_id)
        filler_token = min(eos_ids) if eos_ids else self.tokenizer.eos_token_id
        if filler_token is None:
            raise ValueError("an EOS or pad token is required for static batching")
        batch_size = len(prompts)
        generated: list[list[int]] = [[] for _ in prompts]
        active = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        torch.cuda.synchronize(self.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        end.record(); end.synchronize()
        prefill_ms = start.elapsed_time(end)
        cache = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_ms: list[float] = []

        for step in range(max_new_tokens):
            for row, token in enumerate(next_tokens.squeeze(1).tolist()):
                if active[row]:
                    generated[row].append(token)
                    if token in eos_ids:
                        active[row] = False
            if not bool(active.any()) or step == max_new_tokens - 1:
                break
            decode_input = next_tokens.clone()
            decode_input[~active] = filler_token
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)], dim=1
            )
            position_ids = attention_mask.long().sum(dim=-1, keepdim=True) - 1
            start.record()
            outputs = self.model(
                input_ids=decode_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            end.record(); end.synchronize()
            decode_ms.append(start.elapsed_time(end))
            cache = outputs.past_key_values
            next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return StaticBatchResult(
            token_ids=generated,
            texts=[self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated],
            tokenization_ms=tokenization_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=cpu_elapsed_ms(total_start),
        )

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from engine.metrics.metrics import GenerationMetrics, cpu_elapsed_ms


@dataclass
class PrefillState:
    past_key_values: object
    attention_mask: torch.Tensor
    next_token: torch.Tensor


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    metrics: GenerationMetrics


class ExplicitDecodeRunner:
    """Correctness-first, single-request greedy runtime.

    Model execution is intentionally visible: `prefill` processes the complete prompt;
    `decode_one` processes exactly one input token with the accumulated KV state.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.model.eval()

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> PrefillState:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        return PrefillState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            next_token=outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        )

    @torch.inference_mode()
    def decode_one(self, token: torch.Tensor, state: PrefillState) -> PrefillState:
        """Append `token`, update HF's KV cache, and greedily choose the next token."""
        attention_mask = torch.cat(
            [state.attention_mask, torch.ones((1, 1), device=self.device, dtype=state.attention_mask.dtype)], dim=1
        )
        outputs = self.model(
            input_ids=token,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return PrefillState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            next_token=outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        )

    def generate(self, prompt: str, *, max_new_tokens: int, eos_token_id: int | None = None) -> GenerationResult:
        if not prompt:
            raise ValueError("prompt must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        metrics, total_start = GenerationMetrics(), perf_counter_ns()
        torch.cuda.reset_peak_memory_stats(self.device)
        encode_start = perf_counter_ns()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        metrics.tokenization_ms = cpu_elapsed_ms(encode_start)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        eos = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id

        torch.cuda.synchronize(self.device)
        event_start, event_end = torch.cuda.Event(True), torch.cuda.Event(True)
        event_start.record()
        state = self.prefill(input_ids, attention_mask)
        event_end.record(); event_end.synchronize()
        metrics.prefill_ms = event_start.elapsed_time(event_end)
        metrics.ttft_ms = metrics.tokenization_ms + metrics.prefill_ms
        # The prefill logits contain the first output token; no decode step is needed.
        metrics.first_token_ms = metrics.prefill_ms

        generated: list[int] = []
        for step in range(max_new_tokens):
            token_id = int(state.next_token.item())
            generated.append(token_id)
            if token_id == eos:
                break
            if step == max_new_tokens - 1:
                break
            event_start.record()
            state = self.decode_one(state.next_token, state)
            event_end.record(); event_end.synchronize()
            metrics.decode_ms.append(event_start.elapsed_time(event_end))
        metrics.total_ms = cpu_elapsed_ms(total_start)
        metrics.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        metrics.peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
        return GenerationResult(generated, self.tokenizer.decode(generated, skip_special_tokens=True), metrics)

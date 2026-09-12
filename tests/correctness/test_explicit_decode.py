"""Stage 2: explicit runtime equivalence against Hugging Face greedy generation."""

from __future__ import annotations

import pytest
import torch

from engine.model import ExplicitDecodeRunner, LoadedModel


def hf_greedy(loaded: LoadedModel, prompt: str, max_new_tokens: int) -> list[int]:
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=loaded.tokenizer.eos_token_id,
        )
    return generated[0, inputs.input_ids.shape[1] :].tolist()


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("prompt", "max_new_tokens"),
    [
        ("Hello", 1),
        ("The capital of France is", 8),
        ("Explain why key-value caching reduces decode work in a decoder-only transformer.", 16),
        (" ".join(["cache"] * 64), 4),
    ],
    ids=["short-one-token", "short", "medium", "long-prompt"],
)
def test_explicit_greedy_matches_hf_generate(
    loaded: LoadedModel, prompt: str, max_new_tokens: int
) -> None:
    """Every emitted token must match the trusted HF greedy path exactly."""
    engine = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    ours = engine.generate(prompt, max_new_tokens=max_new_tokens)
    assert ours.token_ids == hf_greedy(loaded, prompt, max_new_tokens)


@pytest.mark.cuda
def test_generation_honors_max_new_tokens(loaded: LoadedModel) -> None:
    engine = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    result = engine.generate("A concise fact:", max_new_tokens=3, eos_token_id=-1)
    assert len(result.token_ids) == 3


@pytest.mark.cuda
def test_generation_stops_when_first_token_is_eos(loaded: LoadedModel) -> None:
    """Force the first predicted token to behave as EOS to exercise the stop branch."""
    engine = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    inputs = loaded.tokenizer("A concise fact:", return_tensors="pt").to(loaded.device)
    state = engine.prefill(inputs.input_ids, inputs.attention_mask)
    forced_eos = int(state.next_token.item())
    result = engine.generate("A concise fact:", max_new_tokens=8, eos_token_id=forced_eos)
    assert result.token_ids == [forced_eos]
    assert result.metrics.decode_ms == []

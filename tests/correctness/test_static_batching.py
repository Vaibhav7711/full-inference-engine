import pytest

from engine.batching import StaticBatchRunner
from engine.model import ExplicitDecodeRunner, LoadedModel


@pytest.mark.cuda
def test_static_batch_matches_individual_greedy_generation(loaded: LoadedModel) -> None:
    prompts = ["Hello", "The capital of France is", "cache cache cache cache"]
    max_new_tokens = 8
    batched = StaticBatchRunner(loaded.model, loaded.tokenizer, loaded.device).generate(
        prompts, max_new_tokens=max_new_tokens
    )
    reference = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    expected = [reference.generate(prompt, max_new_tokens=max_new_tokens).token_ids for prompt in prompts]
    assert batched.token_ids == expected

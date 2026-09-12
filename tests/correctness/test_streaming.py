import pytest

from engine.model import ExplicitDecodeRunner, LoadedModel


@pytest.mark.cuda
def test_streamed_tokens_match_non_streamed_generation(loaded: LoadedModel) -> None:
    prompt = "KV cache means"
    runner = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    expected = runner.generate(prompt, max_new_tokens=8).token_ids
    events = list(runner.stream_generate(prompt, max_new_tokens=8))
    token_ids = [event.token_id for event in events if event.token_id is not None]
    assert token_ids == expected
    assert events[-1].finish_reason in {"EOS", "LENGTH"}

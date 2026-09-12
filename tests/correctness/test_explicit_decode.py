import pytest
import torch

from engine.model import ExplicitDecodeRunner, load_model


@pytest.mark.cuda
def test_explicit_greedy_matches_hf_generate() -> None:
    loaded = load_model()
    prompt = "The capital of France is"
    engine = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    ours = engine.generate(prompt, max_new_tokens=8)
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    with torch.inference_mode():
        reference = loaded.model.generate(**inputs, do_sample=False, max_new_tokens=8, use_cache=True)
    expected = reference[0, inputs.input_ids.shape[1] :].tolist()
    assert ours.token_ids == expected

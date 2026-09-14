"""Batched speculation correctness — gated against per-sequence vanilla speculation.

STAGED so failures localize:
  test_bs_single_sequence  -> N=1: batched engine with one seq == vanilla speculative.
                              (isolates the batched machinery from the ragged-commit logic)
  test_bs_two_identical    -> N=2 same prompt: both sequences should produce identical output
                              and match vanilla. (isolates batching without ragged divergence)
  test_bs_two_different    -> N=2 different prompts: the real ragged case. Each must match
                              its own vanilla speculative output.

If single-sequence passes but two-different fails, the bug is in the ragged commit/rollback.
If single-sequence fails, the bug is in the basic batched draft/verify.

Run:
    python -m pytest tests/batching/test_batched_speculative.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TARGET = "Qwen/Qwen3-1.7B"   # smaller target so the test loads fast; crossover uses 4B
DRAFT = "Qwen/Qwen3-0.6B"


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()
    target.config._attn_implementation = "sdpa"
    draft.config._attn_implementation = "sdpa"
    return target, draft, tok


def _vanilla_reference(target, draft, tok, prompt, max_new, depth):
    """Per-sequence vanilla speculative output — the oracle."""
    from engine.speculative.vanilla import VanillaSpeculativeDecoder
    dec = VanillaSpeculativeDecoder(target, draft, tok, torch.device("cuda"))
    return dec.generate(prompt, max_new_tokens=max_new, speculation_depth=depth).token_ids


@cuda
@requires_cuda
def test_bs_single_sequence():
    """N=1: batched engine == vanilla speculative. Isolates batched machinery."""
    from engine.batching.batched_speculative import BatchedSpeculativeEngine

    target, draft, tok = _load()
    eng = BatchedSpeculativeEngine(target, draft, tok, torch.device("cuda"))

    prompt = "The capital of France is"
    max_new, depth = 20, 4

    ref = _vanilla_reference(target, draft, tok, prompt, max_new, depth)
    out = eng.generate([prompt], max_new_tokens=max_new, speculation_depth=depth).outputs[0]

    # Trim to same length for comparison (batched may over/under-run by rounding)
    n = min(len(ref), len(out))
    assert out[:n] == ref[:n], (
        f"N=1 batched != vanilla\n  ref: {ref[:n]}\n  out: {out[:n]}"
    )


@cuda
@requires_cuda
def test_bs_two_identical_prompts():
    """N=2 same prompt: both outputs identical and match vanilla. Batching, no ragged divergence."""
    from engine.batching.batched_speculative import BatchedSpeculativeEngine

    target, draft, tok = _load()
    eng = BatchedSpeculativeEngine(target, draft, tok, torch.device("cuda"))

    prompt = "The capital of France is"
    max_new, depth = 20, 4

    ref = _vanilla_reference(target, draft, tok, prompt, max_new, depth)
    res = eng.generate([prompt, prompt], max_new_tokens=max_new, speculation_depth=depth)

    for i, out in enumerate(res.outputs):
        n = min(len(ref), len(out))
        assert out[:n] == ref[:n], (
            f"N=2 identical, seq {i} != vanilla\n  ref: {ref[:n]}\n  out: {out[:n]}"
        )


@cuda
@requires_cuda
def test_bs_two_different_prompts():
    """N=2 different prompts: the REAL ragged case. Each matches its own vanilla output."""
    from engine.batching.batched_speculative import BatchedSpeculativeEngine

    target, draft, tok = _load()
    eng = BatchedSpeculativeEngine(target, draft, tok, torch.device("cuda"))

    prompts = ["The capital of France is", "Once upon a time in a land far away"]
    max_new, depth = 20, 4

    refs = [_vanilla_reference(target, draft, tok, p, max_new, depth) for p in prompts]
    res = eng.generate(prompts, max_new_tokens=max_new, speculation_depth=depth)

    for i, (out, ref) in enumerate(zip(res.outputs, refs)):
        n = min(len(ref), len(out))
        assert out[:n] == ref[:n], (
            f"N=2 different, seq {i} ({prompts[i]!r}) != vanilla\n"
            f"  ref: {ref[:n]}\n  out: {out[:n]}"
        )

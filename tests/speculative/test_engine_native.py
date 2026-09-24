"""CUDA gates for speculative verification through the live paged-KV engine."""

from __future__ import annotations

import pytest
import torch

from engine.speculative import Proposal


cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _AlwaysWrong:
    name = "test-wrong"

    def propose(self, history, max_tokens, *, request_id=None):
        return Proposal(tuple([0] * max_tokens), self.name)

    def forget(self, request_id):
        pass


class _ReferenceProposer:
    name = "test-reference"

    def __init__(self, prompt_length: int, reference: list[int]):
        self.prompt_length = prompt_length
        self.reference = reference

    def propose(self, history, max_tokens, *, request_id=None):
        produced = len(history) - self.prompt_length
        return Proposal(tuple(self.reference[produced:produced + max_tokens]), self.name)

    def forget(self, request_id):
        pass


def _engine(model, tok, *, depth=3):
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    return ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=512, block_size=16, max_active=2,
        prefix_cache_blocks=0, cuda_graph_batch_sizes=None,
        speculative_ngram=True, speculation_depth=depth,
    )


@cuda
@requires_cuda
def test_engine_native_rejection_matches_ordinary_greedy(loaded):
    prompt = "The capital of France is"
    model, tok = loaded.model, loaded.tokenizer
    baseline = _engine(model, tok)
    baseline.speculative_proposer = None
    expected = baseline.generate([prompt], max_new_tokens=12)[0]

    speculative = _engine(model, tok)
    speculative.speculative_proposer = _AlwaysWrong()
    actual = speculative.generate([prompt], max_new_tokens=12)[0]

    assert actual == expected
    assert speculative.speculative_rounds > 0
    assert speculative.speculative_accepted_tokens == 0
    assert speculative.block_manager.snapshot()["used_blocks"] == 0


@cuda
@requires_cuda
def test_engine_native_full_acceptance_commits_multiple_tokens(loaded):
    prompt = "Write a short Python loop:"
    model, tok = loaded.model, loaded.tokenizer
    baseline = _engine(model, tok)
    baseline.speculative_proposer = None
    expected = baseline.generate([prompt], max_new_tokens=16)[0]

    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    speculative = _engine(model, tok, depth=3)
    speculative.speculative_proposer = _ReferenceProposer(len(prompt_ids), expected)
    actual = speculative.generate([prompt], max_new_tokens=16)[0]

    assert actual == expected
    assert speculative.speculative_accepted_tokens > 0
    assert speculative.speculative_accepted_tokens <= speculative.speculative_proposed_tokens
    assert speculative.speculative_rounds < len(expected) - 1
    assert speculative.block_manager.snapshot()["used_blocks"] == 0

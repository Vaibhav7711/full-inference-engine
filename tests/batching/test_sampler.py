"""Batched sampling: per-row parameters, and the greedy path left untouched.

These run on CPU: the sampler is pure torch over a `[rows, vocab]` tensor, and the
properties that matter (a row's tokens do not depend on its neighbours; greedy stays
argmax; filters admit exactly the right token sets) are device-independent.
"""

from __future__ import annotations

import math

import pytest
import torch

from engine.batching.sampler import BatchedSampler
from engine.runtime.sampling import GREEDY, SamplingParams


def _logits(rows: int, vocab: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(rows, vocab, generator=torch.Generator().manual_seed(seed))


def test_all_greedy_batch_is_exactly_argmax() -> None:
    logits = _logits(4, 50)
    result = BatchedSampler("cpu").sample(logits, [GREEDY] * 4)
    assert result.token_ids == logits.argmax(dim=-1).tolist()
    assert result.logprobs == {}


def test_zero_temperature_is_greedy_whatever_else_is_set() -> None:
    logits = _logits(2, 32)
    params = [SamplingParams(temperature=0.0, top_p=0.1, top_k=3, seed=5)] * 2
    result = BatchedSampler("cpu").sample(logits, params)
    assert result.token_ids == logits.argmax(dim=-1).tolist()


def test_top_k_one_is_greedy() -> None:
    logits = _logits(3, 40)
    params = [SamplingParams(temperature=1.0, top_k=1)] * 3
    assert BatchedSampler("cpu").sample(logits, params).token_ids == logits.argmax(-1).tolist()


def test_top_k_restricts_choices_to_the_k_best() -> None:
    logits = _logits(1, 100)
    allowed = set(logits[0].topk(5).indices.tolist())
    sampler = BatchedSampler("cpu", generator_seed=0)
    params = [SamplingParams(temperature=1.5, top_k=5)]
    drawn = {sampler.sample(logits, params).token_ids[0] for _ in range(200)}
    assert drawn <= allowed
    assert len(drawn) > 1, "sampling collapsed to one token; the draw is not random"


def test_top_p_keeps_the_smallest_prefix_reaching_the_mass() -> None:
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.10]]))
    sampler = BatchedSampler("cpu", generator_seed=1)
    params = [SamplingParams(temperature=1.0, top_p=0.7)]
    drawn = {sampler.sample(logits, params).token_ids[0] for _ in range(300)}
    # 0.5 alone is below 0.7; adding 0.25 crosses it, so tokens 0 and 1 survive.
    assert drawn == {0, 1}


def test_top_p_always_keeps_the_most_likely_token() -> None:
    logits = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))
    sampler = BatchedSampler("cpu", generator_seed=2)
    params = [SamplingParams(temperature=1.0, top_p=0.3)]
    drawn = {sampler.sample(logits, params).token_ids[0] for _ in range(50)}
    assert drawn == {0}


def test_min_p_drops_tokens_far_below_the_mode() -> None:
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.09, 0.01]]))
    sampler = BatchedSampler("cpu", generator_seed=3)
    params = [SamplingParams(temperature=1.0, min_p=0.2)]     # floor = 0.2 * 0.6 = 0.12
    drawn = {sampler.sample(logits, params).token_ids[0] for _ in range(300)}
    assert drawn == {0, 1}


def test_seeded_rows_are_reproducible_and_batch_independent() -> None:
    logits = _logits(3, 64, seed=7)
    params = [SamplingParams(temperature=1.0, seed=1234)] * 3
    alone = BatchedSampler("cpu").sample(logits[:1], params[:1]).token_ids[0]
    with_others = BatchedSampler("cpu").sample(logits, params).token_ids[0]
    again = BatchedSampler("cpu").sample(logits[:1], params[:1]).token_ids[0]
    assert alone == with_others == again


def test_mixed_batch_greedy_rows_are_unaffected_by_sampled_neighbours() -> None:
    logits = _logits(4, 80, seed=11)
    params = [GREEDY, SamplingParams(temperature=1.2, top_p=0.9),
              GREEDY, SamplingParams(temperature=0.8, top_k=10, seed=99)]
    sampler = BatchedSampler("cpu", generator_seed=0)
    result = sampler.sample(logits, params)
    assert result.token_ids[0] == logits[0].argmax().item()
    assert result.token_ids[2] == logits[2].argmax().item()


def test_repetition_penalty_pushes_seen_tokens_down() -> None:
    logits = torch.tensor([[2.0, 1.9, 0.0]])
    params = [SamplingParams(repetition_penalty=1.5)]
    sampler = BatchedSampler("cpu")
    # Without the penalty token 0 wins; penalising it hands the argmax to token 1.
    assert sampler.sample(logits, [GREEDY], [[0]]).token_ids == [0]
    assert sampler.sample(logits, params, [[0]]).token_ids == [1]


def test_repetition_penalty_moves_negative_logits_down_too() -> None:
    logits = torch.tensor([[-1.0, -1.1]])
    sampler = BatchedSampler("cpu")
    params = [SamplingParams(repetition_penalty=2.0)]
    assert sampler.sample(logits, params, [[0]]).token_ids == [1]


def test_frequency_penalty_scales_with_occurrences() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    sampler = BatchedSampler("cpu")
    params = [SamplingParams(frequency_penalty=0.6)]
    assert sampler.sample(logits, params, [[0]]).token_ids == [0]       # 3.0 - 0.6 = 2.4
    assert sampler.sample(logits, params, [[0, 0]]).token_ids == [1]    # 3.0 - 1.2 = 1.8


def test_presence_penalty_applies_once_per_token() -> None:
    logits = torch.tensor([[3.0, 2.5, 1.0]])
    sampler = BatchedSampler("cpu")
    params = [SamplingParams(presence_penalty=0.75)]
    assert sampler.sample(logits, params, [[0, 0, 0]]).token_ids == [1]


def test_penalties_are_per_row() -> None:
    logits = torch.tensor([[2.0, 1.9, 0.0], [2.0, 1.9, 0.0]])
    sampler = BatchedSampler("cpu")
    params = [SamplingParams(repetition_penalty=1.5), GREEDY]
    assert sampler.sample(logits, params, [[0], [0]]).token_ids == [1, 0]


def test_logprobs_report_chosen_token_first_and_sum_sensibly() -> None:
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    result = BatchedSampler("cpu").sample(logits, [SamplingParams(logprobs=2)])
    entries = result.logprobs[0]
    assert entries[0][0] == result.token_ids[0]
    assert math.isclose(entries[0][1], math.log(0.7), abs_tol=1e-5)
    assert len(entries) == 3 or len(entries) == 2  # chosen + up to 2 alternatives
    assert all(value <= 0.0 for _, value in entries)


def test_logprobs_follow_temperature() -> None:
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    hot = BatchedSampler("cpu").sample(
        logits, [SamplingParams(temperature=2.0, logprobs=1, seed=3)],
    ).logprobs[0][0][1]
    cold = BatchedSampler("cpu").sample(logits, [SamplingParams(logprobs=1)]).logprobs[0][0][1]
    # Flattening the distribution cannot make the argmax more likely than it was.
    assert hot > math.log(0.7) - 1e-6 or hot < cold


def test_invalid_parameters_are_rejected() -> None:
    for kwargs in (
        {"temperature": -0.1}, {"temperature": 2.1}, {"top_p": 0.0}, {"top_p": 1.2},
        {"top_k": -1}, {"min_p": 1.5}, {"repetition_penalty": 0.0},
        {"presence_penalty": 3.0}, {"frequency_penalty": -3.0}, {"logprobs": 21},
    ):
        with pytest.raises(ValueError):
            SamplingParams(**kwargs)


def test_greedy_default_is_deterministic_and_penalty_free() -> None:
    assert GREEDY.greedy and GREEDY.deterministic and not GREEDY.has_penalties


def test_forget_drops_row_generators() -> None:
    sampler = BatchedSampler("cpu")
    sampler.sample(_logits(1, 16), [SamplingParams(temperature=1.0, seed=42)],
                   request_ids=["request-a"])
    assert "request-a" in sampler._seeded
    sampler.forget(["request-a"])
    assert "request-a" not in sampler._seeded


def test_same_seed_is_owned_by_each_request() -> None:
    sampler = BatchedSampler("cpu")
    params = [SamplingParams(temperature=1.0, seed=42)] * 2
    logits = _logits(2, 16)
    first = sampler.sample(logits, params, request_ids=["a", "b"]).token_ids
    sampler.forget(["a", "b"])
    assert sampler.sample(logits, params, request_ids=["a", "b"]).token_ids == first

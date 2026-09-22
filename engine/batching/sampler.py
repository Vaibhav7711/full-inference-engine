"""Batched sampling: one step's logits, many requests, each with its own parameters.

A decode step produces `[N, vocab]` logits for N unrelated requests. Sampling them one
row at a time would be N launches and N device syncs on the step's critical path, which
at a 10 ms step is the same mistake the per-element metadata staging was. So every stage
here is a whole-batch tensor operation, with per-row parameters staged into `[N]` tensors
and disabled stages skipped for the whole batch when no row wants them.

Three properties this has to keep:

* A batch where every request is greedy must take exactly the argmax path the engine has
  always taken. Every benchmark and the token-identity gate depend on it, so the greedy
  fast path returns before any sampling machinery runs.
* A request's tokens must not depend on who else is in the batch. Per-row parameters and
  per-row penalty state give that for free; a per-request `seed` additionally makes the
  draw itself independent, which costs one `torch.multinomial` call for each seeded row
  (their generators cannot be batched) and is why seeds are opt-in.
* Nothing here may synchronize. The result is one `[N]` tensor moved to the host once by
  the caller, together with everything else the step needs.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from engine.runtime.sampling import SamplingParams


@dataclass
class SampleResult:
    """Chosen token per row, plus logprobs for the rows that asked for them."""

    token_ids: list[int]
    # Row index -> [(token_id, logprob), ...], chosen token first, then the top-k
    # alternatives requested by that row.
    logprobs: dict[int, list[tuple[int, float]]]


class BatchedSampler:
    """Applies per-request sampling to one step's logits.

    Holds no per-request state: penalties are computed from the token ids the caller
    passes, so a request that is preempted and rebuilt samples identically.
    """

    def __init__(
        self, device: torch.device | str, *, generator_seed: int | None = None,
        max_generators: int = 1024,
    ):
        self.device = torch.device(device)
        self._generator: torch.Generator | None = None
        if generator_seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(generator_seed)
        # Seed -> generator, least recently used first. A generator must survive for the
        # whole request that uses it (recreating one mid-request would replay its first
        # draws), so eviction is LRU and only ever reaches seeds no live request has
        # used for `max_generators` distinct seeds.
        self._seeded: OrderedDict[int, torch.Generator] = OrderedDict()
        self._max_generators = max_generators

    def _row_generator(self, seed: int) -> torch.Generator:
        generator = self._seeded.get(seed)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            self._seeded[seed] = generator
            while len(self._seeded) > self._max_generators:
                self._seeded.popitem(last=False)
        else:
            self._seeded.move_to_end(seed)
        return generator

    def forget(self, seeds: list[int]) -> None:
        """Drop generators for finished requests, so the map cannot grow without bound."""
        for seed in seeds:
            self._seeded.pop(seed, None)

    # ------------------------------------------------------------------ the step
    def sample(
        self,
        logits: torch.Tensor,                       # [N, vocab], any float dtype
        params: list[SamplingParams],
        context_tokens: list[list[int]] | None = None,
    ) -> SampleResult:
        """Choose one token per row. `context_tokens[i]` is row i's penalty context."""
        if logits.ndim != 2:
            raise ValueError("logits must have shape [rows, vocab]")
        rows = logits.shape[0]
        if len(params) != rows:
            raise ValueError("one SamplingParams per logits row is required")
        if context_tokens is not None and len(context_tokens) != rows:
            raise ValueError("one context per logits row is required")

        wants_logprobs = any(p.logprobs is not None for p in params)
        # Greedy is not the same as "nothing to do": penalties change which token the
        # argmax picks, exactly as they do in `transformers`, so they are applied first
        # and only a batch that wants neither takes the fast path.
        plain = all(p.greedy and not p.has_penalties for p in params)
        if plain and not wants_logprobs:
            # The path every recorded measurement used: one kernel, no copies.
            return SampleResult(logits.argmax(dim=-1).tolist(), {})

        working = logits.float()
        if any(p.has_penalties for p in params) and context_tokens is not None:
            working = self._apply_penalties(working, params, context_tokens)
        working = self._apply_temperature(working, params)

        logprob_source = working if wants_logprobs else None
        filtered = self._filter(working, params)
        tokens = self._draw(filtered, params)
        chosen = tokens.tolist()
        return SampleResult(chosen, self._logprobs(logprob_source, params, chosen))

    # ------------------------------------------------------------------ stages
    def _apply_penalties(
        self, logits: torch.Tensor, params: list[SamplingParams],
        context_tokens: list[list[int]],
    ) -> torch.Tensor:
        """Repetition / presence / frequency penalties, one gather-scatter per batch.

        The seen tokens of every row are packed into one padded `[N, K]` index tensor;
        counts come from a scatter-add over a `[N, K]` ones tensor grouped by unique
        token, so nothing allocates a `[N, vocab]` buffer.
        """
        rows, vocab = logits.shape
        pad = vocab  # an index no real token uses; the row is masked below
        packed: list[list[int]] = []
        counts: list[list[float]] = []
        for row, tokens in enumerate(context_tokens):
            setting = params[row]
            if not setting.has_penalties or not tokens:
                packed.append([])
                counts.append([])
                continue
            occurrences: dict[int, int] = {}
            for token in tokens:
                occurrences[token] = occurrences.get(token, 0) + 1
            packed.append(list(occurrences))
            counts.append([float(value) for value in occurrences.values()])
        width = max((len(row) for row in packed), default=0)
        if width == 0:
            return logits
        index = torch.full((rows, width), pad, dtype=torch.long)
        count = torch.zeros((rows, width), dtype=torch.float32)
        for row, (tokens, occurrences) in enumerate(zip(packed, counts)):
            if tokens:
                index[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                count[row, :len(tokens)] = torch.tensor(occurrences, dtype=torch.float32)
        index = index.to(logits.device, non_blocking=True)
        count = count.to(logits.device, non_blocking=True)
        valid = index < vocab
        safe_index = index.masked_fill(~valid, 0)

        repetition = torch.tensor(
            [p.repetition_penalty for p in params], device=logits.device
        ).unsqueeze(1)
        presence = torch.tensor(
            [p.presence_penalty for p in params], device=logits.device
        ).unsqueeze(1)
        frequency = torch.tensor(
            [p.frequency_penalty for p in params], device=logits.device
        ).unsqueeze(1)

        gathered = logits.gather(1, safe_index)
        # transformers' convention: divide positive logits, multiply negative ones, so
        # the penalty always moves the logit towards -inf.
        penalized = torch.where(gathered > 0, gathered / repetition, gathered * repetition)
        penalized = penalized - presence - frequency * count
        penalized = torch.where(valid, penalized, gathered)
        return logits.scatter(1, safe_index, penalized)

    def _apply_temperature(
        self, logits: torch.Tensor, params: list[SamplingParams],
    ) -> torch.Tensor:
        # Greedy rows keep their logits unscaled; the draw stage takes their argmax, and
        # dividing by a placeholder would only add error.
        scale = torch.tensor(
            [1.0 if p.greedy else p.temperature for p in params], device=logits.device,
        ).unsqueeze(1)
        return logits / scale

    def _filter(self, logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
        """Mask out tokens excluded by top-k, top-p or min-p, per row."""
        rows, vocab = logits.shape
        top_k = [p.top_k if 0 < p.top_k < vocab and not p.greedy else 0 for p in params]
        top_p = [p.top_p if p.top_p < 1.0 and not p.greedy else 1.0 for p in params]
        min_p = [p.min_p if p.min_p > 0.0 and not p.greedy else 0.0 for p in params]

        if any(top_k):
            limit = max(top_k)
            kth = logits.topk(limit, dim=-1).values                       # [N, limit]
            # Row i keeps tokens at least as likely as its own k-th best; rows with
            # top_k disabled index the last column, whose threshold is never binding
            # because they are given -inf instead.
            index = torch.tensor(
                [(k - 1 if k else 0) for k in top_k], device=logits.device,
            ).unsqueeze(1)
            threshold = kth.gather(1, index)
            enabled = torch.tensor(
                [bool(k) for k in top_k], device=logits.device,
            ).unsqueeze(1)
            logits = torch.where(
                enabled & (logits < threshold), torch.full_like(logits, float("-inf")), logits,
            )

        if any(value > 0.0 for value in min_p):
            probabilities = logits.softmax(dim=-1)
            floor = probabilities.max(dim=-1, keepdim=True).values * torch.tensor(
                min_p, device=logits.device,
            ).unsqueeze(1)
            logits = torch.where(
                probabilities < floor, torch.full_like(logits, float("-inf")), logits,
            )

        if any(value < 1.0 for value in top_p):
            ordered, order = logits.sort(dim=-1, descending=True)
            cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
            limit = torch.tensor(top_p, device=logits.device).unsqueeze(1)
            # Keep every token up to and including the one that crosses the mass limit,
            # so the most likely token always survives even if it exceeds top_p alone.
            drop = cumulative - ordered.softmax(dim=-1) >= limit
            ordered = ordered.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(logits).scatter_(1, order, ordered)
        return logits

    def _draw(self, logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
        greedy_rows = [row for row, p in enumerate(params) if p.greedy]
        sampled_rows = [row for row, p in enumerate(params) if not p.greedy]
        tokens = torch.empty(len(params), dtype=torch.long, device=logits.device)
        if greedy_rows:
            index = torch.tensor(greedy_rows, device=logits.device)
            tokens[index] = logits.index_select(0, index).argmax(dim=-1)
        if not sampled_rows:
            return tokens
        probabilities = logits.softmax(dim=-1)
        unseeded = [row for row in sampled_rows if params[row].seed is None]
        if unseeded:
            index = torch.tensor(unseeded, device=logits.device)
            drawn = torch.multinomial(
                probabilities.index_select(0, index), 1, generator=self._generator,
            )
            tokens[index] = drawn.squeeze(1)
        for row in sampled_rows:
            seed = params[row].seed
            if seed is None:
                continue
            # A seeded row draws from its own generator so its output does not depend on
            # how many other rows shared the step. One extra launch per seeded row.
            generator = self._row_generator(seed)
            tokens[row] = torch.multinomial(
                probabilities[row], 1, generator=generator,
            ).squeeze(0)
        return tokens

    def _logprobs(
        self, logits: torch.Tensor | None, params: list[SamplingParams], chosen: list[int],
    ) -> dict[int, list[tuple[int, float]]]:
        """Chosen-token and top-k logprobs, from the penalized and temperature-scaled
        distribution (before top-k/top-p truncation, which is a sampling device rather
        than the model's opinion)."""
        if logits is None:
            return {}
        rows = [row for row, p in enumerate(params) if p.logprobs is not None]
        if not rows:
            return {}
        index = torch.tensor(rows, device=logits.device)
        selected = logits.index_select(0, index).log_softmax(dim=-1)
        wanted = max(params[row].logprobs or 0 for row in rows)
        top = selected.topk(wanted, dim=-1) if wanted else None
        picked = torch.tensor([chosen[row] for row in rows], device=logits.device).unsqueeze(1)
        picked_logprob = selected.gather(1, picked).squeeze(1).tolist()
        top_ids = top.indices.tolist() if top is not None else [[]] * len(rows)
        top_values = top.values.tolist() if top is not None else [[]] * len(rows)
        result: dict[int, list[tuple[int, float]]] = {}
        for position, row in enumerate(rows):
            count = params[row].logprobs or 0
            entries = [(chosen[row], picked_logprob[position])]
            entries.extend(
                (token, value) for token, value in
                zip(top_ids[position][:count], top_values[position][:count])
                if token != chosen[row]
            )
            result[row] = entries[:count + 1]
        return result

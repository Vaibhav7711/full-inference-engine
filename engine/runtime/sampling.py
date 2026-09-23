"""Per-request sampling parameters.

The engine sampled greedily (argmax) for its whole measurement history, which is what
made every A/B comparable against stock Transformers token for token. Real traffic needs
temperature, nucleus and top-k sampling, penalties, stop conditions and logprobs, and it
needs them *per request*, because a batch step advances many requests with different
settings at once. This dataclass is that per-request state; `engine.batching.sampler`
applies it to one step's logits in a batch.

Defaults are exactly greedy decoding, so a request that asks for nothing keeps the
behaviour every recorded benchmark was measured with.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    """What to do with one request's logits at each step.

    `temperature = 0` means greedy and takes precedence over every other knob, matching
    the OpenAI convention. `top_k <= 0` and `top_p >= 1` disable those filters.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    # Minimum probability relative to the most likely token (min-p sampling). 0 disables.
    min_p: float = 0.0
    # Divides (or multiplies, for negative logits) the logit of any token already seen,
    # as in `transformers`' RepetitionPenaltyLogitsProcessor. 1.0 disables.
    repetition_penalty: float = 1.0
    # Subtracted once for any seen token / per occurrence, as in the OpenAI API.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # Penalties apply to the prompt as well as the generated tokens when true. OpenAI
    # counts only the completion; `transformers` counts the whole context.
    penalize_prompt: bool = False
    # Per-request RNG seed. Reproducible regardless of what else is in the batch, at the
    # cost of sampling that row on its own generator (see the sampler's fast/slow paths).
    seed: int | None = None
    stop_token_ids: frozenset[int] = field(default_factory=frozenset)
    ignore_eos: bool = False
    # Number of alternatives to report per step, or None for no logprobs. The chosen
    # token's logprob is always included when this is set.
    logprobs: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative (0 disables it)")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2]")
        for name in ("presence_penalty", "frequency_penalty"):
            value = getattr(self, name)
            if not -2.0 <= value <= 2.0:
                raise ValueError(f"{name} must be in [-2, 2]")
        if self.logprobs is not None and not 0 <= self.logprobs <= 20:
            raise ValueError("logprobs must be in [0, 20]")
        if self.seed is not None and not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a non-negative 63-bit integer")
        object.__setattr__(self, "stop_token_ids", frozenset(self.stop_token_ids))

    @property
    def greedy(self) -> bool:
        """True when this request takes the argmax, whatever the other fields say."""
        return self.temperature == 0.0 or self.top_k == 1

    @property
    def has_penalties(self) -> bool:
        return (self.repetition_penalty != 1.0 or self.presence_penalty != 0.0
                or self.frequency_penalty != 0.0)

    @property
    def deterministic(self) -> bool:
        """True when repeating the request reproduces the tokens exactly."""
        return self.greedy or self.seed is not None

    @property
    def can_reuse_cached_prediction(self) -> bool:
        """Whether a prompt-only cache entry fully determines the next-token result.

        Exact-prefix entries store the model's first-token decision, not the logits.  A
        sampled request must execute sampling so its RNG advances, penalties can change
        the argmax, and requested logprobs cannot be reconstructed from a token id.
        """
        return self.greedy and not self.has_penalties and self.logprobs is None


GREEDY = SamplingParams()

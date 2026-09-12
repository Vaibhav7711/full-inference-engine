from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Iterator

import torch


@dataclass
class GenerationMetrics:
    tokenization_ms: float = 0.0
    prefill_ms: float = 0.0
    first_token_ms: float = 0.0
    ttft_ms: float = 0.0
    decode_ms: list[float] | None = None
    total_ms: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    def __post_init__(self) -> None:
        if self.decode_ms is None:
            self.decode_ms = []

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["mean_decode_ms"] = sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else 0.0
        return data


@contextmanager
def cuda_timed() -> Iterator[callable]:
    """Measure one GPU section using CUDA events, not an unsynchronized CPU timer."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested without CUDA")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    result: dict[str, float] = {}
    yield lambda: result["ms"]
    end.record()
    end.synchronize()
    result["ms"] = start.elapsed_time(end)


def cpu_elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000

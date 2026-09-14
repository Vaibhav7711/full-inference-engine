"""Shared, reproducible benchmark metadata and validation helpers."""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import torch
import transformers


def environment_record(device: torch.device | str = "cuda") -> dict[str, object]:
    """Describe the software and active CUDA device behind a measurement."""
    dev = torch.device(device)
    software: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    try:
        import triton

        software["triton"] = triton.__version__
    except ImportError:
        software["triton"] = None

    properties = torch.cuda.get_device_properties(dev)
    hardware = {
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "framework_bf16_supported": torch.cuda.is_bf16_supported(),
        "native_bf16_tensor_cores": properties.major >= 8,
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "software": software,
    }


def assert_repeatable_tokens(token_runs: list[list[int]]) -> None:
    """Reject a deterministic benchmark if repeated runs emit different tokens."""
    if token_runs and any(tokens != token_runs[0] for tokens in token_runs[1:]):
        raise RuntimeError("greedy output changed between measured benchmark runs")

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


def git_record(root: str | None = None) -> dict[str, object]:
    """Which code produced a measurement: commit, branch, and whether the tree was dirty.

    A result JSON without this cannot be tied back to the kernel or scheduler it measured,
    which is how three stale files ended up being the only committed evidence.
    """
    import subprocess

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "short": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def device_clock_record(index: int = 0) -> dict[str, object]:
    """Current SM/memory clocks, power and temperature from nvidia-smi, or {} off-GPU.

    Recorded before and after each run: a T4 throttles at its 70 W cap, so the same
    configuration measured cold and hot is two different experiments.
    """
    import subprocess

    fields = ["clocks.sm", "clocks.mem", "clocks.max.sm", "power.draw", "power.limit",
              "temperature.gpu", "clocks_throttle_reasons.active"]
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--id={index}", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    values = [item.strip() for item in raw.split(",")]
    record: dict[str, object] = {}
    for name, value in zip(fields, values):
        try:
            record[name] = float(value)
        except ValueError:
            record[name] = value
    return record

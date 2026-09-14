"""Fail early with actionable CUDA information before downloading a model."""

from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> None:
    record: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        record["gpu"] = properties.name
        record["gpu_memory_gib"] = round(properties.total_memory / 2**30, 2)
        record["capability"] = f"{properties.major}.{properties.minor}"
        record["bf16_supported"] = torch.cuda.is_bf16_supported()
        record["recommended_dtype"] = (
            "bfloat16" if properties.major >= 8 and torch.cuda.is_bf16_supported() else "float16"
        )
        if properties.major == 7 and properties.minor == 5:
            record["runtime_profile"] = "turing_t4"
    print(json.dumps(record, indent=2))
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU, then reconnect."
        )


if __name__ == "__main__":
    main()

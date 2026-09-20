"""Fail early with actionable CUDA information before downloading a model."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform

import torch


def _version(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="also write the record as JSON")
    args = parser.parse_args()
    # The journal's target block pins torch/triton/transformers; a re-measurement on a
    # drifted stack must say so next to its numbers rather than hide it.
    record: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": _version("triton"),
        "transformers": _version("transformers"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        record["gpu"] = properties.name
        record["gpu_memory_gib"] = round(properties.total_memory / 2**30, 2)
        record["capability"] = f"{properties.major}.{properties.minor}"
        record["framework_bf16_supported"] = torch.cuda.is_bf16_supported()
        record["native_bf16_tensor_cores"] = properties.major >= 8
        record["recommended_dtype"] = (
            "bfloat16"
            if properties.major >= 8 and torch.cuda.is_bf16_supported()
            else "float16"
        )
        if properties.major == 7 and properties.minor == 5:
            record["runtime_profile"] = "turing_t4"
    print(json.dumps(record, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=2)
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU, then reconnect."
        )


if __name__ == "__main__":
    main()

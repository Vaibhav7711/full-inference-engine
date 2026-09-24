"""Fail early unless a Kaggle-style multi-T4 CUDA environment is ready."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

import torch


def _version(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def _nvidia_smi_rows() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,clocks.sm,clocks.mem,power.limit,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 9:
            continue
        rows.append({
            "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
            "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]),
            "sm_clock_mhz": int(fields[5]), "memory_clock_mhz": int(fields[6]),
            "power_limit_w": float(fields[7]), "driver_version": fields[8],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-t4-count", type=int, default=2)
    parser.add_argument("--max-initial-used-mib", type=int, default=600)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.require_t4_count <= 0:
        parser.error("--require-t4-count must be positive")

    record: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": _version("triton"),
        "transformers": _version("transformers"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "nvidia_smi": _nvidia_smi_rows(),
    }
    errors: list[str] = []
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")
    elif torch.cuda.device_count() != args.require_t4_count:
        errors.append(
            f"requires exactly {args.require_t4_count} visible GPUs, found {torch.cuda.device_count()}"
        )

    if torch.cuda.is_available():
        gpus = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu = {
                "index": index,
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "memory_gib": round(properties.total_memory / 2**30, 3),
                "multiprocessors": properties.multi_processor_count,
            }
            gpus.append(gpu)
            if "T4" not in properties.name.upper():
                errors.append(f"GPU {index} is not a T4: {properties.name}")
            if (properties.major, properties.minor) != (7, 5):
                errors.append(
                    f"GPU {index} capability is {properties.major}.{properties.minor}, expected 7.5"
                )
        record["gpus"] = gpus
        record["peer_access"] = [
            [i == j or torch.cuda.can_device_access_peer(i, j)
             for j in range(torch.cuda.device_count())]
            for i in range(torch.cuda.device_count())
        ]

    smi_rows = record["nvidia_smi"]
    if isinstance(smi_rows, list):
        for row in smi_rows:
            if isinstance(row, dict) and int(row["memory_used_mib"]) >= args.max_initial_used_mib:
                errors.append(
                    f"GPU {row['index']} already uses {row['memory_used_mib']} MiB "
                    f"(limit {args.max_initial_used_mib} MiB)"
                )
    record["ok"] = not errors
    record["errors"] = errors
    payload = json.dumps(record, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    if errors:
        raise SystemExit("Kaggle T4 x2 preflight failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()

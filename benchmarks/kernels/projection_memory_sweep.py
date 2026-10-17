"""Measure weight traffic through the actual model's decode projections.

The complete projection sweep exceeds L2, so repeated graph replays must fetch model
weights from device memory. This isolates projection scheduling from attention and
sampling while retaining the model's real tensor shapes and layouts.

    python -m benchmarks.kernels.projection_memory_sweep \
        --model Qwen/Qwen3-0.6B --batches 1,8 --repeats 30
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")


def _projections(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in PROJECTIONS
    ]


def _capture(projections: list[tuple[str, nn.Linear]], batch: int) -> torch.cuda.CUDAGraph:
    inputs = {
        width: torch.randn(batch, width, device="cuda", dtype=projections[0][1].weight.dtype)
        for width in {module.in_features for _, module in projections}
    }

    def run() -> None:
        # Retain outputs until graph capture finishes; each projection's weights are
        # fetched once per replay, in model order, with no artificial host gaps.
        outputs = [F.linear(inputs[module.in_features], module.weight, module.bias)
                   for _, module in projections]
        if len(outputs) != len(projections):
            raise RuntimeError("incomplete projection sweep")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> list[float]:
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="results/rtx4060/projection_memory_sweep.json")
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if not batches or min(batches) < 1 or args.repeats < 1:
        parser.error("batches and repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    from engine.model import load_model

    loaded = load_model(args.model)
    projections = _projections(loaded.model)
    if not projections:
        raise RuntimeError("no compatible linear projections found")
    weight_bytes = sum(module.weight.numel() * module.weight.element_size()
                       for _, module in projections)
    rows = []
    for batch in batches:
        graph = _capture(projections, batch)
        samples = _time_graph(graph, args.repeats)
        median_ms = statistics.median(samples)
        row = {
            "batch": batch,
            "median_ms": median_ms,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "effective_weight_gb_s": weight_bytes / (median_ms / 1000) / 1e9,
        }
        rows.append(row)
        print(f"batch={batch:2d}  median={median_ms:.3f} ms  "
              f"effective weight rate={row['effective_weight_gb_s']:.1f} GB/s")
        del graph

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "model": args.model,
        "projection_count": len(projections),
        "weight_bytes": weight_bytes,
        "note": "Projection-only CUDA graph; includes arithmetic and output writes. "
                "Effective weight rate is a diagnostic, not measured DRAM traffic.",
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

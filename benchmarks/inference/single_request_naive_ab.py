"""Identical-workload single-request A/B against untouched Transformers DynamicCache.

Arms:
  * naive_fp16: ExplicitDecodeRunner and the stock model
  * engine_fp16: paged engine, one active request, decode CUDA graph
  * engine_w8a16_mlp: the same engine after selective MLP weight quantization

There is never more than one request, so this measures latency rather than continuous-
batching throughput. All arms use the same model instance and prompt in that order.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from time import perf_counter, perf_counter_ns

import torch

from engine.runtime import GenerationRequest


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _run_naive(loaded, prompt: str, max_new_tokens: int, warmups: int, runs: int) -> dict:
    from engine.model import ExplicitDecodeRunner

    runner = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    for _ in range(warmups):
        runner.generate(prompt, max_new_tokens=max_new_tokens)
    measured = [runner.generate(prompt, max_new_tokens=max_new_tokens) for _ in range(runs)]
    return {
        "tokens": measured[0].token_ids,
        "repeatable": all(result.token_ids == measured[0].token_ids for result in measured),
        "ttft_ms": _summary([result.metrics.ttft_ms for result in measured]),
        "end_to_end_ms": _summary([result.metrics.total_ms for result in measured]),
        "itl_ms": _summary([
            value for result in measured for value in result.metrics.decode_ms
        ]),
        "throughput_tokens_per_second": (
            sum(len(result.token_ids) for result in measured)
            / (sum(result.metrics.total_ms for result in measured) / 1000)
        ),
        "peak_allocated_bytes": max(result.metrics.peak_allocated_bytes for result in measured),
        "peak_reserved_bytes": max(result.metrics.peak_reserved_bytes for result in measured),
    }


def _run_engine(loaded, prompt: str, max_new_tokens: int, warmups: int, runs: int,
                num_blocks: int) -> dict:
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device,
        num_blocks=num_blocks, block_size=16, max_active=1, prefix_cache_blocks=0,
        cuda_graph_batch_sizes=(1,), prefill_cuda_graphs=False,
    )
    engine.warmup()

    def one() -> dict:
        engine.reset()
        torch.cuda.reset_peak_memory_stats(loaded.device)
        total_started = perf_counter()
        tokenize_started = perf_counter_ns()
        ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        tokenization_ms = (perf_counter_ns() - tokenize_started) / 1_000_000
        request = GenerationRequest(
            request_id="single", prompt_token_count=len(ids),
            max_new_tokens=max_new_tokens, prompt_token_ids=ids,
        )
        if not engine.submit(request):
            raise RuntimeError("single request was rejected")
        while engine.has_unfinished_requests:
            engine.step()
        torch.cuda.synchronize(loaded.device)
        end_to_end_ms = (perf_counter() - total_started) * 1000
        timestamps = request.token_timestamps_ns
        gaps = [(b - a) / 1_000_000 for a, b in zip(timestamps, timestamps[1:])]
        return {
            "tokens": request.output_token_ids,
            "ttft_ms": (request.time_to_first_token_ms() or 0.0) + tokenization_ms,
            "end_to_end_ms": end_to_end_ms,
            "itl_ms": gaps,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(loaded.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(loaded.device),
        }

    for _ in range(warmups):
        one()
    measured = [one() for _ in range(runs)]
    tokens = measured[0]["tokens"]
    payload = {
        "tokens": tokens,
        "repeatable": all(result["tokens"] == tokens for result in measured),
        "ttft_ms": _summary([result["ttft_ms"] for result in measured]),
        "end_to_end_ms": _summary([result["end_to_end_ms"] for result in measured]),
        "itl_ms": _summary([value for result in measured for value in result["itl_ms"]]),
        "throughput_tokens_per_second": (
            sum(len(result["tokens"]) for result in measured)
            / (sum(result["end_to_end_ms"] for result in measured) / 1000)
        ),
        "peak_allocated_bytes": max(result["peak_allocated_bytes"] for result in measured),
        "peak_reserved_bytes": max(result["peak_reserved_bytes"] for result in measured),
    }
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from benchmarks.common import environment_record
    from engine.kernels.w8a16_linear import quantize_mlp_w8a16
    from engine.model import load_model
    from engine.quantization.int8 import model_storage_bytes

    loaded = load_model(args.model, dtype="float16")
    prompt_tokens = len(loaded.tokenizer(args.prompt).input_ids)
    if args.num_blocks * 16 < prompt_tokens + args.max_new_tokens:
        parser.error("num-blocks is too small for the fixed workload")

    arms = {}
    print("Measuring untouched ExplicitDecodeRunner...")
    arms["naive_fp16"] = _run_naive(
        loaded, args.prompt, args.max_new_tokens, args.warmup_runs, args.runs,
    )
    fp16_storage = model_storage_bytes(loaded.model)

    print("Measuring single-active-request optimized FP16 engine...")
    arms["engine_fp16"] = _run_engine(
        loaded, args.prompt, args.max_new_tokens, args.warmup_runs, args.runs, args.num_blocks,
    )

    replaced = quantize_mlp_w8a16(loaded.model)
    w8_storage = model_storage_bytes(loaded.model)
    print(f"Measuring single-active-request selective W8A16 engine ({replaced} projections)...")
    arms["engine_w8a16_mlp"] = _run_engine(
        loaded, args.prompt, args.max_new_tokens, args.warmup_runs, args.runs, args.num_blocks,
    )

    baseline = arms["naive_fp16"]
    comparisons = {}
    for label in ("engine_fp16", "engine_w8a16_mlp"):
        arm = arms[label]
        comparisons[label] = {
            "itl_speedup": baseline["itl_ms"]["p50"] / arm["itl_ms"]["p50"],
            "end_to_end_speedup": baseline["end_to_end_ms"]["p50"] / arm["end_to_end_ms"]["p50"],
            "throughput_speedup": arm["throughput_tokens_per_second"] / baseline["throughput_tokens_per_second"],
            "ttft_speedup": baseline["ttft_ms"]["p50"] / arm["ttft_ms"]["p50"],
            "first_token_divergence": _divergence(baseline["tokens"], arm["tokens"]),
        }
        print(label, comparisons[label])

    payload = {
        "environment": environment_record(loaded.device),
        "config": vars(args) | {"output": str(args.output), "prompt_tokens": prompt_tokens},
        "model_storage_bytes": {"fp16": fp16_storage, "w8a16_mlp": w8_storage},
        "w8_replaced_projections": replaced,
        "arms": arms,
        "comparisons_vs_naive": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"model_storage_bytes": payload["model_storage_bytes"],
                      "comparisons_vs_naive": comparisons}, indent=2))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

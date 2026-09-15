"""A/B latency and accuracy gate for fused INT8 paged decode attention."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("expected positive comma-separated integers")
    return parsed


def _one_ms(callable_) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _paired_rounds(fp16_call, int8_call, warmup: int, repeats: int, rounds: int) -> tuple[list[float], list[float]]:
    """Interleave variants so clocks or thermal drift cannot favor one side."""
    for index in range(warmup):
        (fp16_call if index % 2 == 0 else int8_call)()
    torch.cuda.synchronize()
    fp16_rounds, int8_rounds = [], []
    for _ in range(rounds):
        fp16_samples, int8_samples = [], []
        for index in range(repeats):
            if index % 2 == 0:
                fp16_samples.append(_one_ms(fp16_call))
                int8_samples.append(_one_ms(int8_call))
            else:
                int8_samples.append(_one_ms(int8_call))
                fp16_samples.append(_one_ms(fp16_call))
        fp16_rounds.append(statistics.median(fp16_samples))
        int8_rounds.append(statistics.median(int8_samples))
    return fp16_rounds, int8_rounds


def _case(batch: int, sequence_length: int, block_size: int = 16):
    q_heads, kv_heads, dim = 16, 8, 128
    blocks_per_sequence = (sequence_length + block_size - 1) // block_size
    total_blocks = batch * blocks_per_sequence
    tables = torch.arange(
        total_blocks - 1, -1, -1, device="cuda", dtype=torch.int32,
    ).view(batch, -1)
    lengths = torch.full((batch,), sequence_length, device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, 1, dim, device="cuda", dtype=torch.float16)
    key_fp16 = torch.randn(total_blocks, block_size, kv_heads, dim, device="cuda", dtype=torch.float16)
    value_fp16 = torch.randn_like(key_fp16)
    key_scale = key_fp16.float().abs().amax(dim=-1).clamp_min(1e-8).div(127).to(torch.float16)
    value_scale = value_fp16.float().abs().amax(dim=-1).clamp_min(1e-8).div(127).to(torch.float16)
    key_int8 = torch.round(key_fp16.float() / key_scale[..., None]).clamp(-127, 127).to(torch.int8)
    value_int8 = torch.round(value_fp16.float() / value_scale[..., None]).clamp(-127, 127).to(torch.int8)
    return query, key_fp16, value_fp16, key_int8, value_int8, key_scale, value_scale, tables, lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default="256,1024,2048")
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="results/int8_paged_decode_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")

    from engine.kernels.int8_paged_kv import paged_decode_batched_int8
    from engine.kernels.paged_decode_batched import paged_decode_batched

    results: dict[str, object] = {"config": vars(args), "rows": []}
    print("\nINT8 paged decode A/B (Qwen3-0.6B geometry: H=16, KVH=8, D=128)")
    print(f"{'context':>8} {'batch':>6} {'fp16 ms':>10} {'int8 ms':>10} {'speedup':>9} {'range':>13} {'rel err':>9} {'KV save':>9}")
    for sequence_length in _parse_ints(args.seq_lens):
        for batch in _parse_ints(args.batches):
            tensors = _case(batch, sequence_length)
            query, key_fp16, value_fp16, key_int8, value_int8, key_scale, value_scale, tables, lengths = tensors
            fp16_call = lambda: paged_decode_batched(query, key_fp16, value_fp16, tables, lengths, block_n=128)
            int8_call = lambda: paged_decode_batched_int8(
                query, key_int8, value_int8, key_scale, value_scale, tables, lengths, block_n=128,
            )
            fp16_output = fp16_call()
            int8_output = int8_call()
            torch.cuda.synchronize()
            fp16_rounds, int8_rounds = _paired_rounds(
                fp16_call, int8_call, args.warmup, args.repeats, args.rounds,
            )
            fp16_ms, int8_ms = statistics.median(fp16_rounds), statistics.median(int8_rounds)
            round_speedups = [left / right for left, right in zip(fp16_rounds, int8_rounds)]
            relative_error = float(
                (int8_output.float() - fp16_output.float()).abs().mean()
                / fp16_output.float().abs().mean().clamp_min(1e-5)
            )
            fp16_bytes = key_fp16.numel() * 2 + value_fp16.numel() * 2
            int8_bytes = key_int8.numel() + value_int8.numel() + (key_scale.numel() + value_scale.numel()) * 2
            row = {
                "sequence_length": sequence_length, "batch": batch,
                "fp16_median_ms": fp16_ms, "int8_median_ms": int8_ms,
                "int8_over_fp16_speedup": fp16_ms / int8_ms,
                "round_speedups": round_speedups,
                "round_speedup_min": min(round_speedups),
                "round_speedup_max": max(round_speedups),
                "relative_output_error": relative_error,
                "fp16_kv_bytes": fp16_bytes, "int8_kv_bytes_including_scales": int8_bytes,
                "kv_storage_reduction_fraction": 1 - int8_bytes / fp16_bytes,
            }
            results["rows"].append(row)
            print(f"{sequence_length:>8} {batch:>6} {fp16_ms:>10.4f} {int8_ms:>10.4f} "
                  f"{row['int8_over_fp16_speedup']:>8.2f}x "
                  f"{min(round_speedups):>5.2f}-{max(round_speedups):>5.2f}x {relative_error:>9.4f} "
                  f"{row['kv_storage_reduction_fraction']:>8.1%}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()

"""Kernel-time share of one decode step by category (eager, so kernels are visible).

Decides whether elementwise/fused kernels are worth tuning: GEMMs, attention, norms,
RoPE, SwiGLU, KV writes, sampling and everything else, as fractions of GPU kernel time.

    python -m benchmarks.kernels.decode_kernel_share --out results/rtx4060/decode_kernel_share_1.7b.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

CATEGORIES = (
    ("attention", ("paged_decode", "decode_batched", "split_kernel", "merge_split")),
    ("kv_write", ("write_decode", "kv_write")),
    ("rmsnorm", ("rmsnorm", "rms_norm")),
    ("rope", ("rope",)),
    ("swiglu", ("swiglu", "silu")),
    ("w8a16_gemm", ("w8a16",)),
    ("gemm", ("gemm", "gemv", "cutlass", "sm80_xmma", "sm89", "ampere", "cublas", "splitk")),
    ("sampling", ("argmax", "reduce", "topk")),
)


def classify(name: str) -> str:
    lowered = name.lower()
    for category, keys in CATEGORIES:
        if any(key in lowered for key in keys):
            return category
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--w8a16-mlp", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from torch.profiler import ProfilerActivity, profile

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model
    from engine.runtime import GenerationRequest

    loaded = load_model(args.model, dtype="float16")
    if args.w8a16_mlp:
        from engine.kernels.w8a16_linear import quantize_mlp_w8a16
        quantize_mlp_w8a16(loaded.model)
    engine = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, "cuda", num_blocks=512,
                                      max_active=args.batch, prefix_cache_blocks=0)
    engine.eos_ids.clear()
    ids = loaded.tokenizer("Paged attention maps logical blocks to physical memory. " * 80,
                           return_tensors="pt").input_ids[0].tolist()[:args.prompt_tokens]
    for index in range(args.batch):
        engine.submit(GenerationRequest(f"r{index}", len(ids), args.steps + 40, prompt_token_ids=ids))
    while not engine.scheduler.active or any(r.state.name != "DECODING" for r in engine.scheduler.active.values()) \
            or len(engine.scheduler.active) < args.batch:
        engine.step()
    for _ in range(4):
        engine.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            engine.step()
        torch.cuda.synchronize()
    totals, counts = defaultdict(float), defaultdict(int)
    for event in prof.key_averages():
        time_us = getattr(event, "device_time_total", None) or getattr(event, "cuda_time_total", 0)
        if time_us and event.key and not event.key.startswith(("ProfilerStep", "cudaMemcpy")):
            category = classify(event.key)
            totals[category] += time_us
            counts[category] += event.count
    total = sum(totals.values())
    shares = {k: {"share": v / total, "us_per_step": v / args.steps, "launches_per_step": counts[k] / args.steps}
              for k, v in sorted(totals.items(), key=lambda item: -item[1])}
    for k, v in shares.items():
        print(f"{k:12s} {v['share']:6.1%}  {v['us_per_step']:8.1f} us/step  {v['launches_per_step']:6.0f} launches/step")
    with open(args.out, "w") as handle:
        json.dump({"args": vars(args), "gpu": torch.cuda.get_device_name(), "note": "eager decode; "
                   "graph replay removes launch gaps but not kernel time", "shares": shares}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

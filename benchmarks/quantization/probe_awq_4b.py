"""Probe a real Qwen3-4B-AWQ projection without changing the model loader.

The checkpoint is read locally. A reference dequantization verifies the packed
order, then an experimental direct-from-packed GEMV is timed against FP16 GEMM.
This is a projection gate, not a model-quality or serving benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from engine.kernels.awq_gemv import awq_gemv
from engine.quantization.awq_tinygemm import repack_awq_to_tinygemm


def _dequant_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=qweight.device)
    shifts = 4 * torch.arange(8, device=qweight.device)
    weight = ((qweight[:, :, None] >> shifts) & 15).reshape(qweight.shape[0], -1)
    zeros = ((qzeros[:, :, None] >> shifts) & 15).reshape(qzeros.shape[0], -1)
    groups = torch.arange(weight.shape[1], device=qweight.device).reshape(-1, 8)[:, order].reshape(-1)
    weight = weight[:, groups]
    zeros = zeros[:, groups]
    return ((weight - zeros.repeat_interleave(128, dim=0)) *
            scales.repeat_interleave(128, dim=0)).to(torch.float16)


def _time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/Qwen3-4B-AWQ/model.safetensors")
    parser.add_argument("--projections", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--arms", default="dequant_fp16,tinygemm_native,direct_awq",
                        help="comma-separated: dequant_fp16,tinygemm_native,direct_awq")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--output", default="results/rtx4060/awq_4b_projection_probe.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    prefix = {
        "q_proj": "model.layers.0.self_attn.q_proj",
        "k_proj": "model.layers.0.self_attn.k_proj",
        "v_proj": "model.layers.0.self_attn.v_proj",
        "o_proj": "model.layers.0.self_attn.o_proj",
        "gate_proj": "model.layers.0.mlp.gate_proj",
        "up_proj": "model.layers.0.mlp.up_proj",
        "down_proj": "model.layers.0.mlp.down_proj",
    }
    chosen = args.projections.split(",")
    if any(name not in prefix for name in chosen):
        parser.error(f"--projections must use {sorted(prefix)}")
    batches = [int(item) for item in args.batches.split(",")]
    if any(batch < 1 for batch in batches):
        parser.error("batches must be positive")
    selected_arms = set(args.arms.split(","))
    valid_arms = {"dequant_fp16", "tinygemm_native", "direct_awq"}
    if "dequant_fp16" not in selected_arms or selected_arms - valid_arms:
        parser.error(f"--arms must include dequant_fp16 and use only {sorted(valid_arms)}")

    rng = random.Random(4406)
    rows = []
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for name in chosen:
            key = prefix[name]
            qweight = handle.get_tensor(f"{key}.qweight").to("cuda")
            qzeros = handle.get_tensor(f"{key}.qzeros").to("cuda")
            scales = handle.get_tensor(f"{key}.scales").to("cuda")
            weight = _dequant_awq(qweight, qzeros, scales)
            tinygemm_weight, tinygemm_qparams = repack_awq_to_tinygemm(qweight, qzeros, scales)
            torch.manual_seed(4406)
            for batch in batches:
                x = torch.randn((batch, qweight.shape[0]), device="cuda", dtype=torch.float16)
                reference = x @ weight
                arms = {"dequant_fp16": lambda: x @ weight}
                if "tinygemm_native" in selected_arms:
                    arms["tinygemm_native"] = lambda: torch.ops.aten._weight_int4pack_mm(
                        x.to(torch.bfloat16), tinygemm_weight, 128, tinygemm_qparams,
                    ).to(x.dtype)
                if "direct_awq" in selected_arms:
                    for block_k, block_n in ((32, 16), (32, 32), (64, 16), (64, 32)):
                        arms[f"awq_{block_k}x{block_n}"] = (
                            lambda bk=block_k, bn=block_n: awq_gemv(
                                x, qweight, qzeros, scales, block_k=bk, block_n=bn,
                            )
                        )
                    for splits in (2, 4, 8):
                        arms[f"awq_64x32_split{splits}"] = (
                            lambda sp=splits: awq_gemv(
                                x, qweight, qzeros, scales, block_k=64, block_n=32,
                                splits=sp,
                            )
                        )
                errors = {}
                for arm, fn in arms.items():
                    actual = fn()
                    rel = float((actual.float() - reference.float()).abs().mean() /
                                reference.float().abs().mean().clamp_min(1e-5))
                    if not torch.isfinite(actual).all() or rel > 0.02:
                        raise RuntimeError(f"{name}/{batch}/{arm} failed relative parity: {rel}")
                    errors[arm] = rel
                    for _ in range(8):
                        fn()
                torch.cuda.synchronize()
                timings = {arm: [] for arm in arms}
                for _ in range(args.rounds):
                    order = list(arms)
                    rng.shuffle(order)
                    for arm in order:
                        timings[arm].append(_time_ms(arms[arm]))
                for arm in arms:
                    row = {
                        "projection": name, "batch": batch, "arm": arm,
                        "median_ms": statistics.median(timings[arm]),
                        "relative_error_vs_dequant_fp16": errors[arm],
                        "packed_weight_bytes": qweight.numel() * qweight.element_size() +
                            qzeros.numel() * qzeros.element_size() +
                            scales.numel() * scales.element_size(),
                        "dequant_weight_bytes": weight.numel() * weight.element_size(),
                    }
                    rows.append(row)
                    print(row, flush=True)
            del weight, qweight, qzeros, scales, tinygemm_weight, tinygemm_qparams

    totals = []
    for batch in batches:
        batch_rows = [row for row in rows if row["batch"] == batch]
        baseline = sum(row["median_ms"] for row in batch_rows if row["arm"] == "dequant_fp16")
        arms = sorted({row["arm"] for row in batch_rows if row["arm"] != "dequant_fp16"})
        for arm in arms:
            selected = [row for row in batch_rows if row["arm"] == arm]
            if len(selected) == len(chosen):
                total = sum(row["median_ms"] for row in selected)
                totals.append({"batch": batch, "arm": arm, "layer_projection_ms": total,
                               "speedup_vs_dequant_fp16": baseline / total})
    payload = {"gpu": torch.cuda.get_device_name(), "checkpoint": str(checkpoint),
               "config": vars(args), "rows": rows, "layer_projection_totals": totals}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

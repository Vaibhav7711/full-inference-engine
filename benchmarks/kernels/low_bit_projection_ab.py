"""Compare FP16, experimental INT8, and PyTorch native INT4 decode projections.

This is a projection microbenchmark, not an end-to-end engine benchmark. All modes
use the same input and original FP16 weight; weights are quantized before timing.
INT4 uses PyTorch's packed tinygemm operation, which is separate from AWQ packing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F


def _int4_pack(weight: torch.Tensor, group_size: int = 128):
    """Pack symmetric groupwise INT4 for aten tinygemm (not AWQ format)."""
    n, k = weight.shape
    if n % 8 or k % group_size or k % 128:
        raise ValueError("INT4 tinygemm requires N multiple of 8 and K aligned to 128")
    grouped = weight.float().reshape(n, k // group_size, group_size)
    minimum, maximum = grouped.amin(dim=-1), grouped.amax(dim=-1)
    scale = ((maximum - minimum) / 15.0).clamp_min(1e-8)
    # tinygemm stores a floating zero point: q=8 reconstructs to `zero`, so
    # q=0 reconstructs to the group minimum and q=15 to its maximum.
    zero = minimum + 8.0 * scale
    unsigned = ((grouped - (zero[:, :, None] - 8.0 * scale[:, :, None]))
                / scale[:, :, None]).round().clamp(0, 15).to(torch.int32)
    unsigned = unsigned.reshape(n, k)
    packed_nibbles = ((unsigned[:, 0::2] << 4) | unsigned[:, 1::2]).to(torch.uint8)
    packed = torch.ops.aten._convert_weight_to_int4pack(packed_nibbles.contiguous(), 8)
    scale_and_zero = torch.stack((scale.T, zero.T), dim=-1).to(torch.bfloat16)
    return packed, scale_and_zero.contiguous()


def _median_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--geometry", choices=("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b"),
                        default="qwen3_0_6b")
    parser.add_argument("--modes", default="fp16,w8a16,w8a8,w4a16_tinygemm,fp8_dynamic")
    parser.add_argument("--real-weights", action="store_true",
                        help="use layer-0 Qwen3-1.7B checkpoint weights from local HF cache")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--output", default="results/rtx4060/low_bit_projection_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    selected_modes = set(args.modes.split(","))
    valid_modes = {"fp16", "w8a16", "w8a8", "w4a16_tinygemm", "fp8_dynamic"}
    if not selected_modes or selected_modes - valid_modes:
        parser.error(f"--modes must be a comma-separated subset of {sorted(valid_modes)}")
    if args.real_weights and args.geometry != "qwen3_1_7b":
        parser.error("--real-weights currently supports only qwen3_1_7b")

    from engine.kernels.w8a16_linear import quantize_weight_per_channel, w8a16_linear
    from engine.kernels.w8a8_linear import pack_w8a8_weight, w8a8_linear

    geometries = {
        "qwen3_0_6b": (1024, 3072, 2048, 1024),
        "qwen3_1_7b": (2048, 6144, 2048, 1024),
        "qwen3_4b": (2560, 9728, 4096, 1024),
    }
    hidden, intermediate, q_width, kv_width = geometries[args.geometry]
    shapes = [
        ("q_proj", q_width, hidden),
        ("kv_proj", kv_width, hidden),
        ("gate_up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
    ]
    if args.include_lm_head:
        shapes.append(("lm_head", 151936, hidden))
    if args.real_weights:
        from huggingface_hub import try_to_load_from_cache
        from safetensors import safe_open

        index_path = try_to_load_from_cache("Qwen/Qwen3-1.7B", "model.safetensors.index.json")
        if not isinstance(index_path, str):
            raise FileNotFoundError("Qwen3-1.7B safetensors index is not cached locally")
        checkpoint = Path(index_path).parent
        index = json.loads(Path(index_path).read_text())["weight_map"]
        paths = {
            "q_proj": "model.layers.0.self_attn.q_proj.weight",
            "kv_proj": "model.layers.0.self_attn.k_proj.weight",
            "gate_up_proj": "model.layers.0.mlp.gate_proj.weight",
            "down_proj": "model.layers.0.mlp.down_proj.weight",
        }
    rows = []
    torch.manual_seed(73)
    for name, n, k in shapes:
        if args.real_weights and name in paths:
            key = paths[name]
            with safe_open(checkpoint / index[key], framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(key).to("cuda", dtype=torch.float16)
            if weight.shape != (n, k):
                raise RuntimeError(f"{key} has shape {tuple(weight.shape)}, expected {(n, k)}")
        else:
            weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
        qweight, w8_scales = quantize_weight_per_channel(weight)
        w8_packed = pack_w8a8_weight(qweight)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        fp8_weight_scale = weight.float().abs().amax().clamp_min(1e-8) / fp8_max
        fp8_weight = (weight.t().float() / fp8_weight_scale).clamp(
            -fp8_max, fp8_max,
        ).to(torch.float8_e4m3fn).contiguous()
        try:
            w4_packed, w4_qparams = _int4_pack(weight)
            w4_error = None
        except Exception as exc:
            w4_packed = w4_qparams = None
            w4_error = repr(exc)

        for batch in (int(item) for item in args.batches.split(",") if item):
            x = torch.randn(batch, k, device="cuda", dtype=torch.float16)
            reference = F.linear(x, weight)
            methods = {
                "fp16": lambda: F.linear(x, weight),
                "w8a16": lambda: w8a16_linear(x, qweight, w8_scales),
                "w8a8": lambda: w8a8_linear(x, w8_packed, w8_scales),
                "fp8_dynamic": lambda: (
                    lambda activation_scale: torch._scaled_mm(
                        (x.float() / activation_scale).clamp(-fp8_max, fp8_max)
                        .to(torch.float8_e4m3fn).contiguous(),
                        fp8_weight, activation_scale, fp8_weight_scale,
                        out_dtype=x.dtype,
                    )
                )(x.float().abs().amax().clamp_min(1e-8) / fp8_max),
            }
            if w4_packed is not None:
                methods["w4a16_tinygemm"] = lambda: torch.ops.aten._weight_int4pack_mm(
                    x.to(torch.bfloat16), w4_packed, 128, w4_qparams
                ).to(x.dtype)
            for mode, fn in methods.items():
                if mode not in selected_modes:
                    continue
                try:
                    output = fn()
                    relative_error = float((reference.float() - output.float()).abs().mean()
                                           / reference.float().abs().mean().clamp_min(1e-5))
                    if mode != "fp16" and (not torch.isfinite(output).all() or relative_error > 0.2):
                        raise RuntimeError(f"relative error {relative_error:.4f} exceeds 0.2")
                    elapsed_ms = _median_ms(fn, args.warmup, args.repeats)
                    row = {"shape": name, "batch": batch, "mode": mode,
                           "median_ms": elapsed_ms, "relative_error": relative_error}
                except Exception as exc:
                    row = {"shape": name, "batch": batch, "mode": mode, "error": repr(exc)}
                rows.append(row)
                print(row)
            if w4_error and "w4a16_tinygemm" in selected_modes:
                row = {"shape": name, "batch": batch, "mode": "w4a16_tinygemm", "error": w4_error}
                rows.append(row)
                print(row)

    payload = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "scope": f"{'Layer-0 checkpoint' if args.real_weights else 'Synthetic'} weights at {args.geometry} projection shapes; "
                        "per-call CUDA events, no model serving or AWQ packing",
               "rows": rows, "config": vars(args)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

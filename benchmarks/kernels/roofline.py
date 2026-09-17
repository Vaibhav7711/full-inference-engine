"""Measure the decode-step floor on the GPU actually present, instead of quoting a spec.

A decode step is memory-bound at small batch: every model weight is read once to produce
one token per sequence, and the arithmetic is a rounding error beside that traffic. So the
floor is (bytes that must be read) / (bytes per second this GPU actually delivers).

Both halves are measured here rather than assumed:

- **Bandwidth** comes from probes run on this device, not from the datasheet. Real
  memory-bound kernels reach some fraction of the sticker number, and which fraction is
  exactly the thing being argued about when someone says "70%" or "85%". The headline
  probe is a large fp16 matrix-vector product, because that *is* the decode operation:
  read an enormous matrix, touch a tiny vector, write almost nothing.
- **Bytes** come from the loaded checkpoint's own parameters, classified by whether a
  decode step actually reads them. An embedding table is not read in full - a decode step
  gathers one row per sequence. A tied `lm_head` shares storage with that table but *is*
  read in full, every step, across the whole vocabulary.

The result is an empirical floor: what this engine would cost per step if it did nothing
but move the necessary bytes at the rate this card demonstrably achieves.

    python -m benchmarks.kernels.roofline --measured-itl-ms 15.5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class BandwidthProbe:
    name: str
    description: str
    bytes_moved: int
    ms: float

    @property
    def gb_per_s(self) -> float:
        return self.bytes_moved / (self.ms / 1000) / 1e9


def _time_ms(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Median wall time of `fn` on the GPU, using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def measure_bandwidth(size_mb: int = 512, dtype=torch.float16) -> list[BandwidthProbe]:
    """Probe achieved bandwidth several ways; the GEMV probe is the representative one.

    Buffers are far larger than the T4's 4 MB L2, so nothing is served from cache - which
    is also true of a 1.2 GB model, and is why a microbenchmark on a small matrix would
    flatter the result.
    """
    element = torch.tensor([], dtype=dtype).element_size()
    count = (size_mb * 1024 * 1024) // element
    probes: list[BandwidthProbe] = []

    src = torch.randn(count, dtype=dtype, device="cuda")
    dst = torch.empty_like(src)
    ms = _time_ms(lambda: dst.copy_(src))
    probes.append(BandwidthProbe(
        "copy", "read + write a large buffer (STREAM-style copy)", 2 * count * element, ms,
    ))

    ms = _time_ms(lambda: torch.sum(src))
    probes.append(BandwidthProbe(
        "reduction", "read-only sweep of a large buffer", count * element, ms,
    ))

    # The decode pattern: one huge matrix read, one tiny vector, negligible output.
    hidden = 4096
    rows = count // hidden
    matrix = src[: rows * hidden].view(rows, hidden)
    vector = torch.randn(hidden, dtype=dtype, device="cuda")
    ms = _time_ms(lambda: torch.mv(matrix, vector))
    probes.append(BandwidthProbe(
        "gemv", "fp16 matrix-vector product - the decode weight-read pattern",
        rows * hidden * element, ms,
    ))
    del src, dst, matrix
    torch.cuda.empty_cache()
    return probes


@dataclass
class WeightBytes:
    layers: int
    lm_head: int
    other: int
    embedding_table: int
    lm_head_is_tied: bool

    @property
    def read_per_decode_step(self) -> int:
        """Bytes a single decode step must read, whatever the batch size.

        The embedding table is excluded: a decode step gathers one row per sequence, which
        is kilobytes. `lm_head` is included in full - every step projects onto the entire
        vocabulary, and when it is tied to the embedding table it is the same memory being
        read for a completely different purpose.
        """
        return self.layers + self.lm_head + self.other


def measure_weight_bytes(model) -> WeightBytes:
    layers = lm_head = other = embedding = 0
    lm_head_param = None
    embed_param = None
    for name, param in model.named_parameters():
        size = param.numel() * param.element_size()
        if "lm_head" in name:
            lm_head += size
            lm_head_param = param
        elif "embed_tokens" in name or "wte" in name:
            embedding += size
            embed_param = param
        elif ".layers." in name or ".h." in name:
            layers += size
        else:
            other += size
    tied = False
    if lm_head == 0 and embed_param is not None:
        # Tied weights: the output projection is not a separate parameter, but the step
        # still reads the whole table to produce logits.
        lm_head = embedding
        tied = True
    elif lm_head_param is not None and embed_param is not None:
        tied = lm_head_param.data_ptr() == embed_param.data_ptr()
    return WeightBytes(layers=layers, lm_head=lm_head, other=other,
                       embedding_table=embedding, lm_head_is_tied=tied)


def kv_bytes_per_step(model, *, batch: int, context_tokens: int, kv_dtype_bytes: int = 2) -> int:
    """KV bytes attention must read for one decode step.

    The ideal figure: each KV group read once. The current batched decode kernel assigns
    one program per query head, so a GQA group is read once per query head that shares it -
    twice for Qwen3-0.6B. That inefficiency belongs in the measured number, not the floor.
    """
    cfg = model.config
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    per_token = 2 * kv_heads * head_dim * kv_dtype_bytes  # K and V
    return per_token * context_tokens * batch * cfg.num_hidden_layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--probe-mb", type=int, default=512)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--context", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--measured-itl-ms", type=float, default=None,
                        help="observed ms between tokens, to report as a multiple of the floor")
    parser.add_argument("--itl-batch", type=int, default=8,
                        help="batch size the measured ITL was taken at")
    parser.add_argument("--itl-context", type=int, default=512,
                        help="approximate context length the measured ITL was taken at")
    parser.add_argument("--out", default="results/roofline.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("roofline requires CUDA")
        return 1
    device_name = torch.cuda.get_device_name(0)
    print(f"device: {device_name}")

    probes = measure_bandwidth(args.probe_mb)
    print("\nachieved bandwidth")
    for probe in probes:
        print(f"  {probe.name:10s} {probe.gb_per_s:7.1f} GB/s   {probe.description}")
    gemv = next(p for p in probes if p.name == "gemv")
    achieved = gemv.gb_per_s
    print(f"\nusing the gemv probe as the decode-representative rate: {achieved:.1f} GB/s")

    from engine.model import load_model
    loaded = load_model(args.model)
    weights = measure_weight_bytes(loaded.model)
    weight_read = weights.read_per_decode_step
    print(f"\nbytes read per decode step (batch-independent)")
    print(f"  transformer layers  {weights.layers / 1e6:8.1f} MB")
    print(f"  lm_head             {weights.lm_head / 1e6:8.1f} MB"
          f"{'  (tied to embedding table)' if weights.lm_head_is_tied else ''}")
    print(f"  other               {weights.other / 1e6:8.1f} MB")
    print(f"  embedding table     {weights.embedding_table / 1e6:8.1f} MB  (gathered, not swept)")
    print(f"  TOTAL               {weight_read / 1e6:8.1f} MB")

    weight_floor_ms = weight_read / (achieved * 1e9) * 1000
    print(f"\nweight-only floor: {weight_floor_ms:.2f} ms/step at {achieved:.0f} GB/s")

    grid = []
    print("\nfloor including KV traffic (ms per decode step = ms between tokens per caller)")
    header = "  batch " + "".join(f"{c:>12}" for c in args.context)
    print(header + "   context tokens each")
    for batch in args.batch:
        cells = []
        for context in args.context:
            kv = kv_bytes_per_step(loaded.model, batch=batch, context_tokens=context)
            floor_ms = (weight_read + kv) / (achieved * 1e9) * 1000
            cells.append(floor_ms)
            grid.append({
                "batch": batch, "context": context, "kv_mb": kv / 1e6,
                # One step advances every sequence by one token, so step time *is* the
                # inter-token latency each caller experiences. Dividing by batch gives
                # aggregate throughput instead, which is a different question.
                "floor_step_ms": floor_ms,
                "floor_itl_ms": floor_ms,
                "floor_ms_per_token_aggregate": floor_ms / batch,
            })
        print(f"  {batch:5d} " + "".join(f"{c:12.2f}" for c in cells))

    payload = {
        "device": device_name,
        "probes": [{**asdict(p), "gb_per_s": p.gb_per_s} for p in probes],
        "achieved_gb_per_s": achieved,
        "weight_bytes": asdict(weights),
        "weight_bytes_read_per_step": weight_read,
        "weight_only_floor_ms": weight_floor_ms,
        "grid": grid,
    }

    if args.measured_itl_ms:
        # Compare against a mid-sized operating point, not the empty-context best case,
        # so the ratio is not flattered. Inter-token latency is compared against *step*
        # time: a step advances every active sequence by exactly one token, so a caller's
        # gap between tokens is one step, whatever the batch size.
        reference = min(
            grid, key=lambda row: abs(row["batch"] - args.itl_batch)
            + abs(row["context"] - args.itl_context) / 100
        )
        ratio = args.measured_itl_ms / reference["floor_itl_ms"]
        payload["measured_itl_ms"] = args.measured_itl_ms
        payload["reference_point"] = reference
        payload["multiple_of_floor"] = ratio
        print(f"\nmeasured {args.measured_itl_ms:.2f} ms/token vs floor "
              f"{reference['floor_itl_ms']:.2f} ms at batch {reference['batch']}, "
              f"{reference['context']} ctx = {ratio:.1f}x the floor")
        print(f"  headroom if closed: {args.measured_itl_ms - reference['floor_itl_ms']:.2f} ms/token")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

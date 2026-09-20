"""Measure the prefill attention kernel in isolation and fit its cost per token.

The engine sweep measures a whole step: attention, MLP, weight reads, the decode the step
also performs, and scheduling. That is the right number for deciding what to build, but the
wrong one for iterating on a kernel, because a 2x kernel win shows up as a fraction of a
step and is easy to lose in noise.

This runs the attention kernel alone across chunk sizes and prefix lengths, reports
effective KV bandwidth, and fits `ms = a + b * chunk_tokens` per kernel so the two can be
compared on the same axis the Phase B sweep used.

    python -m benchmarks.kernels.prefill_attention_ab
    python -m benchmarks.kernels.prefill_attention_ab --sweep-tiles
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from benchmarks.reliability.sweep import fit_line
from engine.kernels.paged_prefill import paged_prefill
from engine.kernels.tiled_paged_prefill import tiled_paged_prefill

HEAD_DIM = 128
BLOCK_SIZE = 16
LAYERS = 28          # Qwen3-0.6B: a step runs the attention kernel once per layer
BYTES = 2


def kernel_diagnostics(kernel_fn) -> dict:
    """Registers, spills and shared memory of the most recently compiled variant.

    Triton records these per compiled kernel and they are the only way to tell an
    occupancy problem from a spilling one from a memory one. Reading them is the
    difference between diagnosing and guessing: a kernel moving 60x less data while
    running 3.5x slower is not memory-bound, and n_spills says so directly.
    """
    try:
        cache = getattr(kernel_fn, "cache", {})
        compiled = [k for device in cache.values() for k in device.values()]
        if not compiled:
            return {}
        latest = compiled[-1]
        report = {
            "n_regs": getattr(latest, "n_regs", None),
            "n_spills": getattr(latest, "n_spills", None),
            "shared_bytes": getattr(latest, "metadata", None)
            and getattr(latest.metadata, "shared", None),
        }
        report.update(ptx_report(getattr(latest, "asm", {}).get("ptx", "")))
        return report
    except Exception:
        return {}


def ptx_report(ptx: str) -> dict:
    """What the compiler actually emitted, counted from the PTX.

    Three questions a timing cannot answer, each visible as an instruction count:

    - `mma_sync`: did `tl.dot` reach the tensor cores? Triton supports sm_80+ officially;
      on Turing a dot can lower to `fma.rn.f32` loops instead, and a "tensor-core" kernel
      that is really an FMA loop through MMA-shaped layout shuffles is slower than the
      naive multiply-reduce it was meant to replace. Zero here is the whole story.
    - `local_ld/st`: register spills. Anything in the hundreds means every loop iteration
      round-trips local memory.
    - `global_ld_vec / global_ld_scalar`: whether the gathered K/V tiles load 16 bytes at a
      time (`ld.global.v4`) or one element at a time (`ld.global.b16`), i.e. whether the
      compiler could prove the page rows contiguous along head_dim.
    """
    if not ptx:
        return {}
    import re

    def count(pattern: str) -> int:
        return len(re.findall(pattern, ptx))

    return {
        "mma_sync": count(r"\bmma\.sync"),
        "fma_f32": count(r"\bfma\.rn\.f32"),
        "local_ld": count(r"\bld\.local"),
        "local_st": count(r"\bst\.local"),
        "global_ld_vec": count(r"\bld\.global[.\w]*\.v[24]\."),
        "global_ld_scalar": count(r"\bld\.global[.\w]*\.(?:b16|u16|f16|b32|u32)\s"),
        "bar_sync": count(r"\bbar\.sync"),
    }


def _time_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _dense_sdpa(query, key_pages, value_pages, tables, starts, chunk_lens):
    """Upper bound: gather the pages into dense tensors once, then let torch attend.

    The gather is outside the timed region because the point is what the attention itself
    should cost on this GPU, not what a Python gather costs. Any Triton kernel landing far
    below this is losing to its own implementation, not to the problem.
    """
    import torch.nn.functional as F

    batch, q_heads, chunk, head_dim = query.shape
    kv_heads = key_pages.shape[2]
    start, total = int(starts[0]), int(starts[0]) + chunk
    pages = tables[:, : -(-total // BLOCK_SIZE)]
    keys = key_pages[pages.reshape(-1)].reshape(batch, -1, kv_heads, head_dim)[:, :total]
    values = value_pages[pages.reshape(-1)].reshape(batch, -1, kv_heads, head_dim)[:, :total]
    keys = keys.transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    values = values.transpose(1, 2).repeat_interleave(q_heads // kv_heads, dim=1)
    positions = torch.arange(total, device=query.device)
    q_positions = start + torch.arange(chunk, device=query.device)
    mask = positions[None, :] <= q_positions[:, None]
    return F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)


def _build(batch, q_heads, kv_heads, start, chunk, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = start + chunk
    pages = -(-total // BLOCK_SIZE) + 2
    key_pages = torch.randn(pages * batch, BLOCK_SIZE, kv_heads, HEAD_DIM,
                            device="cuda", dtype=torch.float16, generator=generator)
    value_pages = torch.randn_like(key_pages)
    tables = torch.arange(pages * batch, device="cuda",
                          dtype=torch.int32).view(batch, pages)
    query = torch.randn(batch, q_heads, chunk, HEAD_DIM,
                        device="cuda", dtype=torch.float16, generator=generator)
    starts = torch.full((batch,), start, device="cuda", dtype=torch.int32)
    chunks = torch.full((batch,), chunk, device="cuda", dtype=torch.int32)
    return query, key_pages, value_pages, tables, starts, chunks


def ideal_kv_bytes(batch, q_heads, start, chunk, block_m):
    """KV a correctly tiled kernel must read: each Q tile loads its prefix once."""
    tiles = -(-chunk // block_m)
    entries = sum(start + min((i + 1) * block_m, chunk) for i in range(tiles))
    return entries * batch * q_heads * 2 * HEAD_DIM * BYTES


def _ptx_only(args) -> int:
    """Compile each kernel on one representative shape and report the instruction mix.

    A kernel that is 100x off its hardware's peak is not mistuned, it is not doing what
    its source says. This answers that before any timing: tensor cores or FMA, spills or
    not, vector or scalar loads - for the tiled kernel and, as a control, the per-token
    kernel it was meant to replace.
    """
    from engine.kernels.paged_prefill import _paged_prefill_kernel
    from engine.kernels.tiled_paged_prefill import _tiled_paged_prefill_kernel

    tensors = _build(args.batch, args.q_heads, args.kv_heads, args.prefix, args.chunks[0])
    paged_prefill(*tensors)
    torch.cuda.synchronize()
    old = kernel_diagnostics(_paged_prefill_kernel)
    tiled_paged_prefill(*tensors, block_m=args.block_m, block_n=args.block_n,
                        num_warps=args.num_warps)
    torch.cuda.synchronize()
    new = kernel_diagnostics(_tiled_paged_prefill_kernel)
    keys = ["n_regs", "n_spills", "shared_bytes", "mma_sync", "fma_f32", "local_ld",
            "local_st", "global_ld_vec", "global_ld_scalar", "bar_sync"]
    print(f"{'':18s}{'per_token':>12s}{'tiled':>12s}")
    for key in keys:
        print(f"{key:18s}{str(old.get(key, '?')):>12s}{str(new.get(key, '?')):>12s}")
    verdicts = []
    if new.get("mma_sync") == 0:
        verdicts.append("tl.dot did NOT reach the tensor cores: no mma.sync in the tiled PTX. "
                        "The kernel is an FMA loop through MMA-shaped layout shuffles.")
    if (new.get("local_ld") or 0) > 0:
        verdicts.append(f"tiled kernel spills: {new['local_ld']} local loads in the PTX.")
    if (new.get("global_ld_scalar") or 0) > (new.get("global_ld_vec") or 0):
        verdicts.append("tiled K/V gathers are scalar loads; contiguity along head_dim was not proven.")
    print("\n" + ("\n".join(verdicts) if verdicts else "no structural defect visible in the PTX; "
                                                   "the loss is in scheduling/occupancy, sweep tiles"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--prefix", type=int, default=896)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--sweep-tiles", action="store_true")
    parser.add_argument("--ptx-only", action="store_true",
                        help="compile both kernels once and print what the compiler emitted; no timing")
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--bandwidth-gbps", type=float, default=258.8,
                        help="measured achieved bandwidth, from roofline.py")
    parser.add_argument("--out", default="results/prefill_attention_ab.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("requires CUDA")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={args.batch} q_heads={args.q_heads} kv_heads={args.kv_heads} "
          f"prefix={args.prefix}\n")

    if args.ptx_only:
        return _ptx_only(args)

    payload: dict = {"config": vars(args), "points": [], "fits": {}}
    print("A dense SDPA reference is included as an upper bound: same maths on gathered\n"
          "contiguous tensors, using whatever attention kernel torch picks. If the tiled\n"
          "kernel is far off *that*, the problem is the kernel rather than the workload.\n")
    header = (f"{'chunk':>6} {'old ms':>9} {'new ms':>9} {'sdpa ms':>9} {'speedup':>8} "
              f"{'grid':>7} {'blk/SM':>7}")
    print(header)
    print("-" * len(header))
    old_ms, new_ms = [], []
    for chunk in args.chunks:
        tensors = _build(args.batch, args.q_heads, args.kv_heads, args.prefix, chunk)
        old = _time_ms(lambda t=tensors: paged_prefill(*t))
        sdpa = _time_ms(lambda t=tensors: _dense_sdpa(*t))
        new = _time_ms(lambda t=tensors: tiled_paged_prefill(
            *t, block_m=args.block_m, block_n=args.block_n, num_warps=args.num_warps))
        ideal = ideal_kv_bytes(args.batch, args.q_heads, args.prefix, chunk, args.block_m)
        gbps = ideal / (new / 1000) / 1e9
        old_ms.append(old)
        new_ms.append(new)
        from engine.kernels.tiled_paged_prefill import _tiled_paged_prefill_kernel
        diag = kernel_diagnostics(_tiled_paged_prefill_kernel)
        spills = diag.get("n_spills")
        note = ""
        if spills:
            note = f"  SPILLS {spills} bytes/thread"
        elif diag.get("n_regs"):
            note = f"  {diag['n_regs']} regs, {diag.get('shared_bytes', 0)} smem"
        if "mma_sync" in diag:
            note += (f", mma={diag['mma_sync']} fma={diag['fma_f32']} "
                     f"spill_ld={diag['local_ld']} vec_ld={diag['global_ld_vec']} "
                     f"scalar_ld={diag['global_ld_scalar']} bar={diag['bar_sync']}")
        import torch as _t
        sms = _t.cuda.get_device_properties(0).multi_processor_count
        blocks = args.batch * args.q_heads * -(-chunk // args.block_m)
        print(f"{chunk:>6} {old:>9.3f} {new:>9.3f} {sdpa:>9.3f} {old / new:>7.1f}x "
              f"{blocks:>7} {blocks / sms:>7.1f}{note}")
        payload["points"].append({
            "chunk": chunk, "old_ms": old, "new_ms": new, "sdpa_ms": sdpa,
            "blocks": blocks, "blocks_per_sm": blocks / sms, "speedup": old / new,
            "ideal_kv_bytes": ideal, "effective_gbps": gbps, "diagnostics": diag,
        })

    # Scale one layer's attention to a whole step, so b is on the Phase B axis.
    print(f"\nper-step cost model, attention only, x{LAYERS} layers, "
          f"batch {args.batch} rows sharing the step")
    for name, series in (("old", old_ms), ("new", new_ms)):
        fit = fit_line([float(c) for c in args.chunks],
                       [ms * LAYERS / args.batch for ms in series])
        payload["fits"][name] = fit
        print(f"  {name:4s} ms = {fit['a']:7.3f} + {fit['b']:.5f} * chunk_tokens   "
              f"(R^2={fit['r2']:.3f})  b = {fit['b'] / 0.018:5.1f}x compute floor")
    if payload["fits"]["new"]["b"] > 0:
        ratio = payload["fits"]["old"]["b"] / payload["fits"]["new"]["b"]
        print(f"\n  marginal cost per prefill token improved {ratio:.1f}x")
        target = 0.05
        verdict = ("MEETS" if payload["fits"]["new"]["b"] <= target else "MISSES")
        print(f"  {verdict} the Phase D2 target of b <= {target} ms/token")

    if args.sweep_tiles:
        print("\ntile shape sweep at chunk 128")
        tensors = _build(args.batch, args.q_heads, args.kv_heads, args.prefix, 128)
        best = None
        from engine.kernels.tiled_paged_prefill import _tiled_paged_prefill_kernel
        old_ref = _time_ms(lambda: paged_prefill(*tensors))
        print(f"  (per-token kernel at this shape: {old_ref:.3f} ms)")
        sweep = []
        # num_stages matters: without cp.async (pre-sm_80) the pipeliner keeps prefetched
        # tiles in registers, which can spill. It was never swept before.
        # Five axes, flattened: nesting them produced an indentation error, and a product
        # reads better anyway. num_stages was never swept before, and the K-load layout is
        # here because a paged gather cannot pre-transpose for free the way contiguous
        # FlashAttention does, so which side pays is an empirical question.
        grid_space = itertools.product((16, 32, 64), (32, 64), (2, 4, 8), (1, 2), (False, True))
        for block_m, block_n, warps, stages, transposed in grid_space:
            try:
                ms = _time_ms(lambda: tiled_paged_prefill(
                    *tensors, block_m=block_m, block_n=block_n, num_warps=warps,
                    num_stages=stages, transposed_k_load=transposed))
            except Exception:
                continue
            diag = kernel_diagnostics(_tiled_paged_prefill_kernel)
            sweep.append({"block_m": block_m, "block_n": block_n, "num_warps": warps,
                          "num_stages": stages, "transposed_k_load": transposed,
                          "ms": ms, **diag})
            marker = ""
            if best is None or ms < best[0]:
                best = (ms, block_m, block_n, warps, stages, transposed)
                marker = "  <-- best"
            spill = diag.get("n_spills") or 0
            print(f"  M={block_m:3d} N={block_n:3d} w={warps} s={stages} "
                  f"kT={'y' if transposed else 'n'} -> {ms:8.3f} ms  "
                  f"regs={str(diag.get('n_regs', '?')):>4} spills={spill:>5} "
                  f"mma={diag.get('mma_sync', '?')} spill_ld={diag.get('local_ld', '?')} "
                  f"vec_ld={diag.get('global_ld_vec', '?')} scalar_ld={diag.get('global_ld_scalar', '?')}{marker}")
        payload["sweep"] = sweep
        if best:
            print(f"\n  best: BLOCK_M={best[1]} BLOCK_N={best[2]} num_warps={best[3]} "
                  f"num_stages={best[4]} transposed_k={best[5]} at {best[0]:.3f} ms "
                  f"({old_ref / best[0]:.2f}x the per-token kernel)")
            if old_ref / best[0] < 1.0:
                print("  NO configuration beats the kernel it replaces. If that survives a "
                      "second device, delete the tiled kernel rather than tune it further.")
            payload["best_tile"] = {"block_m": best[1], "block_n": best[2],
                                    "num_warps": best[3], "num_stages": best[4],
                                    "transposed_k_load": best[5],
                                    "ms": best[0], "per_token_kernel_ms": old_ref}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

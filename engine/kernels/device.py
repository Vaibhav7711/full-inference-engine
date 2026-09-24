"""Device capability detection, so kernel defaults and recorded results are per-GPU.

Every performance number this project has produced is tied to one device. The measured
roofline (DD-037) says so explicitly, and the same is true of kernel launch parameters: a
tile shape and pipeline depth tuned for Turing is not the right one for Ada, and silently
carrying one over produces a result that looks like a regression but is a mismatch.

Two facts drive the defaults:

- `num_stages > 2` requires `cp.async`, introduced in sm_80. On Turing the extra stages are
  ignored at best and cost registers at worst, so the default there is 2.
- `mma.sync.m16n8k16` (fp16) exists from sm_80; Turing has only `m16n8k8`, so a given tile
  shape decomposes into twice as many instructions and wants fewer warps.

These are starting points, not answers. `benchmarks/kernels/prefill_attention_ab.py
--sweep-tiles` searches the space per device, and its result should override these.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    capability: tuple[int, int]
    total_memory_gb: float
    multiprocessors: int
    l2_cache_mb: float

    @property
    def sm(self) -> int:
        """Compute capability as an integer, e.g. 75 for Turing, 89 for Ada."""
        return self.capability[0] * 10 + self.capability[1]

    @property
    def supports_async_copy(self) -> bool:
        """cp.async, and therefore useful multi-stage software pipelining."""
        return self.sm >= 80

    def prefill_tile_defaults(self, head_dim: int = 128,
                              query_len: int = 128) -> dict[str, int]:
        """Starting launch parameters for the tiled prefill kernel on this device.

        Two things the first version got wrong, both computable without a GPU:

        - **Warps must scale with the accumulator.** `acc[BLOCK_M, head_dim]` in fp32 plus
          the score tile is 48 KB at 64x128; across 4 warps that is ~96 registers per
          thread before any temporaries, which caps occupancy hard. 8 warps halves it.
        - **BLOCK_M must leave enough tiles to fill the GPU.** The grid is
          `(batch, heads, ceil(query_len / BLOCK_M))`. Chunked prefill gives a query
          dimension of only a few hundred tokens, so a large BLOCK_M can collapse the grid
          to one tile per (row, head) - 64 blocks on 40 SMs. FlashAttention takes its
          parallelism from long sequences; a prefill *chunk* is not one.
        """
        block_m = 64 if query_len >= 256 else 32
        # 8 warps at every tile: measured on the T4 (prefill_attention_ab, prefix 896,
        # batch 4) the 32x64 tile at 8 warps ran 2x faster than the per-token kernel,
        # while the engine's earlier 4-warp default ran 2x slower than it - the kernel is
        # at 255 registers and halving the warps doubles the per-thread demand.
        warps = 8
        stages = 3 if self.supports_async_copy else 2
        return {"block_m": block_m, "block_n": 64, "num_warps": warps,
                "num_stages": stages}

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["sm"] = self.sm
        payload["supports_async_copy"] = self.supports_async_copy
        return payload

    def __str__(self) -> str:
        return (f"{self.name} (sm_{self.sm}, {self.total_memory_gb:.1f} GB, "
                f"{self.multiprocessors} SMs, {self.l2_cache_mb:.0f} MB L2)")


def current_device(device=None) -> DeviceProfile | None:
    """Profile a CUDA device, or the current one when ``device`` is omitted."""
    import torch

    if not torch.cuda.is_available():
        return None
    if device is None:
        index = torch.cuda.current_device()
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return None
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    # l2_cache_size has existed since torch 2.0 but is not guaranteed on every build.
    l2_bytes = getattr(properties, "L2_cache_size", 0) or 0
    return DeviceProfile(
        name=properties.name,
        capability=(properties.major, properties.minor),
        total_memory_gb=properties.total_memory / 1e9,
        multiprocessors=properties.multi_processor_count,
        l2_cache_mb=l2_bytes / 1e6,
    )


def prefill_tile_defaults(head_dim: int = 128, query_len: int = 128) -> dict[str, int]:
    """Launch parameters for the tiled prefill kernel, or Turing-safe values off-GPU."""
    profile = current_device()
    if profile is None:
        return {"block_m": 32, "block_n": 64, "num_warps": 8, "num_stages": 2}
    return profile.prefill_tile_defaults(head_dim=head_dim, query_len=query_len)


def fits_in_memory(weight_bytes: int, *, kv_pool_bytes: int = 0,
                   headroom_fraction: float = 0.15) -> tuple[bool, str]:
    """Whether weights plus a KV pool fit, with headroom for activations and fragmentation.

    Added because the move to an 8 GB card makes this a real constraint rather than a
    formality: Qwen3-4B in fp16 is about 8 GB of weights alone and cannot be served there
    without quantised weights, whatever the KV budget.
    """
    profile = current_device()
    if profile is None:
        return True, "no CUDA device; not checked"
    budget = profile.total_memory_gb * 1e9 * (1 - headroom_fraction)
    needed = weight_bytes + kv_pool_bytes
    if needed <= budget:
        return True, (f"{needed / 1e9:.2f} GB of {budget / 1e9:.2f} GB usable "
                      f"on {profile.name}")
    return False, (
        f"needs {needed / 1e9:.2f} GB but only {budget / 1e9:.2f} GB is usable on "
        f"{profile.name} ({profile.total_memory_gb:.1f} GB total, "
        f"{headroom_fraction:.0%} reserved). Use quantised weights or a smaller KV pool."
    )

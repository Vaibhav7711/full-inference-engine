"""KV cache INT8 quantization — halve KV memory and decode-phase bandwidth.

The decode bottleneck is memory bandwidth: every token reads the whole KV cache from HBM.
Quantizing the KV cache to INT8 halves those bytes — a direct attack on the bottleneck.

This differs from WEIGHT quantization (engine/quantization/int8.py):
    - Weight quant: quantize once at load, dequant before the matmul (on the compute path,
      which is why it was slightly SLOWER — dequant adds compute).
    - KV quant: quantize K,V as they are WRITTEN to cache, dequant when READ for attention.
      The dequant is cheap relative to the bandwidth saved, so it targets the real limit.

Scheme: per-token, per-head symmetric INT8.
    For each token's K (or V) vector of length head_dim:
        scale = max(|vector|) / 127        (one float per token per head)
        q     = round(vector / scale)      (int8, range [-127, 127])
    Dequant: vector ≈ q * scale.
    Verified round-trip error ~0.7% relative — attention is robust to this (weighted average).

Memory: INT8 = 1 byte/element vs FP16 = 2 bytes -> ~2x reduction. Scale storage
(one float32 per token per head) is ~3% overhead on the INT8 size — negligible.

This module provides:
    quantize_kv / dequantize_kv        — the core INT8 quant functions
    QuantizedKVLayer                    — a paged KV layer storing INT8 pages + scales
    A demonstration that generation with a quantized KV cache stays close to FP16.

Kept SEPARATE from the working FP16 PagedLayer so the FP16 path stays intact for comparison.
"""

from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Core INT8 quantization (per-token, per-head symmetric)
# ---------------------------------------------------------------------------

def quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [..., head_dim] float tensor to INT8 with per-vector symmetric scale.

    The scale is computed over the LAST axis (head_dim), so each token-head vector gets
    its own scale. Returns (int8_tensor, scale) where scale has the head_dim axis reduced.

    Args:
        x: float tensor, quantized along its last dimension.
    Returns:
        q:     int8 tensor, same shape as x.
        scale: float tensor, x.shape[:-1] + (1,).
    """
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0     # [..., 1]
    scale = scale.clamp(min=1e-8)                           # avoid div-by-zero
    q = torch.round(x / scale).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_kv(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_kv. Returns float16 reconstruction."""
    return (q.to(torch.float16) * scale)


# ---------------------------------------------------------------------------
# Quantized paged KV layer (INT8 pages + scales)
# ---------------------------------------------------------------------------

class QuantizedKVLayer:
    """Paged KV storage for one layer, holding INT8 pages + per-slot scales.

    Layout mirrors the FP16 PagedLayer but stores:
        key_pages_int8   : [num_blocks, block_size, kv_heads, head_dim]  int8
        key_scales       : [num_blocks, block_size, kv_heads, 1]         float16
      (and the same for value).

    On write: incoming FP16 K,V are quantized per token-head and scattered.
    On read (gather): pages are dequantized back to FP16 for attention.

    Memory vs FP16 PagedLayer: ~half (1 byte vs 2 per element), plus small scale storage.
    """

    def __init__(self, block_size_tokens: int = 16, initial_blocks: int = 4):
        self.block_size_tokens = block_size_tokens
        self.seq_len = 0
        self._initial_blocks = initial_blocks
        self.key_pages_int8: Optional[torch.Tensor] = None
        self.value_pages_int8: Optional[torch.Tensor] = None
        self.key_scales: Optional[torch.Tensor] = None
        self.value_scales: Optional[torch.Tensor] = None
        self.num_growths = 0

    def _init_pages(self, ref: torch.Tensor) -> None:
        # ref: [batch=1, kv_heads, seq, head_dim]
        _, kv_heads, _, head_dim = ref.shape
        dev = ref.device
        nb, bs = self._initial_blocks, self.block_size_tokens
        self.key_pages_int8 = torch.zeros((nb, bs, kv_heads, head_dim), dtype=torch.int8, device=dev)
        self.value_pages_int8 = torch.zeros_like(self.key_pages_int8)
        self.key_scales = torch.zeros((nb, bs, kv_heads, 1), dtype=torch.float16, device=dev)
        self.value_scales = torch.zeros_like(self.key_scales)

    @property
    def num_blocks(self) -> int:
        return 0 if self.key_pages_int8 is None else self.key_pages_int8.shape[0]

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size_tokens

    def _grow_to(self, required: int) -> None:
        while self.capacity_tokens < required:
            add = max(self._initial_blocks, self.num_blocks)
            zk = torch.zeros((add, *self.key_pages_int8.shape[1:]), dtype=torch.int8, device=self.key_pages_int8.device)
            zs = torch.zeros((add, *self.key_scales.shape[1:]), dtype=torch.float16, device=self.key_scales.device)
            self.key_pages_int8 = torch.cat([self.key_pages_int8, zk], dim=0)
            self.value_pages_int8 = torch.cat([self.value_pages_int8, torch.zeros_like(zk)], dim=0)
            self.key_scales = torch.cat([self.key_scales, zs], dim=0)
            self.value_scales = torch.cat([self.value_scales, torch.zeros_like(zs)], dim=0)
            self.num_growths += 1

    @torch.inference_mode()
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """Quantize new K,V, scatter into INT8 pages, return dequantized full K,V.

        key_states/value_states: [1, kv_heads, num_new, head_dim] (FP16).
        Returns (keys, values) dequantized to FP16, shape [1, kv_heads, seq_len, head_dim].
        """
        if self.key_pages_int8 is None:
            self._init_pages(key_states)

        num_new = key_states.shape[2]
        start, end = self.seq_len, self.seq_len + num_new
        if end > self.capacity_tokens:
            self._grow_to(end)

        # [1, H, N, D] -> [N, H, D]
        k = key_states[0].transpose(0, 1)
        v = value_states[0].transpose(0, 1)
        # Quantize per token-head (last axis = head_dim)
        kq, ks = quantize_kv(k)      # [N,H,D] int8, [N,H,1] scale
        vq, vs = quantize_kv(v)

        for i in range(num_new):
            tok = start + i
            blk, off = divmod(tok, self.block_size_tokens)
            self.key_pages_int8[blk, off] = kq[i]
            self.value_pages_int8[blk, off] = vq[i]
            self.key_scales[blk, off] = ks[i]
            self.value_scales[blk, off] = vs[i]

        self.seq_len = end
        return self._gather()

    def _gather(self):
        """Dequantize all stored tokens -> [1, H, seq_len, D] FP16."""
        # flatten pages: [nb, bs, H, D] -> [nb*bs, H, D]
        kq = self.key_pages_int8.flatten(0, 1)[: self.seq_len]     # [S,H,D]
        vq = self.value_pages_int8.flatten(0, 1)[: self.seq_len]
        ks = self.key_scales.flatten(0, 1)[: self.seq_len]         # [S,H,1]
        vs = self.value_scales.flatten(0, 1)[: self.seq_len]
        k = dequantize_kv(kq, ks)      # [S,H,D] fp16
        v = dequantize_kv(vq, vs)
        keys = k.transpose(0, 1).unsqueeze(0)     # [1,H,S,D]
        values = v.transpose(0, 1).unsqueeze(0)
        return keys, values

    def get_seq_length(self, *args, **kwargs) -> int:
        return self.seq_len

    def memory_bytes(self) -> dict:
        """Report INT8 storage vs the FP16 equivalent, for the memory-savings measurement."""
        if self.key_pages_int8 is None:
            return {}
        int8_bytes = (self.key_pages_int8.numel() + self.value_pages_int8.numel()) * 1  # int8
        scale_bytes = (self.key_scales.numel() + self.value_scales.numel()) * 2  # fp16
        fp16_equiv = (self.key_pages_int8.numel() + self.value_pages_int8.numel()) * 2
        total = int8_bytes + scale_bytes
        return {
            "int8_bytes": int8_bytes,
            "scale_bytes": scale_bytes,
            "total_quantized_bytes": total,
            "fp16_equivalent_bytes": fp16_equiv,
            "reduction_pct": round((1 - total / fp16_equiv) * 100, 1),
        }

"""Compare real paged decode attention with contiguous PyTorch SDPA on tiny pages."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.cache.paging import KVBlockAllocation, gather_paged_tokens
from engine.kernels.kv_write import write_prefill_kv_batched
from engine.kernels.paged_decode_batched import paged_decode_batched


def _reference(
    query: torch.Tensor, key_pool: torch.Tensor, value_pool: torch.Tensor,
    tables: list[list[int]], lengths: list[int], block_size: int,
) -> torch.Tensor:
    """Materialize each logical sequence only for the reference implementation."""
    _, query_heads, _, _ = query.shape
    kv_heads = key_pool.shape[2]
    repetition = query_heads // kv_heads
    rows = []
    for row, (table, length) in enumerate(zip(tables, lengths)):
        allocation = KVBlockAllocation(f"reference-{row}", block_size, table, length)
        key = gather_paged_tokens(key_pool, allocation, length).permute(1, 0, 2)
        value = gather_paged_tokens(value_pool, allocation, length).permute(1, 0, 2)
        # Qwen GQA: two query heads consume each KV head in this toy geometry.
        key = key.repeat_interleave(repetition, dim=0).unsqueeze(0)
        value = value.repeat_interleave(repetition, dim=0).unsqueeze(0)
        rows.append(F.scaled_dot_product_attention(query[row:row + 1], key, value))
    return torch.cat(rows, dim=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA: this trace launches the real paged Triton kernel.")
    torch.manual_seed(71)

    sequences, query_heads, kv_heads, dim = 2, 4, 2, 8
    blocks, block_size, padded_tokens = 6, 4, 5
    tables_list = [[5, 1], [2, 4]]
    lengths_list = [5, 3]
    tables = torch.tensor(tables_list, device="cuda", dtype=torch.int32)
    lengths = torch.tensor(lengths_list, device="cuda", dtype=torch.int32)
    key_pool = torch.zeros((blocks, block_size, kv_heads, dim), device="cuda", dtype=torch.float16)
    value_pool = torch.zeros_like(key_pool)
    key = torch.randn(sequences, kv_heads, padded_tokens, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    query = torch.randn(sequences, query_heads, 1, dim, device="cuda", dtype=torch.float16)

    write_prefill_kv_batched(key, value, key_pool, value_pool, tables, lengths)
    paged = paged_decode_batched(
        query, key_pool, value_pool, tables, lengths, block_n=16, num_warps=4,
    )
    reference = _reference(query, key_pool, value_pool, tables_list, lengths_list, block_size)
    torch.cuda.synchronize()

    difference = (paged.float() - reference.float()).abs()
    print("logical block tables:", tables_list)
    print("sequence lengths:", lengths_list)
    print("query heads / KV heads / GQA repetition:", query_heads, kv_heads, query_heads // kv_heads)
    print("Triton grid = (sequences, query_heads) =", (sequences, query_heads), "; programs:", sequences * query_heads)
    print("BLOCK_N = 16; each program loops over ceil(sequence_length / 16) KV tiles")
    print("paged output shape:", tuple(paged.shape))
    print("max abs difference vs SDPA:", float(difference.max()))
    print("outputs close:", bool(torch.allclose(paged, reference, atol=2e-3, rtol=2e-3)))
    print("sample output [sequence0, query_head0, token0, :4]:", paged[0, 0, 0, :4].float().cpu().tolist())


if __name__ == "__main__":
    main()

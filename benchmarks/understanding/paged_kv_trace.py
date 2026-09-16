"""Show actual Triton paged-KV addresses using a tiny, recognizable CUDA pool."""

from __future__ import annotations

import torch

from engine.cache.paging import KVBlockAllocation, gather_paged_tokens
from engine.kernels.kv_write import write_decode_kv, write_prefill_kv_batched


def _tagged_prefill(batch: int, heads: int, tokens: int, dim: int) -> torch.Tensor:
    result = torch.empty((batch, heads, tokens, dim), device="cuda", dtype=torch.float16)
    for row in range(batch):
        for head in range(heads):
            for token in range(tokens):
                result[row, head, token] = 1000 * row + 100 * head + 10 * token + torch.arange(
                    dim, device="cuda", dtype=torch.float16
                )
    return result


def _show_logical(name: str, pool: torch.Tensor, table: list[int], length: int) -> None:
    allocation = KVBlockAllocation(name, 4, table, sequence_length=length)
    # gather returns [token, head, dim], converting from physical pool order.
    values = gather_paged_tokens(pool, allocation, length)[:, :, 0].cpu().tolist()
    locations = [allocation.physical_location(position) for position in range(length)]
    print(f"{name}: logical positions -> physical (block, offset): {locations}")
    print(f"{name}: gathered K[logical_token, head, dim0]: {values}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA: this trace launches the real Triton KV writer.")

    # Toy dimensions make the address mapping readable. Production Qwen uses
    # [num_blocks, 16, 8, 128] per layer; the layout rule is identical.
    blocks, block_size, heads, dim, batch, padded_tokens = 6, 4, 2, 8, 2, 5
    tables = torch.tensor([[5, 1], [2, 4]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([5, 3], device="cuda", dtype=torch.int32)
    key_pool = torch.full((blocks, block_size, heads, dim), -1.0, device="cuda", dtype=torch.float16)
    value_pool = torch.full_like(key_pool, -2.0)

    key = _tagged_prefill(batch, heads, padded_tokens, dim)
    value = key + 5000
    write_prefill_kv_batched(key, value, key_pool, value_pool, tables, lengths)
    torch.cuda.synchronize()

    print("pool layout: [physical_block, offset, kv_head, head_dim]")
    print("block tables:", tables.cpu().tolist())
    _show_logical("request 0 after prefill", key_pool, [5, 1], 5)
    _show_logical("request 1 after prefill", key_pool, [2, 4], 3)
    print("untouched block 0 remains sentinel:", key_pool[0, :, :, 0].cpu().tolist())

    # Decode writes one KV vector at each current sequence length: positions 5 and 3.
    decode_key = torch.tensor(
        [[[[9000 + dim_id for dim_id in range(dim)]], [[9100 + dim_id for dim_id in range(dim)]]],
         [[[9200 + dim_id for dim_id in range(dim)]], [[9300 + dim_id for dim_id in range(dim)]]]],
        device="cuda", dtype=torch.float16,
    )
    write_decode_kv(decode_key, decode_key + 5000, key_pool, value_pool, tables, lengths)
    torch.cuda.synchronize()
    _show_logical("request 0 after decode write", key_pool, [5, 1], 6)
    _show_logical("request 1 after decode write", key_pool, [2, 4], 4)
    print("decode destinations: request0 pos5 -> block1/offset1; request1 pos3 -> block2/offset3")


if __name__ == "__main__":
    main()

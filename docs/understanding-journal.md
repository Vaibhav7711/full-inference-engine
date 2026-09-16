# Inference engine understanding journal

This journal reconstructs the runtime from code and measured results. It is separate
from `optimization-journal.md`: the goal here is not to justify changes, but to explain
the current engine's data flow, invariants, measurements, and limitations layer by layer.

## Layer 1 — Explicit prefill/decode reference

### Code boundary

- `engine/model/loader.py` selects a CUDA device and resolves `auto` to FP16 on the
  Tesla T4 (`sm75`).
- `engine/model/runner.py` owns the unbatched reference recurrence: full-prompt prefill,
  one-token cached decode, greedy argmax, EOS/length stopping, and per-stage timing.
- `engine/metrics/metrics.py` stores CUDA-event latency and process-visible totals.

### Measured model geometry

The Colab checkpoint was `Qwen/Qwen3-0.6B` under PyTorch 2.11.0+cu128 on a Tesla T4:

| Property | Observed value |
| --- | ---: |
| Parameters | 596,049,920 |
| Loaded parameter memory | 1,136.875 MiB |
| Hidden width | 1,024 |
| Layers | 28 |
| Query heads | 16 |
| KV heads | 8 |
| GQA ratio | 2:1 |
| Head dimension | 128 |
| Vocabulary | 151,936 |
| Runtime dtype | FP16 |

The current checkpoint is therefore 16-query-head/8-KV-head GQA, not the older 32/8
geometry still mentioned in parts of the README. Also, `query_heads * head_dim = 2,048`
is larger than the 1,024 hidden width; Qwen's explicit head dimension means the query
projection width must not be inferred as equal to hidden size.

### Token and cache recurrence

The prompt tokenized to nine IDs. Prefill produced token 7281 from the logits at the
last prompt position and created a 28-layer `DynamicCache`. Each layer's first K/V shape
was `[1, 8, 9, 128]`.

FP16 KV storage per logical token is:

```text
2 (K and V) * 28 layers * 8 KV heads * 128 dimensions * 2 bytes
= 114,688 bytes = 112 KiB/token
```

Observed prefill storage was exactly `9 * 114,688 = 1,032,192` bytes. Each call to
`decode_one` consumed the previously selected token, increased cache sequence length by
one, and added exactly 114,688 bytes. After producing eight output IDs, only seven had
been consumed by decode, so the cache length was `9 + 7 = 16`. The eighth token was
predicted but not yet written into KV. This is the central decode off-by-one invariant.

The explicit tokens were identical to Hugging Face greedy generation:

```text
[7281, 11, 10339, 279, 6672, 1948, 264, 6500]
```

### Measured reference performance

After two warmups and across five measured 32-token runs:

| Metric | Result |
| --- | ---: |
| Repeatable tokens | yes |
| Median TTFT | 43.210 ms |
| Median end-to-end | 1,295.456 ms |
| Median recorded decode forward | 39.094 ms/token |
| Aggregate throughput | 22.766 tok/s |
| Peak allocated | 1,152.424 MiB |
| Peak reserved | 1,168 MiB |

There are 31 recorded decode forwards for a 32-token length-limited request: prefill
predicts token zero, then only the first 31 selected output tokens need to be consumed to
predict tokens 1--31. The final predicted token is returned without another model call.

### What these timings mean

- `ttft_ms` is tokenization plus CUDA-event prefill time.
- Input tensor transfer to CUDA occurs before the prefill event and is excluded from
  `ttft_ms` even though it is included in `total_ms`.
- The first argmax token's `.item()` device-to-host synchronization is also excluded from
  `ttft_ms`; this is kernel TTFT, not fully user-visible TTFT.
- Each `decode_ms` value covers `decode_one`, including attention-mask construction and
  the model forward, but excludes the preceding selected token's `.item()` call.
- `total_ms` includes tokenization, input transfer, CUDA synchronization, scalar token
  reads, and model work, but stops before final `tokenizer.decode()` text conversion.
- Peak allocated memory includes the already-resident 1,136.875 MiB model weights. It is
  not generation-only working memory.
- Aggregate throughput is total tokens divided by total time over all five runs. It need
  not equal `32 / median(end_to_end)` because aggregation and medians are different
  statistics and individual runs vary.

### Layer 1 established invariants

1. Prefill cache length equals prompt length.
2. Prefill logits predict the first output token.
3. Each decode call consumes one selected token, grows KV by one position, and predicts
   the following token.
4. FP16 DynamicCache storage grows by exactly 112 KiB per sequence token for this model.
5. The explicit recurrence is token-identical to Hugging Face greedy generation.
6. Reference benchmark metrics are internal runtime timings, not network-visible service
   latency.

### DynamicCache storage and one-token operator anatomy

The installed Transformers `DynamicLayer.update` implementation was inspected directly:

```python
self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)
```

The `DynamicCache` and its layer object remain mutable containers, but their tensor
attributes are replaced. For layer zero, a decode changed key storage from shape
`[1, 8, 9, 128]` at CUDA pointer `140653626959872` to `[1, 8, 10, 128]` at pointer
`140653627957248`. Retaining the old tensor object confirmed both Python identity and
CUDA storage identity were false after the update. The new tensors remained contiguous;
their strides changed from `(9216, 1152, 128, 1)` to `(10240, 1280, 128, 1)`.

Net allocated memory grew by 115,200 bytes while reserved memory remained unchanged.
The logical KV increase is 114,688 bytes; the small remainder is allocator/tensor
overhead. PyTorch satisfied new storage from its existing CUDA caching-allocator pool,
so a changed `data_ptr` did not require reserved memory to grow.

Profiling one reference decode produced exactly 197 `aten::mm` calls:

```text
28 layers * (4 attention projections + 3 MLP projections) + 1 lm_head = 197
```

It also produced 113 separate RMSNorm reduction sequences:

```text
28 layers * (input norm + post-attention norm + Q norm + K norm) + final norm = 113
```

Python-level tracing attributed all 114 `torch.cat` calls exactly:

| Source | Calls | Purpose |
| --- | ---: | --- |
| `modeling_qwen3.py:rotate_half` | 56 | Q and K RoPE rotation in each of 28 layers |
| `cache_utils.py:DynamicLayer.update` key | 28 | allocate/copy enlarged key cache |
| `cache_utils.py:DynamicLayer.update` value | 28 | allocate/copy enlarged value cache |
| `engine/model/runner.py:decode_one` | 1 | extend the attention mask |
| `modeling_qwen3.py` rotary `forward` | 1 | concatenate duplicated rotary frequencies |

The count is therefore not merely “cache concatenation.” Half of the calls come from
the unfused Hugging Face RoPE implementation, 56 come from cache growth, and two are
per-forward mask/rotary construction. This distinction explains why fused RoPE removed
56 cats while direct paged KV writes later removed the other 56 cache cats.

## Layer 3 — paged KV physical storage and kernels

`ContinuousBatchingEngine` owns one persistent key pool and one value pool per model
layer. In FP16 their common layout is `[physical_block, token_offset, kv_head, head_dim]`:

```python
torch.zeros((num_blocks, block_size, num_kv_heads, head_dim), device=device)
```

For the Qwen3-0.6B engine this is `[num_blocks, 16, 8, 128]` per K or V layer pool.
The CPU `KVBlockManager` owns only the mapping: each request has a list where
`physical_block_ids[logical_position // 16]` selects a row in those shared tensors.

The CUDA toy trace used tables `[[5, 1], [2, 4]]` with a four-token block. Triton's
prefill writer correctly placed request 0 positions 0--4 at `(5,0)` through `(5,3)`,
then `(1,0)`, and request 1 positions 0--2 at `(2,0)` through `(2,2)`. A decode write
at pre-write lengths `[5, 3]` then landed at request 0 `(1,1)` and request 1 `(2,3)`.
Gathering through the same mappings reconstructed the logical order exactly.

The decode writer launches a grid `(batch_size, kv_heads)`: one Triton program copies
one `[head_dim]` K vector and one V vector for one request/KV-head pair. Its destination
is calculated from sequence length, block table, and actual tensor strides; no cache
tensor is grown or copied. The decode attention kernel launches `(active_sequences,
query_heads)`, maps each Q head to its GQA KV head, walks logical positions in tiles,
translates every position through that sequence's block table, and performs online
softmax without gathering a contiguous K/V tensor.

The trace used intentionally large marker values. FP16 represents values near 9,000 in
steps of eight, so `9100` was read back as `9104` and `9300` as `9296`. Those are normal
FP16 rounding effects, not incorrect block-table addressing.

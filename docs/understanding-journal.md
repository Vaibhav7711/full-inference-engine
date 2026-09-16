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

The paged decode reader was then compared directly with PyTorch SDPA. Two sequences
with tables `[[5, 1], [2, 4]]`, lengths `[5, 3]`, four Q heads, two KV heads, and
head dimension eight produced a Triton grid `(2, 4)` (eight programs). The paged output
shape was `[2, 4, 1, 8]`; its maximum absolute difference from an SDPA reference which
first materialized logical contiguous K/V was `0.0009765625`, and `allclose` passed at
`atol=rtol=0.002`. The difference is one normal FP16-scale rounding increment.

For each `(sequence, query_head)` program, `kv_head = query_head // (q_heads // kv_heads)`
implements grouped-query attention. The program loops through logical positions in
`BLOCK_N` tiles, translates each position independently through the request's table,
loads K/V straight from the corresponding physical pool rows, and keeps running
`max`, normalization sum, and weighted-value accumulator in FP32. It does not form a
contiguous K/V tensor or a full attention-score vector.

### Real Qwen engine integration trace

The same path was exercised with the actual Qwen3-0.6B engine, two real prompts, and
128 physical blocks. Its real geometry was 28 layers, 16 Q heads, 8 KV heads, and head
dimension 128; each per-layer K or V pool had shape `[128, 16, 8, 128]`. After prefill,
request `real-0` had length 12 and table `[127]`, and `real-1` had length 9 and table
`[126]`. Their next decode writes therefore targeted `(127,12)` and `(126,9)`.

One subsequent ordinary `engine.step()` produced exactly 28 custom attention-hook calls:
one invocation by each real Qwen attention module. The trace compared only the imminent
K slots before/after the forward and found 56 changed slots out of 56 expected
(`28 layers * 2 requests`), while all 28 K-pool CUDA data pointers were unchanged.
The writer stores V in the same invocation; K was selected for compact observation.

The first prefill outputs were `[8886]` and `[1096]`. The decode forward consumed those
as input IDs at positions 12 and 9, wrote their K/V at those positions, predicted tokens
`2504` and `374`, and only then advanced allocator sequence lengths to 13 and 10. Thus
the real custom engine retains the same one-token generation recurrence established in
Layer 1 while replacing DynamicCache growth with persistent-pool writes.

## Layer 4 — decode metadata staging

With a real two-request Qwen engine and 128 blocks, persistent host buffers were pinned
and persistent GPU buffers had distinct CUDA pointers. Their shapes were input IDs
`[2,1]` int64, position IDs `[2,1]` int64, sequence lengths `[2]` int32, and complete
block tables `[2,128]` int32. The real pre-decode CPU metadata was token IDs `[1096,
1096]`, positions/lengths `[13,12]`, and valid block-table prefixes `[[127],[126]]`.
The GPU views contained the same values and their data pointers exactly matched the
long-lived device buffers. A second staging call reused all four CUDA addresses.

The measured per-iteration H2D payload was 1,064 bytes:

```text
2 * 1 * 8  input IDs       =   16 bytes
2 * 1 * 8  position IDs    =   16 bytes
2 * 4      sequence lengths =   8 bytes
2 * 128 * 4 complete tables = 1024 bytes
                                   ------
                                   1064 bytes
```

The engine deliberately copies full block-table rows, not only currently used columns,
to preserve contiguous fixed-size buffer views suitable for CUDA-graph capture and to
avoid per-step tensor allocations. Entries beyond a request's allocated logical blocks
may be stale, but paged kernels read table positions only for `logical_block` values
reached by masked positions below that request's `seq_len`; they are semantically dead.
Pinned host memory permits the `copy_(..., non_blocking=True)` calls to enqueue DMA
transfers without a host-side wait. On the same CUDA stream, the subsequent model
kernels still observe the copies in order.

## Layer 5 — real padded CUDA-graph replay

The real Qwen engine was configured with `max_active=4` and graph buckets `(2,4)`. It
permanently reserved three pages `[127,126,125]` under the special allocator owner
`__cuda_graph_dummy_rows__`: the maximum bucket can need `4 - 1 = 3` padding rows when
only one customer request is live. With three live requests, their actual first pages
were `[124,123,122]`, and one width-four dummy row used page 127.

The staged fixed-width GPU metadata was input IDs `[[1096],[1096],[576],[151643]]`,
positions `[8,8,12,0]`, sequence lengths `[8,8,12,0]`, and first table entries
`[124,123,122,127]`. `151643` was the tokenizer's pad token. The dummy row has length
zero and position zero, but it has a valid private page so the captured model can execute
the same memory accesses and tensor shapes as four real rows. Its logits are discarded
through `logits[:len(active)]`; the scheduler never commits its length.

First use generated graph key `(4,64,4)`: width four, the selected paged-attention tile
regime of 64, and four Triton warps. The captured logits tensor had fixed shape
`[4,1,151936]`. Capture executes one eager warmup model forward and one captured forward,
therefore 56 Python attention-hook calls (`28 * 2`). Subsequent `graph.replay()` launches
the recorded GPU graph directly, so it makes no new Python hook calls while still running
the 28-layer model computation.

After replay, only live requests advanced from lengths `[8,8,12]` to `[9,9,13]` and
received their next output tokens. The dummy allocation remained at sequence length zero.
The writer may overwrite its physical page at offset zero during every captured/replayed
forward, but that page is permanently isolated from customers and the allocator regards
it as reusable scratch for graph padding.

## Layer 6 — real prefix cache and partial-tail copy-on-write

An actual 103-token Qwen prompt occupied seven 16-token pages: six complete blocks and
a final partial block with seven valid offsets. Its source allocation was
`[127,126,125,124,123,122,121]`. Publishing created six radix nodes for the complete
blocks plus one exact entry holding all seven pages and the first prediction (token
3555). Before source release, complete-page refcounts were three (source + radix node +
exact entry) and the partial tail refcount was two (source + exact entry). After source
release, the cache alone retained all seven physical pages: six radix nodes, one exact
entry, and seven unique cached blocks.

An identical request then matched the exact entry with `cached_prefix_tokens=103`,
attached the same block table, received cached first output token 3555, and made zero
attention-hook calls: it bypassed prefill entirely. Its shared partial-tail refcount was
two (exact entry + request). On the first decode, `_ensure_writable_tail` replaced tail
121 with private page 120, copied the valid existing K prefix correctly in all 28 layers,
then the normal decode wrote the consumed token at offset 7. The old tail refcount became
one (exact cache) and the private tail was one (request); the request length became 104
and predicted output token 374.

Copy-on-write copies an entire physical tail page, not only seven valid offsets: in this
FP16 Qwen geometry that is `16 * 8 * 128 * 2 = 32 KiB` per K or V page per layer, or
`64 KiB * 28 = 1.75 MiB` for K and V across the model. It is paid only when an exact hit
ends inside a shared page and subsequently begins decoding; the dramatic saved work is
the avoided 103-token, 28-layer prompt prefill.

## Layer 7 — real chunked prefill and decode-first scheduling

The real Qwen engine was run with three admitted requests, two 125-token prompts and one
7-token prompt. The scheduler was configured with `prefill_chunk_size=16`,
`max_prefill_tokens_per_iteration=16`, and `max_active=3`. This forces prompt work to
be spread across scheduling iterations instead of letting one long prompt monopolize the
engine until its whole prefill is complete.

`ContinuousBatchingEngine.step()` is intentionally decode-first. It first gathers active
`DECODING` requests and calls one batched `decode_step`; then it admits waiting requests;
only after that does it plan and run prefill chunks. The prefill planner is the
scheduler's round-robin `_prefill_order` deque. Each visit chooses
`min(remaining_prefill_tokens, chunk_size, token_budget)` and subtracts from the
iteration's remaining budget.

The trace showed the exact effect:

| Iteration | Main work performed | Resulting state |
| --- | --- | --- |
| 0 | Admit all three, prefill `long-a` by 16 | `long-a=16/125`, others `0/...` |
| 1 | Prefill `long-b` by 16 | `long-b=16/125` |
| 2 | Prefill `short` by 7, then spend remaining 9 on `long-a` | `short` enters `DECODING`, `long-a=25/125` |
| 3 | Decode `short`, then prefill `long-b` by 16 | `short` length 8, `long-b=32/125` |
| 4 | Decode `short`, then prefill `long-a` by 16 | `short` length 9, `long-a=41/125` |

This is the important service invariant: a short prompt can finish prefill and begin
streaming tokens while long prompts are still only partially prefilling. The engine does
not need to finish all admitted prefill work before decode begins. Once a request reaches
`DECODING`, it gets a decode opportunity at the front of each `step()` before more prompt
chunks are scheduled.

The allocator numbers also matched the paging model. At admission, each request reserved
one 16-token page, so capacity was 48 tokens while only 16 real tokens had been written.
As long prompts crossed block boundaries, new physical pages were appended: `long-a`
moved from `[127]` to `[127,124]` at 25 tokens and then `[127,124,122]` at 41 tokens;
`long-b` moved from `[126]` to `[126,123]` at 32 tokens. Internal fragmentation shrank
from 32 to 14 tokens as the already-reserved pages filled.

Chunked prefill uses the custom prefill attention path only when resumability is needed.
The code keeps a faster complete-prompt SDPA path for batches where every request is
fresh and the planned chunk covers the whole prompt. For partial chunks, it builds padded
IDs, absolute position IDs, block tables, start offsets, and chunk lengths, installs a
prefill context, runs the real model with `use_cache=False`, and lets the custom
attention modules write K/V directly into the persistent paged pools. When a row's prompt
is completed, the logits at that row's final real chunk position produce the first output
token, the request transitions to `DECODING`, and the token is appended.

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

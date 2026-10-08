# User guide

How to install, serve, configure and tune this engine. For *why* it is built this way see
[`engineering-report.md`](engineering-report.md); for the code structure see
[`architecture.md`](architecture.md).

The engine is validated on **Tesla T4 (`sm_75`)** with **Qwen3-0.6B / 1.7B / 4B in FP16**.
It runs on other CUDA GPUs and other Llama-style checkpoints, but those combinations are
not performance-validated — see [Running on other hardware](#running-on-other-hardware).

---

## 1. Install

```bash
git clone https://github.com/Vaibhav7711/full-inference-engine.git
cd full-inference-engine
pip install -e ".[dev,server]"
```

Requirements: Python ≥ 3.10, a CUDA GPU, `torch` ≥ 2.4 matching your CUDA runtime, and
Triton (bundled with Linux `torch` wheels). **Do not reinstall torch on Colab or Kaggle** —
the preinstalled build matches the runtime's CUDA. Use `scripts/setup_colab.sh`, which
installs everything except torch.

The engine requires CUDA and refuses CPU deliberately: every timing and cache behaviour it
records describes the GPU path.

Verify the install before loading a model:

```bash
python -m pytest -q                 # 314 CPU tests; needs neither GPU nor Triton
python scripts/colab_preflight.py   # torch / triton / transformers versions, GPU identity
```

---

## 2. First run

### As a library

```python
from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.model import load_model

loaded = load_model("Qwen/Qwen3-0.6B", dtype="float16")
engine = ContinuousBatchingEngine(
    loaded.model, loaded.tokenizer, loaded.device,
    max_active=8,                       # concurrent sequences
    num_blocks=1024, block_size=16,     # KV pool: 16,384 tokens
    cuda_graph_batch_sizes=(1, 2, 4, 8),
)
engine.warmup()                          # pay Triton JIT + graph capture up front

print(engine.generate(["Explain KV caching in one sentence."], max_new_tokens=64))
```

A fuller worked example — backend resolution, capacity arithmetic, per-request sampling,
and staggered arrivals through the scheduler — is in
[`examples/library_usage.py`](../examples/library_usage.py):

```bash
python examples/library_usage.py --model Qwen/Qwen3-1.7B --out results/library_usage.json
```

### As a server

```bash
uvicorn engine.server.api:create_app --factory --host 127.0.0.1 --port 8000
until curl -sf localhost:8000/ready >/dev/null; do sleep 2; done   # warmup runs first
```

Readiness deliberately waits for warmup and graph capture, so the first real request does
not pay them.

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "Explain paged attention in three sentences."}],
  "temperature": 0.7, "top_p": 0.95, "max_tokens": 120, "stream": true}'
```

Any OpenAI client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
for chunk in client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "Write a haiku about caches."}],
        stream=True, temperature=0.8, max_tokens=64):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 3. Sizing the KV pool

This is the one configuration decision that matters most, because **the KV pool, not the
weights, bounds concurrency**.

KV bytes per token = `2 × layers × kv_heads × head_dim × dtype_bytes`. For Qwen3-0.6B and
1.7B (both 28 layers, 8 KV heads, head_dim 128, FP16) that is **112 KiB per token**.

```
pool_tokens   = num_blocks × block_size
pool_bytes    = pool_tokens × kv_bytes_per_token
tokens_per_row ≈ pool_tokens / max_active        (before preemption starts)
```

Worked example for a 16 GB T4 serving Qwen3-1.7B:

| | |
|---|---|
| weights (FP16) | 3.44 GB |
| activations, CUDA graphs, CUDA context | ~1.5–2.5 GB (see §4) |
| available for KV | ~10 GB → ~89,000 tokens → `num_blocks=5568, block_size=16` |
| at `max_active=8` | ~11,000 tokens per row |

Pick `max_active` and `num_blocks` together so that
`pool_tokens / max_active` comfortably exceeds your longest expected
`prompt + max_new_tokens`. If it does not, the scheduler will admit requests and then
preempt them, which is correct behaviour but costs ~37 ms per rebuild plus queue time.

The engine refuses a pool it cannot allocate, with the arithmetic and the largest
affordable `num_blocks`, rather than raising a CUDA OOM:

```
ValueError: KV pool of 4096 x 16 tokens needs 7.52 GB but only 6.90 GB is free on this
device. Pass num_blocks <= 3758 (leaving nothing for activations or CUDA graphs), or a
smaller block_size, or quantise the KV cache with kv_cache_dtype='int8'.
```

`block_size` should stay at **16** unless you are using the FlashAttention paged backend,
which requires 256. Larger pages cost internal fragmentation — a 300-token request occupies
two 256-token pages (41% waste) against 6% at page 16 — and coarsen both preemption
granularity and prefix-cache reuse.

---

## 4. Memory budget

Four things occupy the GPU:

| | Qwen3-0.6B example | notes |
|---|---:|---|
| weights | 1.19 GB | FP16; `lm_head` is tied to the embedding |
| KV pool | `pool_tokens × 112 KiB` | preallocated at construction |
| CUDA graph pool | 0.3–2 GB | scales with the number and shape of captured graphs |
| CUDA context, cuBLAS, Triton modules | ~0.3–0.5 GB | fixed |

The graph pool is the surprising one. Graphs are captured per shape — decode row bucket,
chunk row bucket, gathered-context bucket, kernel regime — and each captured forward's
intermediate tensors live in a shared pool for the engine's lifetime. If memory is tight:

```python
cuda_graph_batch_sizes=(1, 4, 8)   # fewer decode buckets
prefill_cuda_graphs=False          # keeps decode graphs, drops prefill/fused graphs
fused_step=False                   # two forwards per prefill step instead of one
```

Measured on a T4 with Qwen3-0.6B, `num_blocks=1024`: full graphs reserved 5.09 GiB;
`prefill_cuda_graphs=False` reserved 3.06 GiB; no graphs at all 2.91 GiB. Disabling prefill
graphs therefore saves ~2 GB and costs roughly 57% on prefill-carrying step latency — a
real trade, not a free win.

---

## 5. Configuration reference

### `ContinuousBatchingEngine`

| parameter | default | when to change it |
|---|---|---|
| `max_active` | 16 | Concurrent sequences. Raise for throughput, lower if the pool cannot give each row enough tokens. |
| `num_blocks` | 4096 | KV pages. **Almost always set this explicitly** — the default is 7.5 GB of KV at `block_size=16`. |
| `block_size` | 16 | Tokens per page. 256 only for the FlashAttention paged backend. |
| `prefill_chunk_size` | 128 | Tokens of prompt admitted per request per step. Smaller = steadier inter-token latency, slower first token. |
| `max_prefill_tokens_per_iteration` | 128 | Total prefill tokens per step across requests. Equal to the chunk size means one chunk per step. |
| `cuda_graph_batch_sizes` | `None` | Decode row buckets to capture. `(1,2,4,8,16)` is a good default; `None` disables graphs and costs ~70% of decode step time. |
| `prefill_cuda_graphs` | `True` | Capture the prefill/fused forward. Disable to save ~2 GB at a large latency cost. |
| `fused_step` | `True` | One packed forward per prefill-carrying step. Disable only for A/B comparison. |
| `decode_attention` | `"per_head"` | Decode backend. `"auto"` consults the per-device policy. |
| `prefill_attention` | `None` (→ policy) | `"sdpa"`, `"per_token"`, `"tiled"`, `"flash"`. |
| `verify_attention` | `None` (→ policy) | Attention for speculative verification: an in-place paged kernel (`"tiled"` from sm_80, `"per_token"` below) instead of the prefill path's prefix gather; `"prefill"` keeps the old behaviour as the A/B baseline. Unmeasured; see `ab.py --setting verify_kernel`. |
| `kv_cache_dtype` | `"fp16"` | `"int8"` (any device) or `"fp8"` (E4M3, sm_89+) halves KV bytes per token. INT8 measured neutral on T4; FP8 is unmeasured and refused with the device named below sm_89. |
| `prefix_cache_blocks` | 256 | Pages retained for prefix reuse. Only useful when requests share prefixes. |
| `triton_rmsnorm` / `triton_rope` / `triton_swiglu` | `True` | Fused kernels. Disable only to isolate their contribution. |
| `fuse_mlp_gate_up` | `False` | One GEMM for gate+up instead of two. |
| `sampling_seed` | `None` | Seeds the shared RNG for requests that sample without their own seed. |
| `max_waiting_requests` | `None` | Queue bound; beyond it, submission is rejected rather than queued. |

### `SamplingParams` (per request)

```python
from engine.runtime import SamplingParams

SamplingParams(
    temperature=0.8,          # 0.0 = greedy, and greedy overrides everything else
    top_p=0.95, top_k=40, min_p=0.0,
    repetition_penalty=1.1, presence_penalty=0.0, frequency_penalty=0.0,
    penalize_prompt=False,    # OpenAI counts only the completion
    seed=1234,                # reproducible regardless of batch composition
    stop_token_ids=frozenset({151645}),
    ignore_eos=False,
    logprobs=5,               # chosen token plus top-k alternatives
)
```

Defaults are exactly greedy decoding, which is the path every recorded benchmark measured.
A batch where every request is greedy and asks for no penalties takes a single `argmax`.

### `create_app` (server)

```python
from engine.server.api import create_app

app = create_app(
    model_name="Qwen/Qwen3-1.7B", dtype="float16",
    max_active=8, num_blocks=4096, block_size=16,
    graph_buckets=(1, 2, 4, 8),
    max_prompt_tokens=4096, request_timeout_s=120.0, drain_timeout_s=30.0,
)
```

Serve a custom configuration with a factory:

```python
# myserver.py
from engine.server.api import create_app
def app():
    return create_app(model_name="Qwen/Qwen3-1.7B", num_blocks=4096, max_active=8)
```
```bash
uvicorn myserver:app --factory --port 8000
```

---

## 6. HTTP surface

| endpoint | purpose |
|---|---|
| `POST /v1/completions` | OpenAI completions, streaming supported |
| `POST /v1/chat/completions` | OpenAI chat, uses the model's own chat template |
| `GET /v1/models` | the served model id |
| `POST /generate` | native: returns token ids and per-request metrics |
| `POST /generate/stream` | native SSE stream |
| `GET /health` | liveness — the worker is running |
| `GET /ready` | readiness — loaded, warmed, accepting requests |
| `GET /metrics` | Prometheus text format |
| `GET /` | a small demo UI |

Parameters the engine does not honour are **refused with 400**, not ignored: `n > 1`,
`echo`, `best_of`, `logit_bias`. Status codes: 400 bad request, 413 prompt too long, 429
queue full, 499 client disconnected, 503 not ready or draining, 504 deadline exceeded.

`stop` strings are applied at the HTTP layer and **cancel the request**, so its KV pages
return to the pool instead of generating to the token bound.

### Metrics worth alerting on

| metric | meaning |
|---|---|
| `inference_time_to_first_token_seconds` | TTFT histogram |
| `inference_inter_token_latency_seconds` | per-request mean ITL |
| `inference_e2e_request_latency_seconds` | arrival to final token |
| `inference_requests_waiting` / `_running` | queue depth and batch width |
| `inference_kv_cache_utilization` | pool pressure; sustained >0.9 means preemption |
| `inference_graph_captures_in_service` | **should be 0**; non-zero means a shape warmup missed and a request paid 100–200 ms for it |

---

## 7. Tuning

Start from measurement, not intuition. The A/B harness compares two configurations on the
same closed-loop workload with interleaved runs:

```bash
python -m benchmarks.reliability.ab --setting fused_step --prompt-profile chat \
    --cuda-graphs --repeats 5 --duration 30
```

Read the output with two rules: a change smaller than its reported `spread` is
**unresolved** and is not a result, and a tail percentile (`p99`, `p999`) only counts when
`lazy_graph_captures` is 0 in both arms.

Which knob for which symptom:

| symptom | first thing to try |
|---|---|
| first token too slow | raise `prefill_chunk_size` and `max_prefill_tokens_per_iteration` together |
| inter-token latency spiky when new requests arrive | lower `prefill_chunk_size` (32 or 64) |
| tail latency spikes early in a run | check `inference_graph_captures_in_service`; call `warmup()` and widen `cuda_graph_batch_sizes` |
| throughput below expectation at high concurrency | check `kv_cache_utilization` and `preemptions_total` — you are probably pool-bound, not compute-bound |
| out of memory at construction | reduce `num_blocks`; the error message gives the affordable value |
| out of memory after warmup | reduce `cuda_graph_batch_sizes`, or set `prefill_cuda_graphs=False` |

The chunk-size trade is the one to understand: a bigger chunk finishes prompts in fewer
steps (better TTFT) but each prefill-carrying step interrupts decoding for longer (worse
ITL). On the T4, 128 was measured as the right point because a prefill step's cost is
dominated by a fixed per-invocation overhead rather than by the tokens in it.

---

## 8. Running on other hardware

```bash
python scripts/check_hooks.py --backends-only
```

Seconds, no model warmup. It prints the checkpoint's geometry, what got fused, **every
backend with `ok`/`no` and the reason**, and the defaults this device will use:

```
  ok decode   per_head   p50
  no decode   flash      p80  <- flash_attn wheels require sm_80+; this device is sm_75x
  no prefill  tiled      p60  <- tl.dot does not reach the tensor cores on sm_75
defaults: {'decode_attention': 'per_head', 'prefill_attention': 'sdpa', 'dtype': 'float16'}
```

If the reasons say **`unmeasured`**, the defaults are capability-led guesses, not
measurements. To settle them on your GPU:

```bash
python scripts/verify_hooks.py --dtype float16 --out results/<gpu>/verify_hooks.json
python -m benchmarks.reliability.ab --setting prefill_kernel --cuda-graphs --repeats 5
python -m benchmarks.reliability.ab --setting decode_kernel  --cuda-graphs --repeats 5
```

then add the winners to `MEASURED` in `engine/backends/policy.py` with the artifact that
chose them.

Two traps when comparing a new GPU against the recorded T4 numbers:

1. `dtype="auto"` selects **bf16** on `sm_80+`. Pass `dtype="float16"` for comparability —
   bf16 changes numerics and the token-identity gate behaves differently.
2. `"auto"` backends on an unmeasured architecture pick the highest-priority *available*
   backend, which may be one nobody has measured. Pin `decode_attention` and
   `prefill_attention` explicitly for any baseline comparison.

### Which checkpoints work

The engine is not Qwen-specific; it is **Llama-family-shaped**. A checkpoint is servable
when all of the following hold:

- `head_dim ≤ 128` and divisible by 8 — the paged kernels load a whole head per program
- query heads divisible by KV heads (any GQA ratio, including 1:1)
- no sliding-window attention — the paged kernels attend to the full prefix
- dense feed-forward (no mixture-of-experts)
- standard K/V heads (no multi-head latent attention)
- RMSNorm-style norms, and ideally separate `gate_proj` / `up_proj` / `down_proj`

Check any checkpoint in seconds, **without a GPU**:

```bash
python -c "
from transformers import AutoConfig
from engine.model.adapters import unsupported_reason, geometry_of
c = AutoConfig.from_pretrained('meta-llama/Llama-3.2-3B')
g = geometry_of(c)
print(f'{g.num_layers} layers, {g.num_q_heads}/{g.num_kv_heads} heads, head_dim {g.head_dim}')
print(f'{g.kv_bytes_per_token()/1024:.0f} KiB KV per token')
print(unsupported_reason(c) or 'servable')"
```

| family | status |
|---|---|
| Qwen3 dense, Qwen2.5 (head_dim 128) | **validated** (0.6B / 1.7B / 4B on T4) |
| Llama 3.x, TinyLlama, SmolLM | servable, **not performance-validated** |
| Mistral v0.2+ (no sliding window) | servable, not validated |
| Phi-3 | servable, but its fused `gate_up_proj` means the SwiGLU fusion is skipped — the engine records `swiglu_unavailable_reason` and serves without it |
| Mistral v0.1 | refused — sliding-window attention |
| Gemma 2 / 3 | refused — `head_dim 256` |
| MoE variants (e.g. Qwen3-30B-A3B) | refused — mixture-of-experts |
| DeepSeek-V2 / V3 | refused — multi-head latent attention |

Refusals happen **at load, with the reason**, not at the first request.

Before trusting an unvalidated family, run the hook matrix — it checks fusion discovery,
RoPE patching, and token identity against stock Transformers for every available backend:

```bash
python scripts/verify_hooks.py --model <checkpoint> --dtype float16
```

---

## 9. Troubleshooting

**`RuntimeError: CUDA is required`** — intentional; the engine has no CPU path.

**`pytest -q` errors on import** — run it from the repository root; `engine` must be
importable (`pip install -e .`).

**`ValueError: KV pool of N x M tokens needs X GB but only Y GB is free`** — use the
`num_blocks` value in the message, minus headroom for graphs and activations.

**`ValueError: <model>: head_dim 256 exceeds the 128 the paged kernels support`** — the
checkpoint's geometry is outside what these kernels serve. The message names the cause.

**`ValueError: decode backend 'flash' cannot run here: ...`** — you named a backend that
this GPU or build cannot run. The engine refuses rather than silently substituting.
`check_hooks.py --backends-only` lists what is available.

**Tail latency spikes minutes into a run** — a graph shape warmup did not cover. Check
`inference_graph_captures_in_service`; it should be 0.

**Requests rejected with 429** — the scheduler queue or submission queue is full; raise
`max_waiting_requests`/`max_pending_requests` or reduce offered load.

**Requests failing with `KV_POOL_EXHAUSTED`** — a single request cannot fit the pool even
with everything else evicted. Raise `num_blocks` or lower `max_prompt_tokens`.

**Greedy output differs from Hugging Face by one token** — expected at logit near-ties;
fp16 reduction order differs between kernels. The gate treats a difference at a position
whose stock top-2 margin is under 0.02 as a tie. A difference at a clear margin is a bug —
please report it with the prompt.

---

## 10. What is not supported

No tensor or pipeline parallelism, no multi-GPU, no quantized weights on the serving path,
no structured/constrained output, no LoRA, no multimodal inputs, no `n > 1` sampling. INT8
KV is implemented but disabled. Speculative decoding has an experimental, default-off
greedy path whose speedup is not yet validated. See
[`engineering-report.md` §8](engineering-report.md#8-limitations-and-threats-to-validity)
for the full list with reasons.

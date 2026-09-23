# RTX 4060 final evaluation

Date: 2026-09-24  
Target: NVIDIA GeForce RTX 4060, Ada `sm_89`, 8 GB VRAM  
Models: `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`  
Purpose: release evidence for the single-GPU engine, including accepted changes and
negative results. This document does not turn an isolated kernel win into a serving claim.

## Executive result

The shippable RTX 4060 configuration is FP16 with 256-token KV pages, graphed Triton
`per_head` decode, and eager direct-paged FlashAttention-2 prefill. It is exposed by
`engine.server.api:create_rtx4060_flash_app`.

FlashAttention is not a universal replacement in this engine:

- Direct paged Flash prefill is accepted for the measured long-context configuration.
- Flash decode is rejected because its FA2 kvcache path cannot be CUDA-graph captured on
  the installed build; losing graph replay dominates its narrow kernel-level wins.
- Dense-gather Flash prefill for normal 16-token pages is rejected end to end.
- The normal compact-page configuration therefore remains graphed Triton decode plus SDPA
  prefill. Backend selection is geometry- and device-gated and never silently falls back
  when a named backend is unavailable.

The complete service passes its CPU, CUDA, hook, token-identity and external HTTP gates.
The measurements support publishing this as an experimentally validated Qwen3 single-GPU
engine. Mistral/Llama support is structural and guarded, but must not be advertised as
performance-validated until checkpoints from those families pass the same end-to-end gate.

## Evaluated environment

| item | observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, 24 SMs, compute capability 8.9 |
| VRAM | 8188 MiB reported by the driver |
| L2 | approximately 24-25 MB |
| measured FP16 decode-like bandwidth | 257.4 GB/s |
| Qwen3-0.6B weight-only decode floor | 4.63 ms/token |
| PyTorch | 2.14.0+cu130 |
| Triton | 3.8.0 |
| FlashAttention | source-built 2.8.4, sm89-only inference build |
| Flash page constraint | paged-KV block size divisible by 256 |

The FlashAttention build is a machine-local measurement dependency rather than a portable
wheel guarantee. Capability checks report a precise reason when it is unavailable.

## Architecture changes in this checkpoint

### Backend and device policy

- Decode and prefill are independent registered backends.
- Ada `sm_89` has a measured policy: FP16, `per_head` decode, and Flash prefill when the
  selected 256-token geometry supports it; compact 16-token pages resolve to SDPA prefill.
- Model `dtype="auto"` consumes the measured device policy instead of choosing BF16 only
  because the architecture supports it. BF16 failed the early-token gate for this path.
- Flash backends declare themselves graph-unsafe on this tested FA2 build. Other phases
  retain their CUDA graphs rather than disabling graphs globally.

### FlashAttention adapters

- `flash_attn_with_kvcache` reads the engine's physical paged K/V pools directly for
  compatible page geometry and native GQA.
- Prefill uses `cache_seqlens = existing_context + query_length` and groups unequal query
  lengths so staging padding never becomes causal input.
- Chunk-length groups are formed once from pinned host staging per engine step. The first
  implementation copied lengths from CUDA to CPU once per transformer layer; removing
  those 28 synchronizations was necessary for an end-to-end win.
- Equal-length rows take a direct path without gather, scatter, or padding buffers.
- `num_splits` is exposed for decode research. The measured upstream automatic choice won.
- A dense-gather Flash backend remains registered below the production backends so its
  rejected 16-page experiment stays reproducible.

### Runtime, serving and correctness

- The optimized RTX 4060 application factory fixes the evaluated model, dtype, page size,
  backend choices and graph buckets in one reproducible entry point.
- Sampling generators are owned by request id, not seed, and are discarded at every
  terminal path.
- Exact-prefix entries cache a next-token decision only for parameter-free greedy requests.
  Sampled or penalized requests still reuse complete KV blocks, but rerun the residual
  prompt position so their private RNG advances once per generated token. This prevents a
  cached sampled token leaking into a different policy and fixes same-seed HTTP replay.
- The A/B harness can isolate Flash decode, direct Flash prefill, split-count experiments
  and dense-gather prefill, select dtype explicitly, and record the selected model/dtype.
- Hook verification reports eager-only backends as an explicit property rather than a
  missing-graph failure.

## Measurement protocol

Serving conclusions use interleaved repeated A/B runs from
`benchmarks/reliability/ab.py`, not a single kernel timing. Each arm is warmed, exercises
the same closed-loop prompt profile, records median/min/max/spread, and passes a
stock-Transformers leading-token gate. A change smaller than run-to-run spread is labelled
unresolved. Kernel sweeps explain results but do not override end-to-end latency.

FP16 is used for accepted Ada comparisons because it passed the token gate. Near-ties are
recorded with their stock top-two margin; non-tie early divergence rejects a path.

## Accepted results

### Qwen3-0.6B, long profile, FP16, page 256

| metric, Flash prefill versus SDPA | change | interpretation |
|---|---:|---|
| prefill step p50 | **-7.8%** | clear win beyond spread |
| prefill penalty p50 | **-13.4%** | less decode disruption during prefill |
| fused GPU p50 | **-8.8%** | packed decode+prefill work improves |
| ITL p99 | **-22.6%** | material tail improvement |
| expected gap | -4.2% | favorable, but within 7.6% run spread |
| TTFT p50 | -8.1% | favorable, but unresolved by spread |

The follow-up optimized run independently measured expected gap -5.3%, prefill step -7.3%,
prefill GPU -6.4%, fused GPU -7.4%, and prefill penalty -16.6%.

### Qwen3-1.7B, long profile, FP16, page 256

| metric, Flash prefill versus SDPA | change | interpretation |
|---|---:|---|
| expected gap | **-5.3%** | end-to-end phase-weighted win |
| TTFT p50 | **-13.7%** | stable across repeats |
| ITL p99 / p999 | **-9.9% / -9.0%** | improved tail |
| prefill GPU p50 | **-5.3%** | device-side win |
| fused GPU p50 | **-11.8%** | strong packed-step win |
| prefill step p50 | -17.0% | favorable; spread is 14.8% |

This is the larger practical FP16 target on an 8 GB card. A 4B FP16 model does not fit
with a useful KV pool; evaluating 4B requires the planned W4A16 weight path and its own
quality gate.

## Rejected and unresolved experiments

| experiment | measured outcome | decision / cause |
|---|---|---|
| Flash decode versus graphed Triton, short profile | ITL p50 **+113.9%**, expected gap **+110.0%** | rejected; FA2 kvcache workspace setup/synchronization is not stream-capture safe, so eager execution loses badly |
| Flash decode versus eager Triton, FP16 probe | ITL p50 **+11.9%** | rejected even without the graph advantage |
| Flash decode kernel sweep | about 15% faster only at batch 1, context 4096; neutral/slower elsewhere | too narrow to justify serving selection |
| forced Flash decode split counts | generally worse than automatic | retain upstream automatic split planning |
| first combined Flash arm | ITL p50 **+11.5%**, prefill step **+16.1%** | rejected; changed two phases and included per-layer host synchronization |
| pre-optimization eager Flash prefill | prefill step **+11.4%** | rejected implementation; exposed the adapter synchronization defect |
| dense-gather Flash prefill on page 16 | prefill step **+38.5%**, TTFT **+52.6%**, expected gap **+32.0%** | rejected; gather/materialization cost dominates attention savings |
| BF16 Flash path | early non-tie token divergence | rejected for the published policy |
| Qwen3-4B FP16 | exceeds the 8 GB memory budget before a production KV allocation | not run; requires weight quantization rather than unsafe overcommit |

The direct attention microbenchmark shows why prefill remained worth pursuing: paged Flash
prefill was roughly 1.5-2.6x faster than the SDPA attention component in tested cells.
The smaller end-to-end gain is expected because projections, MLP, scheduling and decode are
outside that component.

## Verification gates

| gate | final result |
|---|---|
| non-CUDA test suite | **296 passed**, 195 deselected |
| CUDA test suite | **193 passed**, 1 skipped, 1 expected failure, 296 deselected |
| expected skip | two-model FP32 speculative parity exceeds the practical 8 GB budget |
| expected failure | known INT8 paged-KV recompute/preemption token drift on Ada; INT8 KV remains disabled |
| compact-page hook gate | no problems; 6 decode, 15 prefill and 90 fused graphs |
| Flash-page hook gate | no problems; decode graphs retained, Flash prefill explicitly eager-only |
| optimized external HTTP smoke | **6/6 passed** |
| HTTP concurrency sample | 8 callers, 1136.6 output tok/s, 6.76x versus measured serial estimate |
| formatting | `git diff --check` clean |

The HTTP smoke covers readiness/model discovery, completion, streaming chat, stop and seed
determinism, unsupported-input refusal, concurrent callers and Prometheus metrics. It is an
outside process and does not import engine internals.

## Reproduction

Run the accepted service:

```bash
.venv/bin/uvicorn engine.server.api:create_rtx4060_flash_app \
  --factory --host 127.0.0.1 --port 8000
```

Run gates and phase-isolated measurements:

```bash
.venv/bin/pytest -q -m 'not cuda'
.venv/bin/pytest -q -m cuda
.venv/bin/python scripts/check_hooks.py --dtype float16 --block-size 256 \
  --decode-attention per_head --prefill-attention flash \
  --out results/rtx4060/check_hooks_final_flash_prefill_block256_fp16.json
.venv/bin/python -m benchmarks.kernels.flash_paged_decode_sweep \
  --out results/rtx4060/flash_paged_decode_sweep.json
.venv/bin/python -m benchmarks.kernels.flash_paged_prefill_sweep \
  --out results/rtx4060/flash_paged_prefill_sweep.json
```

## Evidence index

- `results/rtx4060/roofline.json`: measured bandwidth and decode floor.
- `results/rtx4060/flash_paged_decode_sweep.json`: decode kernel cells and split counts.
- `results/rtx4060/flash_paged_prefill_sweep.json`: prefill component sweep.
- `results/rtx4060/ab_flash_decode_block256_short.json`: production decode rejection.
- `results/rtx4060/probe_flash_decode_fp16.json`: eager FP16 decode rejection.
- `results/rtx4060/ab_flash_prefill_block256_long_eager_fp16.json`: adapter before sync removal.
- `results/rtx4060/ab_flash_prefill_block256_long_graphs_fp16.json`: accepted 0.6B run.
- `results/rtx4060/ab_flash_prefill_block256_long_optimized_fp16.json`: optimized 0.6B confirmation.
- `results/rtx4060/ab_flash_prefill_qwen3_1.7b_long_graphs_fp16.json`: accepted larger-model run.
- `results/rtx4060/ab_flash_dense_prefill_block16_long_graphs_fp16.json`: dense-gather rejection.
- `results/rtx4060/check_hooks_final_block16_fp16.json`: compact-page architecture gate.
- `results/rtx4060/check_hooks_final_flash_prefill_block256_fp16.json`: accepted Flash gate.
- `results/rtx4060/live_smoke_optimized_flash.json`: external service verification.

## Publication boundary and next work

The evidence supports these claims: real continuous batching, paged KV ownership, graph
replay, deterministic per-request sampling, an OpenAI-compatible single-GPU service, and a
measured Ada Flash-prefill policy for Qwen3-0.6B/1.7B. It does not support claims of universal
Flash decode superiority, validated Mistral throughput, 4B FP16 support, production INT8 KV,
or multi-GPU scale.

The next optimization phase must begin with Nsight Compute/System measurements of the live
decode step and K/V metadata path. Candidate CUDA/Triton changes are accepted only after
coalescing, cache behavior, occupancy, shared-memory use, numerical parity and end-to-end A/B
evidence are all recorded alongside rejected variants.

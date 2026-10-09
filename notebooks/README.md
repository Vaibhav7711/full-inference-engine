# Notebooks

Colab/Kaggle notebooks that produce the measurements quoted in
[`../docs/engineering-report.md`](../docs/engineering-report.md). Each one clones the repository,
installs into the host's own PyTorch build, and writes every number it prints to JSON so a claim can
be traced back to the run that produced it.

| notebook | what it measures |
|---|---|
| [`part3_memory_and_bandwidth.ipynb`](part3_memory_and_bandwidth.ipynb) | resident-memory attribution per engine stage (NVML *and* allocator counters) across eight warm-up variants, with the two memory fixes run as in-process A/Bs against their own pre-fix behaviour; then achieved decode bandwidth against a measured fp16 GEMV probe over a (batch × context) sweep |
| [`part4_three_engines_breakdown.ipynb`](part4_three_engines_breakdown.ipynb) | naive HF `generate` vs this engine (graphs on and off) vs vLLM at matched KV: TTFT, TPOT, single-stream and batched throughput, startup, NVML memory with this engine's post-fix stage attribution (graph pool, executables outside the allocator, graph counts, widest capture); then a `torch.profiler` breakdown of one prefill and one decode step per engine by transformer sub-block — FFN / attention-projection / lm_head GEMMs apportioned by shape, attention, norm, RoPE, activation, sampling — plus kernel launches vs `cudaGraphLaunch` per step |

Every timed window in part 3 is validated before it is reported: decode-only steps, constant batch,
and zero CUDA-graph captures inside the window. A window that fails a check is labelled, not averaged
in.

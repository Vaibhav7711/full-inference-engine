# Qwen3-8B at W4A16 on a Kaggle T4 — results

Produced by the CLI-driven iteration kernel (`kaggle kernels push`, one `run.py` that clones
the repo at a pinned SHA); the notebook `scripts/kaggle_qwen3_8b_t4x2.ipynb` runs the same
commands. Read against `docs/qwen3-8b-optimization-map.md` §6.

- `kernel_bench/w4a16_bench_v5.json` — the `tl.dot` kernel the first A/B ran on: 488 ms/step at M = 1.
- `kernel_bench/w4a16_bench_v7.json` — first `tl.sum` GEMV (one row per program): 54 ms at M = 1, 293 at M = 8.
- `kernel_bench/w4a16_bench_v8.json` — the dequantise-once rows kernel, which lost at every M (removed).
- `kernel_bench/w4a16_bench_v10.json` — mantissa-insertion dequant, final defaults: 47 ms at M = 1, 79 at M = 2; dense 187 flat.
- `check_hooks_8b_first_pass.json` — the 8B live gate on the final kernels: `problems: []`, memory, warm-up, graph replay report.
- `ab_fused_step_chat_1x15s.json` — the first 8B A/B that measured the engine rather than one kernel (1 repeat, 15 s).

Per-step numbers in the bench files are the per-shape medians scaled by count (36 layers,
one `lm_head`); CUDA-event timing, 20 iterations after 3 warm-up calls, Triton 3.6 / torch 2.11.

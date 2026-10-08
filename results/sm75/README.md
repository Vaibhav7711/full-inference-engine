# sm75 — Turing (Tesla T4)

Results on this path are filed under [`../t4/`](../t4/) (transcribed Kaggle sessions) and
`../*.json` at the repository root (early phases). They were produced on the engine's
original target and are the basis of every claim in the README and the engineering report.

What this tier cannot do, and why the `sm80plus` path exists: `tl.dot` lowers to scalar FMA
below sm_80 (the tiled prefill kernel compiled with `mma_sync = 0` and 128 spills), there
is no bf16 tensor-core path, and no FP8 storage. The measured memory bandwidth is 258 GB/s.

Tier definitions: [`engine/backends/arch.py`](../../engine/backends/arch.py).

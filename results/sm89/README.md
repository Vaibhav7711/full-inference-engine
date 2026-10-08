# sm89 — Ada (RTX 4060)

Results on this path are filed under [`../rtx4060/`](../rtx4060/) (the FlashAttention
isolation campaign and `roofline.json`) and `../rtx4060_*.json`. The accepted serving
configuration from that work is recorded in `MEASURED[89]` in
[`engine/backends/policy.py`](../../engine/backends/policy.py).

The measured card, from `rtx4060/roofline.json`:

```
NVIDIA GeForce RTX 4060 · sm_89 · 24 SMs · 8.18 GB · L2 25.2 MB
gemv 257.4 GB/s  (T4: 258)      weight-only decode floor 4.63 ms
```

Same bandwidth as the T4, 6.3x the L2, working `mma.sync`, FP8. So on this path decode
cannot get faster at the same byte count; the levers are fewer bytes (FP8 KV, W4A16),
compute-bound prefill on the tensor cores (`tiled`, Flash), and L2-resident working sets.

Round-1 plan and the gate order for first boot: [`docs/sm89-round1.md`](../../docs/sm89-round1.md).
Tier definitions: [`engine/backends/arch.py`](../../engine/backends/arch.py).

"""Architecture paths: which capability tier a device belongs to, by name.

The engine does not fork per GPU - the scheduler, paged KV, graph capture and sampler are
the same code everywhere - but kernels, defaults and recorded results do split along
capability lines, and those lines are compute-capability gates, not product names.
"RTX" spans Ampere (30xx, sm_86), Ada (40xx, sm_89) and Blackwell (50xx, sm_120); "T4"
is one Turing part. Results and plans are therefore filed under the gate they depend on.

Gates this codebase actually checks:

- **sm_80**: `tl.dot` lowers to `mma.sync` (fp16 m16n8k16) and `cp.async` exists. Below
  it the tiled prefill kernel compiles to scalar FMA and spills (journal: "never used the
  tensor cores"), and bf16 has no tensor-core path.
- **sm_89**: FP8 (E4M3/E5M2) storage and tensor cores. Gates `kv_cache_dtype="fp8"`.
- **sm_90**: `wgmma`, TMA, FlashAttention-3. Nothing in this tree uses them yet; the gate
  exists so a registry entry can refuse cleanly rather than fail at first launch.
- **sm_100**: FP4. Same status.

Importable without Triton or a GPU.
"""

from __future__ import annotations

# Path names, chosen so a results directory and a `MEASURED` entry read the same way.
TURING = "sm75"
SM80_PLUS = "sm80plus"


def path_for(profile) -> str:
    """The results/plan path a device belongs to: `"sm75"` or `"sm80plus"`.

    A device below sm_75 is grouped with Turing; nothing here has been built or measured
    for it, and the Turing defaults are the most conservative ones available.
    """
    sm = profile.sm if profile is not None else 0
    return SM80_PLUS if sm >= 80 else TURING


def has_mma_sync(sm: int) -> bool:
    """fp16 `mma.sync.m16n8k16`; the tensor-core path Triton's `tl.dot` needs."""
    return sm >= 80


def has_async_copy(sm: int) -> bool:
    """`cp.async`, and therefore multi-stage software pipelining into shared memory."""
    return sm >= 80


def has_bf16_tensor_cores(sm: int) -> bool:
    return sm >= 80


def has_fp8(sm: int) -> bool:
    """E4M3/E5M2 storage and tensor cores (Ada and later)."""
    return sm >= 89


def has_wgmma(sm: int) -> bool:
    """Hopper warpgroup MMA and TMA. Not used by any kernel here yet."""
    return sm >= 90


def has_fp4(sm: int) -> bool:
    """Blackwell FP4. Not used by any kernel here yet."""
    return sm >= 100


def capabilities(profile) -> dict[str, bool | str | int]:
    """Everything a startup log or result JSON should record about the tier."""
    sm = profile.sm if profile is not None else 0
    return {
        "path": path_for(profile),
        "sm": sm,
        "mma_sync": has_mma_sync(sm),
        "async_copy": has_async_copy(sm),
        "bf16_tensor_cores": has_bf16_tensor_cores(sm),
        "fp8": has_fp8(sm),
        "wgmma": has_wgmma(sm),
        "fp4": has_fp4(sm),
    }

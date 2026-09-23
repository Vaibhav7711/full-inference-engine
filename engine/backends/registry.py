"""Attention backend registry: which kernel serves a phase, and whether it can here.

Until this module the engine chose its attention implementation with an `if/elif` chain
over three string constants, and the defaults were whatever had won on a Tesla T4. That
is a dead end for two reasons. A kernel that only exists on some GPUs - the tiled Triton
prefill needs `tl.dot` to lower to `mma.sync` (sm_80+), FlashAttention needs a wheel
built for the architecture - has to be *conditionally* available rather than a constant
someone remembers not to pass. And a new kernel should be addable without editing the
engine, or nobody writes one.

A backend declares three things: how to run one phase, what it needs (`available`
returns the reason it cannot run here, or None), and how much it should be preferred
when the engine is asked for `"auto"`. `resolve()` then picks, and refuses a named
backend that cannot run *with the reason*, rather than falling back silently - a silent
fallback is how the T4 measured the eager path for a week and called it a graph result.

`describe()` returns the whole table, which `scripts/check_hooks.py` prints, so the
first thing a new GPU tells you is which kernels it can actually run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

Phase = Literal["decode", "prefill"]


@dataclass(frozen=True)
class Geometry:
    """Everything a backend needs to know about the shapes it will be handed."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: str = "float16"
    kv_dtype: str = "float16"

    @property
    def gqa_group(self) -> int:
        return self.num_q_heads // self.num_kv_heads


@dataclass(frozen=True)
class Backend:
    """One attention implementation for one phase."""

    name: str
    phase: Phase
    run: Callable
    # Reason this backend cannot run on this device with this geometry, or None.
    available: Callable[[object, Geometry], str | None] = lambda profile, geometry: None
    # Higher wins when the engine is asked for "auto" and several are available. The
    # numbers encode measured preference, not aspiration: see `policy.py`.
    priority: int = 0
    # False keeps the phase out of CUDA graphs (a kernel that allocates or synchronizes).
    graph_safe: bool = True
    # Free-text, shown by `describe()`; say what the backend is, not that it is good.
    summary: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[tuple[Phase, str], Backend] = {}


def register(backend: Backend, *, replace: bool = False) -> Backend:
    """Add a backend. Re-registering the same name is an error unless `replace`."""
    key = (backend.phase, backend.name)
    if key in _REGISTRY and not replace:
        raise ValueError(f"{backend.phase} backend {backend.name!r} is already registered")
    _REGISTRY[key] = backend
    return backend


def unregister(phase: Phase, name: str) -> None:
    _REGISTRY.pop((phase, name), None)


def get(phase: Phase, name: str) -> Backend:
    try:
        return _REGISTRY[(phase, name)]
    except KeyError:
        known = ", ".join(sorted(n for p, n in _REGISTRY if p == phase)) or "none"
        raise KeyError(f"unknown {phase} backend {name!r}; registered: {known}") from None


def names(phase: Phase) -> tuple[str, ...]:
    return tuple(sorted(name for registered_phase, name in _REGISTRY if registered_phase == phase))


def describe(phase: Phase, profile, geometry: Geometry) -> list[dict]:
    """Every registered backend for a phase, with whether it can run here and why not."""
    rows = []
    for name in names(phase):
        backend = get(phase, name)
        reason = backend.available(profile, geometry)
        rows.append({
            "name": name, "available": reason is None, "reason": reason,
            "priority": backend.priority, "graph_safe": backend.graph_safe,
            "summary": backend.summary, "tags": list(backend.tags),
        })
    return rows


def resolve(phase: Phase, requested: str | None, profile, geometry: Geometry) -> Backend:
    """The backend to use. `None` or `"auto"` picks the highest-priority available one.

    A named backend that cannot run raises, quoting the reason. Serving a request on a
    different kernel than the operator asked for is a measurement and a correctness
    hazard, not a convenience.
    """
    if requested and requested != "auto":
        backend = get(phase, requested)
        reason = backend.available(profile, geometry)
        if reason is not None:
            raise ValueError(f"{phase} backend {requested!r} cannot run here: {reason}")
        return backend
    candidates = [
        backend for backend in (get(phase, name) for name in names(phase))
        if backend.available(profile, geometry) is None
    ]
    if not candidates:
        rows = describe(phase, profile, geometry)
        detail = "; ".join(f"{row['name']}: {row['reason']}" for row in rows)
        raise RuntimeError(f"no {phase} attention backend can run here ({detail})")
    return max(candidates, key=lambda backend: (backend.priority, backend.name))

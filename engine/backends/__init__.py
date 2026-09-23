"""Pluggable attention backends and the per-device policy that selects them.

Importing this package registers the built-in backends, so `resolve("decode", "auto", ...)`
works without the caller knowing what exists. A new kernel is added by calling
`register(Backend(...))` - from here, from a plugin, or from a test - and needs no change
to the engine.
"""

from engine.backends.registry import (
    Backend, Geometry, describe, get, names, register, resolve, unregister,
)
from engine.backends import builtin as _builtin  # noqa: F401  (registers the built-ins)
from engine.backends.policy import DeviceDefaults, MEASURED, defaults_for, report

__all__ = [
    "Backend", "Geometry", "register", "unregister", "get", "names", "describe", "resolve",
    "DeviceDefaults", "MEASURED", "defaults_for", "report",
]

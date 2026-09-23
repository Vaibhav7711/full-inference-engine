"""Per-device serving defaults: what to run when nobody has measured this GPU yet.

Two kinds of knowledge live here and they must not be confused. `MEASURED` holds
settings that an A/B on that exact architecture decided, with the journal entry that
decided them; `auto` resolution falls back to backend priorities and hardware capability
when a device is not in the table, which is a starting point and is labelled as one.

The engine calls `defaults_for()` once at construction. Everything it returns can be
overridden by an explicit keyword, because a benchmark that cannot pin the configuration
cannot compare anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.backends.registry import Geometry, describe, resolve


@dataclass(frozen=True)
class DeviceDefaults:
    decode_attention: str
    prefill_attention: str
    dtype: str
    # Why each choice was made, for the startup log and the result JSON. A default whose
    # provenance is not recorded turns into folklore within two sessions.
    reasons: dict[str, str] = field(default_factory=dict)


# Architecture -> settings an A/B on that architecture chose, with the entry that did it.
MEASURED: dict[int, dict] = {
    75: {  # Tesla T4, Turing. docs/optimization-journal.md, phases 1b-2c.
        "decode_attention": "per_head",
        "prefill_attention": "sdpa",
        "dtype": "float16",
        "reasons": {
            "decode_attention": "T4 A/B: gqa 0.95-1.06x (L2 already dedups the group read)",
            "prefill_attention": "T4 A/B: sdpa -40%/-66% step vs per_token; tiled has no "
                                 "mma.sync on sm_75",
            "dtype": "Turing has no bf16 tensor cores",
        },
    },
    89: {  # RTX 4060, Ada. docs/rtx4060-plan.md, Flash isolation campaign.
        "decode_attention": "per_head",
        "prefill_attention": "flash",
        "dtype": "float16",
        "reasons": {
            "decode_attention": "RTX 4060 A/B: FA2 decode is eager-only and +114% ITL "
                                "against graphed per_head at the short serving point",
            "prefill_attention": "RTX 4060 long-context A/B: optimized FA2 prefill "
                                 "reduced prefill step 7.8% (0.6B) and TTFT 13.7% (1.7B); "
                                 "requires 256-token pages",
            "dtype": "FP16 passed the early stock-token gate; BF16 Flash diverged at token 2",
        },
    },
}


def _pick(phase: str, profile, geometry: Geometry) -> tuple[str, str | None]:
    """Highest-priority runnable backend, or the highest-priority one with a note.

    This is policy, not enforcement: the engine calls `resolve()` with whatever comes
    back, and that is where an unrunnable backend raises with its reason. Returning a
    name here even when nothing can run keeps `report()` and startup logs informative on
    a machine where, say, Triton is missing.
    """
    try:
        return resolve(phase, None, profile, geometry).name, None
    except RuntimeError as error:
        from engine.backends.registry import get, names

        fallback = max(names(phase), key=lambda name: (get(phase, name).priority, name))
        return fallback, str(error)


def defaults_for(profile, geometry: Geometry) -> DeviceDefaults:
    """Serving defaults for this GPU: measured where they exist, capability-led otherwise."""
    if profile is not None and profile.sm in MEASURED:
        entry = MEASURED[profile.sm]
        available = {
            phase: {row["name"] for row in describe(phase, profile, geometry) if row["available"]}
            for phase in ("decode", "prefill")
        }
        decode = entry["decode_attention"]
        prefill = entry["prefill_attention"]
        reasons = dict(entry["reasons"])
        # A measured default that cannot run here (an INT8 pool, a missing wheel) must
        # not be forced: fall back to auto and say so.
        for phase, chosen in (("decode", decode), ("prefill", prefill)):
            if chosen in available[phase]:
                continue
            alternatives = available[phase]
            if alternatives:
                from engine.backends.registry import get

                replacement = max(alternatives, key=lambda name: (get(phase, name).priority, name))
                reasons[f"{phase}_attention"] = (
                    f"measured default {chosen!r} cannot run here; selected {replacement}"
                )
            else:
                # Nothing else can run either (a missing Triton, an unsupported head
                # dimension). Keep the measured name: the engine's own `resolve()` is
                # the enforcement point and reports the specific reason.
                replacement = chosen
                reasons[f"{phase}_attention"] = (
                    f"measured default {chosen!r} cannot run here and no alternative can; "
                    f"the engine will refuse with the reason"
                )
            if phase == "decode":
                decode = replacement
            else:
                prefill = replacement
        return DeviceDefaults(decode, prefill, entry["dtype"], reasons)

    decode, decode_note = _pick("decode", profile, geometry)
    prefill, prefill_note = _pick("prefill", profile, geometry)
    dtype = "bfloat16" if profile is not None and profile.sm >= 80 else "float16"
    sm = f"sm_{profile.sm}" if profile is not None else "no CUDA device"
    return DeviceDefaults(
        decode_attention=decode,
        prefill_attention=prefill,
        dtype=dtype,
        reasons={
            "decode_attention": f"unmeasured on {sm}; highest-priority available backend"
                                + (f"; {decode_note}" if decode_note else ""),
            "prefill_attention": f"unmeasured on {sm}; highest-priority available backend"
                                 + (f"; {prefill_note}" if prefill_note else ""),
            "dtype": "bf16 tensor cores from sm_80" if dtype == "bfloat16"
                     else "fp16 (no bf16 tensor cores below sm_80)",
            "measure": "run `ab.py --setting decode_kernel` and `--setting prefill_kernel` "
                       "on this device, then add the winners to MEASURED",
        },
    )


def report(profile, geometry: Geometry) -> dict:
    """Everything a new GPU should be asked before it serves: what runs, and what will.

    Every backend is listed with whether it can run here and why not, so an unfamiliar
    GPU can be inspected before anything is served on it.
    """
    defaults = defaults_for(profile, geometry)
    return {
        "device": str(profile) if profile is not None else None,
        "measured": profile is not None and profile.sm in MEASURED,
        "defaults": {
            "decode_attention": defaults.decode_attention,
            "prefill_attention": defaults.prefill_attention,
            "dtype": defaults.dtype,
        },
        "reasons": defaults.reasons,
        "decode_backends": describe("decode", profile, geometry),
        "prefill_backends": describe("prefill", profile, geometry),
    }

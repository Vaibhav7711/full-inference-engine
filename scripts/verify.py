"""Run everything, in the order that fails cheapest first. For a machine with a GPU.

The Colab workflow ran one thing at a time and pasted results back, which made ordering a
manual discipline. On a local GPU the whole sequence should be one command, and it should
stop at the first stage that invalidates the ones after it: there is no point measuring a
kernel that does not compile, or comparing configurations on an engine whose tests fail.

    python -m scripts.verify                 # tests only, a few minutes
    python -m scripts.verify --full          # plus roofline, kernel A/B, soaks
    python -m scripts.verify --stage kernels # one stage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from time import perf_counter

# (stage, description, command). Ordered so a failure invalidates only what follows.
STAGES: list[tuple[str, str, list[str]]] = [
    ("static", "kernel source checks: no GPU, no compilation, arithmetic and language rules",
     ["pytest", "tests/kernels/test_kernel_static_checks.py",
      "tests/kernels/test_device.py", "-q"]),
    ("cpu", "pure-Python engine logic, no GPU",
     ["pytest", "tests/runtime", "tests/scheduler", "tests/server", "tests/cache",
      "tests/reliability/test_ab_settings.py", "tests/reliability/test_sweep.py", "-q"]),
    ("kernels", "every Triton kernel, compile smoke tests first",
     ["pytest", "tests/kernels", "-q", "-x"]),
    ("engine", "token-identity against a stock unpatched checkpoint",
     ["pytest", "tests/batching/test_continuous_batching.py", "-q", "-m", "cuda"]),
    ("soak", "mixed-arrival reliability with a KV page audit",
     ["pytest", "tests/reliability/test_soak.py", "-q"]),
]

MEASUREMENTS: list[tuple[str, str, list[str]]] = [
    ("roofline", "achieved bandwidth and the decode floor for this device",
     [sys.executable, "-m", "benchmarks.kernels.roofline"]),
    ("prefill-kernel", "tiled vs per-token prefill attention, with a tile sweep",
     [sys.executable, "-m", "benchmarks.kernels.prefill_attention_ab", "--sweep-tiles"]),
    ("prefill-sweep", "the Phase B matrix: chunk size, prompt profile, budget packing",
     [sys.executable, "-m", "benchmarks.reliability.sweep", "--repeats", "3"]),
]


def run(name: str, description: str, command: list[str]) -> bool:
    argv = [sys.executable, "-m"] + command if command[0] == "pytest" else command
    print(f"\n{'=' * 72}\n{name}: {description}\n{'=' * 72}")
    started = perf_counter()
    code = subprocess.call(argv)
    elapsed = perf_counter() - started
    print(f"-- {name}: {'ok' if code == 0 else f'FAILED ({code})'} in {elapsed:.1f}s")
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also run the measurement suite (tens of minutes)")
    parser.add_argument("--stage", nargs="*", default=None,
                        help="run only these stages")
    parser.add_argument("--keep-going", action="store_true",
                        help="do not stop at the first failing stage")
    args = parser.parse_args()

    try:
        import torch
        from engine.kernels.device import current_device
        profile = current_device()
        print(f"device: {profile}" if profile else "device: none (CPU-only stages will run)")
        if profile:
            print(f"torch {torch.__version__}")
    except Exception as error:
        print(f"could not profile the device: {error}")

    plan = list(STAGES) + (list(MEASUREMENTS) if args.full else [])
    if args.stage:
        wanted = set(args.stage)
        plan = [item for item in plan if item[0] in wanted]
        if not plan:
            print(f"no stage matched {sorted(wanted)}; "
                  f"available: {[s[0] for s in STAGES + MEASUREMENTS]}")
            return 2

    failed = []
    for name, description, command in plan:
        if not run(name, description, command):
            failed.append(name)
            if not args.keep_going:
                print(f"\nstopping at {name}; later stages assume it passes. "
                      f"Use --keep-going to run them anyway.")
                break

    print(f"\n{'=' * 72}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(plan)} stages passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

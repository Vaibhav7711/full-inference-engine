"""Exercise every hook the engine exposes, and report a pass/fail matrix.

`check_hooks.py` answers "is this GPU set up and did warmup capture everything".
This answers the other question: are the *seams* sound - does every registered attention
backend produce correct tokens, does the policy pick what it claims, does the model hook
find its targets and refuse what it cannot serve, and does sampling behave per request.

Run it after any change to `engine/backends`, `engine/model/adapters.py`, the kernels or
the sampler, and on any new GPU before trusting a benchmark from it.

    CUDA_VISIBLE_DEVICES=0 python scripts/verify_hooks.py --out results/<run>/verify_hooks.json

Every check is independent and reports its own reason on failure; the exit code is 1 if
any check failed. Checks whose preconditions are absent (no flash wheel, no second model
downloaded) are SKIPPED, not failed - an unavailable backend is a fact about the machine,
not a defect.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class Report:
    rows: list[dict] = field(default_factory=list)

    def record(self, name: str, status: str, detail: str = "", **extra) -> None:
        self.rows.append({"check": name, "status": status, "detail": detail, **extra})
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip"}[status]
        print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""), flush=True)

    def run(self, name: str, function) -> None:
        try:
            detail = function()
        except _Skip as skip:
            self.record(name, "skip", str(skip))
        except Exception as error:  # a check that raises is a failure, with the traceback
            self.record(name, "fail", f"{type(error).__name__}: {error}",
                        traceback="".join(traceback.format_exception(error))[-2000:])
        else:
            self.record(name, "pass", detail or "")

    @property
    def failed(self) -> list[str]:
        return [row["check"] for row in self.rows if row["status"] == "fail"]


class _Skip(Exception):
    """Raised by a check whose preconditions are not met on this machine."""


# ---------------------------------------------------------------------------
# 1. Model hook
# ---------------------------------------------------------------------------

def model_hook_checks(report: Report, loaded) -> None:
    from engine.model import adapters

    def discovery():
        described = adapters.describe(loaded.model)
        if described["mlp_modules"] == 0:
            raise AssertionError("no SwiGLU MLP modules were found structurally")
        if described["norm_modules"] == 0:
            raise AssertionError("no RMSNorm modules were found")
        if described["rope_module"] is None:
            raise AssertionError("the model's RoPE module could not be located")
        if described["unsupported_reason"]:
            raise AssertionError(described["unsupported_reason"])
        return (f"{described['model_type']}: {described['mlp_modules']} MLP, "
                f"{described['norm_modules']} norm, rope in {described['rope_module']}")

    def refusals():
        # Synthetic configs: each must be refused, and the reason must name the cause.
        class Config:
            def __init__(self, **kwargs):
                self.model_type = "synthetic"
                self.num_hidden_layers = 4
                self.hidden_size = 1024
                self.num_attention_heads = 16
                self.num_key_value_heads = 8
                self.vocab_size = 1000
                self.max_position_embeddings = 2048
                self.__dict__.update(kwargs)

        cases = {
            "head_dim": (Config(head_dim=256), "128"),
            "gqa": (Config(num_key_value_heads=5), "KV heads"),
            "sliding_window": (Config(sliding_window=512), "sliding"),
            "moe": (Config(num_experts=8), "experts"),
            "mla": (Config(kv_lora_rank=64), "latent"),
        }
        for name, (config, expected) in cases.items():
            reason = adapters.unsupported_reason(config)
            if reason is None:
                raise AssertionError(f"{name}: unsupported geometry was accepted")
            if expected.lower() not in reason.lower():
                raise AssertionError(f"{name}: reason {reason!r} does not mention {expected!r}")
        supported = Config(head_dim=128)
        if adapters.unsupported_reason(supported) is not None:
            raise AssertionError("a supported geometry was refused")
        return f"{len(cases)} unsupported geometries refused with reasons"

    def fusion_round_trip():
        from engine.kernels.rope import install_triton_rope, uninstall_triton_rope
        from engine.kernels.swiglu import install_triton_swiglu, uninstall_triton_swiglu

        installed = install_triton_swiglu(loaded.model)
        removed = uninstall_triton_swiglu(loaded.model)
        if installed != removed:
            raise AssertionError(f"installed {installed} SwiGLU modules, restored {removed}")
        rope_installed = install_triton_rope(loaded.model)
        rope_removed = uninstall_triton_rope(loaded.model)
        if rope_installed != rope_removed:
            raise AssertionError("RoPE install/uninstall did not round-trip")
        return f"{installed} MLP modules, {rope_installed} RoPE module"

    print("\nmodel hook")
    report.run("model.structural_discovery", discovery)
    report.run("model.unsupported_geometry_refused", refusals)
    report.run("model.fusion_install_uninstall", fusion_round_trip)


# ---------------------------------------------------------------------------
# 2. Backend and device hooks
# ---------------------------------------------------------------------------

def backend_hook_checks(report: Report, geometry, profile) -> list[dict]:
    from engine.backends import Backend, describe, register, resolve, unregister
    from engine.backends.policy import MEASURED, defaults_for

    tables = {phase: describe(phase, profile, geometry) for phase in ("decode", "prefill")}

    def every_row_explains_itself():
        for phase, rows in tables.items():
            for row in rows:
                if not row["available"] and not row["reason"]:
                    raise AssertionError(f"{phase}/{row['name']} is unavailable without a reason")
        counts = {phase: sum(row["available"] for row in rows) for phase, rows in tables.items()}
        if not counts["decode"] or not counts["prefill"]:
            raise AssertionError(f"no backend can run for some phase: {counts}")
        return ", ".join(f"{phase}: {count}/{len(tables[phase])} available"
                         for phase, count in counts.items())

    def named_unavailable_raises():
        for phase, rows in tables.items():
            blocked = [row for row in rows if not row["available"]]
            if not blocked:
                continue
            name = blocked[0]["name"]
            try:
                resolve(phase, name, profile, geometry)
            except ValueError as error:
                if blocked[0]["reason"].split(";")[0][:20] not in str(error):
                    raise AssertionError(f"{phase}/{name} raised without its reason: {error}")
                return f"{phase}/{name} refused with its reason"
            raise AssertionError(f"{phase}/{name} is unavailable but resolve() accepted it")
        raise _Skip("every backend is available here; nothing to refuse")

    def auto_respects_priority():
        probe = Backend(name="__verify_probe__", phase="decode", run=lambda *a, **k: None,
                        priority=10_000, summary="temporary probe")
        register(probe)
        try:
            chosen = resolve("decode", "auto", profile, geometry).name
            if chosen != "__verify_probe__":
                raise AssertionError(f"auto picked {chosen}, not the highest-priority backend")
        finally:
            unregister("decode", "__verify_probe__")
        after = resolve("decode", "auto", profile, geometry).name
        return f"probe selected, then unregistered (auto now {after})"

    def policy_is_labelled():
        defaults = defaults_for(profile, geometry)
        measured = profile is not None and profile.sm in MEASURED
        for key in ("decode_attention", "prefill_attention"):
            if key not in defaults.reasons:
                raise AssertionError(f"{key} has no recorded reason")
            if not measured and "unmeasured" not in defaults.reasons[key]:
                raise AssertionError(
                    f"{key} on an unmeasured architecture is not labelled as such: "
                    f"{defaults.reasons[key]!r}"
                )
        label = "measured" if measured else "unmeasured"
        return (f"{label}: decode={defaults.decode_attention}, "
                f"prefill={defaults.prefill_attention}, dtype={defaults.dtype}")

    print("\nbackend and device hooks")
    report.run("backend.every_row_explains_itself", every_row_explains_itself)
    report.run("backend.named_unavailable_raises", named_unavailable_raises)
    report.run("backend.auto_respects_priority", auto_respects_priority)
    report.run("device.policy_is_labelled", policy_is_labelled)
    return tables


# ---------------------------------------------------------------------------
# 3. Every available backend produces correct tokens, eagerly and under graphs
# ---------------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "Explain how paged attention, continuous batching and chunked prefill work together "
    "in a production inference engine. Include scheduling and memory details, and say "
    "what happens when the KV pool runs out of free pages.",
    "2 + 2 =",
]


def _reference(loaded, max_new: int) -> list[list[int]]:
    import torch

    from engine.kernels.rope import stock_rope

    model, tokenizer = loaded.model, loaded.tokenizer
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    outputs = []
    with stock_rope(), torch.inference_mode():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(loaded.device)
            generated = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            outputs.append(generated[0, ids.shape[1]:].tolist())
    return outputs


def backend_correctness_checks(report: Report, loaded, tables, args) -> None:
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    reference = _reference(loaded, args.max_new)

    def run_pair(decode_name: str, prefill_name: str, graphs):
        engine = ContinuousBatchingEngine(
            loaded.model, loaded.tokenizer, loaded.device,
            num_blocks=args.num_blocks, block_size=16, max_active=4, prefix_cache_blocks=0,
            prefill_chunk_size=32, max_prefill_tokens_per_iteration=32,
            decode_attention=decode_name, prefill_attention=prefill_name,
            cuda_graph_batch_sizes=graphs,
        )
        if graphs:
            engine.warmup()
            if engine.lazy_graph_captures:
                raise AssertionError(
                    f"{engine.lazy_graph_captures} graph captures happened after warmup"
                )
        outputs = engine.generate(PROMPTS, max_new_tokens=args.max_new)
        divergence = [
            next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
            for out, ref in zip(outputs, reference)
        ]
        leaked = (engine.block_manager.snapshot()["used_blocks"]
                  - engine.prefix_cache.snapshot()["cached_blocks"]
                  - len(engine._graph_dummy_blocks))
        if leaked:
            raise AssertionError(f"{leaked} KV blocks were not returned to the pool")
        early = [index for index in divergence if index is not None and index < args.min_identical]
        if early:
            raise AssertionError(
                f"diverges from stock at {divergence} (within the first "
                f"{args.min_identical} tokens: a wrong kernel, not rounding)"
            )
        return f"divergence vs stock: {divergence}"

    print("\nbackend correctness (tokens vs stock Transformers)")
    for phase, other in (("decode", "prefill"), ("prefill", "decode")):
        fixed = "sdpa" if phase == "decode" else "per_head"
        for row in tables[phase]:
            name = row["name"]
            label = f"tokens.{phase}.{name}"
            if not row["available"]:
                report.record(label, "skip", row["reason"])
                continue
            decode_name = name if phase == "decode" else fixed
            prefill_name = name if phase == "prefill" else fixed
            report.run(label, lambda d=decode_name, p=prefill_name: run_pair(d, p, None))

    print("\nbackend correctness under CUDA graphs")
    for phase in ("decode", "prefill"):
        for row in tables[phase]:
            name = row["name"]
            if not row["available"]:
                continue
            decode_name = name if phase == "decode" else "per_head"
            prefill_name = name if phase == "prefill" else "sdpa"
            report.run(
                f"graphs.{phase}.{name}",
                lambda d=decode_name, p=prefill_name: run_pair(d, p, (2, 4)),
            )


# ---------------------------------------------------------------------------
# 4. Sampling hook
# ---------------------------------------------------------------------------

def sampling_checks(report: Report, loaded, args) -> None:
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import GREEDY, GenerationRequest, RequestState, SamplingParams

    reference = _reference(loaded, args.max_new)

    def build():
        return ContinuousBatchingEngine(
            loaded.model, loaded.tokenizer, loaded.device, num_blocks=args.num_blocks,
            block_size=16, max_active=4, prefix_cache_blocks=0,
        )

    def greedy_rows_unaffected():
        engine = build()
        settings = [GREEDY, SamplingParams(temperature=0.9, top_p=0.95, seed=17), GREEDY]
        outputs = engine.generate(PROMPTS, max_new_tokens=args.max_new, sampling=settings)
        for index in (0, 2):
            if outputs[index] != reference[index]:
                raise AssertionError(f"greedy row {index} changed when a neighbour sampled")
        if len(outputs[1]) != args.max_new:
            raise AssertionError("the sampled row did not produce the requested tokens")
        return "greedy rows identical to stock while sharing a batch with a sampled row"

    def seeded_is_reproducible():
        setting = SamplingParams(temperature=1.0, top_p=0.9, seed=4242)
        runs = [build().generate(PROMPTS[:1], max_new_tokens=args.max_new, sampling=setting)[0]
                for _ in range(2)]
        if runs[0] != runs[1]:
            raise AssertionError("a seeded request produced different tokens across engines")
        return f"seed 4242 reproduced {len(runs[0])} tokens"

    def stop_tokens_and_logprobs():
        engine = build()
        ids = loaded.tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0].tolist()
        greedy = engine.generate([PROMPTS[0]], max_new_tokens=4)[0]
        engine.reset()
        request = GenerationRequest(
            "verify-stop", prompt_token_count=len(ids), max_new_tokens=8, prompt_token_ids=ids,
            sampling=SamplingParams(stop_token_ids=frozenset({greedy[1]}), logprobs=3),
        )
        engine.submit(request)
        steps = 0
        while engine.has_unfinished_requests and steps < 100:
            engine.step()
            steps += 1
        if request.state is not RequestState.FINISHED or request.finish_reason != "STOP":
            raise AssertionError(f"stop token ignored: {request.state}/{request.finish_reason}")
        if request.output_token_ids != greedy[:2]:
            raise AssertionError("generation did not stop at the stop token")
        if len(request.output_logprobs) != len(request.output_token_ids):
            raise AssertionError("logprobs were not recorded for every token")
        return f"stopped at token {greedy[1]} with {len(request.output_logprobs)} logprob rows"

    print("\nsampling hook")
    report.run("sampling.greedy_rows_unaffected", greedy_rows_unaffected)
    report.run("sampling.seeded_reproducible", seeded_is_reproducible)
    report.run("sampling.stop_tokens_and_logprobs", stop_tokens_and_logprobs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float16",
                        help="float16 keeps results comparable with the T4 record; "
                             "'auto' selects bf16 on sm_80+")
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--min-identical", type=int, default=8,
                        help="tokens that must match stock before a difference counts as "
                             "a wrong kernel rather than rounding")
    parser.add_argument("--skip-backends", action="store_true",
                        help="hook checks only; no per-backend generation")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from benchmarks.common import device_clock_record, environment_record, git_record
    from engine.backends import Geometry, report as backend_report
    from engine.kernels.device import current_device
    from engine.model import load_model
    from engine.model.adapters import geometry_of

    started = time.perf_counter()
    loaded = load_model(args.model, dtype=args.dtype)
    shapes = geometry_of(loaded.model.config)
    geometry = Geometry(
        num_q_heads=shapes.num_q_heads, num_kv_heads=shapes.num_kv_heads,
        head_dim=shapes.head_dim, block_size=16,
        dtype=str(loaded.dtype).removeprefix("torch."),
    )
    profile = current_device()
    print(f"model {args.model} ({loaded.dtype}) on {profile}")

    report = Report()
    model_hook_checks(report, loaded)
    tables = backend_hook_checks(report, geometry, profile)
    if not args.skip_backends:
        backend_correctness_checks(report, loaded, tables, args)
        sampling_checks(report, loaded, args)

    passed = sum(row["status"] == "pass" for row in report.rows)
    skipped = sum(row["status"] == "skip" for row in report.rows)
    print(f"\n{passed} passed, {len(report.failed)} failed, {skipped} skipped "
          f"in {time.perf_counter() - started:.0f}s")
    if report.failed:
        print("FAILED: " + ", ".join(report.failed))

    if args.out:
        payload = {
            "model": args.model, "dtype": str(loaded.dtype),
            "geometry": shapes.__dict__,
            "backends": backend_report(profile, geometry),
            "checks": report.rows,
            "passed": passed, "failed": report.failed, "skipped": skipped,
            "git": git_record(), "environment": environment_record(loaded.device),
            "clocks": device_clock_record(),
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"Saved -> {out}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

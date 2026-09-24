"""Kaggle T4 x2 screen for Qwen3-4B draft-model candidates.

This is deliberately a cheap pre-integration screen. The target is placed on GPU 0 and
one draft candidate at a time on GPU 1. It measures acceptance, draft cost, target verify
cost and end-to-end speedup with identical greedy outputs. The winning candidate still
has to pass the live paged-engine integration gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from time import perf_counter

import torch

from engine.model import ExplicitDecodeRunner, load_model


PROMPTS = [
    {"stratum": "chat", "prompt": "Explain why the sky appears blue in three sentences."},
    {"stratum": "code", "prompt": "Write a Python function that returns the first n Fibonacci numbers."},
    {"stratum": "math", "prompt": "A train travels 120 km in 90 minutes. Explain its average speed step by step."},
    {"stratum": "summary", "prompt": "Paged attention stores KV tensors in fixed-size blocks. Continuous batching changes the active rows every step. Summarize these two ideas."},
    {"stratum": "copy", "prompt": "Repeat and then explain this sequence: alpha beta gamma alpha beta gamma alpha beta gamma."},
    {"stratum": "qa", "prompt": "What is the capital of France, and which river runs through it?"},
]


@dataclass
class _State:
    cache: object
    mask: torch.Tensor
    next_token: torch.Tensor


@dataclass
class _Result:
    token_ids: list[int]
    accepted: int
    proposed: int
    rounds: int
    draft_ms: float
    verify_ms: float
    commit_ms: float


def _sync(*devices: torch.device) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _prefill(model, ids: torch.Tensor, mask: torch.Tensor) -> _State:
    out = model(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
    return _State(out.past_key_values, mask, out.logits[:, -1].argmax(-1, keepdim=True))


@torch.inference_mode()
def _decode(model, state: _State, token: torch.Tensor) -> _State:
    device = token.device
    mask = torch.cat((state.mask, torch.ones((1, 1), device=device, dtype=state.mask.dtype)), 1)
    out = model(input_ids=token, attention_mask=mask, past_key_values=state.cache,
                use_cache=True, return_dict=True)
    return _State(out.past_key_values, mask, out.logits[:, -1].argmax(-1, keepdim=True))


def _crop(state: _State, length: int) -> _State:
    remove = state.mask.shape[1] - length
    if remove < 0:
        raise RuntimeError("cannot grow a cache by cropping")
    if remove:
        if not hasattr(state.cache, "crop"):
            raise TypeError("draft screening requires a transformers cache with crop()")
        state.cache.crop(-remove)
    return _State(state.cache, state.mask[:, :length], state.next_token)


@torch.inference_mode()
def _generate(target, draft, tokenizer, prompt: str, *, target_device: torch.device,
              draft_device: torch.device, max_new_tokens: int, depth: int) -> _Result:
    encoded = tokenizer(prompt, return_tensors="pt")
    target_state = _prefill(target, encoded.input_ids.to(target_device),
                            encoded.attention_mask.to(target_device))
    draft_state = _prefill(draft, encoded.input_ids.to(draft_device),
                           encoded.attention_mask.to(draft_device))
    configured_eos = target.generation_config.eos_token_id
    eos = configured_eos if configured_eos is not None else tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    emitted: list[int] = []
    accepted_total = proposed_total = rounds = 0
    draft_ms = verify_ms = commit_ms = 0.0

    while len(emitted) < max_new_tokens:
        round_depth = min(depth, max_new_tokens - len(emitted))
        context_length = target_state.mask.shape[1]

        _sync(draft_device)
        started = perf_counter()
        proposal_tensors = []
        for _ in range(round_depth):
            proposal_tensors.append(draft_state.next_token)
            draft_state = _decode(draft, draft_state, draft_state.next_token)
        proposals_draft = torch.cat(proposal_tensors, dim=1)
        _sync(draft_device)
        draft_ms += (perf_counter() - started) * 1000
        proposed_total += round_depth

        proposals_target = proposals_draft.to(target_device)
        target_mask = torch.cat((
            target_state.mask,
            torch.ones((1, round_depth), device=target_device, dtype=target_state.mask.dtype),
        ), 1)
        _sync(target_device)
        started = perf_counter()
        verified = target(input_ids=proposals_target, attention_mask=target_mask,
                          past_key_values=target_state.cache, use_cache=True, return_dict=True)
        target_tail = verified.logits[0].argmax(-1)
        _sync(target_device)
        verify_ms += (perf_counter() - started) * 1000

        started = perf_counter()
        proposals = proposals_draft[0].tolist()
        target_predictions = [int(target_state.next_token.item())] + target_tail[:-1].tolist()
        bonus = int(target_tail[-1].item())
        accepted = 0
        for draft_token, target_token in zip(proposals, target_predictions):
            if draft_token != target_token:
                break
            accepted += 1
        round_tokens = (
            proposals + [bonus]
            if accepted == round_depth
            else proposals[:accepted] + [target_predictions[accepted]]
        )
        eos_index = next((i for i, token in enumerate(round_tokens) if token in eos_ids), None)
        if eos_index is not None:
            round_tokens = round_tokens[:eos_index + 1]
        retained = min(accepted, len(round_tokens))
        accepted_total += retained
        emitted.extend(round_tokens[:max_new_tokens - len(emitted)])
        rounds += 1

        if eos_index is not None or len(emitted) >= max_new_tokens:
            commit_ms += (perf_counter() - started) * 1000
            break
        accepted_length = context_length + retained
        target_state = _crop(_State(verified.past_key_values, target_mask,
                                    target_tail[-1:].reshape(1, 1)), accepted_length)
        draft_state = _crop(draft_state, accepted_length)
        final = round_tokens[-1]
        target_state = _decode(target, target_state,
                               torch.tensor([[final]], device=target_device))
        draft_state = _decode(draft, draft_state,
                              torch.tensor([[final]], device=draft_device))
        _sync(target_device, draft_device)
        commit_ms += (perf_counter() - started) * 1000

    return _Result(emitted, accepted_total, proposed_total, rounds,
                   draft_ms, verify_ms, commit_ms)


def _timed(fn, *devices):
    _sync(*devices)
    started = perf_counter()
    result = fn()
    _sync(*devices)
    return result, (perf_counter() - started) * 1000


def _load_prompts(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return PROMPTS
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"stratum": str(row.get("stratum", "custom")),
                         "prompt": str(row["prompt"])})
    if not rows:
        raise ValueError("prompt file is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--depths", default="2,3,4")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise SystemExit("pair screening requires two visible CUDA GPUs")
    if args.warmup < 0 or args.runs <= 0 or args.max_new_tokens <= 0:
        parser.error("warmup must be non-negative; runs and max-new-tokens must be positive")
    depths = [int(value) for value in args.depths.split(",")]
    if not depths or any(depth <= 0 for depth in depths):
        parser.error("depths must be positive")

    target_device, draft_device = torch.device(args.target_device), torch.device(args.draft_device)
    if target_device == draft_device:
        parser.error("target and draft devices must be different for the Kaggle T4 x2 screen")
    prompts = _load_prompts(args.prompts)
    target_loaded = load_model(args.target_model, dtype="float16", device=target_device)
    draft_loaded = load_model(args.draft_model, dtype="float16", device=draft_device)
    target, draft, tokenizer = target_loaded.model, draft_loaded.model, target_loaded.tokenizer
    if target_loaded.tokenizer.get_vocab() != draft_loaded.tokenizer.get_vocab():
        raise SystemExit("target and draft token maps differ")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(target_loaded.tokenizer, name) != getattr(draft_loaded.tokenizer, name):
            raise SystemExit(f"target and draft {name} differ")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": args.target_model, "draft_model": args.draft_model,
        "target_revision": target_loaded.resolved_revision,
        "draft_revision": draft_loaded.resolved_revision,
        "target_device": str(target_device), "draft_device": str(draft_device),
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "rows": [],
    }

    baseline_runner = ExplicitDecodeRunner(target, tokenizer, target_device)
    for item in prompts:
        prompt = item["prompt"]
        for _ in range(args.warmup):
            baseline_runner.generate(prompt, max_new_tokens=args.max_new_tokens)
        baseline_times, reference = [], None
        for _ in range(args.runs):
            reference, elapsed = _timed(
                lambda: baseline_runner.generate(prompt, max_new_tokens=args.max_new_tokens),
                target_device,
            )
            baseline_times.append(elapsed)
        baseline_ms = statistics.median(baseline_times)

        for depth in depths:
            for _ in range(args.warmup):
                _generate(target, draft, tokenizer, prompt, target_device=target_device,
                          draft_device=draft_device, max_new_tokens=args.max_new_tokens,
                          depth=depth)
            samples = []
            last = None
            for _ in range(args.runs):
                last, elapsed = _timed(
                    lambda: _generate(
                        target, draft, tokenizer, prompt, target_device=target_device,
                        draft_device=draft_device, max_new_tokens=args.max_new_tokens,
                        depth=depth,
                    ), target_device, draft_device,
                )
                samples.append((last, elapsed))
            elapsed_ms = statistics.median(sample[1] for sample in samples)
            representative = min(samples, key=lambda sample: abs(sample[1] - elapsed_ms))[0]
            mismatch = next((i for i, pair in enumerate(zip(
                representative.token_ids, reference.token_ids
            )) if pair[0] != pair[1]), None)
            result["rows"].append({
                "stratum": item["stratum"], "prompt": prompt, "depth": depth,
                "baseline_ms": baseline_ms, "speculative_ms": elapsed_ms,
                "speedup": baseline_ms / elapsed_ms,
                "tokens_match": representative.token_ids == reference.token_ids,
                "first_mismatch": mismatch,
                "accepted": representative.accepted, "proposed": representative.proposed,
                "acceptance_rate": (representative.accepted / representative.proposed
                                    if representative.proposed else 0.0),
                "mean_accepted_per_round": (representative.accepted / representative.rounds
                                            if representative.rounds else 0.0),
                "rounds": representative.rounds,
                "draft_ms": representative.draft_ms,
                "verify_ms": representative.verify_ms,
                "commit_ms": representative.commit_ms,
                "raw_speculative_ms": [sample[1] for sample in samples],
                "raw_baseline_ms": baseline_times,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(row["tokens_match"] for row in result["rows"]):
        raise SystemExit("pair screen found a token mismatch; inspect the saved artifact")


if __name__ == "__main__":
    main()

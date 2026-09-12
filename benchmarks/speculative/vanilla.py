from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from engine.model import ExplicitDecodeRunner, load_model
from engine.speculative import VanillaSpeculativeDecoder


def timed(callable_: object) -> tuple[object, float]:
    torch.cuda.synchronize()
    start = perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, (perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla greedy draft-model speculation")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--speculation-depth", type=int, default=4)
    parser.add_argument("--target-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("results/vanilla_speculation.json"))
    args = parser.parse_args()
    target = load_model(args.target_model)
    draft = load_model(args.draft_model)
    if target.tokenizer.vocab_size != draft.tokenizer.vocab_size:
        parser.error("target and draft tokenizers must have matching vocabularies")
    reference, reference_ms = timed(lambda: ExplicitDecodeRunner(target.model, target.tokenizer, target.device).generate(args.prompt, max_new_tokens=args.max_new_tokens))
    speculative, speculative_ms = timed(lambda: VanillaSpeculativeDecoder(target.model, draft.model, target.tokenizer, target.device).generate(args.prompt, max_new_tokens=args.max_new_tokens, speculation_depth=args.speculation_depth))
    record = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "speculation_depth": args.speculation_depth,
        "reference_ms": reference_ms,
        "speculative_ms": speculative_ms,
        "speedup": reference_ms / speculative_ms,
        "target_tokens_match_reference": speculative.token_ids == reference.token_ids,
        "output_tokens": len(speculative.token_ids),
        "accepted_draft_tokens": speculative.accepted_draft_tokens,
        "proposed_draft_tokens": speculative.proposed_draft_tokens,
        "acceptance_rate": speculative.acceptance_rate,
        "rounds": speculative.rounds,
        "target_forward_passes_baseline": len(reference.token_ids),
        "target_forward_passes_speculative": speculative.rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

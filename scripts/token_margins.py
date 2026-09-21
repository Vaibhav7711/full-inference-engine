"""Top-2 logit margins along the stock greedy continuation of the identity prompts.

The A/B gate refuses an arm that diverges from stock Transformers within the first
`--min-identical-tokens` tokens, on the reasoning that a wrong kernel shows up at once
while rounding shows up late. That reasoning has a hole: a near-tie between the top two
logits can sit at any position, and then every implementation that is not bit-identical
to stock flips it, early or late. Three different chunked-prefill paths diverged at the
same positions (4 on prompt 0, 13 on prompt 3) on the T4, which is the tie signature.

This prints, for each generated position, the fp32 margin between the best and second
best logit under stock kernels. A margin below ~1e-2 (fp16 logits are ~3 decimal digits
at magnitude 10-30) is a tie that no kernel choice is responsible for.

    CUDA_VISIBLE_DEVICES=0 python scripts/token_margins.py --positions 0:4 1:18 3:13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--positions", nargs="*", default=[],
                        help="prompt:position pairs to highlight, e.g. 0:4 3:13")
    parser.add_argument("--smallest", type=int, default=5,
                        help="also print the N smallest margins per prompt")
    args = parser.parse_args()

    import torch
    from benchmarks.reliability.ab import IDENTITY_PROMPTS
    from engine.kernels.rope import stock_rope
    from engine.model import load_model

    loaded = load_model(args.model)
    model, tokenizer = loaded.model, loaded.tokenizer
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    highlight: dict[int, set[int]] = {}
    for item in args.positions:
        prompt_index, position = item.split(":")
        highlight.setdefault(int(prompt_index), set()).add(int(position))

    with stock_rope(), torch.inference_mode():
        for index, prompt in enumerate(IDENTITY_PROMPTS):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(loaded.device)
            out = model.generate(
                ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, output_scores=True,
                return_dict_in_generate=True,
            )
            margins = []
            for position, scores in enumerate(out.scores):
                top = torch.topk(scores[0].float(), 2).values
                margins.append((position, (top[0] - top[1]).item()))
            print(f"\nprompt {index}: {len(margins)} tokens generated")
            for position, margin in sorted(margins, key=lambda m: m[1])[:args.smallest]:
                print(f"  smallest margin  position {position:3d}  {margin:.4f}")
            for position in sorted(highlight.get(index, ())):
                if position < len(margins):
                    margin = margins[position][1]
                    verdict = "TIE (below fp16 resolution)" if margin < 1e-2 else (
                        "near-tie" if margin < 5e-2 else "clear")
                    print(f"  divergence at    position {position:3d}  {margin:.4f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

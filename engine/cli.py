from __future__ import annotations

import argparse

from engine.model import ExplicitDecodeRunner, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit prefill/decode LLM runtime")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()
    loaded = load_model(args.model, revision=args.revision, dtype=args.dtype)
    result = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device).generate(
        args.prompt, max_new_tokens=args.max_new_tokens
    )
    print(result.text)
    print(result.metrics.as_dict())


if __name__ == "__main__":
    main()

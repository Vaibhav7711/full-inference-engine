from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from engine.graphs import assess_graph_eligibility, capture_decode_graph
from engine.model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-shape CUDA Graph decode experiment")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("results/cuda_graph_decode.json"))
    args = parser.parse_args()
    if args.decode_steps < 3:
        parser.error("decode-steps must be at least 3")
    loaded = load_model(args.model)
    inputs = loaded.tokenizer(args.prompt, return_tensors="pt").input_ids.to(loaded.device)
    eligibility = assess_graph_eligibility(fixed_batch_size=True, fixed_sequence_length=True, static_cache=True, dynamic_arrivals=False)
    # Normal fixed-shape decode baseline. Its first decode is excluded to match graph
    # capture, which executes the first decode while recording the graph.
    from transformers.cache_utils import StaticCache
    prompt_length = inputs.shape[1]
    normal_cache = StaticCache(config=loaded.model.config, max_cache_len=prompt_length + args.decode_steps)
    positions = torch.arange(prompt_length, device=loaded.device)
    with torch.inference_mode():
        normal_prefill = loaded.model(input_ids=inputs, past_key_values=normal_cache, cache_position=positions, use_cache=True, return_dict=True)
        normal_tokens = [int(normal_prefill.logits[:, -1, :].argmax(dim=-1).item())]
        first_decode = loaded.model(input_ids=torch.tensor([[normal_tokens[-1]]], device=loaded.device), past_key_values=normal_cache, cache_position=torch.tensor([prompt_length], device=loaded.device), use_cache=True, return_dict=True)
        normal_tokens.append(int(first_decode.logits[:, -1, :].argmax(dim=-1).item()))
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for index in range(args.decode_steps - 2):
        with torch.inference_mode():
            output = loaded.model(input_ids=torch.tensor([[normal_tokens[-1]]], device=loaded.device), past_key_values=normal_cache, cache_position=torch.tensor([prompt_length + index + 1], device=loaded.device), use_cache=True, return_dict=True)
        normal_tokens.append(int(output.logits[:, -1, :].argmax(dim=-1).item()))
    end.record(); end.synchronize()
    normal_ms = start.elapsed_time(end)

    captured, first_token = capture_decode_graph(loaded.model, inputs, max_cache_len=inputs.shape[1] + args.decode_steps)
    # Capture records the first decode but does not guarantee its static output buffer
    # is materialized for host-side consumption. Replay that same position once before
    # timing; overwriting the same static-cache slot is intentional and idempotent.
    captured.replay(first_token, inputs.shape[1])
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    second_token = int(captured.logits[:, -1, :].argmax(dim=-1).item())
    tokens = [first_token, second_token]
    start.record()
    for index in range(args.decode_steps - 2):
        logits = captured.replay(tokens[-1], inputs.shape[1] + index + 1)
        tokens.append(int(logits[:, -1, :].argmax(dim=-1).item()))
    end.record(); end.synchronize()
    record = {
        "eligibility": {"eligible": eligibility.eligible, "reasons": eligibility.reasons},
        "workload": {"batch_size": 1, "prompt_tokens": inputs.shape[1], "decode_steps": args.decode_steps, "static_cache": True},
        "cuda_graph_decode_ms": start.elapsed_time(end),
        "cuda_graph_decode_ms_per_token": start.elapsed_time(end) / (args.decode_steps - 2),
        "normal_decode_ms": normal_ms,
        "normal_decode_ms_per_token": normal_ms / (args.decode_steps - 2),
        "speedup": normal_ms / start.elapsed_time(end),
        "token_sequences_match": normal_tokens == tokens,
        "token_ids": tokens,
        "note": "Capture overhead excluded; compare only against a normal decode loop with the same fixed shapes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

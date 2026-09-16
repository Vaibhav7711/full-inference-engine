"""Inspect real decode metadata staging in ContinuousBatchingEngine.

Loads Qwen3-0.6B, obtains real DECODING requests, and invokes the exact metadata
preparation method used immediately before the production decode forward.
"""

from __future__ import annotations

import argparse

import torch

from engine.runtime import GenerationRequest, RequestState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-blocks", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA and a Qwen checkpoint.")

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, "cuda", num_blocks=args.num_blocks,
        block_size=16, max_active=2, prefix_cache_blocks=0,
    )
    engine.eos_ids.clear()
    for row, prompt in enumerate((
        "Paged KV block tables route each request to physical GPU pages.",
        "Pinned host buffers stage a batched decode metadata transfer.",
    )):
        ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        assert engine.submit(GenerationRequest(
            f"metadata-{row}", len(ids), 4, prompt_token_ids=ids,
        ))
    engine.step()  # admit + real prefill; both requests now have next_token_id.
    active = [request for request in engine.scheduler.active.values() if request.state is RequestState.DECODING]
    assert len(active) == 2

    host = {
        "input_ids": engine._host_input_ids,
        "position_ids": engine._host_position_ids,
        "seq_lens": engine._host_seq_lens,
        "block_tables": engine._host_block_tables,
    }
    device = {
        "input_ids": engine._device_input_ids,
        "position_ids": engine._device_position_ids,
        "seq_lens": engine._device_seq_lens,
        "block_tables": engine._device_block_tables,
    }
    print("persistent host staging buffers:", {
        name: {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "pinned": tensor.is_pinned(), "ptr": tensor.data_ptr()}
        for name, tensor in host.items()
    })
    print("persistent device buffers:", {
        name: {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device), "ptr": tensor.data_ptr()}
        for name, tensor in device.items()
    })

    base_device_ptrs = {name: tensor.data_ptr() for name, tensor in device.items()}
    input_ids, position_ids, block_tables, seq_lens = engine._prepare_decode_metadata(active)
    torch.cuda.synchronize()
    valid_table_widths = [len(request.block_table) for request in active]
    print("active request CPU metadata:", [
        {
            "id": request.request_id,
            "next_token": request.next_token_id,
            "position_and_prewrite_length": request.allocation.sequence_length if request.allocation else None,
            "valid_block_table": request.block_table,
        }
        for request in active
    ])
    print("staged GPU views:", {
        "input_ids": input_ids.cpu().tolist(),
        "position_ids": position_ids.cpu().tolist(),
        "seq_lens": seq_lens.cpu().tolist(),
        "valid_block_table_prefixes": [block_tables[row, :width].cpu().tolist() for row, width in enumerate(valid_table_widths)],
        "view_pointers_match_persistent_buffers": {
            "input_ids": input_ids.data_ptr() == base_device_ptrs["input_ids"],
            "position_ids": position_ids.data_ptr() == base_device_ptrs["position_ids"],
            "seq_lens": seq_lens.data_ptr() == base_device_ptrs["seq_lens"],
            "block_tables": block_tables.data_ptr() == base_device_ptrs["block_tables"],
        },
    })

    # The second call performs new copies but returns the same storage views.
    repeated = engine._prepare_decode_metadata(active)
    torch.cuda.synchronize()
    print("second staging call reuses all device storage:", [
        tensor.data_ptr() == first.data_ptr()
        for tensor, first in zip(repeated, (input_ids, position_ids, block_tables, seq_lens))
    ])
    bytes_per_iteration = (
        input_ids.numel() * input_ids.element_size()
        + position_ids.numel() * position_ids.element_size()
        + seq_lens.numel() * seq_lens.element_size()
        + block_tables.numel() * block_tables.element_size()
    )
    print("metadata bytes copied for this decode batch:", bytes_per_iteration)


if __name__ == "__main__":
    main()

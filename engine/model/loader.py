from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class LoadedModel:
    model_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    dtype: torch.dtype
    device: torch.device


def load_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    *,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
) -> LoadedModel:
    """Load an actual decoder-only checkpoint for CUDA inference.

    This runtime intentionally refuses CPU: Stage 1 timing and cache behaviour must
    describe the GPU execution path that later stages optimize.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Start a GPU runtime before loading a model.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=dtype
    ).to(device)
    model.eval()
    return LoadedModel(model_name, model, tokenizer, dtype, device)

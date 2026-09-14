from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class LoadedModel:
    model_name: str
    requested_revision: str | None
    resolved_revision: str | None
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    dtype: torch.dtype
    device: torch.device


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    """Resolve a user dtype request, selecting a Tensor-Core dtype for the GPU.

    Turing GPUs such as the Colab T4 (compute capability 7.5) do not have BF16
    Tensor Core support, so ``auto`` deliberately selects FP16 there. Ampere and
    newer devices may use BF16 when PyTorch reports it as supported.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype != "auto":
        try:
            return _DTYPES[dtype]
        except KeyError as error:
            choices = ", ".join(("auto", *_DTYPES))
            raise ValueError(f"unsupported dtype {dtype!r}; choose one of: {choices}") from error
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    *,
    dtype: str | torch.dtype = "auto",
    revision: str | None = None,
) -> LoadedModel:
    """Load an actual decoder-only checkpoint for CUDA inference.

    This runtime intentionally refuses CPU: Stage 1 timing and cache behaviour must
    describe the GPU execution path that later stages optimize.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Start a GPU runtime before loading a model.")
    device = torch.device("cuda")
    resolved_dtype = resolve_dtype(dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=resolved_dtype
    ).to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    return LoadedModel(
        model_name=model_name,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        model=model,
        tokenizer=tokenizer,
        dtype=resolved_dtype,
        device=device,
    )

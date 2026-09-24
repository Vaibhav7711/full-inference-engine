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

    Turing GPUs such as the Kaggle T4 (compute capability 7.5) do not have BF16
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
    major, minor = torch.cuda.get_device_capability(device)
    # A measured serving policy includes numerical validation, not just tensor-core
    # capability. On sm_89, BF16 Flash prefill failed the early token gate while FP16
    # passed and won, so `auto` must honor that result before the generic heuristic.
    try:
        from engine.backends.policy import MEASURED

        measured = MEASURED.get(major * 10 + minor, {}).get("dtype")
        if measured in _DTYPES:
            return _DTYPES[measured]
    except ImportError:  # keep the standalone loader usable during minimal installs
        pass
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def tie_output_embeddings(model) -> bool:
    """Share `lm_head` with the input embedding when the config asks for it.

    Newer transformers builds refuse to tie when a checkpoint ships both tensors (Qwen3
    does), leaving two copies of a 311 MB matrix on the GPU. Tying is only safe when the
    two are identical, so that is checked rather than assumed. Returns True when the
    output projection shares storage with the embedding table afterwards.
    """
    if not getattr(model.config, "tie_word_embeddings", False):
        return False
    output = model.get_output_embeddings()
    inputs = model.get_input_embeddings()
    if output is None or inputs is None:
        return False
    if output.weight.data_ptr() == inputs.weight.data_ptr():
        return True
    if output.weight.shape != inputs.weight.shape or not torch.equal(output.weight, inputs.weight):
        return False
    output.weight = inputs.weight
    return True


def load_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    *,
    dtype: str | torch.dtype = "auto",
    revision: str | None = None,
    device: str | torch.device | None = None,
) -> LoadedModel:
    """Load an actual decoder-only checkpoint for CUDA inference.

    This runtime intentionally refuses CPU: Stage 1 timing and cache behaviour must
    describe the GPU execution path that later stages optimize.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Start a GPU runtime before loading a model.")
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required. load_model(device=...) must name a CUDA device.")
    if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is not visible")
    resolved_dtype = resolve_dtype(dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=resolved_dtype
    ).to(device)
    model.eval()
    tie_output_embeddings(model)
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

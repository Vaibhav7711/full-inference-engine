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


def load_awq_tinygemm_model(model_path: str, *, triton_prefill_rows: int | None = 64,
                            dequant_rows: int | None = 192) -> LoadedModel:
    """Load a local GEMM-format AWQ checkpoint through native PyTorch INT4.

    This experimental path avoids an external AWQ runtime and never materializes the
    full FP16 model. Packed weights are repacked one projection at a time, keeping a
    4B checkpoint within an 8 GB card's loading budget.

    ``triton_prefill_rows``: projections with at least this many rows use the Triton
    int4pack GEMM over the same packed buffers (RTX 4060 Qwen3-4B-AWQ A/B: prefill GPU
    -46%, fused step -34%, ITL p99 -33%); decode keeps tinygemm. ``None`` restores
    tinygemm everywhere. From ``dequant_rows`` rows (whole-prompt prefill) the weight is
    dequantized to FP16 scratch and multiplied by cuBLAS, 1.8x the fused kernel at 2048.
    """
    import json
    from pathlib import Path

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig

    from engine.quantization.awq_tinygemm import AWQTinyGemmLinear, repack_awq_to_tinygemm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(model_path)
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    quantization = getattr(config, "quantization_config", None)
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "awq" \
            or quantization.get("bits") != 4 or quantization.get("group_size") != 128 \
            or quantization.get("version", "").lower() != "gemm":
        raise ValueError("native loader requires a 4-bit, group-128, GEMM-format AWQ checkpoint")
    delattr(config, "quantization_config")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        files = sorted({root / filename for filename in weight_map.values()})
    else:
        files = [root / "model.safetensors"]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError("AWQ safetensors checkpoint is incomplete")

    quantized: dict[str, dict[str, tuple[Path, str]]] = {}
    ordinary: list[tuple[Path, str]] = []
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                suffix = key.rsplit(".", 1)[-1]
                if suffix in {"qweight", "qzeros", "scales"}:
                    quantized.setdefault(key.rsplit(".", 1)[0], {})[suffix] = (path, key)
                else:
                    ordinary.append((path, key))

    device = torch.device("cuda")
    for name, parts in quantized.items():
        if set(parts) != {"qweight", "qzeros", "scales"}:
            raise ValueError(f"incomplete AWQ projection {name}")
        scale_path, scale_key = parts["scales"]
        with safe_open(scale_path, framework="pt", device="cpu") as handle:
            groups, out_features = handle.get_slice(scale_key).get_shape()
        in_features = groups * 128
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, AWQTinyGemmLinear(
            in_features, out_features, group_size=128, device=device,
        ))

    for path, key in ordinary:
        with safe_open(path, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(key)
        set_module_tensor_to_device(model, key, device, value=value, dtype=torch.float16)

    for name, parts in quantized.items():
        tensors = {}
        for part, (path, key) in parts.items():
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensors[part] = handle.get_tensor(key).to(device)
        packed, qparams = repack_awq_to_tinygemm(
            tensors["qweight"], tensors["qzeros"], tensors["scales"],
        )
        module = model.get_submodule(name)
        module.packed_weight.copy_(packed)
        module.scale_and_zero.copy_(qparams)
        del tensors, packed, qparams

    for name, buffer in list(model.named_buffers()):
        if buffer.device.type != "cuda":
            set_module_tensor_to_device(model, name, device, value=buffer)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.config._weight_only_int4_backend = "tinygemm"
    from engine.quantization.awq_tinygemm import install_int4pack_prefill
    install_int4pack_prefill(model, triton_prefill_rows, dequant_rows)
    missing_meta = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    if missing_meta:
        raise RuntimeError(f"unloaded model parameters: {missing_meta[:5]}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LoadedModel(
        model_name=str(root), requested_revision=None, resolved_revision=None,
        model=model, tokenizer=tokenizer, dtype=torch.float16, device=device,
    )

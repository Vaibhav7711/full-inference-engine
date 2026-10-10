"""W4A16 weights installed on the model, and a load path that never materialises fp16.

Qwen3-8B is 16.4 GB of fp16 weights on a card with 15 GB; a decode step sweeps 15.1 GB of
them, 58.6 ms at the T4's 258 GB/s. The standalone kernels in `engine.kernels.w4a16_linear`
cannot change that until they are *in the model*. This module puts them there, two ways:

* `install_w4a16(model)` - replace every attention projection, feed-forward projection
  and the output head of an already-loaded model with a `W4A16Linear`, quantising each
  weight in place and freeing the fp16 copy. Peak memory is the fp16 model plus one
  packed layer, so this is for models that already fit.
* `load_w4a16_model(name)` - build the model on the `meta` device, swap the shells in,
  then stream the safetensors shards one tensor at a time: Linear weights are quantised
  on the GPU as they arrive, everything else moves over in fp16. The fp16 model is never
  assembled anywhere. This is the only way an 8B loads on a Kaggle box.

`reference_mode(model)` makes every `W4A16Linear` run `F.linear` over its dequantised
weights instead of the Triton kernel. That is the identity gate's reference for a
quantised model: same packed weights, same arithmetic contract, torch's matmul - so the
gate tests the kernel and not the quantisation, and a 16 GB fp16 reference is never
needed. Quantisation error is a separate question, measured against the fp16 model's
perplexity, and it is not this module's job to hide it.

Discovery is structural, as in `engine.model.adapters`: a module is a target because of
its attribute name inside a Llama-style block (`q_proj` ... `down_proj`) or because the
model reports it as its output embedding, never because of a class name.
"""

from __future__ import annotations

import contextlib
import glob
import os
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from engine.kernels.w4a16_format import (
    GROUP_SIZE, quantize_weight_w4_grouped, unpack_w4_reference, weight_bytes,
)

ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_TARGETS = ("attention", "mlp", "lm_head")

_KERNEL: bool | None = None


def kernel_available() -> bool:
    """Whether the Triton kernel can be imported here; decided once."""
    global _KERNEL
    if _KERNEL is None:
        try:
            from engine.kernels import w4a16_linear  # noqa: F401
            _KERNEL = True
        except Exception:
            _KERNEL = False
    return _KERNEL


class W4A16Linear(nn.Module):
    """`nn.Linear` with 4-bit grouped weights, dequantised inside the matmul.

    Holds `packed` (uint8, `[out, in // 2]`), `scales` (fp16, `[out, in // group]`) and an
    optional fp16 `bias`. On CUDA with Triton it runs the fused dequant-GEMV; elsewhere, or
    under `reference_mode`, it dequantises with the Triton-free reference and calls
    `F.linear`, which is the arithmetic the kernel is tested against.
    """

    def __init__(self, in_features: int, out_features: int, *, group_size: int = GROUP_SIZE,
                 bias: bool = False, device=None):
        super().__init__()
        if in_features % group_size:
            raise ValueError(f"in_features {in_features} is not a multiple of group_size {group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.reference = False
        # Rows at or above this go through cuBLAS over a dequantised fp16 tile instead of
        # the Triton kernel. On sm_75 `tl.dot` lowers to scalar FMA (the tiled-prefill PTX
        # dump: mma_sync = 0), so the kernel is bandwidth-bound only while M is small: at
        # M = 16 the 8B's FFN is ~220 GFLOP per step, ~27 ms on FMA against a 15 ms
        # weight-read floor, and at prefill widths it is hopeless. Above the threshold the
        # weight is dequantised once per call (100 MB transient for the 8B's down_proj)
        # and the tensor cores do the matmul; the bytes saved on the read are the same.
        self.dense_threshold = 32
        factory = {"device": device}
        self.register_buffer("packed", torch.empty(out_features, in_features // 2, dtype=torch.uint8, **factory))
        self.register_buffer("scales", torch.empty(out_features, in_features // group_size,
                                                   dtype=torch.float16, **factory))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float16, **factory))
        else:
            self.bias = None

    @property
    def loaded(self) -> bool:
        return self.packed.device.type != "meta"

    def quantize_from(self, weight: torch.Tensor) -> None:
        """Replace this layer's weights with the quantisation of `weight` (`[out, in]`)."""
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"weight {tuple(weight.shape)} does not match "
                             f"[{self.out_features}, {self.in_features}]")
        packed, scales = quantize_weight_w4_grouped(weight.detach(), self.group_size)
        self.packed = packed.to(weight.device)
        self.scales = scales.to(weight.device)

    def set_bias(self, bias: torch.Tensor | None) -> None:
        self.bias = None if bias is None else bias.detach().to(torch.float16)

    def dequantized(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return unpack_w4_reference(self.packed, self.scales, self.in_features, self.group_size).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            raise RuntimeError("W4A16Linear has no weights: load or quantize_from first")
        rows = x.reshape(-1, self.in_features)
        use_kernel = (rows.device.type == "cuda" and not self.reference and kernel_available()
                      and rows.shape[0] < self.dense_threshold)
        if use_kernel:
            from engine.kernels.w4a16_linear import w4a16_linear

            out = w4a16_linear(rows.to(torch.float16).contiguous(), self.packed, self.scales,
                               self.bias, group_size=self.group_size)
            out = out.to(x.dtype)
        else:
            out = F.linear(rows, self.dequantized(rows.dtype),
                           None if self.bias is None else self.bias.to(rows.dtype))
        return out.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, group={self.group_size}, "
                f"bias={self.bias is not None}, kernel={'reference' if self.reference else 'triton'}")


@dataclass
class W4A16Report:
    replaced: int = 0
    by_kind: dict = field(default_factory=dict)
    fp16_bytes: int = 0
    packed_bytes: int = 0
    skipped: list = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.fp16_bytes / self.packed_bytes if self.packed_bytes else 0.0

    def as_dict(self) -> dict:
        return {"replaced": self.replaced, "by_kind": dict(self.by_kind),
                "fp16_mib": round(self.fp16_bytes / 2**20), "packed_mib": round(self.packed_bytes / 2**20),
                "ratio": round(self.ratio, 2), "skipped": list(self.skipped)}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _targets(model, targets) -> list[tuple[str, str, nn.Module, str]]:
    """`(full_name, kind, module, attribute)` for every Linear this installer owns."""
    wanted = set(targets)
    head = model.get_output_embeddings() if "lm_head" in wanted else None
    found = []
    for name, module in model.named_modules():
        for attribute in ATTENTION_PROJECTIONS if "attention" in wanted else ():
            child = getattr(module, attribute, None)
            if isinstance(child, nn.Linear):
                found.append((f"{name}.{attribute}" if name else attribute, "attention", child, attribute))
        for attribute in MLP_PROJECTIONS if "mlp" in wanted else ():
            child = getattr(module, attribute, None)
            if isinstance(child, nn.Linear):
                found.append((f"{name}.{attribute}" if name else attribute, "mlp", child, attribute))
    if head is not None and isinstance(head, nn.Linear):
        for name, module in model.named_modules():
            if module is head:
                found.append((name, "lm_head", module, name.rsplit(".", 1)[-1]))
                break
    seen, unique = set(), []
    for item in found:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    return unique


def _parent(model, full_name: str) -> tuple[nn.Module, str]:
    if "." not in full_name:
        return model, full_name
    parent_name, attribute = full_name.rsplit(".", 1)
    return model.get_submodule(parent_name), attribute


def _shell_for(linear: nn.Linear, group_size: int, device) -> W4A16Linear:
    return W4A16Linear(linear.in_features, linear.out_features, group_size=group_size,
                       bias=linear.bias is not None, device=device)


# ---------------------------------------------------------------------------
# Install onto a loaded model
# ---------------------------------------------------------------------------

@torch.no_grad()
def install_w4a16(model: nn.Module, *, group_size: int = GROUP_SIZE,
                  targets=DEFAULT_TARGETS) -> W4A16Report:
    """Quantise every target Linear in place, freeing each fp16 weight as it goes."""
    report = W4A16Report()
    for full_name, kind, linear, _attribute in _targets(model, targets):
        if linear.in_features % group_size:
            report.skipped.append(f"{full_name}: in_features {linear.in_features} % {group_size}")
            continue
        shell = _shell_for(linear, group_size, linear.weight.device)
        shell.quantize_from(linear.weight.data)
        shell.set_bias(None if linear.bias is None else linear.bias.data)
        parent, attribute = _parent(model, full_name)
        setattr(parent, attribute, shell)
        bytes_ = weight_bytes(linear.in_features, linear.out_features, group_size)
        report.replaced += 1
        report.by_kind[kind] = report.by_kind.get(kind, 0) + 1
        report.fp16_bytes += bytes_["fp16"]
        report.packed_bytes += bytes_["w4a16"]
        del linear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------
# Load without ever assembling the fp16 model
# ---------------------------------------------------------------------------

def _assign(model: nn.Module, key: str, tensor: torch.Tensor) -> bool:
    """Put `tensor` where `key` names a parameter or buffer, replacing a meta placeholder."""
    module_name, _, attribute = key.rpartition(".")
    try:
        module = model.get_submodule(module_name) if module_name else model
    except AttributeError:
        return False
    if attribute in module._parameters:
        module._parameters[attribute] = nn.Parameter(tensor, requires_grad=False)
        return True
    if attribute in module._buffers:
        module._buffers[attribute] = tensor
        return True
    return False


def _reinit_rotary(model: nn.Module, config, device) -> int:
    """Rotary tables are computed in `__init__`, so a meta-built model has meta tables."""
    rebuilt = 0
    for name, module in list(model.named_modules()):
        if not hasattr(module, "inv_freq"):
            continue
        parent, attribute = _parent(model, name)
        setattr(parent, attribute, type(module)(config, device=device))
        rebuilt += 1
    return rebuilt


def _meta_tensors(model: nn.Module) -> list[str]:
    names = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    names += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    return names


@torch.no_grad()
def load_w4a16_model(model_name: str, *, device, revision: str | None = None,
                     group_size: int = GROUP_SIZE, targets=DEFAULT_TARGETS,
                     dtype: torch.dtype = torch.float16) -> tuple[nn.Module, W4A16Report]:
    """Stream a checkpoint into a W4A16 model. `model_name` is a Hub id or a local folder."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from engine.model.adapters import ensure_supported

    device = torch.device(device)
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    ensure_supported(config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.eval()

    report = W4A16Report()
    shells: dict[str, W4A16Linear] = {}
    for full_name, kind, linear, _attribute in _targets(model, targets):
        if linear.in_features % group_size:
            report.skipped.append(f"{full_name}: in_features {linear.in_features} % {group_size}")
            continue
        shell = _shell_for(linear, group_size, "meta")
        parent, attribute = _parent(model, full_name)
        setattr(parent, attribute, shell)
        shells[full_name] = shell
        bytes_ = weight_bytes(linear.in_features, linear.out_features, group_size)
        report.replaced += 1
        report.by_kind[kind] = report.by_kind.get(kind, 0) + 1
        report.fp16_bytes += bytes_["fp16"]
        report.packed_bytes += bytes_["w4a16"]

    folder = model_name if os.path.isdir(model_name) else _download(model_name, revision)
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors shards under {folder}")
    from safetensors import safe_open

    loaded, unexpected = set(), []
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                tensor = shard.get_tensor(key)
                owner, _, leaf = key.rpartition(".")
                shell = shells.get(owner)
                if shell is not None and leaf == "weight":
                    shell.quantize_from(tensor.to(device=device, dtype=torch.float16))
                elif shell is not None and leaf == "bias":
                    shell.set_bias(tensor.to(device=device, dtype=torch.float16))
                elif not _assign(model, key, tensor.to(device=device, dtype=dtype)):
                    unexpected.append(key)
                    continue
                loaded.add(key)
                del tensor
    if unexpected:
        raise ValueError(f"checkpoint tensors with no home in the model: {unexpected[:8]}"
                         + (" ..." if len(unexpected) > 8 else ""))

    # A tied head ships no `lm_head.weight`; quantise it from the embedding table so the
    # output projection still reads 4-bit weights.
    head = model.get_output_embeddings()
    if isinstance(head, W4A16Linear) and not head.loaded:
        if getattr(config, "tie_word_embeddings", False):
            head.quantize_from(model.get_input_embeddings().weight.data.to(torch.float16))
        else:
            raise ValueError("lm_head.weight missing from the checkpoint and the head is untied")
    elif isinstance(head, nn.Linear) and head.weight.device.type == "meta" \
            and getattr(config, "tie_word_embeddings", False):
        head.weight = model.get_input_embeddings().weight

    _reinit_rotary(model, config, device)
    missing = _meta_tensors(model)
    if missing:
        raise ValueError(f"tensors never loaded (still on meta): {missing[:8]}"
                         + (" ..." if len(missing) > 8 else ""))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, report


def _download(model_name: str, revision: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, revision=revision,
                             allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])


# ---------------------------------------------------------------------------
# Reference mode and reporting
# ---------------------------------------------------------------------------

def w4a16_modules(model: nn.Module) -> list[W4A16Linear]:
    return [m for m in model.modules() if isinstance(m, W4A16Linear)]


@contextlib.contextmanager
def reference_mode(model: nn.Module):
    """Run every W4A16Linear through torch over its dequantised weights for the duration."""
    modules = w4a16_modules(model)
    previous = [m.reference for m in modules]
    for m in modules:
        m.reference = True
    try:
        yield
    finally:
        for m, was in zip(modules, previous):
            m.reference = was


def describe(model: nn.Module) -> dict:
    modules = w4a16_modules(model)
    packed = sum(m.packed.numel() + m.scales.numel() * 2 for m in modules)
    fp16 = sum(2 * m.in_features * m.out_features for m in modules)
    return {"w4a16_modules": len(modules), "packed_mib": round(packed / 2**20),
            "fp16_equivalent_mib": round(fp16 / 2**20),
            "group_size": modules[0].group_size if modules else None,
            "kernel": "triton" if kernel_available() else "reference (no Triton)"}

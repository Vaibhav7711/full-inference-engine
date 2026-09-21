"""Static checks over every Triton kernel, runnable without a GPU.

These exist because the failure classes they cover were all shipped at least once and all
of them are detectable by reading the source. A kernel that cannot be compiled in this
environment can still be checked against the language's rules and against arithmetic.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

KERNEL_DIR = pathlib.Path(__file__).resolve().parents[2] / "engine" / "kernels"
KERNEL_FILES = sorted(KERNEL_DIR.glob("*.py"))


def _jit_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            "jit" in ast.dump(decorator) for decorator in node.decorator_list
        ):
            yield node


def _bound_names(function: ast.FunctionDef) -> set[str]:
    names = {a.arg for a in function.args.args} | {a.arg for a in function.args.kwonlyargs}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.comprehension,)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _module_assignments(tree: ast.AST) -> tuple[set[str], set[str]]:
    plain, constexpr = set(), set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if isinstance(node.value, ast.Call) and "constexpr" in ast.dump(node.value.func):
            constexpr |= targets
        else:
            plain |= targets
    return plain, constexpr


@pytest.mark.parametrize("path", KERNEL_FILES, ids=lambda p: p.name)
def test_no_plain_module_global_is_read_inside_a_jit_function(path):
    """Triton can only read module globals instantiated as tl.constexpr.

    Shipped once: a masked-score sentinel declared as a module constant, which failed
    every test in its file with one compilation error.
    """
    tree = ast.parse(path.read_text())
    plain, constexpr = _module_assignments(tree)
    offenders = []
    for function in _jit_functions(tree):
        local = _bound_names(function)
        for node in ast.walk(function):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in plain and node.id not in local
                    and node.id not in constexpr):
                offenders.append(f"{node.id} at line {node.lineno} in {function.name}")
    assert not offenders, (
        f"{path.name}: module globals read inside @triton.jit: {offenders}. "
        f"Declare them tl.constexpr(...) or as kernel locals."
    )


@pytest.mark.parametrize("path", KERNEL_FILES, ids=lambda p: p.name)
def test_no_rank_three_broadcast_intermediate(path):
    """A [M, N, D] intermediate is megabytes per program; the operation is a dot.

    Shipped once: an fp32 PV accumulation written as
    `tl.sum(probs[:, :, None] * values[None, :, :], axis=1)`, which is 2 MB per program
    at 64x64x128.

    Confirmed a second time by `paged_decode_gqa.py`: its first version broadcast over
    the GQA group, `[2, BLOCK_N, D]`, small enough in bytes and still 1.5-2.8x slower
    than the rank-2 per-head kernel on the T4. The layout is the cost, not the size.
    """
    tree = ast.parse(path.read_text())
    offenders = []
    for function in _jit_functions(tree):
        for node in ast.walk(function):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                continue
            for side in (node.left, node.right):
                if (isinstance(side, ast.Subscript)
                        and isinstance(side.slice, ast.Tuple)
                        and len(side.slice.elts) >= 3):
                    offenders.append(f"line {node.lineno} in {function.name}")
    assert not offenders, (
        f"{path.name}: rank-3 broadcast inside @triton.jit at {offenders}. Use tl.dot."
    )


def _is_row_tiled(node: ast.AST) -> bool:
    """True when an expression indexes a tile of rows, i.e. contains a `[:, None]`.

    A store indexed only by program_id scalars is safe unmasked, because the grid covers
    the tensor exactly - that is how the per-token kernels are written and they are
    correct. The hazard is specific to tiled stores, where the grid is a `cdiv` and the
    last tile overhangs the tensor.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Tuple):
            elements = sub.slice.elts
            if any(isinstance(e, ast.Slice) for e in elements) and any(
                isinstance(e, ast.Constant) and e.value is None for e in elements
            ):
                return True
    return False


@pytest.mark.parametrize("path", KERNEL_FILES, ids=lambda p: p.name)
def test_tiled_stores_are_masked(path):
    """A tiled store must be masked by the *tensor* bound.

    Both halves of this were shipped. Masking by the logical bound (which rows carry a
    real token) left padded rows holding whatever the output allocation contained;
    dropping the mask entirely would write past the end whenever the query length is not
    a multiple of the tile. They are different numbers.

    Only tiled stores are checked: a store indexed by program_id scalars alone is safe
    unmasked, since such kernels launch a grid that covers the tensor exactly.
    """
    tree = ast.parse(path.read_text())
    unmasked = []
    for function in _jit_functions(tree):
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "store"):
                continue
            if not node.args or not _is_row_tiled(node.args[0]):
                continue
            if not any(kw.arg == "mask" for kw in node.keywords):
                unmasked.append(f"line {node.lineno} in {function.name}")
    assert not unmasked, (
        f"{path.name}: unmasked tiled tl.store at {unmasked}. Mask by the tensor bound "
        f"(offs < query_len), not the logical one (offs < chunk_len)."
    )


def test_the_checks_cover_every_kernel_file():
    assert KERNEL_FILES, "no kernel files found; the glob is wrong"
    with_jit = [p for p in KERNEL_FILES if list(_jit_functions(ast.parse(p.read_text())))]
    assert len(with_jit) >= 8, f"expected most kernel files to define kernels, got {with_jit}"

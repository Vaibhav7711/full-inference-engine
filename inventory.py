"""Repo architecture inventory + redundancy audit.

Run this at the repo root to get GROUND TRUTH on your actual project:
    python inventory.py

It prints:
  1. Every Python file, grouped by directory, with line counts.
  2. What each engine module imports (to see the dependency graph).
  3. Which modules are imported BY others (load-bearing) vs never imported (leaf/maybe-dead).
  4. Test-to-code mapping (which modules have tests).
  5. Redundancy flags: files that look superseded.

Use the output to correct the architecture doc to match reality, and to decide what to
archive/remove.
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict


REPO = "."
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules", "results"}


def find_py_files():
    files = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            if n.endswith(".py"):
                files.append(os.path.join(root, n))
    return sorted(files)


def module_name(path):
    """Convert a path to a dotted module name (engine/cache/foo.py -> engine.cache.foo)."""
    rel = os.path.relpath(path, REPO).replace(os.sep, ".")
    return rel[:-3] if rel.endswith(".py") else rel


def analyze_imports(path):
    """Return (internal_imports, defined_classes, defined_functions, line_count)."""
    try:
        src = open(path, encoding="utf-8").read()
    except Exception:
        return set(), [], [], 0
    lines = len(src.splitlines())
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), [], [], lines

    internal = set()
    classes, funcs = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("engine") or mod.startswith("benchmarks") or node.level > 0:
                internal.add(mod)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("engine") or alias.name.startswith("benchmarks"):
                    internal.add(alias.name)
        elif isinstance(node, ast.ClassDef) and node.col_offset == 0:
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            funcs.append(node.name)
    return internal, classes, funcs, lines


def main():
    files = find_py_files()

    # Collect info
    info = {}
    for f in files:
        internal, classes, funcs, lines = analyze_imports(f)
        info[f] = {
            "module": module_name(f),
            "imports": internal,
            "classes": classes,
            "funcs": funcs,
            "lines": lines,
        }

    # Build reverse import graph: who imports each module
    imported_by = defaultdict(set)
    all_modules = {info[f]["module"]: f for f in files}
    for f in files:
        for imp in info[f]["imports"]:
            # Match imported module against known modules (prefix match for packages)
            for mod, mf in all_modules.items():
                if imp == mod or mod.startswith(imp + ".") or imp.startswith(mod):
                    if mf != f:
                        imported_by[mod].add(info[f]["module"])

    # === 1. File inventory by directory ===
    print("=" * 70)
    print("1. FILE INVENTORY (by directory)")
    print("=" * 70)
    by_dir = defaultdict(list)
    for f in files:
        d = os.path.dirname(f)
        by_dir[d].append(f)
    total_lines = 0
    for d in sorted(by_dir):
        print(f"\n{d}/")
        for f in sorted(by_dir[d]):
            base = os.path.basename(f)
            ln = info[f]["lines"]
            total_lines += ln
            cls = info[f]["classes"]
            cls_str = f"  [{', '.join(cls)}]" if cls else ""
            print(f"    {base:<38} {ln:>5} lines{cls_str}")
    print(f"\nTotal: {len(files)} files, {total_lines} lines")

    # === 2. Dependency graph (engine modules only) ===
    print("\n" + "=" * 70)
    print("2. ENGINE DEPENDENCY GRAPH (what each engine module imports)")
    print("=" * 70)
    for f in sorted(files):
        mod = info[f]["module"]
        if not mod.startswith("engine"):
            continue
        deps = sorted(d for d in info[f]["imports"] if d.startswith("engine"))
        if deps:
            print(f"\n{mod}")
            for d in deps:
                print(f"    -> {d}")

    # === 3. Load-bearing vs possibly-dead ===
    print("\n" + "=" * 70)
    print("3. LOAD-BEARING vs POSSIBLY-DEAD (engine modules)")
    print("=" * 70)
    print("\nImported by others (load-bearing):")
    for f in sorted(files):
        mod = info[f]["module"]
        if not mod.startswith("engine") or mod.endswith("__init__"):
            continue
        importers = imported_by.get(mod, set())
        # also count package-level imports
        pkg = mod.rsplit(".", 1)[0]
        if importers:
            print(f"    {mod:<45} <- {len(importers)} importer(s)")

    print("\nNOT imported by any engine/benchmark module (leaf or possibly dead):")
    for f in sorted(files):
        mod = info[f]["module"]
        if not mod.startswith("engine") or mod.endswith("__init__"):
            continue
        importers = imported_by.get(mod, set())
        if not importers:
            print(f"    {mod:<45} (check: used via __init__ re-export? or dead?)")

    # === 4. Test coverage mapping ===
    print("\n" + "=" * 70)
    print("4. TEST -> CODE MAPPING")
    print("=" * 70)
    test_files = [f for f in files if "test" in os.path.basename(f).lower()]
    for tf in sorted(test_files):
        deps = sorted(d for d in info[tf]["imports"] if d.startswith("engine"))
        print(f"\n{os.path.basename(tf)}")
        for d in deps:
            print(f"    tests -> {d}")

    # === 5. Redundancy flags ===
    print("\n" + "=" * 70)
    print("5. REDUNDANCY FLAGS (heuristic — verify manually)")
    print("=" * 70)
    flags = []
    # Files with 'paged' in name — you have several, check which are active
    paged = [info[f]["module"] for f in files
             if "paged" in os.path.basename(f).lower() and "test" not in os.path.basename(f).lower()]
    if len(paged) > 1:
        flags.append(f"Multiple 'paged' modules — confirm each has a distinct role:\n      "
                     + "\n      ".join(paged))
    # Multiple batching modules
    batching = [info[f]["module"] for f in files
                if os.sep + "batching" + os.sep in f and "test" not in os.path.basename(f).lower()]
    if len(batching) > 1:
        flags.append(f"Multiple batching modules — confirm active vs baseline:\n      "
                     + "\n      ".join(batching))
    # Scratch files at repo root
    root_scripts = [f for f in files if os.path.dirname(f) == "."
                    and os.path.basename(f) not in ("inventory.py",)]
    if root_scripts:
        flags.append("Scripts at repo root (scratch? move or remove):\n      "
                     + "\n      ".join(os.path.basename(f) for f in root_scripts))

    if flags:
        for i, fl in enumerate(flags, 1):
            print(f"\n  [{i}] {fl}")
    else:
        print("\n  No obvious redundancy flags.")

    print("\n" + "=" * 70)
    print("Use this output to correct ARCHITECTURE.md and decide what to archive.")
    print("=" * 70)


if __name__ == "__main__":
    main()

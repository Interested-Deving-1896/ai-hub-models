# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""``configure_files`` must list exactly as many files as ``configure()`` takes.

The list is what the error message tells users to pass, so a mismatch sends them
a command that fails the arity check. Read statically: importing these modules
pulls model dependencies and can trigger source-repo clone prompts.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]

# configure() is inherited from reid.py, so the arity is not in these files.
# reid.py's own configure() takes a single zip; both assets declare one file.
_ARITY_DEFINED_ELSEWHERE = {
    "datasets/reid/market1501.py",
    "datasets/reid/entire_id.py",
}

_EXPECTS_RE = re.compile(r"expects (\d+) file\(s\)")


def _modules_declaring_configure_files() -> list[str]:
    """Dataset modules only -- utils/ defines the parameter rather than using it."""
    out = []
    for root in ("datasets", "models"):
        for path in (PACKAGE_ROOT / root).rglob("*.py"):
            if "test" in path.relative_to(PACKAGE_ROOT).parts:
                continue
            if "configure_files=" in path.read_text(errors="ignore"):
                out.append(path.relative_to(PACKAGE_ROOT).as_posix())
    return sorted(out)


def _list_len(tree: ast.Module, node: ast.expr) -> int | None:
    """Length of *node* as a list literal, resolving a module-level constant."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    if isinstance(node, ast.Name):
        for stmt in tree.body:
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, (ast.List, ast.Tuple))
                and any(
                    isinstance(t, ast.Name) and t.id == node.id for t in stmt.targets
                )
            ):
                return len(stmt.value.elts)
    return None


def _declared_lengths(tree: ast.Module) -> list[int]:
    lengths = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "configure_files":
                length = _list_len(tree, kw.value)
                assert length is not None, "configure_files must be a list literal"
                lengths.append(length)
    return lengths


def test_every_private_dataset_module_declares_configure_files() -> None:
    """configure_files is required, so every asset module must appear here.

    Compares against the modules that build a private dataset asset at all, so
    a new dataset cannot slip in without an ordering.
    """
    declaring = set(_modules_declaring_configure_files())
    building = set()
    for root in ("datasets", "models"):
        for path in (PACKAGE_ROOT / root).rglob("*.py"):
            if "test" in path.relative_to(PACKAGE_ROOT).parts:
                continue
            if "CachedPrivateDatasetAsset(" in path.read_text(errors="ignore"):
                building.add(path.relative_to(PACKAGE_ROOT).as_posix())
    assert building, "no dataset modules found; the scan is broken"
    assert building == declaring, (
        "these modules build a private dataset asset without configure_files: "
        f"{sorted(building - declaring)}"
    )


@pytest.mark.parametrize("rel", _modules_declaring_configure_files())
def test_configure_files_length_matches_configure_arity(rel: str) -> None:
    path = PACKAGE_ROOT / rel
    text = path.read_text()
    lengths = _declared_lengths(ast.parse(text))
    assert lengths, f"{rel}: no configure_files found by the AST walk"

    # Every asset of one dataset shares the list, so all lengths must agree.
    assert len(set(lengths)) == 1, (
        f"{rel}: assets declare different configure_files lengths {lengths}. "
        "Whichever asset fails first prints the whole command, so they must match."
    )

    expected = {int(m) for m in _EXPECTS_RE.findall(text)}
    if rel in _ARITY_DEFINED_ELSEWHERE:
        assert not expected, f"{rel}: now defines its own arity; drop the exemption"
        assert lengths[0] == 1, f"{rel}: reid configure() takes one zip"
        return

    assert expected, (
        f"{rel}: no `expects N file(s)` guard found. Either keep configure()'s "
        "arity check in that wording, or add the file to _ARITY_DEFINED_ELSEWHERE."
    )
    assert expected == {lengths[0]}, (
        f"{rel}: configure_files lists {lengths[0]} file(s) but configure() "
        f"expects {sorted(expected)}."
    )

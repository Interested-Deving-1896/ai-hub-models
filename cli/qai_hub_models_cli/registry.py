# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Local alias registry for standalone model folders.

Maps a short user-chosen name to an absolute folder path so
``qai-hub-models export <name>`` / ``evaluate <name>`` can dispatch to a
folder outside ``qai_hub_models.models`` without retyping its full path.

Storage: ``$QAIHM_CLI_REGISTRY_PATH`` if set, else a JSON file under
``LOCAL_STORE_DEFAULT_PATH/cli/registry.json`` (shared with the heavy
``qai_hub_models`` package via ``qai_hub_models_cli.common``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from qai_hub_models_cli.common import (
    LOCAL_STORE_DEFAULT_PATH,
    is_heavy_package_installed,
)

REGISTRY_PATH_ENVVAR = "QAIHM_CLI_REGISTRY_PATH"

_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _default_registry_path() -> Path:
    return Path(LOCAL_STORE_DEFAULT_PATH) / "cli" / "registry.json"


def registry_path() -> Path:
    override = os.environ.get(REGISTRY_PATH_ENVVAR)
    return Path(override) if override else _default_registry_path()


def load_registry() -> dict[str, str]:
    """Read the registry, returning ``{alias: absolute_path}``. Empty on miss."""
    path = registry_path()
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(
            f"Registry at {path} is not a JSON object; delete it or fix by hand."
        )
    return data


def _save_registry(entries: dict[str, str]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def register_alias(name: str, folder: str, force: bool = False) -> Path:
    """Register ``name`` -> resolved ``folder``. Returns the absolute path stored.

    Raises ``ValueError`` if ``name`` is invalid, collides with a built-in
    model id, or is already registered without ``force``. Raises
    ``FileNotFoundError`` if ``folder`` doesn't exist or isn't a directory.
    """
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid name {name!r}: must match {_NAME_RE.pattern} "
            "(lowercase letters, digits, and underscores only)."
        )

    if is_heavy_package_installed():
        from qai_hub_models.utils.path_helpers import MODEL_IDS

        if name in MODEL_IDS:
            raise ValueError(
                f"{name!r} is a built-in model id; pick a different alias."
            )

    resolved = Path(folder).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Not a directory: {resolved}")

    entries = load_registry()
    if name in entries and not force:
        raise ValueError(
            f"{name!r} is already registered to {entries[name]}. "
            "Use --force to overwrite."
        )
    entries[name] = str(resolved)
    _save_registry(entries)
    return resolved


def unregister_alias(name: str) -> str | None:
    """Remove ``name`` from the registry. Returns the path it was mapped to, or ``None`` if unregistered."""
    entries = load_registry()
    if name not in entries:
        return None
    path = entries.pop(name)
    _save_registry(entries)
    return path


def resolve_alias(name: str) -> Path | None:
    """Return the registered folder for ``name``, or ``None`` if unregistered."""
    entry = load_registry().get(name)
    return Path(entry) if entry else None

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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

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


_LOCK_TIMEOUT_SECONDS = 10.0


@contextmanager
def _registry_lock(timeout: float = _LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """Hold an exclusive lock across one registry read-modify-write.

    Without it, two concurrent registers each read the file, each add their own
    alias, and whichever writes last silently drops the other -- ``os.replace``
    makes the write atomic but does nothing about a lost update.

    Parameters
    ----------
    timeout
        Seconds to wait for the lock before giving up.

    Yields
    ------
    None
        With the lock held.

    Raises
    ------
    TimeoutError
        If the lock is still held after *timeout*.
    """
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(f"{path}.lock", timeout=timeout)
    try:
        lock.acquire()
    except Timeout as e:
        raise TimeoutError(
            f"Timed out after {timeout:g}s waiting for {path}.lock. Another "
            "`qai-hub-models register` or `unregister` is still running; wait "
            "for it to finish and try again."
        ) from e
    try:
        yield
    finally:
        lock.release()


def _assert_name_free(name: str, entries: dict[str, str], force: bool) -> None:
    """Raise if *name* is already registered and *force* was not passed.

    Parameters
    ----------
    name
        Alias being registered.
    entries
        The registry as just read.
    force
        Whether the caller opted into overwriting.

    Raises
    ------
    ValueError
        If *name* is taken and *force* is False.
    """
    if name in entries and not force:
        raise ValueError(
            f"{name!r} is already registered to {entries[name]}. "
            "Use --force to overwrite."
        )


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
    ``FileNotFoundError`` if ``folder`` doesn't exist or isn't a directory,
    and ``TimeoutError`` if another process holds the registry lock.
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

    with _registry_lock():
        entries = load_registry()
        _assert_name_free(name, entries, force)
        entries[name] = str(resolved)
        _save_registry(entries)
    return resolved


def unregister_alias(name: str) -> str | None:
    """Remove ``name`` from the registry. Returns the path it was mapped to, or ``None`` if unregistered."""
    with _registry_lock():
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

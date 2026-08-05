# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import sys
import types
from pathlib import Path

import pytest

from qai_hub_models_cli import registry


@pytest.fixture(autouse=True)
def _tmp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the registry at a per-test temp file so nothing touches user state."""
    monkeypatch.setenv(registry.REGISTRY_PATH_ENVVAR, str(tmp_path / "registry.json"))


@pytest.fixture
def no_heavy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the heavy qai_hub_models package isn't installed.

    The heavy import is blocked in conftest, so we short-circuit the
    ``MODEL_IDS`` collision check.
    """
    monkeypatch.setattr(
        "qai_hub_models_cli.registry.is_heavy_package_installed", lambda: False
    )


@pytest.mark.usefixtures("no_heavy")
def test_register_and_resolve_roundtrip(tmp_path: Path) -> None:
    folder = tmp_path / "my_model"
    folder.mkdir()

    stored = registry.register_alias("my_alias", str(folder))
    assert stored == folder.resolve()

    assert registry.resolve_alias("my_alias") == folder.resolve()
    assert registry.load_registry() == {"my_alias": str(folder.resolve())}


@pytest.mark.usefixtures("no_heavy")
def test_register_rejects_invalid_name(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    with pytest.raises(ValueError, match="Invalid name"):
        registry.register_alias("Bad-Name", str(folder))


def test_register_rejects_builtin_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "m"
    folder.mkdir()

    monkeypatch.setattr(
        "qai_hub_models_cli.registry.is_heavy_package_installed", lambda: True
    )
    # The conftest blocks real imports of qai_hub_models, so pre-register a stub
    # module tree that satisfies the collision-check import path.
    fake_pkg = types.ModuleType("qai_hub_models")
    fake_utils = types.ModuleType("qai_hub_models.utils")
    fake_path_helpers = types.ModuleType("qai_hub_models.utils.path_helpers")
    fake_path_helpers.MODEL_IDS = frozenset({"mobilenet_v2"})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qai_hub_models", fake_pkg)
    monkeypatch.setitem(sys.modules, "qai_hub_models.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules, "qai_hub_models.utils.path_helpers", fake_path_helpers
    )

    with pytest.raises(ValueError, match="built-in model id"):
        registry.register_alias("mobilenet_v2", str(folder))


@pytest.mark.usefixtures("no_heavy")
def test_register_requires_force_on_overwrite(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    registry.register_alias("m", str(a))
    with pytest.raises(ValueError, match="already registered"):
        registry.register_alias("m", str(b))

    stored = registry.register_alias("m", str(b), force=True)
    assert stored == b.resolve()
    assert registry.resolve_alias("m") == b.resolve()


@pytest.mark.usefixtures("no_heavy")
def test_unregister_missing_returns_none(tmp_path: Path) -> None:
    assert registry.unregister_alias("never_registered") is None

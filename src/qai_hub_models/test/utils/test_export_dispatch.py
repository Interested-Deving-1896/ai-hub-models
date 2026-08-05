# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import Mock, patch

import pytest

from qai_hub_models.utils.export.dispatch import (
    ResolvedModel,
    resolve_model,
    select_pipeline,
)


@pytest.fixture
def fake_module(tmp_path: Path) -> Iterator[ModuleType]:
    """Install a fake importable module at tmp_path/test_model/__init__.py."""
    module_dir = tmp_path / "test_model"
    module_dir.mkdir()
    init_path = module_dir / "__init__.py"
    init_path.write_text("")

    module = ModuleType("test_model")
    module.__file__ = str(init_path)
    module.Model = Mock(__name__="Model")  # type: ignore[attr-defined]
    sys.modules["test_model"] = module

    manifest_instance = Mock()
    manifest_instance.name = "test_model"
    manifest_instance.can_use_quantize_job = False
    manifest_instance.supports_quantization = False
    with patch(
        "qai_hub_models.utils.export.dispatch.QAIHMModelManifest.from_yaml",
        return_value=manifest_instance,
    ):
        yield module
    del sys.modules["test_model"]


def test_resolve_model_reads_module_and_configs(fake_module: ModuleType) -> None:
    """resolve_model imports the module and reads configs from its directory."""
    assert fake_module.__file__ is not None
    result = resolve_model("test_model")

    assert result.model_id == "test_model"
    assert result.model_cls is fake_module.Model
    assert result.manifest is not None
    assert result.display_name == "test_model"
    assert result.source_dir == Path(fake_module.__file__).parent
    assert result.app_cls is None


def test_resolve_model_uses_manifest_name(fake_module: ModuleType) -> None:
    """resolve_model reads display name from manifest.yaml."""
    assert fake_module.__file__ is not None
    source_dir = Path(fake_module.__file__).parent

    manifest_instance = Mock()
    manifest_instance.name = "Custom Display Name"
    manifest_instance.can_use_quantize_job = False
    manifest_instance.supports_quantization = False
    with patch(
        "qai_hub_models.utils.export.dispatch.QAIHMModelManifest.from_yaml",
        return_value=manifest_instance,
    ) as mock_from_yaml:
        result = resolve_model("test_model")

    assert result.display_name == "Custom Display Name"
    mock_from_yaml.assert_called_once_with(source_dir / "manifest.yaml")


def test_resolve_model_loads_app_cls_when_present(fake_module: ModuleType) -> None:
    """resolve_model reads App class if the module exposes one."""
    app_cls = Mock(__name__="App")
    fake_module.App = app_cls  # type: ignore[attr-defined]

    assert resolve_model("test_model").app_cls is app_cls


def test_resolve_model_raises_when_no_model_class() -> None:
    """resolve_model raises AttributeError when Model isn't exported."""
    module = ModuleType("empty_module")
    module.__file__ = "/tmp/empty_module/__init__.py"
    sys.modules["empty_module"] = module
    try:
        with pytest.raises(AttributeError):
            resolve_model("empty_module")
    finally:
        del sys.modules["empty_module"]


def test_resolve_model_raises_when_import_fails() -> None:
    """resolve_model surfaces ImportError for non-importable module paths."""
    with pytest.raises(ImportError):
        resolve_model("does_not_exist_anywhere_xyz")


def test_resolve_model_treats_known_id_as_qaihm_models_subpath() -> None:
    """Known model ids resolve under qai_hub_models.models.<id>."""
    with (
        patch("qai_hub_models.utils.export.dispatch.MODEL_IDS", ["mobilenet_v2"]),
        patch(
            "qai_hub_models.utils.export.dispatch.importlib.import_module",
            side_effect=RuntimeError("stop-after-import"),
        ) as mock_import,
        pytest.raises(RuntimeError, match="stop-after-import"),
    ):
        resolve_model("mobilenet_v2")

    mock_import.assert_called_once_with("qai_hub_models.models.mobilenet_v2")


def test_resolve_model_passes_unknown_id_through_verbatim() -> None:
    """An id not in MODEL_IDS is imported as-is."""
    with (
        patch("qai_hub_models.utils.export.dispatch.MODEL_IDS", ["known_model"]),
        patch(
            "qai_hub_models.utils.export.dispatch.importlib.import_module",
            side_effect=RuntimeError("stop-after-import"),
        ) as mock_import,
        pytest.raises(RuntimeError, match="stop-after-import"),
    ):
        resolve_model("my_project.models.foo")

    mock_import.assert_called_once_with("my_project.models.foo")


def test_resolve_model_folder_path_inserts_parent_and_imports_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Path arg prepends its parent to sys.path and imports by basename."""
    monkeypatch.setattr(sys, "path", sys.path[:])
    folder = tmp_path / "standalone_model"
    folder.mkdir()

    with (
        patch(
            "qai_hub_models.utils.export.dispatch.importlib.import_module",
            side_effect=RuntimeError("stop-after-import"),
        ) as mock_import,
        pytest.raises(RuntimeError, match="stop-after-import"),
    ):
        resolve_model(folder)

    mock_import.assert_called_once_with("standalone_model")
    assert sys.path[0] == str(folder.parent)


def test_resolve_model_missing_folder_raises(tmp_path: Path) -> None:
    """A Path arg to a non-existent folder raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Model folder does not exist"):
        resolve_model(tmp_path / "not_there")


def test_select_pipeline_excludes_model_id_from_bindings() -> None:
    """select_pipeline leaves model_id unbound so positional callers don't collide."""

    class FakeModel:
        pass

    resolved = ResolvedModel(
        model_id="foo",
        model_cls=FakeModel,
        manifest=Mock(),
        display_name="Foo",
        source_dir=Path("/fake"),
    )
    bound = select_pipeline(resolved)
    bound_partial = cast(partial, bound)
    # model_id must stay unbound (positional callers pass it explicitly).
    assert "model_id" not in bound_partial.keywords


def test_resolved_model_dataclass_defaults() -> None:
    """ResolvedModel has an optional app_cls that defaults to None."""
    r = ResolvedModel(
        model_id="foo",
        model_cls=Mock(),
        manifest=Mock(),
        display_name="Foo",
        source_dir=Path("/fake"),
    )
    assert r.app_cls is None

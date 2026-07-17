# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
import qai_hub as hub

from qai_hub_models.scripts import compare_inference
from qai_hub_models.scripts.compare_inference import export, get_component_inference_job
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_model import WorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.dispatch import ResolvedModel
from qai_hub_models.utils.export.result import (
    CollectionExportResult,
    ComponentGroup,
    ExportResult,
    LegacyCollectionExportResult,
)


def _fake_inference_job() -> hub.InferenceJob:
    return mock.create_autospec(hub.InferenceJob, instance=True)


# -- Minimal model classes for exercising export()'s branching. Abstract
# methods are left unimplemented on purpose: export()'s validation raises
# before any instance is created, and the success-path tests mock
# from_pretrained, so these are never instantiated. --
class _FakeSingleModel(WorkbenchModel):
    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> Any: ...


class _FakeCollectionModel(CollectionModel):
    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> Any: ...


class _FakeMultiGraphModel(MultiGraphWorkbenchModel):
    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> Any: ...


class _FakeMultiGraphCollectionModel(MultiGraphCollectionModel):
    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> Any: ...


def _fake_device() -> hub.Device:
    return mock.create_autospec(hub.Device, instance=True)


def _patch_resolved(model_cls: type) -> Any:
    """Patch export()'s model resolution to return *model_cls*."""
    resolved = mock.create_autospec(ResolvedModel, instance=True)
    resolved.model_cls = model_cls
    return mock.patch.object(compare_inference, "load_model", return_value=resolved)


def test_collection_export_result_returns_component_job() -> None:
    """CollectionExportResult stores jobs in a per-component ComponentGroup."""
    job = _fake_inference_job()
    result = CollectionExportResult(
        inference_jobs=ComponentGroup(
            {"detector": job, "recognizer": _fake_inference_job()}
        )
    )
    assert get_component_inference_job(result, "detector") is job


def test_collection_export_result_no_inference_jobs_returns_none() -> None:
    """A skipped-inference export has inference_jobs=None -> None, not a crash."""
    result = CollectionExportResult(inference_jobs=None)
    assert get_component_inference_job(result, "detector") is None


def test_legacy_collection_export_result_returns_component_job() -> None:
    """LegacyCollectionExportResult nests an ExportResult per component."""
    job = _fake_inference_job()
    result = LegacyCollectionExportResult(
        components={"part1": ExportResult(inference_job=job)}
    )
    assert get_component_inference_job(result, "part1") is job


def test_legacy_collection_export_result_component_without_job_returns_none() -> None:
    result = LegacyCollectionExportResult(
        components={"part1": ExportResult(inference_job=None)}
    )
    assert get_component_inference_job(result, "part1") is None


def test_unexpected_result_type_raises_type_error() -> None:
    """A non-collection export result is a programming error, not a silent None.

    Multi-graph models are rejected upstream, so any result type other than the
    two per-component collection shapes reaching this helper is a bug.
    """
    result = ExportResult(inference_job=_fake_inference_job())
    with pytest.raises(TypeError, match="Unexpected collection export result"):
        get_component_inference_job(result, "part1")


@pytest.mark.parametrize(
    "model_cls", [_FakeMultiGraphModel, _FakeMultiGraphCollectionModel]
)
def test_export_rejects_multi_graph_model(model_cls: type) -> None:
    """Multi-graph models have no on-device inference path -> ValueError."""
    with (
        _patch_resolved(model_cls),
        pytest.raises(ValueError, match="multi-graph model"),
    ):
        export("some_model", device=_fake_device())


def test_export_collection_without_component_raises() -> None:
    with (
        _patch_resolved(_FakeCollectionModel),
        pytest.raises(ValueError, match=r"collection model\. Use --component"),
    ):
        export("some_model", device=_fake_device())


def test_export_non_collection_with_component_raises() -> None:
    with (
        _patch_resolved(_FakeSingleModel),
        pytest.raises(ValueError, match="not a collection model"),
    ):
        export("some_model", device=_fake_device(), component="detector")

# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import inspect
from functools import partial
from pathlib import Path
from typing import cast
from unittest.mock import patch

from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.evaluate.dispatch import select_evaluate_pipeline


def test_select_evaluate_pipeline_binds_source_dir(tmp_path: Path) -> None:
    """select_evaluate_pipeline binds source_dir and drops it from the signature."""

    class FakeModel:
        pass

    source_dir = tmp_path / "fake_recipe"
    with patch(
        "qai_hub_models.utils.evaluate.dispatch.resolve_model_cls",
        return_value=FakeModel,
    ):
        bound = select_evaluate_pipeline(source_dir)

    sig = inspect.signature(bound)
    assert "source_dir" not in sig.parameters


def test_select_evaluate_pipeline_picks_collection_for_collection_model(
    tmp_path: Path,
) -> None:
    """CollectionModel subclasses go through the collection evaluate pipeline."""

    class FakeCollectionModel(CollectionModel):
        pass

    source_dir = tmp_path / "fake_collection"
    with patch(
        "qai_hub_models.utils.evaluate.dispatch.resolve_model_cls",
        return_value=FakeCollectionModel,
    ):
        bound = select_evaluate_pipeline(source_dir)

    bound_partial = cast(partial, bound)
    assert (
        bound_partial.func.__module__
        == "qai_hub_models.utils.evaluate.collection_pipeline"
    )

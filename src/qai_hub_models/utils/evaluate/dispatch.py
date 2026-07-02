# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Dispatch a ``model_id`` to the ``evaluate_model`` function for its pipeline.

Lives in its own file so callers can import the dispatcher without
triggering top-level imports of every pipeline module.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from qai_hub_models.utils.base_collection_model import CollectionModel


def _resolve_model_cls(model_id: str) -> type:
    """Import the Model class from ``qai_hub_models.models.<model_id>``."""
    module = importlib.import_module(f"qai_hub_models.models.{model_id}")
    return module.Model


def resolve_evaluate_model(model_id: str) -> Callable[..., Any]:
    """Return the ``evaluate_model`` function for the pipeline matching this model."""
    model_cls = _resolve_model_cls(model_id)
    if issubclass(model_cls, CollectionModel):
        from .collection_pipeline import evaluate_model

        return evaluate_model
    from .pipeline import evaluate_model

    return evaluate_model

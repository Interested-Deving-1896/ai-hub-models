# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Resolve a recipe folder and select its export pipeline.

The canonical identity for a model recipe throughout the pipeline stack is
``source_dir: Path`` — the folder that contains ``manifest.yaml``. String
ids (``"mobilenet_v2"``) and CLI targets (``"./my_model/"``) are only
accepted at the top layer (:func:`resolve_recipe_dir`), which converts them
into a folder path before anything else runs.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    PrecompiledCollectionModel,
)
from qai_hub_models.utils.base_model import PrecompiledWorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.collection_pipeline import (
    export_model as collection_export,
)
from qai_hub_models.utils.export.context import resolve_model_cls
from qai_hub_models.utils.export.multi_graph_collection_pipeline import (
    export_model as multi_graph_collection_export,
)
from qai_hub_models.utils.export.multi_graph_pipeline import (
    export_model as multi_graph_export,
)
from qai_hub_models.utils.export.pipeline import export_model as single_export
from qai_hub_models.utils.export.precompiled_pipeline import (
    export_model as precompiled_export,
)


def select_pipeline(source_dir: Path) -> Callable[..., Any]:
    """Return the pipeline's ``export_model`` with ``source_dir`` pre-bound.

    The pipeline is chosen from the recipe's ``Model`` class:

    * :class:`PrecompiledWorkbenchModel` / :class:`PrecompiledCollectionModel`
      → precompiled pipeline
    * :class:`MultiGraphCollectionModel` → sharded-LLM pipeline
    * :class:`CollectionModel` → per-component pipeline
    * :class:`MultiGraphWorkbenchModel` → multi-graph single-model pipeline
    * anything else → single-graph pipeline
    """
    model_cls = resolve_model_cls(source_dir)
    if issubclass(model_cls, (PrecompiledWorkbenchModel, PrecompiledCollectionModel)):
        pipeline_fn: Callable[..., Any] = precompiled_export
    elif issubclass(model_cls, MultiGraphCollectionModel):
        pipeline_fn = multi_graph_collection_export
    elif issubclass(model_cls, CollectionModel):
        pipeline_fn = collection_export
    elif issubclass(model_cls, MultiGraphWorkbenchModel):
        pipeline_fn = multi_graph_export
    else:
        pipeline_fn = single_export

    bound = partial(pipeline_fn, source_dir=source_dir)
    sig = inspect.signature(pipeline_fn)
    bound.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=[p for n, p in sig.parameters.items() if n != "source_dir"]
    )
    bound.__doc__ = pipeline_fn.__doc__
    return bound

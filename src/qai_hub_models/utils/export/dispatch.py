# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Resolve a model and select its export pipeline."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
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
from qai_hub_models.utils.path_helpers import MODEL_IDS


@dataclass
class ResolvedModel:
    """Resolved model metadata for export/evaluate."""

    model_id: str
    model_cls: type
    manifest: QAIHMModelManifest
    display_name: str
    source_dir: Path
    app_cls: type | None = None
    supports_quant_cpu: bool = False


def resolve_model(model_id_or_module: str | Path) -> ResolvedModel:
    """Import the model's module and gather its metadata.

    Parameters
    ----------
    model_id_or_module
        A known model id (e.g. ``"mobilenet_v2"``, resolved under
        ``qai_hub_models.models``), any importable dotted module path, or
        a filesystem ``Path`` to a standalone model folder. When a folder
        path is passed, its parent is prepended to ``sys.path`` and the
        folder's basename is imported as a top-level module.

    Returns
    -------
    ResolvedModel
        The imported ``Model`` class, optional ``App`` class, ``source_dir``
        derived from the module's ``__file__``, and ``manifest.yaml`` read
        from that directory.
    """
    if isinstance(model_id_or_module, Path):
        folder = model_id_or_module
        if not folder.is_dir():
            raise FileNotFoundError(f"Model folder does not exist: {folder}")
        sys.path.insert(0, str(folder.parent))
        module_path = folder.name
    elif model_id_or_module in MODEL_IDS:
        module_path = f"qai_hub_models.models.{model_id_or_module}"
    else:
        module_path = model_id_or_module
    module = importlib.import_module(module_path)
    if module.__file__ is None:
        raise ValueError(
            f"Module {module_path!r} has no __file__ (namespace package?). "
            f"Point at a regular package with an __init__.py instead."
        )
    source_dir = Path(module.__file__).parent
    model_cls = module.Model
    manifest = QAIHMModelManifest.from_yaml(source_dir / "manifest.yaml")
    return ResolvedModel(
        model_id=source_dir.name,
        model_cls=model_cls,
        manifest=manifest,
        display_name=manifest.name or source_dir.name,
        source_dir=source_dir,
        app_cls=getattr(module, "App", None),
        supports_quant_cpu=(
            manifest.can_use_quantize_job and manifest.supports_quantization
        ),
    )


def select_pipeline(resolved: ResolvedModel) -> Callable[..., Any]:
    """Return the pipeline ``export_model`` for *resolved* with its context bound.

    The right ``export_model`` is chosen from ``resolved.model_cls``. Only the
    ``ResolvedModel`` fields that appear in that pipeline's signature are
    bound as kwargs — pipelines that don't need e.g. ``source_dir`` or
    ``app_cls`` don't have to declare placeholder params for them.

    Parameters
    ----------
    resolved
        Model metadata from :func:`resolve_model`.

    Returns
    -------
    Callable[..., Any]
        Callable equivalent to the selected pipeline's ``export_model`` with
        the applicable resolved fields pre-bound.
    """
    model_cls = resolved.model_cls
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

    sig = inspect.signature(pipeline_fn)
    # ``model_id`` is excluded so positional callers like
    # ``export_model(model_id, device=...)`` don't collide with a bound kwarg.
    bind_kwargs = {
        f.name: getattr(resolved, f.name)
        for f in fields(resolved)
        if f.name in sig.parameters and f.name != "model_id"
    }
    bound = partial(pipeline_fn, **bind_kwargs)
    # Give the partial a signature/docstring the CLI parser can introspect:
    #   * __signature__ drops the bound kwargs so inspect.signature(bound)
    #     surfaces only user-facing params.
    #   * __doc__ preserves FunctionDoc parameter descriptions.
    bound.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=[p for n, p in sig.parameters.items() if n not in bind_kwargs]
    )
    bound.__doc__ = pipeline_fn.__doc__
    return bound

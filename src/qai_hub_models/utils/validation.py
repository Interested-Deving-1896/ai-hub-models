# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterable, Mapping
from enum import Enum, unique
from typing import Any, NamedTuple

import numpy as np
import torch

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.protocols import FromPretrainedProtocol
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
    instantiate_dataset,
)
from qai_hub_models.utils.base_model import WorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
    MultiGraphWorkbenchModelCollection,
)
from qai_hub_models.utils.private_asset_loaders import UnfetchableDatasetError


def is_valid_dataset_class(dataset_cls: type) -> bool:
    return (
        isinstance(dataset_cls, type)
        and issubclass(dataset_cls, BaseDataset)
        and not inspect.isabstract(dataset_cls)
    )


def _quantized_precision_names(manifest: QAIHMModelManifest) -> list[str]:
    return [str(p) for p in manifest.supported_precisions if p != Precision.float]


def validate_io_names(instance: WorkbenchModel) -> list[str]:
    """
    Validate channel-last declarations match actual I/O names
    and that names don't contain dashes.

    Parameters
    ----------
    instance
        The model instance to validate.

    Returns
    -------
    list[str]
        Error messages for each failing check.
    """
    input_spec = instance.get_input_spec()
    output_names = list(instance.get_output_spec())

    errors: list[str] = []
    errors.extend(
        f"Input name '{name}' contains '-'. "
        "QNN converts dashes to underscores, causing name mismatches."
        for name in input_spec
        if "-" in name
    )
    errors.extend(
        f"Output name '{name}' contains '-'. "
        "QNN converts dashes to underscores, causing name mismatches."
        for name in output_names
        if "-" in name
    )
    return errors


def validate_io_names_collection(
    model: WorkbenchModelCollection | MultiGraphWorkbenchModelCollection,
) -> list[str]:
    """
    Run I/O name validation on each component of a collection model.

    Parameters
    ----------
    model
        The collection model to validate.

    Returns
    -------
    list[str]
        Error messages for each failing check, prefixed with the component name.
    """
    errors: list[str] = []
    for comp_name, component in model.components.items():
        if not isinstance(component, WorkbenchModel):
            continue
        errors.extend(
            f"[component '{comp_name}'] {err}" for err in validate_io_names(component)
        )
    return errors


def validate_eval_datasets(
    model: WorkbenchModel | CollectionModel | MultiGraphCollectionModel,
) -> list[str]:
    """
    Validate that all dataset classes returned by get_eval_dataset_classes() are valid.

    Parameters
    ----------
    model
        The model instance to validate.

    Returns
    -------
    list[str]
        Error messages for each invalid dataset class.
    """
    return [
        f"get_eval_dataset_classes() includes '{ds_cls.dataset_name()}', which is not "
        "a valid BaseDataset subclass."
        for ds_cls in model.get_eval_dataset_classes()
        if not is_valid_dataset_class(ds_cls)
    ]


def validate_eval_datasets_have_evaluator(
    model: WorkbenchModel,
) -> list[str]:
    """
    Validate that models with eval datasets implement get_evaluator().

    Parameters
    ----------
    model
        The model instance to validate.

    Returns
    -------
    list[str]
        Error messages if get_eval_dataset_classes() is non-empty but
        get_evaluator() is not overridden.
    """
    if not model.get_eval_dataset_classes():
        return []
    if model.get_evaluator is WorkbenchModel.get_evaluator:
        return [
            "get_eval_dataset_classes() is non-empty but get_evaluator() is not implemented."
        ]
    return []


def _litemp_implemented(model: WorkbenchModel, precision: Precision) -> bool:
    try:
        model.get_hub_litemp_percentage(precision)
    except NotImplementedError:
        return False
    return True


def validate_mixed_precision_litemp(
    model: WorkbenchModel,
    manifest: QAIHMModelManifest,
) -> list[str]:
    """
    Validate that models with mixed-precision support implement
    get_hub_litemp_percentage().

    Parameters
    ----------
    model
        The model instance to validate.
    manifest
        The model's manifest.yaml configuration.

    Returns
    -------
    list[str]
        Error messages for each mixed precision missing litemp support.
    """
    mixed_precisions = [
        p
        for p in manifest.supported_precisions
        if isinstance(p, Precision) and p.override_type is not None
    ]
    return [
        f"Precision {p} uses mixed precision (override_type) "
        "but get_hub_litemp_percentage() raises NotImplementedError."
        for p in mixed_precisions
        if not _litemp_implemented(model, p)
    ]


def _component_precision_implemented(component: WorkbenchModel) -> bool:
    try:
        component.component_precision()
    except NotImplementedError:
        return False
    return True


def validate_component_precision(
    model: WorkbenchModelCollection | MultiGraphWorkbenchModelCollection,
    manifest: QAIHMModelManifest,
) -> list[str]:
    """
    Validate that components implement component_precision() when the
    collection model declares mixed or mixed_with_float precision,
    and that components whose per-component precision uses mixed precision
    also implement get_hub_litemp_percentage().

    Parameters
    ----------
    model
        The collection model to validate.
    manifest
        The model's manifest.yaml configuration.

    Returns
    -------
    list[str]
        Error messages for each component missing component_precision()
        or litemp support.
    """
    has_mixed = any(
        p in [Precision.mixed, Precision.mixed_with_float]
        for p in manifest.supported_precisions
    )
    if not has_mixed:
        return []

    errors: list[str] = []
    for comp_name, component in model.components.items():
        if not isinstance(component, WorkbenchModel):
            continue
        if not _component_precision_implemented(component):
            errors.append(
                f"[component '{comp_name}'] Collection model declares mixed precision "
                "but component does not implement component_precision()."
            )
            continue
        comp_precision = component.component_precision()
        if (
            isinstance(comp_precision, Precision)
            and comp_precision.override_type is not None
            and not _litemp_implemented(component, comp_precision)
        ):
            errors.append(
                f"[component '{comp_name}'] Component precision {comp_precision} "
                "uses mixed precision (override_type) "
                "but get_hub_litemp_percentage() raises NotImplementedError."
            )
    return errors


@unique
class DatasetCheck(Enum):
    """The three per-dataset split checks."""

    REQUIRED_SPLIT = "required split loads"
    DISTINCT_SPLITS = "train/val splits are distinct"
    VAL_REPRODUCIBLE = "val split is reproducible"


@unique
class DatasetCheckOutcome(Enum):
    """Verdict for one ``DatasetCheck``.

    ``UNKNOWN`` means the dataset's items could not be compared (e.g. an
    opaque item type); ``UNAVAILABLE`` means the data could not be fetched
    in this environment. Neither is treated as a failure.
    """

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class DatasetCheckResult(NamedTuple):
    """One check's verdict for one dataset class."""

    dataset_name: str
    check: DatasetCheck
    outcome: DatasetCheckOutcome
    detail: str


class DatasetRoles(NamedTuple):
    """How a model uses a dataset class, which fixes the split it must support."""

    is_eval: bool
    is_calibration: bool

    @property
    def label(self) -> str:
        roles = ["eval"] if self.is_eval else []
        if self.is_calibration:
            roles.append("calibration")
        return "+".join(roles)

    @property
    def required_splits(self) -> tuple[DatasetSplit, ...]:
        """Eval datasets must serve VAL; calibration datasets must serve TRAIN."""
        splits = (DatasetSplit.VAL,) if self.is_eval else ()
        return (*splits, DatasetSplit.TRAIN) if self.is_calibration else splits


def collect_dataset_roles(model: Any) -> dict[type[BaseDataset], DatasetRoles]:
    """Map each of the model's dataset classes to the roles it plays.

    Collection models are walked component-by-component too, since each
    component declares its own calibration dataset. A class used for both
    eval and calibration gets both roles, so both splits are required.

    Parameters
    ----------
    model
        An instantiated model (or collection model).

    Returns
    -------
    dict[type[BaseDataset], DatasetRoles]
        Dataset classes in declaration order, mapped to their roles.
    """
    eval_classes: list[Any] = []
    calib_classes: list[Any] = []
    for holder in [model, *getattr(model, "components", {}).values()]:
        eval_getter = getattr(holder, "get_eval_dataset_classes", None)
        if callable(eval_getter):
            with contextlib.suppress(Exception):
                eval_classes.extend(eval_getter())
        calib_getter = getattr(holder, "get_calibration_dataset_cls", None)
        if callable(calib_getter):
            with contextlib.suppress(Exception):
                calib_classes.append(calib_getter())

    roles: dict[type[BaseDataset], DatasetRoles] = {}
    for candidates, is_eval in ((eval_classes, True), (calib_classes, False)):
        for candidate in candidates:
            if not is_valid_dataset_class(candidate):
                continue
            prior = roles.get(candidate, DatasetRoles(False, False))
            roles[candidate] = DatasetRoles(
                prior.is_eval or is_eval, prior.is_calibration or not is_eval
            )
    return roles


def _all_values_match(pairs: Iterable[tuple[Any, Any]]) -> bool | None:
    """Fold ``_values_match`` over pairs; False wins over None wins over True."""
    saw_unknown = False
    for a, b in pairs:
        match = _values_match(a, b)
        if match is False:
            return False
        if match is None:
            saw_unknown = True
    return None if saw_unknown else True


def _values_match(a: Any, b: Any) -> bool | None:
    """Deep-compare two dataset items. ``None`` means "not comparable"."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return bool(np.array_equal(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            return False
        return _all_values_match((a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return _all_values_match(zip(a, b, strict=True))
    if a is None and b is None:
        return True
    if isinstance(a, (bool, int, float, str, bytes)) and isinstance(
        b, (bool, int, float, str, bytes)
    ):
        return a == b
    return None


def _load_first_item(
    dataset_cls: type[BaseDataset], split: DatasetSplit
) -> tuple[Any, int | None]:
    dataset = instantiate_dataset(dataset_cls, split=split)
    try:
        length: int | None = len(dataset)
    except Exception:
        length = None
    return dataset[0], length


def _try_load_first_item(
    dataset_cls: type[BaseDataset], split: DatasetSplit
) -> tuple[tuple[Any, int | None] | None, Exception | None]:
    try:
        return _load_first_item(dataset_cls, split), None
    except Exception as exc:
        return None, exc


def _required_split_result(
    name: str, roles: DatasetRoles, errors: dict[DatasetSplit, Exception]
) -> DatasetCheckResult:
    required = ", ".join(s.name.lower() for s in roles.required_splits)
    failures = [(s, errors[s]) for s in roles.required_splits if s in errors]
    if any(isinstance(exc, UnfetchableDatasetError) for _, exc in failures):
        return DatasetCheckResult(
            name,
            DatasetCheck.REQUIRED_SPLIT,
            DatasetCheckOutcome.UNAVAILABLE,
            "dataset is not fetchable in this environment.",
        )
    if failures:
        detail = "; ".join(
            f"{split.name.lower()}: {exc.__class__.__name__}: {exc}"
            for split, exc in failures
        )
        return DatasetCheckResult(
            name,
            DatasetCheck.REQUIRED_SPLIT,
            DatasetCheckOutcome.FAIL,
            f"role={roles.label} requires {required}, but {detail}",
        )
    return DatasetCheckResult(
        name,
        DatasetCheck.REQUIRED_SPLIT,
        DatasetCheckOutcome.PASS,
        f"role={roles.label}; serves {required}",
    )


def _distinct_splits_result(
    name: str,
    loaded: dict[DatasetSplit, tuple[Any, int | None]],
    errors: dict[DatasetSplit, Exception],
) -> DatasetCheckResult:
    absent = [s for s in (DatasetSplit.TRAIN, DatasetSplit.VAL) if s in errors]
    if absent:
        names = ", ".join(s.name.lower() for s in absent)
        return DatasetCheckResult(
            name,
            DatasetCheck.DISTINCT_SPLITS,
            DatasetCheckOutcome.PASS,
            f"raises on {names} rather than aliasing it",
        )

    train_item, train_len = loaded[DatasetSplit.TRAIN]
    val_item, val_len = loaded[DatasetSplit.VAL]
    lengths = f"len(train)={train_len}, len(val)={val_len}"
    match = _values_match(train_item, val_item)
    if match is None:
        return DatasetCheckResult(
            name,
            DatasetCheck.DISTINCT_SPLITS,
            DatasetCheckOutcome.UNKNOWN,
            f"first items are not comparable "
            f"(type {type(train_item).__name__}); {lengths}",
        )
    if match:
        return DatasetCheckResult(
            name,
            DatasetCheck.DISTINCT_SPLITS,
            DatasetCheckOutcome.FAIL,
            f"TRAIN and VAL return an identical first item ({lengths}) — one "
            "split is silently aliasing the other. Serve distinct data, or "
            "raise on the split this dataset does not support.",
        )
    return DatasetCheckResult(
        name, DatasetCheck.DISTINCT_SPLITS, DatasetCheckOutcome.PASS, lengths
    )


def _val_reproducible_result(
    name: str, val_item: Any, val_again: Any
) -> DatasetCheckResult:
    repro = _values_match(val_item, val_again)
    if repro is None:
        return DatasetCheckResult(
            name,
            DatasetCheck.VAL_REPRODUCIBLE,
            DatasetCheckOutcome.UNKNOWN,
            f"first items are not comparable (type {type(val_item).__name__}).",
        )
    if repro:
        return DatasetCheckResult(
            name,
            DatasetCheck.VAL_REPRODUCIBLE,
            DatasetCheckOutcome.PASS,
            "two from-scratch val instances agree on item 0",
        )
    return DatasetCheckResult(
        name,
        DatasetCheck.VAL_REPRODUCIBLE,
        DatasetCheckOutcome.FAIL,
        "two from-scratch DatasetSplit.VAL instances return different first "
        "items — evaluation numerics will not be reproducible. Sort the sample "
        "list or seed any shuffling.",
    )


def _check_one_dataset(
    dataset_cls: type[BaseDataset], roles: DatasetRoles
) -> list[DatasetCheckResult]:
    name = dataset_cls.dataset_name()
    loaded: dict[DatasetSplit, tuple[Any, int | None]] = {}
    errors: dict[DatasetSplit, Exception] = {}
    for split in (DatasetSplit.VAL, DatasetSplit.TRAIN):
        item, exc = _try_load_first_item(dataset_cls, split)
        if exc is not None:
            errors[split] = exc
        elif item is not None:
            loaded[split] = item

    required = _required_split_result(name, roles, errors)
    results = [required]
    if required.outcome is not DatasetCheckOutcome.PASS:
        detail = "skipped: required split did not load."
        results.append(
            DatasetCheckResult(
                name, DatasetCheck.DISTINCT_SPLITS, required.outcome, detail
            )
        )
        if roles.is_eval:
            results.append(
                DatasetCheckResult(
                    name, DatasetCheck.VAL_REPRODUCIBLE, required.outcome, detail
                )
            )
        return results

    results.append(_distinct_splits_result(name, loaded, errors))
    if not roles.is_eval:
        return results

    val_again, exc = _try_load_first_item(dataset_cls, DatasetSplit.VAL)
    if exc is not None or val_again is None:
        results.append(
            DatasetCheckResult(
                name,
                DatasetCheck.VAL_REPRODUCIBLE,
                DatasetCheckOutcome.UNKNOWN,
                f"val split reload failed: {exc}",
            )
        )
        return results
    results.append(
        _val_reproducible_result(name, loaded[DatasetSplit.VAL][0], val_again[0])
    )
    return results


def validate_dataset_splits(model: Any) -> list[DatasetCheckResult]:
    """Check every eval / calibration dataset against the splits its role requires.

    Eval datasets must serve VAL, calibration datasets must serve TRAIN, and the
    other split must either raise or return different data — it may never
    silently alias the required one. Eval datasets are additionally instantiated
    twice to confirm VAL's first item is stable.

    Only item 0 of each split is compared, so this is cheap relative to a full
    pass, but it does download each dataset.

    Parameters
    ----------
    model
        An instantiated model (or collection model).

    Returns
    -------
    list[DatasetCheckResult]
        One entry per (dataset, check). Empty if the model declares no eval or
        calibration datasets.
    """
    results: list[DatasetCheckResult] = []
    for dataset_cls, roles in collect_dataset_roles(model).items():
        results.extend(_check_one_dataset(dataset_cls, roles))
    return results


def perform_dataset_split_validation(
    model_cls: type[WorkbenchModel | CollectionModel | MultiGraphCollectionModel],
) -> None:
    """Run ``validate_dataset_splits`` and raise on any failure.

    Datasets that cannot be fetched in this environment, and items that cannot
    be compared, are reported as skips rather than failures.

    Parameters
    ----------
    model_cls
        The model class to validate.

    Raises
    ------
    AssertionError
        If any dataset fails a split check.
    """
    assert issubclass(model_cls, FromPretrainedProtocol)
    results = validate_dataset_splits(model_cls.from_pretrained())
    failures = [r for r in results if r.outcome is DatasetCheckOutcome.FAIL]
    if failures:
        header = f"Dataset split validation failed with {len(failures)} error(s):"
        details = "\n".join(
            f"  - [{r.dataset_name}] {r.check.value}: {r.detail}" for r in failures
        )
        raise AssertionError(f"{header}\n{details}")


def perform_runtime_model_validation(
    model_cls: type[WorkbenchModel | CollectionModel | MultiGraphCollectionModel],
    model_id: str,
    app_cls: type | None = None,
    manifest: QAIHMModelManifest | None = None,
) -> None:
    """
    Run all static validation checks on a model's configuration.

    Raises AssertionError with all collected failures.

    Parameters
    ----------
    model_cls
        The model class to validate.
    model_id
        The model identifier used to load manifest.yaml.
    app_cls
        For collection models, the App class so calibration checks
        can verify CollectionAppQuantizeProtocol compliance. Passing ``None``
        is safe for models without quantized precisions; for models
        with quantized precisions, ``None`` will produce an error
        indicating the missing App.
    manifest
        Optional pre-loaded manifest. If None, loads via QAIHMModelManifest.from_model(model_id).

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    if manifest is None:
        manifest = QAIHMModelManifest.from_model(model_id)
    errors: list[str] = []

    assert issubclass(model_cls, FromPretrainedProtocol)
    model = model_cls.from_pretrained()
    if isinstance(
        model,
        (WorkbenchModelCollection, MultiGraphWorkbenchModelCollection),
    ):
        errors.extend(validate_io_names_collection(model))
        errors.extend(validate_component_precision(model, manifest))
    elif isinstance(model, WorkbenchModel):
        errors.extend(validate_io_names(model))
        errors.extend(validate_mixed_precision_litemp(model, manifest))
        errors.extend(validate_eval_datasets_have_evaluator(model))
    else:
        raise NotImplementedError()

    errors.extend(validate_eval_datasets(model))

    if errors:
        header = (
            f"Model validation failed for '{model_id}' with {len(errors)} error(s):"
        )
        details = "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(f"{header}\n{details}")

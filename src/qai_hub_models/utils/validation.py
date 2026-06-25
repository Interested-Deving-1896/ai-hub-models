# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import inspect

from qai_hub_models import Precision
from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.protocols import FromPretrainedProtocol
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    WorkbenchModelCollection,
)
from qai_hub_models.utils.base_model import WorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
    MultiGraphWorkbenchModelCollection,
)


def _is_valid_dataset_class(dataset_cls: type) -> bool:
    return (
        isinstance(dataset_cls, type)
        and issubclass(dataset_cls, BaseDataset)
        and not inspect.isabstract(dataset_cls)
    )


def _quantized_precision_names(code_gen: QAIHMModelCodeGen) -> list[str]:
    return [str(p) for p in code_gen.supported_precisions if p != Precision.float]


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
        if not _is_valid_dataset_class(ds_cls)
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
    code_gen: QAIHMModelCodeGen,
) -> list[str]:
    """
    Validate that models with mixed-precision support implement
    get_hub_litemp_percentage().

    Parameters
    ----------
    model
        The model instance to validate.
    code_gen
        The model's code-gen.yaml configuration.

    Returns
    -------
    list[str]
        Error messages for each mixed precision missing litemp support.
    """
    mixed_precisions = [
        p
        for p in code_gen.supported_precisions
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
    code_gen: QAIHMModelCodeGen,
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
    code_gen
        The model's code-gen.yaml configuration.

    Returns
    -------
    list[str]
        Error messages for each component missing component_precision()
        or litemp support.
    """
    has_mixed = any(
        p in [Precision.mixed, Precision.mixed_with_float]
        for p in code_gen.supported_precisions
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


def perform_runtime_model_validation(
    model_cls: type[WorkbenchModel | CollectionModel | MultiGraphCollectionModel],
    model_id: str,
    app_cls: type | None = None,
) -> None:
    """
    Run all static validation checks on a model's configuration.

    Raises AssertionError with all collected failures.

    Parameters
    ----------
    model_cls
        The model class to validate.
    model_id
        The model identifier used to load code-gen.yaml.
    app_cls
        For collection models, the App class so calibration checks
        can verify CollectionAppQuantizeProtocol compliance. Passing ``None``
        is safe for models without quantized precisions; for models
        with quantized precisions, ``None`` will produce an error
        indicating the missing App.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    code_gen = QAIHMModelCodeGen.from_model(model_id)
    errors: list[str] = []

    assert issubclass(model_cls, FromPretrainedProtocol)
    model = model_cls.from_pretrained()
    if isinstance(
        model,
        (WorkbenchModelCollection, MultiGraphWorkbenchModelCollection),
    ):
        errors.extend(validate_io_names_collection(model))
        errors.extend(validate_component_precision(model, code_gen))
    elif isinstance(model, WorkbenchModel):
        errors.extend(validate_io_names(model))
        errors.extend(validate_mixed_precision_litemp(model, code_gen))
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

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""End-to-end evaluate pipeline for a single (non-collection) model."""

from __future__ import annotations

import argparse
import importlib

import qai_hub as hub

from qai_hub_models import Precision
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.args import get_model_kwargs
from qai_hub_models.utils.evaluate.helpers import (
    _load_quant_cpu_onnx,
    evaluate_on_dataset,
)
from qai_hub_models.utils.inference import AsyncOnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.kwarg_helpers import filter_kwargs


def evaluate_model(
    model_id: str,
    args: argparse.Namespace,
    supports_quant_cpu: bool = False,
) -> None:
    """
    Evaluate a single-graph model's accuracy on a dataset.

    Parameters
    ----------
    model_id
        Model folder name (e.g. ``yolov8_det``).
    args
        Parsed argparse.Namespace from :func:`evaluate_parser`.
    supports_quant_cpu
        If True and precision != float, adds a "quant cpu" executor via
        :func:`_load_quant_cpu_onnx` for the QDQ ONNX accuracy check.
    """
    model_module = importlib.import_module(f"qai_hub_models.models.{model_id}")
    model_cls = model_module.Model

    model_kwargs = get_model_kwargs(model_cls, vars(args))
    input_spec_kwargs = filter_kwargs(model_cls.get_input_spec, vars(args))

    eval_dataset_classes = model_cls.get_eval_dataset_classes()
    if len(eval_dataset_classes) == 0:
        # PSNR fallback: no datasets, just run export with inference enabled
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        export_module = importlib.import_module(
            f"qai_hub_models.models.{model_id}.export"
        )
        export_model = export_module.export_model

        kwargs = {
            "device": args.device,
            "target_runtime": args.target_runtime,
            "skip_downloading": True,
            "skip_profiling": True,
            "compile_options": args.compile_options,
            "profile_options": args.profile_options,
            **model_kwargs,
            **input_spec_kwargs,
        }
        num_calibration_samples = getattr(args, "num_calibration_samples", None)
        if num_calibration_samples is not None:
            kwargs["num_calibration_samples"] = num_calibration_samples

        export_model(**kwargs)
        return

    input_spec: InputSpec | None = None
    torch_model = model_cls.from_pretrained(**model_kwargs)
    model_executors: dict[str, ExecutableModelProtocol] = {}

    if not args.skip_torch_accuracy:
        model_executors["torch"] = torch_model
        input_spec = torch_model.get_input_spec(**input_spec_kwargs)

    # On-device or quant-cpu path
    if not args.skip_device_accuracy or (
        supports_quant_cpu and getattr(args, "compute_quant_cpu_accuracy", False)
    ):
        if args.hub_model_id is not None:
            compiled_model: hub.Model = hub.get_model(args.hub_model_id)
        else:
            compiled_result = compile_model_from_args(
                model_id, args, {**model_kwargs, **input_spec_kwargs}
            )
            assert isinstance(compiled_result, hub.Model)
            compiled_model = compiled_result

        if compiled_model.get_producer() is None:
            raise ValueError(
                "Compiled models must be compiled with AI Hub Workbench; they cannot be uploaded manually."
            )

        on_device_model = AsyncOnDeviceModel(
            model=compiled_model,
            input_names=list(input_spec) if input_spec else None,
            device=args.device,
            inference_options=args.profile_options,
        )
        if not args.skip_device_accuracy:
            model_executors["on-device"] = on_device_model

        if (
            supports_quant_cpu
            and getattr(args, "compute_quant_cpu_accuracy", False)
            and args.precision != Precision.float
        ):
            model_executors["quant cpu"] = _load_quant_cpu_onnx(compiled_model)

        input_spec = on_device_model.get_input_spec()

    if input_spec is None:
        raise ValueError("Cannot extract input spec.")

    evaluate_on_dataset(
        evaluator_func=torch_model.get_evaluator,
        dataset_cls=args.dataset_cls,
        model_executors=model_executors,
        input_spec=input_spec,
        samples_per_job=args.samples_per_job,
        num_samples=args.num_samples,
        seed=args.seed,
        use_cache=args.use_dataset_cache,
    )

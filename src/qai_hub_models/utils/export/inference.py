# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Run on-device inference on sample inputs."""

from __future__ import annotations

import qai_hub as hub

from qai_hub_models import SampleInputsType
from qai_hub_models.utils.export.result import ComponentGroup


def run_inference(
    inputs: SampleInputsType,
    model_name: str,
    device: hub.Device,
    options: str,
    target_model: hub.Model,
) -> hub.client.InferenceJob:
    """Submit an on-device inference job over ``inputs``."""
    print(f"Running inference for {model_name} on a hosted device with example inputs.")
    return hub.submit_inference_job(
        model=target_model,
        inputs=inputs,
        device=device,
        name=model_name,
        options=options,
    )


def run_collection_inference(
    inputs_per_component: ComponentGroup[SampleInputsType],
    model_name: str,
    device: hub.Device,
    options_per_component: dict[str, str],
    target_models: ComponentGroup[hub.Model],
    components: list[str] | None = None,
) -> ComponentGroup[hub.client.InferenceJob]:
    """Submit one inference job per component."""
    components = components if components is not None else list(target_models)
    inference_jobs: ComponentGroup[hub.client.InferenceJob] = ComponentGroup()
    for name in components:
        print(f"Running inference for {name} on a hosted device with example inputs.")
        inference_jobs[name] = hub.submit_inference_job(
            model=target_models[name],
            inputs=inputs_per_component[name],
            device=device,
            name=f"{model_name}_{name}",
            options=options_per_component.get(name, ""),
        )
    return inference_jobs

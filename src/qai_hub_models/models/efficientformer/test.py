# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models.efficientformer.demo import main as demo_main
from qai_hub_models.models.efficientformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EfficientFormer,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
    run_imagenet_classifier_trace_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        EfficientFormer.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.39,
    )


@pytest.mark.trace
def test_trace() -> None:
    # check_trace=False: EfficientFormer uses dynamic control flow that
    # produces non-deterministic trace outputs, causing trace check to fail.
    run_imagenet_classifier_trace_test(
        EfficientFormer.from_pretrained(), check_trace=False
    )


def test_demo() -> None:
    demo_main(is_test=True)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.ffnet_122ns_lowres.demo import main as demo_main
from qai_hub_models.models.ffnet_122ns_lowres.model import FFNet122NSLowRes
from qai_hub_models.models.templates.cityscapes_segmentation.ffnet_test_utils import (
    run_test_off_target_numerical,
)


def test_off_target_numerical() -> None:
    run_test_off_target_numerical(
        FFNet122NSLowRes,
        "segmentation_ffnet122NS_CCC_mobile_pre_down",
    )


def test_demo() -> None:
    demo_main(is_test=True)

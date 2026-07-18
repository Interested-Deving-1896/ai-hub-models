# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models._shared.cityscapes_segmentation.ffnet_test_utils import (
    run_test_off_target_numerical,
)
from qai_hub_models.models.ffnet_40s.demo import main as demo_main
from qai_hub_models.models.ffnet_40s.model import FFNet40S


def test_off_target_numerical() -> None:
    run_test_off_target_numerical(FFNet40S, "segmentation_ffnet40S_dBBB_mobile")


def test_demo() -> None:
    demo_main(is_test=True)

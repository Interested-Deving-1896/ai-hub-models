# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models._shared.cityscapes_segmentation.ffnet_test_utils import (
    run_test_off_target_numerical,
)
from qai_hub_models.models.ffnet_78s.demo import main as demo_main
from qai_hub_models.models.ffnet_78s.model import FFNet78S


def test_off_target_numerical() -> None:
    run_test_off_target_numerical(FFNet78S, "segmentation_ffnet78S_dBBB_mobile")


def test_demo() -> None:
    demo_main(is_test=True)

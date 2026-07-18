# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.cityscapes_segmentation.ffnet_model import (
    FFNetLowRes,
)

MODEL_ID = __name__.split(".")[-2]


class FFNet78SLowRes(FFNetLowRes):
    variant_name = "segmentation_ffnet78S_BCC_mobile_pre_down"

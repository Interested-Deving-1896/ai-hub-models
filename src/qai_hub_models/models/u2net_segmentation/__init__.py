# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.u2net_segmentation.app import (
    SegmentationU2NetApp as App,
)
from qai_hub_models.models.u2net_segmentation.model import MODEL_ID
from qai_hub_models.models.u2net_segmentation.model import (
    SegmentationU2Net as Model,
)

__all__ = ["MODEL_ID", "App", "Model"]

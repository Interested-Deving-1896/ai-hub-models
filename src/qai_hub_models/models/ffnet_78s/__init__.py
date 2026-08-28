# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.cityscapes_segmentation.app import (
    CityscapesSegmentationApp as App,
)

from .model import MODEL_ID
from .model import FFNet78S as Model

__all__ = ["MODEL_ID", "App", "Model"]

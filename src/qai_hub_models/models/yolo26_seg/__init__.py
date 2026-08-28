# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.yolo.app import (
    YoloSegmentationApp as App,
)

from .model import MODEL_ID
from .model import Yolo26Segmentor as Model

__all__ = ["MODEL_ID", "App", "Model"]

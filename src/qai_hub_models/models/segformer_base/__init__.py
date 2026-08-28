# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.segmentation.app import (
    SegmentationApp as App,
)

from .model import MODEL_ASSET_VERSION, MODEL_ID
from .model import SegformerBase as Model

__all__ = ["MODEL_ASSET_VERSION", "MODEL_ID", "App", "Model"]

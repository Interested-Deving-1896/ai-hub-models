# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.selfie_segmentation.app import (
    SelfieSegmentationApp as App,
)

from .model import MODEL_ID
from .model import SINet as Model

__all__ = ["MODEL_ID", "App", "Model"]

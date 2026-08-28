# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.detr_resnet50_dc5.model import (
    DETRResNet50DC5 as Model,
)
from qai_hub_models.models.templates.detr.app import DETRApp as App

from .model import MODEL_ID

__all__ = ["MODEL_ID", "App", "Model"]

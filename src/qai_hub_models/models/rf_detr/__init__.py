# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.rf_detr.model import RF_DETR as Model
from qai_hub_models.models.templates.detr.app import DETRApp as App

from .model import MODEL_ID

__all__ = ["MODEL_ID", "App", "Model"]

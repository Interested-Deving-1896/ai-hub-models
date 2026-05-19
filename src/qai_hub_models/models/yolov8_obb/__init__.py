# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from .app import YoloV8OBBApp as App
from .model import MODEL_ID
from .model import YoloV8OBB as Model

__all__ = ["MODEL_ID", "App", "Model"]

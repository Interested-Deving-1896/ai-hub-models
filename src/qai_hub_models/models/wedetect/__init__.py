# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from .app import WeDetectModelApp
from .model import MODEL_ID, WeDetectDetector, WeDetectModel

App = WeDetectModelApp
Model = WeDetectModel

__all__ = [
    "MODEL_ID",
    "App",
    "Model",
    "WeDetectDetector",
    "WeDetectModel",
    "WeDetectModelApp",
]

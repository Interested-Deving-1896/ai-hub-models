# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.sam.app import SAMApp as App

# Use SamLarge as the default. This really needs to be split into multiple
# directory like llama
from .model import MODEL_ID
from .model import SAM as Model

__all__ = ["MODEL_ID", "App", "Model"]

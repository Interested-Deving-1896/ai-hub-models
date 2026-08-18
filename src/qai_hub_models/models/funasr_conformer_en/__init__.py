# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.funasr_conformer_en.app import FunASRConformerEnApp as App
from qai_hub_models.models.funasr_conformer_en.model import (
    MODEL_ID,
)
from qai_hub_models.models.funasr_conformer_en.model import (
    FunASRConformerEn as Model,
)

__all__ = ["MODEL_ID", "App", "Model"]

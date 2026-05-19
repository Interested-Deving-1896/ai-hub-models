# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloOBBApp


class YoloV8OBBApp(YoloOBBApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """YoloV8-OBB does not check for spatial dim shapes for input image"""

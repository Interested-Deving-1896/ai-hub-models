# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.templates.yolo.app import YoloObjectDetectionApp


class LicensePlateDetectorApp(YoloObjectDetectionApp):
    """End-to-end licence-plate detection application.

    Wraps ``LicensePlateDetector`` (or any compiled on-device equivalent)
    with the standard YOLO pre/post-processing pipeline.
    """

    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        pass

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.datasets.kinetics400 import (
    VIDEOMAE_FRAME_SAMPLE_RATE,
    VIDEOMAE_NUM_CLIPS,
    VIDEOMAE_NUM_CROPS,
    preprocess_video_224,
)
from qai_hub_models.models._shared.video_classifier.app import KineticsClassifierApp
from qai_hub_models.models._shared.video_classifier.model import KineticsClassifier


class VideoMAEApp(KineticsClassifierApp):
    def __init__(
        self,
        model: KineticsClassifier,
        num_frames: int = 16,
        num_clips: int = VIDEOMAE_NUM_CLIPS,
        num_crops: int = VIDEOMAE_NUM_CROPS,
        frame_sample_rate: int = VIDEOMAE_FRAME_SAMPLE_RATE,
    ) -> None:
        super().__init__(model, num_frames, num_clips, num_crops, frame_sample_rate)

    def preprocess_clip(self, clip: torch.Tensor) -> torch.Tensor:
        # Skip center-crop when multi_crop handles the spatial views instead.
        return preprocess_video_224(clip, center_crop=self.num_crops == 1)

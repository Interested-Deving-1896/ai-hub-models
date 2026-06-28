# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.superres_evaluator import SuperResolutionOutputEvaluator
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MetricMetadata


class StereoEvaluator(BaseEvaluator):
    """Evaluator for comparing a batched stereo image output (left + right views).

    Reuses :class:`SuperResolutionOutputEvaluator` (per-image YUV-space PSNR,
    8-bit data range) by composition: the left and right views are
    concatenated along the batch dimension and forwarded to it.
    """

    def __init__(self) -> None:
        self._psnr = SuperResolutionOutputEvaluator()

    def add_batch(self, output: list[torch.Tensor], gt: list[torch.Tensor]) -> None:
        """Evaluate one batch of stereo predictions.

        Parameters
        ----------
        output
            ``[left_pred, right_pred]`` — each tensor of shape ``(B, C, H, W)``.
        gt
            ``[left_gt, right_gt]`` — each tensor of shape ``(B, C, H, W)``.
        """
        assert gt[0].shape == output[0].shape and gt[1].shape == output[1].shape

        combined_output = torch.cat((output[0], output[1]), dim=0)
        combined_gt = torch.cat((gt[0], gt[1]), dim=0)

        self._psnr.add_batch(combined_output, combined_gt)

    def reset(self) -> None:
        self._psnr.reset()

    def get_accuracy_score(self) -> float:
        return self._psnr.get_accuracy_score()

    def formatted_accuracy(self) -> str:
        return self._psnr.formatted_accuracy()

    def get_metric_metadata(self) -> MetricMetadata:
        return self._psnr.get_metric_metadata()

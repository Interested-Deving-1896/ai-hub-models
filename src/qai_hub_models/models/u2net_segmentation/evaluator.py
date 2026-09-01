# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MEAN_ABSOLUTE_ERROR, MetricMetadata


class MeanAbsoluteErrorEvaluator(BaseEvaluator):
    """
    MAE evaluator for salient object detection.

    Matches the evaluation protocol of the U²-Net paper
    (Qin et al., Pattern Recognition 2020).

    MAE = (1/N) * sum_i [ (1/H*W) * sum_xy |P(x,y) - G(x,y)| ]

    Where:
    - P(x,y): predicted saliency map in [0, 1]
              (sigmoid already applied inside u2net_refactor)
    - G(x,y): GT binary mask normalized to [0, 1]
    - N: number of images
    """

    def __init__(self) -> None:
        self.reset()

    def add_batch(
        self,
        output: torch.Tensor,
        gt: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        output
            Model output [B, 1, H, W] — sigmoid already applied, range [0, 1]
        gt
            Ground truth binary mask [B, 1, H, W] — values already in [0, 1]
            as normalized by DUTSDataset
        """
        output = output.cpu().float()
        gt = gt.cpu().float()

        # Vectorized MAE: mean over H,W,C per image, sum over batch
        self.mae_sum += torch.mean(torch.abs(output - gt), dim=(1, 2, 3)).sum().item()
        self.count += output.shape[0]

    def reset(self) -> None:
        self.mae_sum: float = 0.0
        self.count: int = 0

    def formatted_accuracy(self) -> str:
        return f"MAE: {self.get_accuracy_score():.4f}"

    def get_accuracy_score(self) -> float:
        """
        Returns positive MAE score.
        Paper target: MAE=0.044
        """
        return self.mae_sum / max(self.count, 1)

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_ABSOLUTE_ERROR

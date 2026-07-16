# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import ACCURACY, MetricMetadata


class MobileFaceNetEvaluator(BaseEvaluator):
    """Evaluator for MobileFaceNet face verification on the LFW dataset."""

    def __init__(self, threshold: float = 74.18) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._angles: list[float] = []  # cosine angle in degrees per pair
        self._labels: list[int] = []  # 1 = same person, 0 = different

    def add_batch(
        self,
        output: torch.Tensor,
        gt: torch.Tensor | list[int],
    ) -> None:
        """
        Add a batch of model outputs and ground-truth labels.

        Parameters
        ----------
        output
            Embeddings tensor of shape ``(B*2, 128)`` as returned by
            ``MobileFaceNet.forward()``, which stacks both images of each pair
            channel-wise and produces two consecutive embedding rows per pair.
            Even rows (0, 2, 4, …) are embeddings for image 1; odd rows for image 2.
        gt
            Ground-truth labels of shape ``(B,)``: 1 = same person, 0 = different.
        """
        emb = output.cpu().float().numpy()  # (B*2, 128)
        emb1 = emb[0::2]  # (B, 128) — image 1
        emb2 = emb[1::2]  # (B, 128) — image 2

        # Ensure L2-normalised
        norms1 = np.linalg.norm(emb1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(emb2, axis=1, keepdims=True)
        emb1 = emb1 / np.where(norms1 > 0, norms1, 1.0)
        emb2 = emb2 / np.where(norms2 > 0, norms2, 1.0)

        cosines = np.clip((emb1 * emb2).sum(axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cosines))

        if isinstance(gt, torch.Tensor):
            labels = gt.cpu().numpy().astype(int).tolist()
        else:
            labels = [int(l) for l in gt]

        self._angles.extend(angles.tolist())
        self._labels.extend(labels)

    def get_accuracy_score(self) -> float:
        """
        Verification accuracy at the current threshold.

        Returns
        -------
        float
            Accuracy as a percentage in [0, 100].
        """
        if not self._angles:
            return 0.0

        angles = np.array(self._angles)
        labels = np.array(self._labels)
        predicted_same = angles <= self.threshold
        correct = int(((predicted_same) & (labels == 1)).sum()) + int(
            ((~predicted_same) & (labels == 0)).sum()
        )
        return correct / len(labels) * 100.0

    def formatted_accuracy(self) -> str:
        acc = self.get_accuracy_score()
        return (
            f"Accuracy: {acc:.3f}%  "
            f"(threshold: {self.threshold:.2f}°, "
            f"pairs: {len(self._angles)})"
        )

    def get_metric_metadata(self) -> MetricMetadata:
        return ACCURACY.with_description(
            "Verification accuracy on LFW pairs: fraction of same/different "
            "predictions that match ground truth at the cosine-angle threshold."
        )

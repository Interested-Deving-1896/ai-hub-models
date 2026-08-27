# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MetricMetadata


class LeRobotEvaluator(BaseEvaluator):
    """
    Measures Action RMSE between predicted and ground-truth action chunks from a LeRobot dataset

    Each call to add_batch() accepts:
      pred_chunk : np.ndarray  shape (action_horizon, total_dof)
      gt_chunk   : np.ndarray  shape (action_horizon, total_dof)

    RMSE is computed over all accumulated elements.
    """

    def __init__(
        self,
        dof_slices: list[tuple[str, slice]] | None = None,
    ) -> None:
        self._sum_sq_error: float = 0.0
        self._n_elements: int = 0
        self._n_steps: int = 0

        # Per-modality-group tracking
        self._dof_slices: list[tuple[str, slice]] = dof_slices or []
        self._group_sum_sq: list[float] = [0.0] * len(self._dof_slices)
        self._group_n_elem: list[int] = [0] * len(self._dof_slices)

    # BaseEvaluator interface
    def add_batch(
        self,
        pred_chunk: np.ndarray,
        gt_chunk: np.ndarray,
    ) -> None:
        if pred_chunk.shape != gt_chunk.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_chunk.shape} vs gt {gt_chunk.shape}"
            )
        sq_err = (pred_chunk.astype(np.float64) - gt_chunk.astype(np.float64)) ** 2
        self._sum_sq_error += float(sq_err.sum())
        self._n_elements += sq_err.size
        self._n_steps += 1
        # Per-modality-group accumulation — slice last (DOF) axis
        for i, (_, sl) in enumerate(self._dof_slices):
            group_sq = sq_err[..., sl]
            self._group_sum_sq[i] += float(group_sq.sum())
            self._group_n_elem[i] += group_sq.size

    def get_accuracy_score(self) -> float:
        """Returns RMSE (lower is better). Returns 0.0 for no data."""
        if self._n_elements == 0:
            return 0.0
        return float(np.sqrt(self._sum_sq_error / self._n_elements))

    def get_group_scores(self) -> list[tuple[str, float]]:
        """
        Per-modality-group RMSE scores
        Returns [(group_name, rmse), ...] in dof_slices order.
        Returns empty list if no dof_slices were provided.
        """
        results = []
        for i, (name, _) in enumerate(self._dof_slices):
            n = self._group_n_elem[i]
            rmse = float(np.sqrt(self._group_sum_sq[i] / n)) if n > 0 else 0.0
            results.append((name, rmse))
        return results

    def formatted_accuracy(self) -> str:
        lines = [
            f"Action RMSE: {self.get_accuracy_score():.6f} (over {self._n_steps} steps)"
        ]
        group_scores = self.get_group_scores()
        if group_scores:
            lines.append("  Per-modality-group RMSE:")
            for name, rmse in group_scores:
                lines.append(f"    [{name}]: {rmse:.6f}")
        return "\n".join(lines)

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Action RMSE",
            unit="rad",
            description="Root mean squared error between model predicted actions and ground truth actions from the dataset",
            range=(0.0, float("inf")),
            higher_is_better=False,
        )

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._sum_sq_error = 0.0
        self._n_elements = 0
        self._n_steps = 0
        self._group_sum_sq = [0.0] * len(self._dof_slices)
        self._group_n_elem = [0] * len(self._dof_slices)

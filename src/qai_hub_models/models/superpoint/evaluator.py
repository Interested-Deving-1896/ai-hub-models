# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import HOMOGRAPHY_ACCURACY, MetricMetadata


class SuperPointEvaluator(BaseEvaluator):
    """Evaluates SuperPoint on HPatches using homography estimation @3px.

    The dataset yields items in (ref, tgt, ref, tgt, ...) order. This evaluator
    buffers the ref output and evaluates when the matching tgt arrives.
    """

    def __init__(
        self,
        image_height: int = 480,
        image_width: int = 640,
        keep_k_points: int = 500,
        ransac_reproj_threshold: float = 3.0,
        correctness_thresh: float = 3.0,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.keep_k_points = keep_k_points
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.correctness_thresh = correctness_thresh
        self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.reset()

    def reset(self) -> None:
        self._num_total = 0
        self._homo_correct = 0
        # Buffer for the ref outputs of the current pair
        self._ref_buffer: dict[str, np.ndarray] | None = None

    def get_accuracy_score(self) -> float:
        if self._num_total == 0:
            return 0.0
        return 100.0 * self._homo_correct / self._num_total

    def formatted_accuracy(self) -> str:
        return f"Homography@3 {self.get_accuracy_score():.1f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return HOMOGRAPHY_ACCURACY

    def add_batch(
        self,
        output: Sequence[torch.Tensor],
        gt: Sequence[torch.Tensor],
    ) -> None:
        """Accumulate SuperPoint outputs one image at a time.

        Parameters
        ----------
        output
            Model outputs: (keypoints, scores, descriptors, num_keypoints).
            keypoints shape ``(B, MAX_K, 2)``, scores ``(B, MAX_K)``,
            descriptors ``(B, MAX_K, 256)``, num_keypoints ``(B,)``.
        gt
            Ground-truth tuple ``(H_gt, is_ref)`` where H_gt shape ``(B, 3, 3)``
            and is_ref shape ``(B,)`` bool — True for reference, False for target.
        """
        kp_batch, sc_batch, desc_batch, nkp_batch = output
        H_gt_batch, is_ref_batch = gt

        B = kp_batch.shape[0]
        for i in range(B):
            n = int(nkp_batch[i].item())
            kp = kp_batch[i, :n].cpu().numpy()
            sc = sc_batch[i, :n].cpu().numpy()
            desc = desc_batch[i, :n].cpu().numpy()
            H_gt = H_gt_batch[i].cpu().numpy().astype(np.float64)
            is_ref = bool(is_ref_batch[i].item())

            if is_ref:
                if self._ref_buffer is not None:
                    raise RuntimeError(
                        "Two consecutive ref items received — dataset stream is misordered."
                    )
                self._ref_buffer = {"kp": kp, "sc": sc, "desc": desc}
            else:
                if self._ref_buffer is None:
                    # Misaligned stream — skip
                    self._num_total += 1
                    continue
                correct = self._evaluate_pair(
                    self._ref_buffer["kp"],
                    self._ref_buffer["desc"],
                    kp,
                    desc,
                    H_gt,
                )
                self._homo_correct += int(correct)
                self._num_total += 1
                self._ref_buffer = None

    def _evaluate_pair(
        self,
        kp0: np.ndarray,
        desc0: np.ndarray,
        kp1: np.ndarray,
        desc1: np.ndarray,
        H_gt: np.ndarray,
    ) -> bool:
        """Filter to shared points, match, RANSAC, check corner error.

        Parameters
        ----------
        kp0
            Reference keypoints (x, y), shape (N, 2).
        desc0
            Reference descriptors, shape (N, 256).
        kp1
            Target keypoints (x, y), shape (M, 2).
        desc1
            Target descriptors, shape (M, 256).
        H_gt
            Ground-truth homography (3, 3) mapping ref -> tgt.

        Returns
        -------
        bool
            True if the estimated homography is correct within threshold.
        """
        image_hw = (self.image_height, self.image_width)
        kp0, desc0 = self._keep_shared_points(
            kp0, desc0, H_gt, image_hw, self.keep_k_points
        )
        kp1, desc1 = self._keep_shared_points(
            kp1, desc1, np.linalg.inv(H_gt), image_hw, self.keep_k_points
        )
        if len(kp0) < 4 or len(kp1) < 4:
            return False
        matches = self._bf.match(desc0.astype(np.float32), desc1.astype(np.float32))
        if len(matches) < 4:
            return False
        q_idx = np.array([m.queryIdx for m in matches], dtype=np.int32)
        t_idx = np.array([m.trainIdx for m in matches], dtype=np.int32)
        H_est, inliers = cv2.findHomography(
            kp0[q_idx].astype(np.float32),
            kp1[t_idx].astype(np.float32),
            cv2.RANSAC,
            self.ransac_reproj_threshold,
        )
        if H_est is None or inliers is None or inliers.sum() < 4:
            return False
        return self._corner_error(H_est, H_gt, image_hw) < self.correctness_thresh

    @staticmethod
    def _keep_shared_points(
        kp: np.ndarray,
        desc: np.ndarray,
        H: np.ndarray,
        image_hw: tuple[int, int],
        keep_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter keypoints whose projection through H stays inside image bounds.

        Parameters
        ----------
        kp
            Keypoints (x, y), shape (N, 2).
        desc
            Descriptors, shape (N, D).
        H
            Homography to project through.
        image_hw
            Image bounds (H, W).
        keep_k
            Max keypoints to retain (top by order, already score-sorted).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Filtered keypoints and descriptors.
        """
        h, w = image_hw
        if len(kp) == 0:
            return kp, desc
        n = kp.shape[0]
        pts_h = np.concatenate([kp, np.ones((n, 1))], axis=1)
        warped = (H @ pts_h.T).T
        warped_xy = warped[:, :2] / warped[:, 2:3]
        mask = (
            (warped_xy[:, 0] >= 0)
            & (warped_xy[:, 0] < w)
            & (warped_xy[:, 1] >= 0)
            & (warped_xy[:, 1] < h)
        )
        kp, desc = kp[mask], desc[mask]
        if len(kp) > keep_k:
            kp, desc = kp[:keep_k], desc[:keep_k]
        return kp, desc

    @staticmethod
    def _corner_error(
        H_est: np.ndarray,
        H_gt: np.ndarray,
        image_hw: tuple[int, int],
    ) -> float:
        """Mean corner reprojection error between estimated and ground-truth homography.

        Parameters
        ----------
        H_est
            Estimated homography (3, 3).
        H_gt
            Ground-truth homography (3, 3).
        image_hw
            Image bounds (H, W).

        Returns
        -------
        float
            Mean Euclidean distance between projected corners in pixels.
        """
        h, w = image_hw
        H_est = np.asarray(H_est, dtype=np.float64).reshape(3, 3)
        H_gt = np.asarray(H_gt, dtype=np.float64).reshape(3, 3)
        corners = np.array(
            [[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]],
            dtype=np.float64,
        )
        real_warped = corners @ H_gt.T
        real_warped = real_warped[:, :2] / real_warped[:, 2:3]
        est_warped = corners @ H_est.T
        est_warped = est_warped[:, :2] / est_warped[:, 2:3]
        return float(np.mean(np.linalg.norm(real_warped - est_warped, axis=1)))

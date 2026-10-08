# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.datasets.coco import CocoPanopticSegmentationDataset
from qai_hub_models.models.mask2former.app import Mask2FormerApp as app
from qai_hub_models.models.templates.panoptic_seg.external_repos.panopticapi.panopticapi.evaluation import (
    PQStat,
)
from qai_hub_models.models.templates.panoptic_seg.external_repos.panopticapi.panopticapi.utils import (
    rgb2id,
)
from qai_hub_models.models.templates.panoptic_seg.pq_metric import (
    pq_compute_single_image,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    PANOPTIC_QUALITY,
    MetricMetadata,
)


class PanopticSegmentationEvaluator(BaseEvaluator):
    """Evaluator for panoptic segmentation metrics (PQ, SQ, RQ)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dataset = CocoPanopticSegmentationDataset()
        self.annotations = self.dataset.annotations
        self.label_divisor = 1000
        self.ignore_label = 0
        self._init_mappings()
        self.reset()

    def _init_mappings(self) -> None:
        self.img_ann_map = {
            ann["image_id"]: ann for ann in self.annotations["annotations"]
        }
        self.categories = {el["id"]: el for el in self.annotations["categories"]}
        sorted_cats = sorted(self.categories.values(), key=lambda x: x["id"])
        self.coco_to_contiguous = {
            cat["id"]: idx for idx, cat in enumerate(sorted_cats)
        }
        self.contiguous_to_coco = {v: k for k, v in self.coco_to_contiguous.items()}
        self.preprocessed_gt_segments = {}
        for img_id, ann in self.img_ann_map.items():
            self.preprocessed_gt_segments[img_id] = [
                {
                    "id": s["id"],
                    "category_id": self.coco_to_contiguous[s["category_id"]],
                    "area": s["area"],
                    "iscrowd": s.get("iscrowd", 0),
                }
                for s in ann.get("segments_info", [])
            ]

    def reset(self) -> None:
        """Reset the evaluation statistics."""
        self.pq_stat = PQStat()

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt_data: tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Process model predictions and ground truth for panoptic segmentation evaluation.

        Parameters
        ----------
        output
            Model predictions with class logits, class labels, and mask logits.
        gt_data
            Ground truth panoptic masks and image IDs, optionally followed by
            letterbox scale/pad metadata (unused here; CocoPanopticSegmentationDataset
            may return either a 2-tuple or a 4-tuple).
        """
        pred_scores, pred_labels, pred_masks_logits = output
        batch_results = app.post_process_panoptic_segmentation(
            pred_scores, pred_labels, pred_masks_logits
        )
        batched_gt_masks, batched_img_ids = gt_data[0], gt_data[1]

        for i in range(batched_gt_masks.shape[0]):
            result = batch_results[i]
            gt_mask = rgb2id(batched_gt_masks[i].cpu().numpy())
            img_id = batched_img_ids[i].item()

            # Process GT
            gt_segments = self.preprocessed_gt_segments.get(img_id, [])

            # Process predictions
            pred_mask = result["segmentation"].cpu().numpy()

            segments_info: list[dict] = []

            for s in result["segments_info"]:
                coco_cat_id = self.contiguous_to_coco.get(s["label_id"])
                if coco_cat_id is None:
                    continue
                instance_id = len(segments_info) + 1
                panoptic_id = coco_cat_id * self.label_divisor + instance_id
                area = np.sum(pred_mask == (s["id"] % self.label_divisor))
                segments_info.append(
                    {
                        "id": panoptic_id,
                        "category_id": s["label_id"],
                        "area": int(area),
                        "iscrowd": 0,
                        "score": s["score"],
                    }
                )

            # Update pred mask with panoptic IDs
            final_pred_mask = np.zeros_like(pred_mask)
            for seg in segments_info:
                instance_id = seg["id"] % self.label_divisor
                final_pred_mask[pred_mask == instance_id] = seg["id"]

            self._evaluate_single_image(
                pred_mask=final_pred_mask,
                gt_mask=gt_mask,
                pred_segments_info=segments_info,
                gt_segments_info=gt_segments,
            )

    def _evaluate_single_image(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        pred_segments_info: list[dict],
        gt_segments_info: list[dict],
    ) -> PQStat:
        """Evaluate a single image's prediction against ground truth.

        Parameters
        ----------
        pred_mask
            Predicted panoptic mask of shape (H, W) with panoptic IDs (category_id * label_divisor + instance_id).
        gt_mask
            Ground truth panoptic mask of shape (H, W) with panoptic IDs or 0 for ignored regions.
        pred_segments_info
            Predicted segment info with id, category_id, area, iscrowd (0), score.
        gt_segments_info
            Ground truth segment info with id, category_id, area, iscrowd (0 or 1).

        Returns
        -------
        PQStat
            Updated panoptic quality statistics for the image.
        """
        pq_stat = pq_compute_single_image(
            gt_mask,
            pred_mask,
            gt_segments_info,
            pred_segments_info,
            label_divisor=self.label_divisor,
        )
        self.pq_stat += pq_stat
        return pq_stat

    def compute_pq(self) -> dict[str, dict]:
        """Compute panoptic quality metrics for all, things, and stuff categories.

        Returns
        -------
        metrics : dict[str, dict]
            Metrics for "All", "Things", "Stuff" with pq, sq, rq, and n (number of categories).
        """
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        return {
            name: self.pq_stat.pq_average(self.categories, isthing=isthing)[0]
            for name, isthing in metrics
        }

    def get_accuracy_score(self) -> float:
        """Return the PQ score for all categories."""
        return self.compute_pq()["All"]["pq"]

    def formatted_accuracy(self) -> str:
        """Return formatted PQ score as a percentage."""
        return f"{self.get_accuracy_score() * 100:.1f} PQ"

    def get_metric_metadata(self) -> MetricMetadata:
        return PANOPTIC_QUALITY

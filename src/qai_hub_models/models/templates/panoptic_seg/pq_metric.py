# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.templates.panoptic_seg.external_repos.panopticapi.panopticapi.evaluation import (
    PQStat,
)


def pq_compute_single_image(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_segments_info: list[dict],
    pred_segments_info: list[dict],
    label_divisor: int = 1000,
    ignore_label: int = 0,
) -> PQStat:
    """
    Compute panoptic quality (PQ) statistics for a single image.

    Adapted from ``panopticapi.evaluation.pq_compute_single_core``, which
    only operates on file-based PNG masks read from disk via multiprocessing;
    this version takes in-memory numpy arrays so it can be called per-sample
    from a ``BaseEvaluator.add_batch`` loop.
    Ref: https://github.com/cocodataset/panopticapi/blob/7bb4655548f9/panopticapi/evaluation.py#L27

    Parameters
    ----------
    gt_mask
        Ground truth panoptic mask of shape (H, W) with panoptic ids
        (matching the ids in ``gt_segments_info``), or ``ignore_label`` for
        ignored regions.
    pred_mask
        Predicted panoptic mask of shape (H, W) with panoptic ids matching
        the ids in ``pred_segments_info``.
    gt_segments_info
        Ground truth segment dicts with keys ``id``, ``category_id``,
        ``area``, ``iscrowd`` (0 or 1).
    pred_segments_info
        Predicted segment dicts with keys ``id``, ``category_id``, ``area``,
        ``iscrowd`` (always 0), ``score``.
    label_divisor
        Divisor used to encode (category_id, instance_id) into a single
        panoptic id; must match how segment ids were assigned upstream.
    ignore_label
        Panoptic id value used for ignored/void regions.

    Returns
    -------
    PQStat
        Per-category true positives, false positives, false negatives, and
        summed IoU for matched segments in this image.
    """
    pq_stat = PQStat()
    VOID = ignore_label
    OFFSET = label_divisor * label_divisor

    gt_segms = {el["id"]: el for el in gt_segments_info}
    pred_segms = {el["id"]: el for el in pred_segments_info}

    # Update areas from actual mask pixel counts
    for seg_id, count in zip(*np.unique(pred_mask, return_counts=True), strict=False):
        if seg_id in pred_segms and seg_id != VOID:
            pred_segms[seg_id]["area"] = int(count)
    for seg_id, count in zip(*np.unique(gt_mask, return_counts=True), strict=False):
        if seg_id in gt_segms and seg_id != VOID:
            gt_segms[seg_id]["area"] = int(count)

    # Store every (gt_id, pred_id) intersection, including pairs where one
    # side is VOID. The VOID intersections are looked up below to correct the
    # union (matched pairs) and to suppress predictions lying mostly on VOID
    # (false positives). Dropping them here — e.g. only keeping pairs where
    # both ids are real segments — makes those corrections dead, inflating FPs
    # on real data where VOID regions are common.
    pan_gt_pred = gt_mask.astype(np.uint64) * OFFSET + pred_mask.astype(np.uint64)
    gt_pred_map: dict = {}
    labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
    for label, intersection in zip(labels, labels_cnt, strict=False):
        gt_id = int(label // OFFSET)
        pred_id = int(label % OFFSET)
        gt_pred_map[(gt_id, pred_id)] = intersection

    gt_matched: set = set()
    pred_matched: set = set()
    for (gt_id, pred_id), intersection in gt_pred_map.items():
        if gt_id not in gt_segms or pred_id not in pred_segms:
            continue
        if gt_segms[gt_id].get("iscrowd", 0) == 1:
            continue
        gt_cat = gt_segms[gt_id]["category_id"]
        pred_cat = pred_segms[pred_id]["category_id"]
        if gt_cat != pred_cat:
            continue
        union = (
            gt_segms[gt_id]["area"]
            + pred_segms[pred_id]["area"]
            - intersection
            - gt_pred_map.get((VOID, pred_id), 0)
        )
        iou = intersection / union if union > 0 else 0
        if iou > 0.5:
            pq_stat[gt_cat].tp += 1
            pq_stat[gt_cat].iou += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id, gt_info in gt_segms.items():
        if gt_id in gt_matched or gt_info.get("iscrowd", 0) == 1 or gt_id == VOID:
            continue
        pq_stat[gt_info["category_id"]].fn += 1

    crowd_labels_dict = {
        gt_info["category_id"]: gt_id
        for gt_id, gt_info in gt_segms.items()
        if gt_info.get("iscrowd", 0) == 1
    }
    for pred_id, pred_info in pred_segms.items():
        if pred_id in pred_matched or pred_id == VOID:
            continue
        intersection = gt_pred_map.get((VOID, pred_id), 0)
        if pred_info["category_id"] in crowd_labels_dict:
            intersection += gt_pred_map.get(
                (crowd_labels_dict[pred_info["category_id"]], pred_id), 0
            )
        if pred_info["area"] > 0 and intersection / pred_info["area"] <= 0.5:
            pq_stat[pred_info["category_id"]].fp += 1

    return pq_stat

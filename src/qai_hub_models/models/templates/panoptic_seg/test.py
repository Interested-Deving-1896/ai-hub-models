# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.templates.panoptic_seg.pq_metric import (
    pq_compute_single_image,
)


def _segment(seg_id: int, category_id: int, area: int, iscrowd: int = 0) -> dict:
    return {
        "id": seg_id,
        "category_id": category_id,
        "area": area,
        "iscrowd": iscrowd,
    }


def test_perfect_match() -> None:
    """Identical GT and prediction -> one true positive per segment, IoU 1."""
    # id 1 (thing, cat 10) on top half; id 2 (stuff, cat 20) on bottom half.
    gt = np.zeros((4, 4), dtype=np.int64)
    gt[:2, :] = 1
    gt[2:, :] = 2
    pred = gt.copy()

    gt_info = [_segment(1, 10, 8), _segment(2, 20, 8)]
    pred_info = [_segment(1, 10, 8), _segment(2, 20, 8)]

    stat = pq_compute_single_image(gt, pred, gt_info, pred_info)

    for cat in (10, 20):
        assert stat[cat].tp == 1
        assert stat[cat].fp == 0
        assert stat[cat].fn == 0
        assert stat[cat].iou == 1.0


def test_missed_segment_is_false_negative() -> None:
    """A GT segment with no matching prediction is a false negative."""
    gt = np.zeros((4, 4), dtype=np.int64)
    gt[:2, :] = 1  # cat 10
    gt[2:, :] = 2  # cat 20
    # Prediction only reproduces the top (cat 10) segment.
    pred = np.zeros((4, 4), dtype=np.int64)
    pred[:2, :] = 1

    gt_info = [_segment(1, 10, 8), _segment(2, 20, 8)]
    pred_info = [_segment(1, 10, 8)]

    stat = pq_compute_single_image(gt, pred, gt_info, pred_info)

    assert stat[10].tp == 1
    assert stat[20].tp == 0
    assert stat[20].fn == 1


def test_wrong_category_is_fp_and_fn() -> None:
    """A prediction overlapping a real GT segment of a different category
    counts as both a false positive (prediction) and a false negative (GT).
    """
    gt = np.zeros((4, 4), dtype=np.int64)
    gt[:2, :] = 1  # cat 10
    gt[2:, :] = 2  # cat 20
    pred = np.zeros((4, 4), dtype=np.int64)
    pred[:2, :] = 1  # cat 10 - matches GT id 1
    pred[2:, :] = 3  # cat 30 - overlaps GT cat 20 region, different category

    gt_info = [_segment(1, 10, 8), _segment(2, 20, 8)]
    pred_info = [_segment(1, 10, 8), _segment(3, 30, 8)]

    stat = pq_compute_single_image(gt, pred, gt_info, pred_info)

    assert stat[10].tp == 1
    assert stat[20].fn == 1  # GT cat 20 unmatched
    assert stat[30].fp == 1  # pred cat 30 has no GT match


def test_low_iou_is_not_a_match() -> None:
    """Overlap with IoU <= 0.5 does not count as a match: the prediction is a
    false positive and the ground truth is a false negative.
    """
    gt = np.zeros((6, 4), dtype=np.int64)
    gt[:3, :] = 1  # 12 px, cat 10
    pred = np.zeros((6, 4), dtype=np.int64)
    pred[:1, :] = 1  # 4 px, cat 10 -> IoU = 4 / 12 ~= 0.33

    gt_info = [_segment(1, 10, 12)]
    pred_info = [_segment(1, 10, 4)]

    stat = pq_compute_single_image(gt, pred, gt_info, pred_info)

    assert stat[10].tp == 0
    assert stat[10].fp == 1
    assert stat[10].fn == 1


def test_prediction_on_void_is_not_false_positive() -> None:
    """A prediction lying mostly on VOID (unlabeled) GT pixels is ignored,
    not counted as a false positive.

    This exercises the VOID-intersection correction: the ``(VOID, pred_id)``
    pair must be retained in the intersection map for the suppression to fire.
    """
    gt = np.zeros((4, 4), dtype=np.int64)  # entirely VOID (id 0)
    pred = np.full((4, 4), 5, dtype=np.int64)  # one segment covering everything

    pred_info = [_segment(5, 30, 16)]

    stat = pq_compute_single_image(gt, pred, [], pred_info)

    assert stat[30].fp == 0
    assert stat[30].tp == 0
    assert stat[30].fn == 0


def test_void_intersection_excluded_from_union() -> None:
    """VOID pixels overlapping a prediction are removed from the IoU union so
    a segment that fully covers its GT (plus some VOID) still matches.
    """
    # GT id 1 (cat 10) covers the top 2 rows; the rest is VOID.
    gt = np.zeros((4, 4), dtype=np.int64)
    gt[:2, :] = 1
    # Prediction covers the top 3 rows: the 8 GT pixels plus 4 VOID pixels.
    pred = np.zeros((4, 4), dtype=np.int64)
    pred[:3, :] = 1

    gt_info = [_segment(1, 10, 8)]
    pred_info = [_segment(1, 10, 12)]

    stat = pq_compute_single_image(gt, pred, gt_info, pred_info)

    # Without the VOID correction, union = 8 + 12 - 8 = 12 and IoU = 8/12 = 0.67
    # (still a match, but a depressed IoU). With the correction the 4 VOID
    # pixels are removed from the union: union = 12 - 4 = 8, IoU = 8/8 = 1.0.
    assert stat[10].tp == 1
    assert stat[10].iou == 1.0


def test_crowd_ground_truth_not_penalized() -> None:
    """An unmatched ``iscrowd`` GT segment is ignored (no false negative)."""
    gt = np.ones((4, 4), dtype=np.int64)  # id 1 everywhere
    pred = np.zeros((4, 4), dtype=np.int64)  # nothing predicted

    gt_info = [_segment(1, 10, 16, iscrowd=1)]

    stat = pq_compute_single_image(gt, pred, gt_info, [])

    assert stat[10].fn == 0


def test_empty_inputs() -> None:
    """No GT and no predictions produces empty per-category statistics."""
    gt = np.zeros((2, 2), dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)

    stat = pq_compute_single_image(gt, pred, [], [])

    assert len(stat.pq_per_cat) == 0

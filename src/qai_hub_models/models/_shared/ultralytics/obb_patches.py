# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import math
from typing import cast

import torch
from ultralytics.nn.modules.head import OBB
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils.tal import dist2rbox, make_anchors


def patch_ultralytics_obb_head(model: OBBModel) -> None:
    """
    Patch the model's detection head for export compatibility.

    1. Disable Ultralytics postprocessing (we add our own postprocessing for YOLO models).
    2. Enable "export" mode.
    3. Remove a concat that makes quantization impossible.

    Parameters
    ----------
    model
        OBBModel to patch.
    """
    head = cast(OBB, model.model[-1])
    head.export = True
    head.end2end = False
    head._inference = functools.partial(patched_ultralytics_obb_head_inference, head)  # type: ignore[assignment]
    head.forward = functools.partial(patched_ultralytics_obb_head_forward, head)  # type: ignore[assignment]


def patched_ultralytics_obb_head_forward(
    self: OBB, x: list[torch.Tensor]
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]
):
    """
    Adjusted version of OBB::forward that does not concat the bounding boxes, angles and class probs.

    The boxes, angles and probs are very different ranges (int vs [0-1]). Concatenation makes quantization impossible.

    Decode predicted bounding boxes, angles and class probabilities based on multiple-level feature maps.

    Parameters
    ----------
    self
        Detect module instance.
    x
        List of feature maps from different detection layers.

    Returns
    -------
    output : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        If training: (boxes, angles, scores) where each is a list
        If export: (yboxes, yangles, yscores) as tensors
        Otherwise: (yboxes, yangles, yscores, boxes, angles, scores) with yboxes/yangles/yscores as tensors and boxes/angles/scores as lists
    """
    assert not self.end2end

    pred_dist = []
    pred_angles = []
    pred_scores = []

    # Iterate over pyramid levels (P3, P4, P5)
    for i in range(self.nl):
        # 1. Box Branch (cv2)
        b = self.cv2[i](x[i])
        pred_dist.append(b)

        # 2. Class Branch (cv3)
        c = self.cv3[i](x[i])
        pred_scores.append(c)

        # 3. Angle Branch (cv4)
        a = self.cv4[i](x[i])
        pred_angles.append(a)

    if self.training:
        return pred_dist, pred_angles, pred_scores

    # Inference path
    yboxes, yangles, yscores = patched_ultralytics_obb_head_inference(
        self, pred_dist, pred_angles, pred_scores
    )

    return (
        (yboxes, yangles, yscores)
        if self.export
        else (yboxes, yangles, yscores, pred_dist, pred_angles, pred_scores)
    )


def patched_ultralytics_obb_head_inference(
    self: OBB,
    boxes: list[torch.Tensor],
    angles: list[torch.Tensor],
    scores: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adjusted version of OBB::_inference for quantization friendly export.
    Decodes bounding boxes (xywh) and angles (radians) separately.

    Parameters
    ----------
    self
        OBB module instance.
    boxes
        List of box predictions (distributions) from different detection layers.
    angles
        List of angle predictions from different detection layers.
    scores
        List of score predictions from different detection layers.

    Returns
    -------
    dbox : torch.Tensor
        Decoded bounding boxes in xywh format (N, 4).
    angle : torch.Tensor
        Decoded angles in radians (N, 1).
    cls : torch.Tensor
        Class probabilities after sigmoid (N, C).
    """
    shape = boxes[0].shape  # BCHW

    # 1. Flatten and Concatenate across scales (H*W)
    # Box Distribution: [Batch, 4*reg_max, Total_Anchors]
    box = torch.cat(tuple(b.view(shape[0], boxes[0].shape[1], -1) for b in boxes), 2)
    # Scores: [Batch, num_classes, Total_Anchors]
    cls = torch.cat(tuple(s.view(shape[0], scores[0].shape[1], -1) for s in scores), 2)
    # Angles: [Batch, 1, Total_Anchors]
    angle = torch.cat(
        tuple(a.view(shape[0], angles[0].shape[1], -1) for a in angles), 2
    )

    # 2. Generate Anchors
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (
            x.transpose(0, 1) for x in make_anchors(boxes, self.stride, 0.5)
        )
        self.shape = shape  # type: ignore[assignment]

    # 3. Decode Angles
    # Standard YOLOv8 OBB angle decoding: range [-pi/4, 3pi/4]
    angle_decoded = (angle.sigmoid() - 0.25) * math.pi

    # 4. Decode Boxes
    pred_dist = self.dfl(box)
    dbox = (
        dist2rbox(pred_dist, angle_decoded, self.anchors.unsqueeze(0), dim=1)
        * self.strides
    )

    return dbox, angle_decoded, cls.sigmoid()

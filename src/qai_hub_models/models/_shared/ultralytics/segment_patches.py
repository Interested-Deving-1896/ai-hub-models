# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from typing import cast

import torch
from ultralytics.nn.modules.head import Segment, Segment26, YOLOESegment
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.tal import make_anchors

from qai_hub_models.models._shared.ultralytics.detect_patches import (
    patched_ultryaltics_det_head_forward,
    patched_ultryaltics_det_head_inference,
)


def patch_ultralytics_segmentation_head(model: SegmentationModel) -> None:
    """
    Patches the segmentation model head for export / quantization.

    Discussion:
        After patching, the model will return the following:
            boxes:
                Shape [batch_size, 4, num_anchors]
                where 4 = [x, y, w, h] (box coordinates in pixel space)
            scores:
                Shape [batch_size, num classes (1), num_anchors]
                per-anchor class confidence. Single class is box contain object / box does not contain object
            mask_coefficients:
                Shape [batch_size, num_prototype_masks, num_anchors]
                Coefficients for each prototype mask.
            mask_prototypes:
                Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
    """
    head = cast(Segment, model.model[-1])

    # Makes the model traceable
    head.export = True

    # Patch inference head to skip concat of boxes & scores
    # This is required for int8 quantization.
    head.forward = functools.partial(patched_ultralytics_seg_head_forward, head)
    head._inference = functools.partial(patched_ultryaltics_det_head_inference, head)  # type: ignore[assignment]


def patched_ultralytics_seg_head_forward(
    self: Segment, x: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
    #####
    # <Copied from Ultralytics>
    ####
    p = self.proto(x[0])  # mask protos
    bs = p.shape[0]  # batch size

    mc = torch.cat(
        [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
    )  # mask coefficients
    ###
    # /<Copied From Ultralytics>
    ###

    boxes, scores = cast(
        tuple[torch.Tensor, torch.Tensor], patched_ultryaltics_det_head_forward(self, x)
    )
    return boxes, scores, mc, p


def patch_ultralytics_segmentation_head_26(model: SegmentationModel) -> None:
    """
    Patches the YOLO26 segmentation model head for export / quantization.

    Discussion:
        After patching, the model will return the following:
            boxes:
                Shape [batch_size, 4, num_anchors]
                where 4 = [x, y, w, h] (box coordinates in pixel space)
            scores:
                Shape [batch_size, num classes (1), num_anchors]
                per-anchor class confidence. Single class is box contain object / box does not contain object
            mask_coefficients:
                Shape [batch_size, num_prototype_masks, num_anchors]
                Coefficients for each prototype mask.
            mask_prototypes:
                Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
    """
    head = cast(Segment26, model.model[-1])

    # Makes the model traceable
    head.export = True
    head.end2end = False

    # Patch inference head to skip concat of boxes & scores
    # This is required for int8 quantization.
    head.forward = functools.partial(patched_ultralytics_seg_head_26_forward, head)
    head._inference = functools.partial(patched_ultryaltics_det_head_inference, head)  # type: ignore[assignment]


def patched_ultralytics_seg_head_26_forward(
    self: Segment26, x: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
    #####
    # <Copied from Ultralytics>
    ####
    p = self.proto(x)  # mask protos
    bs = p.shape[0]  # batch size

    mc = torch.cat(
        [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
    )  # mask coefficients
    ###
    # /<Copied From Ultralytics>
    ###

    boxes, scores = cast(
        tuple[torch.Tensor, torch.Tensor], patched_ultryaltics_det_head_forward(self, x)
    )
    return boxes, scores, mc, p


def patch_ultralytics_yoloe_segmentation_head(model: SegmentationModel) -> None:
    """
    Patches the YOLOE segmentation model head for export / quantization.

    Discussion:
        After patching, the model will return the following:
            boxes:
                Shape [batch_size, 4, num_anchors]
                where 4 = [x, y, w, h] (box coordinates in pixel space)
            scores:
                Shape [batch_size, num classes, num_anchors]
                per-anchor class confidence with text-guided semantic understanding
            mask_coefficients:
                Shape [batch_size, num_prototype_masks, num_anchors]
                Coefficients for each prototype mask.
            mask_prototypes:
                Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
    """
    head = cast(YOLOESegment, model.model[-1])

    # Makes the model traceable
    head.export = True
    head.end2end = False

    # Patch inference head to skip concat of boxes & scores
    # This is required for int8 quantization.
    head.forward = functools.partial(patched_ultralytics_yoloe_seg_head_forward, head)


def patched_ultralytics_yoloe_seg_head_forward(
    self: YOLOESegment, x: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adjusted version of YOLOESegment::forward that does not concat the bounding boxes and class probs.

    Returns model outputs and mask coefficients with separate boxes and scores for quantization compatibility.

    Parameters
    ----------
    self
        YOLOESegment module instance.
    x
        List of feature maps including 3 feature maps and 1 text embeddings.

    Returns
    -------
    yboxes : torch.Tensor
        Decoded bounding boxes with shape [batch_size, 4, num_anchors].
    yscores : torch.Tensor
        Class probabilities with shape [batch_size, num_classes, num_anchors].
    mc : torch.Tensor
        Mask coefficients with shape [batch_size, num_prototype_masks, num_anchors].
    p : torch.Tensor
        Mask prototypes with shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size].
    """
    assert len(x) == 4, (
        f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
    )

    # Generate mask prototypes
    p = self.proto(x[0])  # mask protos
    bs = p.shape[0]  # batch size

    # Generate mask coefficients
    mc = torch.cat(
        [self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
    )  # mask coefficients

    # Generate boxes and scores separately (not concatenated)
    boxes = []
    scores = []
    for i in range(self.nl):
        boxes.append(self.cv2[i](x[i]))
        # YOLOESegment uses cv4 (contrastive head) with cv3 (embedding) and text
        scores.append(self.cv4[i](self.cv3[i](x[i]), x[-1]))

    # Inference path - decode boxes and scores
    yboxes, yscores = patched_yoloe_seg_head_inference(self, boxes, scores, x[:3])

    return (yboxes, yscores, mc, p)


def patched_yoloe_seg_head_inference(
    self: YOLOESegment,
    boxes: list[torch.Tensor],
    scores: list[torch.Tensor],
    feats: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adjusted version of YOLOESegment::_inference that does not concat the bounding boxes and class probs.

    The boxes and probs are very different ranges (int vs [0-1]). Concatenation makes quantization impossible.

    Parameters
    ----------
    self
        YOLOESegment module instance.
    boxes
        List of box predictions from different detection layers.
    scores
        List of score predictions from different detection layers.
    feats
        Original features for anchor generation.

    Returns
    -------
    dbox : torch.Tensor
        Decoded bounding boxes.
    cls : torch.Tensor
        Class probabilities after sigmoid.
    """
    shape = boxes[0].shape  # BCHW
    box = torch.cat(
        tuple(box.view(shape[0], boxes[0].shape[1], -1) for box in boxes), 2
    )
    cls = torch.cat(
        tuple(score.view(shape[0], scores[0].shape[1], -1) for score in scores), 2
    )

    # See detect_patches.patched_ultryaltics_det_head_inference for rationale.
    anchors, strides = (
        bb.transpose(0, 1) for bb in make_anchors(feats, self.stride, 0.5)
    )
    self.anchors, self.strides = anchors, strides

    dbox = self.decode_bboxes(self.dfl(box), anchors.unsqueeze(0)) * strides
    return dbox, cls.sigmoid()

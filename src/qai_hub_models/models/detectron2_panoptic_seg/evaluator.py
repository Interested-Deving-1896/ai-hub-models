# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torchvision
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.layers.mask_ops import paste_masks_in_image as d2_paste
from detectron2.model_zoo import get_config_file
from detectron2.structures import Boxes

from qai_hub_models.datasets.coco import CocoPanopticSegmentationDataset
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
from qai_hub_models.utils.input_spec import TensorSpec
from qai_hub_models.utils.metrics import (
    PANOPTIC_QUALITY,
    MetricMetadata,
)

# Post-processing thresholds shared by Detectron2PanopticSegApp (full pipeline)
# and Detectron2PanopticSegEvaluator (re-runs the same postprocessing off the
# raw model outputs during eval). Both must apply identical thresholds, or the
# evaluated PQ silently diverges from what the app predicts, so both default to
# these constants rather than to independent literals.
DEFAULT_BOXES_SCORE_THRESHOLD = 0.5
DEFAULT_BOXES_IOU_THRESHOLD = 0.5
DEFAULT_MAX_DET_POST_NMS = 100
DEFAULT_INSTANCES_SCORE_THRESH = 0.5
DEFAULT_OVERLAP_THRESHOLD = 0.5
DEFAULT_STUFF_AREA_THRESH = 4096


class PanopticSegment(NamedTuple):
    """One segment painted into a panoptic id map.

    Attributes
    ----------
    id
        Segment id painted into the panoptic map (1-based).
    is_thing
        True for instance (thing) segments, False for semantic (stuff)
        segments.
    source_index
        Instance index into the instance tensors for a thing segment, or the
        stuff class label for a stuff segment.
    area
        Number of pixels the segment occupies in the painted map.
    """

    id: int
    is_thing: bool
    source_index: int
    area: int


def select_instances(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    mask_logits: torch.Tensor,
    proposals: torch.Tensor,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select final detections from dense per-class ROI head outputs, matching
    detectron2's fast_rcnn_inference_single_image: keep every
    (proposal, class) pair above the score threshold, then per-class NMS.

    Parameters
    ----------
    boxes
        (num_proposals, num_box_classes, 4) per-class refined boxes.
    scores
        (num_proposals, num_classes) per-class probabilities.
    mask_logits
        (num_proposals, num_mask_classes, mask_h, mask_w) per-class mask
        logits, pooled on the proposal boxes.
    proposals
        (num_proposals, 4) proposal boxes the masks were pooled on.
    score_threshold
        Minimum class score to keep a (proposal, class) pair.
    nms_iou_threshold
        Per-class NMS IoU threshold.
    max_detections
        Maximum number of detections to keep.

    Returns
    -------
    torch.Tensor
        Selected boxes of shape (M, 4).
    torch.Tensor
        Selected scores of shape (M,).
    torch.Tensor
        Selected class ids of shape (M,).
    torch.Tensor
        Selected mask probabilities of shape (M, 1, mask_h, mask_w).
    torch.Tensor
        Paste boxes of shape (M, 4): the proposal boxes the masks were
        pooled on; masks must be pasted with these.
    """
    filter_mask = scores > score_threshold  # (N, C)
    inds = filter_mask.nonzero()
    prop_idx, cls_idx = inds[:, 0], inds[:, 1]
    box_cls = cls_idx.clamp(max=boxes.shape[1] - 1)
    sel_boxes = boxes[prop_idx, box_cls]
    sel_scores = scores[filter_mask]

    # torchvision.ops.batched_nms is used directly (not the repo's
    # bounding_box_processing.batched_nms) because the surviving keep indices
    # are needed to gather the matching mask logits and proposal boxes; the
    # repo helper returns filtered values without the indices.
    keep = torchvision.ops.batched_nms(
        sel_boxes.float(), sel_scores.float(), cls_idx, nms_iou_threshold
    )[:max_detections]
    prop_idx = prop_idx[keep]
    cls_idx = cls_idx[keep]

    mask_cls = cls_idx.clamp(max=mask_logits.shape[1] - 1)
    return (
        sel_boxes[keep],
        sel_scores[keep],
        cls_idx,
        mask_logits[prop_idx, mask_cls][:, None].sigmoid(),
        proposals[prop_idx],
    )


def paste_masks_in_image(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    image_h: int,
    image_w: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Paste fixed-resolution masks into an image using detectron2's method.

    Parameters
    ----------
    masks
        (N, 1, mask_h, mask_w) float in [0, 1].
    boxes
        (N, 4) xyxy pixel coords.
    image_h
        Output image height.
    image_w
        Output image width.
    threshold
        Binarisation threshold.

    Returns
    -------
    torch.Tensor
        Bool mask of shape (N, image_h, image_w).
    """
    if masks.shape[0] == 0:
        return masks.new_empty((0, image_h, image_w), dtype=torch.bool)
    return d2_paste(
        masks[:, 0],
        Boxes(boxes),
        (image_h, image_w),
        threshold=threshold,
    )


def build_panoptic_segments(
    instance_masks: torch.Tensor,
    instance_scores: torch.Tensor,
    semantic_map: torch.Tensor,
    instances_score_thresh: float,
    overlap_threshold: float,
    stuff_area_thresh: int,
) -> tuple[torch.Tensor, list[PanopticSegment]]:
    """
    Merge instance masks and a semantic map into one panoptic id map,
    following detectron2's combine_semantic_and_instance_outputs: instances
    are painted first (highest score first, skipping heavily overlapping
    ones), then stuff regions fill the remaining pixels.

    The caller maps ``source_index`` to category ids and builds the
    ``segments_info`` dicts it needs, so this shares the pixel-painting logic
    between the demo app and the evaluator without coupling to either's
    segment metadata schema.

    Parameters
    ----------
    instance_masks
        (N, H, W) bool instance masks, already pasted at the target
        resolution.
    instance_scores
        (N,) instance confidence scores.
    semantic_map
        (H, W) int semantic class labels (argmax of the stuff logits), where
        label 0 is the "things" placeholder and is ignored.
    instances_score_thresh
        Minimum instance score to paint a thing segment.
    overlap_threshold
        Skip an instance if this fraction of its mask is already occupied.
    stuff_area_thresh
        Minimum pixel area to paint a stuff segment.

    Returns
    -------
    torch.Tensor
        (H, W) int32 panoptic id map.
    list[PanopticSegment]
        One entry per painted segment, in paint order.
    """
    height, width = semantic_map.shape
    panoptic_seg = torch.zeros((height, width), dtype=torch.int32)
    segments: list[PanopticSegment] = []
    current_segment_id = 0

    # Instance (thing) segments, highest score first.
    for inst_id in torch.argsort(-instance_scores).tolist():
        if instance_scores[inst_id].item() < instances_score_thresh:
            break
        mask = instance_masks[inst_id]
        mask_area = int(mask.sum().item())
        if mask_area == 0:
            continue
        intersect_area = int(((mask > 0) & (panoptic_seg > 0)).sum().item())
        if intersect_area / mask_area > overlap_threshold:
            continue
        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)
            mask_area = int(mask.sum().item())
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments.append(
            PanopticSegment(current_segment_id, True, int(inst_id), mask_area)
        )

    # Semantic (stuff) segments fill the remaining pixels.
    for stuff_label in torch.unique(semantic_map).cpu().tolist():
        if stuff_label == 0:  # things placeholder
            continue
        mask = (semantic_map == stuff_label) & (panoptic_seg == 0)
        mask_area = int(mask.sum().item())
        if mask_area < stuff_area_thresh:
            continue
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments.append(
            PanopticSegment(current_segment_id, False, int(stuff_label), mask_area)
        )

    return panoptic_seg, segments


class Detectron2PanopticSegEvaluator(BaseEvaluator):
    """Evaluator for Detectron2 panoptic segmentation (PQ metric).

    Accepts the raw outputs of ``Detectron2PanopticROIHead`` together with the
    semantic segmentation logits from ``Detectron2PanopticProposalGenerator``
    and computes the COCO Panoptic Quality metric.

    Expected ``output`` tuple passed to :meth:`add_batch`:
        boxes        - (1, N, num_box_classes, 4) per-class refined boxes,
                       xyxy in model-image pixel space
        scores       - (1, N, num_classes) per-class probabilities
        mask_logits  - (N, num_mask_classes, mask_h, mask_w) per-class
                       instance mask logits, pooled on the proposal boxes
        proposals    - (N, 4) the proposal boxes fed to the ROI head
        sem_seg      - (1, num_sem_classes, H/4, W/4) stride-4 semantic logits
                       (upsampled to model resolution internally)

    Class selection, score filtering, and NMS run here (see
    select_instances); the model emits dense per-class outputs.

    Expected ``gt_data`` tuple:
        batched_gt_masks  - (B, H, W, 3) uint8 RGB panoptic masks
        batched_img_ids   - (B,) int64 COCO image ids
    """

    def __init__(
        self,
        model_image_height: int = 800,
        model_image_width: int = 800,
        score_threshold: float = DEFAULT_BOXES_SCORE_THRESHOLD,
        nms_iou_threshold: float = DEFAULT_BOXES_IOU_THRESHOLD,
        max_detections: int = DEFAULT_MAX_DET_POST_NMS,
        instances_score_thresh: float = DEFAULT_INSTANCES_SCORE_THRESH,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
        stuff_area_thresh: int = DEFAULT_STUFF_AREA_THRESH,
    ) -> None:
        super().__init__()
        self.model_image_height = model_image_height
        self.model_image_width = model_image_width
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.instances_score_thresh = instances_score_thresh
        self.overlap_threshold = overlap_threshold
        self.stuff_area_thresh = stuff_area_thresh

        self.label_divisor = 1000
        self._dataset: CocoPanopticSegmentationDataset | None = None
        self._mappings_initialized = False
        self.reset()

    def _init_mappings(self) -> None:
        """Lazily load COCO annotations and build id-conversion maps.

        Called on first add_batch() so that constructing the evaluator does not
        trigger a COCO download.
        """
        if self._mappings_initialized:
            return
        self._dataset = CocoPanopticSegmentationDataset(
            input_spec={
                "image": TensorSpec(
                    shape=(1, 3, self.model_image_height, self.model_image_width),
                    dtype="float32",
                )
            }
        )
        annotations = self._dataset.annotations
        # GT annotations use raw COCO category ids — keep them as-is.
        self.img_ann_map = {ann["image_id"]: ann for ann in annotations["annotations"]}
        # categories dict keyed by COCO id — used by pq_average
        self.categories = {el["id"]: el for el in annotations["categories"]}

        # Build id-conversion maps using detectron2 metadata
        cfg = get_cfg()
        cfg.merge_from_file(
            get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml")
        )
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        # ROI head outputs thing contiguous ids (0-79) -> COCO id
        self.thing_contiguous_to_coco: dict[int, int] = {
            v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }
        # sem_seg head outputs stuff contiguous ids (0-53) -> COCO id
        # index 0 is the "things" placeholder — skip it
        self.stuff_contiguous_to_coco: dict[int, int] = {
            v: k
            for k, v in metadata.stuff_dataset_id_to_contiguous_id.items()
            if v != 0  # skip things placeholder
        }

        # GT segments already use raw COCO category ids — no remapping needed
        self.gt_segments_by_img: dict[int, list[dict]] = {
            ann["image_id"]: ann.get("segments_info", [])
            for ann in annotations["annotations"]
        }
        self._mappings_initialized = True

    def reset(self) -> None:
        self.pq_stat = PQStat()

    def add_batch(
        self,
        output: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        gt_data: tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]],
    ) -> None:
        """Process model predictions and ground truth for panoptic segmentation evaluation.

        Parameters
        ----------
        output
            Model predictions: ROI head boxes, scores, and mask logits, the
            region proposals, and the semantic segmentation head results.
        gt_data
            Ground truth panoptic masks and image IDs, optionally followed by
            letterbox scale/pad metadata (unused here; CocoPanopticSegmentationDataset
            may return either a 2-tuple or a 4-tuple).
        """
        boxes, scores, mask_logits, proposals, sem_seg_results = output

        self._init_mappings()

        # gt_data may include scale/pad from resize_pad (letterbox) preprocessing
        if len(gt_data) == 4:
            batched_gt_masks, batched_img_ids, batched_scales, batched_pads = gt_data
        else:
            batched_gt_masks, batched_img_ids = gt_data
            batched_scales = None
            batched_pads = None

        batch_size = batched_img_ids.shape[0]
        batched_gt_masks_np = batched_gt_masks.cpu().numpy()
        for i in range(batch_size):
            img_id = int(batched_img_ids[i].item())
            gt_mask = rgb2id(batched_gt_masks_np[i])

            # If letterbox scale/pad provided, crop pad region from GT mask
            # so GT is at the same effective resolution as the model output
            scale = float(batched_scales[i]) if batched_scales is not None else None
            pad = (
                (int(batched_pads[i][0]), int(batched_pads[i][1]))
                if batched_pads is not None
                else None
            )
            if scale is not None and pad is not None:
                left_pad, top_pad = pad
                assert self._dataset is not None  # set by _init_mappings()
                orig_info = self._dataset.img_dict[img_id]
                orig_w, orig_h = orig_info["width"], orig_info["height"]
                # Must match resize_pad's math.floor rounding, or GT/pred are
                # cropped to different pixel dimensions.
                scaled_w = math.floor(orig_w * scale)
                scaled_h = math.floor(orig_h * scale)
                # Crop out the padded region — keep only the valid image area
                gt_mask = gt_mask[
                    top_pad : top_pad + scaled_h, left_pad : left_pad + scaled_w
                ]

            # The model emits dense per-class predictions with no on-device
            # class selection or NMS; select final detections here, matching
            # detectron2's fast_rcnn_inference_single_image.
            (
                _inst_boxes,
                inst_scores,
                inst_classes,
                inst_mask_probs,
                paste_boxes,
            ) = select_instances(
                boxes[i],
                scores[i],
                mask_logits[i],
                proposals[i],
                self.score_threshold,
                self.nms_iou_threshold,
                self.max_detections,
            )

            # Clamp paste boxes to image bounds
            paste_boxes = paste_boxes.clone()
            paste_boxes[:, 0::2] = paste_boxes[:, 0::2].clamp(
                0, self.model_image_width - 1
            )
            paste_boxes[:, 1::2] = paste_boxes[:, 1::2].clamp(
                0, self.model_image_height - 1
            )

            # Paste instance masks with the boxes they were pooled on
            inst_masks = paste_masks_in_image(
                inst_mask_probs,
                paste_boxes,
                self.model_image_height,
                self.model_image_width,
            )  # (M, H, W) bool

            # Semantic segmentation logits are emitted at stride 4; upsample
            # to model resolution, then argmax -> stuff contiguous ids (0-53)
            sem_logits = torch.nn.functional.interpolate(
                sem_seg_results[i : i + 1],
                size=(self.model_image_height, self.model_image_width),
                mode="bilinear",
                align_corners=False,
            )[0]
            sem_result = sem_logits.argmax(dim=0)  # (H, W)

            # Build panoptic map following detectron2 merge logic, then map
            # each segment's contiguous id to its raw COCO category id.
            panoptic_seg, segments = build_panoptic_segments(
                inst_masks,
                inst_scores,
                sem_result,
                self.instances_score_thresh,
                self.overlap_threshold,
                self.stuff_area_thresh,
            )
            segments_info: list[dict] = []
            for segment in segments:
                if segment.is_thing:
                    thing_contiguous = int(inst_classes[segment.source_index].item())
                    coco_cat_id = self.thing_contiguous_to_coco.get(thing_contiguous)
                    if coco_cat_id is None:
                        continue
                    segments_info.append(
                        {
                            "id": segment.id,
                            "category_id": coco_cat_id,  # raw COCO id
                            "iscrowd": 0,
                            "score": inst_scores[segment.source_index].item(),
                        }
                    )
                else:
                    coco_cat_id = self.stuff_contiguous_to_coco.get(
                        segment.source_index
                    )
                    if coco_cat_id is None:
                        continue
                    segments_info.append(
                        {
                            "id": segment.id,
                            "category_id": coco_cat_id,  # raw COCO id
                            "iscrowd": 0,
                            "score": 1.0,
                        }
                    )

            # Crop pred panoptic map to valid (unpadded) region to match GT
            pan_np = panoptic_seg.cpu().numpy()
            if scale is not None and pad is not None:
                pan_np = pan_np[
                    top_pad : top_pad + scaled_h, left_pad : left_pad + scaled_w
                ]

            seg_id_to_area = {
                int(sid): int(cnt)
                for sid, cnt in zip(
                    *np.unique(pan_np, return_counts=True), strict=False
                )
                if sid != 0
            }
            for seg in segments_info:
                seg["area"] = seg_id_to_area.get(seg["id"], 0)

            gt_segments = self.gt_segments_by_img.get(img_id, [])
            pq_stat = pq_compute_single_image(
                gt_mask,
                pan_np,
                gt_segments,
                segments_info,
                label_divisor=self.label_divisor,
            )
            self.pq_stat += pq_stat

    def compute_pq(self) -> dict[str, dict]:
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        return {
            name: self.pq_stat.pq_average(self.categories, isthing=isthing)[0]
            for name, isthing in metrics
        }

    def get_accuracy_score(self) -> float:
        return self.compute_pq()["All"]["pq"]

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score() * 100:.1f} PQ"

    def get_metric_metadata(self) -> MetricMetadata:
        return PANOPTIC_QUALITY

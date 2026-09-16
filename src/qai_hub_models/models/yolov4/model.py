# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.templates.yolo.model import Yolo
from qai_hub_models.models.templates.yolo.utils import (
    detect_postprocess_split_input,
    make_grid,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

from .external_repos.yolov4.nets.yolo import YoloBody

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_HEIGHT = 416
DEFAULT_WIDTH = 416
ANCHORS_MASK = [[3, 4, 5], [1, 2, 3]]
NUM_OF_CLASSES = 80
# Reference: https://github.com/bubbliiiing/yolov4-tiny-pytorch/blob/master/model_data/yolo_anchors.txt
ANCHORS = (
    (10.0, 14.0),
    (23.0, 27.0),
    (37.0, 58.0),
    (81.0, 82.0),
    (135.0, 169.0),
    (344.0, 319.0),
)

# Weights taken from "https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_coco.pth"
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov4_tiny_weights_coco.pth"
)


class _YoloV4TinyDecoder(nn.Module):
    def __init__(self, num_classes: int = NUM_OF_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("anchors", torch.tensor(ANCHORS))

    def forward(
        self, outputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode the two YOLOv4-tiny detection heads into QAI Hub tensors.
        The upstream implementation in ``yolo.py`` performs equivalent anchor
        decoding but continues immediately to its framework-specific NMS
        routine. This implementation separates decoding from postprocessing so
        the model can expose stable, exportable tensors and reuse the shared
        QAI Hub YOLO postprocessor in ``YoloV4Tiny.forward``.

        Parameters
        ----------
        outputs
            The two raw detection-head outputs from ``YoloBody``. Each tensor
            has shape ``(batch, 3 * (5 + num_classes), height, width)`` and
            contains three anchors per grid cell. The first output is the
            coarse feature map using anchors ``[3, 4, 5]``; the second is the
            fine feature map using anchors ``[1, 2, 3]``.

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            xy
                Center coordinates. Shape ``(batch, num_predictions, 2)``.
            wh
                Width and height. Shape ``(batch, num_predictions, 2)``.
            scores
                Objectness and class scores. Shape ``(batch, num_predictions, 1 + num_classes)``.


        """
        xy_outputs: list[torch.Tensor] = []
        wh_outputs: list[torch.Tensor] = []
        score_outputs: list[torch.Tensor] = []
        for output, mask in zip(outputs, ((3, 4, 5), (1, 2, 3)), strict=False):
            batch, _, height, width = output.shape
            prediction = output.reshape(
                batch, 3, 5 + self.num_classes, height, width
            ).permute(0, 1, 3, 4, 2)
            grid = make_grid(width, height, device=output.device, dtype=output.dtype)
            stride = output.new_tensor((DEFAULT_WIDTH / width, DEFAULT_HEIGHT / height))
            # This keeps the upstream anchor masks and decode equations, but
            # returns decoded tensors instead
            # of running upstream NMS here.
            anchor_wh = self.anchors[list(mask)].to(output.dtype).reshape(1, 3, 1, 1, 2)  # type: ignore[index]
            xy = (prediction[..., 0:2].sigmoid() + grid) * stride
            wh = prediction[..., 2:4].exp() * anchor_wh
            scores = torch.cat(
                (prediction[..., 4:5].sigmoid(), prediction[..., 5:].sigmoid()), dim=-1
            )
            xy_outputs.append(xy.reshape(batch, -1, 2))
            wh_outputs.append(wh.reshape(batch, -1, 2))
            score_outputs.append(scores.reshape(batch, -1, 1 + self.num_classes))
        return (
            torch.cat(xy_outputs, dim=1),
            torch.cat(wh_outputs, dim=1),
            torch.cat(score_outputs, dim=1),
        )


class YoloV4Tiny(Yolo):
    """Exportable YoloV4-tiny detector using the bubbliiiing source model."""

    def __init__(
        self,
        model: nn.Module,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        super().__init__(model)
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        self.decoder = _YoloV4TinyDecoder(NUM_OF_CLASSES)

    @classmethod
    def from_pretrained(
        cls,
        weights: CachedWebModelAsset | str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Self:
        weights_path = str(
            weights.fetch() if isinstance(weights, CachedWebModelAsset) else weights
        )
        net = YoloBody(ANCHORS_MASK, NUM_OF_CLASSES)
        net.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        return cls(net, include_postprocessing, split_output)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        return super().get_input_spec(batch_size, height, width)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Run YoloV4-tiny on ``image`` and produce predicted bounding boxes and class probabilities.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor
            If ``self.include_postprocessing`` is True, returns:
            boxes
                Bounding box locations. Shape ``[batch, num_preds, 4]`` where
                4 is ``(left_x, top_y, right_x, bottom_y)``.
            scores
                Class scores multiplied by confidence. Shape ``[batch, num_preds]``.
            class_idx
                Shape ``[batch, num_preds]`` containing the index of the most
                probable class for each prediction.

            If ``self.include_postprocessing`` is False and ``self.split_output``
            is True, returns:
            boxes_xy
                Shape ``[batch, num_preds, 2]`` containing ``[x_center, y_center]``.
            boxes_wh
                Shape ``[batch, num_preds, 2]`` containing ``[width, height]``.
            scores
                Shape ``[batch, num_preds, 81]``: index 0 is objectness and
                indices 1 through 80 are the 80 class scores.

            If ``self.include_postprocessing`` is False and ``self.split_output``
            is False, returns:
            detector_output
                Shape ``[batch, num_preds, 85]`` structured as ``[x_center,
                y_center, width, height, objectness, class_scores]``.
        """
        xy, wh, scores = self.decoder(self.model(image))
        if self.include_postprocessing:
            return detect_postprocess_split_input(xy, wh, scores)
        if self.split_output:
            return xy, wh, scores
        return torch.cat((xy, wh, scores), dim=-1)

    def get_output_spec(self) -> OutputSpec:
        if self.include_postprocessing:
            return {
                "boxes": TensorSpec(),
                "scores": TensorSpec(),
                "class_idx": TensorSpec(),
            }
        if self.split_output:
            return {
                "boxes_xy": TensorSpec(),
                "boxes_wh": TensorSpec(),
                "scores": TensorSpec(),
            }
        return {"detector_output": TensorSpec()}

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.models.yolov5_face.dataset import WiderFaceDataset
from qai_hub_models.models.yolov5_face.evaluator import (
    YoloV5FaceEvaluator,
)
from qai_hub_models.models.yolov5_face.external_repos.yolov5_face.models.experimental import (
    attempt_load,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


SUPPORTED_WEIGHTS = [
    "yolov5n-0.5.pt",
    "yolov5n-face.pt",
    "yolov5s-face.pt",
    "yolov5m-face.pt",
]

# Weights sourced from https://github.com/deepcam-cn/yolov5-face/tree/master#pretrained-models
DEFAULT_WEIGHTS = "yolov5n-face.pt"

INPUT_IMAGE_SIZE = 640


class YoloV5Face(BaseModel):
    """Exportable YoloV5-Face face detector with 5-point facial landmark estimation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        height: int = INPUT_IMAGE_SIZE,
        width: int = INPUT_IMAGE_SIZE,
    ) -> Self:
        local = Path(ckpt_name)
        if local.exists():
            # Local path supplied directly — skip asset-store lookup.
            weights_path = str(local)
        else:
            if ckpt_name not in SUPPORTED_WEIGHTS:
                raise ValueError(
                    f"Unsupported checkpoint: {ckpt_name!r}. "
                    f"Supported: {SUPPORTED_WEIGHTS}"
                )
            weights_path = str(
                CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, ckpt_name
                ).fetch()
            )

        # attempt_load handles Ensemble construction, torch.load, fuse/eval,
        # and compatibility updates (inplace activations, _non_persistent_buffers_set).
        source_model = attempt_load(weights_path, map_location="cpu")

        # torch.load restores anchor_grid from the checkpoint state dict as a
        # rank-6 buffer whose static shape breaks tracing. Replace it with a
        # plain list post-load so _make_grid_new can fill it on the dry run.
        detect = source_model.model[-1]
        delattr(detect, "anchor_grid")
        detect.anchor_grid = [torch.zeros(1)] * detect.nl

        # Dry run — pre-warms self.grid[i] so the shape-mutating conditional
        # in Detect.forward is False during tracing.
        dummy = torch.zeros(1, 3, height, width)
        with torch.no_grad():
            source_model(dummy)

        return cls(source_model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run YoloV5-Face inference.

        Parameters
        ----------
        image
            (1, 3, 640, 640) float32 RGB, values in [0, 1].

        Returns
        -------
        boxes : torch.Tensor
            Shape (1, 25200, 4) - decoded [cx, cy, w, h] in 640x640 pixel space.
        scores : torch.Tensor
            Shape (1, 25200, 1) - obj_conf * cls_conf.
        landmarks : torch.Tensor
            Shape (1, 25200, 10) - decoded [lx1,ly1,...,lx5,ly5] in 640x640 pixel space.
        """
        raw = self.model(image)
        boxes = raw[..., :4]  # cx, cy, w, h
        scores = (
            raw[..., 4:5] * raw[..., 15:16]
        )  # obj_conf * cls_conf (matches upstream NMS)
        landmarks = raw[..., 5:15]  # 5 x (x, y)
        return boxes, scores, landmarks

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = INPUT_IMAGE_SIZE,
        width: int = INPUT_IMAGE_SIZE,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "boxes": TensorSpec(),
            "scores": TensorSpec(),
            "landmarks": TensorSpec(),
        }

    def get_evaluator(self) -> BaseEvaluator:
        return YoloV5FaceEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [WiderFaceDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return WiderFaceDataset

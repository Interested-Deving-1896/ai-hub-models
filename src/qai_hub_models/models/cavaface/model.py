# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.models.cavaface.external_repos.cavaface.backbone.resnet_irse import (
    IR_SE_100,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
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
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "IR_SE_100"
# Plain state_dict converted from the originally-published scripted checkpoint
# (see convert_checkpoint.py), so the eager Backbone loads directly.
DEFAULT_WEIGHTS_FILE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "IR_SE_100_state_dict.pth"
)


class CavaFace(BaseModel):
    def __init__(self, model: torch.nn.Module | None = None) -> None:
        super().__init__(model=model)

    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS) -> Self:
        if weights_name != DEFAULT_WEIGHTS:
            raise NotImplementedError("Unsupported weights")
        net = IR_SE_100((112, 112))
        net.load_state_dict(load_torch(DEFAULT_WEIGHTS_FILE))
        return cls(net).eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for face embeddings.

        Parameters
        ----------
        image
            RGB image of range[0, 1] and shape [batch_size, 3, H, W].

        Returns
        -------
        embeddings : torch.Tensor
            Normalized face embeddings tensor of shape [batch_size, 512].
        """
        # Normalize input [-1, 1]
        image = (image * 255 - 127.5) / 128.0

        # Get raw embeddings
        embeddings = self.model(image)

        # Normalize embeddings to unit length
        norm = torch.norm(embeddings, dim=1, keepdim=True) + 1e-9
        return embeddings / norm

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 112,
        width: int = 112,
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
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "embeddings": TensorSpec(),
        }

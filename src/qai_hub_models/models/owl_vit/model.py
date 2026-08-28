# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import OwlViTForObjectDetection
from typing_extensions import Self

from qai_hub_models.models.templates.owl.model import Owl
from qai_hub_models.models.templates.owl.model_patches import apply_patches
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    InputSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
HF_MODEL_ID = "google/owlvit-base-patch32"

# OwlViT-base-patch32 constants
IMAGE_SIZE = 768
PATCH_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 576
TEXT_SEQ_LEN = 16


class OwlViT(Owl):
    """OwlViT open-vocabulary object detector."""

    def __init__(self, model: OwlViTForObjectDetection) -> None:
        super().__init__(model)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pixel_values
            Preprocessed image. Shape: [B, 3, 768, 768]
            Range: float[0, 1]
            3-channel Color Space: RGB
        input_ids
            Tokenized text queries. Shape: [B, 16]
        attention_mask
            Attention mask for text queries. Shape: [B, 16]

        Returns
        -------
        boxes : torch.Tensor
            Shape (B, 576, 4) representing the bounding box coordinates (x1, y1, x2, y2) in pixel space.
        scores : torch.Tensor
            Shape (B, 576) representing the confidence scores.
        labels : torch.Tensor
            Shape (B, 576) representing the class labels.
        """
        boxes, scores, labels = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
        )
        _, _, h, w = pixel_values.shape

        boxes *= torch.tensor([w, h, w, h])

        # Cast output tensors to float32 as supported by Qualcomm AI Hub
        boxes = boxes.to(torch.float32)
        scores = scores.to(torch.float32)
        labels = labels.to(torch.int32)

        return boxes, scores, labels

    def get_input_spec(
        self,
        batch_size: int = 1,
        image_height: int = IMAGE_SIZE,
        image_width: int = IMAGE_SIZE,
        text_seq_len: int = TEXT_SEQ_LEN,
    ) -> InputSpec:
        return Owl._get_input_spec(batch_size, image_height, image_width, text_seq_len)

    @classmethod
    def from_pretrained(cls) -> Self:
        """Load the pretrained google/owlvit-base-patch32 checkpoint."""
        apply_patches()
        model = OwlViTForObjectDetection.from_pretrained(
            HF_MODEL_ID,
            attn_implementation="eager",
        )
        model.eval()
        return cls(model)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [cls.get_dataset_class(HF_MODEL_ID, Owl._get_input_spec(), False)]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return self.__class__.get_dataset_class(
            HF_MODEL_ID, self.get_input_spec(), True
        )

    def get_evaluator(self) -> BaseEvaluator:
        return self.detection_evaluator(self.get_input_spec())

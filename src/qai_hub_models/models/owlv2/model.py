# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import TypeVar, cast

import torch
from qai_hub.client import Device
from transformers import Owlv2ForObjectDetection
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.owlv2.model_patches import apply_patches, prepare_conv
from qai_hub_models.models.templates.owl.model import (
    Owl,
    owl_postprocess,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
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
HF_MODEL_ID = "google/owlv2-base-patch16-ensemble"

# OwlV2-base-patch16 constants
IMAGE_SIZE = 960
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 3600
TEXT_SEQ_LEN = 16
VISION_HIDDEN_DIM = 768  # OwlV2-base vision hidden size


class OwlV2VisionEncoder(BaseModel):
    """OwlV2 vision encoder component.

    Takes pixel_values and produces image feature embeddings (feature map).
    """

    def __init__(self, model: Owlv2ForObjectDetection) -> None:
        super().__init__(model)
        self.model: Owlv2ForObjectDetection = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values
            Preprocessed image. Shape: [B, 3, 960, 960]
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        image_embeds : torch.Tensor
            Image feature map. Shape: [B, num_patches_h, num_patches_w, hidden_dim]
        """
        hf_model = self.model
        image_embeds, _ = hf_model.image_embedder(  # type: ignore[misc]
            pixel_values=pixel_values,  # type: ignore[arg-type]
            interpolate_pos_encoding=True,
        )
        return image_embeds

    @staticmethod
    def _get_input_spec(
        batch_size: int = 1,
        image_height: int = IMAGE_SIZE,
        image_width: int = IMAGE_SIZE,
    ) -> InputSpec:
        return {
            "pixel_values": TensorSpec(
                shape=(batch_size, 3, image_height, image_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_input_spec(
        self,
        batch_size: int = 1,
        image_height: int = IMAGE_SIZE,
        image_width: int = IMAGE_SIZE,
    ) -> InputSpec:
        return OwlV2VisionEncoder._get_input_spec(
            batch_size=batch_size, image_height=image_height, image_width=image_width
        )

    def get_output_spec(self) -> OutputSpec:
        return {
            "image_embeds": TensorSpec(),
        }

    @classmethod
    def from_pretrained(cls) -> Self:
        return OwlV2Loader.load()[1]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return Owl._get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
            self,
        )

    def component_precision(self) -> Precision:
        return Precision.float


class OwlV2TextDetector(BaseModel):
    """OwlV2 text encoder + detection heads component.

    Takes text queries and image embeddings from the vision encoder,
    and produces object detection results (boxes, scores, labels).
    """

    def __init__(self, model: Owlv2ForObjectDetection) -> None:
        super().__init__(model)
        hf_model = cast(Owlv2ForObjectDetection, self.model)
        self.text_model = hf_model.owlv2.text_model
        self.text_projection = hf_model.owlv2.text_projection
        self.class_predictor = hf_model.class_predictor
        self.box_predictor = hf_model.box_predictor

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids
            Tokenized text queries. Shape: [B, L]
        attention_mask
            Attention mask for text queries. Shape: [B, L]
        image_embeds
            Image feature map from vision encoder.
            Shape: [B, num_patches_h, num_patches_w, hidden_dim]

        Returns
        -------
        boxes : torch.Tensor
            Shape [B, num_patches, 4] bounding box coordinates (x1, y1, x2, y2) in pixel space.
        scores : torch.Tensor
            Shape [B, num_patches] confidence scores.
        labels: torch.Tensor
            Shape [B, num_patches] class labels.
        """
        batch_size = image_embeds.shape[0]
        num_patches_h = image_embeds.shape[1]
        num_patches_w = image_embeds.shape[2]
        hidden_dim = image_embeds.shape[3]

        # --- Text encoding ---
        text_outputs = cast(torch.nn.Module, self.text_model)(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            return_dict=True,
        )
        text_embeds = text_outputs[1]
        # Project to shared embedding space and L2-normalise
        text_embeds = cast(torch.nn.Module, self.text_projection)(text_embeds)
        text_embeds = text_embeds / (
            torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        )

        # --- Reshape image features ---
        image_feats = image_embeds.reshape(
            batch_size, num_patches_h * num_patches_w, hidden_dim
        )

        # --- Reshape text embeddings ---
        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = text_embeds.reshape(
            batch_size, max_text_queries, text_embeds.shape[-1]
        )

        # Query mask: first token == 0 means padded query
        input_ids_reshaped = input_ids.reshape(
            batch_size, max_text_queries, input_ids.shape[-1]
        )
        query_mask = input_ids_reshaped[..., 0] > 0

        # --- Detection heads ---
        pred_logits, _ = cast(torch.nn.Module, self.class_predictor)(
            image_feats, query_embeds, query_mask
        )

        # Box prediction: [B, num_patches, 4] in (cx, cy, w, h) normalised [0, 1]
        pred_boxes = cast(torch.nn.Module, self.box_predictor)(
            image_feats, image_embeds, interpolate_pos_encoding=True
        )

        # --- Post-processing ---
        # Convert (cx, cy, w, h) → (x1, y1, x2, y2), extract scores and labels
        boxes, scores, labels = owl_postprocess(pred_logits, pred_boxes)

        # Scale normalised boxes to pixel coordinates
        max_size = max(num_patches_h, num_patches_w) * PATCH_SIZE
        boxes *= torch.tensor([max_size, max_size, max_size, max_size])

        # Cast output tensors to types supported by Qualcomm AI Hub
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
        num_patches_h = image_height // PATCH_SIZE
        num_patches_w = image_width // PATCH_SIZE
        return {
            "input_ids": TensorSpec(
                shape=(batch_size, text_seq_len),
                dtype="int32",
            ),
            "attention_mask": TensorSpec(
                shape=(batch_size, text_seq_len),
                dtype="int32",
            ),
            "image_embeds": TensorSpec(
                shape=(batch_size, num_patches_h, num_patches_w, VISION_HIDDEN_DIM),
                dtype="float32",
            ),
        }

    def component_precision(self) -> Precision:
        return Precision.w8a16

    def get_output_spec(self) -> OutputSpec:
        return Owl._get_output_spec()

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return Owl._get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
            self,
        )

    @classmethod
    def from_pretrained(cls) -> Self:
        return OwlV2Loader.load()[2]


visionT = TypeVar("visionT", bound=OwlV2VisionEncoder)
textT = TypeVar("textT", bound=OwlV2TextDetector)


class OwlV2Loader:
    """Helper class for loading and preparing a HTP-compatible OwlV2 model.

    Loads the HuggingFace checkpoint once and shares it between both
    the vision encoder and text detector components.
    """

    @classmethod
    def load(
        cls,
        encoder_cls: type[visionT] = OwlV2VisionEncoder,  # type: ignore[assignment]
        decoder_cls: type[textT] = OwlV2TextDetector,  # type: ignore[assignment]
    ) -> tuple[Owlv2ForObjectDetection, visionT, textT]:
        apply_patches()
        model = Owlv2ForObjectDetection.from_pretrained(
            HF_MODEL_ID,
            attn_implementation="eager",
        )
        model.eval()
        prepare_conv(model)
        vision_encoder = encoder_cls(model)
        text_detector = decoder_cls(model)
        return model, vision_encoder, text_detector


class OwlV2(WorkbenchModelCollection):
    """OwlV2 collection model with separate vision encoder and text detector components."""

    def __init__(
        self,
        model: Owlv2ForObjectDetection,
        vision_encoder: OwlV2VisionEncoder,
        text_detector: OwlV2TextDetector,
    ) -> None:
        super().__init__({"vision": vision_encoder, "text": text_detector})
        self.model = model
        self.vision_encoder = vision_encoder
        self.text_detector = text_detector

    @classmethod
    def from_pretrained(cls) -> Self:
        """Load the pretrained OwlV2 collection model."""
        return cls(*OwlV2Loader.load())

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
    ) -> ComponentGroup[InputSpec]:
        return ComponentGroup(
            {
                "vision": self.vision_encoder.get_input_spec(
                    batch_size=batch_size, image_height=height, image_width=width
                ),
                "text": self.text_detector.get_input_spec(
                    batch_size=batch_size, image_height=height, image_width=width
                ),
            }
        )

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Owl.get_dataset_class(
            HF_MODEL_ID, self.vision_encoder.get_input_spec(), True
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [
            Owl.get_dataset_class(
                HF_MODEL_ID, OwlV2VisionEncoder._get_input_spec(), True
            )
        ]

    def get_evaluator(self) -> BaseEvaluator:
        return Owl.detection_evaluator(self.vision_encoder.get_input_spec())

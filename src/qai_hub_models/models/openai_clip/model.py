# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator

import clip
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch import Tensor
from typing_extensions import Self

from qai_hub_models.datasets.cococaptions.cococaptions import (
    CAPTIONS_PER_IMAGE,
    CocoCaptionsDataset,
)
from qai_hub_models.models.openai_clip.evaluator import CLIPRetrievalEvaluator
from qai_hub_models.models.openai_clip.model_patches import (
    _patched_build_attention_mask,
)
from qai_hub_models.utils.asset_loaders import callback_with_retry
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

PRETRAINED_WEIGHTS = "ViT-B/16"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CONTEXT_LENGTH = 77

_CLIP_NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_clip_image(
    image_tensor: torch.Tensor,
    image_tensor_has_batch: bool = True,
) -> torch.Tensor:
    """
    Normalizes an image tensor using CLIP's channel mean and std constants.

    Parameters
    ----------
    image_tensor
        Image tensor to normalize, values in [0.0, 1.0].
    image_tensor_has_batch
        Whether the tensor has a leading batch dimension.

    Returns
    -------
    normalized_tensor : torch.Tensor
        Normalized image tensor.
    """
    shape = [-1, 1, 1]
    if image_tensor_has_batch:
        shape.insert(0, 1)
    mean = torch.tensor(_CLIP_NORMALIZE_MEAN).reshape(*shape)
    std = torch.tensor(_CLIP_NORMALIZE_STD).reshape(*shape)
    return (image_tensor - mean) / std


class OpenAIClip(BaseModel):
    def __init__(
        self,
        clip: CLIP,
        text_tokenizer: Callable[[str], torch.Tensor],
    ) -> None:
        """Wrapper for OpenAI CLIP."""
        super().__init__()
        self.clip = clip
        self.eot_token = 49407  # Token ID of CLIP's vocab-end token
        self.text_tokenizer = text_tokenizer

    def forward(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward call on Open AI CLIP model.

        Parameters
        ----------
        image
            Raw image tensor, values in [0.0, 1.0] (ToTensor output, no normalization applied).
            Shape: [1, 3, 224, 224]
            Channel Layout: RGB
        text
            Tokenized text tensor.
            Shape: [1, captions_per_image, 77]

        Returns
        -------
        image_features : torch.Tensor
            L2-normalized image embeddings. Shape: [1, D]
        text_features : torch.Tensor
            L2-normalized text embeddings. Shape: [1 * num_text_prompts, D]
        """
        with patched_in_projection_packed():
            image = normalize_clip_image(image).to(next(self.clip.parameters()).device)
            # Flatten [1, captions_per_image, 77] -> [1*captions_per_image , 77]
            text = torch.flatten(text, start_dim=0, end_dim=1)
            clipped_text = torch.clip(text, min=0, max=self.eot_token).to(
                next(self.clip.parameters()).device
            )
            text_features = self.clip.encode_text(clipped_text)
            # text_features: torch.Tensor [num_text_prompts,512 (transformer_width)]
            # Raw text features.
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            image_features = self.clip.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features, text_features

    def get_input_spec(
        self,
        image_batch_size: int = 1,
        image_height: int = DEFAULT_IMAGE_SIZE,
        image_width: int = DEFAULT_IMAGE_SIZE,
        captions_per_image: int = CAPTIONS_PER_IMAGE,
        text_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "image": TensorSpec(
                shape=(image_batch_size, 3, image_height, image_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
            "text": TensorSpec(
                shape=(image_batch_size, captions_per_image, text_length),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "image_features": TensorSpec(),
            "text_features": TensorSpec(),
        }

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoCaptionsDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoCaptionsDataset

    def get_evaluator(self) -> BaseEvaluator:
        """Returns a CLIPRetrievalEvaluator measuring Recall@K on COCO Captions."""
        return CLIPRetrievalEvaluator()

    @classmethod
    def from_pretrained(cls) -> Self:
        def load_clip() -> tuple[CLIP, Callable]:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            CLIP.build_attention_mask = _patched_build_attention_mask
            tokenizer = clip.tokenize
            net, _ = clip.load(PRETRAINED_WEIGHTS, device=device)
            return net, tokenizer

        net, tokenizer = callback_with_retry(num_retries=5, callback=load_clip)
        assert isinstance(net, CLIP)
        return cls(net, tokenizer)


@contextlib.contextmanager
def patched_in_projection_packed() -> Generator[None]:
    """
    Avoid unflatten that causes ONNX export failure.
    https://github.com/pytorch/pytorch/issues/135764
    """
    original_in_projection_packed = torch.nn.functional._in_projection_packed  # type: ignore[attr-defined]

    def patched_in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        E = q.size(-1)
        if k is v and q is k:
            proj = F.linear(q, w, b)
            proj = proj.view(*proj.shape[:-1], 3, E).permute((2, 0, 1, 3)).contiguous()
            return proj[0], proj[1], proj[2]
        return original_in_projection_packed(q, k, v, w, b)

    torch.nn.functional._in_projection_packed = patched_in_projection_packed  # type: ignore[attr-defined]

    try:
        yield
    finally:
        torch.nn.functional._in_projection_packed = original_in_projection_packed  # type: ignore[attr-defined]

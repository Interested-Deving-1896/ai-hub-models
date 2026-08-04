# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from PIL.Image import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from qai_hub_models.models.openai_clip.model import DEFAULT_IMAGE_SIZE
from qai_hub_models.utils.input_spec import InputSpec

_DEFAULT_IMAGE_PREPROCESSOR = Compose(
    [
        Resize(DEFAULT_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(DEFAULT_IMAGE_SIZE),
        lambda img: img.convert("RGB"),
        ToTensor(),
    ]
)


class ClipApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Clip.

    The app uses 1 model:
        * Clip

    For a given image input, the app will:
        * pre-process the image
        * pre-process the text
        * Run Clip inference
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        text_tokenizer: Callable[[str], torch.Tensor],
        input_spec: InputSpec,
        logit_scale: float = 100.0,
    ) -> None:
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.image_preprocessor = _DEFAULT_IMAGE_PREPROCESSOR
        self._captions_per_image: int = input_spec["text"][0][1]

        # logit_scale: Temperature scale applied to cosine similarities. Hardcoded to 100.0
        # because CLIP training clamps the learnable scale to this maximum, so
        # released weights always saturate the cap.
        # Reference: https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py#L367

        self._logit_scale = logit_scale

    def predict(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # See predict_best_caption.
        return self.predict_best_caption(*args, **kwargs)

    def predict_best_caption(
        self,
        image: Image,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """
        Given one image and a list of text prompts, return similarity scores.

        Passes image [1, 3, H, W] and text [1, len(texts), 77] to the model —
        matching the on-device compiled input spec (B=1, captions_per_image=5).

        Parameters
        ----------
        image
            PIL Image
        texts
            Text prompt candidates.

        Returns
        -------
        similarities : torch.Tensor
            Shape [1, len(texts)]. Cosine similarity x logit_scale for each prompt.
        """
        img = self.image_preprocessor(image).unsqueeze(0)

        if len(texts) != self._captions_per_image:
            raise ValueError(
                f"Expected {self._captions_per_image} text prompts (from input spec), "
                f"got {len(texts)}."
            )

        tokenized = torch.cat([self.text_tokenizer(t) for t in texts])
        text = tokenized.unsqueeze(0)

        image_features, text_features = self.model(img, text)
        return (self._logit_scale * image_features) @ text_features.t()

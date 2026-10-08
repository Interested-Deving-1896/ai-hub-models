# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import torch

from qai_hub_models.models.vjepa2.model import FRAMES, IMG_SIZE


class VJEPA2App:
    """
    Wraps a V-JEPA 2 encoder to turn a preprocessed video clip into patch-token
    embeddings.

    Preprocessing (short-side resize + center-crop to ``[3, T, H, W]``) is handled
    by ``dataset._preprocess_vjepa2``, which is the single shared path for both demo
    and eval.  ImageNet normalization is performed inside the model.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        num_frames: int = FRAMES,
        image_size: int = IMG_SIZE,
    ) -> None:
        self.model = model
        self.num_frames = num_frames
        self.image_size = image_size

    def preprocess_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Prepare a preprocessed clip for model input.

        Parameters
        ----------
        clip
            Shape ``[3, T, H, W]``, pixel values in [0, 1], RGB.
            Must already be resized and center-cropped via
            ``dataset._preprocess_vjepa2``.

        Returns
        -------
        video : torch.Tensor
            Shape ``[1, 3, T, H, W]``.
        """
        if clip.dim() != 4 or clip.shape[0] != 3:
            raise ValueError(
                f"Expected clip of shape [3, T, H, W], got {tuple(clip.shape)}. "
                "Run dataset._preprocess_vjepa2 before calling this method."
            )
        return clip.unsqueeze(0)  # [1, 3, T, H, W]

    def extract_features(self, clip: torch.Tensor) -> torch.Tensor:
        """Return per-token embeddings ``[1, num_tokens, hidden]`` for a preprocessed clip."""
        video = self.preprocess_clip(clip)
        with torch.no_grad():
            return self.model(video)

    def predict(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        # Alias so the app matches the common QAIHM app interface.
        return self.extract_features(*args, **kwargs)

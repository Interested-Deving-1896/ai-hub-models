# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.input_spec import InputSpec


class SuperPointApp:
    """End-to-end SuperPoint keypoint detection application.

    Accepts a PIL image (RGB or grayscale), runs SuperPoint, and returns
    keypoints, scores, and descriptors as numpy arrays.
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        input_spec: InputSpec | None = None,
    ) -> None:
        self.model = model
        spec = input_spec if input_spec is not None else model.get_input_spec()  # type: ignore[attr-defined]
        _, _, h, w = spec["image"][0]
        self.input_height: int = h
        self.input_width: int = w

    def predict(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SuperPoint on *image*.

        Parameters
        ----------
        image
            PIL image (RGB or grayscale).
        raw_output
            If True, return padded tensors including zero-padding.

        Returns
        -------
        keypoints : np.ndarray
            Shape (N, 2) with (x, y) pixel coordinates in the original image space.
        scores : np.ndarray
            Shape (N,) detection confidence scores.
        descriptors : np.ndarray
            Shape (N, 256) L2-normalised feature descriptors.
        """
        orig_w, orig_h = image.size
        gray = image.convert("L").resize(
            (self.input_width, self.input_height), Image.BILINEAR
        )
        tensor = preprocess_PIL_image(gray)  # (1, 1, H, W)

        kp, sc, desc, num_kp = self.model(tensor)

        n = int(num_kp[0].item())
        if raw_output:
            return (
                kp[0].detach().numpy(),
                sc[0].detach().numpy(),
                desc[0].detach().numpy(),
            )

        kp_np = kp[0, :n].detach().numpy()  # (N, 2) in model-resolution coords
        # Scale keypoints from model resolution back to original image resolution.
        kp_np[:, 0] *= orig_w / self.input_width
        kp_np[:, 1] *= orig_h / self.input_height
        return (
            kp_np,
            sc[0, :n].detach().numpy(),
            desc[0, :n].detach().numpy(),
        )

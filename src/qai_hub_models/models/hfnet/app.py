# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw


class HFNetApp:
    """HFNet app wrapper for preprocessing, postprocessing, and visualization."""

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]],
        input_size: tuple[int, int] = (640, 480),
        output_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.input_size = input_size
        self.output_names = output_names or self._infer_output_names()

    def _infer_output_names(self) -> list[str]:
        get_output_spec = getattr(self.model, "get_output_spec", None)
        if callable(get_output_spec):
            output_spec = get_output_spec()
            if isinstance(output_spec, dict) and len(output_spec) == 6:
                return list(output_spec.keys())

        get_output_names = getattr(self.model, "get_output_names", None)
        if callable(get_output_names):
            names = get_output_names()
            if len(names) == 6:
                return list(names)

        return [f"output_{idx}" for idx in range(6)]

    @staticmethod
    def preprocess(
        image: Image.Image,
        input_size: tuple[int, int] = (640, 480),
    ) -> torch.Tensor:
        """Convert PIL image to grayscale NHWC float32 [1, H, W, 1] in 0..255."""
        gray = image.convert("L").resize(input_size, Image.Resampling.BILINEAR)
        x = np.asarray(gray, dtype=np.float32)[None, :, :, None]
        return torch.from_numpy(x)

    @staticmethod
    def filter_topk_keypoints(
        keypoints: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int,
        topk: int = 300,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return valid in-frame keypoints and scores sorted by descending score."""
        keypoints_2d = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
        scores_1d = np.asarray(scores, dtype=np.float32).reshape(-1)

        valid = np.isfinite(keypoints_2d).all(axis=1) & np.isfinite(scores_1d)
        valid &= scores_1d > 0
        valid &= (keypoints_2d[:, 0] >= 0) & (keypoints_2d[:, 0] < width)
        valid &= (keypoints_2d[:, 1] >= 0) & (keypoints_2d[:, 1] < height)

        keypoints_valid = keypoints_2d[valid]
        scores_valid = scores_1d[valid]
        if scores_valid.size:
            idx = np.argsort(-scores_valid)[:topk]
            keypoints_valid = keypoints_valid[idx]
            scores_valid = scores_valid[idx]
        return keypoints_valid, scores_valid

    def _outputs_to_dict(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, np.ndarray]:
        if isinstance(outputs, torch.Tensor):
            raise TypeError("HFNet expected 6 outputs but received a single tensor.")

        outputs_tuple = tuple(outputs)
        if len(outputs_tuple) != 6:
            raise ValueError(f"HFNet expected 6 outputs, got {len(outputs_tuple)}.")

        return {
            name: (
                out.detach().cpu().numpy()
                if isinstance(out, torch.Tensor)
                else np.asarray(out)
            )
            for name, out in zip(self.output_names, outputs_tuple, strict=True)
        }

    def predict(self, image: Image.Image) -> dict[str, np.ndarray]:
        """Run HFNet inference and return named numpy outputs."""
        x = self.preprocess(image, self.input_size)
        return self._outputs_to_dict(self.model(x))

    def predict_keypoint_overlay(
        self,
        image: Image.Image,
        topk: int = 300,
    ) -> Image.Image:
        """Run inference and return RGB image overlay with top-k keypoints."""
        x = self.preprocess(image, self.input_size)
        out = self._outputs_to_dict(self.model(x))

        gray = Image.fromarray(
            x[0, :, :, 0].detach().cpu().numpy().astype(np.uint8),
            mode="L",
        )
        rgb = gray.convert("RGB")

        keypoints, _scores = self.filter_topk_keypoints(
            out["keypoints"],
            out["scores"],
            width=gray.width,
            height=gray.height,
            topk=topk,
        )

        draw = ImageDraw.Draw(rgb)
        for pt in keypoints:
            x0, y0 = float(pt[0]), float(pt[1])
            r = 2
            draw.ellipse(
                (x0 - r, y0 - r, x0 + r, y0 + r),
                outline=(255, 0, 0),
                fill=(255, 0, 0),
            )

        return rgb

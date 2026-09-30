# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import torch
from qai_hub.client import Device
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.superpoint.dataset import HPatchesDataset
from qai_hub_models.models.superpoint.evaluator import SuperPointEvaluator
from qai_hub_models.models.superpoint.external_repos.SuperPoint.superpoint_pytorch import (
    VGGBlock,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

# Maximum number of keypoints returned per image (pads with zeros if fewer detected).
MAX_KEYPOINTS = 500

DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "superpoint_v6_from_tf.pth"
)
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "superpoint.jpg"
)

_DEFAULT_CONF = {
    "nms_radius": 4,
    "detection_threshold": 0.005,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
}


def _sample_descriptors(
    keypoints: torch.Tensor, descriptors: torch.Tensor, s: int = 8
) -> torch.Tensor:
    """Sample dense descriptors at keypoint locations via bilinear interpolation.

    Normalises keypoint pixel coordinates into the [-1, 1] grid expected by
    ``grid_sample``, samples ``descriptors`` (which live on a downsampled feature
    map of stride ``s``), and L2-normalises the result along the channel axis.

    Parameters
    ----------
    keypoints:
        Pixel-space (x, y) coordinates, shape ``(B, N, 2)``.
    descriptors:
        Dense descriptor map from the descriptor head, shape ``(B, C, H, W)``.
    s:
        Backbone stride — the ratio between image pixels and descriptor cells.

    Returns
    -------
    torch.Tensor
        L2-normalised descriptors, shape ``(B, C, N)``.
    """
    b, c, h, w = descriptors.shape
    kp = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    kp = kp * 2 - 1
    desc = torch.nn.functional.grid_sample(
        descriptors, kp.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    return torch.nn.functional.normalize(desc.reshape(b, c, -1), p=2, dim=1)


def _batched_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Suppress non-maximum score values within a local neighbourhood.

    Runs two rounds of iterative suppression: a score survives only if it equals
    the max-pooled value in its ``nms_radius``-pixel window and has not already
    been suppressed by a neighbouring peak.  Operates on the full batch at once.

    Parameters
    ----------
    scores:
        Detection score map, shape ``(B, H, W)``.
    nms_radius:
        Half-width (in pixels) of the suppression window.

    Returns
    -------
    torch.Tensor
        Suppressed score map of the same shape; suppressed positions are zero.
    """

    def max_pool(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class SuperPoint(BaseModel):
    """SuperPoint keypoint detector and descriptor extractor.

    Input:  Grayscale image, float32 [0, 1], shape (B, 1, H, W).
    Output (fixed-size tensors, padded to MAX_KEYPOINTS=500):
        keypoints     — (B, MAX_KEYPOINTS, 2)   x/y pixel coordinates
        scores        — (B, MAX_KEYPOINTS)       detection confidence
        descriptors   — (B, MAX_KEYPOINTS, 256)  L2-normalised feature vectors
        num_keypoints — (B,)                     number of valid (unpadded) keypoints
    """

    def __init__(self, **conf: object) -> None:
        super().__init__()
        cfg = {**_DEFAULT_CONF, **conf}
        self.conf = SimpleNamespace(**cfg)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers: list[nn.Module] = [
                VGGBlock(channels[i - 1], c, 3),
                VGGBlock(c, c, 3),
            ]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

    @classmethod
    def from_pretrained(
        cls,
        weights: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
    ) -> Self:
        """Load SuperPoint from a .pth checkpoint."""
        if isinstance(weights, CachedWebModelAsset):
            weights_path = str(weights.fetch())
        else:
            weights_path = weights

        model = cls()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        return model

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run SuperPoint on *image*.

        Parameters
        ----------
        image
            Pre-processed pixel values. Range float[0, 1], grayscale,
            shape (B, 1, H, W).

        Returns
        -------
        keypoints : torch.Tensor
            Shape (B, MAX_KEYPOINTS, 2). Padded with zeros.
        scores : torch.Tensor
            Shape (B, MAX_KEYPOINTS). Padded with zeros.
        descriptors : torch.Tensor
            Shape (B, MAX_KEYPOINTS, 256). Padded with zeros.
        num_keypoints : torch.Tensor
            Shape (B,). Number of valid keypoints per image (before padding).
        """
        conf = self.conf

        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        # Decode detector scores
        scores = self.detector(features)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        s = self.stride
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, s, s)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * s, w * s)
        scores = _batched_nms(scores, conf.nms_radius)

        # Remove border keypoints
        if conf.remove_borders:
            pad = conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints: topk on flat score map — always static shape (k,).
        # torch.where() produces variable-length dim that breaks torch.export.
        W = scores.shape[2]
        scores_flat = scores.reshape(b, -1)  # (B, H*W)
        # Clamp so tiny images (H*W < MAX_KEYPOINTS) don't raise a topk range error.
        k = min(MAX_KEYPOINTS, scores_flat.shape[1])

        top_scores, top_flat_idx = torch.topk(
            scores_flat, k, dim=1, sorted=True
        )  # (B, k)

        # Pad to MAX_KEYPOINTS so output shape is always static.
        if k < MAX_KEYPOINTS:
            pad_len = MAX_KEYPOINTS - k
            top_scores = torch.cat(
                [top_scores, top_scores.new_zeros(b, pad_len)], dim=1
            )
            top_flat_idx = torch.cat(
                [top_flat_idx, top_flat_idx.new_zeros(b, pad_len)], dim=1
            )

        # Convert flat indices to (y, x) pixel coords, then flip to (x, y).
        top_y = (top_flat_idx // W).float()
        top_x = (top_flat_idx % W).float()
        keypoints = torch.stack([top_x, top_y], dim=-1)  # (B, MAX_KEYPOINTS, 2)

        # Count valid keypoints per image (score above threshold); no Python bool on sym dim.
        valid_mask = top_scores > conf.detection_threshold  # (B, MAX_KEYPOINTS) bool
        num_kp = valid_mask.sum(dim=1).to(torch.int32)  # (B,)

        # Zero out sub-threshold entries so padded outputs are clean.
        kp_scores = top_scores * valid_mask.float()
        keypoints = keypoints * valid_mask.float().unsqueeze(-1)

        # Sample descriptors at all MAX_KEYPOINTS locations (padded locs give ~zero desc).
        descriptors = _sample_descriptors(
            keypoints, descriptors_dense, self.stride
        )  # (B, 256, MAX_KEYPOINTS)
        descriptors = descriptors.permute(0, 2, 1)  # (B, MAX_KEYPOINTS, 256)
        descriptors = descriptors * valid_mask.float().unsqueeze(-1)

        return (
            keypoints,
            kp_scores,
            descriptors,
            num_kp,
        )

    def get_evaluator(self) -> BaseEvaluator:
        spec = self.get_input_spec()
        h, w = spec["image"][0][2], spec["image"][0][3]
        return SuperPointEvaluator(image_height=h, image_width=w)

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [HPatchesDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return HPatchesDataset

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    @staticmethod
    def get_input_spec(
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        if height * width < MAX_KEYPOINTS:
            raise ValueError(
                f"Image resolution {height}x{width} = {height * width} pixels is too small; "
                f"must be at least {MAX_KEYPOINTS} pixels total (height * width >= {MAX_KEYPOINTS})."
            )
        # The backbone downsamples by stride=8; non-multiples produce a score map
        # smaller than get_output_spec declares, causing a confusing shape mismatch.
        stride = 8
        if height % stride != 0 or width % stride != 0:
            raise ValueError(
                f"height and width must each be divisible by {stride} (backbone stride), "
                f"got {height}x{width}."
            )
        return {"image": TensorSpec(shape=(1, 1, height, width), dtype="float32")}

    def get_output_spec(self) -> OutputSpec:
        k = MAX_KEYPOINTS
        return {
            "keypoints": TensorSpec(shape=(1, k, 2)),
            "scores": TensorSpec(shape=(1, k)),
            "descriptors": TensorSpec(shape=(1, k, 256)),
            "num_keypoints": TensorSpec(shape=(1,)),
        }

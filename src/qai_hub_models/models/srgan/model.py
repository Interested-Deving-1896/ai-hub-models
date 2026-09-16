# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from typing_extensions import Self

from qai_hub_models.models.srgan.external_repos.srgan_pytorch import model
from qai_hub_models.models.srgan.external_repos.srgan_pytorch.utils import (
    load_pretrained_state_dict,
)
from qai_hub_models.models.templates.super_resolution.model import (
    DEFAULT_SCALE_FACTOR,
    SuperResolutionModel,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
BASE_ASSET_URL = "https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRGAN_x{scale_factor}-ImageNet.pth.tar"


def validate_srgan_scale_factor(scale_factor: int) -> None:
    """Only these scales have pre-trained checkpoints available."""
    valid_scales = [2, 4, 8]
    assert scale_factor in valid_scales, "`scale_factor` must be in : " + ", ".join(
        [str(s) for s in valid_scales]
    )


class SRGAN(SuperResolutionModel):
    """Exportable SRGAN super resolution model, end-to-end."""

    @classmethod
    def from_pretrained(cls, scale_factor: int = DEFAULT_SCALE_FACTOR) -> Self:
        validate_srgan_scale_factor(scale_factor)

        model_arch_name = f"srresnet_x{scale_factor}"

        sr_model = model.__dict__[model_arch_name]()

        url = BASE_ASSET_URL.format(scale_factor=scale_factor)
        checkpoint_asset = CachedWebModelAsset(
            url,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            Path(url).name,
        ).fetch()
        sr_model = load_pretrained_state_dict(sr_model, False, str(checkpoint_asset))

        return cls(sr_model, scale_factor)

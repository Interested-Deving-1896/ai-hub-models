# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qai_hub_models.models.templates.nafnet.denoising_evaluator import (
    DenoisingEvaluator,
)
from qai_hub_models.models.templates.nafnet.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.templates.nafnet.external_repos.nafnet.basicsr.models.archs import (
    NAFNet_arch as NAFarch,
)
from qai_hub_models.models.templates.nafnet.external_repos.nafnet.basicsr.models.archs import (
    NAFSSR_arch as NAFSSRarch,
)
from qai_hub_models.models.templates.nafnet.model_patches import (
    AutoLayerNorm2d,
    NAFLocal_Base,
    ssrforward,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_torch,
    load_yaml,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class NAFNetModelBase(BaseModel):
    """Shared constructor for NAFNet models."""

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(model)


class NAFNetModel(NAFNetModelBase):
    """Single-image NAFNet restoration model."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run NAFNet on image, and produce denoised/deblurred image.

        Parameters
        ----------
        image
            Input image to be processed
            Pixel values in range [0, 1], RGB color space.

        Returns
        -------
        restored_image : torch.Tensor
            Restored image
            Pixel values in range [0, 1], RGB color space.
        """
        return self.model(image)

    def get_evaluator(self) -> BaseEvaluator:
        return DenoisingEvaluator()


def _load_nafnet_source_model(
    nafnet_weights: str | Path,
    yaml_path_nafnet: str | Path,
    MODEL_ID: str,
    MODEL_ASSET_VERSION: int,
) -> torch.nn.Module:
    """Load the NAFNet source model.

    Parameters
    ----------
    nafnet_weights
        Remote asset filename (``str``) fetched from the asset store, or a
        local ``Path`` to an existing ``.pth`` weights file.
    yaml_path_nafnet
        Repo-relative config path under ``options/test/`` (``str``), or a
        local ``Path`` to an existing ``.yml`` config file.
    MODEL_ID
        Model identifier used for asset store.
    MODEL_ASSET_VERSION
        Asset version number used for asset store.


    Returns
    -------
    torch.nn.Module
        The ``net_g`` submodule of the loaded NAFNet model.
    """
    if isinstance(nafnet_weights, Path):
        weights_path_nafnet = nafnet_weights
        if not weights_path_nafnet.exists():
            raise FileNotFoundError(f"Local weights file not found: {nafnet_weights}")
    else:
        weights_path_nafnet = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, nafnet_weights
        ).fetch()

    naf_arch: Any = NAFarch
    nafssr_arch: Any = NAFSSRarch
    naf_arch.LayerNorm2d = AutoLayerNorm2d
    nafssr_arch.LayerNorm2d = AutoLayerNorm2d
    naf_arch.Local_Base.convert = NAFLocal_Base.convert
    nafssr_arch.Local_Base.convert = NAFLocal_Base.convert
    nafssr_arch.NAFNetSR.forward = ssrforward

    # Handle config path
    if isinstance(yaml_path_nafnet, Path):
        yaml_path = yaml_path_nafnet.resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Local config file not found: {yaml_path_nafnet}")
    else:
        # Load YAML from the cloned repo
        yaml_path = Path(
            EXTERNAL_REPO_PATHS["nafnet"], "options/test", yaml_path_nafnet
        ).resolve()

    opt = load_yaml(yaml_path)

    # Build net_g directly from the config instead of basicsr's create_model,
    # which relies on dynamic imports that don't survive the external_repos move.
    network_g = dict(opt["network_g"])
    builders = {
        "NAFNet": NAFarch.NAFNet,
        "NAFNetLocal": NAFarch.NAFNetLocal,
        "NAFSSR": NAFSSRarch.NAFSSR,
    }
    model = builders[network_g.pop("type")](**network_g)

    state_dict = load_torch(weights_path_nafnet)["params"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state_dict.items()})
    return model.eval()

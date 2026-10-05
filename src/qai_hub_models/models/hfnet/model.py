# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

from .hfnet_torch import HFNetTorch

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_SOURCE_MODEL_FILENAME = "hfnet.onnx"
DEFAULT_TORCH_WEIGHTS_FILENAME = "hfnet_torch_state_dict.pt"

SOURCE_MODEL = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    DEFAULT_SOURCE_MODEL_FILENAME,
)

TORCH_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    DEFAULT_TORCH_WEIGHTS_FILENAME,
)


class HFNetSourceModelNotFoundError(RuntimeError):
    """Raised when HFNet source ONNX cannot be resolved for export/eval."""


class HFNet(BaseModel):
    """HFNet model wrapper using torch-native execution."""

    def __init__(
        self,
        torch_model: HFNetTorch | None,
    ) -> None:
        super().__init__()
        self._torch_model = torch_model

    @staticmethod
    def _load_torch_model_from_source(source_model_path: str) -> HFNetTorch:
        path = Path(source_model_path).resolve()
        if path.suffix.lower() != ".onnx":
            raise ValueError(f"Expected an .onnx file, got: {path}")
        return HFNetTorch.from_onnx_weights(path)

    @staticmethod
    def _load_torch_model_from_weights(torch_weights_path: str) -> HFNetTorch:
        weights_path = Path(torch_weights_path).resolve()
        if weights_path.suffix.lower() not in (".pt", ".pth"):
            raise ValueError(
                f"Expected a .pt/.pth file for torch weights, got: {weights_path}"
            )
        torch_model = HFNetTorch()
        state_dict = torch.load(weights_path, map_location="cpu")
        torch_model.load_state_dict(state_dict)
        torch_model.eval()
        return torch_model

    @classmethod
    def from_pretrained(
        cls,
        source_model_path: str | None = None,
        torch_weights_path: str | None = None,
        require_source_model: bool = False,
    ) -> Self:
        """
        Construct HFNet model wrapper.

        Parameters
        ----------
        source_model_path
            Optional path to source ONNX model used to load torch-native
            HFNet weights.
        torch_weights_path
            Optional path to serialized torch state dict. If not provided and
            no source ONNX path is passed, `from_pretrained` attempts to fetch
            the default state dict asset first and falls back to ONNX mapping.
        require_source_model
            If True, raise HFNetSourceModelNotFoundError when neither torch
            weights nor source ONNX can be resolved.

        Returns
        -------
        Self
            HFNet wrapper with local torch runtime.
        """
        if source_model_path is not None:
            return cls(cls._load_torch_model_from_source(source_model_path))

        if torch_weights_path is None:
            try:
                torch_weights_path = str(TORCH_WEIGHTS.fetch())
            except Exception:
                torch_weights_path = None

        if torch_weights_path is not None:
            return cls(cls._load_torch_model_from_weights(torch_weights_path))

        try:
            source_model_path = str(SOURCE_MODEL.fetch())
        except Exception as exc:
            if require_source_model:
                raise HFNetSourceModelNotFoundError(
                    "HFNet model weights could not be resolved. "
                    "Provide `torch_weights_path` (.pt/.pth) or `source_model_path` (.onnx), "
                    f"or upload `{DEFAULT_TORCH_WEIGHTS_FILENAME}` / "
                    f"`{DEFAULT_SOURCE_MODEL_FILENAME}` to models/{MODEL_ID}/v{MODEL_ASSET_VERSION}/."
                ) from exc

            torch_model = HFNetTorch()
            torch_model.eval()
            return cls(torch_model)

        return cls(cls._load_torch_model_from_source(source_model_path))

    def get_input_spec(self) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(1, 480, 640, 1),
                dtype="float32",
                io_type=IoType.IMAGE,
                # Upstream HFNet normalization is (x - 128) / 128.
                value_range=(0.0, 255.0),
                image_metadata=ImageMetadata(color_format=ColorFormat.GRAYSCALE),
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "global_descriptor": TensorSpec(),
            "keypoints": TensorSpec(),
            "scores": TensorSpec(),
            "scores_dense": TensorSpec(),
            "logits": TensorSpec(),
            "prob_full": TensorSpec(),
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run local HFNet inference on `image` and return 6 tensors.

        Input is NHWC grayscale float32 in 0..255. Outputs are
        `(global_descriptor, keypoints, scores, scores_dense, logits, prob_full)`.
        """
        if self._torch_model is None:
            raise RuntimeError(
                "HFNet local FP runtime was not initialized. "
                "Pass `source_model_path` to from_pretrained() for local torch inference, "
                "or run demo with --eval-mode on-device."
            )

        return self._torch_model(image)


__all__ = [
    "DEFAULT_SOURCE_MODEL_FILENAME",
    "DEFAULT_TORCH_WEIGHTS_FILENAME",
    "MODEL_ASSET_VERSION",
    "MODEL_ID",
    "HFNet",
    "HFNetSourceModelNotFoundError",
]

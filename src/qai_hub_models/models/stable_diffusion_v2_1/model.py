# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# (module level import not at top of file)

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from qai_hub_models.models._shared.diffusion.stable_diffusion import (
    StableDiffusionBase,
    TextEncoderQuantizableBase,
    UnetQuantizableBase,
    VaeDecoderQuantizableBase,
)
from qai_hub_models.utils.onnx.helpers import ONNXBundle

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

MODEL_ASSET_VERSION = 4
MODEL_ID = __name__.split(".")[-2]
HF_REPO = "sd2-community/stable-diffusion-2-1"


def make_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(HF_REPO, subfolder="tokenizer")


# SEQ_LEN = make_tokenizer().model_max_length
SEQ_LEN = 77  # Tokenizer is no longer on Hugging Face.


class TextEncoderQuantizable(TextEncoderQuantizableBase):
    hf_repo_id = HF_REPO
    hf_model_cls = CLIPTextModel
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        super().__init__(sim_model, host_device, onnx_bundle, seq_len=SEQ_LEN)


class UnetQuantizable(UnetQuantizableBase):
    hf_repo_id = HF_REPO
    hf_model_cls = UNet2DConditionModel  # type: ignore[assignment]
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        super().__init__(sim_model, host_device, onnx_bundle, seq_len=SEQ_LEN)


class VaeDecoderQuantizable(VaeDecoderQuantizableBase):
    hf_repo_id = HF_REPO
    hf_model_cls = AutoencoderKL  # type: ignore[assignment]
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION


# Align component names with Huggingface's repo's subfolder names
class StableDiffusionV2_1_Quantized(StableDiffusionBase):
    hf_repo_id = HF_REPO
    component_classes = {
        "text_encoder": TextEncoderQuantizable,
        "unet": UnetQuantizable,
        "vae": VaeDecoderQuantizable,
    }

    def __init__(
        self,
        text_encoder: TextEncoderQuantizable,
        unet: UnetQuantizable,
        vae: VaeDecoderQuantizable,
    ) -> None:
        super().__init__(text_encoder, unet, vae)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "DEFAULT",
    ) -> StableDiffusionV2_1_Quantized:
        return cls(
            TextEncoderQuantizable.from_pretrained(checkpoint=checkpoint),
            UnetQuantizable.from_pretrained(checkpoint=checkpoint),
            VaeDecoderQuantizable.from_pretrained(checkpoint=checkpoint),
        )

    @staticmethod
    def make_tokenizer() -> CLIPTokenizer:
        return make_tokenizer()

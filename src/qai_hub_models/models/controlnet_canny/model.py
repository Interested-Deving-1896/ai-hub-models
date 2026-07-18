# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# (module level import not at top of file)

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from qai_hub_models.models._shared.diffusion.controlnet import (
    ControlNetQuantizableBase,
    ControlUnetQuantizableBase,
)
from qai_hub_models.models._shared.diffusion.stable_diffusion import (
    StableDiffusionBase,
    TextEncoderQuantizableBase,
    VaeDecoderQuantizableBase,
)
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

HF_REPO_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HF_REPO_CN = "lllyasviel/sd-controlnet-canny"


def make_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(HF_REPO_SD, subfolder="tokenizer")


SEQ_LEN = make_tokenizer().model_max_length


class TextEncoderQuantizable(TextEncoderQuantizableBase):
    hf_repo_id = HF_REPO_SD
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


class ControlUnetQuantizable(ControlUnetQuantizableBase):
    hf_repo_id = HF_REPO_SD
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
    hf_repo_id = HF_REPO_SD
    hf_model_cls = AutoencoderKL  # type: ignore[assignment]
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION


class ControlNetQuantizable(ControlNetQuantizableBase):
    hf_repo_id = HF_REPO_CN
    hf_model_cls = ControlNetModel  # type: ignore[assignment]
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
        onnx_bundle: ONNXBundle | None = None,
    ) -> None:
        super().__init__(sim_model, host_device, onnx_bundle, seq_len=SEQ_LEN)


# Align component names with Huggingface's repo's subfolder names
class ControlNetCannyQuantized(StableDiffusionBase):
    hf_repo_id = HF_REPO_SD
    component_classes = {
        "text_encoder": TextEncoderQuantizable,
        "unet": ControlUnetQuantizable,
        "vae": VaeDecoderQuantizable,
        "controlnet": ControlNetQuantizable,
    }

    def __init__(
        self,
        text_encoder: TextEncoderQuantizable,
        unet: ControlUnetQuantizable,
        vae: VaeDecoderQuantizable,
        controlnet: ControlNetQuantizable,
    ) -> None:
        super().__init__(text_encoder, unet, vae)
        self.components["controlnet"] = controlnet
        self.controlnet = controlnet

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(batch_size=batch_size)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "DEFAULT",
    ) -> ControlNetCannyQuantized:
        return cls(
            TextEncoderQuantizable.from_pretrained(checkpoint=checkpoint),
            ControlUnetQuantizable.from_pretrained(checkpoint=checkpoint),
            VaeDecoderQuantizable.from_pretrained(checkpoint=checkpoint),
            ControlNetQuantizable.from_pretrained(checkpoint=checkpoint),
        )

    @staticmethod
    def make_tokenizer() -> CLIPTokenizer:
        return make_tokenizer()

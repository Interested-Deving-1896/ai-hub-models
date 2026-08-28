# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# flake8: noqa: E402
# (module level import not at top of file)

from __future__ import annotations

import os
from pathlib import Path

# isort: off
# This verifies aimet is installed, and this must be included first.
MODEL_ID = __name__.split(".")[-2]
from qai_hub_models.utils.quantization_aimet_onnx import ensure_aimet_onnx_installed

ensure_aimet_onnx_installed(model_id=MODEL_ID)
# isort: on

from transformers import WhisperConfig
from typing_extensions import Self

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models.templates.hf_whisper.model import (
    TIKTOKEN_URL,
    HfWhisperDecoder,
    HfWhisperEncoder,
)
from qai_hub_models.models.templates.hf_whisper.utils import (
    write_whisper_supplementary_files,
)
from qai_hub_models.models.templates.hf_whisper.whisper_metadata_json import (
    WhisperCapabilities,
)
from qai_hub_models.models.templates.hf_whisper_quantized.model import (
    WhisperDecoderQuantizableBase,
    WhisperEncoderQuantizableBase,
)
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection

MODEL_ASSET_VERSION = 3
WHISPER_VERSION = "openai/whisper-large-v3-turbo"
ENCODER_AIMET = "encoder.aimet"
DECODER_AIMET = "decoder.aimet"

WHISPER_LARGE_V3_TURBO_QUANTIZED_CAPABILITIES = WhisperCapabilities(
    streaming=False,
    file_based=True,
    language_detection=True,
    confidence_scores=False,
)


class WhisperLargeV3TurboEncoderQuantizable(WhisperEncoderQuantizableBase):
    @classmethod
    def make_torch_model(cls) -> HfWhisperEncoder:
        return WhisperLargeV3Turbo.from_pretrained().encoder

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[Path, Path]:  # type: ignore[override]
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(ENCODER_AIMET, "model.onnx"),
        ).fetch()
        # Encoder weights exceed ONNX 2GB protobuf limit and are stored as
        # external data. Both files must be in the same directory for
        # onnx.load_model to resolve the weights automatically.
        CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(ENCODER_AIMET, "model.onnx.data"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(ENCODER_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings


class WhisperLargeV3TurboDecoderQuantizable(WhisperDecoderQuantizableBase):
    @classmethod
    def make_torch_model(cls) -> HfWhisperDecoder:
        return WhisperLargeV3Turbo.from_pretrained().decoder

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[Path, Path]:  # type: ignore[override]
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(DECODER_AIMET, "model.onnx"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(DECODER_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings


class WhisperLargeV3TurboQuantized(WorkbenchModelCollection):
    def __init__(
        self,
        encoder: WhisperEncoderQuantizableBase,
        decoder: WhisperDecoderQuantizableBase,
        config: WhisperConfig,
        hf_source: str,
    ) -> None:
        super().__init__({"encoder": encoder, "decoder": decoder})
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.hf_source = hf_source

    @classmethod
    def get_hf_whisper_version(cls) -> str:
        return WHISPER_VERSION

    @classmethod
    def from_pretrained(cls) -> Self:
        encoder = WhisperLargeV3TurboEncoderQuantizable.from_pretrained()
        decoder = WhisperLargeV3TurboDecoderQuantizable.from_pretrained()
        return cls(encoder, decoder, encoder.config, WHISPER_VERSION)

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_whisper_supplementary_files(
            output_dir,
            metadata,
            "whisper-large-v3-turbo-quantized",
            WHISPER_LARGE_V3_TURBO_QUANTIZED_CAPABILITIES,
            TIKTOKEN_URL,
            display_name="Whisper Large V3 Turbo (Quantized)",
        )

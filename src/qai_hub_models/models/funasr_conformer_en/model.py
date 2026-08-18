# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any

import torch
from funasr import AutoModel
from typing_extensions import Self

from qai_hub_models.models.funasr_conformer_en.evaluator import (
    FunASRConformerEvaluator,
)
from qai_hub_models.models.funasr_conformer_en.model_patches import (
    _ConformerEncoderWithCTC,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel, SerializationSettings
from qai_hub_models.utils.input_spec import InputSpec, IoType, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
HF_MODEL_ID = "funasr/conformer-en"

# Conformer-en frontend: 16kHz, 80 mel bins, LFR m=7 n=6 → input_size=560
SAMPLE_RATE = 16000
FBANK_BINS = 80
LFR_M = 7
LFR_N = 6
INPUT_SIZE = FBANK_BINS * LFR_M  # 560

# Fixed-length input: 10 seconds of raw audio at 16kHz
DEFAULT_AUDIO_LENGTH = 160000  # 10s x 16kHz

# Actual frame count produced by WavFrontend for DEFAULT_AUDIO_LENGTH samples.
# WavFrontend: 25ms window / 10ms shift → ~998 fbank frames, LFR n=6 → ceil(998/6)=167.
# This must match the model's traced input shape and the OnDeviceModel input spec.
DEFAULT_NUM_FRAMES = 167


def audio_len_to_valid_frames(real_len: int) -> int:
    """Frame count for real_len samples at 16kHz with 25ms/10ms windows and LFR n=6."""
    num_fbank = max(real_len - 400, 0) // 160 + 1
    return math.ceil(num_fbank / LFR_N)


class FunASRConformerEn(BaseModel):
    """
    FunASR Conformer-EN English ASR model.

    Architecture:
    - WavFrontend preprocessing: 16kHz audio → 80-dim mel + LFR(m=7,n=6) → 560-dim features
    - ConformerEncoder: 32 blocks, 512 hidden dim, 16 heads
    - CTC head: Linear(512 → 4200) + log_softmax

    The exported model takes pre-computed mel-LFR features as input (features
    are produced by the WavFrontend and are not part of the exported graph, as
    the frontend uses non-differentiable DSP ops that don't trace cleanly).
    CTC decoding is performed in the App layer.

    """

    def __init__(
        self,
        model: _ConformerEncoderWithCTC,
        token_list: list[str],
        frontend: Any,
    ) -> None:
        # use_pt2=False: torch.export symbolic tracing breaks on data-dependent
        # control flow in FunASR's make_pad_mask (uses .tolist()). TorchScript
        # trace (jit.trace) handles this correctly with concrete inputs.
        super().__init__(serialization_settings=SerializationSettings(use_pt2=False))
        self.model = model
        self.token_list = token_list
        self.frontend = frontend

    @classmethod
    def from_pretrained(cls) -> Self:
        """Load FunASR conformer-en from HuggingFace."""
        auto = AutoModel(
            model=HF_MODEL_ID,
            hub="hf",
            disable_update=True,
            device="cpu",
        )
        inner = auto.model.cpu().eval()
        token_list = auto.kwargs["token_list"]
        frontend = auto.kwargs["frontend"]
        model = _ConformerEncoderWithCTC(inner.encoder, inner.ctc.ctc_lo)
        return cls(model, token_list, frontend)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Run Conformer encoder and CTC on pre-computed mel-LFR features.

        Parameters
        ----------
        feats
            Shape (1, T, 560) — mel-LFR features, float32.

        Returns
        -------
        log_probs: torch.Tensor
            Shape (1, T, 4200) — CTC log-probabilities.
        """
        return self.model(feats)

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_frames: int = DEFAULT_NUM_FRAMES,
    ) -> InputSpec:
        """
        Input spec for the exported encoder+CTC model.

        num_frames defaults to DEFAULT_NUM_FRAMES (167), the actual frame count
        produced by WavFrontend for a full 10-second / 160,000-sample chunk.
        """
        return {
            "feats": TensorSpec(
                shape=(batch_size, num_frames, INPUT_SIZE),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"log_probs": TensorSpec()}

    def get_evaluator(self) -> BaseEvaluator:
        return FunASRConformerEvaluator(self.token_list or [])

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        from qai_hub_models.models.funasr_conformer_en.dataset import (
            ConformerLibriSpeechDataset,
        )

        return [ConformerLibriSpeechDataset]

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor
from typing_extensions import Self

from qai_hub_models.models.wav2vec2_conformer.dataset import (
    Wav2Vec2ConformerLibriSpeechDataset,
)
from qai_hub_models.models.wav2vec2_conformer.evaluator import (
    Wav2Vec2ConformerEvaluator,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, IoType, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
HF_MODEL_ID = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"

SAMPLE_RATE = 16000

# Fixed-length input: 10 seconds of raw audio at 16kHz
DEFAULT_AUDIO_LENGTH = 160000  # 10s x 16kHz


class Wav2Vec2Conformer(BaseModel):
    """
    Facebook Wav2Vec2-Conformer-Large fine-tuned on 960h LibriSpeech.

    Architecture:
    - Wav2Vec2 feature extractor: raw 16kHz audio → convolutional features
    - Conformer encoder with relative-position attention (large: 24 layers, 1024 dim)
    - CTC head: Linear(1024 → vocab) + log_softmax

    The exported model takes raw float32 audio samples as input.
    Processor feature extraction (normalization) is handled in the App layer
    since it uses non-differentiable ops. CTC decoding is also done in App.
    """

    def __init__(
        self,
        model: Wav2Vec2ConformerForCTC,
    ) -> None:
        super().__init__(model)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str = HF_MODEL_ID) -> Self:
        model = Wav2Vec2ConformerForCTC.from_pretrained(checkpoint_path)
        model = model.cpu()
        return cls(model)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Run Wav2Vec2-Conformer encoder and CTC head on normalized audio.

        Parameters
        ----------
        input_values
            Shape (N, T) — normalized float32 audio samples
            N is the no of audio chunks

        Returns
        -------
        logits: torch.Tensor
            Shape (N, T', vocab_size) — CTC logits (pre-softmax).
        """
        return self.model(input_values).logits

    def get_input_spec(
        self,
        batch_size: int = 1,
        audio_length: int = DEFAULT_AUDIO_LENGTH,
    ) -> InputSpec:
        return {
            "input_values": TensorSpec(
                shape=(batch_size, audio_length),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"logits": TensorSpec()}

    def get_evaluator(self) -> BaseEvaluator:
        processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_ID)
        return Wav2Vec2ConformerEvaluator(processor)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [Wav2Vec2ConformerLibriSpeechDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Wav2Vec2ConformerLibriSpeechDataset

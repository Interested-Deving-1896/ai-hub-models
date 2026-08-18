# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from funasr import AutoModel

from qai_hub_models.datasets.librispeech.librispeech import (
    DEFAULT_MAX_TEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LibriSpeechDataset,
)
from qai_hub_models.models.funasr_conformer_en.model import (
    HF_MODEL_ID,
    audio_len_to_valid_frames,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


class ConformerLibriSpeechDataset(LibriSpeechDataset):
    """LibriSpeech dataset pre-processed for the FunASR Conformer-EN model."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        target_sample_rate: int = 16000,
        max_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(
            split=split,
            target_sample_rate=target_sample_rate,
            max_sequence_length=max_sequence_length,
            max_text_length=max_text_length,
            input_spec=input_spec,
        )

        self.frontend = AutoModel(
            model=HF_MODEL_ID,
            hub="hf",
            disable_update=True,
            device="cpu",
        ).kwargs["frontend"]

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        index
            The index of the audio file and transcription in the dataset.

        Returns
        -------
        torch.Tensor
            Mel-LFR feature tensor, shape (T, 560), float32.
            Conforms to FunASRConformerEn.forward(feats) input signature.
        tuple[torch.Tensor, torch.Tensor]
            (gt_text, valid_frames): ASCII ground-truth tensor of shape
            (max_text_length,) int32, and scalar int64 valid frame count
            derived from the real (pre-padding) audio length. The frame
            count is used by the evaluator to bound CTC decoding to the
            same region that app._transcribe_chunk would decode.
        """
        (audio, attention_mask), gt = super().__getitem__(index)

        # attention_mask is 1 for real samples, 0 for padding; sum = real sample count.
        real_len = int(attention_mask.sum().item())

        audio_tensor = audio.unsqueeze(0)
        audio_len = torch.tensor([audio_tensor.shape[-1]], dtype=torch.int64)

        feats, _ = self.frontend(audio_tensor, audio_len)

        # Compute valid frame count using the same formula as app._transcribe_chunk.
        valid_frames = torch.tensor(
            audio_len_to_valid_frames(real_len), dtype=torch.int64
        )

        # frontend returns (1, T, 560); squeeze batch dim so DataLoader stacks
        # correctly to (B, T, 560) matching forward(feats).
        return feats.squeeze(0), (gt, valid_frames)

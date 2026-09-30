# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import Wav2Vec2Processor

from qai_hub_models.models.wav2vec2_conformer.utils import normalize_audio
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

_HF_PARQUET_BASE = (
    "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/clean"
)

# Parquet shards for each split — avoids the script-based builder that downloads
# every split when any single split is requested.
_PARQUET_URLS: dict[str, list[str]] = {
    "test": [f"{_HF_PARQUET_BASE}/test/0000.parquet"],
    "train.100": [f"{_HF_PARQUET_BASE}/train.100/{i:04d}.parquet" for i in range(14)],
}
HF_MODEL_ID = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"

MAX_TEXT_LENGTH = (
    600  # covers longest transcription in LibriSpeech test-clean (576 chars)
)
MAX_AUDIO_LENGTH = 160000  # 10s at 16kHz — matches the on-device fixed-length input


class Wav2Vec2ConformerLibriSpeechDataset(BaseDataset):
    """LibriSpeech test-clean dataset for Wav2Vec2-Conformer evaluation."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        target_sample_rate: int = 16000,
        max_audio_length: int = MAX_AUDIO_LENGTH,
        max_text_length: int = MAX_TEXT_LENGTH,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        if input_spec is not None and "input_values" in input_spec:
            max_audio_length = input_spec["input_values"][0][1]
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length

        hf_split = "train.100" if split == DatasetSplit.TRAIN else "test"
        self.ds = load_dataset(
            "parquet",
            data_files={"data": _PARQUET_URLS[hf_split]},
            split="data",
        )

        BaseDataset.__init__(self, "non_existent_dir", split, input_spec)
        self.processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_ID)

    def _validate_data(self) -> bool:
        return hasattr(self, "ds")

    def _download_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        torch.Tensor
            (input_values), shape (MAX_AUDIO_LENGTH,).
        torch.Tensor
            ASCII ground-truth int32 tensor, shape (MAX_TEXT_LENGTH,).
        """
        sample = self.ds[index]
        audio_array = sample["audio"]["array"].astype(np.float32)
        text = sample["text"]

        inputs = normalize_audio(
            self.processor, audio_array, self.target_sample_rate, self.max_audio_length
        ).input_values.squeeze(0)

        text_truncated = text[: self.max_text_length]
        gt = torch.tensor([ord(c) for c in text_truncated], dtype=torch.int32)

        if len(gt) < self.max_text_length:
            pad_len = self.max_text_length - len(gt)
            gt = F.pad(gt, (0, pad_len), value=0)

        return inputs, gt

    @staticmethod
    def default_samples_per_job() -> int:
        return 50

    @staticmethod
    def default_num_calibration_samples() -> int:
        return 20

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.openslr.org/12",
            split_description="test-clean",
        )

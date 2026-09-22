# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

EARNINGS21_FOLDER_NAME = "earnings21"
EARNINGS21_VERSION = 2

# Earnings-21 eval-10 subset: 11 files, ~10.3 hours, 136 MB total (MP3).
# Audio sourced from: https://github.com/revdotcom/speech-datasets
# RTTM ground-truth labels sourced from the same repository.
# Transcripts licensed under CC-BY-SA 4.0.
_GITHUB_RAW = "https://raw.githubusercontent.com/revdotcom/speech-datasets/d5d09d014bb2fd2e8837ba097b75527174677ff1/earnings21"

EVAL10_FILE_IDS = [
    "4320211",
    "4341191",
    "4346818",
    "4359971",
    "4365024",
    "4366522",
    "4366893",
    "4367535",
    "4383161",
    "4384964",
    "4387332",
]

# CachedWebDatasetAsset entries for each eval-10 audio file (MP3)
EVAL10_AUDIO_ASSETS: dict[str, CachedWebDatasetAsset] = {
    fid: CachedWebDatasetAsset(
        f"{_GITHUB_RAW}/media/{fid}.mp3",
        EARNINGS21_FOLDER_NAME,
        EARNINGS21_VERSION,
        f"{fid}.mp3",
    )
    for fid in EVAL10_FILE_IDS
}


class Earnings21Dataset(BaseDataset):
    """Earnings-21 eval-10 dataset base class.

    Manages download and access to the 11 eval-10 earnings-call recordings.
    Subclass this to add model-specific preprocessing or calibration logic.

    Parameters
    ----------
    split:
        Dataset split.  The split value is accepted but not used — all
        audio files are loaded regardless of split.
    file_ids:
        Subset of eval-10 file IDs to include.  Defaults to all 11 files.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        file_ids: list[str] | None = None,
    ) -> None:
        self.file_ids = file_ids if file_ids is not None else list(EVAL10_FILE_IDS)
        base_path = EVAL10_AUDIO_ASSETS[self.file_ids[0]].path.parent
        BaseDataset.__init__(self, base_path, split)

    def _validate_data(self) -> bool:
        """Validate that all audio files are present."""
        return all(EVAL10_AUDIO_ASSETS[fid].path.exists() for fid in self.file_ids)

    def _download_data(self) -> None:
        """Download all audio files."""
        for fid in self.file_ids:
            EVAL10_AUDIO_ASSETS[fid].fetch()

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError(
            "Earnings21Dataset is a base class. Use a model-specific subclass."
        )

    @classmethod
    def dataset_name(cls) -> str:
        return "earnings21"

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/revdotcom/speech-datasets/tree/main/earnings21",
            split_description="eval-10 subset (11 files, ~10.3 hours)",
        )

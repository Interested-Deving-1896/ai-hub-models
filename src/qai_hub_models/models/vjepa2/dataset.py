# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from qai_hub_models.datasets.kinetics400.kinetics400 import (
    CORRUPTED_TRAIN_VIDEOS,
    _get_labeled_data,
)
from qai_hub_models.datasets.kinetics400.video_utils import (
    normalize,
    read_video_per_second,
    sample_video,
)
from qai_hub_models.models.vjepa2.model import FRAMES, IMG_SIZE
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

_KINETICS400_FOLDER_NAME = "kinetics400"
_KINETICS400_VERSION = 2


def _preprocess_vjepa2(
    raw: torch.Tensor, num_frames: int, img_size: int
) -> torch.Tensor:
    """
    Preprocess a raw ``[T, H, W, C]`` uint8 video for V-JEPA 2.

    Steps:
      1. Uniform frame sampling to ``num_frames``.
      2. Normalize to float32 [0, 1] and permute to ``[C, T, H, W]``.
      3. Bilinear resize so the shorter side == ``img_size``.
      4. Center-crop to ``[C, T, img_size, img_size]``.

    Parameters
    ----------
    raw
        Raw video tensor of shape ``[T, H, W, C]``, uint8 values 0-255.
    num_frames
        Number of frames to sample from the video.
    img_size
        Spatial size for the square center crop.

    Returns
    -------
    torch.Tensor
        Shape ``[C, T, img_size, img_size]``.
    """
    clip = sample_video(raw, num_frames)  # [T, H, W, C]
    video = normalize(clip)  # [C, T, H, W]

    _, _, h, w = video.shape
    if h < w:
        new_h, new_w = img_size, int(w * img_size / h)
    else:
        new_h, new_w = int(h * img_size / w), img_size

    # F.interpolate expects [B, C, H, W]; treat T as batch
    frames = video.permute(1, 0, 2, 3)  # [T, C, H, W]
    frames = F.interpolate(
        frames, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    # center crop
    top = (new_h - img_size) // 2
    left = (new_w - img_size) // 2
    frames = frames[:, :, top : top + img_size, left : left + img_size]
    return frames.permute(1, 0, 2, 3)  # [C, T, img_size, img_size]


class Kinetics400VJEPA2Dataset(BaseDataset):
    """
    Kinetics-400 wrapper for V-JEPA 2 calibration.

    Bypasses ``Kinetics400Dataset``'s 112/224 size restriction and applies
    V-JEPA 2's own 256x256 preprocessing.

    Frame count and spatial resolution are fixed to V-JEPA 2's input spec
    (``FRAMES`` frames, ``IMG_SIZE`` x ``IMG_SIZE``).
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.split_str = split.name.lower()
        self.num_frames = FRAMES
        self.img_size = IMG_SIZE

        self.videos_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/{self.split_str}/part_0.tar.gz",
            _KINETICS400_FOLDER_NAME,
            _KINETICS400_VERSION,
            f"{self.split_str}/part_0.tar.gz",
        )
        self.csv_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/annotations/{self.split_str}.csv",
            _KINETICS400_FOLDER_NAME,
            _KINETICS400_VERSION,
            f"annotations/{self.split_str}.csv",
        )
        self.videos_folder = self.videos_asset.extracted_path
        self.mp4_files: list[str] = []

        BaseDataset.__init__(
            self,
            str(self.videos_folder),
            split=split,
            input_spec=None,  # don't forward input_spec — size validation lives upstream
        )

    def __len__(self) -> int:
        return 993 if self.split == DatasetSplit.TRAIN else 1000

    def _validate_data(self) -> bool:
        if not self.csv_asset.path.exists():
            return False
        if not self.videos_asset.path.parent.exists():
            return False
        self.mp4_files, _ = _get_labeled_data(self.videos_folder, self.csv_asset.path)
        return len(self.mp4_files) == len(self)

    def _download_data(self) -> None:
        self.videos_asset.fetch(extract=True)
        self.csv_asset.fetch()

        # The TRAIN tar ships 1000 videos, 7 of which are corrupted and cannot be
        # decoded. Kinetics400Dataset removes them on download and its __len__
        # expects the resulting 993; we mirror that here so _validate_data's
        # len(mp4_files) == 993 check passes. (VAL has no corrupted videos.)
        if self.split == DatasetSplit.TRAIN:
            for video in CORRUPTED_TRAIN_VIDEOS:
                video_path = self.videos_folder / video
                if video_path.exists():
                    os.remove(video_path)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        idx
            Index of the video clip in the dataset.

        Returns
        -------
        video : torch.Tensor
            Shape ``[3, T, 256, 256]`` — the DataLoader adds the batch dim.
        gt_video : torch.Tensor
            Unused placeholder (clone of video) kept for BaseDataset compatibility.
        """
        video_path = str(self.videos_folder / self.mp4_files[idx])
        raw = read_video_per_second(video_path)  # [T, H, W, C]
        clip = _preprocess_vjepa2(raw, self.num_frames, self.img_size)  # [C, T, H, W]
        return clip, clip.clone()

    @staticmethod
    def default_samples_per_job() -> int:
        return 40

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/cvdfoundation/kinetics-dataset",
            split_description="part 0 of the validation split (single clip, 256x256)",
        )

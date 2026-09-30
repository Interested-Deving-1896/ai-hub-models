# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

HPATCHES_FOLDER_NAME = "hpatches"
HPATCHES_VERSION = 1
HPATCHES_ASSET = CachedWebDatasetAsset.from_asset_store(
    HPATCHES_FOLDER_NAME,
    HPATCHES_VERSION,
    "hpatches-sequences-release.zip",
)

# Number of images per sequence (1 reference + 5 target)
IMAGES_PER_SEQ = 6
# Total sequences in the full dataset
NUM_SEQUENCES = 116

# Fixed stratified split: ~20% for calibration (TRAIN), ~80% for evaluation
# (VAL/TEST). Sequences are first partitioned by type (illumination i_* and
# viewpoint v_*) then the first N of each type go to TRAIN, keeping the same
# illumination/viewpoint ratio in both partitions.
NUM_TRAIN_ILLUM = 12  # out of 57 illumination sequences
NUM_TRAIN_VIEW = 12  # out of 59 viewpoint sequences
NUM_TRAIN_SEQUENCES = NUM_TRAIN_ILLUM + NUM_TRAIN_VIEW  # 24
NUM_VAL_SEQUENCES = NUM_SEQUENCES - NUM_TRAIN_SEQUENCES  # 92


class HPatchesDataset(BaseDataset):
    """HPatches dataset for homography estimation evaluation.

    https://github.com/hpatches/hpatches-dataset

    HPatches has 116 sequences: 57 illumination (i_*) and 59 viewpoint (v_*).
    A stratified split is applied to keep the same illumination/viewpoint ratio
    in both partitions:

      TRAIN (calibration): first 12 i_* + first 12 v_* = 24 sequences
      VAL/TEST (evaluation): remaining 45 i_* + remaining 47 v_* = 92 sequences

    Sequences are sorted alphabetically within each type before splitting, so
    the partition is deterministic across machines. This avoids data pollution
    between quantisation calibration and accuracy benchmarking, and ensures
    calibration data is representative of both scene-change types.

    Each pair is split into two consecutive items: the reference image
    (is_ref=True) followed immediately by the target image (is_ref=False).
    This keeps the model input shape at (1, 1, H, W) — matching the compiled
    model — while the evaluator re-pairs outputs using an internal buffer.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        num_samples: int = -1,
    ) -> None:
        """Initialize the HPatches dataset.

        Parameters
        ----------
        split
            TRAIN → first 24 sequences (calibration).
            VAL/TEST → remaining 92 sequences (evaluation).
        input_spec
            Optional input specification. Height and width are extracted from
            the last two dimensions of the ``image`` entry. Defaults to 480x640.
        num_samples
            Total number of items to expose (must be even; each pair contributes
            2 items). Defaults to -1 (use all sequences in the split).
            get_dataloader() subsets further via default_samples_per_job().
        """
        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 1, 480, 640), dtype="float32")
        }
        self.height = input_spec["image"][0][-2]
        self.width = input_spec["image"][0][-1]
        self.num_samples = num_samples
        self.hpatches_path = HPATCHES_ASSET.extracted_path
        super().__init__(self.hpatches_path, split, input_spec)
        self.hpatches_path = self._find_sequences_root(self.hpatches_path)
        self._build_pairs()

    def _download_data(self) -> None:
        HPATCHES_ASSET.fetch(extract=True)

    @staticmethod
    def _is_sequence_dir(d: Path) -> bool:
        """Return True if directory is an HPatches sequence (i_* or v_* prefix)."""
        return d.is_dir() and (d.name.startswith("i_") or d.name.startswith("v_"))

    def _validate_data(self) -> bool:
        """Return True if the extracted dataset contains all 116 expected sequences."""
        if not HPATCHES_ASSET.extracted_path.exists():
            return False
        root = self._find_sequences_root(HPATCHES_ASSET.extracted_path)
        seqs = [d for d in root.iterdir() if self._is_sequence_dir(d)]
        return len(seqs) == NUM_SEQUENCES

    def _find_sequences_root(self, path: Path) -> Path:
        """Walk down from path until the directory containing i_*/v_* folders is found.

        The extracted zip may wrap sequences inside one or two extra directories
        (e.g. ``hpatches-sequences-release/``). This method descends up to three
        levels along single-child directory chains until sequence folders appear.
        """
        current = path
        for _ in range(3):
            entries = list(current.iterdir())
            seq_dirs = [e for e in entries if self._is_sequence_dir(e)]
            if seq_dirs:
                return current
            sub_dirs = [e for e in entries if e.is_dir()]
            if len(sub_dirs) == 1:
                current = sub_dirs[0]
            else:
                break
        return current

    @classmethod
    def dataset_name(cls) -> str:
        return "hpatches"

    @staticmethod
    def default_samples_per_job() -> int:
        # 50 pairs x 2 images
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """Return metadata describing the dataset and its evaluation split.

        The split_description covers the VAL partition used for benchmarking.
        The TRAIN partition (calibration) is kept separate to prevent data
        pollution between quantisation calibration and accuracy evaluation.
        """
        return DatasetMetadata(
            link="https://github.com/hpatches/hpatches-dataset",
            split_description=(
                f"{NUM_VAL_SEQUENCES} of {NUM_SEQUENCES} sequences "
                f"({57 - NUM_TRAIN_ILLUM} illumination + {59 - NUM_TRAIN_VIEW} viewpoint)"
            ),
        )

    def _build_pairs(self) -> None:
        """Build the list of (sequence_path, target_index) pairs for the active split.

        Applies a stratified train/val split to avoid data pollution between
        quantisation calibration (TRAIN) and accuracy benchmarking (VAL/TEST):

          1. All 116 sequences are collected and sorted alphabetically.
          2. They are partitioned by prefix into illumination sequences (i_*)
             and viewpoint sequences (v_*).
          3. The first NUM_TRAIN_ILLUM (12) illumination sequences and the first
             NUM_TRAIN_VIEW (12) viewpoint sequences form the TRAIN partition.
             The remaining 45 + 47 = 92 sequences form VAL/TEST.

        Sorting within each type is deterministic across machines (lexicographic
        order of sequence names). This ensures calibration data is representative
        of both scene-change types in roughly the same ratio as the full dataset.
        """
        all_seqs = sorted(
            d for d in self.hpatches_path.iterdir() if self._is_sequence_dir(d)
        )
        illum_seqs = [s for s in all_seqs if s.name.startswith("i_")]
        view_seqs = [s for s in all_seqs if s.name.startswith("v_")]

        if self.split == DatasetSplit.TRAIN:
            seqs = illum_seqs[:NUM_TRAIN_ILLUM] + view_seqs[:NUM_TRAIN_VIEW]
        else:
            seqs = illum_seqs[NUM_TRAIN_ILLUM:] + view_seqs[NUM_TRAIN_VIEW:]

        all_pairs: list[tuple[Path, int]] = [
            (seq, idx) for seq in seqs for idx in range(2, IMAGES_PER_SEQ + 1)
        ]

        if self.num_samples == -1:
            self._pairs = all_pairs
        else:
            max_pairs = self.num_samples // 2
            self._pairs = all_pairs[:max_pairs]

    def get_dataloader(
        self, num_samples: int, samples_per_job: int | None = None
    ) -> DataLoader:
        """Return a DataLoader that exposes num_samples items from this split.

        Sampling is done at pair granularity (stride over pairs, then expanded
        to consecutive ref/tgt index pairs) so reference and target images are
        never separated across batches or jobs. Batch size is rounded up to the
        nearest even number to guarantee each batch contains complete pairs.
        """
        num_pairs = len(self._pairs)
        want_pairs = max(1, min(num_samples // 2, num_pairs))
        stride = max(1, num_pairs // want_pairs)
        selected_pairs = list(range(0, num_pairs, stride))[:want_pairs]
        indices = [idx for p in selected_pairs for idx in (p * 2, p * 2 + 1)]
        batch = samples_per_job if samples_per_job is not None else len(indices)
        batch = max(2, batch + batch % 2)
        return DataLoader(
            Subset(self, indices),
            batch_size=batch,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def __len__(self) -> int:
        return len(self._pairs) * 2

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return a single image and pairing metadata.

        Items are ordered as ref0, tgt0, ref1, tgt1, ... so the evaluator can
        buffer ref outputs and pair them with the immediately following tgt.

        Parameters
        ----------
        item
            Dataset index (even = reference image, odd = target image).

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            image ``(1, H, W)`` float32 in ``[0, 1]``, and gt tuple
            ``(H_gt, is_ref)`` where H_gt is ``(3, 3)`` and is_ref is a
            scalar bool tensor (True for reference, False for target).
        """
        pair_idx = item // 2
        is_ref = item % 2 == 0
        seq_path, tgt_idx = self._pairs[pair_idx]

        img_name = "1.ppm" if is_ref else f"{tgt_idx}.ppm"

        with Image.open(seq_path / img_name) as pil:
            orig_w, orig_h = pil.size
            image = preprocess_PIL_image(
                pil.convert("L").resize(
                    (self.width, self.height), resample=Image.BILINEAR
                )
            )[0]  # (1, H, W)

        if is_ref:
            # Evaluator discards H_gt for ref items; skip the extra file I/O.
            H_gt = torch.zeros(3, 3, dtype=torch.float32)
        else:
            with Image.open(seq_path / "1.ppm") as pil_ref:
                ref_w, ref_h = pil_ref.size
            H_raw = np.loadtxt(str(seq_path / f"H_1_{tgt_idx}"), dtype=np.float64)
            H_gt = self._rescale_homography(
                H_raw,
                src_hw=(ref_h, ref_w),
                dst_hw=(orig_h, orig_w),
            )

        return image, (H_gt, torch.tensor(is_ref))

    def _rescale_homography(
        self,
        H: np.ndarray,
        src_hw: tuple[int, int],
        dst_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Rescale a raw homography from original resolution to evaluation resolution.

        Parameters
        ----------
        H
            Raw (3, 3) homography at original image resolution.
        src_hw
            Original (H, W) of the source (reference) image.
        dst_hw
            Original (H, W) of the destination (target) image.

        Returns
        -------
        torch.Tensor
            (3, 3) homography rescaled to (self.height, self.width).
        """
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        eval_h, eval_w = self.height, self.width

        S_src = np.array(
            [[eval_w / src_w, 0, 0], [0, eval_h / src_h, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        S_dst = np.array(
            [[eval_w / dst_w, 0, 0], [0, eval_h / dst_h, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        H_resized = S_dst @ H @ np.linalg.inv(S_src)
        return torch.from_numpy(H_resized.astype(np.float32))

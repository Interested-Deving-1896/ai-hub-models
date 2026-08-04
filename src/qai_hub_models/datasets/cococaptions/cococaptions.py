# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import clip
import torch
from torchvision.datasets import CocoCaptions
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from qai_hub_models.datasets.coco.coco import (
    COCO_ANNOTATIONS,
    COCO_VAL_DATASET,
    CocoDatasetBase,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

CAPTIONS_PER_IMAGE = 5


def add_prefix_to_captions(texts: list[str], prefix: str = "a photo of") -> list[str]:
    """Add a prefix to each caption.

    The default prefix "a photo of" is a prompt engineering technique from the
    CLIP paper that improves zero-shot retrieval performance by framing
    raw captions as natural image descriptions.
    """
    return [f"{prefix} {text}" for text in texts]


class CocoCaptionsDataset(CocoDatasetBase):
    """
    COCO 2017 val split with captions, for image-text retrieval evaluation.

    Each item is a single image paired with its 5 tokenized captions.
    Used to compute CLIP Recall@K (text-to-image and image-to-text).
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.coco_base = COCO_VAL_DATASET.extracted_path.parent

        anno_file = (
            "captions_val2017.json"
            if split == DatasetSplit.VAL
            else "captions_train2017.json"
        )
        self.root = (
            COCO_VAL_DATASET.extracted_path
            if split == DatasetSplit.VAL
            else self.coco_base / "train2017"
        )
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 224, 224))}
        n_px = input_spec["image"][0][2]

        self.transforms_pipeline = Compose(
            [
                Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(n_px),
                lambda img: img.convert("RGB"),
                ToTensor(),
            ]
        )
        super().__init__(split, input_spec)

        self._coco = CocoCaptions(
            root=str(self.root),
            annFile=str(COCO_ANNOTATIONS.extracted_path / anno_file),
        )

        if split == DatasetSplit.TRAIN:
            # Only a subset of train images are downloaded; filter to those IDs.
            downloaded_ids = {int(s["id"]) for s in self.train_samples}
            self._coco.ids = [id_ for id_ in self._coco.ids if id_ in downloaded_ids]

    def __len__(self) -> int:
        return len(self._coco)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the image, its tokenized captions, and the sample index.

        Parameters
        ----------
        index
            Dataset index.

        Returns
        -------
        (image, tokenized_captions) : tuple[torch.Tensor, torch.Tensor]
            image: float tensor [3, H, W], preprocessed values in [0,1].
            tokenized_captions: int tensor [CAPTIONS_PER_IMAGE, 77], CLIP tokens.

        index : torch.Tensor
            scalar tensor, used to build the ground-truth retrieval mapping.
        """
        pil_image, captions = self._coco[index]
        image = self.transforms_pipeline(pil_image)
        tokenized = clip.tokenize(
            add_prefix_to_captions(captions[:CAPTIONS_PER_IMAGE]), truncate=True
        )

        return (image, tokenized), torch.tensor(index)

    @staticmethod
    def default_samples_per_job() -> int:
        return 500

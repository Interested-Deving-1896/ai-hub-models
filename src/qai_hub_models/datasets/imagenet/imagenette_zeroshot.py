# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Imagenette dataset with optional joint image+text mode for calibration.

This is a calibration-only companion to :class:`ImagenetZeroshotDataset`.
It wraps :class:`ImagenetteDataset` (10-class, publicly downloadable, supports
TRAIN split) and adds the same joint-mode API used by
``ImagenetZeroshotDataset``:

* **Image-only** (default, ``tokenizer_id=None``):
  ``__getitem__`` returns ``(image, label)`` — same as ``ImagenetteDataset``.

* **Joint image+text** (``tokenizer_id`` provided):
  ``__getitem__`` returns ``(image, input_ids, label)``.
  ``input_ids`` is the tokenized ``"a photo of a {label}"`` prompt for the
  ground-truth class, making the dataset directly usable for calibrating
  *both* the image encoder and the text encoder of a vision-language model
  such as SigLIP2.

Only the 10 Imagenette class prompts are tokenized (not all 1000 ImageNet
classes), keeping construction fast and memory-light.
"""

from __future__ import annotations

import torch

from qai_hub_models.datasets.imagenet.imagenet_zeroshot import (
    PROMPT_TEMPLATE,
    tokenize_prompts,
)
from qai_hub_models.datasets.imagenet.imagenette import (
    IMAGENETTE_CLASS_MAP,
    ImagenetteDataset,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.labels import get_class_names


class ImagenetteZeroshotDataset(ImagenetteDataset):
    """Imagenette dataset for zero-shot image(-text) calibration.

    Parameters
    ----------
    split
        Dataset split.  Defaults to ``TRAIN`` (recommended for calibration).
    tokenizer_id
        Hugging Face model id whose tokenizer is used to encode class prompts.
        When ``None`` (default) the dataset operates in *image-only* mode and
        ``__getitem__`` returns ``(image, label)``.
        When provided the dataset operates in *joint* mode and ``__getitem__``
        returns ``(image, input_ids, label)``.
    text_seq_len
        Fixed sequence length for tokenized prompts.  Only used when
        ``tokenizer_id`` is not ``None``.

    Notes
    -----
    In joint mode the 10 Imagenette class prompts are tokenized once at
    construction time and cached in memory.  The ``input_ids`` returned for a
    given sample correspond to the ground-truth class of that sample (mapped to
    its ImageNet-1K id via ``IMAGENETTE_CLASS_MAP``), so the same dataset can
    be used to calibrate both the image encoder (use ``image``) and the text
    encoder (use ``input_ids``) of a vision-language model.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        tokenizer_id: str | None = None,
        text_seq_len: int = 64,
    ) -> None:
        super().__init__(split=split)

        # Pre-tokenize the 10 Imagenette class prompts when a tokenizer is
        # requested.  We index into the full ImageNet-1K label list using
        # IMAGENETTE_CLASS_MAP so the prompts match those used at eval time.
        self._class_input_ids: dict[int, torch.Tensor] | None = None
        if tokenizer_id is not None:
            imagenet_labels = get_class_names("imagenet")
            prompts = [
                PROMPT_TEMPLATE.format(label=imagenet_labels[imagenet_id])
                for imagenet_id in IMAGENETTE_CLASS_MAP.values()
            ]
            # Map each ImageNet-1K id → its tokenized prompt tensor [text_seq_len].
            input_ids = tokenize_prompts(prompts, tokenizer_id, text_seq_len)
            self._class_input_ids = {
                imagenet_id: input_ids[i]
                for i, imagenet_id in enumerate(IMAGENETTE_CLASS_MAP.values())
            }

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Return a single sample.

        Parameters
        ----------
        index
            Sample index into the dataset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(image, label)`` in image-only mode, or ``(image, input_ids, label)``
            in joint mode.  ``image`` is float32 ``[3, 224, 224]``; ``input_ids``
            is int32 ``[text_seq_len]`` (joint mode only); ``label`` is the
            ground-truth ImageNet-1K class index.
        """
        image, label = super().__getitem__(index)
        if self._class_input_ids is not None:
            # label is already the ImageNet-1K id (applied by target_transform).
            return image, self._class_input_ids[label], label
        return image, label

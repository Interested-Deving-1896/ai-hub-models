# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Collection

import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import RECALL_AT_1, MetricMetadata

DEFAULT_K_VALS = [1, 5, 10]


class CLIPRetrievalEvaluator(BaseEvaluator):
    """
    Evaluates CLIP using image-text retrieval Recall@K on COCO Captions.

    Accumulates normalized image and text embeddings across batches, then
    computes a global pairwise similarity matrix to measure Recall@K in
    both text-to-image and image-to-text directions.

    """

    def __init__(self, k_vals: list[int] | None = None) -> None:
        self.k_vals = k_vals or DEFAULT_K_VALS
        self.reset()

    def reset(self) -> None:
        self._image_feats: list[torch.Tensor] = []
        self._text_feats: list[torch.Tensor] = []
        self._image_indices: list[torch.Tensor] = []
        self._recall_cache: tuple[list[float], list[float]] | None = None

    def add_batch(
        self,
        output: torch.Tensor | Collection[torch.Tensor],
        gt: torch.Tensor | Collection[torch.Tensor],
    ) -> None:
        """
        Accumulate embeddings from one batch.

        Parameters
        ----------
        output
            Tuple of (image_features, text_features) from the model forward pass.
            image_features: [B, D] normalized image embeddings.
            text_features:  [B * captions_per_image, D] normalized text embeddings.
        gt
            [B] tensor of dataset image indices from CocoCaptionsDataset.__getitem__.
            Used to build the image↔text ground-truth retrieval maps.
        """
        image_features, text_features = output
        self._image_feats.append(image_features.detach().cpu())
        self._text_feats.append(text_features.detach().cpu())
        if isinstance(gt, torch.Tensor):
            self._image_indices.append(gt.cpu())
        else:
            self._image_indices.append(torch.tensor(gt))
        self._recall_cache = None

    def _compute_recall(self) -> tuple[list[float], list[float]]:
        if self._recall_cache is not None:
            return self._recall_cache

        image_encodings = torch.cat(self._image_feats)
        text_encodings = torch.cat(self._text_feats)
        image_indices = torch.cat(self._image_indices)

        num_im = image_encodings.shape[0]
        captions_per_image = text_encodings.shape[0] // num_im

        # Build retrieval maps from dataset image indices.
        # text_to_image_map[i] = dataset image index for the i-th text
        # image_to_text_map[j] = list of global text indices for the j-th image
        text_to_image_map = image_indices.repeat_interleave(captions_per_image)
        # Global text index for image j, caption c = j * captions_per_image + c
        image_to_text_map = torch.arange(num_im).unsqueeze(
            1
        ) * captions_per_image + torch.arange(captions_per_image)

        # text-to-image: dist[i] = similarity of i-th text vs all images
        dist_matrix = text_encodings @ image_encodings.T
        dist_matrix_cpu = dist_matrix.cpu()
        k_max = max(self.k_vals)
        _, inds = torch.topk(dist_matrix_cpu, k=k_max, dim=1, largest=True, sorted=True)

        # For each text, map top-k column indices back to dataset image indices
        retrieved_image_indices = image_indices[inds]

        t2i_recall = []
        for k in self.k_vals:
            topk = retrieved_image_indices[:, :k]
            correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
            t2i_recall.append(correct.float().mean().item() * 100.0)

        # image-to-text: dist[i] = similarity of i-th image vs all texts
        dist_matrix_T = dist_matrix_cpu.T
        _, inds_i2t = torch.topk(
            dist_matrix_T, k=k_max, dim=1, largest=True, sorted=True
        )

        i2t_recall = []
        for k in self.k_vals:
            topk = inds_i2t[:, :k]
            correct = torch.zeros(num_im, dtype=torch.bool)
            for cap_idx in range(captions_per_image):
                contains = torch.eq(
                    topk, image_to_text_map[:, cap_idx].unsqueeze(-1)
                ).any(dim=1)
                correct = torch.logical_or(correct, contains)
            i2t_recall.append(correct.float().mean().item() * 100.0)

        self._recall_cache = (t2i_recall, i2t_recall)
        return self._recall_cache

    def get_accuracy_score(self) -> float:
        """Returns text-to-image Recall@1 as the primary metric."""
        if not self._image_feats:
            return 0.0
        t2i, _ = self._compute_recall()
        return t2i[0]

    def formatted_accuracy(self) -> str:
        if not self._image_feats:
            return "No data"
        t2i, i2t = self._compute_recall()
        t2i_parts = ", ".join(
            f"R@{k}: {v:.1f}%" for k, v in zip(self.k_vals, t2i, strict=False)
        )
        i2t_parts = ", ".join(
            f"R@{k}: {v:.1f}%" for k, v in zip(self.k_vals, i2t, strict=False)
        )
        return f"T->I [{t2i_parts}] | I->T [{i2t_parts}]"

    def get_metric_metadata(self) -> MetricMetadata:
        return RECALL_AT_1

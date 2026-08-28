# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.owlvit.modeling_owlvit import OwlViTClassPredictionHead

MASK_FILL: float = -1e4


def _patched_class_prediction_head_forward(
    self: OwlViTClassPredictionHead,
    image_embeds: torch.FloatTensor,
    query_embeds: torch.FloatTensor | None,
    query_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for ``OwlViTClassPredictionHead.forward``.

    Identical to the upstream implementation except that the ``einsum``
    computing the dot-product between image patch embeddings and text query
    embeddings is replaced by an explicit ``unsqueeze + mul + sum`` sequence
    for better on-device compiler compatibility.

    Parameters
    ----------
    self
        OwlViTClassPredictionHead module instance.
    image_embeds
        Image patch features from the vision backbone.
        Shape: ``[batch, num_patches, query_dim]``
    query_embeds
        Normalised text query embeddings.
        Shape: ``[batch, num_queries, text_hidden_size]``
        When ``None`` a zero logit tensor is returned (no-query path).
    query_mask
        Boolean mask indicating valid queries.
        Shape: ``[batch, num_queries]``

    Returns
    -------
    pred_logits : torch.Tensor
        Per-patch per-query classification logits.
        Shape: ``[batch, num_patches, num_queries]``
    image_class_embeds : torch.Tensor
        Normalised image class embeddings (before logit shift/scale).
        Shape: ``[batch, num_patches, text_hidden_size]``
    """
    image_class_embeds = self.dense0(image_embeds)

    # No-query fast path (used during image-guided detection)
    if query_embeds is None:
        device = image_class_embeds.device
        batch_size, num_patches = image_class_embeds.shape[:2]
        pred_logits = torch.zeros(
            (batch_size, num_patches, self.query_dim), device=device
        )
        return (pred_logits, image_class_embeds)

    # Normalize image and text features
    image_class_embeds = image_class_embeds / (
        torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
    )
    query_embeds = query_embeds / (
        torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
    )

    # Compute per-patch per-query dot-products.
    # Upstream uses:
    #   pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
    #     pred_logits = (image_class_embeds.unsqueeze(-2) * query_embeds.unsqueeze(-3)).sum(
    #         dim=-1
    #     )

    pred_logits = torch.bmm(
        image_class_embeds,  # [B, P, D]
        query_embeds.transpose(1, 2),  # [B, D, Q]
    )

    # Apply learnable per-patch logit shift and scale
    logit_shift = self.logit_shift(image_embeds)
    logit_scale = self.logit_scale(image_embeds)
    logit_scale = self.elu(logit_scale) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale

    # Mask out padding queries
    if query_mask is not None:
        if query_mask.ndim > 1:
            query_mask = torch.unsqueeze(query_mask, dim=-2)
        pred_logits = pred_logits.masked_fill(query_mask == 0, MASK_FILL)
        pred_logits = pred_logits.to(torch.float32)

    return pred_logits, image_class_embeds


@staticmethod  # type: ignore[misc]
def _patched_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """
    Patched ``AttentionMaskConverter._make_causal_mask``.

    Identical to the upstream implementation except that the mask is
    initialised with ``MASK_FILL = -1e4`` instead of
    ``torch.finfo(dtype).min``.  This avoids NaN / overflow when the mask
    is later cast to float16 on-device.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), MASK_FILL, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )

    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window - 1
        context_mask = torch.tril(
            torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal
        )
        mask.masked_fill_(context_mask, MASK_FILL)

    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


@staticmethod  # type: ignore[misc]
def _patched_expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int | None = None,
) -> torch.Tensor:
    """
    Patched ``AttentionMaskConverter._expand_mask``.

    Identical to the upstream implementation except that the inverted mask
    is filled with ``MASK_FILL = -1e4`` instead of ``torch.finfo(dtype).min``
    to avoid overflow in float16 on-device paths.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), MASK_FILL)


def apply_patches() -> None:
    """
    Apply all OwlViT model patches.


    Patches applied
    ---------------
    1. ``OwlViTClassPredictionHead.forward`` - replaces ``einsum`` with
       ``unsqueeze + mul + sum`` for on-device compiler compatibility.
    2. ``AttentionMaskConverter._make_causal_mask`` - uses finite
       ``MASK_FILL = -1e4`` instead of ``torch.finfo(dtype).min``.
    3. ``AttentionMaskConverter._expand_mask`` - uses finite
       ``MASK_FILL = -1e4`` instead of ``torch.finfo(dtype).min``.
    """
    OwlViTClassPredictionHead.forward = _patched_class_prediction_head_forward  # type: ignore[assignment]
    AttentionMaskConverter._make_causal_mask = _patched_make_causal_mask
    AttentionMaskConverter._expand_mask = _patched_expand_mask

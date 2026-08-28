# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Model adaptations for Qwen2.5-VL text model: Split-Head Attention (SHA),
RMSNorm rank-4, and Conv2d conversions.

Reuses SHAQwen2Attention from the Qwen2 template since the SHA logic
is identical. The only difference is the parent class (Qwen2_5_VLAttention
vs Qwen2Attention) needed for the monkey-patch to work.
"""

from __future__ import annotations

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2MLP,
)

from qai_hub_models.models.templates.llm.model_adaptations import ConvInplaceLinear
from qai_hub_models.models.templates.qwen2.model_adaptations import SHAQwen2Attention


def _preprocess_mrope(
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-process M-RoPE cos/sin into format compatible with _apply_rope_single.

    Input cos/sin shape: (3, batch, seq, head_dim) — raw from Qwen2_5_VLRotaryEmbedding
    Output: (cos_half, sin_half) each with shape (batch, 1, seq, head_dim//2)

    M-RoPE selects temporal/height/width dimensions for different portions
    of head_dim. Since the underlying freqs are doubled (cat(freqs, freqs)),
    the first and second halves of the processed cos/sin are identical,
    so we only need the first half for _apply_rope_single.
    """
    mrope_section_2x = mrope_section * 2
    cos_full = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section_2x, dim=-1))],
        dim=-1,
    ).unsqueeze(1)  # (batch, 1, seq, head_dim)
    sin_full = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section_2x, dim=-1))],
        dim=-1,
    ).unsqueeze(1)  # (batch, 1, seq, head_dim)

    head_dim = cos_full.shape[-1]
    cos_half = cos_full[..., : head_dim // 2]
    sin_half = sin_full[..., : head_dim // 2]

    return cos_half, sin_half


class SHAQwen2_5_VLAttention(Qwen2_5_VLAttention):
    """Split-Head Attention for Qwen2.5-VL.

    Reuses prepare_conv, prepare_sha, and forward_sha from SHAQwen2Attention.
    Must inherit from Qwen2_5_VLAttention so the monkey-patch in model.py
    replaces the correct class.
    """

    prepare_conv = SHAQwen2Attention.prepare_conv
    prepare_sha = SHAQwen2Attention.prepare_sha
    forward_sha = SHAQwen2Attention.forward_sha


class QCQwen2_5_VLMLP(Qwen2MLP):
    """Qwen2.5-VL MLP with Conv2d adaptation for HTP backend."""

    def prepare_conv(self) -> None:
        # TODO (https://github.com/qcom-ai-hub/tetracode/issues/17113)
        # Temporarily commented out due to AISW-148745.
        # self.up_proj = ConvInplaceLinear(self.up_proj)
        self.down_proj = ConvInplaceLinear(self.down_proj)  # type: ignore[has-type, unused-ignore]
        # self.gate_proj = ConvInplaceLinear(self.gate_proj)

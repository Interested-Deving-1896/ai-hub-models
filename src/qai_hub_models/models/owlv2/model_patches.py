# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.owlv2.modeling_owlv2 import (
    Owlv2Attention,
    Owlv2ClassPredictionHead,
    Owlv2ForObjectDetection,
    Owlv2MLP,
)

from qai_hub_models.models.templates.owl.model_patches import (
    MASK_FILL,
    _patched_class_prediction_head_forward,
    _patched_expand_mask,
    _patched_make_causal_mask,
)

# Number of attention heads to process simultaneously in the SHA forward.
# Separate values for the vision and text encoders allow tuning each
# independently — the vision encoder has many more tokens (3601 for 960x960)
# so needs smaller groups to stay within on-device memory budgets, while the
# text encoder has only 16 tokens and can afford larger groups.
#
# Vision encoder (base: 12 heads, large: 16 heads)
#   VISION_HEAD_GROUP_SIZE = 3  →  4 groups (base) / ~5 groups (large)
#   attn matrix [3, 3601, 3601] ≈ 156 MB per group  ← w8a8 OK
#
# Text encoder (8 heads, seq_len = 16)
#   TEXT_HEAD_GROUP_SIZE = 2  →  4 groups
#   attn matrix [2, 16, 16] ≈ negligible
#
# Rules:
#   • group_size must evenly divide num_heads.
#   • If the chosen value does not divide num_heads, _split_attention_heads()
#     automatically walks it down to the nearest valid divisor.
VISION_HEAD_GROUP_SIZE: int = 3
TEXT_HEAD_GROUP_SIZE: int = 2


class _QuickGELU(nn.Module):
    """x * sigmoid(1.702 * x) — the activation used in OwlV2 / CLIP MLPs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(
            torch.tensor(1.702, dtype=x.dtype, device=x.device) * x
        )


class Owlv2MLPAIMET(nn.Module):
    """
    Two-layer MLP with QuickGELU activation — shared by VE and TE.

    Supports Linear mode (default) and Conv2d(1x1) mode after
    ``_linear_to_conv()``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = _QuickGELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.is_conv = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            return self.forward_conv(hidden_states)
        return self.forward_linear(hidden_states)

    def forward_linear(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)

    def forward_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (B, seq, C) → (B, C, seq, 1) for Conv2d
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(-1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.conv2(hidden_states)
        # (B, C, seq, 1) → (B, seq, C)
        return hidden_states.squeeze(-1).permute(0, 2, 1)

    def _linear_to_conv(self) -> None:
        """Convert fc1/fc2 (Linear) → conv1/conv2 (Conv2d 1x1) in-place."""
        self.conv1 = nn.Conv2d(
            self.hidden_size,
            self.intermediate_size,
            1,
            bias=(self.fc1.bias is not None),
        )
        self.conv2 = nn.Conv2d(
            self.intermediate_size,
            self.hidden_size,
            1,
            bias=(self.fc2.bias is not None),
        )
        with torch.no_grad():
            self.conv1.weight.copy_(self.fc1.weight[:, :, None, None])
            self.conv2.weight.copy_(self.fc2.weight[:, :, None, None])
            if self.fc1.bias is not None:
                self.conv1.bias.copy_(self.fc1.bias)  # type: ignore[union-attr]
            if self.fc2.bias is not None:
                self.conv2.bias.copy_(self.fc2.bias)  # type: ignore[union-attr]

        self.is_conv = True
        del self.fc1, self.fc2


def _split_attention_heads(
    attn: Owlv2Attention,
    group_size: int,
) -> None:
    """
    Convert an ``Owlv2Attention`` module from Linear MHA to per-head
    Conv2d(1x1) SHA in-place.

    After this call:
      - ``attn.q_convs``, ``attn.k_convs``, ``attn.v_convs`` are
        ``nn.ModuleList`` of ``num_heads`` Conv2d(embed_dim → head_dim, 1x1).
      - ``attn.out_conv`` is a Conv2d(embed_dim → embed_dim, 1x1).
      - ``attn.head_group_size`` is the resolved group size (largest divisor
        of ``num_heads`` that is ≤ ``group_size``).
      - ``attn.is_sha = True``.
      - The original ``q_proj``, ``k_proj``, ``v_proj``, ``out_proj``
        Linear layers are deleted.

    Parameters
    ----------
    attn
        The ``Owlv2Attention`` instance to convert.
    group_size
        Desired number of heads per group.
        If this value does not evenly divide ``num_heads``, it is walked down
        to the nearest divisor so that all heads are always covered.

    """
    H = attn.num_heads
    D = attn.head_dim
    E = attn.embed_dim

    # ── Per-head Conv2d projections ──────────────────────────────────────
    attn.q_convs = nn.ModuleList(
        [nn.Conv2d(E, D, 1, bias=(attn.q_proj.bias is not None)) for _ in range(H)]
    )
    attn.k_convs = nn.ModuleList(
        [nn.Conv2d(E, D, 1, bias=(attn.k_proj.bias is not None)) for _ in range(H)]
    )
    attn.v_convs = nn.ModuleList(
        [nn.Conv2d(E, D, 1, bias=(attn.v_proj.bias is not None)) for _ in range(H)]
    )

    # ── Copy sliced weights from original Linear layers ──────────────────
    q_convs_list = cast(nn.ModuleList, attn.q_convs)
    k_convs_list = cast(nn.ModuleList, attn.k_convs)
    v_convs_list = cast(nn.ModuleList, attn.v_convs)
    with torch.no_grad():
        for i in range(H):
            s, e = i * D, (i + 1) * D
            cast(nn.Conv2d, q_convs_list[i]).weight.copy_(
                attn.q_proj.weight[s:e, :, None, None]
            )
            cast(nn.Conv2d, k_convs_list[i]).weight.copy_(
                attn.k_proj.weight[s:e, :, None, None]
            )
            cast(nn.Conv2d, v_convs_list[i]).weight.copy_(
                attn.v_proj.weight[s:e, :, None, None]
            )
            if attn.q_proj.bias is not None:
                cast(nn.Conv2d, q_convs_list[i]).bias.copy_(attn.q_proj.bias[s:e])  # type: ignore[union-attr]
            if attn.k_proj.bias is not None:
                cast(nn.Conv2d, k_convs_list[i]).bias.copy_(attn.k_proj.bias[s:e])  # type: ignore[union-attr]
            if attn.v_proj.bias is not None:
                cast(nn.Conv2d, v_convs_list[i]).bias.copy_(attn.v_proj.bias[s:e])  # type: ignore[union-attr]

    # ── Output projection as Conv2d ──────────────────────────────────────
    attn.out_conv = nn.Conv2d(E, E, 1, bias=(attn.out_proj.bias is not None))
    with torch.no_grad():
        attn.out_conv.weight.copy_(attn.out_proj.weight[:, :, None, None])
        if attn.out_proj.bias is not None:
            attn.out_conv.bias.copy_(attn.out_proj.bias)  # type: ignore[union-attr]

    gs = min(group_size, H)
    while H % gs != 0:
        gs -= 1
    attn.head_group_size = gs

    attn.is_sha = True  # type: ignore[assignment]

    # Remove original Linear projections
    del attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj


def _patched_attention_forward(
    self: Owlv2Attention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    causal_attention_mask: torch.Tensor | None = None,
    output_attentions: bool | None = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Drop-in replacement for ``Owlv2Attention.forward``.

    SHA mode (``is_sha=True``, set by ``_split_attention_heads()``):
      - Reshapes input to ``[B, E, S, 1]`` for Conv2d.
      - Applies per-head Conv2d to get Q/K/V of shape ``[B, S, head_dim]``.
      - Processes VISION_HEAD_GROUP_SIZE=3 heads at a time:
          * Stacks Q/K/V → [B*group_size, S, head_dim]
          * Attention matrix [B*group_size, S, S] ≈ 208 MB per group
          * 3 groups total → 3x fewer MatMul nodes than per-head
      - Applies causal mask via ``masked_fill`` (→ MatMul→Where, not MatMul→Add).
      - Concatenates group outputs and applies output Conv2d.

    MHA fallback (``is_sha=False``):
      - Original HuggingFace MHA with mask support.
    """
    if not getattr(self, "is_sha", False):
        # ── Fallback: original HuggingFace MHA ──────────────────────────
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = cast(nn.Linear, self.q_proj)(hidden_states) * self.scale
        key_states = cast(nn.Linear, self.k_proj)(hidden_states)
        value_states = cast(nn.Linear, self.v_proj)(hidden_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = (
            query_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(*proj_shape)
        )
        key_states = (
            key_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(*proj_shape)
        )
        value_states = (
            value_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(*proj_shape)
        )

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if causal_attention_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
                + causal_attention_mask
            ).view(bsz * self.num_heads, tgt_len, tgt_len)
        if attention_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
                + attention_mask
            ).view(bsz * self.num_heads, tgt_len, tgt_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )
        attn_output = cast(nn.Linear, self.out_proj)(attn_output)
        return attn_output, None

    # ── Grouped SHA forward: per-head Conv2d + grouped attention ─────────
    B, seq_len, _ = hidden_states.shape

    # Collapse the two optional masks into one [B, S, S] bias.
    # With Patch 2 active, causal_attention_mask is always None for the text
    # encoder; the combined mask arrives as attention_mask.
    combined_mask: torch.Tensor | None = None
    if causal_attention_mask is not None:
        combined_mask = causal_attention_mask[:, 0, :, :]  # [B, S, S]
    if attention_mask is not None:
        attn_bias = attention_mask[:, 0, :, :]  # [B, S, S]
        combined_mask = (
            attn_bias if combined_mask is None else combined_mask + attn_bias
        )

    # Reshape input to (B, embed_dim, seq_len, 1) for Conv2d
    x = hidden_states.permute(0, 2, 1).unsqueeze(-1)

    # Use the group size resolved at setup time by _split_attention_heads().
    # That function already walked the value down to the nearest divisor of
    # num_heads, so every head is guaranteed to be covered here.
    group_size = int(self.head_group_size)  # type: ignore[arg-type]
    num_groups = self.num_heads // group_size

    group_outputs: list[torch.Tensor] = []
    for g in range(num_groups):
        # ── Per-head Conv2d projections for this group ───────────────────
        # Each Conv2d: (B, embed_dim, S, 1) → (B, head_dim, S, 1)
        # → squeeze + permute → (B, S, head_dim)
        q_convs = cast(nn.ModuleList, self.q_convs)
        k_convs = cast(nn.ModuleList, self.k_convs)
        v_convs = cast(nn.ModuleList, self.v_convs)
        q_heads = [
            cast(nn.Conv2d, q_convs[g * group_size + i])(x).squeeze(-1).permute(0, 2, 1)
            for i in range(group_size)
        ]
        k_heads = [
            cast(nn.Conv2d, k_convs[g * group_size + i])(x).squeeze(-1).permute(0, 2, 1)
            for i in range(group_size)
        ]
        v_heads = [
            cast(nn.Conv2d, v_convs[g * group_size + i])(x).squeeze(-1).permute(0, 2, 1)
            for i in range(group_size)
        ]

        # Stack → [B, group_size, S, head_dim] → reshape → [B*group_size, S, head_dim]
        q_g = torch.stack(q_heads, dim=1).reshape(
            B * group_size, seq_len, self.head_dim
        )
        k_g = torch.stack(k_heads, dim=1).reshape(
            B * group_size, seq_len, self.head_dim
        )
        v_g = torch.stack(v_heads, dim=1).reshape(
            B * group_size, seq_len, self.head_dim
        )

        # Attention weights: [B*group_size, S, S]
        attn_w = torch.bmm(q_g * self.scale, k_g.transpose(-2, -1))

        if combined_mask is not None:
            # Use masked_fill to avoid MatMul→Add pattern in ONNX.
            # Broadcast mask [B, S, S] → [B, group_size, S, S] → [B*group_size, S, S]
            attn_w = (
                attn_w.view(B, group_size, seq_len, seq_len)
                .masked_fill(combined_mask.unsqueeze(1) < 0, MASK_FILL)
                .view(B * group_size, seq_len, seq_len)
            )

        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = F.dropout(attn_w, p=self.dropout, training=self.training)

        # Context: [B*group_size, S, head_dim]
        ctx = torch.bmm(attn_w, v_g)

        # Reshape back: [B, group_size, S, head_dim] → [B, S, group_size*head_dim]
        ctx = (
            ctx.view(B, group_size, seq_len, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(B, seq_len, group_size * self.head_dim)
        )
        group_outputs.append(ctx)

    # Concatenate all groups: [B, S, embed_dim]
    out = torch.cat(group_outputs, dim=-1)

    # Output projection via Conv2d
    out = out.permute(0, 2, 1).unsqueeze(-1)  # (B, embed_dim, S, 1)
    out = cast(nn.Conv2d, self.out_conv)(out)
    out = out.squeeze(-1).permute(0, 2, 1)  # (B, S, embed_dim)

    return out, None


def apply_patches() -> None:
    """
    Apply class-level monkey-patches.

    Must be called **before** ``Owlv2ForObjectDetection.from_pretrained()``.

    Patches applied
    ---------------
    1. ``Owlv2Attention.forward`` - SHA with per-head weight slicing.
    2. ``Owlv2ClassPredictionHead.forward`` - einsum → unsqueeze+mul+sum.
    3. ``AttentionMaskConverter._make_causal_mask`` - finite mask fill.
    4. ``AttentionMaskConverter._expand_mask`` - finite mask fill.

    """
    Owlv2Attention.forward = _patched_attention_forward  # type: ignore[assignment]
    Owlv2ClassPredictionHead.forward = _patched_class_prediction_head_forward  # type: ignore[assignment]
    AttentionMaskConverter._make_causal_mask = _patched_make_causal_mask
    AttentionMaskConverter._expand_mask = _patched_expand_mask


def prepare_conv(model: Owlv2ForObjectDetection) -> None:
    """
    Prepare the model for on-device compilation.

    Transformations applied
    -----------------------
    A. **MLP**: Every ``Owlv2MLP`` → ``Owlv2MLPAIMET`` + ``_linear_to_conv()``.
       Conv2d(1x1) allows the QNN compiler to tile along the sequence dimension.

    B. **Attention**: ``_split_attention_heads()`` on every ``Owlv2Attention``
       → per-head Conv2d(1x1) projections for Q, K, V and output.
       The SHA forward processes them in groups of VISION_HEAD_GROUP_SIZE=3.

    Parameters
    ----------
    model
        Loaded ``Owlv2ForObjectDetection`` instance (after ``from_pretrained``).
    """
    # ── A. Replace Owlv2MLP → Owlv2MLPAIMET + Conv2d ────────────────────
    for parent_module in model.modules():
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, Owlv2MLP):
                continue

            hidden_size = child_module.fc1.in_features
            intermediate_size = child_module.fc1.out_features

            # Build replacement with the same dimensions
            new_mlp = Owlv2MLPAIMET(hidden_size, intermediate_size)

            # Copy pretrained weights from the original Linear layers
            with torch.no_grad():
                new_mlp.fc1.weight.copy_(child_module.fc1.weight)
                new_mlp.fc2.weight.copy_(child_module.fc2.weight)
                if child_module.fc1.bias is not None and new_mlp.fc1.bias is not None:
                    new_mlp.fc1.bias.copy_(child_module.fc1.bias)
                if child_module.fc2.bias is not None and new_mlp.fc2.bias is not None:
                    new_mlp.fc2.bias.copy_(child_module.fc2.bias)

            # Convert Linear → Conv2d(1x1) in-place
            new_mlp._linear_to_conv()

            # Swap the module in the parent
            setattr(parent_module, child_name, new_mlp)

    # ── B. Split Owlv2Attention → per-head Conv2d SHA ────────────────────
    # Apply encoder-specific group sizes:
    #   • Text encoder  (num_heads=8,  seq_len=16)   → TEXT_HEAD_GROUP_SIZE
    #   • Vision encoder (num_heads=12 base / 16 large, seq_len=3601) → VISION_HEAD_GROUP_SIZE
    for module in model.owlv2.text_model.modules():
        if isinstance(module, Owlv2Attention):
            _split_attention_heads(module, group_size=TEXT_HEAD_GROUP_SIZE)

    for module in model.owlv2.vision_model.modules():
        if isinstance(module, Owlv2Attention):
            _split_attention_heads(module, group_size=VISION_HEAD_GROUP_SIZE)

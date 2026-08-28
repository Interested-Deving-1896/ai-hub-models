# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Vision encoder adaptations for on-device export of Qwen2.5-VL.

These adaptations are required for QNN/HTP compatibility:
1. Conv3d -> Conv2d for patch embedding (Conv3d not supported on HTP)
2. Split fused QKV attention into separate Q, K, V Conv2d projections
3. Replace dynamic cu_seqlens attention with static attention masks
4. Convert all Linear layers to Conv2d(1x1) for HTP backend
5. Pre-compute RoPE and attention masks on host

Reference: Tutorial_for_Qwen2_5_VL_7B_IoT/example1/Example1A/veg_utils/qc_adaptation.py
"""

from __future__ import annotations

import functools
import math
from typing import Any

import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
)

from qai_hub_models.models.templates.llm.model_adaptations import ConvInplaceLinear


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to vision tensor."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.reshape(1, 1, sin.shape[0], sin.shape[1])
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


def _apply_rope_single(
    x: torch.Tensor, rope_vals: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Apply RoPE using pre-computed (cos, sin) tuple."""
    rope_real = rope_vals[0]  # shape: 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1]  # shape: 1, 1, seqlen, head_dim/2

    x_real = x[:, :, :, : x.shape[-1] // 2]
    x_im = x[:, :, :, x.shape[-1] // 2 :]
    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    return torch.cat((x_prod_real, x_prod_im), dim=3).view(*x.shape)


class Conv2dInplaceConv3d(nn.Module):
    """
    Replaces Conv3d patch embedding with Conv2d equivalent.

    Conv3d is not supported on QNN/HTP, so we convert the patch embedding
    to use Conv2d by:
    1. Flattening the temporal dimension into the channel dimension
    2. Converting and interleaving weights to match the new data layout
    """

    def __init__(
        self,
        conv3d: nn.Conv3d,
        in_channels: int = 3,
        temporal_patch_size: int = 2,
        patch_size: int = 14,
    ) -> None:
        """Only stride=kernel_size (non-overlapping) and bias=False are supported."""
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size

        # New Conv2d: temporal dim folded into channel dim
        inc_2d = in_channels * temporal_patch_size
        outc = conv3d.out_channels
        self.conv2d = nn.Conv2d(
            in_channels=inc_2d,
            out_channels=outc,
            kernel_size=(patch_size, patch_size),
            bias=False,
        )

        # Convert 3D weights to 2D with proper channel interleaving
        with torch.no_grad():
            # Original: (outc, in_channels, temporal, patch_h, patch_w)
            # Concat temporal slices along channel dim
            temporal_slices = [
                conv3d.weight.data[:, :, t, :, :] for t in range(temporal_patch_size)
            ]
            # (outc, in_channels * temporal_patch_size, patch_h, patch_w)
            concat_weights = torch.cat(temporal_slices, dim=1)

            # Interleave channels to match input data layout:
            # Input is laid out as (c0_t0, c0_t1, c1_t0, c1_t1, c2_t0, c2_t1, ...)
            # After cat, weights are (c0_t0, c1_t0, c2_t0, c0_t1, c1_t1, c2_t1, ...)
            # Need to reorder weight channels to match input layout
            indices: list[int] = []
            for c in range(in_channels):
                indices.extend(t * in_channels + c for t in range(temporal_patch_size))

            interleaved = torch.cat(
                [concat_weights[:, idx : idx + 1, :, :] for idx in indices],
                dim=1,
            )
            self.conv2d.weight.data.copy_(interleaved)
        self.conv2d.to(conv3d.weight.data.device)

    @property
    def weight(self) -> torch.Tensor:
        """Expose weight for compatibility with HF PatchEmbed that accesses self.proj.weight."""
        return self.conv2d.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, in_channels, temporal, patch_h, patch_w)
        # from PatchEmbed reshape. Flatten to (batch, C*T, H, W)
        x = torch.reshape(
            x,
            (
                -1,
                self.in_channels * self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ),
        )
        output = self.conv2d(x)
        # Add back temporal dim to match Conv3d output shape
        # Conv3d output: (batch, outc, 1, 1, 1) -> we need (batch, outc, 1, 1, 1)
        return output.unsqueeze(2)


class Qwen2_5_VLVisionAttentionAdaptation(nn.Module):
    """
    Adapted vision attention with split Q/K/V Conv2d projections.

    Replaces the fused QKV linear layer with separate Q, K, V Conv2d(1x1)
    projections and uses explicit attention masks instead of dynamic cu_seqlens.
    """

    def __init__(self, visual: Qwen2_5_VLVisionAttention) -> None:
        super().__init__()
        self.num_heads = visual.num_heads
        self.head_dim = visual.head_dim
        self.dim = visual.qkv.in_features
        self.bias = visual.qkv.bias is not None

        # Output projection -> Conv2d
        o = nn.Conv2d(
            visual.proj.in_features,
            visual.proj.out_features,
            1,
            bias=visual.proj.bias is not None,
        )
        o.weight.data.copy_(visual.proj.weight.data[:, :, None, None])
        if visual.proj.bias is not None:
            assert o.bias is not None
            o.bias.data.copy_(visual.proj.bias.data)
        o.to(visual.proj.weight.device)
        self.proj = o

        # Split fused QKV into separate Q, K, V Conv2d projections
        self.q, self.k, self.v = self._split_qkv(visual)
        del visual

    def _split_qkv(
        self, visual: Qwen2_5_VLVisionAttention
    ) -> tuple[nn.Conv2d, nn.Conv2d, nn.Conv2d]:
        """Split fused QKV linear into separate Q, K, V Conv2d layers."""
        q = nn.Conv2d(self.dim, self.dim, 1, bias=self.bias)
        k = nn.Conv2d(self.dim, self.dim, 1, bias=self.bias)
        v = nn.Conv2d(self.dim, self.dim, 1, bias=self.bias)

        qkv_weights = visual.qkv.weight.data
        q.weight.data.copy_(qkv_weights[: self.dim, :, None, None])
        k.weight.data.copy_(qkv_weights[self.dim : self.dim * 2, :, None, None])
        v.weight.data.copy_(qkv_weights[self.dim * 2 :, :, None, None])

        if self.bias:
            assert visual.qkv.bias is not None
            qkv_bias = visual.qkv.bias.data
            assert q.bias is not None and k.bias is not None and v.bias is not None
            q.bias.data.copy_(qkv_bias[: self.dim])
            k.bias.data.copy_(qkv_bias[self.dim : self.dim * 2])
            v.bias.data.copy_(qkv_bias[self.dim * 2 :])

        device = qkv_weights.device
        q.to(device)
        k.to(device)
        v.to(device)
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        # Reshape for Conv2d: (seq, dim) -> (1, dim, 1, seq)
        hidden_states = torch.reshape(
            hidden_states, (-1, seq_length, 1, self.num_heads * self.head_dim)
        ).transpose(1, 3)

        # Project Q, K, V
        q = (
            self.q(hidden_states)
            .reshape(-1, self.num_heads, self.head_dim, seq_length)
            .transpose(2, 3)
        )
        k = (
            self.k(hidden_states)
            .reshape(-1, self.num_heads, self.head_dim, seq_length)
            .transpose(2, 3)
        )
        v = (
            self.v(hidden_states)
            .reshape(-1, self.num_heads, self.head_dim, seq_length)
            .transpose(2, 3)
        )

        # Apply RoPE
        if isinstance(rotary_pos_emb, (tuple, list)):
            # Pre-computed (cos, sin) from VEG
            q = _apply_rope_single(q, rotary_pos_emb)
            k = _apply_rope_single(k, rotary_pos_emb)
        else:
            # position_embeddings from HF model (before VEG adaptation)
            assert position_embeddings is not None
            cos, sin = position_embeddings
            q = apply_rotary_pos_emb_vision(q, cos, sin)
            k = apply_rotary_pos_emb_vision(k, cos, sin)

        # Compute attention with explicit mask instead of cu_seqlens
        if attention_mask is None:
            # Fallback: build mask from cu_seqlens
            attention_mask = torch.full(
                [1, seq_length, seq_length],
                torch.finfo(q.dtype).min,
                device=q.device,
                dtype=q.dtype,
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask[
                    ...,
                    cu_seqlens[i - 1] : cu_seqlens[i],
                    cu_seqlens[i - 1] : cu_seqlens[i],
                ] = 0

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            -1, seq_length, 1, self.num_heads * self.head_dim
        )
        attn_output = attn_output.transpose(1, 3)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.transpose(1, 3)
        return attn_output.reshape(seq_length, self.num_heads * self.head_dim)


class Qwen2_5_VLVisionBlockAdaptation(nn.Module):
    """
    Adapted vision block that accepts pre-computed attention masks and RoPE.

    Wraps original block components but with a new forward signature
    that accepts attention_mask instead of relying on cu_seqlens.
    """

    def __init__(self, block: Qwen2_5_VLVisionBlock) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.attn = block.attn
        self.mlp = block.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


# Utility functions for replacing modules


def _rsetattr(obj: Any, attr: str, val: Any) -> None:
    pre, _, post = attr.rpartition(".")
    setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def replace_visual_attention_with_adaptation(
    model: nn.Module,
) -> nn.Module:
    """
    Replace all VisionBlock and VisionAttention modules with adapted versions.

    Must replace blocks first (which preserves attn reference),
    then replace attention modules with the Conv2d/split-QKV version.
    """
    # Replace blocks first
    for name, module in model.named_modules():
        if isinstance(module, Qwen2_5_VLVisionBlock):
            _rsetattr(model, name, Qwen2_5_VLVisionBlockAdaptation(module))

    # Then replace attention modules
    for name, module in model.named_modules():
        if isinstance(module, Qwen2_5_VLVisionAttention):
            _rsetattr(model, name, Qwen2_5_VLVisionAttentionAdaptation(module))

    return model


def replace_linears_with_convs(model: nn.Module) -> nn.Module:
    """Replace all remaining Linear layers with Conv2d(1x1) for HTP backend."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _rsetattr(model, name, ConvInplaceLinear(module))
    return model

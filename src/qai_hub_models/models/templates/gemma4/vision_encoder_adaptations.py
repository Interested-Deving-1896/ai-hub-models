# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""QNN-friendly adaptations for the Gemma4 vision tower (VEG).

Three transformations turn the HF vision path into an ONNX graph the QNN
converter maps efficiently, keeping numerics identical (cosine ~1.0 vs FP32):

1. Attention: static pre-computed RoPE + float additive mask instead of dynamic;
   the pooler becomes a static ``AvgPool2d`` so the graph shape is fixed.
2. RMSNorm -> :class:`QNNCompatibleRMSNorm`, whose op sequence the converter
   fuses into a single RmsNorm.
3. Linear -> :class:`ConvInplaceLinear` (1x1 Conv2d), which HTP maps efficiently.
"""

from __future__ import annotations

import functools
import warnings

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ClippableLinear,
    Gemma4RMSNorm,
    Gemma4VisionAttention,
    Gemma4VisionEncoder,
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
)


def _rsetattr(obj: object, attr: str, val: object) -> None:
    pre, _, post = attr.rpartition(".")
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj: object, attr: str, *args: object) -> object:
    def _getattr(obj: object, attr: object) -> object:
        return getattr(obj, attr, *args)  # type: ignore[call-overload, unused-ignore]

    return functools.reduce(_getattr, [obj, *attr.split(".")])


# ---------------------------------------------------------------------------
# RoPE helpers (explicit / static, ONNX-traceable)
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb_simple(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 2
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multidim_rope_simple(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    half = cos.shape[-1] // 2
    cos_x = cos[..., :half]
    cos_y = cos[..., half:]
    sin_x = sin[..., :half]
    sin_y = sin[..., half:]
    x_x = x[..., :half]
    x_y = x[..., half:]
    return torch.cat(
        [
            apply_rotary_pos_emb_simple(x_x, cos_x, sin_x),
            apply_rotary_pos_emb_simple(x_y, cos_y, sin_y),
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Attention / encoder / patch-embedder / pooler adaptations
# ---------------------------------------------------------------------------


class Gemma4VisionAttentionAdaptation(nn.Module):
    """Explicit-RoPE, static-mask attention replacing ``Gemma4VisionAttention``."""

    def __init__(self, attn: Gemma4VisionAttention) -> None:
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.v_norm = attn.v_norm
        self.num_heads = attn.config.num_attention_heads
        self.head_dim = attn.head_dim
        self.embed_dim = attn.config.hidden_size
        self.scaling = attn.scaling
        self.attention_dropout = attn.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_norm(key_states)
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        value_states = self.v_norm(value_states)

        query_states = apply_multidim_rope_simple(query_states, cos, sin)
        key_states = apply_multidim_rope_simple(key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, self.embed_dim)
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def _make_float_attn_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    mask_value: float = -10000.0,
) -> torch.Tensor:
    batch, seq = attention_mask.shape
    mask_float = attention_mask.to(dtype=dtype)
    mask_float = mask_float.view(batch, 1, 1, seq).expand(batch, 1, seq, seq)
    inverted = 1.0 - mask_float
    return inverted * mask_value


class Gemma4VisionEncoderAdaptation(nn.Module):
    """Encoder wrapper that builds a static float mask + RoPE once, up front."""

    def __init__(self, encoder: Gemma4VisionEncoder) -> None:
        super().__init__()
        self.rotary_emb = encoder.rotary_emb
        self.layers = encoder.layers
        self.config = encoder.config

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> BaseModelOutputWithPast:
        float_attn_mask = _make_float_attn_mask(
            attention_mask, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_out = decoder_layer(
                hidden_states,
                attention_mask=float_attn_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                **kwargs,
            )
            hidden_states = (
                layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            )

        return BaseModelOutputWithPast(last_hidden_state=hidden_states.float())  # type: ignore[arg-type, unused-ignore]


class Gemma4VisionPatchEmbedderAdaptation(nn.Module):
    """Patch embedder that gathers 2-D position embeddings statically."""

    def __init__(self, embedder: Gemma4VisionPatchEmbedder) -> None:
        super().__init__()
        self.input_proj = embedder.input_proj
        self.position_embedding_table = embedder.position_embedding_table
        self.position_embedding_size = embedder.position_embedding_size

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped_positions = pixel_position_ids * (pixel_position_ids >= 0).long()
        table_flat = self.position_embedding_table.reshape(
            -1, self.position_embedding_table.shape[-1]
        )
        offset = torch.tensor(
            [0, self.position_embedding_size],
            dtype=torch.long,
            device=clamped_positions.device,
        )
        idx_flat = clamped_positions + offset
        position_embeddings = (
            table_flat[idx_flat].to(self.position_embedding_table.dtype).sum(dim=2)
        )
        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(
            pixel_position_ids, padding_positions
        )
        return hidden_states + position_embeddings


class Gemma4VisionPoolerStaticAdaptation(nn.Module):
    """Static ``AvgPool2d`` pooler with a fixed grid (shape known at export)."""

    def __init__(
        self,
        pooler: Gemma4VisionPooler,
        k: int,
        output_length: int,
        grid_W: int,
        grid_H: int,
    ) -> None:
        super().__init__()
        self.root_hidden_size = pooler.root_hidden_size
        self.k = k
        self.grid_W = grid_W
        self.grid_H = grid_H
        self.grid_size = grid_W * grid_H
        self.pooled_length = (grid_H // k) * (grid_W // k)
        self.pad_length = output_length - self.pooled_length
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=k)
        if self.pad_length > 0:
            self.register_buffer(
                "pad_zeros",
                torch.zeros(
                    1,
                    self.pad_length,
                    pooler.root_hidden_size.shape[0]
                    if hasattr(pooler.root_hidden_size, "shape")
                    else 768,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = hidden_states.shape
        grid = hidden_states[:, : self.grid_size, :].reshape(
            batch, self.grid_H, self.grid_W, channels
        )
        pooled = (
            self.avg_pool(grid.permute(0, 3, 1, 2).float())
            .permute(0, 2, 3, 1)
            .reshape(batch, self.pooled_length, channels)
            .to(hidden_states.dtype)
        )
        if self.pad_length > 0:
            assert self.pad_zeros is not None
            pad = self.pad_zeros.expand(batch, -1, -1).to(  # type: ignore[operator, unused-ignore]
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            pooled = torch.cat([pooled, pad], dim=1)
        output = pooled * self.root_hidden_size  # type: ignore[operator, unused-ignore]
        mask = output.any(dim=-1)
        return output, mask


def replace_gemma4_attention_with_adaptation(
    model: torch.nn.Module,
    k: int = 3,
    output_length: int = 280,
    pixel_position_ids: torch.Tensor | None = None,
) -> torch.nn.Module:
    """Replace attention / encoder / pooler / patch-embedder with adaptations.

    ``pixel_position_ids`` is required so the pooler grid (grid_W/grid_H) can be
    frozen as static constants matching the export resolution.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4VisionAttention):
            _rsetattr(model, name, Gemma4VisionAttentionAdaptation(module))
    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4VisionEncoder):
            _rsetattr(model, name, Gemma4VisionEncoderAdaptation(module))
    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4VisionPooler):
            if pixel_position_ids is None:
                raise ValueError(
                    "pixel_position_ids must be provided so grid_W and grid_H "
                    "can be fixed as static constants."
                )
            clamped = pixel_position_ids * (pixel_position_ids >= 0).long()
            grid_W = int(clamped[..., 0].max().item()) + 1
            grid_H = int(clamped[..., 1].max().item()) + 1
            _rsetattr(
                model,
                name,
                Gemma4VisionPoolerStaticAdaptation(
                    module,
                    k=k,
                    output_length=output_length,
                    grid_W=grid_W,
                    grid_H=grid_H,
                ),
            )
    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4VisionPatchEmbedder):
            _rsetattr(model, name, Gemma4VisionPatchEmbedderAdaptation(module))
    return model


# ---------------------------------------------------------------------------
# RMSNorm adaptation
# ---------------------------------------------------------------------------


class QNNCompatibleRMSNorm(nn.Module):
    """Drop-in ``Gemma4RMSNorm`` replacement the QNN converter fuses to RmsNorm.

    Two changes vs the original ``forward`` so the converter's pattern matches:
    no ``float()``/``type_as()`` Cast ops (Cast breaks the matcher), and
    ``x / sqrt(mean_sq + eps)`` instead of ``x * pow(mean_sq, -0.5)`` (the
    pattern needs a Sqrt op; ``Pow(-0.5)`` does not match).
    """

    def __init__(self, eps: float, weight: torch.nn.Parameter | None = None) -> None:
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        normed = hidden_states / torch.sqrt(mean_squared)
        if hasattr(self, "weight"):
            normed = normed * self.weight
        return normed


def replace_gemma4_rmsnorm_with_standard(model: torch.nn.Module) -> torch.nn.Module:
    """Replace all ``Gemma4RMSNorm`` instances with :class:`QNNCompatibleRMSNorm`."""
    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4RMSNorm):
            new_norm = QNNCompatibleRMSNorm(
                eps=module.eps,
                weight=module.weight if module.with_scale else None,
            )
            _rsetattr(model, name, new_norm)
    return model


# ---------------------------------------------------------------------------
# Clippable-linear bound freezing (works around an ONNX exporter fusion bug)
# ---------------------------------------------------------------------------


class ScalarClippedLinear(nn.Module):
    """``Gemma4ClippableLinear`` with its clamp bounds frozen to Python floats.

    Numerically identical to the original: HF stores each bound as a 0-dim buffer
    and calls ``torch.clamp``; this passes the very same values as Python floats.
    What changes is the exported graph. With tensor bounds the exporter emits
    ``Max``/``Min`` pairs that onnxscript's optimizer then fuses into ``Clip``,
    deriving the bound initializer names from the *data* tensor being clamped
    (``mul_7_min``, ``mul_7_min_1``, ...). Every clipped linear reading a shared
    hidden state therefore competes for the same name family, and the fused graph
    ends up referencing bound initializers that were never registered -- 96 of
    them for the E2B VEG -- so ``onnx.checker`` rejects the model with
    "Nodes in a graph must be topologically sorted". Float bounds are emitted as
    a per-call-site ``Clip`` with its own constants, so no fusion or renaming
    happens and each linear keeps its own ranges.

    Non-finite bounds drop the clamp entirely: HF initializes the buffers to
    +/-inf, and clamping to infinity is a no-op.
    """

    def __init__(self, clippable: Gemma4ClippableLinear) -> None:
        super().__init__()
        self.linear = clippable.linear
        self.input_bounds = _finite_bounds(clippable, "input")
        self.output_bounds = _finite_bounds(clippable, "output")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.input_bounds is not None:
            hidden_states = torch.clamp(hidden_states, *self.input_bounds)
        hidden_states = self.linear(hidden_states)
        if self.output_bounds is not None:
            hidden_states = torch.clamp(hidden_states, *self.output_bounds)
        return hidden_states


def _finite_bounds(
    clippable: Gemma4ClippableLinear, side: str
) -> tuple[float, float] | None:
    """``(min, max)`` floats for ``side`` ("input"/"output"), or None if unclipped."""
    if not clippable.use_clipped_linears:
        return None
    low = float(getattr(clippable, f"{side}_min").item())
    high = float(getattr(clippable, f"{side}_max").item())
    if low == -float("inf") and high == float("inf"):
        return None
    return low, high


def replace_clippable_linears_with_scalar_bounds(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """Swap every ``Gemma4ClippableLinear`` for a :class:`ScalarClippedLinear`."""
    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4ClippableLinear):
            _rsetattr(model, name, ScalarClippedLinear(module))
    return model


# ---------------------------------------------------------------------------
# Linear -> Conv2d replacement (QNN-friendly)
# ---------------------------------------------------------------------------


class ConvInplaceLinear(torch.nn.Module):
    """A 1x1 Conv2d module that replaces a Linear layer in place."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.conv2d = torch.nn.Conv2d(
            linear.in_features,
            linear.out_features,
            1,
            bias=linear.bias is not None,
        )
        self.conv2d.weight.data.copy_(linear.weight.data[:, :, None, None])
        if linear.bias is not None:
            self.conv2d.bias.data.copy_(linear.bias.data)  # type: ignore[union-attr, unused-ignore]
        self.conv2d.to(linear.weight.data.device)

    def __getattr__(self, attr: str) -> torch.Tensor | torch.nn.Module:
        conv2d = self._modules["conv2d"]
        if attr == "conv2d":
            return conv2d  # type: ignore[return-value, unused-ignore]
        assert conv2d is not None
        return getattr(conv2d, attr)  # type: ignore[return-value, unused-ignore]

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(2)  # (seq, C) -> (1, C, 1, seq)
            x = x.permute(0, 3, 2, 1)
        elif ndim == 3:
            x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, seq, C) -> (B, C, seq, 1)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            warnings.warn(
                "ConvInplaceLinear received an unexpected 4d input, "
                "assuming channels-last and proceeding.",
                stacklevel=2,
            )
        else:
            raise NotImplementedError(
                f"ConvInplaceLinear could not handle input with shape {x.shape}"
            )

        x = self.conv2d(x)

        if ndim == 2:
            return x.permute(0, 3, 2, 1).squeeze(0).squeeze(1)
        if ndim == 3:
            return x.squeeze(-1).permute(0, 2, 1)
        if ndim == 4:
            x = x.permute(0, 2, 3, 1)
        return x


def replace_linears_with_convs(model: torch.nn.Module) -> torch.nn.Module:
    """Replace every ``nn.Linear`` with a :class:`ConvInplaceLinear` (1x1 Conv2d).

    ``Gemma4ClippableLinear`` / :class:`ScalarClippedLinear` wrap an inner
    ``nn.Linear`` (``.linear``); this walks all submodules and swaps the leaf
    ``nn.Linear`` layers, so the clippable wrappers are handled transparently.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            _rsetattr(model, name, ConvInplaceLinear(module))
    return model


__all__ = [
    "ConvInplaceLinear",
    "Gemma4ClippableLinear",
    "Gemma4VisionAttentionAdaptation",
    "Gemma4VisionEncoderAdaptation",
    "Gemma4VisionPatchEmbedderAdaptation",
    "Gemma4VisionPoolerStaticAdaptation",
    "QNNCompatibleRMSNorm",
    "ScalarClippedLinear",
    "apply_multidim_rope_simple",
    "replace_clippable_linears_with_scalar_bounds",
    "replace_gemma4_attention_with_adaptation",
    "replace_gemma4_rmsnorm_with_standard",
    "replace_linears_with_convs",
]

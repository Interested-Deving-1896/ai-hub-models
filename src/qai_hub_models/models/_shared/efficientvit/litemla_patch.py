# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared quantization fix for EfficientViT's LiteMLA attention normalization.

All EfficientViT variants (b2_cls, l2_cls, l2_seg) build the same ``LiteMLA``
ReLU-linear attention block, whose normalization divides by
``out[..., -1:] + self.eps`` with ``eps = 1e-15`` (see the vendored
``efficientvit/models/nn/ops.py``). Because q and k are ReLU'd, that denominator
is >= 0 and frequently lands at (or arbitrarily close to) 0. Under w8a16 the
denominator is a uint16 activation with zero_point 0, so any value below
``scale / 2`` quantizes to integer 0:

  * On CPU / fake-quant the divide is fp32 -> 0/0 = NaN, which the very next
    QuantizeLinear clamps back to integer 0, so accuracy survives (~77%).
  * On the HTP the fused int16 fixed-point reciprocal of integer 0 saturates to
    garbage with no sanitizing requantize step, so on-device accuracy collapses
    (b2_cls w8a16 measured ~1% top1 before this fix).

Flooring the denominator with ``clamp(min=eps)`` for a quant-safe ``eps``
guarantees a nonzero quantized denominator and traces to an ONNX Clip/Max, so the
fix reaches the AI Hub server quantize path. The optimal eps is model-specific
(it trades int reciprocal resolution against a small denominator bias and must
stay below the value that trips the server's mixed-precision auto-promotion), so
it is passed in by each model rather than hardcoded here.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def _make_floored_relu_linear_att(
    eps: float,
) -> Callable[[Any, torch.Tensor], torch.Tensor]:
    """Build a replacement LiteMLA.relu_linear_att that floors the denominator.

    Mirrors the upstream method verbatim except the division denominator uses
    ``torch.clamp(min=eps)`` instead of ``+ self.eps``.

    Parameters
    ----------
    eps
        Quant-safe denominator floor.

    Returns
    -------
    Callable[[Any, torch.Tensor], torch.Tensor]
        A function with the LiteMLA.relu_linear_att signature, suitable for
        binding as an instance method.
    """

    def _floored_relu_linear_att(self: Any, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        denom = torch.clamp(out[..., -1:], min=eps)
        out = out[..., :-1] / denom

        out = torch.transpose(out, -1, -2)
        return torch.reshape(out, (B, -1, H, W))

    return _floored_relu_linear_att


def apply_attention_denominator_floor(model: torch.nn.Module, eps: float) -> int:
    """Floor the LiteMLA attention-normalization denominator on every LiteMLA module.

    Rebinds ``relu_linear_att`` on each ``LiteMLA`` instance to a copy whose
    division floors the denominator with ``clamp(min=eps)``. Call this from a
    model's ``from_pretrained`` for quantized precisions only (the float graph
    should stay identical to upstream).

    Parameters
    ----------
    model
        The EfficientViT torch model (classification or segmentation).
    eps
        Quant-safe denominator floor. Tuned per model.

    Returns
    -------
    int
        Number of LiteMLA modules patched (for sanity-checking / logging).
    """
    floored = _make_floored_relu_linear_att(eps)
    count = 0
    for module in model.modules():
        if type(module).__name__ == "LiteMLA":
            module.relu_linear_att = types.MethodType(floored, module)  # type: ignore[assignment]
            count += 1
    return count

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch


def _patch_vjepa2_rope(modeling_module: Any) -> None:
    """
    Replace transformers' ``rotate_queries_or_keys`` to drop an illegal ``squeeze(-1)``.

    The stock HF implementation calls ``emb_sin.squeeze(-1)`` / ``emb_cos.squeeze(-1)``
    on a ``[B, num_heads, N, D/2]`` tensor whose last dim (D/2) is not 1. Under eager
    PyTorch this squeeze is a silent no-op, but AI Hub's cloud Torch->ONNX exporter
    rejects "squeeze on a non-unit dimension" and aborts. Dropping the no-op squeeze is
    numerically identical (verified cos == 1.0, max abs diff 0.0) and lets the model
    export through the AI Hub cloud path.
    """
    if getattr(modeling_module, "_qaihm_rope_patched", False):
        return

    def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, num_heads, n, d = x.size()
        omega = torch.arange(d // 2, dtype=x.dtype, device=x.device)
        omega /= d / 2.0
        omega = 1.0 / 10000**omega
        freq = pos.unsqueeze(-1) * omega
        emb_sin = freq.sin().repeat(1, 1, 1, 2)
        emb_cos = freq.cos().repeat(1, 1, 1, 2)
        # Original uses x.unflatten(-1, (-1, 2)) + y.flatten(-2); negative dims force
        # the cloud ONNX exporter to know the rank at trace time and it aborts with
        # "ONNX export of operator dim ... Input rank must be known". Use explicit
        # static shapes (all dims are known here) so the export succeeds. Numerically
        # identical to the stock rotate-half.
        y = x.reshape(b, num_heads, n, d // 2, 2)
        y1 = y[..., 0]
        y2 = y[..., 1]
        y = torch.stack((-y2, y1), dim=-1)
        y = y.reshape(b, num_heads, n, d)
        return (x * emb_cos) + (y * emb_sin)

    modeling_module.rotate_queries_or_keys = rotate_queries_or_keys
    modeling_module._qaihm_rope_patched = True

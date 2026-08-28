# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.nn.modules.block import MaxSigmoidAttnBlock
from ultralytics.nn.modules.head import BNContrastiveHead


class MaxSigmoidAttnBlockInf(MaxSigmoidAttnBlock):
    def __init__(self, source: MaxSigmoidAttnBlock) -> None:
        torch.nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Patched forward - replaces einsum with equivalent ops for QAIRT compatibility.

        Parameters
        ----------
        x
            Image features (b, c, h, w).
        guide
            Text/guide features (b, n, c).

        Returns
        -------
        torch.Tensor
            Attended image features (b, c, h, w).

        Notes
        -----
        Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L593-L619
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        # Begin Qualcomm modification
        # Replaced einsum operator with follow expression ("bmchw,bnmc->bmhwn", embed, guide)
        embed_flat = embed.permute(0, 1, 3, 4, 2).reshape(bs, self.nh, h * w, self.hc)
        guide_t = guide.permute(0, 2, 3, 1)

        # NOTE: torch.matmul was intentionally avoided here.
        # Using torch.matmul (or the equivalent via permute) on batched 4D x 3D tensors
        # triggers q::Batch_MatMul_Bias_w_scale.B in QAIRT, which is not registered
        # and causes a graph preparation failure in w8a16 quantized inference:
        #   tcm_migration.cc:2034::ERROR: no properties registered for q::Batch_MatMul_Bias_w_scale.B
        #   graph_prepare.cc:198::ERROR: could not create op: q::Batch_MatMul_Bias_w_scale.B
        # The unsqueeze + element-wise multiply + sum pattern below avoids this op
        # and produces equivalent output while remaining compatible with QAIRT w8a16.
        # Equivalent to matmul: (b, n, c) x (b, c, k) -> (b, n, k)
        aw = (embed_flat.unsqueeze(-1) * guide_t.unsqueeze(2)).sum(dim=-2)

        aw = aw.view(bs, self.nh, h, w, -1)
        # End Qualcomm modification

        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class BNContrastiveHeadInf(BNContrastiveHead):
    def __init__(self, source: BNContrastiveHead) -> None:
        torch.nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Patched forward - replaces einsum with equivalent ops for QAIRT compatibility.

        Parameters
        ----------
        x
            Image features (b, c, h, w).
        w
            Text features (b, k, c).

        Returns
        -------
        torch.Tensor
            Similarity scores (b, k, h, w).

        Notes
        -----
        Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L811-#L825
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        # Begin Qualcomm modification
        # Replaced einsum operator
        # x: (b, c, h, w) -> (b, h, w, c)
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
        # w: (b, k, c) -> (b, c, k)
        w = w.permute(0, 2, 1)  # (b, c, k)

        # NOTE: torch.matmul was intentionally avoided here.
        # Using torch.matmul (or the equivalent via permute) on batched 4D x 3D tensors
        # triggers q::Batch_MatMul_Bias_w_scale.B in QAIRT, which is not registered
        # and causes a graph preparation failure in w8a16 quantized inference:
        #   tcm_migration.cc:2034::ERROR: no properties registered for q::Batch_MatMul_Bias_w_scale.B
        #   graph_prepare.cc:198::ERROR: could not create op: q::Batch_MatMul_Bias_w_scale.B
        # The unsqueeze + element-wise multiply + sum pattern below avoids this op
        # and produces equivalent output while remaining compatible with QAIRT w8a16.
        # matmul: (b, h, w, c) x (b, c, k) -> (b, h, w, k)
        x = (x.unsqueeze(-1) * w.unsqueeze(1).unsqueeze(1)).sum(dim=-2)
        # Restore to (b, k, h, w)
        x = x.permute(0, 3, 1, 2)  # (b, k, h, w)
        # End Qualcomm modification

        return x * self.logit_scale.exp() + self.bias

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import types
from typing import Any

import torch
import torch.nn.functional as F
from funasr.models.transformer.utils.repeat import MultiSequential


def layer_drop_forward(self: MultiSequential, *args: Any) -> Any:
    """
    Serialization-safe replacement for MultiSequential.forward.

    Upstream draws the layer-drop probabilities with `torch.empty(n).uniform_()`.
    The in-place `uniform_` survives torch.export as an `aten.uniform_` node whose
    `from`/`to` kwargs are dropped on save, so torch.export.load fails with
    KeyError: 'from'. `torch.rand` produces the same U[0, 1) draw out of place.
    """
    _probs = torch.rand(len(self))
    for idx, m in enumerate(self):
        if not self.training or (_probs[idx] >= self.layer_drop_rate):
            args = m(*args)
    return args


def patch_layer_drop(model: torch.nn.Module) -> None:
    """Swap every MultiSequential in `model` onto the forward above."""
    for module in model.modules():
        if isinstance(module, MultiSequential):
            module.forward = types.MethodType(layer_drop_forward, module)


class _ConformerEncoderWithCTC(torch.nn.Module):
    """
    Thin wrapper around the FunASR Conformer encoder + CTC head.

    Accepts mel-LFR features (already preprocessed by WavFrontend) and
    returns CTC log-probabilities.  This is the part that gets exported
    to ONNX/QNN.

    feats_len is derived internally from feats.shape[1] so the exported
    model has a single tensor input.  make_pad_mask's .tolist() call gets
    traced with concrete value T=num_frames and baked as a constant, which
    is correct for fixed-window inference.
    """

    def __init__(self, encoder: Any, ctc_lo: torch.nn.Linear) -> None:
        super().__init__()
        self.encoder = encoder
        self.ctc_lo = ctc_lo

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feats
            Shape (1, T, INPUT_SIZE=560) — mel-LFR features (float32).

        Returns
        -------
        log_probs: torch.Tensor
            Shape (1, T, vocab_size=4200) — CTC log-probabilities.
        """
        feats_len = torch.tensor(
            [feats.shape[1]], dtype=torch.int64, device=feats.device
        )
        enc_out, _, _ = self.encoder(feats, feats_len)
        ctc_logits = self.ctc_lo(enc_out)
        return F.log_softmax(ctc_logits, dim=-1)

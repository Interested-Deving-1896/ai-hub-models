# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Monkey-patches applied to the openai/CLIP library at import time.
# Import this module before calling clip.load().

from __future__ import annotations

import torch
from clip.model import CLIP

# Use a finite fill value instead of -inf to avoid NaN in softmax on some
# Qualcomm runtimes where -inf * 0 = NaN rather than 0.
_MASK_FILL = -1e4


def _patched_build_attention_mask(self: CLIP) -> torch.Tensor:
    """
    Build a causal attention mask for the text transformer.

    Replaces the original CLIP implementation which fills masked positions with
    ``-inf``. On some Qualcomm runtimes ``-inf * 0`` evaluates to ``NaN``
    instead of ``0`` during softmax, corrupting attention outputs. Using a
    finite fill value (``_MASK_FILL = -1e4``) avoids this while still
    effectively zeroing out future-token attention weights after softmax.

    Parameters
    ----------
    self
        The CLIP model instance (bound ``self`` supplied by the monkey-patch).

    Returns
    -------
    torch.Tensor
        Upper-triangular causal mask of shape
        ``(context_length, context_length)`` with ``_MASK_FILL`` above the
        diagonal and ``0`` on and below it.
    """
    mask = torch.empty(self.context_length, self.context_length)
    mask.fill_(_MASK_FILL)
    mask.triu_(1)
    return mask

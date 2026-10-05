# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math

import torch

# <blank> is always index 0 in FunASR models.
CTC_BLANK_ID = 0


def ctc_bpe_decode(
    log_probs: torch.Tensor,
    token_list: list[str],
    blank_id: int = CTC_BLANK_ID,
) -> str:
    """
    CTC greedy decode + BPE detokenization for FunASR models.

    Collapses blanks/repeats then joins BPE pieces: `@@` suffix = continuation
    (no space), no suffix = word boundary (trailing space).

    Parameters
    ----------
    log_probs
        Shape (T, vocab_size) — log-probabilities for a single sequence.
        Pass a pre-sliced tensor (e.g. ``log_probs[:valid_len]``) to bound
        decoding to valid frames only.
    token_list
        Vocabulary list mapping token IDs to string pieces.
    blank_id
        Token ID for the CTC blank symbol. Defaults to 0.

    Returns
    -------
    str
        Decoded and detokenized text with BPE pieces joined and leading/
        trailing whitespace stripped.
    """
    ids = torch.argmax(log_probs, dim=-1).tolist()
    _special = {"<blank>", "<s>", "</s>", "<unk>", "<space>"}
    collapsed: list[str] = []
    prev = -1
    for t in ids:
        if t not in (blank_id, prev) and t < len(token_list):
            tok = token_list[t]
            if tok not in _special:
                collapsed.append(tok)
        prev = t
    parts = [tok[:-2] if tok.endswith("@@") else tok + " " for tok in collapsed]
    return "".join(parts).strip()


def audio_len_to_valid_frames(real_len: int, lfr_n: int) -> int:
    """
    Frame count for real_len samples at 16kHz with 25ms/10ms windows and LFR n=6.

    Parameters
    ----------
    real_len
        Number of audio samples at 16kHz.
    lfr_n
        LFR frame-stacking factor.

    Returns
    -------
    int
        Number of valid LFR frames.
    """
    num_fbank = max(real_len - 400, 0) // 160 + 1
    return math.ceil(num_fbank / lfr_n)

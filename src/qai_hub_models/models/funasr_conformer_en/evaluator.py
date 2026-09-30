# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.funasr_conformer_en.utils import CTC_BLANK_ID, ctc_bpe_decode
from qai_hub_models.models.templates.conformer.evaluator import (
    ConformerCTCWERBase,
)


class FunASRConformerEvaluator(ConformerCTCWERBase):
    """WER evaluator for FunASR Conformer-EN using BPE CTC decode."""

    def __init__(self, token_list: list[str]) -> None:
        self.token_list = token_list
        self._blank_id = CTC_BLANK_ID
        super().__init__()

    def decode_predictions(
        self,
        logits: torch.Tensor,
        target: tuple[torch.Tensor, torch.Tensor],
    ) -> list[str]:
        _, valid_frames_batch = target
        predictions = []
        for i in range(logits.shape[0]):
            valid_len = int(valid_frames_batch[i].item())
            predictions.append(
                ctc_bpe_decode(logits[i, :valid_len], self.token_list, self._blank_id)
            )
        return predictions

    def decode_references(self, target: tuple[torch.Tensor, torch.Tensor]) -> list[str]:
        gt_text_batch, _ = target
        return ["".join(chr(int(c)) for c in t if int(c) != 0) for t in gt_text_batch]

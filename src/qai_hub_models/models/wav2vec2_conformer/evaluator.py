# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import Wav2Vec2Processor

from qai_hub_models.models.templates.conformer.evaluator import (
    ConformerCTCWERBase,
)


class Wav2Vec2ConformerEvaluator(ConformerCTCWERBase):
    """WER evaluator using the HF processor's batch_decode."""

    def __init__(self, processor: Wav2Vec2Processor) -> None:
        self.processor = processor
        super().__init__()

    def decode_predictions(self, logits: torch.Tensor, target: object) -> list[str]:
        del target
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)

    def decode_references(self, target: torch.Tensor) -> list[str]:
        return ["".join(chr(int(c)) for c in row if int(c) != 0) for row in target]

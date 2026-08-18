# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import jiwer
import torch

from qai_hub_models.models.funasr_conformer_en.utils import CTC_BLANK_ID, ctc_bpe_decode
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    WORD_ERROR_RATE,
    MetricMetadata,
)


class FunASRConformerEvaluator(BaseEvaluator):
    """
    WER evaluator for FunASR Conformer-EN using BPE CTC decode.

    Uses the conformer's own CTC greedy decode + BPE detokenization rather than
    the WavLM-specific processor.batch_decode() in LibriSpeechEvaluator.
    """

    def __init__(self, token_list: list[str]) -> None:
        self.token_list = token_list
        self._blank_id = CTC_BLANK_ID
        self.reset()

    def add_batch(
        self,
        output: torch.Tensor,
        target: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        logits = output if isinstance(output, torch.Tensor) else output[0]
        gt_text_batch, valid_frames_batch = target
        for i in range(logits.shape[0]):
            valid_len = int(valid_frames_batch[i].item())
            decoded = ctc_bpe_decode(
                logits[i, :valid_len], self.token_list, self._blank_id
            )
            self.predictions.append(decoded)
        clean_targets = [
            "".join(chr(int(c)) for c in t if int(c) != 0) for t in gt_text_batch
        ]
        self.references.extend(clean_targets)

    def reset(self) -> None:
        self.predictions: list[str] = []
        self.references: list[str] = []

    def get_accuracy_score(self) -> float:
        return jiwer.wer(self.references, self.predictions) * 100

    def formatted_accuracy(self) -> str:
        return f"Word Error Rate: {self.get_accuracy_score():.3f}"

    def get_metric_metadata(self) -> MetricMetadata:
        return WORD_ERROR_RATE

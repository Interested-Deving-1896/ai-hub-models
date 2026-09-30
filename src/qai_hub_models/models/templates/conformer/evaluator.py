# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jiwer
import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import WORD_ERROR_RATE, MetricMetadata


class ConformerCTCWERBase(BaseEvaluator, ABC):
    """Shared WER evaluator for conformer models with CTC outputs."""

    def __init__(self) -> None:
        self.reset()

    def add_batch(
        self,
        output: torch.Tensor | tuple[torch.Tensor, ...],
        target: Any,
    ) -> None:
        """Decode one model-output batch and add it to the WER state.

        Parameters
        ----------
        output
            CTC logits, either as a tensor or as the first tensor in a tuple.
        target
            Ground-truth data in the format required by the concrete evaluator.
        """
        logits = output if isinstance(output, torch.Tensor) else output[0]
        self.predictions.extend(self.decode_predictions(logits, target))
        self.references.extend(self.decode_references(target))

    @abstractmethod
    def decode_predictions(self, logits: torch.Tensor, target: Any) -> list[str]:
        """Decode a batch of CTC logits into transcribed strings."""

    @abstractmethod
    def decode_references(self, target: Any) -> list[str]:
        """Decode the model-specific ground-truth representation."""

    def reset(self) -> None:
        """Reset accumulated predictions and references."""
        self.predictions: list[str] = []
        self.references: list[str] = []

    def get_accuracy_score(self) -> float:
        """Return word error rate as a percentage."""
        return jiwer.wer(self.references, self.predictions) * 100

    def formatted_accuracy(self) -> str:
        """Return formatted word error rate."""
        return f"Word Error Rate: {self.get_accuracy_score():.3f}"

    def get_metric_metadata(self) -> MetricMetadata:
        """Return metadata for the word error rate metric."""
        return WORD_ERROR_RATE

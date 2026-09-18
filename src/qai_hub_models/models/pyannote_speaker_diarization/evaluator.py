# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from qai_hub_models.models.pyannote_speaker_diarization.dataset import decode_path
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import DIARIZATION_ERROR_RATE, MetricMetadata


class PyannoteDERMetricEvaluator(BaseEvaluator):
    """Corpus-level Diarization Error Rate using pyannote.metrics."""

    def __init__(self) -> None:
        self.metric = DiarizationErrorRate()

    def add_batch(self, output: Annotation, gt: tuple[torch.Tensor, ...]) -> None:
        rttm_tensor = gt[0] if isinstance(gt, tuple) else gt
        rttm_path = decode_path(rttm_tensor.squeeze(0))
        annotations = load_rttm(rttm_path)
        uri = next(iter(annotations.keys()))
        reference: Annotation = annotations[uri]
        self.metric(reference, output)

    def reset(self) -> None:
        self.metric = DiarizationErrorRate()

    def get_accuracy_score(self) -> float:
        return 100.0 * abs(self.metric)

    def formatted_accuracy(self) -> str:
        return f"DER: {100.0 * abs(self.metric):.2f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return DIARIZATION_ERROR_RATE

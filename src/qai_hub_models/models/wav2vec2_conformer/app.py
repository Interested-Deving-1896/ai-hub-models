# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor

from qai_hub_models.models.templates.conformer.utils import chunk_audio
from qai_hub_models.models.wav2vec2_conformer.model import (
    DEFAULT_AUDIO_LENGTH,
    HF_MODEL_ID,
    SAMPLE_RATE,
)
from qai_hub_models.models.wav2vec2_conformer.utils import normalize_audio
from qai_hub_models.utils.inference import OnDeviceModel


class Wav2Vec2ConformerApp:
    """
    End-to-end Wav2Vec2-Conformer application.

    Pipeline:
    1. Resample audio to 16kHz if needed
    2. Processor: normalize raw audio → input_values
    3. Model: input_values → CTC logits
    4. CTC greedy decode → text

    The processor's feature extraction is kept out of the exported model
    because it uses non-differentiable normalization ops.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.model = model
        self.processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_ID)

    def predict(self, *args: Any, **kwargs: Any) -> str:
        return self.transcribe(*args, **kwargs)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """
        Transcribe raw audio to text, chunking long audio into 10-second windows.

        Parameters
        ----------
        audio
            1-D float32 numpy array of raw audio samples.
        sample_rate
            Sample rate of the input audio. Resampled to 16kHz if needed.

        Returns
        -------
        text: str
            Transcribed text.
        """
        if audio.ndim == 2:
            audio = audio.mean(-1)
        audio = audio.astype(np.float32)

        if sample_rate != SAMPLE_RATE:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sample_rate, SAMPLE_RATE)
            audio = audio_t.squeeze(0).numpy()

        chunks = chunk_audio(DEFAULT_AUDIO_LENGTH, audio)
        chunk_tensors = [
            normalize_audio(self.processor, c, SAMPLE_RATE).input_values
            for c, _ in chunks
        ]

        if isinstance(self.model, OnDeviceModel):
            # Fan-out: submit all inference jobs without blocking.
            async_results = [self.model.async_model(t) for t in chunk_tensors]
            # Fan-in: wait for all jobs and collect per-chunk argmax ids.
            all_ids = []
            for r in async_results:
                result = r.wait()
                logits = result[0] if isinstance(result, tuple) else result
                all_ids.append(torch.argmax(logits[0], dim=-1))
        else:
            # Local: single batched forward pass.
            batch = torch.cat(chunk_tensors, dim=0)
            with torch.no_grad():
                output = self.model(batch)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            all_ids = [torch.argmax(logits[i], dim=-1) for i in range(logits.shape[0])]

        decoded = self.processor.batch_decode(torch.stack(all_ids))
        parts = [t.lower().strip() for t in decoded if t.strip()]
        return " ".join(parts)

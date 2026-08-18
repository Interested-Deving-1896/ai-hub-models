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

from qai_hub_models.models.funasr_conformer_en.model import (
    DEFAULT_AUDIO_LENGTH,
    SAMPLE_RATE,
    audio_len_to_valid_frames,
)
from qai_hub_models.models.funasr_conformer_en.utils import CTC_BLANK_ID, ctc_bpe_decode


class FunASRConformerEnApp:
    """
    End-to-end FunASR conformer-en application.

    Pipeline:
    1. Resample + pad/truncate audio to fixed length
    2. WavFrontend: raw audio → mel-LFR features (560-dim)
    3. Model: features → CTC log-probabilities (4200-dim)
    4. CTC greedy decode → text

    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        frontend: Any,
        token_list: list[str],
    ) -> None:
        self.model = model
        self.frontend = frontend
        self.token_list = token_list
        self._blank_id = CTC_BLANK_ID

    def predict(self, *args: Any, **kwargs: Any) -> str:
        return self.transcribe(*args, **kwargs)

    def _preprocess_audio(
        self,
        audio: torch.Tensor,
        audio_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply WavFrontend to convert raw audio to mel-LFR features.

        This is NOT part of the exported model (frontend uses non-traceable DSP).
        Call this separately before calling forward() for app usage.
        """
        with torch.no_grad():
            return self.frontend(audio, audio_len)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """
        Transcribe raw audio to text, chunking long audio into 10-second windows.

        The exported model has a fixed 10-second (160,000 sample) input window.
        Audio longer than this is split into consecutive 10-second chunks; each
        chunk is transcribed independently and the results are concatenated.

        Parameters
        ----------
        audio
            1-D float32 numpy array of raw audio samples.
        sample_rate
            Sample rate of the input audio. Will be resampled to 16kHz if needed.

        Returns
        -------
        text: str
            transcribed text
        """
        if audio.ndim == 2:
            audio = audio.mean(-1)

        audio = audio.astype(np.float32)

        if sample_rate != SAMPLE_RATE:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sample_rate, SAMPLE_RATE)
            audio = audio_t.squeeze(0).numpy()

        chunks = self._chunk_audio(audio)
        parts: list[str] = []
        for chunk, real_len in chunks:
            result = self._transcribe_chunk(chunk, real_len)
            if result:
                parts.append(result)

        return " ".join(parts)

    def _chunk_audio(self, audio: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """
        Split 16kHz audio into (chunk, real_len) pairs of DEFAULT_AUDIO_LENGTH samples.
        The last chunk is zero-padded if shorter; real_len tracks actual samples.
        """
        if len(audio) == 0:
            return [(np.zeros(DEFAULT_AUDIO_LENGTH, dtype=np.float32), 0)]

        chunks = []
        for start in range(0, len(audio), DEFAULT_AUDIO_LENGTH):
            chunk = audio[start : start + DEFAULT_AUDIO_LENGTH]
            real_len = len(chunk)
            if real_len < DEFAULT_AUDIO_LENGTH:
                chunk = np.pad(chunk, (0, DEFAULT_AUDIO_LENGTH - real_len))
            chunks.append((chunk, real_len))
        return chunks

    def _transcribe_chunk(self, chunk: np.ndarray, real_len: int) -> str:
        """Run model inference on a single fixed-size audio chunk.

        chunk is always DEFAULT_AUDIO_LENGTH samples (zero-padded if needed).
        real_len is used to derive the valid frame count analytically, bounding
        CTC decoding without running a second WavFrontend pass.
        """
        audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
        # Always pass DEFAULT_AUDIO_LENGTH so the frontend produces exactly
        # DEFAULT_NUM_FRAMES (167) frames — matching the compiled model shape.
        full_lengths = torch.tensor([DEFAULT_AUDIO_LENGTH], dtype=torch.int64)

        feats, _ = self._preprocess_audio(audio_tensor, full_lengths)

        # Derive valid frame count analytically (25ms window / 10ms shift at 16kHz,
        # LFR n=6) — avoids a second full WavFrontend pass just to get feats_len.
        valid_frames = audio_len_to_valid_frames(real_len)
        output = self.model(feats)

        # OnDeviceModel may return a tuple; unwrap to the log_probs tensor.
        log_probs = output[0] if isinstance(output, (tuple, list)) else output

        if self.token_list is None:
            return ""
        return ctc_bpe_decode(
            log_probs[0][:valid_frames], self.token_list, self._blank_id
        )

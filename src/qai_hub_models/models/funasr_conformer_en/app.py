# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torchaudio

from qai_hub_models.models.funasr_conformer_en.model import (
    DEFAULT_AUDIO_LENGTH,
    LFR_N,
    SAMPLE_RATE,
)
from qai_hub_models.models.funasr_conformer_en.utils import (
    CTC_BLANK_ID,
    audio_len_to_valid_frames,
    ctc_bpe_decode,
)
from qai_hub_models.models.templates.conformer.utils import chunk_audio
from qai_hub_models.utils.inference import OnDeviceModel


class FunASRConformerEnApp:
    """
    End-to-end FunASR conformer-en application.

    Pipeline:
    1. Resample + pad/truncate audio to fixed length
    2. WavFrontend: raw audio -> mel-LFR features (560-dim)
    3. Model: features -> CTC log-probabilities (4200-dim)
    4. CTC greedy decode -> text

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

        chunks = chunk_audio(DEFAULT_AUDIO_LENGTH, audio)

        # Preprocess: run WavFrontend on every chunk before any inference.
        feats_list: list[torch.Tensor] = []
        valid_frames_list = []
        for chunk, real_len in chunks:
            audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
            # Always pass DEFAULT_AUDIO_LENGTH so the frontend produces exactly
            # DEFAULT_NUM_FRAMES (167) frames, matching the compiled model shape.
            full_lengths = torch.tensor([DEFAULT_AUDIO_LENGTH], dtype=torch.int64)
            feats, _ = self._preprocess_audio(audio_tensor, full_lengths)
            feats_list.append(feats)
            # Derive valid frame count analytically (25ms window / 10ms shift at
            # 16kHz, LFR n=6), avoiding a second WavFrontend pass for feats_len.
            valid_frames_list.append(audio_len_to_valid_frames(real_len, LFR_N))

        if isinstance(self.model, OnDeviceModel):
            # One job over all chunks (each a batch-1 entry), like evaluate/helpers.py.
            output = self.model.async_model(
                cast(list[torch.Tensor | np.ndarray], feats_list)
            ).wait()
        else:
            output = self.model(torch.cat(feats_list, dim=0))

        log_probs = output[0] if isinstance(output, (tuple, list)) else output
        all_log_probs = [log_probs[i] for i in range(log_probs.shape[0])]

        if self.token_list is None:
            return ""

        parts = [
            ctc_bpe_decode(lp[:vf], self.token_list, self._blank_id)
            for lp, vf in zip(all_log_probs, valid_frames_list, strict=True)
        ]
        return " ".join(p for p in parts if p)

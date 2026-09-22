# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch
import torchaudio

if TYPE_CHECKING:
    from qai_hub_models.models.silero_vad.model import SileroVAD

from qai_hub_models.models.silero_vad.model import (
    CHUNK_SIZE,
    CONTEXT_SIZE,
    SAMPLE_RATE,
    STATE_SHAPE,
)

# Default VAD hysteresis parameters (match Silero-VAD defaults)
DEFAULT_THRESHOLD = 0.5
DEFAULT_NEG_THRESHOLD = 0.35  # threshold - 0.15
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_SILENCE_MS = 100
DEFAULT_SPEECH_PAD_MS = 30


def _load_audio_raw(audio_path: str) -> tuple[torch.Tensor, int]:
    """Load audio file and return (waveform [C, N], sample_rate).

    Tries ``torchaudio.load`` first; if that fails (e.g. torchcodec/FFmpeg
    shared-library mismatch in the current environment), falls back to
    decoding via the ``ffmpeg`` command-line binary and ``scipy.io.wavfile``.
    """
    try:
        return torchaudio.load(audio_path)
    except Exception:
        pass

    # Fallback: decode to a temporary WAV via the ffmpeg binary.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-f",
                "wav",
                tmp_path,
            ],
            check=True,
            capture_output=True,
        )
        sr, data = scipy.io.wavfile.read(tmp_path)
    finally:
        os.unlink(tmp_path)

    # scipy returns int16/int32/float32 depending on WAV encoding
    if data.dtype != np.float32:
        orig_dtype = data.dtype
        data = data.astype(np.float32)
        if np.issubdtype(orig_dtype, np.integer):
            data /= float(np.iinfo(orig_dtype).max + 1)

    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)  # [1, N]
    else:
        wav = torch.from_numpy(data.T).float()  # [C, N]
    return wav, sr


def load_audio(audio_path: str) -> torch.Tensor:
    """Load an audio file and return a mono 16 kHz float32 1-D tensor.

    Converts multi-channel audio to mono by averaging channels, then
    resamples to ``SAMPLE_RATE`` (16 kHz) using
    ``scipy.signal.resample_poly`` to avoid a floating-point exception in
    torchaudio's sinc resampler on certain sample-rate ratios (e.g.
    24000 -> 16000).

    Parameters
    ----------
    audio_path:
        Path to an audio file (any format supported by torchaudio, with
        ffmpeg as a fallback).

    Returns
    -------
    torch.Tensor
        1-D float32 tensor of shape ``[num_samples]``, normalised to
        ``[-1, 1]``.
    """
    wav, sr = _load_audio_raw(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav_np = wav.squeeze(0).numpy()
        gcd = math.gcd(sr, SAMPLE_RATE)
        up = SAMPLE_RATE // gcd
        down = sr // gcd
        wav_np = scipy.signal.resample_poly(wav_np, up, down).astype(np.float32)
        wav = torch.from_numpy(wav_np).unsqueeze(0)
    return wav.squeeze(0)  # [num_samples]


class SileroVADApp:
    """End-to-end Voice Activity Detection application.

    Wraps a SileroVAD callable (PyTorch model or on-device inference) and
    provides a high-level ``predict`` method that accepts a raw audio file
    and returns a list of detected speech segments.

    The callable must match the stateless SileroVAD interface::

        speech_prob, state_out, context_out = callable(audio_chunk, state, context)

    Parameters
    ----------
    model:
        SileroVAD model instance, or ``None`` when only segment-detection
        helpers (``_probs_to_segments``) are needed (e.g. QDC path).
    threshold:
        Speech probability threshold.  Probabilities above this value are
        classified as speech.
    neg_threshold:
        Exit threshold.  While in speech mode, probabilities below this
        value trigger a potential speech-end.
    min_speech_ms:
        Minimum speech segment duration in milliseconds.  Shorter segments
        are discarded.
    min_silence_ms:
        Minimum silence duration in milliseconds before ending a speech
        segment.
    speech_pad_ms:
        Padding added to both sides of each detected speech segment.
    """

    def __init__(
        self,
        model: SileroVAD | Callable | None,
        threshold: float = DEFAULT_THRESHOLD,
        neg_threshold: float = DEFAULT_NEG_THRESHOLD,
        min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
        min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
        speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_speech_samples = int(SAMPLE_RATE * min_speech_ms / 1000)
        self.min_silence_samples = int(SAMPLE_RATE * min_silence_ms / 1000)
        self.speech_pad_samples = int(SAMPLE_RATE * speech_pad_ms / 1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, audio_path: str) -> list[dict[str, float]]:
        """Detect speech segments in an audio file.

        Parameters
        ----------
        audio_path:
            Path to an audio file (any format supported by torchaudio).

        Returns
        -------
        list[dict[str, float]]
            List of ``{"start": <seconds>, "end": <seconds>}`` dicts, one
            per detected speech segment, sorted by start time.
        """
        audio = load_audio(audio_path)
        probs = self._run_chunks(audio)
        return self._probs_to_segments(probs, len(audio))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_chunks(self, audio: torch.Tensor) -> list[float]:
        """Slice audio into CHUNK_SIZE windows and run the model on each.

        Returns a list of speech probabilities, one per chunk.
        """
        if self.model is None:
            raise RuntimeError(
                "_run_chunks requires a model; pass model=None only when "
                "using _probs_to_segments directly (e.g. QDC path)."
            )
        state = torch.zeros(1, *STATE_SHAPE)
        context = torch.zeros(1, CONTEXT_SIZE)
        probs: list[float] = []

        num_samples = audio.shape[0]
        with torch.no_grad():
            # TODO(depends on PR #4050): consolidate this state-threading loop with the
            # equivalent Linux and Android device-script implementations.
            for start in range(0, num_samples, CHUNK_SIZE):
                chunk = audio[start : start + CHUNK_SIZE]
                if chunk.shape[0] < CHUNK_SIZE:
                    chunk = torch.nn.functional.pad(
                        chunk, (0, CHUNK_SIZE - chunk.shape[0])
                    )
                chunk = chunk.unsqueeze(0)  # [1, 512]
                speech_prob, state, context = self.model(chunk, state, context)
                probs.append(float(speech_prob.squeeze()))

        return probs

    def _probs_to_segments(
        self, probs: list[float], num_samples: int
    ) -> list[dict[str, float]]:
        """Convert per-chunk probabilities to speech segment timestamps.

        Implements the same hysteresis logic as ``silero_vad.get_speech_timestamps``
        but operates on the pre-computed probability list.
        """
        triggered = False
        speeches: list[dict[str, int]] = []
        current_speech: dict[str, int] = {}
        temp_end = 0

        for i, prob in enumerate(probs):
            cur_sample = i * CHUNK_SIZE

            if prob >= self.threshold and temp_end:
                temp_end = 0

            if prob >= self.threshold and not triggered:
                triggered = True
                current_speech["start"] = cur_sample
                continue

            if prob < self.neg_threshold and triggered:
                if not temp_end:
                    temp_end = cur_sample
                if cur_sample - temp_end < self.min_silence_samples:
                    continue
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                    > self.min_speech_samples
                ):
                    speeches.append(current_speech)
                current_speech = {}
                temp_end = 0
                triggered = False
                continue

        # Handle speech that reaches end of audio
        if (
            current_speech
            and (num_samples - current_speech["start"]) > self.min_speech_samples
        ):
            current_speech["end"] = num_samples
            speeches.append(current_speech)

        # Apply padding and merge overlapping segments
        for i, seg in enumerate(speeches):
            if i == 0:
                seg["start"] = max(0, seg["start"] - self.speech_pad_samples)
            if i < len(speeches) - 1:
                silence = speeches[i + 1]["start"] - seg["end"]
                if silence < 2 * self.speech_pad_samples:
                    seg["end"] += silence // 2
                    speeches[i + 1]["start"] = max(
                        0, speeches[i + 1]["start"] - silence // 2
                    )
                else:
                    seg["end"] = min(num_samples, seg["end"] + self.speech_pad_samples)
                    speeches[i + 1]["start"] = max(
                        0, speeches[i + 1]["start"] - self.speech_pad_samples
                    )
            else:
                seg["end"] = min(num_samples, seg["end"] + self.speech_pad_samples)

        # Convert sample indices to seconds
        return [
            {
                "start": round(seg["start"] / SAMPLE_RATE, 3),
                "end": round(seg["end"] / SAMPLE_RATE, 3),
            }
            for seg in speeches
        ]

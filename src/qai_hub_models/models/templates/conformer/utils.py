# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np


def chunk_audio(audio_length: int, audio: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Split audio into fixed-length chunks.

    The last chunk is zero-padded if shorter than ``audio_length``. The
    corresponding ``real_len`` value tracks the number of actual, unpadded
    samples in each chunk.

    Parameters
    ----------
    audio_length
        Number of samples in each output chunk.
    audio
        One-dimensional audio samples to split.

    Returns
    -------
    audio_chunks: list[tuple[np.ndarray, int]]
        A list of ``(chunk, real_len)`` pairs, where each ``chunk`` is a
        NumPy array of ``audio_length`` samples and ``real_len`` is the
        number of unpadded samples. Empty input returns one zero-padded chunk
        with ``real_len`` equal to zero.
    """
    if len(audio) == 0:
        return [(np.zeros(audio_length, dtype=np.float32), 0)]

    chunks = []
    for start in range(0, len(audio), audio_length):
        chunk = audio[start : start + audio_length]
        real_len = len(chunk)
        if real_len < audio_length:
            chunk = np.pad(chunk, (0, audio_length - real_len))
        chunks.append((chunk, real_len))
    return chunks

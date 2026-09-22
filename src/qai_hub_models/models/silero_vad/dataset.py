# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.earnings21 import (
    EVAL10_AUDIO_ASSETS,
    Earnings21Dataset,
)

# Cached calibration tensors stored alongside the audio files. The version
# suffix is bumped whenever the cache schema or contents change so that stale
# caches are ignored and rebuilt rather than loaded.  When bumping this suffix,
# also increment EARNINGS21_VERSION so the asset-loader cache is invalidated.
_CALIB_CACHE_FILE = "calib_samples_v3.pt"


class SileroVADCalibrationDataset(Earnings21Dataset):
    """Earnings-21 eval-10 dataset for Silero-VAD quantization calibration.

    Yields fixed-size 512-sample audio chunks (32 ms at 16 kHz) drawn from
    the 11 eval-10 earnings-call recordings.  Used exclusively for
    quantization calibration — the activation distribution of each chunk is
    what matters, not sequential ordering or ground-truth labels.

    Each sample includes the real accumulated LSTM state and look-back
    context from the preceding chunks in the same audio file.  This ensures
    the quantizer observes the true dynamic range of the LSTM cell state
    ``c`` (which can reach ~60 in streaming inference) rather than the
    near-zero values seen when state is always reset to zeros.

    The accumulated states are computed once and cached to disk as
    ``calib_samples_v3.pt`` alongside the audio files.  Subsequent
    instantiations load from cache without re-running the model.

    For end-to-end streaming accuracy evaluation, use ``app.py`` directly
    with the full audio files.  Accurate VAD metrics require stateful
    inference (LSTM state threaded across all chunks of each file), which
    the standard evaluate harness does not support.

    Parameters
    ----------
    split:
        Dataset split.  The split value is accepted but not used — all
        audio chunks are loaded regardless of split.
    file_ids:
        Subset of eval-10 file IDs to include.  Defaults to all 11 files.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        file_ids: list[str] | None = None,
    ) -> None:
        self.chunks: list[torch.Tensor] = []
        self.states: list[torch.Tensor] = []
        self.contexts: list[torch.Tensor] = []
        super().__init__(split=split, file_ids=file_ids)

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    @property
    def _cache_path(self) -> Path:
        return self.dataset_path / _CALIB_CACHE_FILE

    def _validate_data(self) -> bool:
        """Return True only when both audio files and the calibration cache exist.

        Kept as a pure existence check so ``BaseDataset.download_data`` can
        decide whether to call ``_download_data``.  Cache loading happens in
        ``_download_data`` (after the build) and here (on a warm-cache hit).
        """
        if not super()._validate_data() or not self._cache_path.exists():
            return False
        data = torch.load(self._cache_path, weights_only=True)
        self.chunks = data["chunks"]
        self.states = data["states"]
        self.contexts = data["contexts"]
        return len(self.chunks) > 0

    def _download_data(self) -> None:
        """Download audio files then build the calibration cache.

        Building the cache here (rather than in ``_validate_data``) ensures
        ``BaseDataset.download_data`` calls it at most once — the base class
        calls ``_validate_data`` twice (before and after ``_download_data``),
        so any expensive work in ``_validate_data`` would run twice on a cold
        start.
        """
        super()._download_data()
        self._build_calib_cache()
        data = torch.load(self._cache_path, weights_only=True)
        self.chunks = data["chunks"]
        self.states = data["states"]
        self.contexts = data["contexts"]

    def _build_calib_cache(self) -> None:
        """Run the model over each audio file and cache a calibration subset.

        The model is streamed through the full audio so the accumulated LSTM
        state ``c`` reaches the true dynamic range seen at inference time
        (~0 when reset each chunk, up to ~60 over a file). Only
        ``default_num_calibration_samples()`` chunks are ever used, so a
        uniform random subset of that size is kept via reservoir sampling
        (seed 42, matching ``sample_dataset``) instead of storing all ~1.2M.
        """
        from qai_hub_models.models.silero_vad.app import load_audio
        from qai_hub_models.models.silero_vad.model import (
            CHUNK_SIZE,
            CONTEXT_SIZE,
            STATE_SHAPE,
            SileroVAD,
        )

        capacity = self.default_num_calibration_samples()
        print(
            f"Building stateful calibration cache (runs model once, "
            f"keeping {capacity} sampled chunks)..."
        )
        model = SileroVAD.from_pretrained()
        model.eval()

        generator = torch.Generator().manual_seed(42)
        chunks: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        contexts: list[torch.Tensor] = []
        seen = 0

        for fid in self.file_ids:
            wav = load_audio(str(EVAL10_AUDIO_ASSETS[fid].path))

            state = torch.zeros(*STATE_SHAPE)  # [2, 128]
            context = torch.zeros(CONTEXT_SIZE)  # [64]

            with torch.no_grad():
                for start in range(0, wav.shape[0], CHUNK_SIZE):
                    chunk = wav[start : start + CHUNK_SIZE]
                    if chunk.shape[0] < CHUNK_SIZE:
                        chunk = F.pad(chunk, (0, CHUNK_SIZE - chunk.shape[0]))

                    # Record inputs *before* the forward pass so the
                    # calibrator sees the full range of states encountered
                    # at inference time (including accumulated cell state).
                    if len(chunks) < capacity:
                        chunks.append(chunk.clone())
                        states.append(state.clone())
                        contexts.append(context.clone())
                    else:
                        j = int(
                            torch.randint(0, seen + 1, (1,), generator=generator).item()
                        )
                        if j < capacity:
                            chunks[j] = chunk.clone()
                            states[j] = state.clone()
                            contexts[j] = context.clone()
                    seen += 1

                    _, state_out, context_out = model(
                        chunk.unsqueeze(0),
                        state.unsqueeze(0),
                        context.unsqueeze(0),
                    )
                    state = state_out.squeeze(0)
                    context = context_out.squeeze(0)

        torch.save(
            {"chunks": chunks, "states": states, "contexts": contexts},
            self._cache_path,
        )
        print(
            f"Saved {len(chunks)} calibration samples (from {seen} total chunks) "
            f"to {self._cache_path}"
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], list]:
        """Return model inputs for one 512-sample audio chunk.

        Parameters
        ----------
        index
            Index of the chunk to retrieve.

        Returns
        -------
        inputs : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            audio_chunk : shape [512], float32 — the DataLoader adds the batch dim.
            state       : shape [2, 128], float32 — real accumulated LSTM state.
            context     : shape [64], float32 — real look-back context.
        ground_truth : list
            Empty list — no ground truth for calibration.
        """
        return (self.chunks[index], self.states[index], self.contexts[index]), []

    @staticmethod
    def default_samples_per_job() -> int:
        """Number of chunks used for a single calibration job."""
        return 500

    @staticmethod
    def default_num_calibration_samples() -> int:
        """Number of chunks to use for quantization calibration."""
        return 500

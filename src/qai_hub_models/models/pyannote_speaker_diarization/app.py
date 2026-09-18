# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pyannote.audio import Audio, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.pyannote_speaker_diarization.dataset import (
    _apply_speaker_masks,
    decode_path,
)
from qai_hub_models.models.pyannote_speaker_diarization.model import (
    load_pyannote_pipeline,
)
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.inference import AsyncOnDeviceModel


class PyannoteSpeakerDiarizationApp(CollectionAppEvaluateProtocol):
    """End-to-end speaker diarization: segmentation + embedding callables → annotation."""

    def __init__(
        self,
        segmentation: Callable,
        embedding: Callable,
        pipeline: Pipeline | None = None,
    ) -> None:
        self.segmentation = segmentation
        self.embedding = embedding
        self.pipeline = pipeline

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> PyannoteSpeakerDiarizationApp:
        """Create app from [segmentation, embedding] component list."""
        return cls(
            segmentation=models[0],
            embedding=models[1],
            pipeline=load_pyannote_pipeline(),
        )

    def run_model_for_eval(
        self,
        model_input: Generator | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        if isinstance(model_input, tuple) and model_input[0].dtype == torch.uint8:
            # Eval path: input is an encoded audio file path from AMIDiarizationEvalDataset.
            assert model_input[0].shape[0] == 1, "DER eval requires batch_size=1"
            audio_path = decode_path(model_input[0][0])
            output = self.diarize(audio_path)
        elif isinstance(model_input, tuple):
            # Local torch path: dispatch by component index (segmentation=0, embedding=1).
            x = model_input[0]
            with torch.inference_mode():
                output = (
                    self.embedding(x)
                    if x.ndim == 3 and x.shape[-1] == 80
                    else self.segmentation(x)
                )
        else:
            # On-device path: re-assemble split batches for compiled fixed-batch models.
            chunks = next(model_input)
            batch = (
                torch.cat(list(chunks), dim=0) if isinstance(chunks, tuple) else chunks
            )
            with torch.inference_mode():
                output = (
                    self.embedding(batch)
                    if batch.ndim == 3 and batch.shape[-1] == 80
                    else self.segmentation(batch)
                )
        yield output
        return output

    def predict(self, audio_file: AudioFile, *args: Any, **kwargs: Any) -> Annotation:
        return self.diarize(audio_file, *args, **kwargs)

    def diarize(
        self,
        audio_file: AudioFile,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Annotation:
        """Run full speaker diarization on an audio file.

        Parameters
        ----------
        audio_file
            Path or dict accepted by pyannote Audio.validate_file.
        num_speakers
            Exact number of speakers (optional).
        min_speakers
            Minimum number of speakers (optional).
        max_speakers
            Maximum number of speakers (optional).

        Returns
        -------
        Annotation
            Pyannote Annotation with speaker labels and time segments.
        """
        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("pipeline must be set to call diarize()")
        file = Audio.validate_file(audio_file)

        num_speakers, min_speakers, max_speakers = pipeline.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        _segmentation = pipeline._segmentation
        waveform, sample_rate = _segmentation.model.audio(file)

        window_size = round(_segmentation.duration * sample_rate)
        step_size = round(_segmentation.step * sample_rate)
        _, num_samples = waveform.shape

        if num_samples > window_size:
            chunks = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks = chunks.shape[0]
        else:
            num_chunks = 0
            chunks = torch.empty(
                (0, waveform.shape[0], window_size),
                device=waveform.device,
                dtype=waveform.dtype,
            )

        has_last_chunk = (num_chunks == 0) or (
            (num_samples - window_size) % step_size > 0
        )
        if has_last_chunk:
            last_chunk = waveform[:, num_chunks * step_size :]
            last_pad = window_size - last_chunk.shape[1]
            last_chunk = F.pad(last_chunk, (0, last_pad))
        else:
            last_chunk = None

        if has_last_chunk:
            all_chunks = torch.cat([chunks, last_chunk.unsqueeze(0)], dim=0)
        else:
            all_chunks = chunks

        original_num_chunks = all_chunks.shape[0]
        batch_sz = _segmentation.batch_size
        remainder = original_num_chunks % batch_sz
        pad_needed = (batch_sz - remainder) % batch_sz
        if pad_needed > 0:
            pad = torch.zeros(
                (pad_needed, *all_chunks.shape[1:]),
                device=all_chunks.device,
                dtype=all_chunks.dtype,
            )
            all_chunks = torch.cat([all_chunks, pad], dim=0)

        seg_outputs: list[Any] = []
        for c in range(0, all_chunks.shape[0], batch_sz):
            batch = all_chunks[c : c + batch_sz]
            with torch.inference_mode():
                batch_out = self.segmentation(batch)
            batch_out = _segmentation.conversion(batch_out).cpu().numpy()
            seg_outputs.append(batch_out)

        outputs = np.vstack(seg_outputs)
        if pad_needed > 0:
            outputs = outputs[:original_num_chunks]

        frames = SlidingWindow(
            start=0.0,
            duration=_segmentation.duration,
            step=_segmentation.step,
        )
        segmentations = SlidingWindowFeature(outputs, frames)
        num_chunks_, num_frames_, _ = segmentations.data.shape

        count = pipeline.speaker_count(
            segmentations,
            _segmentation.model.receptive_field,
            warm_up=(0.0, 0.0),
        )
        if np.nanmax(count.data) == 0.0:
            return Annotation(uri=file["uri"])

        duration_ = segmentations.sliding_window.duration
        min_num_samples_ = pipeline._embedding.min_num_samples
        num_samples_ = duration_ * pipeline._embedding.sample_rate
        min_num_frames_ = math.ceil(num_frames_ * min_num_samples_ / num_samples_)

        clean_frames_ = 1.0 * (np.sum(segmentations.data, axis=2, keepdims=True) < 2)
        clean_segmentations = SlidingWindowFeature(
            segmentations.data * clean_frames_,
            segmentations.sliding_window,
        )

        def _iter_fbank() -> Generator[torch.Tensor, None, None]:
            for (chunk, masks), (_, clean_masks) in zip(
                segmentations, clean_segmentations, strict=False
            ):
                wv, _ = pipeline._audio.crop(
                    file, chunk, duration=duration_, mode="pad"
                )
                if wv.shape[1] > window_size:
                    wv = wv[:, :window_size]
                elif wv.shape[1] < window_size:
                    wv = F.pad(wv, (0, window_size - wv.shape[1]))

                fbank_chunk = pipeline._embedding.model_.compute_fbank(
                    wv[None]
                ).squeeze(0)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for fbank in _apply_speaker_masks(
                    fbank_chunk, masks, clean_masks, min_num_frames_
                ):
                    yield fbank.cpu()

        def _batchify(
            iterable: Iterable[torch.Tensor], batch_size: int
        ) -> Iterator[tuple[torch.Tensor | None, ...]]:
            args = [iter(iterable)] * batch_size
            return itertools.zip_longest(*args, fillvalue=None)

        fixed_batch_size = pipeline.embedding_batch_size
        emb_outputs: list[Any] = []
        for batch_raw in _batchify(_iter_fbank(), fixed_batch_size):
            valid_items: list[torch.Tensor] = [b for b in batch_raw if b is not None]
            actual = len(valid_items)
            fbank = torch.stack(valid_items)

            with torch.inference_mode(), warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if actual < fixed_batch_size:
                    pad_count = fixed_batch_size - actual
                    fbank_in = torch.cat(
                        [
                            fbank,
                            torch.zeros(pad_count, *fbank.shape[1:], dtype=fbank.dtype),
                        ],
                        dim=0,
                    )
                else:
                    fbank_in = fbank
                emb = self.embedding(fbank_in)
            emb_outputs.append(emb[:actual].detach().cpu().numpy())

        embedding_batches = np.vstack(emb_outputs)
        embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks_)

        hard_clusters, _, _centroids = pipeline.clustering(
            embeddings=embeddings,
            segmentations=segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            file=file,
            frames=_segmentation.model.receptive_field,
        )

        num_different_speakers = np.max(hard_clusters) + 1
        if (
            num_different_speakers < min_speakers
            or num_different_speakers > max_speakers
        ):
            warnings.warn(
                f"Detected {num_different_speakers} speakers, outside [{min_speakers}, {max_speakers}].",
                stacklevel=2,
            )

        count.data = np.minimum(count.data, max_speakers).astype(np.int8)
        inactive_speakers = np.sum(segmentations.data, axis=1) == 0
        hard_clusters[inactive_speakers] = -2

        discrete_diarization = pipeline.reconstruct(segmentations, hard_clusters, count)
        diarization = pipeline.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=pipeline.segmentation.min_duration_off,
        )
        diarization.uri = file["uri"]

        mapping = dict(zip(diarization.labels(), pipeline.classes(), strict=False))
        return diarization.rename_labels(mapping=mapping)

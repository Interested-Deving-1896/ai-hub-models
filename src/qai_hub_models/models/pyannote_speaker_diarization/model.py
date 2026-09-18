# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB
from einops import rearrange
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from torch import nn

from qai_hub_models import Precision
from qai_hub_models.models.pyannote_speaker_diarization.dataset import (
    AMIDiarizationEvalDataset,
    AMIEmbeddingDataset,
)
from qai_hub_models.models.pyannote_speaker_diarization.evaluator import (
    PyannoteDERMetricEvaluator,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel, SerializationSettings
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec
from qai_hub_models.utils.model_adapters import (
    Conv2dFromConv1d,
    InstanceNorm2dFromInstanceNorm1d,
)

MODEL_ID = __name__.split(".")[-2]

SAMPLE_RATE = 16000
SEG_DURATION = 5.0
SEG_STEP_RATIO = 0.2
SEG_DURATION_FRAMES = round(SEG_DURATION * SAMPLE_RATE)
BATCH_SIZE = 32


class _QCSincNet(nn.Module):
    """SincNet reimplemented with Conv2d because QNN/HTP doesn't support native 1D conv ops."""

    def __init__(self, sincnet_instance: nn.Module) -> None:
        super().__init__()
        s: Any = sincnet_instance  # nn.Module attrs are typed as Tensor|Module; Any lets us access dynamic attrs

        self.wav_norm2d = InstanceNorm2dFromInstanceNorm1d(s.wav_norm1d)

        self.conv2d = nn.ModuleList()
        self.pool2d = nn.ModuleList()
        self.norm2d = nn.ModuleList()

        num_layer = len(s.conv1d)

        for idx in range(num_layer):
            conv: nn.Module
            if idx == 0:
                conv = self._paramsincfb_to_conv2d(s.conv1d[idx])
            else:
                conv = Conv2dFromConv1d(s.conv1d[idx])
            self.conv2d.append(conv)
            self.pool2d.append(
                nn.MaxPool2d(
                    kernel_size=(1, s.pool1d[idx].kernel_size),
                    stride=(1, s.pool1d[idx].stride),
                    padding=(0, s.pool1d[idx].padding),
                )
            )

            self.norm2d.append(InstanceNorm2dFromInstanceNorm1d(s.norm1d[idx]))

    def _paramsincfb_to_conv2d(self, enc: Encoder) -> nn.Conv2d:
        fb: ParamSincFB = enc.filterbank
        filters_attr = fb.filters
        filters = filters_attr() if callable(filters_attr) else filters_attr

        ks = fb.kernel_size
        n_out = fb.n_feats_out
        stride = fb.stride or ks // 2
        in_channels = 1

        conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_out,
            kernel_size=(1, ks),
            stride=(1, stride),
            padding=(0, 0),
            bias=False,
        )
        with torch.no_grad():
            if filters.ndim == 2:
                filters = filters.view(n_out, in_channels, 1, ks)
            elif filters.ndim == 3:
                filters = filters.unsqueeze(-2)
            conv2d.weight.copy_(filters.to(dtype=conv2d.weight.dtype))
        return conv2d

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = waveforms.unsqueeze(-2)
        outputs = self.wav_norm2d(outputs)

        for idx, (conv2d, pool2d, norm2d) in enumerate(
            zip(self.conv2d, self.pool2d, self.norm2d, strict=False)
        ):
            outputs = conv2d(outputs)
            if idx == 0:
                outputs = torch.abs(outputs)
            outputs = pool2d(outputs)
            outputs = norm2d(outputs)
            outputs = F.leaky_relu(outputs)

        return outputs.squeeze(-2)


class PyannoteSegmentation(BaseModel):
    """Pyannote segmentation model: waveforms (B,1,80000) → powerset output (B,293,7)."""

    def __init__(self, orig_model: nn.Module) -> None:
        super().__init__(serialization_settings=SerializationSettings(use_pt2=False))
        orig: Any = orig_model
        self.sincnet = _QCSincNet(orig.sincnet)
        self.lstm: nn.LSTM = orig.lstm
        self.linear: nn.ModuleList = orig.linear
        self.classifier: nn.Linear = orig.classifier
        self.activation: nn.Module = orig.activation

    @classmethod
    def from_pretrained(cls) -> PyannoteSegmentation:
        pipeline = load_pyannote_pipeline()
        orig = pipeline._segmentation.model
        orig.cpu().eval()
        return cls(orig)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.sincnet(waveforms)
        outputs, _ = self.lstm(
            rearrange(outputs, "batch feature frame -> batch frame feature")
        )
        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))
        return self.activation(self.classifier(outputs))

    def get_input_spec(
        self,
        batch_size: int = BATCH_SIZE,
        duration_frames: int = SEG_DURATION_FRAMES,
    ) -> InputSpec:
        return {
            "waveforms": TensorSpec(
                shape=(batch_size, 1, duration_frames), dtype="float32"
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {"segments": TensorSpec()}

    def component_precision(self) -> Precision:
        return Precision.float


@lru_cache(maxsize=1)
def _cached_embedding_dataset() -> AMIEmbeddingDataset:
    """Building AMIEmbeddingDataset runs the full segmentation pipeline; cache it."""
    return AMIEmbeddingDataset()


class PyannoteEmbedding(BaseModel):
    """Pyannote speaker embedding model: fbank (B,T,80) → embedding (B,256)."""

    def __init__(self, orig_model: nn.Module) -> None:
        super().__init__(serialization_settings=SerializationSettings(use_pt2=False))
        resnet: Any = orig_model.resnet
        self.conv1: nn.Conv2d = resnet.conv1
        self.bn1: nn.BatchNorm2d = resnet.bn1
        self.layer1: nn.Module = resnet.layer1
        self.layer2: nn.Module = resnet.layer2
        self.layer3: nn.Module = resnet.layer3
        self.layer4: nn.Module = resnet.layer4
        self.seg_1: nn.Linear = resnet.seg_1

    @classmethod
    def from_pretrained(cls, orig_model: nn.Module | None = None) -> PyannoteEmbedding:
        if orig_model is None:
            pipeline = load_pyannote_pipeline()
            orig_model = pipeline._embedding.model_
            orig_model.cpu().eval()
        return cls(orig_model)

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        x = fbank.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = rearrange(
            out, "batch dimension channel frames -> batch (dimension channel) frames"
        )
        num_frames = features.shape[-1]
        mean = features.mean(dim=-1)
        var_biased = (features - mean.unsqueeze(-1)).square().mean(dim=-1)
        var_unbiased = var_biased * (num_frames / (num_frames - 1))
        std = torch.sqrt(var_unbiased)
        stats = torch.cat([mean, std], dim=-1)
        return self.seg_1(stats)

    def get_input_spec(
        self,
        batch_size: int = BATCH_SIZE,
        num_frames: int = 498,
        num_mels: int = 80,
    ) -> InputSpec:
        return {
            "fbank": TensorSpec(
                shape=(batch_size, num_frames, num_mels), dtype="float32"
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {"embedding": TensorSpec()}

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return AMIEmbeddingDataset

    def component_precision(self) -> Precision:
        return Precision.w8a16

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> dict[str, list[np.ndarray]]:
        ds = _cached_embedding_dataset()
        if len(ds) > 0:
            fbank, _ = ds[0]
            return {"fbank": [fbank.numpy()]}
        spec = input_spec or self.get_input_spec()
        return {"fbank": [torch.randn(spec["fbank"].shape).numpy()]}


class PyannoteSpeakerDiarization(WorkbenchModelCollection):
    """Pyannote Speaker Diarization 3.1 — CollectionModel with segmentation and embedding."""

    def __init__(
        self,
        segmentation: PyannoteSegmentation,
        embedding: PyannoteEmbedding,
    ) -> None:
        super().__init__({"segmentation": segmentation, "embedding": embedding})
        self.segmentation = segmentation
        self.embedding = embedding

    @classmethod
    def from_pretrained(cls) -> PyannoteSpeakerDiarization:
        pipeline = load_pyannote_pipeline()

        orig_seg = pipeline._segmentation.model
        orig_seg.cpu().eval()

        orig_emb = pipeline._embedding.model_
        orig_emb.cpu().eval()

        seg_model = PyannoteSegmentation(orig_seg)
        emb_model = PyannoteEmbedding(orig_emb)
        return cls(seg_model, emb_model)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [AMIDiarizationEvalDataset]

    def get_evaluator(self) -> PyannoteDERMetricEvaluator:
        return PyannoteDERMetricEvaluator()


# diarize() (app.py) and AMIEmbeddingDataset (dataset.py) reimplement Pipeline.apply()
# via private attributes (_segmentation, _embedding, .model_, .conversion). Tied to pyannote.audio==3.3.2.
@lru_cache(maxsize=1)
def load_pyannote_pipeline() -> Pipeline:
    # Allowlist the classes present in pyannote checkpoints so torch.load
    # can deserialize them under weights_only=True.
    torch.serialization.add_safe_globals(
        [
            torch.torch_version.TorchVersion,
            Specifications,
            Problem,
            Resolution,
        ]
    )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    )
    if pipeline is None:
        raise RuntimeError(
            "Failed to load pyannote/speaker-diarization-3.1. "
            "Run `huggingface-cli login` and accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 "
            "and https://huggingface.co/pyannote/segmentation-3.0"
        )
    pipeline._segmentation.duration = SEG_DURATION
    pipeline._segmentation.step = SEG_DURATION * SEG_STEP_RATIO
    return pipeline

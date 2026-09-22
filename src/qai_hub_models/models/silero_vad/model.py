# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from silero_vad import load_silero_vad
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel, SerializationSettings
from qai_hub_models.utils.input_spec import InputSpec, IoType, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Silero-VAD operates on 512-sample chunks at 16 kHz (32 ms per chunk).
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # samples per inference step at 16 kHz
CONTEXT_SIZE = 64  # samples of look-back context prepended to each chunk
STATE_SHAPE = (
    2,
    128,
)  # LSTM state: (h_and_c, hidden_dim) — batch dim is dim 0 in get_input_spec


class _SileroInnerNative(nn.Module):
    """Native PyTorch reimplementation of Silero-VAD's inner 16 kHz model.

    The Silero-VAD package ships a TorchScript (``torch.jit.script``) model.
    When a scripted submodule is inlined into a traced parent, its graph is
    copied verbatim — so the ``aten::lstm_cell`` op survives tracing and the
    TorchScript→ONNX converter emits an ONNX ``LSTM`` op.  That op carries an
    unused ``Y`` output slot that gets assigned the empty string ``''``,
    causing duplicate tensor names in the QDQ ONNX compiler.

    The fix is to leave TorchScript entirely: any eager rewrite of the LSTM
    cell (whether ``nn.LSTMCell`` or plain ``torch.lstm_cell``) causes tracing
    to run the dispatcher and capture the gate-op decomposition
    (``aten::linear`` x2 → ``add_`` → ``unsafe_chunk`` → ``sigmoid_`` x3 /
    ``tanh_`` → ``mul``/``add``) instead of the monolithic ``aten::lstm_cell``.
    ``nn.LSTMCell`` is used here for clarity, but the key axis is
    scripted vs eager, not module vs functional form.

    Weights are copied from the pretrained JIT model; numerical output is
    bit-exact.
    """

    def __init__(self, jit_inner: torch.jit.RecursiveScriptModule) -> None:
        super().__init__()
        # STFT and encoder are conv-based — no LSTM op issue, keep as JIT.
        self.stft = jit_inner.stft
        self.encoder = jit_inner.encoder

        # Replace torch.lstm_cell with nn.LSTMCell (module form).
        # nn.LSTMCell traces to raw gate ops; torch.lstm_cell traces to
        # aten::lstm_cell which the ONNX converter maps to the LSTM op.
        self.lstm_cell = nn.LSTMCell(input_size=128, hidden_size=128)
        self.lstm_cell.weight_ih = nn.Parameter(
            jit_inner.decoder.rnn.weight_ih.detach().clone()
        )
        self.lstm_cell.weight_hh = nn.Parameter(
            jit_inner.decoder.rnn.weight_hh.detach().clone()
        )
        self.lstm_cell.bias_ih = nn.Parameter(
            jit_inner.decoder.rnn.bias_ih.detach().clone()
        )
        self.lstm_cell.bias_hh = nn.Parameter(
            jit_inner.decoder.rnn.bias_hh.detach().clone()
        )

        # Downstream MLP: relu -> Linear(128,1) -> sigmoid.
        # The original JIT decoder uses Conv1d(128,1,1), which the QAIRT converter
        # maps to a Conv2d with weight layout [out=1, kH=1, kW=128, in=1] on HTP.
        # HTP interprets this as a 1x128 spatial conv with 1 input channel instead
        # of a 1x1 conv with 128 input channels, producing wrong speech_prob values
        # (PSNR ~11 dB). Replacing with nn.Linear traces to FullyConnected in the
        # DLC, which has no layout ambiguity and gives correct results on HTP.
        # Extract Conv1d(128,1,1) weights via named_parameters (JIT modules
        # do not support __getitem__ indexing into Sequential children).
        _conv_w, _conv_b = None, None
        for _name, _p in jit_inner.decoder.decoder.named_parameters():
            if _name == "2.weight":
                _conv_w = _p.detach().clone()  # [1, 128, 1]
            elif _name == "2.bias":
                _conv_b = _p.detach().clone()  # [1]
        assert _conv_w is not None and _conv_b is not None, (
            "Could not find Conv1d weights in decoder.decoder"
        )
        linear = nn.Linear(128, 1)
        linear.weight = nn.Parameter(_conv_w.squeeze(-1))  # [1,128,1] -> [1,128]
        linear.bias = nn.Parameter(_conv_b)
        self.decoder_mlp = nn.Sequential(
            nn.ReLU(),
            linear,
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,  # [1, 576] context + chunk
        h: torch.Tensor,  # [1, 128] LSTM hidden state
        c: torch.Tensor,  # [1, 128] LSTM cell state
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.stft(x)  # [1, 129, 4]
        x1 = self.encoder(x0)  # [1, 128, 1]
        x2 = x1.squeeze(-1)  # [1, 128]
        h_new, c_new = self.lstm_cell(x2, (h, c))  # both [1, 128]
        x3 = h_new.float()  # [1, 128]
        x4 = self.decoder_mlp(x3)  # [1, 1]
        prob = x4.view(1, 1)  # [1, 1]
        return prob, h_new, c_new


class SileroVAD(BaseModel):
    """Silero Voice Activity Detection model.

    Processes one 32 ms audio chunk per inference call and returns a speech
    probability.  LSTM state (``h``, ``c``) and look-back context are explicit
    inputs and outputs so the caller can thread them across chunks for
    streaming inference — matching the official Silero-VAD ONNX interface.

    Interface (get_input_spec / compiled model)
    -------------------------------------------
    Inputs:
        audio_chunk : float32 [1, 512]    — 32 ms of audio at 16 kHz
        state       : float32 [1, 2, 128] — LSTM state (zeros on first call)
        context     : float32 [1, 64]     — look-back context (zeros on first call)

    Outputs:
        speech_prob : float32 [1, 1]      — speech probability in [0, 1]
        state_out   : float32 [1, 2, 128] — updated LSTM state
        context_out : float32 [1, 64]     — updated look-back context

    Evaluation note
    ---------------
    Accurate VAD evaluation requires stateful streaming inference — the LSTM
    state must be threaded across all chunks of each audio file.  The standard
    evaluate harness does not support cross-call state threading, so this model
    does not register an eval dataset.  Use ``app.py`` with the full Earnings-21
    eval-10 audio files for end-to-end streaming accuracy measurement.
    Calibration uses ``SileroVADCalibrationDataset`` for quantization (state is not
    required for calibration — only the activation distribution matters).
    """

    def __init__(self, inner: _SileroInnerNative) -> None:
        super().__init__(serialization_settings=SerializationSettings(use_pt2=False))
        self.inner = inner

    def forward(
        self,
        audio_chunk: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one 32 ms inference step.

        Parameters
        ----------
        audio_chunk:
            Raw audio samples, shape [1, 512], float32, range [-1, 1].
        state:
            LSTM state from the previous step, shape [1, 2, 128].
            Pass ``torch.zeros(1, 2, 128)`` for the first chunk.
        context:
            Look-back context from the previous step, shape [1, 64].
            Pass ``torch.zeros(1, 64)`` for the first chunk.

        Returns
        -------
        speech_prob : torch.Tensor
            Speech probability for this chunk, shape [1, 1], range [0, 1].
        state_out : torch.Tensor
            Updated LSTM state, shape [1, 2, 128].
        context_out : torch.Tensor
            Updated look-back context, shape [1, 64].
        """
        h = state[:, 0]  # [batch, 128]
        c = state[:, 1]  # [batch, 128]
        x = torch.cat([context, audio_chunk], dim=-1)  # [1, 576]
        prob, h_new, c_new = self.inner(x, h, c)
        state_out = torch.stack([h_new, c_new], dim=1)  # [batch, 2, 128]
        context_out = x[:, -CONTEXT_SIZE:]  # [1, 64]
        return prob, state_out, context_out

    @classmethod
    def from_pretrained(cls) -> Self:
        """Load Silero VAD with bundled pretrained weights."""
        jit_model = load_silero_vad(onnx=False)
        inner = _SileroInnerNative(jit_model._model)
        return cls(inner)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
    ) -> InputSpec:
        return {
            "audio_chunk": TensorSpec(
                shape=(batch_size, CHUNK_SIZE), dtype="float32", io_type=IoType.TENSOR
            ),
            "state": TensorSpec(
                shape=(batch_size, *STATE_SHAPE), dtype="float32", io_type=IoType.TENSOR
            ),
            "context": TensorSpec(
                shape=(batch_size, CONTEXT_SIZE), dtype="float32", io_type=IoType.TENSOR
            ),
        }

    @staticmethod
    def get_output_spec() -> OutputSpec:
        return {
            "speech_prob": TensorSpec(io_type=IoType.TENSOR),
            "state_out": TensorSpec(io_type=IoType.TENSOR),
            "context_out": TensorSpec(io_type=IoType.TENSOR),
        }

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        from qai_hub_models.models.silero_vad.dataset import SileroVADCalibrationDataset

        return SileroVADCalibrationDataset

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 25.0

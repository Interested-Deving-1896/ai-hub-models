# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Gives pyannote.audio==3.3.2 the torchaudio<2.9 APIs it uses, via soundfile.

Wrap ``pyannote.audio`` imports in ``torchaudio_compat()``: ``torchaudio`` is patched
only inside the block (pyannote reads ``torchaudio.AudioMetaData`` at import time).
On exit the global module is restored and pyannote's own ``torchaudio`` references
point at a proxy, so other models in the same process are unaffected.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import soundfile as sf
import torch
import torchaudio

PYANNOTE_PACKAGE = "pyannote.audio"
_MISSING = object()


@dataclass
class _AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 0
    encoding: str = "UNKNOWN"


def _info(uri: Any, **_: Any) -> _AudioMetaData:
    meta = sf.info(uri)
    return _AudioMetaData(meta.samplerate, meta.frames, meta.channels)


def _load(
    uri: Any,
    frame_offset: int = 0,
    num_frames: int = -1,
    channels_first: bool = True,
    **_: Any,
) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(
        uri,
        start=frame_offset,
        frames=num_frames,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(data)
    return (waveform.T.contiguous() if channels_first else waveform), sample_rate


def _compat_attrs() -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if not hasattr(torchaudio, "AudioMetaData"):
        attrs["AudioMetaData"] = _AudioMetaData
    if not hasattr(torchaudio, "info"):
        attrs["info"] = _info
    if not hasattr(torchaudio, "list_audio_backends"):
        attrs["list_audio_backends"] = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        attrs["set_audio_backend"] = lambda *_, **__: None
    if torch.__version__ >= "2.9":
        attrs["load"] = _load
    return attrs


class _TorchaudioProxy:
    """Forwards to the real ``torchaudio``, except for the compat overrides."""

    def __init__(self, overrides: dict[str, Any]) -> None:
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(torchaudio, name)


@contextmanager
def torchaudio_compat() -> Generator[None, None, None]:
    """Temporarily patch ``torchaudio``; rebind pyannote's references on exit."""
    attrs = _compat_attrs()
    originals = {name: getattr(torchaudio, name, _MISSING) for name in attrs}
    for name, value in attrs.items():
        setattr(torchaudio, name, value)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is _MISSING:
                delattr(torchaudio, name)
            else:
                setattr(torchaudio, name, original)
        proxy = _TorchaudioProxy(attrs)
        for module_name, module in list(sys.modules.items()):
            if (
                module_name.startswith(PYANNOTE_PACKAGE)
                and getattr(module, "torchaudio", None) is torchaudio
            ):
                setattr(module, "torchaudio", proxy)  # noqa: B010

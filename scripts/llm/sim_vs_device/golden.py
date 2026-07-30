# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Capture golden per-Part tensors from the AIMET QuantSim split model.

We drive the *quantized* split model (each Part backed by its own QuantSim ONNX
session) through the shared ``HubCompatibleGenerator`` on real prefill windows.
Because the generator's ``prepare_inputs`` emits the feed already in on-device
layout (``position_ids_cos`` / ``position_ids_sin`` and Hub-transposed KV), the
tensors we record at each Part boundary are drop-in inputs for an on-device
inference job -- this is the "inject golden inputs at every boundary" step.

For each Part we record, per generator call (one per prefill slice per window):
  * inputs:  {onnx_input_name -> np.ndarray}   (the golden injected feed)
  * outputs: {onnx_output_name -> np.ndarray}  (the golden reference)

Part N's captured *inputs* already contain the golden hidden-state produced by
Part N-1 (the generator threads it through), so injected-only comparison needs
nothing beyond these records -- no separate Part 1 job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from qai_hub_models.models._shared.llm.generator_factory import make_generator


@dataclass
class PartCapture:
    """Golden inputs/outputs for one Part across all captured calls."""

    part_index: int  # 0-based index into the parts list
    part_name: str
    onnx_input_names: list[str]
    onnx_output_names: list[str]
    num_layers: int
    # One dict per (window, slice) call; each maps name -> np.ndarray.
    inputs: list[dict[str, np.ndarray]] = field(default_factory=list)
    outputs: list[dict[str, np.ndarray]] = field(default_factory=list)

    @property
    def hidden_output_name(self) -> str:
        """The residual-stream / logits output (always output[0])."""
        return self.onnx_output_names[0]

    @property
    def kv_output_names(self) -> list[str]:
        return [
            n
            for n in self.onnx_output_names
            if n.startswith(("past_key", "past_value"))
        ]


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").float().numpy()


def capture_golden(
    split_model: object,
    parts: list,
    windows: list[tuple[torch.Tensor, torch.Tensor]],
    sequence_length: int,
    context_length: int,
    layers_per_part: dict[int, int],
    max_samples_per_part: int | None = None,
    last_slice_only: bool = True,
) -> list[PartCapture]:
    """Run the QuantSim split model over ``windows`` and record Part I/O.

    ``split_model`` is a ``SplitForwardMixin`` instance (e.g.
    ``QuantizedSplitModelWrapper``) whose ``forward`` chains ``parts``.
    ``parts`` is the already-instantiated Part list (sharing one presplit).

    ``last_slice_only`` (default True): keep only the final prefill slice of
    each window. When context_length > sequence_length a window is sliced into
    multiple forwards; only the last reads a fully-populated (real) KV cache,
    which is both the KV-read path we probe and half/less the output bytes
    (earlier slices carry a zero-filled cache). Set False to keep every slice.
    """
    # Force the parts to exist / be the exact objects we instrument.
    split_model._ensure_parts()
    assert split_model._parts is parts or split_model._parts is not None
    parts = split_model._parts  # authoritative list the wrapper will call

    captures: list[PartCapture] = []
    originals: list = []
    for idx, part in enumerate(parts):
        in_names = part._get_onnx_input_names()
        out_names = [
            n.replace("/", "_").replace(".", "_") for n in part._get_onnx_output_names()
        ]
        captures.append(
            PartCapture(
                part_index=idx,
                part_name=f"part{idx + 1}_of_{len(parts)}",
                onnx_input_names=in_names,
                onnx_output_names=out_names,
                num_layers=layers_per_part.get(idx, 0),
            )
        )

    def make_wrapped(idx: int, part: object, orig_forward: object) -> Callable:
        cap = captures[idx]

        def wrapped(*args: object, **kwargs: object) -> object:
            out = orig_forward(*args, **kwargs)
            out_list = [out] if isinstance(out, torch.Tensor) else list(out)
            # Positional args align to ONNX input names (mock_torch_onnx_inference
            # consumes them positionally in graph-input order).
            in_rec = {
                name: _to_np(a)
                for name, a in zip(cap.onnx_input_names, args, strict=False)
                if isinstance(a, torch.Tensor)
            }
            out_rec = {
                name: _to_np(o)
                for name, o in zip(cap.onnx_output_names, out_list, strict=False)
                if isinstance(o, torch.Tensor)
            }
            if in_rec and out_rec:
                cap.inputs.append(in_rec)
                cap.outputs.append(out_rec)
            return out

        return wrapped

    for idx, part in enumerate(parts):
        orig = part.forward
        originals.append(orig)
        part.forward = make_wrapped(idx, part, orig)  # type: ignore[method-assign]

    try:
        generator = make_generator(
            split_model,
            sequence_length=sequence_length,
            context_length=context_length,
        )
        with torch.no_grad():
            for input_ids, attention_mask in windows:
                # A ctx_len window is sliced into ctx/seq forwards; each fires
                # every Part.forward and is captured. KV accumulates across
                # slices, so only the LAST slice sees a full (real) KV cache --
                # earlier slices read a zero-filled cache, which both wastes
                # bytes and doesn't exercise the KV-read path we're probing.
                pre = [len(c.inputs) for c in captures]
                generator(input_ids=input_ids, attention_mask=attention_mask)
                if last_slice_only:
                    for c, n0 in zip(captures, pre, strict=False):
                        if len(c.inputs) > n0 + 1:
                            c.inputs[n0:] = c.inputs[-1:]
                            c.outputs[n0:] = c.outputs[-1:]
    finally:
        for part, orig in zip(parts, originals, strict=False):
            part.forward = orig  # type: ignore[method-assign]

    if max_samples_per_part is not None:
        for cap in captures:
            cap.inputs = cap.inputs[:max_samples_per_part]
            cap.outputs = cap.outputs[:max_samples_per_part]

    return captures

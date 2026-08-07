# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared assertion helpers for the Gemma4 KV-cache tensor layout.

Gemma4 attention is exported as single-head-attention (SHA), with the KV cache
exposed as graph I/O tensors that Genie's ring-buffer cache manager reads/writes.
Each KV head is emitted as a SEPARATE graph tensor named
``{pfx}_key_{layer}_h{head}_in`` / ``_out``, shaped ``(1, 1, head_dim, kv_len)``,
so every KV tensor has a leading batch dim of 1 regardless of num_kv_heads (a
single stacked ``(num_kv_heads, 1, ...)`` tensor breaks Genie's ring buffer for
GQA models).

These helpers assert that contract on the pure static spec builders. Each model's
``test.py`` supplies its own geometry via ``KVLayoutConfig`` and calls them, so
both the GQA path (E4B) and the MQA path (E2B) are covered.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from qai_hub_models.models._shared.gemma4 import model
from qai_hub_models.models._shared.gemma4.model import (
    _LUT_QUANT_ROWS,
    Gemma4Base,
    _global_uint16_encoding,
    _quantize_uint16,
    _write_uint16_lut,
    get_non_shared_layer_indices,
)
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

_KV_RE = re.compile(r"^(?:past|swa)_(key|value)_(\d+)_h(\d+)_(in|out)$")
# The old (buggy) single-tensor-per-layer naming — must NOT appear anymore.
_OLD_KV_RE = re.compile(r"^(?:past|swa)_(?:key|value)_\d+_(in|out)$")


@dataclass(frozen=True)
class KVLayoutConfig:
    """The geometry the KV-layout specs are built from.

    Mirrors the per-model ``model.py`` constants of the same names, so a model
    passes its own values rather than duplicating a literal here.
    """

    num_hidden_layers: int
    hidden_size: int
    num_key_value_heads: int
    num_attention_heads: int
    head_dim: int
    global_head_dim: int
    num_kv_shared_layers: int
    sliding_window_pattern: int


def input_spec(cfg: KVLayoutConfig) -> InputSpec:
    return Gemma4Base._get_input_spec(
        num_hidden_layers=cfg.num_hidden_layers,
        sequence_length=128,
        context_length=4096,
        hidden_size=cfg.hidden_size,
        num_key_value_heads=cfg.num_key_value_heads,
        num_attention_heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim,
        global_head_dim=cfg.global_head_dim,
        num_kv_shared_layers=cfg.num_kv_shared_layers,
        sliding_window_pattern=cfg.sliding_window_pattern,
    )


def output_spec(cfg: KVLayoutConfig) -> dict[str, TensorSpec]:
    return Gemma4Base._get_output_spec(
        cfg.num_hidden_layers,
        cfg.num_kv_shared_layers,
        cfg.sliding_window_pattern,
        cfg.num_key_value_heads,
    )


def kv_items(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if "_key_" in k or "_value_" in k}


def assert_kv_input_tensors_have_batch_dim_one(cfg: KVLayoutConfig) -> None:
    """Every KV cache input tensor must have a leading batch dim of 1.

    Per-head separate tensors keep both the batch and per-head dims at 1;
    a leading dim > 1 (e.g. num_kv_heads=2 for GQA) breaks Genie's ring buffer.

    Parameters
    ----------
    cfg
        The model's KV-cache geometry.
    """
    kv = kv_items(input_spec(cfg))
    assert kv, "expected KV input tensors in the spec"
    for name, (shape, _dtype) in kv.items():
        assert shape[0] == 1, (
            f"{name} has leading (batch) dim {shape[0]} != 1 -> Genie ring-buffer "
            f"cache cannot execute this (HTP Error 6004). shape={shape}"
        )
        # dim 1 is the per-head axis and must also be 1 (one head per tensor).
        assert shape[1] == 1, (
            f"{name} has per-head dim {shape[1]} != 1; heads must be separate "
            f"tensors, not stacked. shape={shape}"
        )


def assert_kv_output_tensors_use_per_head_naming(cfg: KVLayoutConfig) -> None:
    """KV outputs must use per-head naming and never the old stacked naming.

    Parameters
    ----------
    cfg
        The model's KV-cache geometry.
    """
    kv = kv_items(output_spec(cfg))
    assert kv, "expected KV output tensors in the spec"
    for name in kv:
        assert _KV_RE.match(name), (
            f"{name} does not match per-head naming {{pfx}}_key_{{layer}}_h{{head}}_out"
        )
        assert not _OLD_KV_RE.match(name), (
            f"{name} uses the old stacked single-tensor-per-layer naming "
            f"(regressed the GQA fix)"
        )


def assert_kv_tensor_count_scales_with_num_kv_heads(cfg: KVLayoutConfig) -> None:
    """There must be exactly num_kv_heads tensors per (layer, key/value).

    This is what makes GQA (kv=2) emit twice as many KV tensors as MQA (kv=1),
    and is the structural counterpart of the per-head split.

    Parameters
    ----------
    cfg
        The model's KV-cache geometry.
    """
    num_non_shared = len(
        get_non_shared_layer_indices(cfg.num_hidden_layers, cfg.num_kv_shared_layers)
    )

    for spec_fn in (input_spec, output_spec):
        kv = kv_items(spec_fn(cfg))
        # non-shared layers * num_kv_heads * {key, value}
        expected = num_non_shared * cfg.num_key_value_heads * 2
        assert len(kv) == expected, (
            f"{spec_fn.__name__}: got {len(kv)} KV tensors, expected {expected} "
            f"({num_non_shared} non-shared layers * "
            f"{cfg.num_key_value_heads} kv-heads * 2)"
        )


def assert_gqa_emits_twice_the_kv_tensors_of_mqa(cfg: KVLayoutConfig) -> None:
    """Direct GQA-vs-MQA comparison: kv=2 must double the per-layer KV tensors.

    Guards against a future change that collapses per-head tensors back into a
    stacked layout (which would make E4B and E2B tensor counts equal again).

    Parameters
    ----------
    cfg
        The model's KV-cache geometry. Only ``num_key_value_heads`` is
        overridden, so the layer/share config is held constant and the
        comparison isolates the effect of the KV-head count.
    """
    n_mqa = len(kv_items(input_spec(replace(cfg, num_key_value_heads=1))))
    n_gqa = len(kv_items(input_spec(replace(cfg, num_key_value_heads=2))))
    assert n_gqa == 2 * n_mqa, (
        f"GQA(kv=2) should emit 2x the KV tensors of MQA(kv=1): got {n_gqa} vs {n_mqa}"
    )


def assert_kv_key_value_inner_shapes(cfg: KVLayoutConfig) -> None:
    """Key tensors are (1,1,head_dim,kv_len); value tensors are transposed.

    Locks the per-layer-type head_dim (SWA=head_dim, global=global_head_dim) and
    the key/value transpose, so a layout change can't silently swap them.

    Parameters
    ----------
    cfg
        The model's KV-cache geometry.
    """
    kv = kv_items(input_spec(cfg))
    valid_hds = {cfg.head_dim, cfg.global_head_dim}
    for name, (shape, _dtype) in kv.items():
        assert len(shape) == 4, f"{name} must be rank-4, got {shape}"
        if "_key_" in name:
            # (1, 1, head_dim, kv_len)
            assert shape[2] in valid_hds, f"{name} key head_dim {shape[2]} unexpected"
        else:  # value: (1, 1, kv_len, head_dim)
            assert shape[3] in valid_hds, f"{name} value head_dim {shape[3]} unexpected"


def assert_embedding_lut_is_written_blockwise() -> None:
    """``_write_uint16_lut`` matches the one-shot path without a huge temporary.

    ``_quantize_uint16`` promotes to float64 (``np.where`` yields int64, which
    upcasts the float32 operand), so quantizing a whole embedding table at once
    needs ~6x its float32 size in temporaries -- ~74 GiB for E4B's PLE table,
    which the OOM killer takes out with no traceback. The blocked writer must
    stay byte-identical while keeping that bounded, including on a row count
    that is not a multiple of the block size.
    """
    rng = np.random.default_rng(0)
    for rows in (_LUT_QUANT_ROWS, _LUT_QUANT_ROWS + 1, 3):
        fp = (rng.standard_normal((rows, 8), dtype=np.float32) * 4.0).astype(np.float32)
        enc = _global_uint16_encoding(fp)
        expected = _quantize_uint16(fp, enc["scale"], enc["offset"]).tobytes()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lut.bin"
            _write_uint16_lut(fp, enc["scale"], enc["offset"], path)
            assert path.read_bytes() == expected, (
                f"blocked LUT write differs from the one-shot path at {rows} rows"
            )

    # The bound that actually matters: no single quantize call ever sees more
    # than one block, so peak memory is independent of table height. Asserted
    # structurally rather than by measuring RSS, which is too noisy for CI.
    fp = np.zeros((_LUT_QUANT_ROWS * 3 + 7, 4), dtype=np.float32)
    fp[:] = rng.standard_normal(4, dtype=np.float32)
    enc = _global_uint16_encoding(fp)
    widest = 0
    real_quantize = model._quantize_uint16

    def _spy(block: np.ndarray, scale: float, offset: int) -> np.ndarray:
        nonlocal widest
        widest = max(widest, block.shape[0])
        return real_quantize(block, scale, offset)

    model._quantize_uint16 = _spy  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as td:
            _write_uint16_lut(fp, enc["scale"], enc["offset"], Path(td) / "lut.bin")
    finally:
        model._quantize_uint16 = real_quantize

    assert 0 < widest <= _LUT_QUANT_ROWS, (
        f"quantized {widest} rows at once (block size is {_LUT_QUANT_ROWS}); the "
        f"whole-table float64 temporary is back"
    )

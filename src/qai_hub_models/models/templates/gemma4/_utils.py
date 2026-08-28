# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Gemma4 helpers: load original (non-QAT) weights from safetensors into the FP
(SHA-adapted) PyTorch model.

The original ``google/gemma-4-*-it`` checkpoint stores plain float weights.
This module casts them to FP32 and maps them onto the QC-adapted model
structure (after prepare_conv + prepare_sha):
  - q_proj.weight  -> q_proj_sha.{0..7}.weight (Conv2d, per-head)
  - k/v_proj.weight -> k/v_proj_sha.{0..num_kv-1}.weight
  - o_proj.weight  -> o_proj_conv.weight
  - mlp gate/up/down -> same names (down_proj may be ConvInplaceLinear)
  - embed_tokens / lm_head -> embedding/linear weights
  - norms / layer_scalar -> loaded as-is (bfloat16 -> float32)
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from typing_extensions import Self


def resolve_shard_paths(path: str) -> list[str]:
    """Resolve a checkpoint path to one or more safetensors shard files.

    Accepts a direct ``.safetensors`` file, a directory with a single
    ``model.safetensors``, a sharded HF directory with
    ``model.safetensors.index.json``, or a directory containing loose
    ``*.safetensors`` files (fallback glob, e.g. the vision-tower layout).
    """
    p = Path(path)
    if p.is_file():
        return [str(p)]
    single = p / "model.safetensors"
    if single.exists():
        return [str(single)]
    index_path = p / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        # ``p / fn`` silently escapes ``p`` when fn is absolute or contains
        # "..", so confine the resolved shard paths to the checkpoint dir.
        root = p.resolve()
        shard_paths = []
        for fn in sorted(set(weight_map.values())):
            sp = (p / fn).resolve()
            if root not in (sp, *sp.parents):
                raise ValueError(
                    f"Shard '{fn}' in {index_path} resolves to {sp}, outside the "
                    f"checkpoint directory {root}."
                )
            shard_paths.append(str(sp))
        return shard_paths
    shards = sorted(str(x) for x in p.glob("*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No safetensors files found under {p}")


class _MultiShardSafetensors:
    """Merge multiple safetensors shards behind a single keys()/get_tensor() view."""

    def __init__(self, paths: list[str]) -> None:
        self._paths = paths
        self._stack: ExitStack | None = None
        self._key_to_handle: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self._stack = ExitStack()
        for p in self._paths:
            handle = self._stack.enter_context(safe_open(p, framework="pt"))
            for key in list(handle.keys()):
                self._key_to_handle.setdefault(key, handle)
        return self

    def __exit__(self, *exc_info: object) -> None:
        assert self._stack is not None
        self._stack.close()

    def keys(self) -> Any:
        return self._key_to_handle.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._key_to_handle

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._key_to_handle[name].get_tensor(name)


def load_dequantized_state_dict(
    safetensors_path: str,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    global_head_dim: int,
    layer_types: list[str],
    num_kv_shared_layers: int,
) -> dict[str, torch.Tensor]:
    """Build a FP32 state_dict for the SHA-adapted Gemma4 model.

    Keys match the post-prepare_sha module structure (q_proj_sha.N, o_proj_conv,
    etc.) with the "model." prefix (Gemma4ForCausalLM.model.layers...).

    ``safetensors_path`` may be a single file, a directory with
    ``model.safetensors`` / ``model.safetensors.index.json`` (sharded), or a
    directory of loose ``*.safetensors`` files.

    Returns a dict ready for model.load_state_dict(strict=False).
    """
    first_shared = num_layers - num_kv_shared_layers
    sd: dict[str, torch.Tensor] = {}

    def attn_hd(layer_idx: int) -> int:
        return (
            head_dim
            if layer_types[layer_idx] == "sliding_attention"
            else global_head_dim
        )

    shard_paths = resolve_shard_paths(safetensors_path)
    with _MultiShardSafetensors(shard_paths) as f:
        keys = set(f.keys())

        def get(name: str) -> torch.Tensor | None:
            return f.get_tensor(name) if name in keys else None

        def deq_proj(base: str) -> torch.Tensor | None:
            """Load an original (float) projection weight, cast to fp32.

            Returns the (out, in) FP32 weight, or None if ``{base}.weight`` is
            not present.
            """
            weight = get(f"{base}.weight")
            return weight.float() if weight is not None else None

        for layer_idx in range(num_layers):
            lm = f"model.language_model.layers.{layer_idx}"
            dst = f"model.layers.{layer_idx}"
            hd = attn_hd(layer_idx)
            is_shared = layer_idx >= first_shared

            # Q projection -> per-head SHA convs (Conv2d weight: [hd, hidden, 1, 1])
            fp_q = deq_proj(f"{lm}.self_attn.q_proj")
            if fp_q is not None:  # (num_heads*hd, hidden)
                for h in range(num_attention_heads):
                    head_w = fp_q[h * hd : (h + 1) * hd]  # (hd, hidden)
                    sd[f"{dst}.self_attn.q_proj_sha.{h}.weight"] = head_w[
                        :, :, None, None
                    ]

            # K/V projection -> SHA convs (only non-shared layers)
            if not is_shared:
                for proj, sha in (("k_proj", "k_proj_sha"), ("v_proj", "v_proj_sha")):
                    fp_p = deq_proj(f"{lm}.self_attn.{proj}")
                    if fp_p is not None:
                        for h in range(num_key_value_heads):
                            head_w = fp_p[h * hd : (h + 1) * hd]
                            sd[f"{dst}.self_attn.{sha}.{h}.weight"] = head_w[
                                :, :, None, None
                            ]

            # O projection -> o_proj_conv (Conv2d: [hidden, num_heads*hd, 1, 1])
            fp_o = deq_proj(f"{lm}.self_attn.o_proj")
            if fp_o is not None:
                sd[f"{dst}.self_attn.o_proj_conv.weight"] = fp_o[:, :, None, None]

            # Q/K/V norms (bfloat16 -> float32). K/V norms only for non-shared.
            norm_list = ("q_norm",) if is_shared else ("q_norm", "k_norm", "v_norm")
            for norm in norm_list:
                nw = get(f"{lm}.self_attn.{norm}.weight")
                if nw is not None:
                    n_heads = (
                        num_attention_heads if norm == "q_norm" else num_key_value_heads
                    )
                    sha_norm = f"{norm}_sha"
                    for h in range(n_heads):
                        sd[f"{dst}.self_attn.{sha_norm}.{h}.weight"] = nw.float()

            # MLP: gate/up/down are all ConvInplaceLinear (Conv2d weight
            # [out, in, 1, 1]) after prepare_conv.
            fp_gate = deq_proj(f"{lm}.mlp.gate_proj")
            if fp_gate is not None:
                sd[f"{dst}.mlp.gate_proj.weight"] = fp_gate[:, :, None, None]

            fp_up = deq_proj(f"{lm}.mlp.up_proj")
            if fp_up is not None:
                sd[f"{dst}.mlp.up_proj.weight"] = fp_up[:, :, None, None]

            fp_down = deq_proj(f"{lm}.mlp.down_proj")
            if fp_down is not None:
                # down_proj is ConvInplaceLinear (Conv2d): [hidden, intermediate, 1, 1]
                sd[f"{dst}.mlp.down_proj.weight"] = fp_down[:, :, None, None]

            # PLE gate / projection -> ConvInplaceLinear Conv2d weight
            # [out, in, 1, 1] after prepare_conv. Original checkpoints store
            # these as plain float .weight.
            for proj in ("per_layer_input_gate", "per_layer_projection"):
                pw = get(f"{lm}.{proj}.weight")
                if pw is None:
                    continue
                sd[f"{dst}.{proj}.weight"] = pw.float()[:, :, None, None]

            # Norms + layer_scalar (unquantized)
            for norm in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_per_layer_input_norm",
            ):
                nw = get(f"{lm}.{norm}.weight")
                if nw is not None:
                    sd[f"{dst}.{norm}.weight"] = nw.float()
            ls = get(f"{lm}.layer_scalar")
            if ls is not None:
                sd[f"{dst}.layer_scalar"] = ls.float()

        # Embeddings (original float tables).
        fp_embed = deq_proj("model.language_model.embed_tokens")
        if fp_embed is not None:
            sd["model.embed_tokens.weight"] = fp_embed

        fp_ple = deq_proj("model.language_model.embed_tokens_per_layer")
        if fp_ple is not None:
            sd["model.embed_tokens_per_layer.weight"] = fp_ple

        # per_layer_model_projection (unquantized bfloat16)
        plmp = get("model.language_model.per_layer_model_projection.weight")
        if plmp is not None:
            sd["model.per_layer_model_projection.weight"] = plmp.float()

        # per_layer_projection_norm (RMSNorm after PLE projection)
        plpn = get("model.language_model.per_layer_projection_norm.weight")
        if plpn is not None:
            sd["model.per_layer_projection_norm.weight"] = plpn.float()

        # Final norm
        nw = get("model.language_model.norm.weight")
        if nw is not None:
            sd["model.norm.weight"] = nw.float()

        # LM head (ConvInplaceLinear -> Conv2d weight [vocab, hidden, 1, 1]).
        # Original checkpoints commonly tie lm_head to embed_tokens
        # (tie_word_embeddings=True) and ship no separate lm_head key.
        fp_lm = deq_proj("lm_head")
        if fp_lm is None:
            fp_lm = fp_embed
        if fp_lm is not None:
            sd["lm_head.weight"] = fp_lm[:, :, None, None]

    # Projection weights must have loaded for every layer or the model is
    # silently on random init; the caller's strict=False load_state_dict would
    # never raise on this. (Norms/scalars legitimately vary, so skip those.)
    projection_keys_loaded = sum(
        1
        for k in sd
        if any(
            tag in k
            for tag in (
                "q_proj_sha",
                "k_proj_sha",
                "v_proj_sha",
                "o_proj_conv",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
    )
    expected_projections = num_layers * 4  # q, o, gate/up (per-layer minimum)
    if projection_keys_loaded < expected_projections:
        raise ValueError(
            f"Only {projection_keys_loaded} projection weights loaded from "
            f"{safetensors_path} (expected at least {expected_projections} "
            f"across {num_layers} layers). The checkpoint's weight names or "
            f"quantization layout are not recognized by deq_proj -- the model "
            f"would otherwise silently run with random projection weights."
        )
    return sd

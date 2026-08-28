# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Gemma4 model adaptations for on-device deployment.

Follows the same SHA (Split-Head Attention) + Conv2d pattern as Llama3/Qwen3:
- KV stored transposed: key shape [num_kv_heads, 1, head_dim, kv_seq_len]
- Value stored normal: [num_kv_heads, 1, kv_seq_len, head_dim]
- Attention mask: additive 4D float mask (1, 1, seq_len, kv_seq_len)
- Linear→Conv2d for all projections
- Per-head SHA splits for Q/K/V

Gemma4-specific:
- Dual head_dim: 256 (SWA) / 512 (global)
- q_norm, k_norm, v_norm per head
- Partial RoPE on global layers (25% of head_dim)
- KV shared layers (skip K/V projection, reuse from cache)
- MQA (1 KV head)
"""

from __future__ import annotations

from typing import Any

import torch
import transformers.models.gemma4.modeling_gemma4 as _mg4
from torch import nn
from transformers.activations import ACT2FN
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from qai_hub_models.models.templates.llm.common import TORCH_SUPPORTS_DYNAMIC_SHAPE
from qai_hub_models.models.templates.llm.model_adaptations import (
    ConvInplaceLinear,
    _apply_rope_single,
    repeat_kv,
)


class Gemma4RMSNorm(nn.Module):
    """Gemma4 RMSNorm - can optionally have no learnable scale (with_scale=False).

    v_norm uses with_scale=False (no learnable parameters, just normalization).
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            x = x * self.weight.float()
        return x.to(input_dtype)


class SHAGemma4Attention(nn.Module):
    """Split-Head Attention for Gemma4.

    Fully aligned with Llama3 SHALlamaAttention pattern:
    - KV transposed storage: key [1, 1, head_dim, seq], value [1, 1, seq, head_dim]
    - Additive 4D attention mask
    - Per-head Conv2d projections (after prepare_sha)
    - Q @ K (no extra transpose needed since K is already transposed)

    Gemma4-specific:
    - MQA (1 KV head) or GQA
    - q_norm, k_norm, v_norm per-head
    - Dual head_dim: 256 for SWA, 512 for global
    - Partial RoPE on global layers (25% of head_dim)
    - KV shared layers (skip K/V projection, reuse from cache)
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        )
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = getattr(config, "sliding_window", None)

        self.head_dim = (
            config.global_head_dim
            if not self.is_sliding and config.global_head_dim
            else config.head_dim
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout

        # KV sharing configuration
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        if self.is_kv_shared_layer:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            self.kv_shared_layer_index = (
                len(prev_layers)
                - 1
                - prev_layers[::-1].index(config.layer_types[layer_idx])
            )
        else:
            self.kv_shared_layer_index = None

        # Partial RoPE for global layers
        if not self.is_sliding and hasattr(config, "rope_parameters"):
            rope_params = config.rope_parameters.get("full_attention", {})
            partial_factor = rope_params.get("partial_rotary_factor", 1.0)
            self.n_rope_dims = int(self.head_dim * partial_factor)
        else:
            self.n_rope_dims = self.head_dim  # Full rotation for SWA

        # Projections (will be replaced by prepare_conv/prepare_sha)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Norms (q_norm, k_norm have scale; v_norm does not)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, with_scale=False
        )

    def prepare_conv(self) -> None:
        """Convert Linear projections to Conv2d (kernel_size=1)."""
        if hasattr(self, "q_proj_conv"):
            return

        self.q_proj_conv = nn.Conv2d(
            self.hidden_size, self.num_heads * self.head_dim, 1, bias=False
        )
        self.k_proj_conv = nn.Conv2d(
            self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=False
        )
        self.v_proj_conv = nn.Conv2d(
            self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=False
        )
        self.o_proj_conv = nn.Conv2d(
            self.num_heads * self.head_dim, self.hidden_size, 1, bias=False
        )

        self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
        self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
        self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
        self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

        del self.q_proj
        del self.k_proj
        del self.v_proj
        del self.o_proj

    def prepare_sha(self) -> None:
        """Split Conv2d into per-head Conv2d modules + per-head norms."""
        if not hasattr(self, "q_proj_conv"):
            raise RuntimeError("Must call prepare_conv() before prepare_sha().")
        if hasattr(self, "q_proj_sha"):
            return

        # Per-head Q projections (always present)
        self.q_proj_sha = nn.ModuleList(
            [
                nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False)
                for _ in range(self.num_heads)
            ]
        )
        self.q_norm_sha = nn.ModuleList(
            [
                Gemma4RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                for _ in range(self.num_heads)
            ]
        )

        # K/V projections + norms: only for NON-shared layers. Shared layers
        # reuse KV from their source layer and never project K/V, so creating
        # these modules would leave dead (uninitialized) parameters.
        if not self.is_kv_shared_layer:
            self.k_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False)
                    for _ in range(self.num_key_value_heads)
                ]
            )
            self.v_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False)
                    for _ in range(self.num_key_value_heads)
                ]
            )
            self.k_norm_sha = nn.ModuleList(
                [
                    Gemma4RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                    for _ in range(self.num_key_value_heads)
                ]
            )
            self.v_norm_sha = nn.ModuleList(
                [
                    Gemma4RMSNorm(
                        self.head_dim, eps=self.config.rms_norm_eps, with_scale=False
                    )
                    for _ in range(self.num_key_value_heads)
                ]
            )

        # Copy weights from conv into per-head splits. The isinstance asserts are
        # for mypy: ModuleList.__getitem__ returns a bare Module, hiding .weight.
        for i in range(self.num_heads):
            start = i * self.head_dim
            end = (i + 1) * self.head_dim
            q_proj = self.q_proj_sha[i]
            q_norm = self.q_norm_sha[i]
            assert isinstance(q_proj, nn.Conv2d)
            assert isinstance(q_norm, Gemma4RMSNorm)
            q_proj.weight.data.copy_(self.q_proj_conv.weight[start:end, :])
            q_norm.weight.data.copy_(self.q_norm.weight.data)

        if not self.is_kv_shared_layer:
            for i in range(self.num_key_value_heads):
                start = i * self.head_dim
                end = (i + 1) * self.head_dim
                k_proj = self.k_proj_sha[i]
                v_proj = self.v_proj_sha[i]
                k_norm = self.k_norm_sha[i]
                assert isinstance(k_proj, nn.Conv2d)
                assert isinstance(v_proj, nn.Conv2d)
                assert isinstance(k_norm, Gemma4RMSNorm)
                k_proj.weight.data.copy_(self.k_proj_conv.weight[start:end, :])
                v_proj.weight.data.copy_(self.v_proj_conv.weight[start:end, :])
                k_norm.weight.data.copy_(self.k_norm.weight.data)

        del self.q_proj_conv
        del self.k_proj_conv
        del self.v_proj_conv
        del self.q_norm
        del self.k_norm
        del self.v_norm

        # unused-ignore because whether mypy flags this depends on which torch
        # stubs are installed (same as templates/qwen3/model_adaptations.py).
        self.forward = self.forward_sha  # type: ignore[assignment, unused-ignore]

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """SHA forward pass (matches Llama3 pattern).

        Parameters
        ----------
        hidden_states
            (B, seq_len, hidden_size)
        position_embeddings
            (cos, sin) precomputed RoPE
        attention_mask
            (B, 1, seq_len, kv_seq_len) additive float mask
        past_key_values
            SHADynamicCacheNewValueOnly
        **kwargs
            unused; accepted for compatibility with the HF attention signature.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            (attn_output, attn_weights); attn_weights is None (not returned by
            the SHA path).
        """
        bsz, q_len, _ = hidden_states.size()

        # Reshape for Conv2d: (B, seq, hidden) → (B, hidden, 1, seq)
        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            hidden_states_4d = hidden_states.unsqueeze(2)
        else:
            hidden_states_4d = torch.reshape(
                hidden_states, (bsz, -1, 1, self.hidden_size)
            )
        hidden_states_4d = hidden_states_4d.transpose(1, 3)

        # KV-shared layers only project Q and reuse KV from cache; they never
        # create k/v_proj_sha.
        if self.is_kv_shared_layer:
            assert past_key_values is not None, (
                f"layer {self.layer_idx} is KV-shared and requires a cache "
                f"containing layer {self.kv_shared_layer_index}"
            )
            query_states = [
                q_norm(q_proj(hidden_states_4d).permute(0, 2, 3, 1))
                for q_proj, q_norm in zip(
                    self.q_proj_sha, self.q_norm_sha, strict=False
                )
            ]

            # cos/sin span the full head_dim//2 for both SWA (128 freqs) and
            # global (256, first 64 real and the rest zero-frequency -- the Genie
            # proportional-RoPE layout).
            assert position_embeddings is not None
            query_states = [
                _apply_rope_single(q, position_embeddings) for q in query_states
            ]

            # Take the source layer's FULL-LENGTH (past + new) KV, which it
            # stashed in past_key_values.shared_layers. key_cache would give only
            # the new tokens (new-value-only cache) and mismatch the mask.
            shared = getattr(past_key_values, "shared_layers", None)
            if shared is not None and self.kv_shared_layer_index in shared:
                past_key, past_value = shared[self.kv_shared_layer_index]
            elif hasattr(past_key_values, "key_cache"):
                past_key = past_key_values.key_cache[self.kv_shared_layer_index]
                past_value = past_key_values.value_cache[self.kv_shared_layer_index]
            elif hasattr(past_key_values.layers[self.kv_shared_layer_index], "keys"):
                past_key = past_key_values.layers[self.kv_shared_layer_index].keys
                past_value = past_key_values.layers[self.kv_shared_layer_index].values
            else:
                past_key = past_key_values.layers[self.kv_shared_layer_index][0]
                past_value = past_key_values.layers[self.kv_shared_layer_index][1]

            key_states = list(past_key)
            value_states = list(past_value)

        # Non-shared layers: project Q, K, V
        else:
            query_states = [
                q_norm(q_proj(hidden_states_4d).permute(0, 2, 3, 1))
                for q_proj, q_norm in zip(
                    self.q_proj_sha, self.q_norm_sha, strict=False
                )
            ]
            key_states = [
                k_norm(k_proj(hidden_states_4d).permute(0, 2, 3, 1))
                for k_proj, k_norm in zip(
                    self.k_proj_sha, self.k_norm_sha, strict=False
                )
            ]
            # The value cache output must come from a Transpose (memory op), not
            # the v_norm Mul: QNN propagates the upstream encoding through memory
            # ops, giving identical quant params across the ar1/ar128 +
            # part1/part2 graphs, whereas a compute op is re-estimated per graph
            # and fails Genie I/O validation.
            value_states = [
                v_norm(v_proj(hidden_states_4d).permute(0, 3, 2, 1)).transpose(1, 2)
                for v_proj, v_norm in zip(
                    self.v_proj_sha, self.v_norm_sha, strict=False
                )
            ]

            assert position_embeddings is not None
            query_states = [
                _apply_rope_single(q, position_embeddings) for q in query_states
            ]
            key_states = [
                _apply_rope_single(k, position_embeddings) for k in key_states
            ]

            # Transpose keys for cache: (B, 1, seq, hd) → (B, 1, hd, seq)
            transposed_key_states = [k.transpose(2, 3) for k in key_states]

            # KV cache update
            if past_key_values is not None:
                if hasattr(past_key_values, "key_cache"):
                    past_key = past_key_values.key_cache[self.layer_idx]
                    past_value = past_key_values.value_cache[self.layer_idx]
                elif hasattr(past_key_values.layers[self.layer_idx], "keys"):
                    past_key = past_key_values.layers[self.layer_idx].keys
                    past_value = past_key_values.layers[self.layer_idx].values
                else:
                    past_key = past_key_values.layers[self.layer_idx][0]
                    past_value = past_key_values.layers[self.layer_idx][1]

                cos, sin = position_embeddings
                cache_kwargs = {"sin": sin, "cos": cos}
                past_key_values.update(
                    transposed_key_states, value_states, self.layer_idx, cache_kwargs
                )

                # Concatenate with past KV
                key_states = [
                    torch.cat([pk, tk], dim=3)
                    for pk, tk in zip(past_key, transposed_key_states, strict=False)
                ]
                value_states = [
                    torch.cat([pv, v], dim=2)
                    for pv, v in zip(past_value, value_states, strict=False)
                ]
            else:
                key_states = transposed_key_states

            # Stash the full-length (past + new) KV: the cache only retains the
            # new tokens, but KV-shared layers need their source layer's
            # full-length KV to match the externally-prepared mask width.
            if past_key_values is not None:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = (
                    key_states,
                    value_states,
                )

        # GQA/MQA expansion
        key_states = list(repeat_kv(key_states, self.num_kv_groups))
        value_states = list(repeat_kv(value_states, self.num_kv_groups))

        # Attention: Q @ K^T (K already transposed in cache).
        # Gemma4 text attention uses scaling=1.0 (no 1/sqrt(head_dim)):
        # query_pre_attn_scalar is None and Gemma4RMSNorm folds no scale.
        attn_weights = [
            torch.matmul(q, k) for q, k in zip(query_states, key_states, strict=False)
        ]

        # Additive mask (0 = attend, -inf = mask), prepared OUTSIDE the model at
        # the correct (q_len, kv_len) size, so no in-model resizing here.
        if attention_mask is not None:
            attn_weights = [aw + attention_mask for aw in attn_weights]

        # Softmax in fp32
        attn_weights = [
            nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
                query_states[0].dtype
            )
            for aw in attn_weights
        ]

        # Attention output
        attn_output = [
            torch.matmul(aw, v)
            for aw, v in zip(attn_weights, value_states, strict=False)
        ]

        # Concat heads → o_proj_conv → reshape back
        attn_output_cat: torch.Tensor = torch.cat(attn_output, dim=3)
        attn_output_cat = attn_output_cat.permute(
            0, 3, 1, 2
        )  # (B, num_heads*hd, 1, seq)
        attn_output_cat = self.o_proj_conv(attn_output_cat)
        attn_output_cat = attn_output_cat.transpose(1, 3)  # (B, seq, 1, hidden)
        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            attn_output_cat = attn_output_cat.squeeze(2)
        else:
            attn_output_cat = attn_output_cat.reshape(bsz, q_len, self.hidden_size)

        return attn_output_cat, None


class QCGemma4MLP(nn.Module):
    """Gemma4 MLP with Conv2d support.

    Follows Llama3 pattern: only down_proj converted to Conv2d.
    Handles double-wide MLP for KV-shared layers.
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        # Double-wide MLP for shared layers
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        self.intermediate_size = config.intermediate_size * (
            2 if use_double_wide else 1
        )

        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def prepare_conv(self) -> None:
        """Convert gate/up/down projections to Conv2d.

        All three MLP projections become ConvInplaceLinear so their weights
        appear as named Conv initializers in the exported ONNX, which lets the
        weight encodings attach by name. ConvInplaceLinear handles (B, S, C)
        input transparently, so the forward above is unchanged.
        """
        self.gate_proj = ConvInplaceLinear(self.gate_proj)  # type: ignore[assignment]
        self.up_proj = ConvInplaceLinear(self.up_proj)  # type: ignore[assignment]
        self.down_proj = ConvInplaceLinear(self.down_proj)  # type: ignore[assignment]


def qc_gemma4_text_model_forward(
    self: Any,
    input_ids: torch.Tensor | None = None,
    attention_mask: Any = None,
    position_ids: Any = None,
    past_key_values: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    per_layer_inputs: torch.Tensor | None = None,
    use_cache: bool | None = None,
    swa_attention_mask: torch.Tensor | None = None,
    swa_position_ids: Any = None,
    **kwargs: Any,
) -> Any:
    """QC replacement for Gemma4TextModel.forward.

    Mirrors the reference QcGemma4TextModel: takes the global and sliding-window
    attention masks as TWO explicit inputs and routes each per layer_type, and
    takes the global / SWA RoPE embeddings as precomputed (cos, sin) tuples
    (position_ids = global, swa_position_ids = sliding). This replaces the
    fragile "bypass rotary_emb + smuggle a dict through position_ids" approach
    with an explicit, export-friendly contract.

    Accepts external embeddings: inputs_embeds (token embeddings) and
    per_layer_inputs (raw per-layer embeddings); the PLE projection/gating stay
    in-graph. Either input_ids OR inputs_embeds must be given.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if input_ids is not None:
        inputs_embeds = self.embed_tokens(input_ids)
    # The XOR check above guarantees this, but mypy can't see through it.
    assert inputs_embeds is not None

    # PLE: use externally-provided per_layer_inputs if given, else compute from
    # input_ids. Projection + gating always run in-graph.
    if self.hidden_size_per_layer_input:
        if per_layer_inputs is None:
            per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = self.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

    if position_ids is None:
        past_seen = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        position_ids = (
            torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            + past_seen
        )
        position_ids = position_ids.unsqueeze(0)

    # Build the per-layer-type causal mask mapping. The two masks are distinct
    # external inputs (global full-causal + sliding-window causal).
    if isinstance(attention_mask, dict):
        causal_mask_mapping = attention_mask
    elif swa_attention_mask is not None:
        causal_mask_mapping = {
            "full_attention": attention_mask,
            "sliding_attention": swa_attention_mask,
        }
    else:
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    # Resolve per-layer-type RoPE embeddings. Precomputed (cos, sin) tuples are
    # used directly: position_ids -> global, swa_position_ids -> sliding.
    if isinstance(position_ids, (tuple, list)):
        position_embeddings_global = position_ids
        position_embeddings_sliding = (
            swa_position_ids if swa_position_ids is not None else position_ids
        )
    else:
        position_embeddings_global = self.rotary_emb(
            inputs_embeds, position_ids, "full_attention"
        )
        position_embeddings_sliding = self.rotary_emb(
            inputs_embeds,
            swa_position_ids if swa_position_ids is not None else position_ids,
            "sliding_attention",
        )

    hidden_states = inputs_embeds
    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_type = self.config.layer_types[i]
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )
        pos_emb = (
            position_embeddings_sliding
            if layer_type == "sliding_attention"
            else position_embeddings_global
        )
        hidden_states = decoder_layer(
            hidden_states,
            per_layer_input,
            position_embeddings=pos_emb,
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    # `shared_kv_states` is left empty: KV-sharing is handled in-graph via
    # past_key_values.shared_layers. Falls back to BaseModelOutputWithPast when
    # the subclass is absent, so the valid kwargs are only known at runtime.
    output_cls: Any = getattr(
        _mg4, "Gemma4TextModelOutputWithPast", BaseModelOutputWithPast
    )
    if output_cls is BaseModelOutputWithPast:
        return output_cls(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    return output_cls(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        shared_kv_states=None,
    )

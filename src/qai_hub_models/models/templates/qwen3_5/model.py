# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3.5 text decoder as a single static-shape graph (float).

Qwen3.5 interleaves Gated DeltaNet (linear attention) layers with gated full
attention layers. Linear layers carry a fixed-size conv state and recurrent
state instead of a KV cache; the I/O names follow lm_driver's layer_cache
conventions so HubCompatibleGenerator can drive the model unchanged.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_5 import modeling_qwen3_5
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Embedding,
    LLMBase,
    sample_input,
)
from qai_hub_models.models.templates.llm.model_adaptations import ConvInplaceLinear
from qai_hub_models.models.templates.lm_driver.generator import HubCompatibleGenerator
from qai_hub_models.models.templates.lm_driver.utils.layer_cache import (
    AttentionType,
    LayerCacheDescriptor,
    build_layer_cache_descriptors,
    cache_state_names,
)
from qai_hub_models.models.templates.qwen3_5.model_adaptations import (
    SHAFullAttention,
    full_attention_forward,
    linear_attention_forward,
)
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

END_TOKENS = {"<|im_end|>", "<|endoftext|>"}


class Qwen3_5RopeEmbedding(Embedding):
    """Precomputed RoPE tables for the rotated slice of each head.

    Qwen3.5 uses 3D interleaved MRoPE, but for text the three position axes
    are equal, which reduces exactly to 1D RoPE over ``partial_rotary_factor *
    head_dim`` channels.
    """

    def __init__(
        self,
        head_dim: int | None = None,
        max_length: int = 2048,
        config: PretrainedConfig | None = None,
    ) -> None:
        assert config is not None
        rope = config.rope_parameters
        rotary_dim = int(config.head_dim * rope.get("partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (
            rope["rope_theta"]
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        freqs = torch.arange(max_length, dtype=torch.float32)[:, None] * inv_freq
        self.cos = freqs.cos()[None, None]
        self.sin = freqs.sin()[None, None]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, rotary_dim // 2][2]
        """
        cos = self.cos[0, 0].to(position_ids.device)
        sin = self.sin[0, 0].to(position_ids.device)
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class Qwen3_5Base(LLMBase):
    """FP Qwen3.5 text decoder with explicit per-layer state I/O."""

    LMClass = modeling_qwen3_5.Qwen3_5ForCausalLM
    EmbeddingClass = Qwen3_5RopeEmbedding
    GeneratorClass = HubCompatibleGenerator

    default_user_prompt = "What is gravity? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    hf_repo_name: str = ""

    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        # The config declares bfloat16, so from_pretrained would round the
        # checkpoint's fp32 tensors (e.g. linear_attn.norm) before .float().
        llm_config.dtype = torch.float32
        return llm_config

    def _verify_ckpt(self) -> None:
        if self.llm_config.model_type != "qwen3_5_text":
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT_UNQUANTIZED",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        """
        Load the float model.

        Parameters
        ----------
        checkpoint
            Hugging Face repo name or local folder. DEFAULT_UNQUANTIZED loads
            the model's Hugging Face repo.
        sequence_length
            Tokens processed per forward call.
        context_length
            Total tokens of context (KV cache size plus sequence length).
        host_device
            Torch device to load the model on.
        _skip_optimizations
            Pass ["sha_attention"] to keep the batched multi-head attention.

        Returns
        -------
        Self
            The loaded model.
        """
        if checkpoint == "DEFAULT_UNQUANTIZED":
            checkpoint = cls.hf_repo_name
        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            _skip_optimizations=_skip_optimizations,
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        use_sha = not (
            self.skip_optimizations and "sha_attention" in self.skip_optimizations
        )
        causal_lm: Any = self.model
        text_model = causal_lm.model
        # Same conversions as QCQwen3MLP / QCQwen3ForCausalLM. up_proj and
        # gate_proj stay Linear there too (AISW-148745).
        for layer in text_model.layers:
            layer.mlp.down_proj = ConvInplaceLinear(layer.mlp.down_proj)
        causal_lm.lm_head = ConvInplaceLinear(causal_lm.lm_head)
        self.sha_attention = nn.ModuleDict(
            {
                str(desc.layer_idx): SHAFullAttention(
                    text_model.layers[desc.layer_idx].self_attn
                )
                for desc in self.layer_cache_descriptors
                if use_sha and desc.attention_type != AttentionType.LINEAR
            }
        )

    @property
    def layer_cache_descriptors(self) -> list[LayerCacheDescriptor]:
        return build_layer_cache_descriptors(self.llm_config)

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        rope_cos, rope_sin, *states = args
        seq_len = input_tokens.shape[1]
        # A token is real iff it attends to itself, for left or right padding.
        # Kept in float: HTP has no integer ReduceMax, which any() lowers to.
        attends = (attention_mask[:, 0, :, -seq_len:] == 0).to(torch.float32)
        eye = torch.eye(seq_len, device=attention_mask.device)
        valid = (attends * eye).sum(dim=-1)

        causal_lm: Any = self.model
        text_model = causal_lm.model
        hidden_states = text_model.embed_tokens(input_tokens)
        new_states: list[torch.Tensor] = []
        for desc, layer in zip(
            self.layer_cache_descriptors, text_model.layers, strict=True
        ):
            state_a, state_b = states[2 * desc.layer_idx : 2 * desc.layer_idx + 2]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            if desc.attention_type == AttentionType.LINEAR:
                hidden_states, state_a, state_b = linear_attention_forward(
                    layer.linear_attn, hidden_states, valid, state_a, state_b
                )
            else:
                sha = (
                    self.sha_attention[str(desc.layer_idx)]
                    if self.sha_attention
                    else None
                )
                attention = sha or partial(full_attention_forward, layer.self_attn)
                hidden_states, state_a, state_b = attention(
                    hidden_states,
                    attention_mask,
                    rope_cos,
                    rope_sin,
                    state_a,
                    state_b,
                )
            hidden_states = residual + hidden_states
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
            new_states += [state_a, state_b]

        logits = causal_lm.lm_head(text_model.norm(hidden_states))
        return [logits, *new_states]

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Parameters
        ----------
        llm_config
            Unused; the spec is derived from the loaded config.
        sequence_length
            Tokens processed per forward call.
        context_length
            Total tokens of context (KV cache size plus sequence length).
        llm_io_type
            Only genie_input_ids is supported.

        Returns
        -------
        InputSpec
            Input specification for the model.
        """
        assert llm_io_type == LLMIOType.genie_input_ids
        assert sequence_length < context_length
        rotary_half = self.embedding.cos.shape[-1]  # type: ignore[attr-defined]
        spec: InputSpec = {
            "input_ids": TensorSpec(shape=(1, sequence_length), dtype="int32"),
            "attention_mask": TensorSpec(
                shape=(1, 1, sequence_length, context_length), dtype="float32"
            ),
            "position_ids_cos": TensorSpec(
                shape=(1, 1, sequence_length, rotary_half), dtype="float32"
            ),
            "position_ids_sin": TensorSpec(
                shape=(1, 1, sequence_length, rotary_half), dtype="float32"
            ),
        }
        descs = self.layer_cache_descriptors
        names = cache_state_names(descs, "in")
        for desc, name_a, name_b in zip(descs, names[::2], names[1::2], strict=True):
            shape_a, shape_b = desc.dummy_state_shapes(
                1, context_length, sequence_length
            )
            if desc.attention_type != AttentionType.LINEAR:
                # Hub KV layout: keys (H, B, D, S), values (H, B, S, D).
                b, h, s, d = shape_a
                shape_a, shape_b = (h, b, d, s), (h, b, s, d)
            spec[name_a] = TensorSpec(shape=shape_a, dtype="float32")
            spec[name_b] = TensorSpec(shape=shape_b, dtype="float32")
        return spec

    def get_output_spec(self) -> OutputSpec:
        names = ["logits", *cache_state_names(self.layer_cache_descriptors, "out")]
        return {name: TensorSpec() for name in names}

    def _get_output_spec(self, num_hidden_layers: int) -> OutputSpec:  # type: ignore[override]
        # get_onnx_model names ONNX outputs through this; the LLMBase version
        # would call every layer's state past_key/past_value.
        return self.get_output_spec()

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
            )
        inputs = sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )
        for name, (shape, _) in input_spec.items():
            if name.startswith(("conv_state_", "recurrent_state_")):
                inputs[name] = [torch.zeros(shape).numpy()]
        return {name: inputs[name] for name in input_spec}

    @classmethod
    def get_input_prompt_with_tags(
        cls,
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        tokenizer: Any = None,
        include_image: bool = False,
        **kwargs: Any,
    ) -> str:
        # Qwen3.5 thinks by default; demos and sample inputs answer directly.
        kwargs.setdefault("enable_thinking", False)
        return super().get_input_prompt_with_tags(
            user_input_prompt=user_input_prompt,
            system_context_prompt=system_context_prompt,
            tokenizer=tokenizer,
            include_image=include_image,
            **kwargs,
        )

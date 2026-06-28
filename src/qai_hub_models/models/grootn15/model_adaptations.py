# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import deprecate
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
    repeat_kv,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionEmbeddings,
    SiglipVisionTransformer,
)

from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.embodiment_tags import (
    EMBODIMENT_TAG_MAPPING,
    EmbodimentTag,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.action_head.cross_attention_dit import (
    BasicTransformerBlock,
    DiT,
    SelfAttentionTransformer,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.backbone.eagle_backbone import (
    DEFAULT_EAGLE_PATH,
    EagleBackbone,
)


# LLM
class EagleBackboneOpt(EagleBackbone):
    """
    Create Qwen3ForCausalLM instance truncated to `select_layer`,
    pre-computes and registers RoPE embeddings as buffers, and monkey-patches
    causal mask and RoPE forward methods to bypass dynamic control flow
    """

    def __init__(self, backbone_cfg: dict, vlm_seq_len: int) -> None:
        super().__init__(**backbone_cfg)

        # Modify LlamamModel architecture for ONNX export
        self.eagle_config = AutoConfig.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
        )
        self.eagle_config._attn_implementation = "eager"  # not use flash attention

        assert self.eagle_config.text_config.architectures[0] == "Qwen3ForCausalLM"
        self.eagle_model.language_model = Qwen3ForCausalLM(
            self.eagle_config.text_config
        )

        # remove parts of the LLM
        while (
            len(self.eagle_model.language_model.model.layers)
            > backbone_cfg["select_layer"]
        ):
            self.eagle_model.language_model.model.layers.pop(-1)

        # Bypass attention mask prep in forward
        self.eagle_model.language_model.model._update_causal_mask = MethodType(
            bypass_prepare_attention_mask, self.eagle_model.language_model.model
        )

        # Prepare rope embds
        cos, sin = prepare_qwen3_rotary_embeddings(
            self.eagle_model.language_model.model.rotary_emb, vlm_seq_len
        )
        self.register_buffer("position_embds_cos", cos)
        self.register_buffer("position_embds_sin", sin)

        # Bypass rope embd prep in forward
        self.eagle_model.language_model.model.rotary_emb.forward = MethodType(
            bypass_prepare_rope_embeddings,
            self.eagle_model.language_model.model.rotary_emb,
        )

        # Update attn impl. (uses optimized impl. where the scaling is moved after attn_mask add)
        for layer in self.eagle_model.language_model.model.layers:
            bind_method_to_instance(
                layer.self_attn, "eager_attention_forward", eager_attention_forward_opt
            )

    def forward(  # type: ignore[override, unused-ignore]
        self, input_embeds: torch.Tensor, llm_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.eagle_model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=llm_attention_mask,
            position_ids=(self.position_embds_cos, self.position_embds_sin),
            output_hidden_states=True,
        )
        eagle_features = outputs.hidden_states[self.select_layer]

        return self.eagle_linear(eagle_features)


# DiT-Opt (StateEnc + (ActionEnc + DiT + ActionDec) unrolled for K=4 iterations)
class ActionHeadDiTOpt(FlowmatchingActionHead):
    """
    Replaces the default attention processor and transformer block forward methods
    with adapted variants. Pre-registers timestep tensors and
    embodiment ID as buffers to eliminate dynamic inputs during ONNX tracing.
    Unrolls the full denoising loop (K iterations) into a single static forward pass.
    """

    def __init__(
        self, config: FlowmatchingActionHeadConfig, embodiment_tag: str | EmbodimentTag
    ) -> None:
        super().__init__(config)

        # Replace default attn processor with adapted processor for CrossAttnKV
        for block in self.model.transformer_blocks:
            block.attn1.processor = AttnProcessorOpt()

        # Replace all transformer block forward() with adapted block forward()
        for block in self.model.transformer_blocks:
            block.forward = MethodType(BasicTransformerBlockCrossAttnKV.forward, block)

        # Replace DiT forward() with adapted DiT forward()
        self.model.forward = MethodType(DiTCrossAttnKV.forward, self.model)

        # Action timesteps pos id
        self.register_buffer(
            "pos_ids", torch.arange(self.action_horizon, dtype=torch.long)
        )

        # Embodiment id
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag(embodiment_tag)

        self.register_buffer(
            "embodiment_id",
            torch.tensor(
                [EMBODIMENT_TAG_MAPPING[embodiment_tag.value]], dtype=torch.int32
            ),
        )

        # Pre-registers timestep tensors for the denoising loop.
        # t_cont uses a half-open interval [0, 1) — t=0 is pure noise, and the
        # final step evaluates velocity at t=(N-1)/N but never at t=1.0 (clean data).
        self.timesteps_tensor_names = []
        for t in range(self.num_inference_timesteps):
            name = f"timesteps_tensor_{t}"
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            self.register_buffer(name, torch.full(size=(1,), fill_value=t_discretized))
            self.timesteps_tensor_names.append(name)

    @property
    def timesteps_tensors(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in self.timesteps_tensor_names]

    def forward(  # type: ignore[override, unused-ignore]
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        vlm_proj_keys: list[torch.Tensor],
        vlm_proj_values: list[torch.Tensor],
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = int(state.shape[0])
        assert isinstance(self.embodiment_id, torch.Tensor)
        embodiment_id = self.embodiment_id.expand(B)

        # State Encoder
        state_features = self.state_encoder(state, embodiment_id)

        for t in range(self.num_inference_timesteps):
            timestep = self.timesteps_tensors[t].expand(B)
            action_features = self.action_encoder(actions, timestep, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_embs = self.position_embedding(self.pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
                state.shape[0], -1, -1
            )
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states_keys=vlm_proj_keys,
                encoder_hidden_states_values=vlm_proj_values,
                timestep=timestep,
                encoder_attention_mask=cross_attention_mask,
            )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + (1 / self.num_inference_timesteps) * pred_velocity

        return actions


### Supporting adaptations
class SiglipVisionTransformerOpt(SiglipVisionTransformer):
    """
    Forces eager attention (disables flash attention) and swaps in
    `SiglipVisionEmbeddingsOpt` to use int32 position IDs registered
    as a buffer for static ONNX tracing.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        config._attn_implementation = "eager"
        super().__init__(config)
        self.embeddings = SiglipVisionEmbeddingsOpt(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = False,
    ) -> torch.Tensor:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        return self.post_layernorm(last_hidden_state)


class SiglipVisionEmbeddingsOpt(SiglipVisionEmbeddings):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__(config)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches, dtype=torch.int32).expand((1, -1)),
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SelfAttentionTransformerOpt(SelfAttentionTransformer):
    """Add attention mask arg in the DiT self-attention block forward."""

    def forward(  # type: ignore[override, unused-ignore]
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states


class AttnProcessorOpt(AttnProcessor2_0):
    """
    Attention processor adapted for split key/value cross-attention inputs.

    Extends `AttnProcessor2_0` to accept pre-split `encoder_hidden_states_key`
    and `encoder_hidden_states_value` tensors separately, enabling the DiT
    cross-attention precomputed KV caching pattern.
    """

    def __call__(  # type: ignore[override, unused-ignore]
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states_key: torch.Tensor | None = None,
        encoder_hidden_states_value: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale") is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if encoder_hidden_states_key is None:
            batch_size, _, *_ = hidden_states.shape
        else:
            batch_size, _, _, _ = encoder_hidden_states_key.shape

        if attention_mask is not None:
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states_key is None:
            encoder_hidden_states = hidden_states

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                key = attn.norm_k(key)

        else:
            key = encoder_hidden_states_key
            value = encoder_hidden_states_value

            head_dim = key.shape[-1]

            # Use encoder attn mask for cross-attn block
            attention_mask = encoder_attention_mask

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


class BasicTransformerBlockCrossAttnKV(BasicTransformerBlock):
    """
    Transformer block adapted to accept pre-split cross-attention key/value inputs.

    Replaces the standard `encoder_hidden_states` argument with separate
    `encoder_hidden_states_key` and `encoder_hidden_states_value` tensors,
    enabling the DiT cross-attention precomputed KV caching pattern.
    """

    def forward(  # type: ignore[override, unused-ignore]
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states_key: torch.Tensor | None = None,
        encoder_hidden_states_value: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states_key=encoder_hidden_states_key,
            encoder_hidden_states_value=encoder_hidden_states_value,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiTCrossAttnKV(DiT):
    """
    DiT adapted to accept pre-split cross-attention key/value tensors per block.

    Replaces the standard `encoder_hidden_states` input with stacked
    `encoder_hidden_states_keys` and `encoder_hidden_states_values` tensors
    (one per cross-attention block).
    """

    def forward(  # type: ignore[override, unused-ignore]
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states_keys: list[torch.Tensor],  # Shape: (N, B, S, D)
        encoder_hidden_states_values: list[torch.Tensor],  # Shape: (N, B, S, D)
        timestep: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # Encode timesteps
        temb = self.timestep_encoder(timestep)
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and getattr(
                self.config, "interleave_self_attention", False
            ):
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states_key=None,
                    encoder_hidden_states_value=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states_key=encoder_hidden_states_keys[idx // 2],
                    encoder_hidden_states_value=encoder_hidden_states_values[idx // 2],
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


@torch.no_grad()
def prepare_qwen3_rotary_embeddings(
    rope_embd: Qwen3RotaryEmbedding, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len).unsqueeze(0)
    assert isinstance(rope_embd.inv_freq, torch.Tensor)
    inv_freq_expanded = (
        rope_embd.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    with torch.autocast(device_type="cpu", enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * rope_embd.attention_scaling
        sin = emb.sin() * rope_embd.attention_scaling

    return cos, sin


# Updated from transformers/models/qwen3/modeling_qwen3.py
def prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    min_dtype: float = -100.0,
) -> torch.Tensor:
    """
    Builds a 4D causal additive attention mask with optional padding mask.

    If the input mask is already 4D, it is returned as-is. Otherwise, constructs
    a causal mask of shape [B, 1, seq_len, target_len] filled with `min_dtype`
    for masked positions and 0 for attended positions. Incorporates padding from
    a 1D `attention_mask` if provided.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        diagonal_attend_mask = torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                :, None, None, :
            ].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)
    return causal_mask


def eager_attention_forward_opt(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized eager attention forward with reordered scaling.

    Moves the scaling factor application after the causal mask addition
    (instead of before) to reduce numerical operations (gets absorbed into softmax during conversion).
    """
    key_states = repeat_kv(key, module.num_key_value_groups)  # type: ignore[arg-type]
    value_states = repeat_kv(value, module.num_key_value_groups)  # type: ignore[arg-type]

    attn_weights = torch.matmul(query, key_states.transpose(2, 3))
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = (attn_weights + causal_mask * (1.0 / scaling)) * scaling

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def prepare_4d_bidirectional_mask(
    attention_mask: torch.Tensor,
    target_len: int | None = None,
    min_dtype: float = -100.0,
) -> torch.Tensor:
    """
    Convert [1, seq_len] mask to [1, 1, row_len, seq_len] additive mask.
    Valid positions = 0, masked positions = -inf.
    """
    B, _ = attention_mask.shape
    if target_len is None:
        # Expand to [1, 1, seq_len, seq_len]
        mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
    else:
        # Expand to [1, 1, target_len, seq_len]
        tgt_mask = torch.ones(
            (B, target_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        mask = tgt_mask[:, None, :, None] * attention_mask[:, None, None, :]

    # Convert to additive mask: 0 for valid, -inf for masked
    return torch.where(
        mask > 0,
        torch.zeros_like(mask, dtype=torch.float32),
        torch.full_like(mask, float(min_dtype), dtype=torch.float32),
    )


# Bypass methods
def bypass_prepare_attention_mask(
    self: object,
    attention_mask: torch.Tensor,
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    """Return attention_mask unchanged (bypass hook)."""
    return attention_mask


def bypass_prepare_rope_embeddings(
    self: object,
    hidden_states: torch.Tensor,
    position_ids: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position_ids unchanged (bypass hook)."""
    return position_ids


# Utils
# Helper to swap a model attribute and restore it afterwards
@contextmanager
def swap_attr(obj: object, name: str, new_value: object) -> Generator[None, None, None]:
    old_value = getattr(obj, name)
    setattr(obj, name, new_value)
    try:
        yield
    finally:
        setattr(obj, name, old_value)


# Allow swapping a module level method with adapted method for a scoped model instance
def bind_method_to_instance(
    model_instance: nn.Module, method_str: str, method_new: object
) -> None:
    """
    Rebinds model_instance.forward so that during this instance's forward call,
    the module-level symbol method_str is temporarily method_new.
    """
    # Grab the original bound forward function
    orig_forward = model_instance.forward

    def wrapped_forward(self: object, *args: object, **kwargs: object) -> object:
        # Find the defining module
        mod = sys.modules[self.__class__.__module__]
        # Temporarily substitute original just for this call
        with swap_attr(mod, method_str, method_new):
            return orig_forward(*args, **kwargs)

    # Bind the wrapper to this instance only
    model_instance.forward = MethodType(wrapped_forward, model_instance)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import qai_hub as hub
import torch
from transformers import PreTrainedTokenizerBase

from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.policy import (
    Gr00tPolicy,
    unsqueeze_dict_values,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.transforms import (
    GR00TTransform,
)
from qai_hub_models.models.grootn15.model import (
    DEFAULT_DATASET_ASSET,
    LoadGrootMixin,
    compute_vlm_seq_len,
    preprocess_dit,
    preprocess_llm,
    preprocess_validate_inputs,
    preprocess_vlm_proj,
)
from qai_hub_models.models.grootn15.model import GrootCollection as Model
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.evaluate.helpers import EvalMode
from qai_hub_models.utils.inference import OnDeviceModel


def _to_device_tree(x: Any, device: str, dtype: torch.dtype) -> Any:
    """Recursively move tensors in a nested structure to device."""
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            return x.to(device=device, dtype=dtype, non_blocking=True)
        # Keep original dtype
        return x.to(device=device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device, dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device_tree(v, device, dtype) for v in x)
    return x


def _split_to_list(t: torch.Tensor, B: int) -> list[torch.Tensor]:
    """Split (B, ...) tensor into list of B tensors each (1, ...)"""
    return [t[i : i + 1] for i in range(B)]


def _call_component(
    component: ExecutableModelProtocol,
    *tensors: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Call a model component with stacked (B, ...) tensors.
    - FP nn.Module  : forward() accepts (B, ...) directly.
    - OnDeviceModel : split B -> list[B x (1,...)] for B=1 on-device inference.
    """
    if isinstance(component, OnDeviceModel):
        device = tensors[0].device
        B = tensors[0].shape[0]
        result = component(*[_split_to_list(t, B) for t in tensors])  # type: ignore[arg-type]

        if isinstance(result, torch.Tensor):
            result = result.to(device)
        elif isinstance(result, (list, tuple)):
            result = type(result)(t.to(device) for t in result)
    else:
        result = component(*tensors)

    return result


@dataclass
class GrootAppConfig:
    """
    Lightweight configuration for GrootApp, extracted from Gr00tPolicy
    once at build time to avoid storing Gr00tPolicy at inference time
    """

    # Action config
    action_horizon: int
    action_dim: int

    # Padding
    vlm_pad_len: int
    pad_token_id: int

    # preprocess_llm
    image_token_index: int

    # preprocess_vlm_proj
    vlm_proj_num_heads: int

    # preprocess_dit
    dit_num_heads: int
    num_target_vision_tokens: int

    # Transforms
    modality_transform: Any
    modality_config: dict

    dtype: torch.dtype

    @staticmethod
    def from_policy(policy: Gr00tPolicy) -> GrootAppConfig:
        """Extract all scalars and callables from a loaded Gr00tPolicy."""
        action_head = policy.model.action_head
        vl_self_attention = action_head.vl_self_attention

        transform = policy._modality_transform.transforms[-1]
        if not isinstance(transform, GR00TTransform):
            raise TypeError("GR00TTransform not found in modality_transform pipeline.")

        eagle_processor = cast(Any, transform.eagle_processor)
        tokenizer = cast(PreTrainedTokenizerBase, eagle_processor.tokenizer)

        return GrootAppConfig(
            action_horizon=policy.model.action_horizon,
            action_dim=policy.model.action_dim,
            vlm_pad_len=compute_vlm_seq_len(policy),
            pad_token_id=tokenizer.pad_token_id,
            image_token_index=policy.model.backbone.eagle_model.image_token_index,
            vlm_proj_num_heads=vl_self_attention.transformer_blocks[0].attn1.heads,
            dit_num_heads=action_head.model.transformer_blocks[0].attn1.heads,
            num_target_vision_tokens=action_head.config.num_target_vision_tokens,
            modality_transform=policy._modality_transform,
            modality_config=policy.modality_config,
            dtype=policy.model.action_head.dtype,
        )


class GrootApp:
    """
    Assembles GrootCollection components to reproduce Gr00tPolicy
    inference (ViT -> LLM -> VLMProjection -> DiT diffusion loop).

    Expected inputs to predict_action_chunk (after prepare_groot_inputs):
      - "pixel_values"   : torch.Tensor [V, 3, H, W]  (V = num_cameras)
      - "input_ids"      : torch.Tensor [1, vlm_pad_len]
      - "attention_mask" : torch.Tensor [1, vlm_pad_len]
      - "state"          : torch.Tensor [1, state_dim]
      - "actions"        : torch.Tensor [1, action_horizon, action_dim]
    """

    def __init__(
        self,
        config: GrootAppConfig,
        vit: ExecutableModelProtocol,
        llm: ExecutableModelProtocol,
        vlm_proj: ExecutableModelProtocol,
        dit: ExecutableModelProtocol,
        llm_embedding_weight: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        config
            GrootAppConfig containing model configuration.
        vit
            SiglipViT vision encoder component.
        llm
            Eagle2 LLM backbone component.
        vlm_proj
            VLM projection / cross-attention KV component.
        dit
            Diffusion Transformer (DiT) action denoiser component.
        llm_embedding_weight
            Detached copy of the LLM token embedding weight matrix for preprocess
        device
            Host device
        """
        self.config = config
        self.vit = vit
        self.llm = llm
        self.vlm_proj = vlm_proj
        self.dit = dit
        self.llm_embedding_weight = llm_embedding_weight
        self.device = device

    # Internal helpers
    def _preprocess(
        self,
        step_data: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """
        Apply policy transforms and prepare model-ready inputs.
        Accepts a single dict or a list of dicts (B=N).
        Each dict is preprocessed independently then stacked — required because
        modality_transform operates per-sample.

        Returns (B, ...) stacked tensors.
        """
        if isinstance(step_data, dict):
            step_data = [step_data]

        per_sample = []
        for sd in step_data:
            sd = unsqueeze_dict_values(sd)
            inputs = self.config.modality_transform(sd)

            preprocess_validate_inputs(
                inputs, self.config.action_horizon, self.config.action_dim
            )
            eagle_prefix = "eagle_"
            inputs = {
                (k.removeprefix(eagle_prefix) if k.startswith(eagle_prefix) else k): v
                for k, v in inputs.items()
            }
            del inputs["image_sizes"]

            B = inputs["input_ids"].shape[0]
            inputs["actions"] = torch.randn(
                (B, self.config.action_horizon, self.config.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            cur_len = inputs["input_ids"].shape[1]
            if cur_len > self.config.vlm_pad_len:
                raise ValueError(
                    f"Input sequence length {cur_len} exceeds VLM pad length "
                    f"{self.config.vlm_pad_len}. Check GrootAppConfig.vlm_pad_len."
                )

            pad_ids = torch.full(
                (B, self.config.vlm_pad_len - cur_len),
                self.config.pad_token_id,
                dtype=inputs["input_ids"].dtype,
            )

            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad_ids], dim=-1).to(
                self.device
            )

            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"].to(torch.int32),
                    torch.zeros(
                        (B, self.config.vlm_pad_len - cur_len),
                        dtype=torch.int32,
                    ),
                ],
                dim=-1,
            ).to(self.device)
            per_sample.append(_to_device_tree(inputs, self.device, self.config.dtype))

        # Stack all samples into (B, ...) tensors
        stacked: dict[str, torch.Tensor] = {}
        for key in per_sample[0]:
            stacked[key] = torch.cat([s[key] for s in per_sample], dim=0)
        return stacked

    def _postprocess(
        self,
        outputs: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Post-process raw model outputs back to unnormalized action space.
        Returns (B, H, dof) shaped values
        """
        return self.config.modality_transform.unapply({"action": outputs.float().cpu()})

    def encode_vlm(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run ViT -> LLM -> VLMProjection to produce cross-attention KV pairs."""
        vit_embeds = _call_component(self.vit, pixel_values)
        assert isinstance(vit_embeds, torch.Tensor)

        input_embeds, llm_attention_mask = preprocess_llm(
            input_ids,
            vit_embeds,
            attention_mask,
            self.llm_embedding_weight,
            self.config.image_token_index,
        )
        vlm_embeds = _call_component(
            self.llm,
            input_embeds,
            llm_attention_mask,
        )
        assert isinstance(vlm_embeds, torch.Tensor)

        vlm_attention_mask = preprocess_vlm_proj(
            attention_mask,
            self.config.vlm_proj_num_heads,
        )
        vlm_proj_out = _call_component(
            self.vlm_proj,
            vlm_embeds,
            vlm_attention_mask,
        )
        assert isinstance(vlm_proj_out, tuple)

        return vlm_proj_out

    def denoise_steps(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        vlm_proj_kv_flat: tuple[torch.Tensor, ...],
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run DiT denoising steps."""
        cross_attention_mask = preprocess_dit(
            cross_attention_mask,
            self.config.dit_num_heads,
            self.config.action_horizon,
            self.config.num_target_vision_tokens,
        )

        dit_inputs_tensors: list[torch.Tensor] = [
            state,
            actions,
            cross_attention_mask,
            *vlm_proj_kv_flat,
        ]

        actions_out = _call_component(
            self.dit,
            *dit_inputs_tensors,
        )
        assert isinstance(actions_out, torch.Tensor)

        return actions_out

    def predict_action_chunk(
        self,
        step_data: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Full inference pipeline: preprocess -> VLM encode -> DiT denoise -> postprocess."""
        inputs = self._preprocess(step_data)
        vlm_proj_kv_flat = self.encode_vlm(
            inputs["pixel_values"],
            inputs["input_ids"],
            inputs["attention_mask"],
        )
        actions = self.denoise_steps(
            state=inputs["state"],
            actions=inputs["actions"],
            vlm_proj_kv_flat=vlm_proj_kv_flat,
            cross_attention_mask=inputs["attention_mask"],
        )
        return self._postprocess(actions)


# App builder
def build_app(
    policy: Gr00tPolicy,
    eval_mode: EvalMode,
    device: hub.Device | None = None,
    hub_model_id: str | None = None,
    host_device: str = "cpu",
) -> GrootApp:
    """
    Build a GrootApp with either host PyTorch components (EvalMode.FP)
    or on-device hub.InferenceJob components (EvalMode.ON_DEVICE).
    """
    config = GrootAppConfig.from_policy(policy)

    # Extract LLM embedding weights used during preprocess
    llm_embedding_weight = (
        policy.model.backbone.eagle_model.language_model.get_input_embeddings()
        .weight.detach()
        .clone()
    )

    if eval_mode == EvalMode.FP:
        print("Host FP eval")
        # Host path: use the PyTorch BaseModel instances directly.
        collection = Model.from_policy(policy)

        return GrootApp(
            config=config,
            vit=collection.components["vit"],
            llm=collection.components["llm"],
            vlm_proj=collection.components["vlm_proj"],
            dit=collection.components["dit"],
            llm_embedding_weight=llm_embedding_weight,
            device=host_device,
        )

    print("On device eval")

    # On-device path: wrap each component's InferenceJob as the callable.
    if device is None:
        raise ValueError("device must be provided for ON_DEVICE eval mode")
    if hub_model_id is None:
        raise ValueError("hub_model_id must be provided for ON_DEVICE eval mode")

    hub_ids = hub_model_id.split(",")

    component_classes: dict[str, type[LoadGrootMixin]] = Model.component_classes  # type: ignore[attr-defined]
    if len(hub_ids) != len(component_classes):
        raise ValueError(
            f"Expected {len(component_classes)} comma-separated hub model IDs "
            f"({', '.join(component_classes)}), got {len(hub_ids)}"
        )
    on_device_components = {}
    for (name, cls), hub_id in zip(component_classes.items(), hub_ids, strict=False):
        on_device_components[name] = OnDeviceModel(
            model=hub.get_model(hub_id),
            input_names=list(cls.get_input_names()),  # type: ignore[attr-defined]
            device=device,
        )

    return GrootApp(
        config=config,
        vit=on_device_components["vit"],
        llm=on_device_components["llm"],
        vlm_proj=on_device_components["vlm_proj"],
        dit=on_device_components["dit"],
        llm_embedding_weight=llm_embedding_weight,
        device=host_device,
    )


### Utils
def get_default_dataset_path() -> str:
    """Download & unpack the default dataset if not cached."""
    return str(DEFAULT_DATASET_ASSET.fetch(extract=True))

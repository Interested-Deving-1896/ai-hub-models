# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from qai_hub.client import Device
from torch import nn
from transformers import AutoConfig

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.lerobot_evaluator import LeRobotEvaluator
from qai_hub_models.models.grootn15.constants import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EMBODIMENT_TAG,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    VLM_LANGUAGE_TOKENS,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.transform.base import (
    ComposedModalityTransform,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.experiment.data_config import (
    DATA_CONFIG_MAP,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.backbone.eagle_backbone import (
    DEFAULT_EAGLE_PATH,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.policy import (
    Gr00tPolicy,
)
from qai_hub_models.models.grootn15.model_adaptations import (
    ActionHeadDiTOpt,
    EagleBackboneOpt,
    SelfAttentionTransformerOpt,
    SiglipVisionTransformerOpt,
    bypass_prepare_attention_mask,
    prepare_4d_bidirectional_mask,
    prepare_4d_causal_attention_mask_with_cache_position,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.checkpoint import CheckpointSpec, FromPretrainedMixin
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, make_torch_inputs
from qai_hub_models.utils.qai_hub_helpers import (
    ensure_hexagon_version,
    export_torch_to_onnx_zip,
)

DEFAULT_CHECKPOINT_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "model_finetuned_gr1_picknplace.tar.gz"
)

DEFAULT_DATASET_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "robot_sim_PickNPlace.tar.gz"
)


@lru_cache(maxsize=1)
def load_checkpoint(
    checkpoint: str,
    data_config: str,
    embodiment_tag: str,
    device: str = "cpu",
) -> Gr00tPolicy:
    """Load and cache a Gr00tPolicy from a model path or HF hub ID."""
    if checkpoint == "DEFAULT":
        checkpoint = str(DEFAULT_CHECKPOINT_ASSET.fetch(extract=True))
    print(f"Loading Groot checkpoint: {checkpoint}")
    data_cfg = DATA_CONFIG_MAP[data_config]
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()
    assert isinstance(modality_transform, ComposedModalityTransform)
    policy = Gr00tPolicy(
        model_path=checkpoint,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    policy.model.eval()
    policy.model = policy.model.float()
    return policy


class LoadGrootMixin(FromPretrainedMixin):
    """
    Shared mixin for all Groot component classes.
    Handles checkpoint loading and ONNX export to AIHub format.
    """

    @classmethod
    def torch_from_pretrained(  # type: ignore[override, unused-ignore]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
    ) -> Gr00tPolicy:
        return load_checkpoint(
            str(checkpoint),
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            device=str(host_device),
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        class_name = self.__class__.__name__
        path = Path(output_path) / f"{class_name}.onnx"
        assert input_spec is not None
        if path.exists():
            return str(path)
        return export_torch_to_onnx_zip(
            self.to("cpu"),  # type: ignore[attr-defined]
            str(path),
            make_torch_inputs(input_spec),
            input_names=list(input_spec.keys()),
            skip_zip=False,
            output_names=output_names,
        )

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        return ensure_hexagon_version(
            min_version=73,
            target_runtime=target_runtime,
            device=device,
            model_name="GR00T-N1.5",
        )

    def get_input_spec(self) -> InputSpec:
        """Return ordered list of input tensor names for this component."""
        raise NotImplementedError

    def get_output_spec(self) -> OutputSpec:
        """Return ordered list of output tensor names for this component."""
        raise NotImplementedError


class GrootViT(LoadGrootMixin, BaseModel):
    """
    Eagle ViT vision encoder for Groot N1.5.
    Accepts pixel_values [B*num_cameras, C, H, W] and returns
    visual embeddings [B*num_cameras, num_patches, embed_dim].

    Preprocessing is handled upstream in GrootApp.
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        self.eagle_config = AutoConfig.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
        )

        self.vision_model = SiglipVisionTransformerOpt(self.eagle_config.vision_config)

        vit_hidden_size = self.eagle_config.vision_config.hidden_size
        llm_hidden_size = self.eagle_config.text_config.hidden_size

        if self.eagle_config.mlp_connector_layers == 2:
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(
                    vit_hidden_size * int(1 / self.eagle_config.downsample_ratio) ** 2
                ),
                nn.Linear(
                    vit_hidden_size * int(1 / self.eagle_config.downsample_ratio) ** 2,
                    llm_hidden_size,
                ),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        elif (
            self.eagle_config.mlp_connector_layers == 1
            and self.eagle_config.use_pixel_shuffle
        ):
            self.mlp1 = nn.Sequential(
                nn.Linear(
                    vit_hidden_size * int(1 / self.eagle_config.downsample_ratio) ** 2,
                    llm_hidden_size,
                ),
            )
        elif (
            self.eagle_config.mlp_connector_layers == 1
            and not self.eagle_config.use_pixel_shuffle
        ):
            self.mlp1 = nn.Sequential(
                nn.Linear(vit_hidden_size, llm_hidden_size),
            )
        else:
            raise NotImplementedError(
                f"{self.eagle_config.mlp_connector_layers} is not implemented."
            )

        self.num_cameras = len(policy.modality_config["video"].modality_keys)
        self.pixel_shuffle = policy.model.backbone.eagle_model.pixel_shuffle

        # Weight copy
        self.vision_model.load_state_dict(
            policy.model.backbone.eagle_model.vision_model.vision_model.state_dict(),
            strict=False,
        )
        self.mlp1.load_state_dict(policy.model.backbone.eagle_model.mlp1.state_dict())

        self.to(policy.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values
            [B*num_cameras, C, H, W] — preprocessed images.

        Returns
        -------
        torch.Tensor
            [B*num_cameras, num_patches, embed_dim]
        """
        vit_embeds = self.vision_model(pixel_values)

        if self.eagle_config.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.eagle_config.downsample_ratio
            )  # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], -1, vit_embeds.shape[-1]
            )  # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])

        return self.mlp1(vit_embeds)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        img_size = self.eagle_config.vision_config.image_size
        return {
            "pixel_values": TensorSpec(
                shape=(self.num_cameras, 3, img_size, img_size), dtype="float32"
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {"vit_embeds": TensorSpec()}


class GrootLLMBackbone(LoadGrootMixin, BaseModel):
    """
    Groot Eagle LLM backbone (Qwen3ForCausalLM, truncated to select_layer).
    Accepts preprocessed input_embeds and 4D attention mask.
    Returns vlm_features [B, seq_len, llm_hidden_size].
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        self.vlm_seq_len = compute_vlm_seq_len(policy)
        backbone = EagleBackboneOpt(
            backbone_cfg=policy.model.config.backbone_cfg,
            vlm_seq_len=self.vlm_seq_len,
        )

        # copy weights
        backbone.load_state_dict(policy.model.backbone.state_dict(), strict=False)
        self.model: EagleBackboneOpt = backbone
        self.to(policy.device)

    def forward(
        self,
        input_embeds: torch.Tensor,  # [B, seq_len, llm_hidden_size]
        llm_attention_mask: torch.Tensor,  # [B, 1, seq_len, seq_len] 4D additive
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_embeds
            [B, seq_len, llm_hidden_size] — merged token + visual embeddings.
        llm_attention_mask
            [B, 1, seq_len, seq_len] 4D additive causal mask.

        Returns
        -------
        torch.Tensor
            [B, seq_len, llm_hidden_size]
        """
        return self.model.forward(input_embeds, llm_attention_mask)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        llm_hidden_size = self.model.eagle_config.text_config.hidden_size
        return dict(
            input_embeds=TensorSpec(
                shape=(batch_size, self.vlm_seq_len, llm_hidden_size), dtype="float32"
            ),
            llm_attention_mask=TensorSpec(
                shape=(batch_size, 1, self.vlm_seq_len, self.vlm_seq_len),
                dtype="float32",
            ),
        )

    def get_output_spec(self) -> OutputSpec:
        return {"vlm_embeds": TensorSpec()}


class GrootVLMProjection(LoadGrootMixin, BaseModel):
    """
    VLM Projection head for Groot N1.5.
    Combines VLLN + VL Self-Attention to produce cross-attention
    key/value pairs consumed by the DiT action head.

    Inputs:
      - vlm_embeds [B, seq_len, hidden_dim]
      - vlm_attention_mask [B, num_heads, seq_len, seq_len]

    Outputs:
      - vlm_proj_keys   [B, num_heads, seq_len, head_dim]
      - vlm_proj_values [B, num_heads, seq_len, head_dim]
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        action_head = policy.model.action_head

        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.dit = policy.model.action_head.model

        self.vlm_seq_len = compute_vlm_seq_len(policy)

        self.eagle_config = AutoConfig.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
        )

        self.vl_self_attention.forward = MethodType(
            SelfAttentionTransformerOpt.forward, self.vl_self_attention
        )

        # Bypass attention mask prep in attn forward
        for block in self.vl_self_attention.transformer_blocks:
            block.attn1.prepare_attention_mask = MethodType(
                bypass_prepare_attention_mask, block.attn1
            )

    def forward(
        self,
        vlm_embeds: torch.Tensor,  # [B, seq_len, hidden_dim]
        vlm_attention_mask: torch.Tensor,  # [B, H, seq_len, seq_len]
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        vlm_embeds
            [B, seq_len, hidden_dim] — VLM feature embeddings.
        vlm_attention_mask
            [B, H, seq_len, seq_len] — 4D bidirectional attention mask.

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        x = self.vlln(vlm_embeds)
        x = self.vl_self_attention(x, attention_mask=vlm_attention_mask)

        vlm_proj_keys, vlm_proj_values = [], []
        batch_size, *_ = x.shape

        for idx, block in enumerate(self.dit.transformer_blocks):
            if idx % 2 == 1 and self.dit.config.interleave_self_attention:
                pass
            else:
                key = block.attn1.to_k(x)
                value = block.attn1.to_v(x)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // block.attn1.heads

                key = key.view(batch_size, -1, block.attn1.heads, head_dim).transpose(
                    1, 2
                )
                value = value.view(
                    batch_size, -1, block.attn1.heads, head_dim
                ).transpose(1, 2)

                if block.attn1.norm_k is not None:
                    key = block.attn1.norm_k(key)

                vlm_proj_keys.append(key)
                vlm_proj_values.append(value)

        return (*vlm_proj_keys, *vlm_proj_values)

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        llm_hidden_size = self.eagle_config.text_config.hidden_size
        return dict(
            vlm_embeds=TensorSpec(
                shape=(batch_size, self.vlm_seq_len, llm_hidden_size), dtype="float32"
            ),
            vlm_attention_mask=TensorSpec(
                shape=(
                    batch_size,
                    self.vl_self_attention.transformer_blocks[0].attn1.heads,
                    self.vlm_seq_len,
                    self.vlm_seq_len,
                ),
                dtype="float32",
            ),
        )

    def get_output_spec(self) -> OutputSpec:
        n_blocks = 8  # known constant (8 cross-attn transformer_blocks)
        return {
            **{f"vlm_proj_keys_{i}": TensorSpec() for i in range(n_blocks)},
            **{f"vlm_proj_values_{i}": TensorSpec() for i in range(n_blocks)},
        }


class GrootDiT(LoadGrootMixin, BaseModel):
    """
    Diffusion Transformer (DiT) action head for Groot N1.5.
    Denoises noisy actions conditioned on VLM cross-attention keys/values,
    robot state, and a cross-attention mask.

    Inputs:
      - state               [B, 1, max_state_dim]
      - actions             [B, action_horizon, action_dim]
      - vlm_proj_keys       [B, num_heads, seq_len, head_dim]
      - vlm_proj_values     [B, num_heads, seq_len, head_dim]
      - cross_attention_mask [B, seq_len]

    Output:
      - actions_out [B, action_horizon, action_dim]
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        action_head = ActionHeadDiTOpt(
            config=policy.model.action_head.config, embodiment_tag=policy.embodiment_tag
        )

        self.config = policy.model.action_head.config
        self.vlm_seq_len = compute_vlm_seq_len(policy)

        # copy weights
        action_head.load_state_dict(policy.model.action_head.state_dict(), strict=False)
        self.model: ActionHeadDiTOpt = action_head
        self.to(policy.device)

    def forward(
        self,
        state: torch.Tensor,  # [B, state_dim]
        actions: torch.Tensor,  # [B, action_horizon, action_dim]
        cross_attention_mask: torch.Tensor,  # [B, H, sa_seq_len, vlm_seq_len]
        vlm_proj_keys_0: torch.Tensor,
        vlm_proj_keys_1: torch.Tensor,
        vlm_proj_keys_2: torch.Tensor,
        vlm_proj_keys_3: torch.Tensor,
        vlm_proj_keys_4: torch.Tensor,
        vlm_proj_keys_5: torch.Tensor,
        vlm_proj_keys_6: torch.Tensor,
        vlm_proj_keys_7: torch.Tensor,
        vlm_proj_values_0: torch.Tensor,
        vlm_proj_values_1: torch.Tensor,
        vlm_proj_values_2: torch.Tensor,
        vlm_proj_values_3: torch.Tensor,
        vlm_proj_values_4: torch.Tensor,
        vlm_proj_values_5: torch.Tensor,
        vlm_proj_values_6: torch.Tensor,
        vlm_proj_values_7: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        state
            [B, 1, max_state_dim] — robot state tensor.
        actions
            [B, action_horizon, action_dim] — noisy actions for diffusion init.
        cross_attention_mask
            [B, H, sa_seq_len, vlm_seq_len] — cross-attention mask.
        vlm_proj_keys_0
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_1
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_2
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_3
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_4
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_5
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_6
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_7
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_0
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_1
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_2
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_3
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_4
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_5
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_6
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_7
            [B, num_heads, seq_len, head_dim]

        Returns
        -------
        torch.Tensor
            [B, action_horizon, action_dim] denoised actions.
        """
        vlm_proj_keys = [
            vlm_proj_keys_0,
            vlm_proj_keys_1,
            vlm_proj_keys_2,
            vlm_proj_keys_3,
            vlm_proj_keys_4,
            vlm_proj_keys_5,
            vlm_proj_keys_6,
            vlm_proj_keys_7,
        ]
        vlm_proj_values = [
            vlm_proj_values_0,
            vlm_proj_values_1,
            vlm_proj_values_2,
            vlm_proj_values_3,
            vlm_proj_values_4,
            vlm_proj_values_5,
            vlm_proj_values_6,
            vlm_proj_values_7,
        ]

        return self.model(
            state=state,
            actions=actions,
            vlm_proj_keys=vlm_proj_keys,
            vlm_proj_values=vlm_proj_values,
            cross_attention_mask=cross_attention_mask,
        )

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        num_heads = self.model.model.config.num_attention_heads
        head_dim = self.model.model.config.attention_head_dim
        action_horizon = self.model.action_horizon
        num_target_vision_tokens = self.config.num_target_vision_tokens

        spec: InputSpec = dict(
            state=TensorSpec(
                shape=(batch_size, 1, self.config.max_state_dim), dtype="float32"
            ),
            actions=TensorSpec(
                shape=(batch_size, self.config.action_horizon, self.config.action_dim),
                dtype="float32",
            ),
            cross_attention_mask=TensorSpec(
                shape=(
                    batch_size,
                    num_heads,
                    action_horizon + 1 + num_target_vision_tokens,
                    self.vlm_seq_len,
                ),
                dtype="float32",
            ),
        )
        n_blocks = len(self.model.model.transformer_blocks) // 2
        for i in range(n_blocks):
            spec[f"vlm_proj_keys_{i}"] = TensorSpec(
                shape=(batch_size, num_heads, self.vlm_seq_len, head_dim),
                dtype="float32",
            )
        for i in range(n_blocks):
            spec[f"vlm_proj_values_{i}"] = TensorSpec(
                shape=(batch_size, num_heads, self.vlm_seq_len, head_dim),
                dtype="float32",
            )

        return spec

    def get_output_spec(self) -> OutputSpec:
        return {"actions_out": TensorSpec()}


### Preprocess methods
def preprocess_llm(
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    image_token_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merges vit_embeds into input_embeds and builds the 4D causal attention mask."""
    input_embeds = torch.nn.functional.embedding(
        input_ids, embedding_weight
    )  # [B, N, C]
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)

    # Replace image token positions with ViT embeddings
    selected = input_ids_flat == image_token_index
    input_embeds_flat[selected] = vit_embeds.reshape(-1, C)
    input_embeds = input_embeds_flat.reshape(B, N, C)

    # Build 4D causal attention mask
    llm_attention_mask = prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=N,
        target_length=N,
        dtype=input_embeds.dtype,
        device=input_embeds.device,
        cache_position=torch.arange(N, device=input_embeds.device),
        batch_size=B,
    )
    return input_embeds, llm_attention_mask


def preprocess_vlm_proj(
    attention_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Builds the 4D bidirectional attention mask, repeated for each attention head."""
    vlm_attention_mask = prepare_4d_bidirectional_mask(attention_mask)
    return vlm_attention_mask.repeat_interleave(num_heads, dim=1)


def preprocess_dit(
    attention_mask: torch.Tensor,
    num_heads: int,
    action_horizon: int,
    num_target_vision_tokens: int,
) -> torch.Tensor:
    """Builds the 4D bidirectional mask for DiT cross-attention, repeated per head."""
    cross_attention_mask = prepare_4d_bidirectional_mask(
        attention_mask,
        target_len=action_horizon
        + 1
        + num_target_vision_tokens,  # actions + state token + future tokens
    )
    return cross_attention_mask.repeat_interleave(num_heads, dim=1)


def preprocess_validate_inputs(
    inputs: dict,
    action_horizon: int,
    action_dim: int,
) -> None:
    detected_error = False
    error_msg = ""
    if "action" in inputs:
        action = inputs["action"]
        type_ok = isinstance(action, torch.Tensor)
        shape_ok = (
            len(action.shape) == 3
            and action.shape[1] == action_horizon
            and action.shape[2] == action_dim
        )
        if not type_ok:
            error_msg += f"\n{action.dtype=}"
            detected_error = True
        if not shape_ok:
            error_msg += f"\n{action.shape=}"
            detected_error = True
    if "video" in inputs:
        video = inputs["video"]
        if not isinstance(video, np.ndarray):
            error_msg += f"\n{type(video)=}"
            detected_error = True
        if video.dtype != np.uint8:
            error_msg += f"\n{video.dtype=}"
            detected_error = True
        if not (len(video.shape) == 6 and video.shape[3] == 3):
            error_msg += f"\n{video.shape=}"
            detected_error = True
    if detected_error:
        raise ValueError(error_msg)


### Utils
def compute_vlm_seq_len(policy: Gr00tPolicy) -> int:
    """
    Compute the padded VLM input sequence length from a Gr00tPolicy instance.
    The sequence is laid out as:
        [language tokens (128)] + [vision tokens per camera x num_cameras]
    """
    eagle_cfg = AutoConfig.from_pretrained(
        DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
    )
    num_cameras = len(policy.modality_config["video"].modality_keys)
    raw_patches = (
        eagle_cfg.vision_config.image_size // eagle_cfg.vision_config.patch_size
    ) ** 2
    if getattr(eagle_cfg, "use_pixel_shuffle", False):
        vit_tokens = int(raw_patches * eagle_cfg.downsample_ratio**2)
    else:
        vit_tokens = raw_patches
    return VLM_LANGUAGE_TOKENS + num_cameras * vit_tokens


# -----------------------------------------------------------------
# GrootCollection — registers all sub-models as components
# -----------------------------------------------------------------
class GrootCollection(WorkbenchModelCollection):
    """
    AIHub CollectionModel for Groot N1.5.

    Components:
      - vit       : Eagle vision encoder (ViT)
      - llm       : Eagle LLM backbone (produces VLM embeddings)
      - vlm_proj  : VLM projection + self-attention (keys/values for DiT)
      - dit       : DiT action head (denoising diffusion transformer)

    Usage::

        model = GrootCollection.from_pretrained()
        # model.components["vit"], ["llm"], ["vlm_proj"], ["dit"]
    """

    def __init__(
        self,
        vit: GrootViT,
        llm: GrootLLMBackbone,
        vlm_proj: GrootVLMProjection,
        dit: GrootDiT,
    ) -> None:
        super().__init__({"vit": vit, "llm": llm, "vlm_proj": vlm_proj, "dit": dit})

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        **kwargs: object,
    ) -> GrootCollection:
        policy = load_checkpoint(
            checkpoint=str(checkpoint),
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            device=str(host_device),
        )

        return cls(
            GrootViT(policy),
            GrootLLMBackbone(policy),
            GrootVLMProjection(policy),
            GrootDiT(policy),
        )

    @classmethod
    def from_policy(cls, policy: Gr00tPolicy) -> GrootCollection:
        """Build a GrootCollection from an already-loaded policy."""
        return cls(
            GrootViT(policy),
            GrootLLMBackbone(policy),
            GrootVLMProjection(policy),
            GrootDiT(policy),
        )

    def get_evaluator(self) -> BaseEvaluator:
        return LeRobotEvaluator()

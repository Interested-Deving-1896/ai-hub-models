# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models._shared.llm.llm_helpers import get_kv_cache_names
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import (
    PrecompiledWorkbenchModelCollection,
)
from qai_hub_models.utils.base_model import PrecompiledWorkbenchModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2


def get_cached_asset(part: int) -> str:
    model_path = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "snapdragon_8_elite/models.zip"
    ).fetch(extract=True)
    return os.path.join(model_path, f"weight_sharing_model_{part}_of_4.serialized.bin")


class PromptProcessor_Part1(PrecompiledWorkbenchModel):
    """
    Prompt Processor Part1 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> PromptProcessor_Part1:
        return PromptProcessor_Part1(get_cached_asset(part=1))

    def get_input_spec(self) -> InputSpec:
        return {
            "input_ids": TensorSpec(shape=(1, 128), dtype="int32"),
            "past_key_1_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_1_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_0_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_0_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_2_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_2_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_3_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_3_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_4_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_4_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_5_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_5_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_6_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_6_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_7_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_7_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 128, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [
                *get_kv_cache_names(start=0, end=8),
                "_model_layers_7_Add_1_Add_output_0",
            ]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_1_of_4"
        )


class PromptProcessor_Part2(PrecompiledWorkbenchModel):
    """
    Prompt Processor Part2 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> PromptProcessor_Part2:
        return PromptProcessor_Part2(get_cached_asset(part=2))

    def get_input_spec(self) -> InputSpec:
        return {
            "_model_layers_7_Add_1_Add_output_0": TensorSpec(
                shape=(1, 128, 4096), dtype="uint16"
            ),
            "past_key_9_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_9_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_8_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_8_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_10_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_10_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_11_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_11_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_12_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_12_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_13_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_13_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_14_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_14_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_15_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_15_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 128, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [
                *get_kv_cache_names(start=8, end=16),
                "_model_layers_15_Add_1_Add_output_0",
            ]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_2_of_4"
        )


class PromptProcessor_Part3(PrecompiledWorkbenchModel):
    """
    Prompt Processor Part3 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> PromptProcessor_Part3:
        return PromptProcessor_Part3(get_cached_asset(part=3))

    def get_input_spec(self) -> InputSpec:
        return {
            "_model_layers_15_Add_1_Add_output_0": TensorSpec(
                shape=(1, 128, 4096), dtype="uint16"
            ),
            "past_key_17_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_17_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_16_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_16_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_18_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_18_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_19_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_19_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_20_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_20_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_21_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_21_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_22_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_22_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_23_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_23_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 128, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [
                *get_kv_cache_names(start=16, end=24),
                "_model_layers_23_Add_1_Add_output_0",
            ]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_3_of_4"
        )


class PromptProcessor_Part4(PrecompiledWorkbenchModel):
    """
    Prompt Processor Part4 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> PromptProcessor_Part4:
        return PromptProcessor_Part4(get_cached_asset(part=4))

    def get_input_spec(self) -> InputSpec:
        return {
            "_model_layers_23_Add_1_Add_output_0": TensorSpec(
                shape=(1, 128, 4096), dtype="uint16"
            ),
            "past_key_25_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_25_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_24_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_24_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_26_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_26_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_27_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_27_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_28_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_28_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_29_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_29_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_30_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_30_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "past_key_31_in": TensorSpec(shape=(8, 1, 128, 3968), dtype="uint8"),
            "past_value_31_in": TensorSpec(shape=(8, 1, 3968, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 128, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 128, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [*get_kv_cache_names(start=24, end=32), "logits"]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_4_of_4"
        )


class TokenGenerator_Part1(PrecompiledWorkbenchModel):
    """
    Token Generator Part1 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> TokenGenerator_Part1:
        return TokenGenerator_Part1(get_cached_asset(part=1))

    def get_input_spec(self) -> InputSpec:
        return {
            "input_ids": TensorSpec(shape=(1, 1), dtype="int32"),
            "past_key_1_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_1_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_0_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_0_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_2_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_2_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_3_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_3_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_4_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_4_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_5_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_5_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_6_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_6_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_7_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_7_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 1, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [
                *get_kv_cache_names(start=0, end=8),
                "_model_layers_7_Add_1_Add_output_0",
            ]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_1_of_4"
        )


class TokenGenerator_Part2(PrecompiledWorkbenchModel):
    """
    Token Generator Part2 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> TokenGenerator_Part2:
        return TokenGenerator_Part2(get_cached_asset(part=2))

    def get_input_spec(self) -> InputSpec:
        return {
            "_model_layers_7_Add_1_Add_output_0": TensorSpec(
                shape=(1, 1, 4096), dtype="uint16"
            ),
            "past_key_9_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_9_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_8_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_8_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_10_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_10_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_11_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_11_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_12_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_12_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_13_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_13_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_14_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_14_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_15_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_15_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 1, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [
                *get_kv_cache_names(start=8, end=16),
                "_model_layers_15_Add_1_Add_output_0",
            ]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_2_of_4"
        )


class TokenGenerator_Part3(PrecompiledWorkbenchModel):
    """
    Token Generator Part3 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> TokenGenerator_Part3:
        return TokenGenerator_Part3(get_cached_asset(part=3))

    def get_input_spec(self) -> InputSpec:
        return {
            "_model_layers_15_Add_1_Add_output_0": TensorSpec(
                shape=(1, 1, 4096), dtype="uint16"
            ),
            "past_key_17_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_17_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_16_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_16_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_18_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_18_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_19_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_19_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_20_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_20_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_21_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_21_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_22_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_22_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_23_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_23_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 1, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [
                *get_kv_cache_names(start=16, end=24),
                "_model_layers_23_Add_1_Add_output_0",
            ]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_3_of_4"
        )


class TokenGenerator_Part4(PrecompiledWorkbenchModel):
    """
    Token Generator Part4 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_pretrained(cls) -> TokenGenerator_Part4:
        return TokenGenerator_Part4(get_cached_asset(part=4))

    def get_input_spec(self) -> InputSpec:
        return {
            "_model_layers_23_Add_1_Add_output_0": TensorSpec(
                shape=(1, 1, 4096), dtype="uint16"
            ),
            "past_key_25_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_25_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_24_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_24_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_26_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_26_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_27_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_27_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_28_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_28_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_29_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_29_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_30_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_30_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "past_key_31_in": TensorSpec(shape=(8, 1, 128, 4095), dtype="uint8"),
            "past_value_31_in": TensorSpec(shape=(8, 1, 4095, 128), dtype="uint8"),
            "position_ids_cos": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "position_ids_sin": TensorSpec(shape=(1, 1, 1, 64), dtype="uint16"),
            "attention_mask": TensorSpec(shape=(1, 1, 1, 4096), dtype="uint16"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            name: TensorSpec()
            for name in [*get_kv_cache_names(start=24, end=32), "logits"]
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_4_of_4"
        )


class Mistral_7B_Instruct_v0_3(PrecompiledWorkbenchModelCollection):
    """
    Mistral_7B_Instruct_v0_3 class consists of
        - Prompt Processor (split into 4 parts)
        - Token Generator (split into 4 parts)

    All models are pre-trained, quantized (int4/int8 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    def __init__(
        self,
        prompt_processor_part_1: PromptProcessor_Part1,
        prompt_processor_part_2: PromptProcessor_Part2,
        prompt_processor_part_3: PromptProcessor_Part3,
        prompt_processor_part_4: PromptProcessor_Part4,
        token_generator_part_1: TokenGenerator_Part1,
        token_generator_part_2: TokenGenerator_Part2,
        token_generator_part_3: TokenGenerator_Part3,
        token_generator_part_4: TokenGenerator_Part4,
    ) -> None:
        super().__init__(
            {
                "prompt_processor_part_1": prompt_processor_part_1,
                "prompt_processor_part_2": prompt_processor_part_2,
                "prompt_processor_part_3": prompt_processor_part_3,
                "prompt_processor_part_4": prompt_processor_part_4,
                "token_generator_part_1": token_generator_part_1,
                "token_generator_part_2": token_generator_part_2,
                "token_generator_part_3": token_generator_part_3,
                "token_generator_part_4": token_generator_part_4,
            }
        )

    @classmethod
    def from_pretrained(cls) -> Mistral_7B_Instruct_v0_3:
        return cls(
            PromptProcessor_Part1.from_pretrained(),
            PromptProcessor_Part2.from_pretrained(),
            PromptProcessor_Part3.from_pretrained(),
            PromptProcessor_Part4.from_pretrained(),
            TokenGenerator_Part1.from_pretrained(),
            TokenGenerator_Part2.from_pretrained(),
            TokenGenerator_Part3.from_pretrained(),
            TokenGenerator_Part4.from_pretrained(),
        )

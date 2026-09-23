# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torchvision.models as tv_models
from qai_hub.client import Device
from torchvision.models.swin_transformer import PatchMergingV2, ShiftedWindowAttention
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.templates.common import replace_module_recursively
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models.templates.swin.swin_transformer import (
    AutoSplitLinear,
    ShiftedWindowAttentionInf,
)

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class SwinV2Base(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = tv_models.swin_v2_b(weights=weights)
        replace_module_recursively(
            net, ShiftedWindowAttention, ShiftedWindowAttentionInf
        )
        replace_module_recursively(
            net, torch.nn.Linear, AutoSplitLinear, parent_module=PatchMergingV2
        )
        return cls(net)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        # -O2 works around severe QAIRT numerical regressions on qnn_dlc (tetracode #21465).
        if target_runtime == TargetRuntime.QNN_DLC:
            other_compile_options += " -O2"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

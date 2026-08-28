# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models.templates.efficientvit.external_repos.efficientvit.efficientvit.cls_model_zoo import (
    create_cls_model,
)
from qai_hub_models.models.templates.efficientvit.litemla_patch import (
    apply_attention_denominator_floor,
)
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]

DEFAULT_WEIGHTS = "b2-r288.pt"
MODEL_ASSET_VERSION = 1

# Quant-safe floor for the LiteMLA attention-normalization denominator (see
# qai_hub_models.models.templates.efficientvit.litemla_patch for the full rationale).
# The floor must clear the int-0 / coarse-reciprocal regime without tripping the
# server mixed-precision promotion. Chosen by an on-device w8a16 sweep: device
# top1 climbs with eps (1->53%, 8->68%, 30->74%) and reaches ~79% at 90.
ATTENTION_EPS = 90


class EfficientViT(ImagenetClassifier):
    """Exportable EfficientViT Image classifier, end-to-end."""

    @classmethod
    def from_pretrained(
        cls, weights: str | None = None, precision: Precision = Precision.float
    ) -> Self:
        """Load EfficientViT from a weightfile created by the source repository.

        The LiteMLA attention-normalization denominator is floored (see
        ATTENTION_EPS) so it cannot quantize to integer 0 and trigger a
        divide-by-zero on the HTP fixed-point reciprocal. The floor is applied
        unconditionally rather than only for quantized precisions: the scorecard
        quantize path traces the model at float precision and then quantizes that
        ONNX graph, so the Clip must already be present in the float-traced graph
        for the downstream quantized model to inherit it. The float accuracy cost
        is small (~0.8% top-1).
        """
        if not weights:
            weights = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()

        efficientvit_model = create_cls_model(name="b2", weight_url=weights)
        efficientvit_model.to(torch.device("cpu"))
        efficientvit_model.eval()
        apply_attention_denominator_floor(efficientvit_model, ATTENTION_EPS)
        return cls(efficientvit_model)

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10

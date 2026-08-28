# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.templates.llm.evaluate import llm_evaluate
from qai_hub_models.models.templates.llm.model import LLM_QNN


def vlm_evaluate(
    *,
    quantized_model_cls: type,
    fp_model_cls: type,
    default_sequence_length: int | list[int] | None = None,
    vision_encoder_cls: type | None = None,
    hf_repo_name: str | None = None,
    vlm_image_size: tuple[int, int] | None = None,
    end_tokens: set[str] | None = None,
) -> None:
    """Run VLM evaluation via the shared ``llm_evaluate``.

    The caller (per-model ``evaluate.py``) is responsible for resolving the
    ``--use-presplit`` toggle and passing the chosen ``quantized_model_cls`` /
    ``fp_model_cls`` (PreSplit or Split-wrapper), exactly as the Llama models do.
    """
    llm_evaluate(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        default_sequence_length=default_sequence_length,
        vision_encoder_cls=vision_encoder_cls,
        hf_repo_name=hf_repo_name,
        vlm_image_size=vlm_image_size,
        end_tokens=end_tokens,
    )

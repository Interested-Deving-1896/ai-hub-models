# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.llm.demo import llm_chat_demo
from qai_hub_models.models._shared.llm.model import LLM_QNN, LLM_AIMETOnnx, LLMBase
from qai_hub_models.models.falcon_v3_7b_instruct.model import (
    END_TOKENS,
    HF_REPO_NAME,
    HF_REPO_URL,
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Falcon3_7B_PreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec


def falcon_3_7b_chat_demo(
    model_cls: type[LLM_AIMETOnnx] = QuantizedSplitModelWrapper,
    fp_model_cls: type[LLMBase] = Falcon3_7B_PreSplit,
    qnn_model_cls: type[LLM_QNN] = LLM_QNN,
    model_id: str = MODEL_ID,
    end_tokens: set = END_TOKENS,
    hf_repo_name: str = HF_REPO_NAME,
    hf_repo_url: str = HF_REPO_URL,
    default_prompt: str | None = None,
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    llm_chat_demo(
        model_cls=model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=qnn_model_cls,
        model_id=model_id,
        end_tokens=end_tokens,
        hf_repo_name=hf_repo_name,
        hf_repo_url=hf_repo_url,
        supported_precisions=SUPPORTED_PRECISIONS,
        default_prompt=default_prompt,
        test_checkpoint=test_checkpoint,
    )


if __name__ == "__main__":
    falcon_3_7b_chat_demo()

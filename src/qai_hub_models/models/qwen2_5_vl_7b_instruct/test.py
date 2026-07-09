# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.llm_helpers import (
    log_evaluate_test_result,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    LLM_QNN,
)
from qai_hub_models.models.qwen2_5_vl_7b_instruct import (
    MODEL_ID,
    VisionEncoder,
)
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
)

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen2_5_VL_7B_PreSplit.release()
    Qwen2_5_VL_7B_QuantizablePreSplit.release()
    Qwen2_5_VL_7B_QuantizablePreSplit.from_pretrained(checkpoint=checkpoint)


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        pytest.param("DEFAULT", "wikitext", 10.38, 0, marks=pytest.mark.nightly),
        ("DEFAULT", "mmlu", 0.689, 1000),
        ("DEFAULT", "mmmu", 0.525, 200),
        # Image+prompt generation + LLM-grader smoke test (5 samples). Weekly
        # (evaluate-only) since VLM generation is slow. The grader label can
        # flip across hosts, so expected_metric is a floor.
        ("DEFAULT", "multimodal_prompts", 0.88, 5),
        ("DEFAULT_UNQUANTIZED", "wikitext", 8.86, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.73, 0),
        ("DEFAULT_UNQUANTIZED", "mmmu", 0.525, 200),
        ("DEFAULT_UNQUANTIZED", "multimodal_prompts", 0.84, 5),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
    tmp_path: Path,
) -> None:
    dataset_cls = next(
        d
        for d in Qwen2_5_VL_7B_PreSplit.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen2_5_VL_7B_PreSplit.release()
    Qwen2_5_VL_7B_QuantizablePreSplit.release()
    # The prompt-generation tasks persist responses and grade them in a
    # separate venv; everything else scores a forward-only metric inline.
    task_kwargs = (
        {"output_dir": str(tmp_path)}
        if task in {"prompts", "multimodal_prompts"}
        else None
    )
    actual_metric, _ = evaluate(
        quantized_model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_model_cls=Qwen2_5_VL_7B_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]  # placeholder — no QNN variant yet
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        kwargs=dict(
            checkpoint=checkpoint,
        ),
        vision_encoder_cls=VisionEncoder,
        hf_repo_name=HF_REPO_NAME,
        vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        task_kwargs=task_kwargs,
    )
    log_evaluate_test_result(
        model_name=MODEL_ID,
        checkpoint="DEFAULT_W4A16" if checkpoint == "DEFAULT" else checkpoint,
        metric=task,
        value=actual_metric,
    )
    if task in {"prompts", "multimodal_prompts"}:
        # Grader score is monotonic (higher = better); assert a floor.
        assert actual_metric >= expected_metric, (
            f"{task} grader score {actual_metric:.3f} below floor {expected_metric}"
        )
    else:
        np.testing.assert_allclose(actual_metric, expected_metric, rtol=0.06, atol=0)

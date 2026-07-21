# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest
import torch

from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.models.qwen3_vl_8b_instruct import (
    MODEL_ID,
    VisionEncoder,
)
from qai_hub_models.models.qwen3_vl_8b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    Qwen3_VL_8B_PreSplit,
    Qwen3_VL_8B_QuantizablePreSplit,
)

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        pytest.param("DEFAULT", "wikitext", 8.34, 0, marks=pytest.mark.nightly),
        ("DEFAULT", "mmlu", 0.755, 1000),
        ("DEFAULT", "mmmu", 0.55, 200),
        # Image+prompt generation + LLM-grader smoke test (5 samples). Weekly
        # (evaluate-only) since VLM generation is slow. The grader label can
        # flip across hosts, so expected_metric is a floor.
        ("DEFAULT_UNQUANTIZED", "wikitext", 7.78, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.73, 0),
        ("DEFAULT_UNQUANTIZED", "mmmu", 0.54, 200),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    dataset_cls = next(
        d
        for d in Qwen3_VL_8B_PreSplit.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen3_VL_8B_PreSplit.release()
    Qwen3_VL_8B_QuantizablePreSplit.release()
    # This VLM has no split-Parts wrapper; the monolithic PreSplit classes serve
    # both the forward-only and prompt-generation paths.
    test.run_llm_evaluate_test(
        task=task,
        checkpoint=checkpoint,
        expected_metric=expected_metric,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        quantized_split_cls=Qwen3_VL_8B_QuantizablePreSplit,
        fp_split_cls=Qwen3_VL_8B_PreSplit,
        quantized_presplit_cls=Qwen3_VL_8B_QuantizablePreSplit,
        fp_presplit_cls=Qwen3_VL_8B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        model_id=MODEL_ID,
        log_checkpoint="DEFAULT_W4A16" if checkpoint == "DEFAULT" else checkpoint,
        add_unquantized_extra_kwargs=False,
        evaluate_kwargs=dict(
            vision_encoder_cls=VisionEncoder,
            hf_repo_name=HF_REPO_NAME,
            vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        ),
    )

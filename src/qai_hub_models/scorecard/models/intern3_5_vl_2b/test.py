# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest
import torch

from qai_hub_models.models.intern3_5_vl_2b import (
    MODEL_ID,
    VisionEncoder,
)
from qai_hub_models.models.intern3_5_vl_2b.demo import (
    intern3_5_vl_2b_chat_demo,
)
from qai_hub_models.models.intern3_5_vl_2b.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    Intern3_5_VL_2B_PreSplit,
    Intern3_5_VL_2B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Intern3_5_VL_2B_PreSplit.release()
    Intern3_5_VL_2B_QuantizablePreSplit.release()
    Intern3_5_VL_2B_QuantizablePreSplit.from_pretrained(checkpoint=checkpoint)


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        pytest.param("DEFAULT", "wikitext", 12.16, 0, marks=pytest.mark.nightly),
        ("DEFAULT", "mmlu", 0.552, 1000),
        ("DEFAULT", "mmmu", 0.520, 200),
        # Image+prompt generation + LLM-grader smoke test (5 samples). Weekly
        # (evaluate-only) since VLM generation is slow. The grader label can
        # flip across hosts, so expected_metric is a floor.
        ("DEFAULT_UNQUANTIZED", "wikitext", 11.27, 0),
        ("DEFAULT_UNQUANTIZED", "mmlu", 0.624, 1000),
        ("DEFAULT_UNQUANTIZED", "mmmu", 0.53, 200),
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
        for d in Intern3_5_VL_2B_PreSplit.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Intern3_5_VL_2B_PreSplit.release()
    Intern3_5_VL_2B_QuantizablePreSplit.release()
    # This VLM has no split-Parts wrapper; the monolithic PreSplit classes serve
    # both the forward-only and prompt-generation paths.
    test.run_llm_evaluate_test(
        task=task,
        checkpoint=checkpoint,
        expected_metric=expected_metric,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        quantized_split_cls=Intern3_5_VL_2B_QuantizablePreSplit,
        fp_split_cls=Intern3_5_VL_2B_PreSplit,
        quantized_presplit_cls=Intern3_5_VL_2B_QuantizablePreSplit,
        fp_presplit_cls=Intern3_5_VL_2B_PreSplit,
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


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Intern3_5_VL_2B_PreSplit.release()
    Intern3_5_VL_2B_QuantizablePreSplit.release()
    intern3_5_vl_2b_chat_demo(test_checkpoint=checkpoint)
    captured = capsys.readouterr()
    assert any(
        line.startswith("    + ") and line[6:].strip()
        for line in captured.out.splitlines()
    ), "Demo did not stream any non-empty generated output."

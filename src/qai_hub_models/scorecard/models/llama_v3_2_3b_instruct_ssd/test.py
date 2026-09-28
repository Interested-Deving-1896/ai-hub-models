# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest
import torch

from qai_hub_models.models.llama_v3_2_3b_instruct_ssd import Model
from qai_hub_models.models.llama_v3_2_3b_instruct_ssd.demo import llama_3_2_3b_chat_demo
from qai_hub_models.models.llama_v3_2_3b_instruct_ssd.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    Llama3_2_3B_SSD_PreSplit,
    Llama3_2_3B_SSD_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.model import DEFAULT_CONTEXT_LENGTH
from qai_hub_models.utils.checkpoint import CheckpointSpec

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Llama3_2_3B_SSD_PreSplit.release()
    Llama3_2_3B_SSD_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained()


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        # W4A16 split quantsim OOMs the GPU runner after ~9 wikitext / ~20 mmlu samples.
        ("DEFAULT_W4A16", "wikitext", 11.96, 8),
        ("DEFAULT_W4A16", "mmlu", 0.4, 10),
        ("DEFAULT_UNQUANTIZED", "wikitext", 10.14, 0),
        ("DEFAULT_UNQUANTIZED", "mmlu", 0.607, 1000),
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
        for d in FPSplitModelWrapper.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Llama3_2_3B_SSD_PreSplit.release()
    Llama3_2_3B_SSD_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    test.run_llm_evaluate_test(
        task=task,
        checkpoint=checkpoint,
        expected_metric=expected_metric,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        quantized_split_cls=QuantizedSplitModelWrapper,
        fp_split_cls=FPSplitModelWrapper,
        quantized_presplit_cls=Llama3_2_3B_SSD_QuantizablePreSplit,
        fp_presplit_cls=Llama3_2_3B_SSD_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        model_id=MODEL_ID,
    )


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Llama3_2_3B_SSD_PreSplit.release()
    Llama3_2_3B_SSD_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    llama_3_2_3b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out

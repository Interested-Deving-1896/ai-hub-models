# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)
from qai_hub_models.models.gemma_4_e2b_it.demo import gemma_4_e2b_it_chat_demo
from qai_hub_models.models.gemma_4_e2b_it.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    Gemma4_E2B_PreSplit,
    Gemma4_E2B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec

DEFAULT_EVAL_SEQLEN = [DEFAULT_SEQUENCE_LENGTH, 1]


# Parked until E2B's eval path is fixed. The presplit and split-Parts paths
# disagree 2.2x on identical FP weights (161.24 vs 72.26) even though E2B sets
# NUM_SPLITS = 1, so its "split" is the whole graph and the two should be
# numerically indistinguishable. The presplit side is the trustworthy one (it
# matches an independent QAT reference of 164.7), so no E2B number here can be
# pinned as accuracy yet. E4B is unaffected -- both its paths agree to 0.1% --
# and carries the real gate. Expected values below are placeholders, not
# measurements; set them from a run once the path bug is closed.
@pytest.mark.skip(
    reason="E2B eval path: presplit vs split-Parts disagree 2.2x on identical FP "
    "weights at NUM_SPLITS=1. No trustworthy metric to assert until fixed."
)
@pytest.mark.nightly
@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4A16", "wikitext_chat", 74.41, 0),
        ("DEFAULT_UNQUANTIZED", "wikitext_chat", 161.24, 0),
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
    Gemma4_E2B_PreSplit.release()
    Gemma4_E2B_QuantizablePreSplit.release()
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
        quantized_presplit_cls=Gemma4_E2B_QuantizablePreSplit,
        fp_presplit_cls=Gemma4_E2B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        model_id=MODEL_ID,
    )


@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_quantize_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quantize the model and verify it can respond with 'Paris'."""
    Gemma4_E2B_PreSplit.release()
    Gemma4_E2B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    checkpoint_path = test.setup_test_quantization(
        Gemma4_E2B_QuantizablePreSplit,
        Gemma4_E2B_PreSplit,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint="DEFAULT",
        use_seq_mse=False,
    )
    gemma_4_e2b_it_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint_path,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
    Gemma4_E2B_PreSplit.release()
    Gemma4_E2B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT_W4A16", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Gemma4_E2B_PreSplit.release()
    Gemma4_E2B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    gemma_4_e2b_it_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out

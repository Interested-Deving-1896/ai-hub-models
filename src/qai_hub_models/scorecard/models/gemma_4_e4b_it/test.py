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
from qai_hub_models.models.gemma_4_e4b_it.demo import gemma_4_e4b_it_chat_demo
from qai_hub_models.models.gemma_4_e4b_it.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    Gemma4_E4B_PreSplit,
    Gemma4_E4B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec

DEFAULT_EVAL_SEQLEN = [DEFAULT_SEQUENCE_LENGTH, 1]


# wikitext_chat, not raw wikitext: E4B is instruction-tuned and is degenerate on
# raw prose at BOTH precisions (E2B measures 94.66 FP / 99.94 w4a16), so raw
# perplexity scores a prose/chat mismatch rather than quantization quality. See
# datasets/wikitext/wikitext_chat.py.
#
# Both values measured on the full test split (37 blocks, context_length=4096),
# w4a16 calibrated Seq-MSE + chat. The FP row runs on the monolithic PreSplit
# (run_llm_evaluate_test's fp_baseline_uses_presplit default), which is why it is
# the presplit 55.66 and not the split-Parts 55.60; the two agree to 0.1%.
# Calibration used --num-samples 20, so these may shift once a full-size
# calibration lands.
@pytest.mark.nightly
@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4A16", "wikitext_chat", 58.63, 0),
        ("DEFAULT_UNQUANTIZED", "wikitext_chat", 55.66, 0),
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
    Gemma4_E4B_PreSplit.release()
    Gemma4_E4B_QuantizablePreSplit.release()
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
        quantized_presplit_cls=Gemma4_E4B_QuantizablePreSplit,
        fp_presplit_cls=Gemma4_E4B_PreSplit,
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
    Gemma4_E4B_PreSplit.release()
    Gemma4_E4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    checkpoint_path = test.setup_test_quantization(
        Gemma4_E4B_QuantizablePreSplit,
        Gemma4_E4B_PreSplit,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint="DEFAULT",
        use_seq_mse=False,
    )
    gemma_4_e4b_it_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint_path,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
    Gemma4_E4B_PreSplit.release()
    Gemma4_E4B_QuantizablePreSplit.release()
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
    Gemma4_E4B_PreSplit.release()
    Gemma4_E4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    gemma_4_e4b_it_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out

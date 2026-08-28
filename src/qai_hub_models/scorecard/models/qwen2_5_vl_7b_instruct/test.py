# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from qai_hub_models import Precision
from qai_hub_models.models.qwen2_5_vl_7b_instruct import (
    MODEL_ID,
    VisionEncoder,
)
from qai_hub_models.models.qwen2_5_vl_7b_instruct.demo import (
    qwen2_5_vl_7b_instruct_chat_demo,
)
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    SAMPLE_IMAGE,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
)
from qai_hub_models.models.qwen2_5_vl_7b_instruct.quantize import (
    quantize_vision_encoder,
)
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.model import DEFAULT_CONTEXT_LENGTH
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export.context import resolve_recipe_dir
from qai_hub_models.utils.export.dispatch import select_pipeline

export_model = select_pipeline(resolve_recipe_dir(MODEL_ID))

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
        ("DEFAULT", "multimodal_prompts", 0.84, 5),
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
    # This VLM has no split-Parts wrapper; the monolithic PreSplit classes serve
    # both the forward-only and prompt-generation paths.
    test.run_llm_evaluate_test(
        task=task,
        checkpoint=checkpoint,
        expected_metric=expected_metric,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        quantized_split_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_split_cls=Qwen2_5_VL_7B_PreSplit,
        quantized_presplit_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_presplit_cls=Qwen2_5_VL_7B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        tmp_path=tmp_path,
        model_id=MODEL_ID,
        log_checkpoint="DEFAULT_W4A16" if checkpoint == "DEFAULT" else checkpoint,
        add_unquantized_extra_kwargs=False,
        evaluate_kwargs=dict(
            vision_encoder_cls=VisionEncoder,
            hf_repo_name=HF_REPO_NAME,
            vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
        ),
    )


@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_quantize_and_demo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quantize the full VLM (text splits + vision encoder), then run the demo
    on a sample image and verify it produces a non-empty response.

    Drives the real CLI (``--checkpoint`` / ``--image`` / ``--prompt`` via
    ``sys.argv``) rather than the ``test_checkpoint`` shortcut, so the image
    input path is exercised end-to-end.
    """
    Qwen2_5_VL_7B_PreSplit.release()
    Qwen2_5_VL_7B_QuantizablePreSplit.release()
    # Pass 1: LLM text model.
    checkpoint_path = test.setup_test_quantization(
        Qwen2_5_VL_7B_QuantizablePreSplit,
        Qwen2_5_VL_7B_PreSplit,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint="DEFAULT",
        use_seq_mse=False,
        image_size=(DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH),
    )
    # Pass 2: vision encoder. Writes vision_encoder.{onnx,data,encodings} into
    # the same checkpoint dir so the checkpoint is a complete quantized VLM.
    quantize_vision_encoder(output_dir=checkpoint_path, num_calibration_samples=10)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo.py",
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(SAMPLE_IMAGE.fetch()),
            "--prompt",
            "Describe this image.",
            "--max-output-tokens",
            "10",
        ],
    )
    qwen2_5_vl_7b_instruct_chat_demo()
    captured = capsys.readouterr()
    assert "white dog" in captured.out
    Qwen2_5_VL_7B_PreSplit.release()
    Qwen2_5_VL_7B_QuantizablePreSplit.release()


@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    """Text-only smoke test of the demo (no image)."""
    Qwen2_5_VL_7B_PreSplit.release()
    Qwen2_5_VL_7B_QuantizablePreSplit.release()
    qwen2_5_vl_7b_instruct_chat_demo(test_checkpoint=checkpoint)
    captured = capsys.readouterr()
    assert "LLM" in captured.out

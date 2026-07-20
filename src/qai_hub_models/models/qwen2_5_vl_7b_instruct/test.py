# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.llm_helpers import (
    create_genie_config,
    log_evaluate_test_result,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    LLM_QNN,
)
from qai_hub_models.models._shared.qwen2_vl.model import get_vlm_config
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
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export.dispatch import resolve_model, select_pipeline

export_model = select_pipeline(resolve_model(MODEL_ID))

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    """The VLM builds a Qwen2.5-VL MRoPE genie config under the
    ``text-generator`` key (not the plain ``dialog`` config the text-only
    LLMs emit). Assert that structure end-to-end.
    """
    context_length = 2048
    # Base transformers can't resolve ``qwen2_5_vl`` via AutoConfig; the model
    # loads it through this helper (which registers the architecture).
    llm_config = get_vlm_config(HF_REPO_NAME)
    text_config = llm_config.text_config
    model_list = [f"qwen2_5_vl_7b_instruct_part_{i}_of_5.bin" for i in range(1, 6)]
    vlm_rope_config: dict[str, Any] = {
        "rope-type": "qwen2vl-mrope",
        "time-step": 50,
        "spatial-merge-size": 2,
        "mrope-section": text_config.rope_scaling["mrope_section"],
    }
    actual_config = create_genie_config(
        context_length=context_length,
        llm_config=text_config,
        embedding_type="rope",
        model_list=model_list,
        embedding_size=text_config.hidden_size,
        top_level_key="text-generator",
        embedding_lut_path="embedding_weights.raw",
        vlm_rope_config=vlm_rope_config,
    )
    kv_dim = text_config.hidden_size // text_config.num_attention_heads
    expected_config: dict[str, Any] = {
        "text-generator": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": context_length,
                "n-vocab": text_config.vocab_size,
                "bos-token": text_config.bos_token_id,
                "eos-token": text_config.eos_token_id,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": kv_dim,
                        "allow-async-init": False,
                        "enable-graph-switching": False,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                    "positional-encoding": {
                        "type": "rope",
                        "rope-dim": kv_dim // 2,
                        "rope-theta": int(text_config.rope_theta),
                        "rope-scaling": vlm_rope_config,
                    },
                },
            },
            "embedding": {
                "version": 1,
                "type": "lut",
                "lut-path": "embedding_weights.raw",
                "size": text_config.hidden_size,
                "datatype": "float32",
            },
        }
    }

    assert expected_config == actual_config


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

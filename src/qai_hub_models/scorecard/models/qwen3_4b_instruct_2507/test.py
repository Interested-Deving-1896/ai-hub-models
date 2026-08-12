# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.common import cleanup, get_qdc_api_token
from qai_hub_models.models._shared.llm.llm_helpers import (
    log_perf_on_device_result,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)
from qai_hub_models.models.qwen3_4b_instruct_2507 import Model
from qai_hub_models.models.qwen3_4b_instruct_2507.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_4B_Instruct_2507_Part1_Of_4,
    Qwen3_4B_Instruct_2507_Part4_Of_4,
    Qwen3_4B_Instruct_2507_PartBase,
    Qwen3_4B_Instruct_2507_PreSplit,
    Qwen3_4B_Instruct_2507_QuantizablePreSplit,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import DEFAULT_QDC_DEVICE, cs_8_elite_qrd
from qai_hub_models.scorecard.utils.testing_export_eval import run_llm_compile
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export.context import resolve_recipe_dir
from qai_hub_models.utils.export.dispatch import select_pipeline
from qai_hub_models.utils.export.result import MultiGraphCollectionExportResult

export_model = select_pipeline(resolve_recipe_dir(MODEL_ID))

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_4B_Instruct_2507_PreSplit.release()
    Qwen3_4B_Instruct_2507_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


@pytest.mark.evaluate
@pytest.mark.parametrize(
    "part_cls",
    [Qwen3_4B_Instruct_2507_Part1_Of_4, Qwen3_4B_Instruct_2507_Part4_Of_4],
)
def test_part_quantsim_loads_encodings(
    part_cls: type[Qwen3_4B_Instruct_2507_PartBase],
) -> None:
    """Building a Part's QuantSim must load the migrated encodings.

    Qwen3-4B-Instruct-2507 ties lm_head.weight to the embedding table, so the
    dynamo graph names the single tied initializer ``model.lm_head.weight`` and
    feeds it to both the embedding ``Gather`` (Part1) and the lm_head ``MatMul``
    (Part4). The migrated per-channel lm_head encoding is loadable in Part4 but
    must be relaxed/stripped for the Gather input in Part1 (which has no
    ``tensor_quantizer_params``). This exercises both ends; Part1 regression-
    tests the tied-embedding fix in
    ``Qwen3_4B_Instruct_2507_PartBase._get_quant_sim``.
    """
    Qwen3_4B_Instruct_2507_PreSplit.release()
    Qwen3_4B_Instruct_2507_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    part = part_cls.from_pretrained(
        checkpoint="DEFAULT_W4A16",
        _skip_quantsim_creation=True,
        sequence_lengths=[DEFAULT_SEQUENCE_LENGTH],
        context_lengths=[DEFAULT_CONTEXT_LENGTH],
    )
    # Must not raise (regression: per-channel load on a Gather-fed tied weight).
    quant_sim = part._get_quant_sim()
    assert quant_sim is not None


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4A16", "wikitext", 10.39, 0),
        ("DEFAULT_W4A16", "mmlu", 0.690, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 9.39, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.74, 0),
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
    Qwen3_4B_Instruct_2507_PreSplit.release()
    Qwen3_4B_Instruct_2507_QuantizablePreSplit.release()
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
        quantized_presplit_cls=Qwen3_4B_Instruct_2507_QuantizablePreSplit,
        fp_presplit_cls=Qwen3_4B_Instruct_2507_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        model_id=MODEL_ID,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test can be run on GPU only.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device", "checkpoint"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_8_elite_qrd, "DEFAULT_W4A16"),
    ],
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    checkpoint: CheckpointSpec,
) -> None:
    Qwen3_4B_Instruct_2507_PreSplit.release()
    Qwen3_4B_Instruct_2507_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    result = run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint=checkpoint,
            _skip_quantsim_creation=True,
            output_dir=test.GENIE_BUNDLES_ROOT,
        ),
        skip_compile_options=True,
        skip_downloading=False,
    )
    assert os.path.exists(test.GENIE_BUNDLES_ROOT)
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    assert (genie_bundle_path / "tokenizer.json").exists()
    assert (genie_bundle_path / "genie_config.json").exists()
    assert (genie_bundle_path / "htp_backend_ext_config.json").exists()
    assert (genie_bundle_path / "sample_prompt.txt").exists()

    assert isinstance(result, MultiGraphCollectionExportResult)


@pytest.mark.skipif(
    not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_8_elite_qrd),
    ],
)
@pytest.mark.qdc
def test_qdc(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    cleanup()
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    if not (genie_bundle_path / "genie_config.json").exists():
        pytest.fail("The genie bundle does not exist.")
    from qai_hub_models.models._shared.llm.qdc.genie_jobs import (
        _USE_DEFAULT_PROMPTS,
        submit_genie_bundle_to_qdc_device,
    )

    qdc_job_name = f"Genie {MODEL_ID} {precision}"
    tps, prefill_tps, min_ttft_ms, _ = submit_genie_bundle_to_qdc_device(
        get_qdc_api_token(device),
        device.reference_device.name,
        str(genie_bundle_path),
        job_name=qdc_job_name,
        eval_prompts=(_USE_DEFAULT_PROMPTS if device == DEFAULT_QDC_DEVICE else None),
    )
    assert tps is not None and min_ttft_ms is not None, "QDC execution failed."
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        prefill_tps=prefill_tps,
        ttft_ms=min_ttft_ms,
    )
    assert tps > 6.0
    assert min_ttft_ms < 250.0

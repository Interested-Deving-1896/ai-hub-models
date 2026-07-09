# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared LLM performance collection test.

Replaces the per-model test_llm_perf tests so the whole LLM fleet runs in
one venv with one pytest invocation. The per-model copies were import-heavy
(each pulled the model's source-repo pins through model.py / export.py),
which is why CollectLLMPerfTask used to spin up a fresh venv per model.
This file only needs qaihm + qai_hub + qualcomm_device_cloud_sdk +
transformers (tokenizer) — no AIMET, no source-repo deps.
"""

from __future__ import annotations

import importlib
import os

import pytest

from qai_hub_models import Precision
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.llm_helpers import log_perf_on_device_result
from qai_hub_models.models._shared.llm.perf_collection import (
    LLMPerfConfig,
    get_llm_perf_parametrization,
)
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.device import (
    cs_8_elite_gen_5_qrd,
    cs_8_elite_qrd,
    cs_x_elite,
)
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

# Sampled LLM coverage for the shared perf test. One entry per representative
# model/device/precision combo -- membership here is also the allowlist, so
# other LLMs in the repo (model_type_llm=True with a quantize.py) are skipped.
# CI overrides device/precision via QAIHM_TEST_DEVICES / QAIHM_LLM_PERF_PRECISIONS.
_LLM_DEFAULTS: dict[str, tuple[list[ScorecardDevice], list[Precision]]] = {
    "llama_v3_2_1b_instruct": ([cs_8_elite_qrd], [Precision.w4]),
    "qwen2_5_vl_7b_instruct": ([cs_8_elite_gen_5_qrd], [Precision.w4a16]),
    "qwen3_8b": ([cs_x_elite], [Precision.w4a16]),
}


def _llm_model_ids() -> list[str]:
    """Sampled LLM model IDs (see _LLM_DEFAULTS), verified against the repo."""
    return [
        model_id
        for model_id in _LLM_DEFAULTS
        if (QAIHM_MODELS_ROOT / model_id / "quantize.py").exists() and _is_llm(model_id)
    ]


def _is_llm(model_id: str) -> bool:
    try:
        info = QAIHMModelInfo.from_model(model_id)
    except Exception:
        return False
    return info.model_type_llm


def _build_params() -> list[tuple[str, Precision, ScorecardDevice]]:
    params: list[tuple[str, Precision, ScorecardDevice]] = []
    for model_id in _llm_model_ids():
        defaults = _LLM_DEFAULTS.get(model_id, (None, None))
        combos = get_llm_perf_parametrization(
            model_id,
            default_devices=defaults[0],
            default_precisions=defaults[1],
        )
        for precision, device in combos:
            params.append((model_id, precision, device))
    return params


def _param_id(val: object) -> str:
    if isinstance(val, ScorecardDevice):
        return val.name
    return str(val)


@pytest.fixture(scope="session")
def llm_perf_config() -> LLMPerfConfig:
    return LLMPerfConfig.from_environment()


@pytest.mark.llm_perf
@pytest.mark.skipif(
    not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(
    ("model_id", "precision", "device"),
    _build_params(),
    ids=_param_id,
)
def test_llm_perf(
    model_id: str,
    precision: Precision,
    device: ScorecardDevice,
    llm_perf_config: LLMPerfConfig,
) -> None:
    tps, ttft, prefill_tps = test.run_llm_perf_test(
        model_id=model_id,
        device=device,
        precision=precision,
        output_dir=os.path.join(model_id, test.GENIE_BUNDLES_ROOT),
        qairt_sdk_path=llm_perf_config.qairt_sdk_path,
        skip_perf_update=llm_perf_config.skip_perf_update,
    )
    log_perf_on_device_result(
        model_name=model_id,
        precision=str(precision),
        device=device.name,
        tps=tps,
        prefill_tps=prefill_tps,
        ttft_ms=ttft,
    )

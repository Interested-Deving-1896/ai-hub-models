# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.models.templates.llm.perf_collection import (
    get_llm_perf_parametrization,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import LLM_COMPILE_DEVICES, ScorecardDevice
from qai_hub_models.scorecard.envvars import IgnoreKnownFailuresEnvvar
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_profile_parameterized_pytest_config,
)
from qai_hub_models.utils.set_env import set_temp_env

MODEL_ID = "mobilenet_v2"
ISSUE = "https://github.com/qcom-ai-hub/tetracode/issues/1234"
DISABLED_DEVICE = "cs_8_elite"
DISABLED_HUB_DEVICE = ScorecardDevice.get(DISABLED_DEVICE).device_name
OTHER_HUB_DEVICE = ScorecardDevice.get("cs_8_gen_3").device_name


def _manifest_with_device_disable(timeout: bool = False) -> QAIHMModelManifest:
    manifest = QAIHMModelManifest.from_model(MODEL_ID)
    reasons = manifest.disabled_paths.get_disable_reasons(
        Precision.float, TargetRuntime.TFLITE
    )
    reasons.issue = ISSUE
    reasons.causes_timeout = timeout
    reasons.disable_devices = [DISABLED_HUB_DEVICE]
    return manifest


def test_is_supported_respects_device() -> None:
    manifest = _manifest_with_device_disable()
    assert not manifest.is_supported(
        Precision.float, TargetRuntime.TFLITE, device=DISABLED_HUB_DEVICE
    )
    assert manifest.is_supported(
        Precision.float, TargetRuntime.TFLITE, device=OTHER_HUB_DEVICE
    )
    assert manifest.is_supported(Precision.float, TargetRuntime.TFLITE)


def test_device_issue_ignored_when_user_failures_excluded() -> None:
    manifest = _manifest_with_device_disable()
    assert manifest.is_supported(
        Precision.float,
        TargetRuntime.TFLITE,
        consider_user_defined_failures=False,
        device=DISABLED_HUB_DEVICE,
    )


def test_device_timeout_applies_even_when_user_failures_excluded() -> None:
    manifest = _manifest_with_device_disable(timeout=True)
    assert not manifest.is_supported(
        Precision.float,
        TargetRuntime.TFLITE,
        consider_user_defined_failures=False,
        device=DISABLED_HUB_DEVICE,
    )


def _profile_devices(manifest: QAIHMModelManifest) -> set[str]:
    paths = manifest.get_supported_paths_for_testing(only_include_passing=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(QAIHMModelManifest, "from_model", lambda _id: manifest)
        configs = get_profile_parameterized_pytest_config(MODEL_ID, paths, paths)
    return {
        device.name
        for precision, path, device in configs
        if precision == Precision.float
        and isinstance(path, ScorecardProfilePath)
        and path.runtime == TargetRuntime.TFLITE
    }


def test_parameterization_skips_disabled_device() -> None:
    with set_temp_env({IgnoreKnownFailuresEnvvar.VARNAME: "0"}):
        baseline = _profile_devices(QAIHMModelManifest.from_model(MODEL_ID))
        disabled = _profile_devices(_manifest_with_device_disable())
    assert DISABLED_DEVICE in baseline
    assert DISABLED_DEVICE not in disabled
    assert disabled == baseline - {DISABLED_DEVICE}


def test_parameterization_keeps_device_when_ignoring_known_failures() -> None:
    with set_temp_env({IgnoreKnownFailuresEnvvar.VARNAME: "1"}):
        assert DISABLED_DEVICE in _profile_devices(_manifest_with_device_disable())
        assert DISABLED_DEVICE not in _profile_devices(
            _manifest_with_device_disable(timeout=True)
        )


LLM_MODEL_IDS = ["qwen3_8b", "qwen3_vl_8b_instruct"]
LLM_DISABLED_DEVICES = {"cs_8_elite", "cs_8_elite_gen_5"}
LLM_PERF_ENV = {"QAIHM_TEST_DEVICES": "all", "QAIHM_LLM_MODELS": ""}


def _llm_perf_devices(model_id: str, ignore_known_failures: str) -> set[str]:
    env: dict[str, str | None] = {
        IgnoreKnownFailuresEnvvar.VARNAME: ignore_known_failures,
        **LLM_PERF_ENV,
    }
    with set_temp_env(env):
        return {d.name for _, d in get_llm_perf_parametrization(model_id)}


@pytest.mark.parametrize("model_id", LLM_MODEL_IDS)
def test_llm_compile_parameterization_skips_disabled_devices(model_id: str) -> None:
    manifest = QAIHMModelManifest.from_model(model_id)
    paths = manifest.get_supported_paths_for_testing(only_include_passing=False)
    with set_temp_env({IgnoreKnownFailuresEnvvar.VARNAME: "0"}):
        configs = get_compile_parameterized_pytest_config(
            model_id, paths, paths, is_llm=True
        )
    assert not {device.name for _, _, device in configs} & LLM_DISABLED_DEVICES


@pytest.mark.parametrize("model_id", LLM_MODEL_IDS)
def test_llm_perf_parametrization_skips_disabled_devices(model_id: str) -> None:
    assert not _llm_perf_devices(model_id, "0") & LLM_DISABLED_DEVICES


@pytest.mark.parametrize("model_id", LLM_MODEL_IDS)
def test_llm_perf_parametrization_keeps_devices_when_ignoring_known_failures(
    model_id: str,
) -> None:
    expected = {d.name for d in LLM_COMPILE_DEVICES if d.qdc_enabled}
    assert LLM_DISABLED_DEVICES & expected <= _llm_perf_devices(model_id, "1")


def test_llm_perf_parametrization_follows_geniex_qairt_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = QAIHMModelManifest.from_model("qwen3_4b")
    reasons = manifest.disabled_paths.get_disable_reasons(
        Precision.w4a16, TargetRuntime.GENIEX_QAIRT
    )
    reasons.issue = ISSUE
    reasons.disable_devices = [
        ScorecardDevice.get(name).device_name for name in sorted(LLM_DISABLED_DEVICES)
    ]
    monkeypatch.setattr(QAIHMModelManifest, "from_model", lambda _id: manifest)
    assert not _llm_perf_devices("qwen3_4b", "0") & LLM_DISABLED_DEVICES

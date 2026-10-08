# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_disable_reasons import (
    ModelDisableReasons,
    ModelDisableReasonsMapping,
)

ISSUE = "https://github.com/qcom-ai-hub/tetracode/issues/1234"
S24 = "Samsung Galaxy S24 (Family)"
S25 = "Samsung Galaxy S25 (Family)"
S26 = "Samsung Galaxy S26 (Family)"


def test_disable_devices_round_trips() -> None:
    mapping = ModelDisableReasonsMapping()
    reasons = mapping.get_disable_reasons(Precision.w8a8, TargetRuntime.TFLITE)
    reasons.issue = ISSUE
    reasons.causes_timeout = True
    reasons.disable_devices = [S25, S26]
    dumped = mapping.model_dump(mode="json", exclude_defaults=True)
    assert dumped["w8a8"]["tflite"]["disable_devices"] == [
        S25,
        S26,
    ]
    reloaded = ModelDisableReasonsMapping(**dumped)
    assert reloaded.data[Precision.w8a8][TargetRuntime.TFLITE].disable_devices == [
        S25,
        S26,
    ]


def test_no_disable_devices_is_omitted_from_yaml() -> None:
    mapping = ModelDisableReasonsMapping()
    mapping.get_disable_reasons(Precision.w8a8, TargetRuntime.TFLITE).issue = ISSUE
    dumped = mapping.model_dump(mode="json", exclude_defaults=True)
    assert "disable_devices" not in dumped["w8a8"]["tflite"]


def test_disable_devices_requires_issue() -> None:
    with pytest.raises(ValueError, match="issue must also be provided"):
        ModelDisableReasons(disable_devices=[S25])


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        ModelDisableReasons(issue=ISSUE, disable_devices=["not_a_device"])


def test_scorecard_alias_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown device 'cs_8_elite'"):
        ModelDisableReasons(issue=ISSUE, disable_devices=["cs_8_elite"])


def test_issue_applies_to_device() -> None:
    everywhere = ModelDisableReasons(issue=ISSUE)
    assert everywhere.issue_applies_to_device(None)
    assert everywhere.issue_applies_to_device(S25)

    scoped = ModelDisableReasons(issue=ISSUE, disable_devices=[S25])
    assert scoped.issue_applies_to_device(S25)
    assert not scoped.issue_applies_to_device(S24)
    assert not scoped.issue_applies_to_device(None)


def test_failure_reason_for_device_scopes_issue() -> None:
    scoped = ModelDisableReasons(issue=ISSUE, disable_devices=[S25])
    assert scoped.failure_reason_for_device(S25) == ISSUE
    assert scoped.failure_reason_for_device(S24) is None
    assert scoped.failure_reason_for_device() is None


def test_failure_reason_for_device_keeps_scorecard_failures_runtime_wide() -> None:
    reasons = ModelDisableReasons(
        scorecard_failure="Failing jobs: abc", issue=ISSUE, disable_devices=[S25]
    )
    assert reasons.failure_reason_for_device(S24) == "Failing jobs: abc"
    assert reasons.failure_reason_for_device(S25) == "Failing jobs: abc"

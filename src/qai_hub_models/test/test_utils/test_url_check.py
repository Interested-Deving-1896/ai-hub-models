# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from requests.adapters import HTTPAdapter

from qai_hub_models.utils.url_check import _make_url_check_session


def test_transient_server_errors_are_retried() -> None:
    """A throttled or briefly unavailable host must not read as a missing asset.

    S3 answered a HEAD burst with 503 and six models' banners were reported
    absent; 503/504 belong in the forcelist alongside 502.
    """
    adapter = _make_url_check_session().get_adapter("https://example.com")
    assert isinstance(adapter, HTTPAdapter)
    assert {502, 503, 504} <= set(adapter.max_retries.status_forcelist or [])

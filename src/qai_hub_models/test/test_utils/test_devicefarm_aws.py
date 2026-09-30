# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import datetime
from unittest import mock

import boto3
import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse, HTTPHeaders
from botocore.credentials import DeferredRefreshableCredentials

from qai_hub_models.utils.devicefarm.backends.aws import aws
from qai_hub_models.utils.devicefarm.backends.aws.aws import (
    AwsDeviceFarmConfig,
    _make_session,
)

OIDC_ENV = {
    "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example/?x=1",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "gh-token",
}
RUN_ARN = "arn:aws:devicefarm:us-west-2:000000000000:run:project/run"


def _creds(expiry: datetime.datetime, key: str) -> dict[str, str]:
    return {
        "access_key": key,
        "secret_key": "secret",
        "token": "session",
        "expiry_time": expiry.isoformat(),
    }


def test_make_session_without_oidc_uses_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in OIDC_ENV:
        monkeypatch.delenv(var, raising=False)
    config = AwsDeviceFarmConfig(project_arn="p", role_arn="arn:role")
    with mock.patch.object(boto3, "Session") as session_cls:
        _make_session(config)
    session_cls.assert_called_once_with(profile_name=None, region_name="us-west-2")


def test_make_session_with_oidc_refreshes_expired_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in OIDC_ENV.items():
        monkeypatch.setenv(var, value)
    now = datetime.datetime.now(datetime.timezone.utc)
    fetcher = mock.MagicMock()
    fetcher.return_value.fetch_credentials.side_effect = [
        _creds(now + datetime.timedelta(minutes=1), "first"),
        _creds(now + datetime.timedelta(hours=1), "second"),
    ]
    config = AwsDeviceFarmConfig(project_arn="p", role_arn="arn:role")
    with mock.patch.object(aws, "AssumeRoleWithWebIdentityCredentialFetcher", fetcher):
        session = _make_session(config)

    creds = session.get_credentials()
    assert isinstance(creds, DeferredRefreshableCredentials)
    assert fetcher.call_args.kwargs["role_arn"] == "arn:role"
    assert creds.get_frozen_credentials().access_key == "first"
    # The first credentials are inside the refresh window, so the next use re-assumes.
    assert creds.get_frozen_credentials().access_key == "second"


def test_long_lived_client_signs_each_call_with_refreshed_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in OIDC_ENV.items():
        monkeypatch.setenv(var, value)
    now = datetime.datetime.now(datetime.timezone.utc)
    fetcher = mock.MagicMock()
    fetcher.return_value.fetch_credentials.side_effect = [
        _creds(now + datetime.timedelta(minutes=1), "AKIAFIRST"),
        _creds(now + datetime.timedelta(hours=1), "AKIASECOND"),
    ]
    config = AwsDeviceFarmConfig(project_arn="p", role_arn="arn:role")
    with mock.patch.object(aws, "AssumeRoleWithWebIdentityCredentialFetcher", fetcher):
        client = _make_session(config).client("devicefarm", region_name="us-west-2")

    signed_keys: list[str] = []

    def _capture(request: AWSPreparedRequest, **kwargs: object) -> AWSResponse:
        auth = request.headers["Authorization"]
        auth = auth.decode() if isinstance(auth, bytes) else auth
        signed_keys.append(auth.split("Credential=")[1].split("/")[0])
        return AWSResponse(
            request.url, 200, HTTPHeaders(), mock.MagicMock(content=b"{}")
        )

    client.meta.events.register("before-send.devicefarm", _capture)
    client.get_run(arn=RUN_ARN)
    client.get_run(arn=RUN_ARN)
    assert signed_keys == ["AKIAFIRST", "AKIASECOND"]

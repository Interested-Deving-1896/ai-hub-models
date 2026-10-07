# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import datetime
from unittest import mock

import boto3
import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse, HTTPHeaders

from qai_hub_models_cli._internal import aws
from qai_hub_models_cli._internal.aws import (
    DEFAULT_SESSION_DURATION,
    MAX_SESSION_DURATION,
    MIN_SESSION_DURATION,
    QAIHM_AWS_PROFILE,
    QAIHM_AWS_ROLE_ARN_ENVVAR,
    _get_session_duration,
    qaihm_session,
)
from qai_hub_models_cli.envvars import AWS_SESSION_DURATION_ENVVAR

OIDC_ENV = {
    "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example/?x=1",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "gh-token",
}


def _creds(expiry: datetime.datetime, key: str) -> dict[str, str]:
    return {
        "access_key": key,
        "secret_key": "secret",
        "token": "session",
        "expiry_time": expiry.isoformat(),
    }


@pytest.mark.parametrize(
    ("envvar", "expected"),
    [
        (None, DEFAULT_SESSION_DURATION),
        ("7200", 7200),
        ("1000", MIN_SESSION_DURATION),
        ("50000", MAX_SESSION_DURATION),
        ("not-a-number", DEFAULT_SESSION_DURATION),
        ("", DEFAULT_SESSION_DURATION),
    ],
)
def test_get_session_duration(
    monkeypatch: pytest.MonkeyPatch, envvar: str | None, expected: int
) -> None:
    if envvar is None:
        monkeypatch.delenv(AWS_SESSION_DURATION_ENVVAR, raising=False)
    else:
        monkeypatch.setenv(AWS_SESSION_DURATION_ENVVAR, envvar)
    assert _get_session_duration() == expected


def test_qaihm_session_without_oidc_uses_static_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(QAIHM_AWS_ROLE_ARN_ENVVAR, "arn:aws:iam::1:role/ci")
    for var in OIDC_ENV:
        monkeypatch.delenv(var, raising=False)
    with mock.patch.object(boto3, "Session") as session_cls:
        qaihm_session()
    session_cls.assert_called_once_with(profile_name=QAIHM_AWS_PROFILE)


def test_cached_bucket_survives_credential_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Run 37102182278: get_bucket is cached for the whole job, so its credentials
    # must refresh in place rather than expire 12h in.
    monkeypatch.setenv(QAIHM_AWS_ROLE_ARN_ENVVAR, "arn:aws:iam::1:role/ci")
    for var, value in OIDC_ENV.items():
        monkeypatch.setenv(var, value)
    now = datetime.datetime.now(datetime.timezone.utc)
    fetcher = mock.MagicMock()
    fetcher.return_value.fetch_credentials.side_effect = [
        _creds(now + datetime.timedelta(minutes=1), "AKIAFIRST"),
        _creds(now + datetime.timedelta(hours=1), "AKIASECOND"),
    ]
    with mock.patch.object(aws, "AssumeRoleWithWebIdentityCredentialFetcher", fetcher):
        bucket = qaihm_session().resource("s3").Bucket("my-bucket")

    assert fetcher.call_args.kwargs["role_arn"] == "arn:aws:iam::1:role/ci"
    signed_keys: list[str] = []

    def _capture(request: AWSPreparedRequest, **kwargs: object) -> AWSResponse:
        auth = request.headers["Authorization"]
        auth = auth.decode() if isinstance(auth, bytes) else auth
        signed_keys.append(auth.split("Credential=")[1].split("/")[0])
        return AWSResponse(request.url, 200, HTTPHeaders(), mock.MagicMock())

    bucket.meta.client.meta.events.register("before-send.s3", _capture)  # type: ignore[arg-type]
    bucket.Object("a").load()
    bucket.Object("b").load()
    assert signed_keys == ["AKIAFIRST", "AKIASECOND"]

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from qai_hub_models.scripts import publish_release_assets
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG


def test_publish_latest_version_pointer_only_touches_latest_txt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pointer publish must not touch any key other than latest.txt.
    S3 IAM still allows versioned-key overwrites, so this catches a code bug before it clobbers assets.
    """
    monkeypatch.setattr(
        publish_release_assets, "attempt_with_s3_credentials_warning", lambda fn: fn()
    )
    bucket = MagicMock()
    bucket.name = "qaihub-public-assets"
    bucket.Object.return_value.get.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey"}}, "GetObject"
    )

    publish_release_assets.publish_latest_version_pointer(bucket, "v0.61.0")

    keys_touched = {call.args[0] for call in bucket.Object.call_args_list}
    assert keys_touched == {publish_release_assets.LATEST_VERSION_S3_KEY}


def test_release_asset_skips_write_when_target_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary code-side guard for release archives: never s3_copy over an existing versioned
    key unless --overwrite is explicitly set.
    """
    s3_copy_mock = MagicMock()
    monkeypatch.setattr(publish_release_assets, "s3_copy", s3_copy_mock)
    monkeypatch.setattr(publish_release_assets, "s3_file_exists", lambda *a, **k: True)
    monkeypatch.setattr(
        ASSET_CONFIG,
        "get_release_asset_s3_key",
        lambda **_: "qai-hub-models/releases/v0.61.0/foo",
    )
    sc_path = MagicMock()
    sc_path.is_published = True
    asset_details = MagicMock()
    asset_details.s3_key = "private/foo"
    asset_details.download_url = None

    publish_release_assets.release_asset(
        s3_private_bucket=MagicMock(),
        s3_public_bucket=MagicMock(),
        qaihm_version="v0.61.0",
        model_id="fake_model",
        precision=MagicMock(),
        chipset=None,
        sc_path=sc_path,
        asset_details=asset_details,
        replace_existing=False,
    )

    s3_copy_mock.assert_not_called()


def test_overwrite_flag_requires_envvar(monkeypatch: pytest.MonkeyPatch) -> None:
    """--overwrite must SystemExit before touching S3 when the envvar isn't set.
    Guarantees CI (which never sets the envvar) can't overwrite even if the flag leaks in.
    """
    monkeypatch.delenv(publish_release_assets.OVERWRITE_ASSETS_ENVVAR, raising=False)
    monkeypatch.setattr(
        publish_release_assets,
        "parse_args",
        lambda: argparse.Namespace(models=None, overwrite=True, version="v0.61.0"),
    )
    monkeypatch.setattr(
        publish_release_assets,
        "validate_and_split_enabled_models",
        lambda models: ({"fake_model"}, set()),
    )
    get_qaihm_s3_mock = MagicMock()
    monkeypatch.setattr(publish_release_assets, "get_qaihm_s3", get_qaihm_s3_mock)

    with pytest.raises(SystemExit):
        publish_release_assets.main()
    get_qaihm_s3_mock.assert_not_called()

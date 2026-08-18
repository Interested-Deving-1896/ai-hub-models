# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os
from pathlib import Path

from qai_hub_models import QAIRTVersion, TargetRuntime
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3,
    s3_download,
    s3_file_exists,
)

QAIRT_AUTO_SDK_S3_PREFIX = "qai-hub-models/qairt/"
QAIRT_AUTO_SDK_S3_ZIP_NAME = "artifact.zip"
QAIRT_AUTO_SDK_FILENAME = "qairt_auto_sdk.zip"


def qairt_auto_sdk_s3_key(version: QAIRTVersion) -> str:
    """S3 key of the auto SDK zip for the given QAIRT version.

    Auto SDKs are uploaded to "<api_version>-auto/artifact.zip", e.g.
    "2.45-auto/artifact.zip". Keyed on api_version rather than full_version
    because the latter's trailing build ident changes on every QAIRT rebuild,
    which would orphan an already-uploaded SDK.
    """
    folder = f"{version.api_version}-auto"
    return f"{QAIRT_AUTO_SDK_S3_PREFIX}{folder}/{QAIRT_AUTO_SDK_S3_ZIP_NAME}"


def download_qairt_auto_sdk(local_path: str) -> None:
    """Download the QAIRT SDK for automotive devices from S3.

    Auto devices have no internet access, so the SDK is bundled into the QDC
    artifact instead of being fetched on-device. The version tracks the GENIE
    runtime's default QAIRT version, so it stays in step with the rest of the
    scorecard. Per tetracode#20332 GenieX cannot yet select a QAIRT version, so
    there is deliberately no override here.
    """
    version = TargetRuntime.GENIE.default_qairt_version
    bucket, _ = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)
    key = qairt_auto_sdk_s3_key(version)
    if not s3_file_exists(bucket, key):
        raise ValueError(
            f"No QAIRT auto SDK for version {version.api_version} at "
            f"s3://{bucket.name}/{key}. Upload the auto SDK zip there to unblock "
            "auto devices, or point the default QAIRT version at a version that "
            "has one."
        )
    print(
        f"Downloading QAIRT {version.api_version} auto SDK from s3://{bucket.name}/{key}"
    )
    s3_download(bucket, key, local_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.environ.get("QAIRT_SDK_PATH")
        or str(Path.home() / QAIRT_AUTO_SDK_FILENAME),
        help="Local path to write the downloaded SDK zip. Defaults to "
        "$QAIRT_SDK_PATH if set, else $HOME/qairt_auto_sdk.zip.",
    )
    args = parser.parse_args()
    download_qairt_auto_sdk(args.output)


if __name__ == "__main__":
    main()

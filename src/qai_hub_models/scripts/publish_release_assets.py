# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import os
import sys
import threading

from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket
from packaging.version import Version

from qai_hub_models import Precision
from qai_hub_models._version import __version__
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.envvars import EnabledModelsEnvvar, SpecialModelSetting
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    QAIHM_PUBLIC_S3_BUCKET,
    attempt_with_s3_credentials_warning,
    get_qaihm_s3,
    s3_copy,
    s3_file_exists,
)
from qai_hub_models.utils.version_helpers import QAIHMVersion

LATEST_VERSION_S3_KEY = "qai-hub-models/releases/latest.txt"
OVERWRITE_ASSETS_ENVVAR = "QAIHM_ALLOW_ASSET_OVERWRITE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser, {SpecialModelSetting.PYTORCH})
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        default=False,
        help=(
            "Overwrite existing released assets. Requires "
            f"{OVERWRITE_ASSETS_ENVVAR}=1 in the environment plus admin creds; "
            "CI runs never set the envvar, so this flag is a no-op there."
        ),
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=__version__,
        help="AI Hub Models version to publish assets for.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pytorch_models, _ = validate_and_split_enabled_models(args.models)
    version = QAIHMVersion.tag_from_string(args.version)
    replace_existing_assets: bool = args.overwrite
    if replace_existing_assets and not os.environ.get(OVERWRITE_ASSETS_ENVVAR):
        sys.exit(
            f"--overwrite requires {OVERWRITE_ASSETS_ENVVAR}=1 in the environment "
            "to prevent accidental release-asset overwrites from CI."
        )

    private_s3 = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)[0]
    public_s3 = get_qaihm_s3(
        QAIHM_PUBLIC_S3_BUCKET, requires_admin=replace_existing_assets
    )[0]

    for model_id in sorted(pytorch_models):
        try:
            release_assets_for_model(
                private_s3, public_s3, version, model_id, replace_existing_assets
            )
        except Exception as e:  # noqa: PERF203
            print(f"Unable to upload results for {model_id}: {e}")

    publish_latest_version_pointer(public_s3, args.version)


def publish_latest_version_pointer(public_s3: Bucket, version: str) -> None:
    """Write releases/latest.txt so GenieX can discover the latest release without the CLI.

    Written last so the pointer only flips once every model's assets are uploaded.
    Skips if the pointer already names a newer release (avoids regressing on a
    patched older version).
    """
    new_version = Version(version.lstrip("v"))
    try:
        current = attempt_with_s3_credentials_warning(
            lambda: public_s3.Object(LATEST_VERSION_S3_KEY)
            .get()["Body"]
            .read()
            .decode()
            .strip()
        )
        if Version(current.lstrip("v")) > new_version:
            print(f"SKIPPED latest.txt: existing {current} is newer than {new_version}")
            return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "NoSuchKey":
            raise

    attempt_with_s3_credentials_warning(
        lambda: public_s3.Object(LATEST_VERSION_S3_KEY).put(
            Body=f"{new_version}\n".encode(),
            ACL="public-read",
            ContentType="text/plain",
        )
    )
    print(f"WROTE s3://{public_s3.name}/{LATEST_VERSION_S3_KEY} = {new_version}")


def release_asset(
    s3_private_bucket: Bucket,
    s3_public_bucket: Bucket,
    qaihm_version: str,
    model_id: str,
    precision: Precision,
    chipset: str | None,
    sc_path: ScorecardProfilePath,
    asset_details: QAIHMModelReleaseAssets.AssetDetails,
    replace_existing: bool = False,
) -> None:
    """Release a single asset to the public S3 bucket."""
    if not sc_path.is_published:
        return

    if asset_details.s3_key is None and asset_details.download_url is not None:
        # No publish step is required.
        return

    assert (
        asset_details.s3_key is not None
    )  # s3_key always present in release-assets.yaml

    s3_key = ASSET_CONFIG.get_release_asset_s3_key(
        version=qaihm_version,
        model_id=model_id,
        runtime=sc_path.runtime,
        precision=precision,
        chipset=chipset,
    )

    if not replace_existing and s3_file_exists(s3_public_bucket, s3_key):
        print(
            f"    SKIPPED: s3://{s3_private_bucket.name}/{asset_details.s3_key}; asset exists already at s3://{s3_public_bucket.name}/{s3_key}"
        )
    else:
        print(
            f"    COPYING: s3://{s3_private_bucket.name}/{asset_details.s3_key} to s3://{s3_public_bucket.name}/{s3_key}"
        )
        s3_copy(
            src_bucket=s3_private_bucket,
            src_key=asset_details.s3_key,
            dst_bucket=s3_public_bucket,
            dst_key=s3_key,
            make_dst_public=True,
        )


def release_assets_for_model(
    s3_private_bucket: Bucket,
    s3_public_bucket: Bucket,
    qaihm_version: str,
    model_id: str,
    replace_existing: bool = False,
) -> None:
    """Release all assets for a given model ID."""
    info = QAIHMModelManifest.from_model(model_id)
    if info.restrict_model_sharing:
        print(f"{model_id} SKIPPED; restrict_model_sharing is set in info.yaml\n")
        return

    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    if not assets.precisions:
        print(f"{model_id} SKIPPED; no release assets found\n")
        return

    print(f"{model_id}")
    copy_threads = []
    try:
        for precision, precision_details in assets.precisions.items():
            for chipset, chipset_details in precision_details.chipset_assets.items():
                for sc_path, asset_details in chipset_details.items():
                    thread = threading.Thread(
                        target=release_asset,
                        args=(
                            s3_private_bucket,
                            s3_public_bucket,
                            qaihm_version,
                            model_id,
                            precision,
                            chipset,
                            sc_path,
                            asset_details,
                            replace_existing,
                        ),
                    )
                    copy_threads.append(thread)
                    thread.start()

            for sc_path, asset_details in precision_details.universal_assets.items():
                thread = threading.Thread(
                    target=release_asset,
                    args=(
                        s3_private_bucket,
                        s3_public_bucket,
                        qaihm_version,
                        model_id,
                        precision,
                        None,
                        sc_path,
                        asset_details,
                        replace_existing,
                    ),
                )
                copy_threads.append(thread)
                thread.start()
    finally:
        for thread in copy_threads:
            thread.join()
    print()


if __name__ == "__main__":
    main()

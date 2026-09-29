# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Query and download historical scorecard artifacts from S3.

This script reads manifest.json files to discover available scorecard runs.
Each run's manifest lists the artifacts that were uploaded and metadata about
the run (date, branch, deployment, commit SHA, GitHub Actions URL). See
upload_scorecard_history.py for the manifest schema.

Usage:
    # List last 10 runs
    python -m qai_hub_models.scripts.download_scorecard_results list --last 10

    # Download all artifacts from a run
    python -m qai_hub_models.scripts.download_scorecard_results download --run-id 25594376004-weekly-prod

    # Download a specific artifact
    python -m qai_hub_models.scripts.download_scorecard_results download --run-id 25594376004-weekly-prod --artifact perf-regressions-2x.json

    # Show manifest for a run
    python -m qai_hub_models.scripts.download_scorecard_results show --run-id 25594376004-weekly-prod
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from prettytable import PrettyTable

from qai_hub_models.scorecard.history import ScorecardManifest
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3_or_exit,
    list_s3_files_in_folder_recursive,
    s3_download,
    s3_file_exists,
)

logger = logging.getLogger(__name__)

S3_PREFIX = "scorecard-history"


def list_runs(
    last: int = 10,
    since: str | None = None,
    until: str | None = None,
    run_name: str | None = None,
) -> list[ScorecardManifest]:
    """List scorecard runs by scanning S3 for manifest.json files.

    Returns list of ScorecardManifest objects, sorted by date descending.
    """
    bucket, _ = get_qaihm_s3_or_exit(QAIHM_PRIVATE_S3_BUCKET)
    objects = list_s3_files_in_folder_recursive(bucket, f"{S3_PREFIX}/")

    # Filter to only manifest.json files, sorted descending by key
    suffix = f"-{run_name}/manifest.json" if run_name else "/manifest.json"
    manifest_keys = sorted(
        (obj.key for obj in objects if obj.key.endswith(suffix)),
        reverse=True,
    )

    # Download only the manifests we might need (apply a generous upper bound
    # to avoid downloading the entire history when filters are loose).
    max_to_fetch = last if not since and not until else len(manifest_keys)
    manifests: list[ScorecardManifest] = []
    for key in manifest_keys[:max_to_fetch]:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            s3_download(bucket, key, tmp.name, verbose=False)
            manifests.append(ScorecardManifest.from_json(tmp.name))

    # Apply date filters on manifest content
    if since:
        manifests = [m for m in manifests if m.date >= since]
    if until:
        manifests = [m for m in manifests if m.date <= until]

    # Sort by date descending
    manifests.sort(key=lambda m: m.date, reverse=True)

    return manifests[:last]


def find_latest_run(
    deployment: str,
    exclude_run_id: str = "",
    scheduled_run_ids: list[str] | None = None,
) -> ScorecardManifest | None:
    """Return the most recent weekly-{deployment} manifest, or None.

    Used to source the "previous" baseline for the toolchain-version diff and
    the cross-deployment context table. Excludes exclude_run_id so a partial
    re-run of the same scorecard doesn't compare against itself.

    Filtering on the S3 key before download keeps nightly/manual runs from
    pushing the real weekly run out of the 25-manifest window.

    scheduled_run_ids (newest first, from GitHub's scheduled runs) is the
    authoritative list: nightly/manual uploads were historically also keyed
    weekly-prod, so the key alone can't tell them apart.
    """
    if scheduled_run_ids is not None:
        run_name = f"weekly-{deployment}"
        bucket, _ = get_qaihm_s3_or_exit(QAIHM_PRIVATE_S3_BUCKET)
        for run_id in scheduled_run_ids:
            if run_id == exclude_run_id:
                continue
            key = f"{S3_PREFIX}/{run_id}-{run_name}/manifest.json"
            if not s3_file_exists(bucket, key):
                continue
            with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
                s3_download(bucket, key, tmp.name, verbose=False)
                return ScorecardManifest.from_json(tmp.name)
        return None

    for manifest in list_runs(last=25, run_name=f"weekly-{deployment}"):
        if manifest.deployment != deployment:
            continue
        if exclude_run_id and manifest.run_id == exclude_run_id:
            continue
        return manifest
    return None


def download_single_artifact(
    manifest: ScorecardManifest, artifact: str, dest: Path
) -> Path | None:
    """Download one artifact from a specific run, or return None if missing.

    Takes the manifest (not just the run_id) because the S3 layout keys on
    {run_id}-{run_name} — see upload_scorecard_history.upload_scorecard_to_s3.
    Passing the manifest keeps the key construction in one place and prevents
    callers from silently missing artifacts when they only have the run_id.

    Returns the local path on success, None if either the run or the artifact
    is absent (so callers can fall back gracefully).
    """
    bucket, _ = get_qaihm_s3_or_exit(QAIHM_PRIVATE_S3_BUCKET)
    s3_key = f"{S3_PREFIX}/{manifest.run_id}-{manifest.run_name}/{artifact}"
    if not s3_file_exists(bucket, s3_key):
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3_download(bucket, s3_key, dest, verbose=False)
    return dest


def download_artifacts(
    run_id: str,
    artifact: str | None = None,
    output_dir: Path = Path("scorecard-history-downloads"),
) -> None:
    """Download artifact(s) from a specific scorecard run."""
    bucket, _ = get_qaihm_s3_or_exit(QAIHM_PRIVATE_S3_BUCKET)

    # Construct key directly — layout is deterministic: {PREFIX}/{run_id}/manifest.json
    s3_root = f"{S3_PREFIX}/{run_id}"
    manifest_key = f"{s3_root}/manifest.json"
    if not s3_file_exists(bucket, manifest_key):
        print(f"Error: Run {run_id} not found in S3.")
        sys.exit(1)

    # Read manifest to get artifact list
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        s3_download(bucket, manifest_key, tmp.name, verbose=False)
        manifest = ScorecardManifest.from_json(tmp.name)

    # Determine which files to download
    if artifact:
        if artifact not in manifest.artifacts:
            print(
                f"Error: Artifact '{artifact}' not found. Available: {manifest.artifacts}"
            )
            sys.exit(1)
        files_to_download = [artifact]
    else:
        files_to_download = [*manifest.artifacts, "manifest.json"]

    # Download
    run_output_dir = output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    for filename in files_to_download:
        s3_key = f"{s3_root}/{filename}"
        local_path = run_output_dir / filename
        s3_download(bucket, s3_key, local_path)

    print(f"Downloaded {len(files_to_download)} file(s) to {run_output_dir}/")


def show_manifest(run_id: str) -> ScorecardManifest:
    """Download and display the manifest for a given run."""
    bucket, _ = get_qaihm_s3_or_exit(QAIHM_PRIVATE_S3_BUCKET)

    # Construct key directly — layout is deterministic
    manifest_key = f"{S3_PREFIX}/{run_id}/manifest.json"
    if not s3_file_exists(bucket, manifest_key):
        print(f"Error: Run {run_id} not found in S3.")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        s3_download(bucket, manifest_key, tmp.name, verbose=False)
        manifest = ScorecardManifest.from_json(tmp.name)

    print(manifest.model_dump_json(indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query historical scorecard artifacts from S3."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List historical scorecard runs")
    list_parser.add_argument(
        "--last", type=int, default=10, help="Number of recent runs to show"
    )
    list_parser.add_argument("--since", type=str, help="Start date filter (YYYY-MM-DD)")
    list_parser.add_argument("--until", type=str, help="End date filter (YYYY-MM-DD)")

    # download
    dl_parser = subparsers.add_parser("download", help="Download artifacts from a run")
    dl_parser.add_argument("--run-id", type=str, required=True)
    dl_parser.add_argument("--artifact", type=str, help="Specific artifact to download")
    dl_parser.add_argument(
        "--output-dir", type=Path, default=Path("scorecard-history-downloads")
    )

    # show
    show_parser = subparsers.add_parser("show", help="Show manifest for a run")
    show_parser.add_argument("--run-id", type=str, required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "list":
        manifests = list_runs(last=args.last, since=args.since, until=args.until)
        if not manifests:
            print("No scorecard runs found.")
            return
        table = PrettyTable(["Date", "Run ID", "Deployment", "Branch", "Artifacts"])
        for m in manifests:
            table.add_row([m.date, m.run_id, m.deployment, m.branch, len(m.artifacts)])
        print(table)

    elif args.command == "download":
        download_artifacts(args.run_id, args.artifact, args.output_dir)

    elif args.command == "show":
        show_manifest(args.run_id)


if __name__ == "__main__":
    main()

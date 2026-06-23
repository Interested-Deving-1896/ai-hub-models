#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qai_hub import Client
from utils import (
    DEFAULT_MAX_WORKERS,
    JOB_STATUS_FAILED,
    JOB_STATUS_SUCCESS,
    extract_tag_and_dir_from_yaml,
    load_client,
    load_yaml_safe,
    log_and_print,
    save_yaml_results,
    setup_script_logging,
    wait_for_job_with_timeout,
)

logger = logging.getLogger(__name__)


def wait_for_jobs(
    client: Client, job_map: dict[str, dict], max_workers: int = DEFAULT_MAX_WORKERS
) -> dict[str, str]:
    logger.info(f"Waiting for {len(job_map)} jobs to complete")
    status_map = {}

    def wait_for_single_job(model_name: str, job_info: dict) -> tuple[str, str]:
        job = client.get_job(job_info["dev_job"])
        return model_name, wait_for_job_with_timeout(job, model_name)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(wait_for_single_job, model_name, job_info)
            for model_name, job_info in job_map.items()
        ]

        completed_count = 0
        total_jobs = len(job_map)
        for future in as_completed(futures):
            try:
                model_name, status_code = future.result()
                status_map[model_name] = status_code
                completed_count += 1
                log_and_print(
                    f"  [{completed_count}/{total_jobs}] {model_name}: {status_code}",
                    logger,
                )
            except Exception:
                logger.exception("  Error waiting for job")

    logger.info(f"All {len(status_map)} jobs completed")
    return status_map


def categorize_results(
    dev_status_map: dict[str, str], job_map: dict[str, dict]
) -> tuple[dict, dict, dict, dict, dict]:
    regressions: dict[str, dict] = {}
    progressions: dict[str, dict] = {}
    failures_both: dict[str, dict] = {}
    passed: dict[str, dict] = {}
    other_changes: dict[str, dict] = {}

    for model_name, job_info in job_map.items():
        if model_name not in dev_status_map:
            continue

        prod_status = job_info["prod_job_status"]
        dev_status = dev_status_map[model_name]

        entry = job_info.copy()
        entry["dev_status"] = dev_status

        if prod_status == JOB_STATUS_SUCCESS and dev_status == JOB_STATUS_SUCCESS:
            passed[model_name] = entry
        elif prod_status == JOB_STATUS_FAILED and dev_status == JOB_STATUS_FAILED:
            failures_both[model_name] = entry
        elif prod_status == JOB_STATUS_SUCCESS and dev_status == JOB_STATUS_FAILED:
            regressions[model_name] = entry
        elif prod_status == JOB_STATUS_FAILED and dev_status == JOB_STATUS_SUCCESS:
            progressions[model_name] = entry
        elif prod_status != dev_status:
            other_changes[model_name] = entry

    return regressions, progressions, failures_both, passed, other_changes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for compile jobs and collect results"
    )
    parser.add_argument(
        "--dev-profile",
        type=str,
        default="dev",
        help="Hub client profile for dev environment (default: dev)",
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        required=True,
        help="Path to dev-compile-jobs__<tag>.yaml file from run_compile_jobs.py",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers for waiting on jobs (default: {DEFAULT_MAX_WORKERS})",
    )

    args = parser.parse_args()

    tag, output_dir = extract_tag_and_dir_from_yaml(args.jobs_file)
    log_file = setup_script_logging(
        output_dir, "collect-compile-results", args.verbose, tag
    )
    log_and_print(f"Logging to {log_file}", logger)

    try:
        dev_client = load_client(args.dev_profile)
        job_map = load_yaml_safe(args.jobs_file)
        dev_status_map = wait_for_jobs(dev_client, job_map, args.workers)
        regressions, progressions, failures_both, passed, other_changes = (
            categorize_results(dev_status_map, job_map)
        )

        total_collected = len(dev_status_map)
        log_and_print(
            f"Collected {total_collected} jobs: "
            f"{len(passed)} passed, "
            f"{len(regressions)} regressions, "
            f"{len(progressions)} progressions, "
            f"{len(failures_both)} failures",
            logger,
        )

        results_to_save = [
            (regressions, "dev-regressions", "regressions"),
            (progressions, "dev-progressions", "progressions"),
            (failures_both, "failures-dev-and-prod", "failures"),
            (passed, "passed-dev-and-prod", "passed"),
            (other_changes, "other-status-changes", "other changes"),
        ]

        for data, filename_prefix, label in results_to_save:
            if data:
                path = output_dir / f"{filename_prefix}__{tag}.yaml"
                save_yaml_results(data, path)
                log_and_print(f"Saved {label}: {path}", logger)

        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

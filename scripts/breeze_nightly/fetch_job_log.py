# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Fetch a GitHub Actions job log, filter for error markers, print hits.

Written for the nightly-analyst Breeze agent Step 1.5. The Breeze allowlist
permits `Bash(python3:*)` but treats bash pipelines as split commands, so
collapsing fetch + filter into one Python call keeps the agent well under
its 50-turn cap.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_FILTER = (
    r"Traceback|ImportError|ModuleNotFoundError|FileNotFoundError|"
    r"##\[error\]|FAILED|Error:|failed\."
)
JOB_ID_RE = re.compile(r"^\d+$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="GitHub Actions job database id (numeric)")
    parser.add_argument(
        "--filter",
        default=DEFAULT_FILTER,
        help="Regex applied per line (default matches common error markers)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Print last N matching lines to stdout (0 = all)",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="OWNER/REPO for gh api (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--log-out",
        default=None,
        help="Path to write raw log (default /tmp/breeze-<job_id>.log)",
    )
    parser.add_argument(
        "--hits-out",
        default=None,
        help="Path to write all filtered hits (default /tmp/breeze-<job_id>-hits.txt)",
    )
    args = parser.parse_args()

    if not JOB_ID_RE.fullmatch(args.job_id):
        sys.exit(f"Invalid job_id {args.job_id!r} — must be a positive integer")
    if not args.repo:
        sys.exit("--repo not set and $GITHUB_REPOSITORY is empty")

    log_out = Path(args.log_out or f"/tmp/breeze-{args.job_id}.log")
    hits_out = Path(args.hits_out or f"/tmp/breeze-{args.job_id}-hits.txt")

    api_path = f"repos/{args.repo}/actions/jobs/{args.job_id}/logs"
    result = subprocess.run(
        ["gh", "api", api_path], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"gh api failed (rc={result.returncode}): {result.stderr.strip()}")

    log_out.write_text(result.stdout)

    pattern = re.compile(args.filter)
    hits = [line for line in result.stdout.splitlines() if pattern.search(line)]
    hits_out.write_text("\n".join(hits) + "\n")

    tail = hits[-args.tail :] if args.tail > 0 else hits
    print(f"raw log: {log_out} ({len(result.stdout)} bytes)")
    print(f"hits:    {hits_out} ({len(hits)} lines total, showing last {len(tail)})")
    print("---")
    for line in tail:
        print(line)


if __name__ == "__main__":
    main()

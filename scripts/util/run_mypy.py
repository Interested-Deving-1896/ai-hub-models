# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {"src": "qai_hub_models", "cli": "qai_hub_models_cli"}


def venv_python() -> str:
    venv = Path(
        os.environ.get("VENV_PATH") or os.environ.get("VIRTUAL_ENV") or "qaihm-dev"
    )
    if not venv.is_absolute():
        venv = REPO_ROOT / venv
    for candidate in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in PACKAGES:
        print("Usage: run_mypy.py <src|cli> [files...]", file=sys.stderr)
        return 1
    pkg_dir, files = sys.argv[1], sys.argv[2:]
    pkg_root = REPO_ROOT / pkg_dir

    cmd = [
        venv_python(),
        "-m",
        "mypy",
        "--warn-unused-configs",
        f"--config-file={pkg_root / 'pyproject.toml'}",
    ]
    if not files or len(files) > 100:
        cmd += ["-p", PACKAGES[pkg_dir]]
    else:
        # Paths are relative to pkg_root since we run from there. Generated protobuf
        # files are skipped so that mypy uses their .pyi stubs.
        prefix = f"{pkg_dir}/"
        files = [f.replace("\\", "/") for f in files]
        files = [f.removeprefix(prefix) for f in files if not f.endswith("_pb2.py")]
        if not files:
            return 0
        cmd += files

    return subprocess.call(cmd, cwd=pkg_root)


if __name__ == "__main__":
    sys.exit(main())

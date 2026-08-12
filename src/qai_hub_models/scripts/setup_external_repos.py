# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Setup external repos for models.

Usage:
    python -m qai_hub_models.scripts.setup_external_repos -a
    python -m qai_hub_models.scripts.setup_external_repos -m gkt track_anything

Useful for:
    - CI setup step (clone all external repos before tests)
    - Local development (pre-populate all repos for IDE support)
"""

from __future__ import annotations

import argparse
import sys

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.external_repo import setup_external_repos
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT

SHARED_DIR = QAIHM_MODELS_ROOT / "_shared"


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup external repos for models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        help="Models for which to setup external repos.",
    )
    group.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Setup external repos for all models and shared repos.",
    )
    args = parser.parse_args()

    failed = []

    if args.all and SHARED_DIR.exists():
        for shared_folder in sorted(SHARED_DIR.iterdir()):
            manifest_path = shared_folder / "manifest.yaml"
            if not manifest_path.exists():
                continue
            manifest = QAIHMModelManifest.from_yaml(manifest_path)
            if not manifest.external_repos:
                continue
            print(f"Setting up shared external repo: {shared_folder.name}...")
            try:
                setup_external_repos(shared_folder)
            except Exception as e:
                print(f"  FAILED: {e}")
                failed.append(f"_shared/{shared_folder.name}")

    models = args.models if args.models else MODEL_IDS
    for model_id in models:
        print(f"Setting up external repos for {model_id}...")
        try:
            setup_external_repos(QAIHM_MODELS_ROOT / model_id)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(model_id)

    if failed:
        print(f"\nFailed to set up external repos for: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

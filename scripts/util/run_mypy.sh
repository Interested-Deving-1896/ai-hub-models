#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# shellcheck source=/dev/null

REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"

set_strict_mode

cd "$(dirname "$0")/../.."

venv="${VENV_PATH:-${VIRTUAL_ENV:-qaihm-dev}}"
source "${venv}/bin/activate"

# First argument is the package directory (src or cli).
pkg_dir="$1"
shift

if [[ "$pkg_dir" == "src" ]]; then
  package="qai_hub_models"
elif [[ "$pkg_dir" == "cli" ]]; then
  package="qai_hub_models_cli"
else
  echo "Usage: run_mypy.sh <src|cli> [files...]" >&2
  exit 1
fi

cd "${REPO_ROOT}/${pkg_dir}"

if [[ "$#" -eq 0 || "$#" -gt 100 ]]; then
  mypy --warn-unused-configs --config-file="${REPO_ROOT}/${pkg_dir}/pyproject.toml" -p "${package}"
else
  # Strip the package prefix (src/ or cli/) since we cd into that directory.
  # Also filter out generated protobuf files (which enforces use of .pyi type stubs)
  files=()
  for f in "$@"; do
    f="${f#${pkg_dir}/}"
    [[ "$f" == *_pb2.py ]] && continue
    files+=("$f")
  done
  if [[ "${#files[@]}" -eq 0 ]]; then
    # Exit early (with status 0 / success) if there are no Python files to check.
    exit 0
  fi
  mypy --warn-unused-configs --config-file="${REPO_ROOT}/${pkg_dir}/pyproject.toml" "${files[@]}"
fi

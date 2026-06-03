# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"
. "${REPO_ROOT}/scripts/util/github.sh"
. "${REPO_ROOT}/scripts/ci/install-aws-cli.sh"

set_strict_mode

start_group "Ubuntu install: update repositories"
    APT_GPG_FILE=/etc/apt/trusted.gpg.d/apt.github-cli.gpg
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | run_as_root tee "$APT_GPG_FILE" > /dev/null

    APT_SOURCE_FILE=/etc/apt/sources.list.d/github-cli.list
    APT_ARCH=$(dpkg --print-architecture)
    echo "deb [arch=$APT_ARCH signed-by=$APT_GPG_FILE] https://cli.github.com/packages stable main" | run_as_root tee "$APT_SOURCE_FILE" > /dev/null

    run_as_root apt-get update
end_group

start_group "Ubuntu Install: install/upgrade packages"
    install_aws_cli
    run_as_root apt-get install -y gh python3-opencv cmake libportaudio2 ffmpeg libegl-dev
end_group

start_group "Ubuntu install: uv"
    if [ ! -x /usr/local/bin/uv ] || [ "$(uv --version)" != "uv 0.6.14" ]; then
        wget https://github.com/astral-sh/uv/releases/download/0.6.14/uv-installer.sh
        chmod +x uv-installer.sh
        run_as_root UV_UNMANAGED_INSTALL="/usr/local/bin" ./uv-installer.sh
        rm -rf uv-installer.sh
    else
        log_info "uv up to date."
    fi
end_group

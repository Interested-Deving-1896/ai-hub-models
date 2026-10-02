#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

PY_VERSION="${QAIHM_POD_PYTHON_VERSION:?must be set by the calling workflow step}"

echo "=== Installing system dependencies ==="
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  ca-certificates \
  curl \
  git \
  jq \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  lsb-release \
  python3 \
  python3-pip \
  python3-venv \
  software-properties-common \
  unzip \
  zip \
  ffmpeg

echo "=== Configuring git ==="
git config --global --add safe.directory "$GITHUB_WORKSPACE"

echo "=== Installing Python $PY_VERSION ==="
$SUDO add-apt-repository -y ppa:deadsnakes/ppa
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "python$PY_VERSION" "python$PY_VERSION-venv" "python$PY_VERSION-dev"
$SUDO update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python$PY_VERSION" 1
$SUDO ln -sf "/usr/bin/python$PY_VERSION" /usr/bin/python

echo "=== Installing AWS CLI ==="
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
$SUDO /tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

echo "=== Installing uv 0.6.14 ==="
curl -fsSL https://github.com/astral-sh/uv/releases/download/0.6.14/uv-installer.sh | $SUDO env UV_UNMANAGED_INSTALL="/usr/local/bin" bash

echo "=== Pod setup complete ==="

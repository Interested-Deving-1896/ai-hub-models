# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models_cli.envvars import USE_INTERNAL_RELEASES_ENVVAR, bool_envvar_value


def use_internal_releases() -> bool:
    """
    Check if the internal (private) S3 release should be used instead of the public release.
    Returns True if the QAIHM_CLI_USE_INTERNAL_RELEASES env var is truthy.
    """
    return bool_envvar_value(USE_INTERNAL_RELEASES_ENVVAR)

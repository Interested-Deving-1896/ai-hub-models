# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Help text for ``qai-hub-models <script>`` invoked without a target.

Kept out of :mod:`qai_hub_models.cli.dispatch` on purpose. Printing help must
not pay for the export/evaluate pipelines dispatch imports -- scipy, sympy,
sklearn and friends, roughly 1.4s -- when all it needs are four argument
parsers.
"""

from __future__ import annotations

from typing import TextIO

from qai_hub_models.cli.generate_files import build_parser as generate_files_parser
from qai_hub_models.cli.install import build_parser as install_parser
from qai_hub_models.cli.upload_to_hf import build_parser as upload_to_hf_parser
from qai_hub_models.cli.validate import build_parser as validate_parser

# The recipe commands whose flags do not depend on the target, and their
# parsers. Membership is the sole declaration of that split: every other recipe
# command builds its parser from the resolved recipe, so the lean CLI needs no
# list of its own.
_STATIC_PARSERS = {
    "install": install_parser,
    "generate-files": generate_files_parser,
    "validate": validate_parser,
    "upload-to-hf": upload_to_hf_parser,
}


def print_command_help(script: str, stream: TextIO) -> bool:
    """Print *script*'s flag list, if it has one that needs no target.

    Parameters
    ----------
    script
        The recipe command to describe.
    stream
        Where to write the help text.

    Returns
    -------
    bool
        False if *script* builds its parser from the resolved recipe, in which
        case nothing is written and the caller should explain that instead.
    """
    build_parser = _STATIC_PARSERS.get(script)
    if build_parser is None:
        return False
    build_parser().print_help(file=stream)
    return True

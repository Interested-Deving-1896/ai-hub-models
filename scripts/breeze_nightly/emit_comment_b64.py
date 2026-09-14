# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Emit a Breeze analyst comment base64-encoded with recovery markers.

Used by the nightly-analyze and scorecard-analyze Breeze agents in their Step 5.
The Breeze allowlist permits `Bash(python3:*)` but not `echo` / `base64` /
compound shell, so a single-command Python emit is the only reliable path. The
downstream poster job recovers the file from the agent job's log with
`extract_comment_from_log.py` and posts it via `gh issue comment`.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

BEGIN = "===BREEZE_COMMENT_B64_BEGIN==="
END = "===BREEZE_COMMENT_B64_END==="


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <comment.md>", file=sys.stderr)
        sys.exit(1)
    data = Path(sys.argv[1]).read_bytes()
    print(BEGIN)
    print(base64.b64encode(data).decode())
    print(END)


if __name__ == "__main__":
    main()

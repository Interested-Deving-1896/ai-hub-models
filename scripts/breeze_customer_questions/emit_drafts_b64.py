# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Emit drafts.json base64-encoded with recovery markers.

Written for the customer-questions Breeze agent Step 4. The Breeze
allowlist permits `Bash(python3:*)` but not `echo` / `base64` / compound
shell, so a single-command Python emit is the only reliable path.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

BEGIN = "===DRAFTS_B64_BEGIN==="
END = "===DRAFTS_B64_END==="


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <drafts.json>", file=sys.stderr)
        sys.exit(1)
    data = Path(sys.argv[1]).read_bytes()
    print(BEGIN)
    print(base64.b64encode(data).decode())
    print(END)


if __name__ == "__main__":
    main()

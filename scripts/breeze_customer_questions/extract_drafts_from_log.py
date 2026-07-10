# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Extract a base64-encoded drafts.json from a Breeze agent job log.

The Claude Code SDK captures the agent's Bash tool stdout into structured
JSON tool results (a JSON line containing `"stdout": "..."`), NOT raw lines
in the GitHub Actions log. This script searches for a tool result whose
stdout contains our marker pair, JSON-unescapes the inner string,
extracts the base64 between markers, and writes the decoded bytes.

Usage:
    extract_drafts_from_log.py <log_path> <out_path>

Exit codes:
    0  success — drafts.json written
    1  marker pair not found in any tool stdout (drafting probably failed)
    2  base64 decode failed (corrupted markers or truncated stdout)
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

BEGIN = "===DRAFTS_B64_BEGIN==="
END = "===DRAFTS_B64_END==="

# Match the JSON-encoded stdout field that contains our marker pair. The SDK
# escapes newlines as \n and quotes as \", so the regex matches across the
# escaped form (single line in the log file).
STDOUT_RE = re.compile(
    r'"stdout"\s*:\s*"((?:\\.|[^"\\])*'
    + re.escape(BEGIN)
    + r'(?:\\.|[^"\\])*'
    + re.escape(END)
    + r'(?:\\.|[^"\\])*)"',
    re.DOTALL,
)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <log_path> <out_path>", file=sys.stderr)
        sys.exit(1)
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    log_text = log_path.read_text(errors="replace")
    matches = list(STDOUT_RE.finditer(log_text))
    if not matches:
        print(
            f"ERROR: no tool stdout containing {BEGIN!r} found in {log_path}.\n"
            "The agent may have failed before emitting drafts, or the SDK "
            "log format changed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Take last match: earlier matches may be the agent echoing its own script.
    raw = matches[-1].group(1).encode("utf-8").decode("unicode_escape")

    # Strip everything outside the markers; what's left is the base64 dump
    # `base64` produced (with embedded newlines from the standard wrap).
    body = raw.split(BEGIN, 1)[1].split(END, 1)[0]
    # base64 module ignores whitespace by default with validate=False.
    try:
        decoded = base64.b64decode(body, validate=False)
    except Exception as e:
        print(f"ERROR: base64 decode failed: {e}", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(decoded)
    print(f"Wrote {len(decoded)} bytes to {out_path}")


if __name__ == "__main__":
    main()

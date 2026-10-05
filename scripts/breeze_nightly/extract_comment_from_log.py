# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Extract a base64-encoded Breeze analyst comment from an agent job log.

The Claude Code SDK captures the agent's Bash tool stdout into structured
JSON tool results (a JSON line containing `"stdout": "..."`), NOT raw lines
in the GitHub Actions log. This script searches for a tool result whose
stdout contains our marker pair, JSON-unescapes the inner string, extracts
the base64 between markers, and writes the decoded bytes.

Usage:
    extract_comment_from_log.py <log_path> <out_path> [tag]

Exit codes:
    0  success — comment file written
    1  marker pair not found in any tool stdout (agent probably failed
       before emitting)
    2  base64 decode failed (corrupted markers or truncated stdout)
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

BEGIN = "===BREEZE_COMMENT_B64_BEGIN==="
END = "===BREEZE_COMMENT_B64_END==="


def _markers(tag: str | None) -> tuple[str, str]:
    suffix = f":{tag}" if tag else ""
    return f"{BEGIN[:-3]}{suffix}===", f"{END[:-3]}{suffix}==="


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(f"usage: {sys.argv[0]} <log_path> <out_path> [tag]", file=sys.stderr)
        sys.exit(1)
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    begin, end = _markers(sys.argv[3] if len(sys.argv) == 4 else None)
    stdout_re = re.compile(
        r'"stdout"\s*:\s*"((?:\\.|[^"\\])*'
        + re.escape(begin)
        + r'(?:\\.|[^"\\])*'
        + re.escape(end)
        + r'(?:\\.|[^"\\])*)"',
        re.DOTALL,
    )

    log_text = log_path.read_text(errors="replace")
    matches = list(stdout_re.finditer(log_text))
    if not matches:
        print(
            f"ERROR: no tool stdout containing {begin!r} found in {log_path}.\n"
            "The agent may have failed before emitting the comment, or the "
            "SDK log format changed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Take last match: earlier matches may be the agent echoing its own script.
    raw = matches[-1].group(1).encode("utf-8").decode("unicode_escape")
    body = raw.split(begin, 1)[1].split(end, 1)[0]
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

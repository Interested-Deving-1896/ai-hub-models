#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# SessionStart hook.
#
# Docs and other hooks reference ${TMPDIR:-/tmp}/claude as the sanctioned
# temp dir. The committed .claude/settings.json only grants Read/Edit on
# //tmp/claude/**, which covers Linux (where $TMPDIR is usually unset). If
# $TMPDIR is set to something else (macOS default, or a per-user override on
# Linux), the resolved path is NOT covered and every temp-file access will
# trigger a permission prompt.
#
# Output strategy: emit JSON on stdout with BOTH `additionalContext` (loaded
# into Claude's context so it can surface it) AND `systemMessage` (rendered
# directly in the transcript by Claude Code versions that support it). This
# maximizes the odds the user actually sees the warning at session start.
set -e

TMP_ROOT="${TMPDIR:-/tmp}"
TMP_ROOT="${TMP_ROOT%/}"
RESOLVED="$TMP_ROOT/claude"

# Fast path: on Linux with $TMPDIR unset, the resolved path is /tmp/claude,
# which the committed settings.json already covers. Nothing to warn about.
if [ "$RESOLVED" = "/tmp/claude" ]; then
    exit 0
fi

# Check every settings file that could grant it. In a git worktree, cwd has a
# checked-out settings.json but no settings.local.json -- it's gitignored, so
# `git worktree add` never copies it and the grant lives in the main tree.
MAIN_ROOT="$(dirname "$(git rev-parse --git-common-dir 2>/dev/null || echo .git)")"
USER_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
SETTINGS_FILES=(
    .claude/settings.json
    .claude/settings.local.json
    "$MAIN_ROOT/.claude/settings.json"
    "$MAIN_ROOT/.claude/settings.local.json"
    "$USER_DIR/settings.json"
)

covered=0
for f in "${SETTINGS_FILES[@]}"; do
    [ -f "$f" ] || continue
    if grep -qE "Read\\(//${RESOLVED#/}(/\\*\\*)?\\)" "$f"; then
        covered=1
        break
    fi
done

if [ "$covered" -eq 1 ]; then
    exit 0
fi

# JSON-encode the two message strings.
# Uses python3 to avoid escaping headaches with jq / bash quoting.
python3 - "$TMP_ROOT" "$RESOLVED" "${SETTINGS_FILES[@]}" <<'PY'
import json, os, sys
tmp_root, resolved = sys.argv[1], sys.argv[2]
_paths = dict.fromkeys(os.path.normpath(p) for p in sys.argv[3:])
checked = "\n".join(f"      {p}" for p in _paths)
resolved_no_slash = resolved.lstrip("/")

user_msg = (
    f"⚠️  $TMPDIR is set to '{tmp_root}', so this project's sanctioned temp "
    f"dir resolves to '{resolved}'.\n"
    f"No settings file grants Read on it, so every temp-file access will "
    f"trigger a permission prompt. Checked:\n"
    f"{checked}\n"
    f"Add these two lines to the \"allow\" list in "
    f".claude/settings.local.json:\n"
    f'    "Read(//{resolved_no_slash}/**)",\n'
    f'    "Edit(//{resolved_no_slash}/**)"'
)

claude_ctx = (
    "SessionStart notice — TELL THE USER THIS AS YOUR VERY FIRST MESSAGE, "
    "before running any tools:\n\n"
    f"Their $TMPDIR is '{tmp_root}', so the project's sanctioned temp dir "
    f"resolves to '{resolved}'. None of these settings files grant Read on "
    f"it, so every temp-file access will trigger a permission prompt:\n"
    f"{checked}\n\n"
    "Offer to add these two lines to the \"allow\" list in "
    ".claude/settings.local.json (wait for confirmation before editing):\n\n"
    f'    "Read(//{resolved_no_slash}/**)",\n'
    f'    "Edit(//{resolved_no_slash}/**)"'
)

print(json.dumps({
    "systemMessage": user_msg,
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": claude_ctx,
    },
}))
PY
exit 0

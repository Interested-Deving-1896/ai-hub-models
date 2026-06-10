#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# PreToolUse hook for Bash commands.
#
# Blocks (exit 2) commands that violate the rules in CLAUDE.md, with stderr
# explaining why. The harness shows stderr to the agent so it can adjust.
#
# Reads JSON from stdin: {"tool_input": {"command": "..."}, ...}
#
set -e

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# Empty command (defensive) — let it through; the tool will error on its own.
[ -z "$CMD" ] && exit 0

# ----------------------------------------------------------------------
# Rule 1: forbid `python3 -c "..."` with quoted code.
# CLAUDE.md "Inline Python" rule: write to /tmp/claude/<name>.py instead.
# Detection: `python` or `python3`, then `-c`, then a quote char.
# ----------------------------------------------------------------------
if echo "$CMD" | grep -qE 'python3?[[:space:]]+-c[[:space:]]+["'"'"']'; then
    cat >&2 <<EOF
[hook] BLOCKED: python3 -c with quoted code.

The "Inline Python" rule in CLAUDE.md forbids this — the bash permission
matcher chokes on quotes and triggers repeated permission prompts.

Instead, write the script to /tmp/claude/<name>.py and run it:
    Write(file_path="/tmp/claude/check_env.py", content="...")
    Bash(command="python3 /tmp/claude/check_env.py")
EOF
    exit 2
fi

# ----------------------------------------------------------------------
# Rule 2: forbid filesystem-wide scans whose top-level path is /, /afs, or
# /mnt/share. Subdirectories of those roots (e.g. /afs/foo) are allowed —
# only the top level is dangerous because that's what would walk every
# subtree under the mount.
# ----------------------------------------------------------------------
FIRST_TOK=$(echo "$CMD" | awk '{print $1}')

case "$FIRST_TOK" in
    find|bfs)
        # First non-flag argument after the command — the path being searched.
        PATH_ARG=$(echo "$CMD" | awk '{for(i=2;i<=NF;i++) if(substr($i,1,1)!="-") {print $i; exit}}')
        case "$PATH_ARG" in
            /|/afs|/afs/|/mnt/share|/mnt/share/)
                cat >&2 <<EOF
[hook] BLOCKED: $FIRST_TOK against '$PATH_ARG' would traverse the entire root or a shared filesystem.

These walks can take many minutes, generate heavy load on shared file servers,
and may be terminated by infrastructure admins.

Scope the search to a specific subtree:
    $FIRST_TOK . -name <pattern>
    $FIRST_TOK /tmp/claude -name <pattern>
    $FIRST_TOK /afs/<specific-cell> -name <pattern>
EOF
                exit 2
                ;;
        esac
        ;;
esac

# Recursive grep / ripgrep against /, /afs, or /mnt/share at the top level.
if echo "$CMD" | grep -qE '^(grep[[:space:]]+-[rR]+[A-Za-z]*|rg)[[:space:]].*[[:space:]](/|/afs|/mnt/share)/?([[:space:]]|$)'; then
    cat >&2 <<EOF
[hook] BLOCKED: recursive grep/rg against /, /afs, or /mnt/share top level.

Scope to a specific subtree:
    grep -r <pattern> .
    rg <pattern> /tmp/claude
    grep -r <pattern> /afs/<specific-cell>
EOF
    exit 2
fi

# ----------------------------------------------------------------------
# Rule 3: forbid command chaining (&&, ||, ;, |) outside quotes.
# The Bash permission matcher splits on these operators and rechecks each
# side independently, so a permitted command can fail when chained even
# though it would have been allowed standalone. Redirects (>, >>) don't
# trigger splitting and are fine.
# ----------------------------------------------------------------------
# Strip single-quoted and double-quoted substrings before checking, so
# operators inside string literals don't trip the rule.
STRIPPED=$(echo "$CMD" | sed -E "s/'[^']*'//g; s/\"[^\"]*\"//g")

if echo "$STRIPPED" | grep -qE '(&&|\|\||;|[[:space:]]\|[[:space:]])'; then
    cat >&2 <<EOF
[hook] BLOCKED: command chaining (&&, ||, ;, |) breaks permission checks.

Pick one:
  - Run each command as a separate Bash call (preferred for 2-3 commands)
  - For sequences that genuinely need pipes/chaining, write the pipeline to
    /tmp/claude/<name>.sh and execute that file:
        Write(file_path="/tmp/claude/foo.sh", content="#!/bin/bash\\nset -e\\n...")
        Bash(command="bash /tmp/claude/foo.sh")
  - For piping to python, write the script to /tmp/claude/<name>.py
  - For filtering, use the source command's own flags (e.g. \`gh api ... --jq '<filter>'\`)
EOF
    exit 2
fi

# ----------------------------------------------------------------------
# Rule 4: forbid /tmp/ writes outside /tmp/claude/.
# CLAUDE.md "Temporary files" rule: always use /tmp/claude/.
# ----------------------------------------------------------------------
# Match: redirects to /tmp/X (where X != claude), mkdir /tmp/X, touch /tmp/X.
# We allow /tmp/claude/... and /tmp/claude (without trailing /).
# Find any reference to /tmp/X in a write context (>, >>, mkdir, touch),
# extract the path token, and check it doesn't match /tmp/claude{,/...}.
TMP_VIOLATION=$(echo "$STRIPPED" | grep -oE '(>{1,2}[[:space:]]*|mkdir[^|]*[[:space:]]|touch[^|]*[[:space:]])/tmp/[^[:space:]]*' | grep -oE '/tmp/[^[:space:]]*' | grep -vE '^/tmp/claude(/|$)' | head -1)
if [ -n "$TMP_VIOLATION" ]; then
    cat >&2 <<EOF
[hook] BLOCKED: writing to /tmp/ outside /tmp/claude/.

Use /tmp/claude/ for temporary files so they're scoped to this session.
EOF
    exit 2
fi

exit 0

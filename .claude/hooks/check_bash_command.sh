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
# Resolve ${VAR}, ${VAR:-default}, and ${VAR:?...} expansions in-place so
# the permission matcher sees the literal path (e.g. /tmp/claude/foo.diff)
# instead of the expansion (e.g. ${TMPDIR:-/tmp}/claude/foo.diff). Without
# this, an otherwise-allowlisted command like `wc -l "${TMPDIR:-/tmp}/..."`
# triggers a prompt because the matcher does not run shell expansion.
#
# Pure textual substitution — no shell exec, no $(...) or backticks.
# Values come from the hook process's own env (inherited from Claude Code),
# which matches the env the Bash tool would evaluate the command in.
# ----------------------------------------------------------------------
resolve_shell_vars() {
    local cmd="$1"
    local prev=""
    local iter=0
    while [ "$cmd" != "$prev" ] && [ $iter -lt 50 ]; do
        prev="$cmd"
        if [[ "$cmd" =~ (\$\{([A-Za-z_][A-Za-z0-9_]*):-([^{}]*)\}) ]]; then
            local full="${BASH_REMATCH[1]}"
            local name="${BASH_REMATCH[2]}"
            local dflt="${BASH_REMATCH[3]}"
            local val="${!name:-}"
            [ -z "$val" ] && val="$dflt"
            cmd="${cmd//"$full"/$val}"
        elif [[ "$cmd" =~ (\$\{([A-Za-z_][A-Za-z0-9_]*):\?[^{}]*\}) ]]; then
            local full="${BASH_REMATCH[1]}"
            local name="${BASH_REMATCH[2]}"
            local val="${!name:-}"
            [ -z "$val" ] && break
            cmd="${cmd//"$full"/$val}"
        elif [[ "$cmd" =~ (\$\{([A-Za-z_][A-Za-z0-9_]*)\}) ]]; then
            local full="${BASH_REMATCH[1]}"
            local name="${BASH_REMATCH[2]}"
            local val="${!name:-}"
            cmd="${cmd//"$full"/$val}"
        else
            break
        fi
        iter=$((iter+1))
    done
    printf '%s' "$cmd"
}

RESOLVED=$(resolve_shell_vars "$CMD")
CMD="$RESOLVED"

# ----------------------------------------------------------------------
# Rule 1: forbid `python3 -c "..."` with quoted code.
# CLAUDE.md "Inline Python" rule: write to ${TMPDIR:-/tmp}/claude/<name>.py instead.
# Detection: `python` or `python3`, then `-c`, then a quote char.
# ----------------------------------------------------------------------
if echo "$CMD" | grep -qE 'python3?[[:space:]]+-c[[:space:]]+["'"'"']'; then
    TMPD="${TMPDIR:-/tmp}/claude"
    cat >&2 <<EOF
[hook] BLOCKED: python3 -c with quoted code.

The "Inline Python" rule in CLAUDE.md forbids this — the bash permission
matcher chokes on quotes and triggers repeated permission prompts.

Instead, write the script to $TMPD/<name>.py and run it:
    Write(file_path="$TMPD/check_env.py", content="...")
    Bash(command="python3 $TMPD/check_env.py")
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
                TMPD="${TMPDIR:-/tmp}/claude"
                cat >&2 <<EOF
[hook] BLOCKED: $FIRST_TOK against '$PATH_ARG' would traverse the entire root or a shared filesystem.

These walks can take many minutes, generate heavy load on shared file servers,
and may be terminated by infrastructure admins.

Scope the search to a specific subtree:
    $FIRST_TOK . -name <pattern>
    $FIRST_TOK $TMPD -name <pattern>
    $FIRST_TOK /afs/<specific-cell> -name <pattern>
EOF
                exit 2
                ;;
        esac
        ;;
esac

# Recursive grep / ripgrep against /, /afs, or /mnt/share at the top level.
if echo "$CMD" | grep -qE '^(grep[[:space:]]+-[rR]+[A-Za-z]*|rg)[[:space:]].*[[:space:]](/|/afs|/mnt/share)/?([[:space:]]|$)'; then
    TMPD="${TMPDIR:-/tmp}/claude"
    cat >&2 <<EOF
[hook] BLOCKED: recursive grep/rg against /, /afs, or /mnt/share top level.

Scope to a specific subtree:
    grep -r <pattern> .
    rg <pattern> $TMPD
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
    TMPD="${TMPDIR:-/tmp}/claude"
    cat >&2 <<EOF
[hook] BLOCKED: command chaining (&&, ||, ;, |) breaks permission checks.

Pick one:
  - Run each command as a separate Bash call (preferred for 2-3 commands)
  - For sequences that genuinely need pipes/chaining, write the pipeline to
    $TMPD/<name>.sh and execute that file:
        Write(file_path="$TMPD/foo.sh", content="#!/bin/bash\\nset -e\\n...")
        Bash(command="bash $TMPD/foo.sh")
  - For piping to python, write the script to $TMPD/<name>.py
  - For filtering, use the source command's own flags (e.g. \`gh api ... --jq '<filter>'\`)
EOF
    exit 2
fi

# ----------------------------------------------------------------------
# Rule 4: forbid system-temp writes outside ${TMPDIR:-/tmp}/claude/.
# CLAUDE.md "Temporary files" rule: always use ${TMPDIR:-/tmp}/claude/.
# ----------------------------------------------------------------------
# Match: redirects to <tmp>/X (X != claude), mkdir <tmp>/X, touch <tmp>/X,
# where <tmp> is /tmp or the current $TMPDIR value.
TMP_ROOT="${TMPDIR:-/tmp}"
TMP_ROOT="${TMP_ROOT%/}"
# Escape regex special characters in the resolved root so grep -E treats it literally.
TMP_ROOT_RE=$(printf '%s' "$TMP_ROOT" | sed 's/[.[\*^$()+?{|\\]/\\&/g')
# Check both /tmp (the hardcoded fallback) and the resolved TMPDIR, in case
# they differ (e.g. macOS with TMPDIR=/var/folders/...).
if [ "$TMP_ROOT" = "/tmp" ]; then
    ROOTS_RE="/tmp"
else
    ROOTS_RE="(/tmp|$TMP_ROOT_RE)"
fi
TMP_VIOLATION=$(echo "$STRIPPED" | grep -oE "(>{1,2}[[:space:]]*|mkdir[^|]*[[:space:]]|touch[^|]*[[:space:]])$ROOTS_RE/[^[:space:]]*" | grep -oE "$ROOTS_RE/[^[:space:]]*" | grep -vE "^$ROOTS_RE/claude(/|\$)" | head -1)
if [ -n "$TMP_VIOLATION" ]; then
    cat >&2 <<EOF
[hook] BLOCKED: writing to system-temp outside ${TMP_ROOT}/claude/.

Use ${TMP_ROOT}/claude/ for temporary files so they're scoped to this session.
EOF
    exit 2
fi

# ----------------------------------------------------------------------
# Env-var prefixes: the matcher treats `VAR=1 cmd ...` as its own command
# string, so an allowlisted `cmd` rule does not cover it. Strip assignment
# prefixes into a scratch copy; if what remains is an allowlisted base
# command, decide "allow" here. The command itself runs unmodified.
#
# Only these names may be stripped. Never PATH/LD_*/PYTHONPATH/BASH_ENV/IFS
# — those change which binary runs, so stripping them would turn an
# allowlisted base command into an arbitrary-code allow.
# ----------------------------------------------------------------------
STRIPPABLE_VAR='^(QAIHM_[A-Z0-9_]+|RUN_SLOW_TESTS|SKIP)=[^[:space:]"'"'"'`$;&|]*$'

ALLOWED_BASE="qai-hub-models qai-hub python python3 pytest pre-commit"

strip_env_prefixes() {
    local cmd="$1"
    local tok
    while :; do
        cmd="${cmd#"${cmd%%[![:space:]]*}"}"   # drop leading whitespace
        tok="${cmd%% *}"
        [ "$tok" = "$cmd" ] && break            # single token left
        case "$tok" in
            *=*)
                if echo "$tok" | grep -qE "$STRIPPABLE_VAR"; then
                    cmd="${cmd#* }"
                else
                    break
                fi
                ;;
            env)
                cmd="${cmd#* }"                 # `env VAR=1 cmd` form
                ;;
            *) break ;;
        esac
    done
    printf '%s' "$cmd"
}

STRIPPED=$(strip_env_prefixes "$CMD")
DECIDE_ALLOW=0
BASE_CMD=""
if [ "$STRIPPED" != "$CMD" ]; then
    BASE_CMD="${STRIPPED%% *}"
    for allowed in $ALLOWED_BASE; do
        if [ "$BASE_CMD" = "$allowed" ]; then
            DECIDE_ALLOW=1
            break
        fi
    done
fi

# Emit one combined hookSpecificOutput: the resolved command (so the matcher
# sees literal paths instead of ${VAR} expansions) and/or the allow decision.
# Silent no-op when there is nothing to say — keeps the transcript clean.
ORIG_CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""')
FILTER='{hookSpecificOutput: ({hookEventName: "PreToolUse"}'
ARGS=()
if [ "$DECIDE_ALLOW" = "1" ]; then
    FILTER="$FILTER + {permissionDecision: \"allow\", permissionDecisionReason: \$reason}"
    ARGS+=(--arg reason "env-var prefix stripped; base command '$BASE_CMD' is allowlisted")
fi
if [ "$RESOLVED" != "$ORIG_CMD" ]; then
    FILTER="$FILTER + {updatedInput: {command: \$cmd}}"
    ARGS+=(--arg cmd "$RESOLVED")
fi
FILTER="$FILTER)}"

if [ ${#ARGS[@]} -gt 0 ]; then
    jq -n "${ARGS[@]}" "$FILTER"
fi

exit 0

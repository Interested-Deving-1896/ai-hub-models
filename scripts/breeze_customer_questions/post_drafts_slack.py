# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Post a single reminder message listing unanswered customer questions.

Reads the classified `drafts.json` (see customer-questions-remind SKILL.md
for schema), filters to items still needing follow-up, and posts ONE
parent message to the target Slack channel — a header line plus a
bulleted list of threads. For each reminder that has a `draft_answer`, an
additional threaded reply is posted under the parent message carrying the
draft + KB citation.

Channel routing:
    --test-mode true  -> #tungsten-debug (debug)
    --test-mode false -> #ai-hub-github-notify (production)
Override with --channel <id> if needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


REMIND_STATUSES = {"no-reply", "ack-only"}

# Slack message hard cap is 40k chars; leave headroom. If the bullet list
# overruns this, we truncate with a pointer to the workflow artifact.
MAX_MESSAGE_CHARS = 3500

DEFAULT_PROD_CHANNEL = "C09GG81GW05"  # #ai-hub-github-notify
DEFAULT_TEST_CHANNEL = "C09MKAM6AGP"  # #tungsten-debug

SLACK_API = "https://slack.com/api"


def _slack_post(token: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to Slack Web API and return the parsed JSON response."""
    req = urllib.request.Request(
        f"{SLACK_API}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Slack {method} transport error: {e}") from e
    if not data.get("ok"):
        raise RuntimeError(
            f"Slack {method} failed: {data.get('error', 'unknown')} (response={data})"
        )
    return data


def _has_draft(item: dict[str, Any]) -> bool:
    """Whether a reminder item carries a non-empty KB-grounded draft."""
    draft = item.get("draft_answer")
    return isinstance(draft, str) and bool(draft.strip())


def _bullet_line(item: dict[str, Any], position: int | None = None) -> str:
    """Render one reminder bullet — concise, no question text.

    *position* is the 1-based index of this reminder; it's printed at the
    end so the matching draft reply (when present) can reference it. The
    customer's verbatim question lives in the linked thread; the agent's
    `topic` field gives 1-2 words of context to scan past.

    If *position* is None, the bullet is for an informational item (e.g.
    answered threads) and won't be numbered.
    """
    status = str(item.get("status", "")).lower()
    age_h = item.get("age_hours") or 0
    try:
        age_h = int(age_h)
    except (TypeError, ValueError):
        age_h = 0
    age_str = f"{age_h}h" if age_h < 48 else f"{age_h // 24}d"

    if status == "no-reply":
        status_tag = "🆕 no reply"
    elif status == "ack-only":
        status_tag = "🔁 ack only"
    elif status == "answered":
        status_tag = "✅ answered"
    else:
        status_tag = f"[{status}]"

    draft_tag = " 📝 draft" if _has_draft(item) else ""

    channel = item.get("channel_or_repo") or "?"
    submitted_by = item.get("submitted_by") or "?"
    thread_url = item.get("thread_url") or ""
    topic = (item.get("topic") or "").strip()
    topic_tag = f" _{topic}_" if topic else ""

    link = f"<{thread_url}|view thread>" if thread_url else "(no link)"

    pos_prefix = f"[#{position}] " if position is not None else ""
    return (
        f"• {pos_prefix}[{age_str}] [{status_tag}]{draft_tag} "
        f"`{channel}` — *{submitted_by}*:{topic_tag} {link}"
    )


def _draft_reply_text(item: dict[str, Any], position: int) -> str:
    """Render a threaded-reply body carrying the KB-grounded draft."""
    draft = (item.get("draft_answer") or "").strip()
    kb = item.get("kb_citation") or ""
    confidence = item.get("confidence") or ""
    submitted_by = item.get("submitted_by") or "?"
    parts = [
        f"*📝 Draft for #{position} — {submitted_by}*",
        "",
        draft,
        "",
    ]
    meta_bits = []
    if kb:
        meta_bits.append(f"KB: `{kb}`")
    if confidence:
        meta_bits.append(f"confidence: {confidence}")
    if meta_bits:
        parts.append("_" + " · ".join(meta_bits) + "_")
    parts.append("_Review and edit before replying to the customer._")
    return "\n".join(parts)


def _message_text(
    reminders: list[dict[str, Any]],
    answered: list[dict[str, Any]],
    totals: dict[str, int],
    run_url: str,
    test_mode: bool,
) -> str:
    test_prefix = "[TEST] " if test_mode else ""
    n = len(reminders)
    n_drafts = sum(1 for r in reminders if _has_draft(r))
    header_lines = [
        f"*{test_prefix}Customer-questions reminder — {n} thread"
        f"{'s' if n != 1 else ''} need follow-up*",
    ]
    if run_url:
        header_lines.append(f"Run: {run_url}")
    header_lines.append(
        f"🆕 {totals.get('no-reply', 0)} no reply  "
        f"🔁 {totals.get('ack-only', 0)} ack only  "
        f"✅ {totals.get('answered', 0)} answered"
    )
    if n_drafts:
        header_lines.append(
            f"📝 {n_drafts} draft{'s' if n_drafts != 1 else ''} "
            "— KB-grounded starters in thread below, review before replying"
        )

    sections = []
    sections.append("\n".join(header_lines))

    if reminders:
        sections.append("**Need follow-up:**")
        bullets = [_bullet_line(item, i + 1) for i, item in enumerate(reminders)]
        sections.append("\n".join(bullets))

    if answered:
        sections.append("**Already answered (no action needed):**")
        answered_bullets = [_bullet_line(item, position=None) for item in answered]
        sections.append("\n".join(answered_bullets))

    text = "\n\n".join(sections)

    if len(text) > MAX_MESSAGE_CHARS:
        keep = MAX_MESSAGE_CHARS - 200
        text = (
            text[:keep] + "\n\n_… reminder list truncated for Slack. See the "
            "`reminders.json` artifact on the workflow run for the full list._"
        )
    return text


def _tally(items: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        s = str(item.get("status", "")).lower()
        out[s] = out.get(s, 0) + 1
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument(
        "--run-url",
        default="",
        help="GitHub Actions run URL (linked from the header).",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help=(
            "Slack channel ID to post to. If unset, derives from --test-mode "
            f"(prod={DEFAULT_PROD_CHANNEL}, test={DEFAULT_TEST_CHANNEL})."
        ),
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Route to #tungsten-debug and prefix header with [TEST].",
    )
    parser.add_argument(
        "--max-drafts",
        type=int,
        default=50,
        help="Hard cap on bullets. Default: 50",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message body; do not call Slack.",
    )
    args = parser.parse_args()

    if not args.drafts.exists():
        logger.error("reminders file not found: %s", args.drafts)
        sys.exit(1)

    payload = json.loads(args.drafts.read_text())
    items = payload.get("drafts", [])
    if not isinstance(items, list):
        logger.error("input JSON missing 'drafts' list")
        sys.exit(1)

    totals = _tally([d for d in items if isinstance(d, dict)])
    reminders = [
        d
        for d in items
        if isinstance(d, dict) and str(d.get("status", "")).lower() in REMIND_STATUSES
    ]
    answered = [
        d
        for d in items
        if isinstance(d, dict) and str(d.get("status", "")).lower() == "answered"
    ]

    if not reminders:
        logger.info("No reminder-worthy threads (counts=%s). Skipping post.", totals)
        return

    if len(reminders) > args.max_drafts:
        logger.warning(
            "Truncating from %d reminders to max-drafts=%d",
            len(reminders),
            args.max_drafts,
        )
        reminders = reminders[: args.max_drafts]

    channel = args.channel or (
        DEFAULT_TEST_CHANNEL if args.test_mode else DEFAULT_PROD_CHANNEL
    )
    text = _message_text(
        reminders=reminders,
        answered=answered,
        totals=totals,
        run_url=args.run_url,
        test_mode=args.test_mode,
    )

    if args.dry_run:
        print(f"=== CHANNEL === {channel}")
        print("=== PARENT MESSAGE ===")
        print(text)
        for i, item in enumerate(reminders, 1):
            if _has_draft(item):
                print(f"\n=== THREADED REPLY #{i} ===")
                print(_draft_reply_text(item, i))
        return

    token = os.environ.get("SLACK_NOTIFIER_TOKEN", "")
    if not token:
        logger.error("SLACK_NOTIFIER_TOKEN env var not set; cannot post.")
        sys.exit(1)

    resp = _slack_post(
        token,
        "chat.postMessage",
        {"channel": channel, "text": text, "unfurl_links": False},
    )
    ts = resp.get("ts", "")
    n_drafts_posted = 0
    for i, item in enumerate(reminders, 1):
        if not _has_draft(item):
            continue
        try:
            _slack_post(
                token,
                "chat.postMessage",
                {
                    "channel": channel,
                    "text": _draft_reply_text(item, i),
                    "thread_ts": ts,
                    "unfurl_links": False,
                },
            )
            n_drafts_posted += 1
        except RuntimeError as e:
            logger.warning("Draft #%d post failed: %s", i, e)
    logger.info(
        "Posted reminder. channel=%s ts=%s reminders=%d drafts=%d test_mode=%s",
        channel,
        ts,
        len(reminders),
        n_drafts_posted,
        args.test_mode,
    )

    try:
        perma = _slack_post(
            token,
            "chat.getPermalink",
            {"channel": channel, "message_ts": ts},
        )
        print(perma.get("permalink", ""))
    except RuntimeError:
        print(f"slack://channel?id={channel}&message={ts}")


if __name__ == "__main__":
    main()

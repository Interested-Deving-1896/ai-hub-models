# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import shutil
from typing import Any

from transformers import TextStreamer


class IndentedTextStreamer(TextStreamer):
    def __init__(self, line_start: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal_width = shutil.get_terminal_size().columns
        self.printed_width = 0
        self.line_start = line_start

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if len(text) == 0:
            return

        # If the incoming text would cause the printed output to wrap around, start a new line
        if self.printed_width + len(text) >= self.terminal_width:
            print(flush=True)
            self.printed_width = 0

        # If we are on a new line, print the line starter before the text
        if self.printed_width == 0:
            text = self.line_start + text

        # If there are multiple newlines, make sure that the line starter is present at every new line
        # (except the last one, since that will be taken care of when we try to print the something to that new line
        # for the first time)
        if text.count("\n") > 1:
            last_index = text.rfind("\n")
            before_last = text[:last_index]
            after_last = text[last_index:]
            modified_before_last = before_last.replace("\n", "\n" + self.line_start)
            text = modified_before_last + after_last

        print(text, flush=True, end="" if not stream_end else None)

        # Update the counter of characters on this line
        if text.endswith("\n"):
            self.printed_width = 0
        else:
            self.printed_width += len(text)

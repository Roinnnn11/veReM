import re
from typing import List

from .base import Span

_SENT_DELIMITERS = re.compile(r"(?<=[.!?])\s+|(?<=\n)")


def split_sentences(text: str) -> List[Span]:
    """Split text into sentence-level spans by newline or sentence boundary."""
    spans = []
    # Prefer newline-based splitting first
    lines = text.split("\n")
    offset = 0
    span_id = 0
    for line in lines:
        stripped = line.strip()
        if stripped:
            start = text.index(line, offset)
            end = start + len(line)
            spans.append(Span(
                span_id=span_id,
                start=start,
                end=end,
                text=line,
                span_type="sentence",
            ))
            span_id += 1
        offset += len(line) + 1  # +1 for the \n

    # If only one line, fall back to sentence splitting
    if len(spans) <= 1 and text.strip():
        spans = []
        span_id = 0
        for m in re.finditer(r"[^.!?\n]+[.!?\n]?", text):
            seg = m.group()
            if seg.strip():
                spans.append(Span(
                    span_id=span_id,
                    start=m.start(),
                    end=m.end(),
                    text=seg,
                    span_type="sentence",
                ))
                span_id += 1

    return spans

import re
from typing import List

from .base import Span
from .sentence_splitter import split_sentences


def is_candidate_span(span_text: str) -> bool:
    has_number = bool(re.search(r"\d", span_text))
    has_operator = any(op in span_text for op in ["=", "+", "-", "*", "/", "%"])
    is_too_short = len(span_text.strip()) < 5
    is_final_answer = "####" in span_text
    return (has_number or has_operator) and not is_too_short and not is_final_answer


def math_span_score(span_text: str) -> float:
    """Higher score = more likely to be a critical math span."""
    score = 0.0
    score += len(re.findall(r"\d+", span_text)) * 1.0
    score += len(re.findall(r"[=+\-*/]", span_text)) * 0.5
    if re.search(r"\d+\s*[=+\-*/]\s*\d+", span_text):
        score += 2.0
    return score


def get_math_candidate_spans(text: str) -> List[Span]:
    """Return candidate spans sorted by math relevance (highest first)."""
    all_spans = split_sentences(text)
    candidates = [s for s in all_spans if is_candidate_span(s.text)]
    candidates.sort(key=lambda s: math_span_score(s.text), reverse=True)
    return candidates


def mask_span(text: str, span: Span, mask_token: str = "[MASK]") -> str:
    """Replace the span in text with a single mask token."""
    return text[: span.start] + mask_token + text[span.end :]

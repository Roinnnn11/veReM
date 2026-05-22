import re
from typing import List

from .base import Span
from .sentence_splitter import split_sentences

MASK_TOKEN = "<|mdm_mask|>"


def is_candidate_span(span_text: str) -> bool:
    has_number = bool(re.search(r"\d", span_text))
    has_operator = any(op in span_text for op in ["=", "+", "-", "*", "/", "%"])
    is_too_short = len(span_text.strip()) < 5
    return (has_number or has_operator) and not is_too_short


def math_span_score(span_text: str) -> float:
    """Higher score = more likely to be a critical math span."""
    # Answer line always gets highest priority so verifier can see a changed number
    if "####" in span_text:
        return 1000.0
    score = 0.0
    score += len(re.findall(r"\d+", span_text)) * 1.0
    score += len(re.findall(r"[=+\-*/]", span_text)) * 0.5
    if re.search(r"\d+\s*[=+\-*/]\s*\d+", span_text):
        score += 2.0
    return score


def get_suffix_span(text: str, n_lines: int = 2) -> "Span | None":
    """Return a span covering the last n_lines non-empty lines of text.

    This is the primary remasking target for VeReM: masking the suffix
    (final reasoning step + answer line) forces the model to regenerate
    the conclusion rather than restoring the same wrong answer.
    """
    from .base import Span
    lines = text.split("\n")
    # Collect non-empty line indices from the end
    tail_indices = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            tail_indices.append(i)
        if len(tail_indices) == n_lines:
            break
    if not tail_indices:
        return None
    first_idx = min(tail_indices)
    # Character offset of the first line in the suffix
    start = sum(len(l) + 1 for l in lines[:first_idx])
    end = len(text)
    suffix_text = text[start:]
    return Span(span_id=-1, start=start, end=end, text=suffix_text, span_type="suffix")


def get_answer_span(text: str) -> "Span | None":
    """Return a span for the final answer region.

    Priority 1: the number after '####' (narrow span, just the digits).
    Priority 2: the entire '####' line (when #### exists but has no number yet).
    Priority 3: the last \\boxed{N} number.
    Returns None if no answer marker found.
    """
    from .base import Span

    # Priority 1 & 2: #### marker
    hash_m = list(re.finditer(r"####", text))
    if hash_m:
        last_hash = hash_m[-1]
        tail = text[last_hash.end():]
        num_m = re.search(r"(-?\d[\d,]*\.?\d*)", tail)
        if num_m:
            # Narrow span: just the number
            abs_start = last_hash.end() + num_m.start(1)
            abs_end = last_hash.end() + num_m.end(1)
            all_spans = split_sentences(text)
            span_id = next((s.span_id for s in all_spans if s.start <= abs_start < s.end), 0)
            return Span(span_id=span_id, start=abs_start, end=abs_end,
                        text=num_m.group(1), span_type="answer")
        else:
            # #### exists but no number — remask the whole #### line
            all_spans = split_sentences(text)
            for s in all_spans:
                if "####" in s.text:
                    return Span(span_id=s.span_id, start=s.start, end=s.end,
                                text=s.text, span_type="answer")

    # Priority 3: \boxed{N}
    box_matches = list(re.finditer(r"\\boxed\{(-?\d[\d,]*\.?\d*)\}", text))
    if box_matches:
        m = box_matches[-1]
        all_spans = split_sentences(text)
        span_id = next((s.span_id for s in all_spans if s.start <= m.start(1) < s.end), 0)
        return Span(span_id=span_id, start=m.start(1), end=m.end(1),
                    text=m.group(1), span_type="answer")

    return None


def get_math_candidate_spans(text: str, suffix_lines: int = 2) -> List[Span]:
    """Return candidate spans sorted by math relevance (highest first).

    The suffix span (last N lines) is always first — masking the conclusion
    forces the model to regenerate the final reasoning step and answer,
    breaking the context lock that causes the same wrong answer to be restored.
    Fallback intermediate spans follow in case suffix remasking doesn't help.
    """
    all_spans = split_sentences(text)
    candidates = [s for s in all_spans if is_candidate_span(s.text)]
    candidates.sort(key=lambda s: math_span_score(s.text), reverse=True)

    # Remove spans that overlap with the suffix region
    suffix_span = get_suffix_span(text, n_lines=suffix_lines)
    if suffix_span is not None:
        candidates = [s for s in candidates if s.end <= suffix_span.start]
        candidates.insert(0, suffix_span)

    return candidates


def mask_span(text: str, span: Span, mask_token: str = MASK_TOKEN, tokenizer=None) -> str:
    """
    Replace the span in text with mask token(s).
    If tokenizer is provided, expands to one mask per span token so the dLLM
    infills the same number of positions. Otherwise uses a single placeholder.
    """
    if tokenizer is not None:
        num_tokens = len(tokenizer(span.text, add_special_tokens=False).input_ids)
        placeholder = mask_token * num_tokens
    else:
        placeholder = mask_token
    return text[: span.start] + placeholder + text[span.end :]

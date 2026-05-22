import re


def extract_answer(text: str) -> str | None:
    # Priority 1: GSM8K "#### <number>" format — use the LAST occurrence,
    # since the model sometimes emits #### at the start before reasoning.
    # Also allow short prose between #### and the number (e.g. "#### The answer is 42")
    # but require the number within 60 chars to avoid grabbing unrelated numbers.
    for m in reversed(list(re.finditer(r"####", text))):
        tail = text[m.end(): m.end() + 80]
        nm = re.search(r"(-?\d[\d,]*\.?\d*)", tail)
        if nm:
            return _normalize(nm.group(1))

    # Priority 2: \boxed{42} (LaTeX) — use the last occurrence
    matches = list(re.finditer(r"\\boxed\{(-?\d[\d,]*\.?\d*)\}", text))
    if matches:
        return _normalize(matches[-1].group(1))

    # Priority 3: "the answer is X" / "answer: X"
    m = re.search(r"(?:the answer is|answer:)\s*(-?\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))

    # Priority 4: last number in text (weakest signal)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        return _normalize(nums[-1])

    return None


def _normalize(s: str) -> str:
    s = s.replace(",", "").strip()
    # Remove trailing dot
    s = s.rstrip(".")
    # Normalize float: "3.0" -> "3"
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def is_correct(pred_text: str, gold_text: str) -> bool:
    pred = extract_answer(pred_text)
    gold = extract_answer(gold_text)
    if pred is None or gold is None:
        return False
    return pred == gold

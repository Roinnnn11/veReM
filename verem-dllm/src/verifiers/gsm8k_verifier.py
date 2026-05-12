import re


def extract_answer(text: str) -> str | None:
    # Priority 1: GSM8K "#### 42" format
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return _normalize(m.group(1))

    # Priority 2: "The answer is X" pattern
    m = re.search(r"(?:the answer is|answer:|=)\s*([\-\d,\.]+)", text, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))

    # Priority 3: last number in text
    nums = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
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

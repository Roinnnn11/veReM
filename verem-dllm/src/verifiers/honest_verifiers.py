"""
Honest verifiers for GSM8K reranking.

Distinct from `gsm8k_verifier.is_correct`, which compares to gold and is
therefore an *oracle*. These verifiers operate on the candidate alone and
return a real-valued score (higher = more likely correct).
"""

import re
from typing import Callable

from .gsm8k_verifier import extract_answer


# ---------- (1) Format / well-formedness verifier ----------

def format_score(text: str) -> float:
    """1.0 if output has the canonical '#### <number>' format, else softer fallback.

    This is the weakest honest verifier: it only checks that the model
    followed the expected output format, not that the answer is correct.
    """
    if re.search(r"####\s*-?\d", text):
        return 1.0
    if extract_answer(text) is not None:
        return 0.5
    return 0.0


# ---------- (2) Arithmetic-consistency verifier ----------

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_OP_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def arithmetic_consistency_score(text: str) -> float:
    """Fraction of inline equations 'a OP b = c' that are arithmetically correct.

    Returns 1.0 when there are no equations (no penalty for short/format-only
    outputs); we combine with format_score downstream so format-only gets
    score < 1.

    This is honest: we never look at gold. We only check that the model's own
    arithmetic is internally consistent.
    """
    eqs = _OP_RE.findall(text)
    if not eqs:
        return 1.0  # nothing to check
    correct = 0
    for a, op, b, c in eqs:
        try:
            a, b, c = _to_float(a), _to_float(b), _to_float(c)
            if op in ("*", "x", "×"):
                pred = a * b
            elif op in ("/", "÷"):
                pred = a / b if b != 0 else None
            elif op == "+":
                pred = a + b
            elif op == "-":
                pred = a - b
            else:
                continue
            if pred is not None and abs(pred - c) < 1e-3:
                correct += 1
        except Exception:
            continue
    return correct / len(eqs)


# ---------- (3) Combined honest score ----------

def honest_score(text: str) -> float:
    """Weighted combination of format and arithmetic consistency."""
    f = format_score(text)
    a = arithmetic_consistency_score(text)
    return 0.4 * f + 0.6 * a


# ---------- (4) Length-normalised log-probability (placeholder) ----------
# Computing real LM logprob requires an extra forward pass per candidate;
# we expose the API but compute it lazily in the decoder when needed.


def make_logprob_verifier(model) -> Callable[[str, str], float]:
    """Build a verifier that uses the model's own log-prob of the candidate
    given the prompt. Higher = more likely under the model.

    Note: this needs a forward pass per candidate, so it is *not* free.
    """
    import torch

    def _score(prompt: str, text: str) -> float:
        chat = model._build_chat_input(prompt)
        prompt_ids = model.tokenizer(chat, return_tensors="pt").input_ids.to(model.model.device)
        cand_ids = model.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.model.device)
        full = torch.cat([prompt_ids, cand_ids], dim=1)
        with torch.no_grad():
            logits = model.model(full).logits  # (1, L, V)
        # Score the candidate tokens given the (clean, no-mask) context — this
        # is a coarse approximation since LLaDA was not trained for this.
        lp = torch.log_softmax(logits.float(), dim=-1)
        plen = prompt_ids.shape[1]
        target = cand_ids[0]
        gathered = lp[0, plen - 1 : plen - 1 + target.shape[0], :].gather(
            -1, target.unsqueeze(-1)
        ).squeeze(-1)
        return float(gathered.mean().item())  # length-normalised

    return _score

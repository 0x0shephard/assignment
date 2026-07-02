"""GSM8K loading + answer extraction.

Gold answers in `openai/gsm8k` (main config) come at the end of the `answer`
field after "####". Model outputs may follow different conventions:
    - "#### 42"
    - "The answer is 42"
    - "The answer is \\boxed{42}"
    - Trailing number at end of last line.

Extractor returns None if nothing looks like a number (treated as wrong).
"""
from __future__ import annotations

import re
from typing import Optional

from datasets import load_dataset


# Order matters: try structured hints first, fall back to trailing-number.
_PATTERNS = [
    re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)"),                              # gold format
    re.compile(r"\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}"),                    # \boxed{42}
    re.compile(r"[Tt]he answer is\s*[:=]?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)"),    # "The answer is 42"
    re.compile(r"[Aa]nswer\s*[:=]\s*\$?\s*(-?[\d,]+(?:\.\d+)?)"),            # "Answer: 42"
]
_TRAILING_NUM = re.compile(r"(-?[\d,]+(?:\.\d+)?)\s*[.)!]?\s*$")


def _to_number(s: str) -> Optional[float]:
    s = s.replace(",", "").strip()
    if not s:
        return None
    try:
        v = float(s)
        return v
    except ValueError:
        return None


def extract_answer(text: str) -> Optional[float]:
    if not text:
        return None
    for pat in _PATTERNS:
        # take LAST match — models sometimes rehearse the problem
        matches = pat.findall(text)
        if matches:
            v = _to_number(matches[-1])
            if v is not None:
                return v
    # trailing number on the last non-empty line
    last_line = next((ln for ln in reversed(text.strip().splitlines()) if ln.strip()), "")
    m = _TRAILING_NUM.search(last_line)
    if m:
        v = _to_number(m.group(1))
        if v is not None:
            return v
    return None


def extract_gold(gsm8k_answer_field: str) -> Optional[float]:
    """The gold answer always uses `#### <num>`. We enforce that."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", gsm8k_answer_field)
    return _to_number(m.group(1)) if m else None


def answers_equal(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------- dataset -------------------

def load_gsm8k(split: str = "train", limit: Optional[int] = None):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for ex in ds:
        gold = extract_gold(ex["answer"])
        if gold is None:
            continue
        items.append({"question": ex["question"], "answer_text": ex["answer"], "gold": gold})
        if limit and len(items) >= limit:
            break
    return items


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end, write your final answer as a single number.\n\n"
    "Problem: {question}\n"
    "Solution:"
)


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


# ------------------- verifier sanity -------------------

def _verify_extractor():
    """Manual §4.10 C6.1 step 4: run extract on 20 golds and 20 wrongs."""
    items = load_gsm8k("train", limit=20)
    ok = 0
    for it in items:
        v = extract_answer(it["answer_text"])
        ok += int(answers_equal(v, it["gold"]))
    wrongs = ["I have no idea", "the cat sat on the mat", "",
              "There are seventeen apples", "banana"]
    wrong_ok = 0
    for w in wrongs:
        v = extract_answer(w)
        wrong_ok += int(not answers_equal(v, 42.0))   # arbitrary wrong gold
    return ok, len(items), wrong_ok, len(wrongs)


if __name__ == "__main__":
    ok, n, wok, wn = _verify_extractor()
    print(f"gold extractor: {ok}/{n}")
    print(f"wrong-string extractor: {wok}/{wn}")

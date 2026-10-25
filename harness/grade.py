"""Graders for the KV-cache test. Deterministic; no model in the loop. Written before any model was run.

The answer is the last 'ANSWER: ...' line of the visible reply (never the reasoning trace).
"""
import re
from decimal import Decimal, InvalidOperation

ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _clean(s):
    s = s.strip()
    s = re.sub(r"^[\s*_`\"'\[(<]+", "", s)
    s = re.sub(r"[\s*_`\"'\])>.]+$", "", s)
    return s.strip()


def extract_answer(content):
    found = ANSWER_RE.findall(content or "")
    return _clean(found[-1]) if found else None


def _norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()


def _first_number(s):
    m = NUM_RE.search(s or "")
    if not m:
        return None
    try:
        return Decimal(m.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def grade(question, content):
    """Returns (correct: bool, detail: dict). detail['partial'] is the share of needles right (R only)."""
    got = extract_answer(content)
    detail = {"extracted": got}
    if got is None:
        if question["answer_type"] == "list":
            detail["partial"] = 0.0
        return False, detail
    kind, key = question["answer_type"], question["answer"]
    if kind == "list":
        parts = [re.sub(r"[^A-Z0-9-]", "", p) for p in re.split(r"[,;\s]+", got.upper())]
        parts = [p for p in parts if p and p != "AND"]
        want = [k.upper() for k in key]
        detail["partial"] = sum(1 for i, k in enumerate(want) if i < len(parts) and parts[i] == k) / len(want)
        return parts == want, detail
    if kind == "choice":
        g = " %s " % _norm_text(got)
        k = " %s " % _norm_text(key)
        if g.strip() == k.strip():
            return True, detail
        # lenient containment: the key appears as whole words and no other option does
        others = [o for o in question["meta"]["options"] if o != key]
        hit = k in g and not any(" %s " % _norm_text(o) in g for o in others)
        detail["containment"] = hit
        return hit, detail
    if kind in ("int", "num"):
        n = _first_number(got)
        detail["number"] = None if n is None else str(n)
        return n is not None and n == Decimal(str(key)), detail
    raise ValueError(kind)


def selftest():
    q_list = {"answer_type": "list", "answer": ["K7-QX4", "B2-AA9", "M3-ZZ2"], "meta": {}}
    assert grade(q_list, "...\nANSWER: K7-QX4, B2-AA9, M3-ZZ2")[0]
    assert grade(q_list, "ANSWER: k7-qx4; b2-aa9 and m3-zz2.")[0]
    ok, d = grade(q_list, "ANSWER: K7-QX4, M3-ZZ2, B2-AA9")
    assert not ok and abs(d["partial"] - 1 / 3) < 1e-9
    assert not grade(q_list, "no answer line")[0]
    q_choice = {"answer_type": "choice", "answer": "Cobalt Hall", "meta": {"options": ["Cobalt Hall", "Beacon House"]}}
    assert grade(q_choice, "ANSWER: Cobalt Hall")[0]
    assert grade(q_choice, "ANSWER: **cobalt hall**")[0]
    assert grade(q_choice, "ANSWER: the lead is based in Cobalt Hall")[0]
    assert not grade(q_choice, "ANSWER: Cobalt Hall or Beacon House")[0]
    assert not grade(q_choice, "ANSWER: Beacon House")[0]
    q_int = {"answer_type": "int", "answer": 2036, "meta": {}}
    assert grade(q_int, "ANSWER: 2,036")[0]
    assert grade(q_int, "ANSWER: EUR 2036")[0]
    assert not grade(q_int, "ANSWER: 2035")[0]
    q_num = {"answer_type": "num", "answer": "0.50", "meta": {}}
    assert grade(q_num, "ANSWER: 0.5%")[0]
    assert not grade(q_num, "ANSWER: 5")[0]
    reasoning_only = {"answer_type": "int", "answer": 4, "meta": {}}
    assert not grade(reasoning_only, "")[0]
    return "grade selftest OK"


if __name__ == "__main__":
    print(selftest())

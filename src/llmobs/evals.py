"""Online evaluators — cheap, deterministic quality checks run on sampled generation spans.

Each evaluator takes a span dict and returns {name, score (0..1), passed, detail}. They are
heuristic and key-free so observability works offline; an LLM-as-judge evaluator can be
added with the same signature for production-grade faithfulness scoring.
"""
from __future__ import annotations

import re

_REFUSAL = ("i can't", "i cannot", "i'm unable", "i am unable", "as an ai")
_TOXIC = ("idiot", "stupid", "hate you", "shut up", "moron")
_STOP = set("the a an of to and or in on for is are was were be this that with your you i "
            "it as at by an be can will do does what how why when".split())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def completeness(span: dict) -> dict:
    out = span.get("output") or ""
    refused = any(r in out.lower() for r in _REFUSAL)
    empty = len(out.strip()) < 3
    ok = not (refused or empty)
    detail = "empty" if empty else ("refusal" if refused else "answered")
    return {"name": "completeness", "score": 1.0 if ok else 0.0, "passed": ok, "detail": detail}


def verbosity(span: dict) -> dict:
    n = len(_words(span.get("output") or ""))
    ok = 4 <= n <= 400
    return {"name": "verbosity", "score": 1.0 if ok else 0.0, "passed": ok,
            "detail": f"{n} words"}


def toxicity(span: dict) -> dict:
    out = (span.get("output") or "").lower()
    hits = [t for t in _TOXIC if t in out]
    return {"name": "toxicity", "score": 0.0 if hits else 1.0, "passed": not hits,
            "detail": ("clean" if not hits else f"flagged: {hits}")}


def groundedness(span: dict) -> dict:
    """Fraction of the answer's content words present in the provided context (input).
    A cheap faithfulness proxy; skipped (passes) when the span has no input context."""
    ctx = set(_words(span.get("input") or ""))
    if not ctx:
        return {"name": "groundedness", "score": 1.0, "passed": True, "detail": "no context"}
    ans = [w for w in _words(span.get("output") or "") if w not in _STOP and len(w) > 2]
    if not ans:
        return {"name": "groundedness", "score": 0.0, "passed": False, "detail": "empty answer"}
    grounded = sum(1 for w in ans if w in ctx) / len(ans)
    return {"name": "groundedness", "score": round(grounded, 3), "passed": grounded >= 0.4,
            "detail": f"{grounded:.0%} of content words in context"}


EVALUATORS = [completeness, verbosity, toxicity, groundedness]

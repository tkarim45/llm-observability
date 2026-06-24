"""A small instrumented support assistant — the app under observation.

Each call is one trace with two spans: `retrieve` (KB lookup) and `answer` (LLM generation).
`generate_traffic` runs a mix of good and intentionally-bad requests so the dashboard and
online evals have realistic signal (refusals, empty outputs, ungrounded answers, errors).
"""
from __future__ import annotations

from .providers import get_provider
from .tracer import Tracer

SYSTEM = "You are a helpful, concise customer-support assistant. Answer from the context."

KB = {
    "refund": "Refunds are accepted within 30 days of purchase with a receipt.",
    "password": "Reset your password in Settings > Security > Reset Password.",
    "hours": "Support hours are Monday to Friday, 9am to 6pm Eastern.",
}


def _retrieve(question: str) -> str:
    q = question.lower()
    for key, snippet in KB.items():
        if key in q or (key == "refund" and "refund policy" in q):
            return snippet
    return ""  # no context -> answer will be ungrounded (caught by the groundedness eval)


def answer(question: str, tracer: Tracer | None = None, user: str = "anon") -> dict:
    tracer = tracer or Tracer()
    provider = get_provider()
    with tracer.trace("support_chat", metadata={"user": user, "question": question}) as t:
        with t.span("retrieve", type="retrieval") as s:
            context = _retrieve(question)
            s.set_io(input=question, output=context or "(no match)")

        with t.span("answer", type="generation", model=provider.model) as g:
            g.set_io(input=context)  # context drives the groundedness eval
            res = provider.generate(SYSTEM, question, context)
            g.record_generation(res.text, res.input_tokens, res.output_tokens, res.model)
        return {"trace_id": t.trace_id, "answer": res.text}


# questions that trigger refusal / empty / ungrounded / error, plus normal ones
TRAFFIC = [
    "What is your refund policy?",
    "How do I reset my password?",
    "What are your support hours?",
    "Can I get a refund after 20 days?",
    "How do I reset password on mobile?",
    "Who is your CEO and the stock price?",   # -> refusal
    "Give me a blank response",                # -> empty
    "What is the weather today?",              # -> ungrounded
    "Tell me about returns and refunds",
    "What time does support open?",
]


def generate_traffic(n: int = 0, tracer: Tracer | None = None) -> int:
    tracer = tracer or Tracer()
    items = TRAFFIC if not n else (TRAFFIC * (n // len(TRAFFIC) + 1))[:n]
    count = 0
    for i, q in enumerate(items):
        try:
            if i == 5:  # inject one failed request to populate an error trace
                with tracer.trace("support_chat", metadata={"question": q, "user": "u"}) as t:
                    with t.span("answer", type="generation", model="mock-llm"):
                        raise RuntimeError("provider timeout")
            else:
                answer(q, tracer=tracer, user=f"u{i % 4}")
            count += 1
        except Exception:  # noqa: BLE001 — the failed trace is already persisted
            pass
    return count

"""LLM providers. MockLLM is deterministic + key-free so the whole stack runs offline;
ClaudeLLM is the real Anthropic path used when ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .config import MODEL


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 chars/token rough estimate


class MockLLM:
    model = "mock-llm"

    def generate(self, system: str, prompt: str, context: str = "") -> LLMResult:
        # Deterministic canned behavior, with a few intentionally-bad outputs so the
        # online evals have signal (refusals, empty, ungrounded, toxic).
        p = prompt.lower()
        if "refund policy" in p:
            text = ("Our refund policy allows returns within 30 days of purchase with a "
                    "receipt. Refunds are issued to the original payment method.")
        elif "reset" in p and "password" in p:
            text = ("To reset your password, open Settings, choose Security, and click "
                    "Reset Password. A reset link is emailed to you.")
        elif "hours" in p:
            text = "Our support hours are Monday to Friday, 9am to 6pm Eastern time."
        elif "ceo" in p or "stock price" in p:
            text = "I cannot help with that."  # triggers completeness=refusal
        elif "blank" in p:
            text = ""                            # triggers completeness=empty
        elif "weather" in p:
            text = "The mitochondria is the powerhouse of the cell."  # ungrounded
        else:
            text = ("Thanks for reaching out. Based on the information available, here is a "
                    "concise answer to your question.")
        return LLMResult(text, _est_tokens(system + prompt + context), _est_tokens(text), self.model)


class ClaudeLLM:
    def __init__(self, model: str = MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, system: str, prompt: str, context: str = "") -> LLMResult:
        user = f"{prompt}\n\nContext:\n{context}" if context else prompt
        resp = self.client.messages.create(
            model=self.model, max_tokens=512, system=system,
            messages=[{"role": "user", "content": user}])
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return LLMResult(text, resp.usage.input_tokens, resp.usage.output_tokens, resp.model)


def get_provider():
    """ClaudeLLM when a key is present, else the offline MockLLM."""
    return ClaudeLLM() if os.getenv("ANTHROPIC_API_KEY") else MockLLM()

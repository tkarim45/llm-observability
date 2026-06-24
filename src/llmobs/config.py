"""Paths, model, pricing, and the online-eval sampling rate."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("LLMOBS_ROOT", Path(__file__).resolve().parents[2]))
DB_PATH = Path(os.getenv("LLMOBS_DB", ROOT / "data" / "traces.db"))

MODEL = os.getenv("LLMOBS_MODEL", "claude-opus-4-8")

# Fraction of generations to score with online evals (1.0 = every request).
EVAL_SAMPLE_RATE = float(os.getenv("LLMOBS_EVAL_SAMPLE_RATE", "1.0"))

# USD per 1M tokens, by model substring (first match wins).
PRICING = {
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus": {"input": 5.0, "output": 25.0},
    "claude-sonnet": {"input": 3.0, "output": 15.0},
    "claude-haiku": {"input": 1.0, "output": 5.0},
    "mock": {"input": 0.0, "output": 0.0},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rate = next((v for k, v in PRICING.items() if k in (model or "")), {"input": 0.0, "output": 0.0})
    return (input_tokens / 1e6) * rate["input"] + (output_tokens / 1e6) * rate["output"]

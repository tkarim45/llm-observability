"""Aggregate observability metrics over the stored traces + evals."""
from __future__ import annotations

from collections import defaultdict

from . import store


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(q * len(s)))
    return float(s[idx])


def summary() -> dict:
    traces = store.all_traces()
    evals = store.all_evals()
    n = len(traces)
    if n == 0:
        return {"n_traces": 0}

    latencies = [t["latency_ms"] for t in traces]
    errors = sum(1 for t in traces if t["status"] == "error")
    cost = sum(t["cost_usd"] for t in traces)
    tokens = sum(t["input_tokens"] + t["output_tokens"] for t in traces)

    by_eval: dict[str, list[int]] = defaultdict(list)
    for e in evals:
        by_eval[e["name"]].append(e["passed"])
    eval_pass = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_eval.items())}

    return {
        "n_traces": n,
        "error_rate": round(errors / n, 3),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "total_cost_usd": round(cost, 6),
        "mean_cost_per_trace_usd": round(cost / n, 6),
        "total_tokens": tokens,
        "eval_pass_rate": eval_pass,
    }

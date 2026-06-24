"""Tracing primitives. Wrap a request in a `trace`, wrap each step in a `span`; latency,
tokens, and cost are captured automatically, spans persist on exit, and online evals run
on sampled generation spans.

    from llmobs.tracer import Tracer
    tracer = Tracer()
    with tracer.trace("chat", metadata={"user": "u1"}) as t:
        with t.span("retrieve", type="retrieval") as s:
            s.set_io(input=query, output=docs)
        with t.span("answer", type="generation", model="claude-opus-4-8") as g:
            resp = call_llm(...)
            g.record_generation(resp.text, resp.in_tokens, resp.out_tokens)
"""
from __future__ import annotations

import random
import time
import uuid
from contextlib import contextmanager

from . import store
from .config import EVAL_SAMPLE_RATE, cost_usd


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Span:
    def __init__(self, trace_id: str, name: str, type: str, model: str | None):
        self.span_id = _id("span")
        self.trace_id = trace_id
        self.name = name
        self.type = type
        self.model = model
        self.started_at = time.time()
        self._t0 = time.perf_counter()
        self.input = self.output = None
        self.input_tokens = self.output_tokens = 0
        self.status = "ok"
        self.error: str | None = None

    def set_io(self, input=None, output=None):
        if input is not None:
            self.input = str(input)
        if output is not None:
            self.output = str(output)

    def record_generation(self, output: str, input_tokens: int, output_tokens: int, model: str | None = None):
        """Record an LLM generation's output text + token usage (drives cost)."""
        self.output = output
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if model:
            self.model = model

    def _finalize(self) -> dict:
        latency_ms = int((time.perf_counter() - self._t0) * 1000)
        return {
            "span_id": self.span_id, "trace_id": self.trace_id, "name": self.name,
            "type": self.type, "model": self.model, "started_at": self.started_at,
            "latency_ms": latency_ms, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": cost_usd(self.model or "", self.input_tokens, self.output_tokens),
            "status": self.status, "error": self.error,
            "input": self.input, "output": self.output,
        }


class Trace:
    def __init__(self, name: str, metadata: dict | None):
        self.trace_id = _id("trace")
        self.name = name
        self.metadata = metadata or {}
        self.started_at = time.time()
        self._t0 = time.perf_counter()
        self.spans: list[dict] = []
        self.status = "ok"

    @contextmanager
    def span(self, name: str, type: str = "span", model: str | None = None):
        s = Span(self.trace_id, name, type, model)
        try:
            yield s
        except Exception as e:  # noqa: BLE001 — record the failure on the span + trace
            s.status = "error"
            s.error = f"{e.__class__.__name__}: {e}"  # note: `type` is shadowed by the param
            self.status = "error"
            self.spans.append(s._finalize())
            store.save_span(self.spans[-1])
            raise
        rec = s._finalize()
        self.spans.append(rec)
        store.save_span(rec)


class Tracer:
    def __init__(self, eval_sample_rate: float = EVAL_SAMPLE_RATE, evaluators=None):
        self.eval_sample_rate = eval_sample_rate
        self._rng = random.Random()
        self._evaluators = evaluators  # lazy default to avoid import cycle

    @contextmanager
    def trace(self, name: str, metadata: dict | None = None):
        store.init()
        t = Trace(name, metadata)
        try:
            yield t
        except Exception:
            t.status = "error"
            self._persist(t)
            raise
        self._persist(t)
        self._run_online_evals(t)

    def _persist(self, t: Trace) -> None:
        latency_ms = int((time.perf_counter() - t._t0) * 1000)
        store.save_trace({
            "trace_id": t.trace_id, "name": t.name, "status": t.status,
            "started_at": t.started_at, "ended_at": time.time(), "latency_ms": latency_ms,
            "input_tokens": sum(s["input_tokens"] for s in t.spans),
            "output_tokens": sum(s["output_tokens"] for s in t.spans),
            "cost_usd": sum(s["cost_usd"] for s in t.spans), "metadata": t.metadata,
        })

    def _run_online_evals(self, t: Trace) -> None:
        if self._rng.random() > self.eval_sample_rate:
            return
        from .evals import EVALUATORS

        evaluators = self._evaluators if self._evaluators is not None else EVALUATORS
        for span in t.spans:
            # evaluate every generation span — including empty output, which is exactly
            # what the completeness eval is meant to catch. Skip only un-generated spans.
            if span["type"] != "generation" or span["output"] is None:
                continue
            for ev in evaluators:
                res = ev(span)
                store.save_eval(t.trace_id, span["span_id"], res["name"],
                                res["score"], res["passed"], res["detail"])

"""FastAPI service: an instrumented chat endpoint + observability read APIs.

  POST /chat          run the support assistant (fully traced) -> {answer, trace_id}
  GET  /metrics       aggregate summary (latency p50/p95, cost, error rate, eval pass rates)
  GET  /traces        recent traces
  GET  /traces/{id}   one trace with its spans + eval scores (drill-down)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llmobs import store
from llmobs.demo_app import answer
from llmobs.metrics import summary

app = FastAPI(title="LLM Observability", version="0.1.0")
store.init()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user: str = "anon"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    return answer(req.question, user=req.user)


@app.get("/metrics")
def metrics():
    return summary()


@app.get("/traces")
def traces(limit: int = 50):
    return store.all_traces()[:limit]


@app.get("/traces/{trace_id}")
def trace(trace_id: str):
    spans = store.spans_for(trace_id)
    if not spans:
        raise HTTPException(404, "trace not found")
    return {"trace_id": trace_id, "spans": spans, "evals": store.evals_for(trace_id)}

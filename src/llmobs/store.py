"""SQLite store for traces, spans, and online-eval scores.

One **trace** = one top-level request (e.g. a chat turn). Each trace has ordered **spans**
(retrieval, llm generation, tool calls). **Evals** attach quality scores to a span.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY, name TEXT, status TEXT, started_at REAL, ended_at REAL,
  latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  metadata TEXT
);
CREATE TABLE IF NOT EXISTS spans (
  span_id TEXT PRIMARY KEY, trace_id TEXT, name TEXT, type TEXT, model TEXT,
  started_at REAL, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, status TEXT, error TEXT, input TEXT, output TEXT,
  FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
);
CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, span_id TEXT, name TEXT,
  score REAL, passed INTEGER, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_evals_trace ON evals(trace_id);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


def reset() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    init()


def save_trace(t: dict) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO traces VALUES (:trace_id,:name,:status,:started_at,"
            ":ended_at,:latency_ms,:input_tokens,:output_tokens,:cost_usd,:metadata)",
            {**t, "metadata": json.dumps(t.get("metadata", {}))})


def save_span(s: dict) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO spans VALUES (:span_id,:trace_id,:name,:type,:model,"
            ":started_at,:latency_ms,:input_tokens,:output_tokens,:cost_usd,:status,:error,"
            ":input,:output)", s)


def save_eval(trace_id: str, span_id: str, name: str, score: float, passed: bool, detail: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO evals (trace_id, span_id, name, score, passed, detail) "
                  "VALUES (?,?,?,?,?,?)", (trace_id, span_id, name, score, int(passed), detail))


def all_traces() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM traces ORDER BY started_at DESC")]


def spans_for(trace_id: str) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM spans WHERE trace_id=? ORDER BY started_at", (trace_id,))]


def evals_for(trace_id: str) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM evals WHERE trace_id=?", (trace_id,))]


def all_evals() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM evals")]

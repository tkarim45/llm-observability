# 🔭 LLM Observability & Tracing

> Production monitoring for LLM apps. Instrument any LLM call to capture **prompt, model,
> tokens, latency, and cost** as structured **traces + spans**, run **online quality evals**
> on sampled live traffic, and drill into any request in a dashboard. Core is **stdlib +
> SQLite** (zero deps to trace and store); the demo runs **fully offline** (mock provider),
> and points at **Claude** when a key is present.

You can't operate what you can't see. Once an LLM app is live, the questions are: *what's
my p95 latency? what's this costing? what fraction of answers are ungrounded or refusals,
and which traces?* This gives you all of it per-request — the Langfuse-style trace view,
self-contained.

---

## What it captures

Per **trace** (one request) and its **spans** (retrieval, generation, tools):

| Dimension | How |
|---|---|
| **Latency** | per-span + per-trace, p50 / p95 |
| **Cost** | tokens × per-model pricing, rolled up per trace |
| **Tokens** | input/output per generation span |
| **Errors** | failed spans mark the trace `error`; error-rate tracked |
| **Quality** | **online evals** on sampled generation spans — completeness, verbosity, toxicity, groundedness |

Online evaluators are heuristic + key-free (so observability works offline); an
LLM-as-judge evaluator drops in with the same `(span) -> {score, passed}` signature.

---

## Instrument your app in 6 lines

```python
from llmobs.tracer import Tracer
tracer = Tracer()

with tracer.trace("chat", metadata={"user": "u1"}) as t:
    with t.span("retrieve", type="retrieval") as s:
        s.set_io(input=query, output=docs)
    with t.span("answer", type="generation", model="claude-opus-4-8") as g:
        g.set_io(input=docs)                       # context -> groundedness eval
        res = call_llm(...)
        g.record_generation(res.text, res.in_tokens, res.out_tokens)
# trace + spans persisted; online evals scored automatically on exit
```

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions — never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

# seed demo traffic (offline mock provider) + print the summary
llmobs seed --reset --n 40
#   { "n_traces": 40, "error_rate": 0.025, "latency_p95_ms": ...,
#     "total_cost_usd": ..., "eval_pass_rate": {"completeness": 0.8, "groundedness": 0.7, ...} }

# explore every trace + span + eval in the dashboard
$PY -m streamlit run app/dashboard.py

# or the API: instrumented chat + observability read endpoints
$PY -m uvicorn api.main:app --port 8000
#   POST /chat {"question": "..."}  ·  GET /metrics  ·  GET /traces  ·  GET /traces/{id}

# real LLM instead of the mock:
export ANTHROPIC_API_KEY=sk-ant-...    # generations now hit claude-opus-4-8
```

---

## Architecture

```
  your LLM app  ──@ tracer.trace / t.span ──►  Trace + Spans  (latency · tokens · cost)
        │                                            │  persist
        ▼                                            ▼
  providers.py  (Claude / MockLLM)            store.py  →  SQLite (traces · spans · evals)
        │                                            ▲
        └─ generation span ──► evals.py ────online evals (sampled) ──┘
           completeness · verbosity · toxicity · groundedness
                                                     │
                            metrics.py · FastAPI /metrics · Streamlit dashboard (drill-down)
```

---

## What the dashboard shows

- **Overview** — trace count, error rate, latency p95, total + per-trace cost
- **Online eval pass rates** — completeness / verbosity / toxicity / groundedness across traffic
- **Trace list** — every request with status, latency, tokens, cost
- **Drill-down** — each span's input/output, tokens, cost, errors + the eval scores for that trace

The demo traffic deliberately includes a refusal, an empty response, an ungrounded answer,
and a failed request — so the eval pass rates and error rate are non-trivial out of the box.

---

## Repo layout

```
llm-observability/
├── src/llmobs/
│   ├── tracer.py    Trace/Span context managers — auto latency/tokens/cost, online evals on exit
│   ├── store.py     SQLite store (traces · spans · evals)
│   ├── evals.py     online evaluators (completeness · verbosity · toxicity · groundedness)
│   ├── providers.py MockLLM (offline) + ClaudeLLM
│   ├── demo_app.py  instrumented support assistant + traffic generator
│   ├── metrics.py   latency p50/p95 · cost · error rate · eval pass rates
│   ├── config.py    model, pricing, sample rate
│   └── cli.py       `llmobs seed | summary | serve | dashboard`
├── api/main.py      FastAPI: /chat (traced) · /metrics · /traces · /traces/{id}
├── app/dashboard.py Streamlit observability dashboard
├── tests/           tracer + evals + metrics (key-free) — 6 cases
└── pyproject.toml · Dockerfile · Makefile · .github/workflows/ci.yml
```

---

## Résumé framing

> *Built an LLM observability layer — per-request tracing of prompt/tokens/latency/cost with
> online quality evals (completeness, groundedness, toxicity) on sampled traffic, a metrics
> API (p50/p95 latency, cost, error + eval pass rates) and a drill-down dashboard; stdlib +
> SQLite core, offline-capable, Claude-instrumented.*

## License
MIT (`LICENSE`).

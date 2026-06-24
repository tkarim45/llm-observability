"""Key-free tests: tracer persistence + cost, evaluators, and end-to-end metrics."""
import pytest

from llmobs import store
from llmobs.demo_app import answer, generate_traffic
from llmobs.evals import completeness, groundedness, toxicity, verbosity
from llmobs.metrics import summary
from llmobs.tracer import Tracer


@pytest.fixture(autouse=True)
def fresh_store():
    store.reset()


def test_trace_and_spans_persist_with_cost():
    tracer = Tracer()
    with tracer.trace("t", metadata={"x": 1}) as t:
        with t.span("gen", type="generation", model="claude-opus-4-8") as g:
            g.record_generation("hello world", 1000, 200)
    traces = store.all_traces()
    assert len(traces) == 1
    tr = traces[0]
    # cost = 1000/1e6*5 + 200/1e6*25 = 0.005 + 0.005 = 0.01
    assert tr["cost_usd"] == pytest.approx(0.01)
    assert len(store.spans_for(tr["trace_id"])) == 1


def test_online_evals_run_on_generation():
    tracer = Tracer(eval_sample_rate=1.0)
    with tracer.trace("t") as t:
        with t.span("gen", type="generation", model="mock") as g:
            g.set_io(input="refund within 30 days")
            g.record_generation("Refunds within 30 days are accepted.", 10, 10)
    evals = store.all_evals()
    names = {e["name"] for e in evals}
    assert {"completeness", "verbosity", "toxicity", "groundedness"} <= names


def test_error_span_marks_trace_failed():
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        with tracer.trace("t") as t:
            with t.span("gen", type="generation"):
                raise RuntimeError("boom")
    assert store.all_traces()[0]["status"] == "error"


def test_evaluators_detect_problems():
    assert completeness({"output": "I can't help with that."})["passed"] is False
    assert completeness({"output": ""})["passed"] is False
    assert toxicity({"output": "you are stupid"})["passed"] is False
    assert verbosity({"output": "ok"})["passed"] is False  # too short
    g = groundedness({"input": "refund within 30 days receipt",
                      "output": "Refunds accepted within 30 days with a receipt."})
    assert g["passed"] is True
    g2 = groundedness({"input": "refund policy", "output": "mitochondria powerhouse cell"})
    assert g2["passed"] is False


def test_demo_answer_returns_trace():
    out = answer("What is your refund policy?")
    assert out["trace_id"].startswith("trace_") and "refund" in out["answer"].lower()


def test_traffic_populates_metrics_with_errors():
    generate_traffic(10)
    s = summary()
    assert s["n_traces"] >= 9
    assert s["error_rate"] > 0          # one injected failure
    assert "completeness" in s["eval_pass_rate"]
    assert s["eval_pass_rate"]["completeness"] < 1.0  # refusal + empty cases caught

"""Observability dashboard: overview metrics, online-eval pass rates, trace list + drill-down."""
from __future__ import annotations

import streamlit as st

from llmobs import store
from llmobs.demo_app import generate_traffic
from llmobs.metrics import summary

st.set_page_config(page_title="LLM Observability", layout="wide")
st.title("🔭 LLM Observability & Tracing")
st.caption("Every request traced — prompt, tokens, latency, cost, and quality scored online.")

store.init()
if not store.all_traces():
    st.warning("No traces yet.")
    if st.button("Generate demo traffic"):
        generate_traffic(40)
        st.rerun()
    st.stop()

s = summary()
c = st.columns(5)
c[0].metric("traces", s["n_traces"])
c[1].metric("error rate", f"{s['error_rate']*100:.0f}%")
c[2].metric("latency p95", f"{s['latency_p95_ms']} ms")
c[3].metric("total cost", f"${s['total_cost_usd']:.4f}")
c[4].metric("cost / trace", f"${s['mean_cost_per_trace_usd']:.5f}")

st.subheader("Online eval pass rates")
ec = st.columns(len(s["eval_pass_rate"]) or 1)
for i, (name, rate) in enumerate(s["eval_pass_rate"].items()):
    ec[i].metric(name, f"{rate*100:.0f}%")

st.subheader("Traces")
traces = store.all_traces()
st.dataframe([{
    "trace_id": t["trace_id"], "name": t["name"], "status": t["status"],
    "latency_ms": t["latency_ms"], "tokens": t["input_tokens"] + t["output_tokens"],
    "cost$": round(t["cost_usd"], 6),
} for t in traces], use_container_width=True, hide_index=True)

st.subheader("Trace drill-down")
tid = st.selectbox("trace", [t["trace_id"] for t in traces])
spans, evals = store.spans_for(tid), store.evals_for(tid)
meta = next((t for t in traces if t["trace_id"] == tid), {})
st.caption(f"status: {meta.get('status')} · latency {meta.get('latency_ms')}ms · "
           f"cost ${meta.get('cost_usd', 0):.6f}")

for sp in spans:
    icon = "🟢" if sp["status"] == "ok" else "🔴"
    with st.expander(f"{icon} {sp['name']} · {sp['type']} · {sp['latency_ms']}ms"
                     + (f" · {sp['model']}" if sp["model"] else "")):
        if sp["input"]:
            st.markdown("**input**"); st.code(sp["input"])
        if sp["output"]:
            st.markdown("**output**"); st.code(sp["output"])
        if sp["error"]:
            st.error(sp["error"])
        st.caption(f"tokens in/out: {sp['input_tokens']}/{sp['output_tokens']} · "
                   f"cost ${sp['cost_usd']:.6f}")

if evals:
    st.markdown("**online evals**")
    st.dataframe([{"eval": e["name"], "score": e["score"],
                   "passed": "✅" if e["passed"] else "❌", "detail": e["detail"]}
                  for e in evals], use_container_width=True, hide_index=True)

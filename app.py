"""
insightx/app.py  —  InsightX UPI Payment Intelligence
"""
import sys, re
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="InsightX BI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
from src.data_loader        import load_data, get_schema_info
from src.nlp_engine         import parse_query
from src.analytics_engine   import execute_query, METRIC_AGG
from src.response_generator import generate_response
from src.visualizer         import (
    bar_chart, line_chart, gauge_chart,
    anomaly_chart, donut_chart, comparison_bar_chart,
)
from src.context_memory import ContextMemory

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #0F172A !important; }
.block-container { padding-top: 1rem !important; max-width: 1100px; }
[data-testid="stSidebar"] { background: #020817 !important; border-right: 1px solid #1E293B; }
.stButton > button {
    background: #1E293B !important; border: 1px solid #334155 !important;
    color: #94A3B8 !important; border-radius: 8px !important; font-size: 12px !important;
}
.stButton > button:hover { border-color: #00D4FF !important; color: #00D4FF !important; }
.stChatInput textarea {
    background: #1E293B !important; border: 1px solid #334155 !important;
    color: #E2E8F0 !important; border-radius: 12px !important;
}
.stProgress > div > div { background: linear-gradient(90deg,#00D4FF,#7C3AED) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DATA + STATE
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def cached_load():
    return load_data()

df     = cached_load()
schema = get_schema_info(df)

if "messages" not in st.session_state: st.session_state.messages = []
if "memory"   not in st.session_state: st.session_state.memory   = ContextMemory()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ InsightX BI")
    st.caption("Conversational analytics for UPI payments")
    st.divider()
    st.markdown("**📊 Dataset**")
    st.caption(f"Rows: {schema['total_rows']:,}  |  {schema['date_range'][0]} → {schema['date_range'][1]}")
    st.caption(f"States: {len(schema['states'])}  |  Banks: {len(schema['banks'])}  |  Categories: {len(schema['categories'])}")
    st.divider()
    st.markdown("**🎯 Sample Queries**")
    for sq in [
        "Overall summary", "Compare fraud rate by device",
        "Which bank has highest failure rate?", "Avg amount by category",
        "Fraud trend by day", "Top 5 states by volume",
        "Peak transaction hour", "Failure rate on 3G vs 5G",
        "Fraud rate for 18-25 age group", "Compare P2P vs P2M count",
        "Evening transactions fraud rate", "Maharashtra transaction analysis",
        "HDFC vs SBI comparison", "Education category avg amount",
    ]:
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state["trigger_query"] = sq
    st.divider()
    st.markdown("**🕓 Query History**")
    history = st.session_state.memory.get_history_display()
    if not history:
        st.caption("No queries yet")
    else:
        for item in history[:6]:
            st.caption(f"▸ {item['query'][:40]}")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    st.divider()
    show_json_global = st.toggle("Show Query JSON",     value=False)
    show_anomalies   = st.toggle("Highlight Anomalies", value=True)

# ══════════════════════════════════════════════════════════════
#  HEADER (Fixed + Dynamic + Safe)
# ══════════════════════════════════════════════════════════════

st.markdown("## ⚡ InsightX — UPI Payment Intelligence")
st.caption(f"{schema['total_rows']:,} transactions · {schema['date_range'][0]} to {schema['date_range'][1]}")

# --- SAFE COLUMN HANDLING ---
cols_lower = {c.lower().strip(): c for c in df.columns}

# Detect amount column safely
amount_col = None
for c in df.columns:
    if "amount" in c.lower():
        amount_col = c
        break

# Compute metrics safely
total_txns = len(df)

failure_rate = (
    df["transaction_status"].str.lower().eq("failed").mean()
    if "transaction_status" in df.columns else 0
)

fraud_rate = (
    df["fraud_flag"].mean()
    if "fraud_flag" in df.columns else 0
)

avg_amount = (
    df[amount_col].mean()
    if amount_col else 0
)

# --- FORMATTERS ---
def format_lakh(value):
    return f"{value/100000:.1f}L" if value >= 100000 else f"{value:,}"

def format_percent(value):
    return f"{value*100:.2f}%"

# --- KPI ROW ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Transactions", format_lakh(total_txns))

with col2:
    st.metric("Failure Rate", format_percent(failure_rate))

with col3:
    st.metric("Fraud Rate", format_percent(fraud_rate))

with col4:
    st.metric("Avg Amount", f"₹{avg_amount:,.0f}")

st.divider()
# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def _clean(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", str(text))

def _conf_emoji(pct: int) -> str:
    return "🟢" if pct >= 75 else "🟡" if pct >= 50 else "🔴"

# ══════════════════════════════════════════════════════════════
#  RENDER AI MESSAGE  —  100% native Streamlit, no raw HTML
# ══════════════════════════════════════════════════════════════
def render_ai_message(msg: dict, idx: int):
    result       = msg["result"]
    response     = msg["response"]
    query        = msg["query"]
    is_followup  = getattr(query, "followup", False)
    compare_vals = getattr(query, "compare", [])
    conf_pct     = int(response.confidence_bar * 100)

    # ── Top bar: headline + confidence ───────────────────────
    h_col, c_col = st.columns([7, 3])
    with h_col:
        tag = "  ↩ *follow-up*" if is_followup else ""
        st.markdown(f"### {_clean(response.headline)}{tag}")
    with c_col:
        st.caption(f"{_conf_emoji(conf_pct)} **{response.confidence_label}** ({conf_pct}%)")
        st.progress(response.confidence_bar)

    # ── Comparison segment pills ──────────────────────────────
    if compare_vals and len(compare_vals) >= 2:
        pill_cols = st.columns(len(compare_vals))
        colors = ["🔵", "🟣", "🟡", "🟢", "🔴"]
        for i, v in enumerate(compare_vals):
            pill_cols[i].info(f"{colors[i % len(colors)]} **{v}**")

    # ── Narrative ─────────────────────────────────────────────
    st.caption(_clean(response.narrative))
    st.write("")

    # ── Chart + Insights ──────────────────────────────────────
    if result.result_df is not None and not result.result_df.empty:
        chart_col, insight_col = st.columns([6, 4])

        with chart_col:
            if compare_vals and len(compare_vals) >= 2 and response.chart_type != "line":
                fig = comparison_bar_chart(result, compare_vals)
            elif response.chart_type == "line":
                fig = line_chart(result)
            elif response.chart_type == "anomaly":
                fig = anomaly_chart(result)
            elif response.chart_type == "donut":
                fig = donut_chart(result.result_df, result.group_by, result.metric)
            else:
                fig = bar_chart(result)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")

        with insight_col:
            if response.bullets:
                st.markdown("**Key Insights**")
                for b in response.bullets[:4]:
                    st.markdown(f"- {_clean(b)}")

            if response.risk_flags and show_anomalies:
                st.markdown("---")
                st.markdown("**⚠️ Risk Flags**")
                for f in response.risk_flags[:2]:
                    st.error(_clean(f))

            if response.recommendations:
                st.markdown("---")
                st.markdown("**✅ Actions**")
                for r in response.recommendations[:2]:
                    st.success(_clean(r))

    elif result.scalar_value is not None:
        g_col, t_col = st.columns([4, 6])
        with g_col:
            st.plotly_chart(
                gauge_chart(result.scalar_value, result.metric, result.metric_label),
                use_container_width=True, key=f"sg_{idx}",
            )
        with t_col:
            if response.bullets:
                st.markdown("**Highlights**")
                for b in response.bullets:
                    st.markdown(f"- {_clean(b)}")

    # ── Expanders ─────────────────────────────────────────────
    if result.result_df is not None:
        with st.expander("📋 Data Table"):
            display_df = result.result_df.copy()
            if "value" in display_df.columns:
                fmt = METRIC_AGG.get(result.metric, {}).get("fmt", "{:.2f}")
                display_df["value"] = display_df["value"].apply(lambda v: fmt.format(v))
                display_df = display_df.rename(columns={"value": result.metric_label})
                if "count" in display_df.columns:
                    display_df = display_df.drop(columns=["count"])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    if show_json_global:
        with st.expander("🔧 Query JSON"):
            st.code(result.query_json, language="json")

    if result.anomalies and show_anomalies:
        with st.expander(f"⚠️ {len(result.anomalies)} Anomalies Detected"):
            for a in result.anomalies:
                icon = "🔴" if a["direction"] == "high" else "🔵"
                st.markdown(f"{icon} **{a['group']}** — {a['formatted']} (z = `{a['z_score']:+.2f}`)")

    # ── Follow-ups ────────────────────────────────────────────
    if response.follow_ups:
        st.markdown("**💡 Follow-up**")
        fu_cols = st.columns(min(len(response.follow_ups[:4]), 4))
        for fi, fq in enumerate(response.follow_ups[:4]):
            with fu_cols[fi]:
                if st.button(fq, key=f"fu_{idx}_{fi}", use_container_width=True):
                    st.session_state["trigger_query"] = fq

# ══════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ══════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("<div style='text-align:center;padding:50px 20px'><h1>⚡</h1><h3 style='color:#E2E8F0'>Ask anything about your UPI data</h3><p style='color:#64748B'>Type below or pick a sample from the sidebar</p></div>", unsafe_allow_html=True)
    quick = ["📊 Overall summary","🔴 Fraud by device","❌ Failure rate by bank","💰 Avg amount by category","📈 Transaction trend by day","🗺️ Top states by volume"]
    cols = st.columns(3)
    for qi, q in enumerate(quick):
        with cols[qi % 3]:
            if st.button(q, key=f"qs_{qi}", use_container_width=True):
                st.session_state["trigger_query"] = q.split(" ", 1)[1]
else:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="⚡"):
                render_ai_message(msg, i)

# ══════════════════════════════════════════════════════════════
#  CHAT INPUT + PROCESSING
# ══════════════════════════════════════════════════════════════
if "trigger_query" in st.session_state:
    st.session_state["pending_query"] = st.session_state.pop("trigger_query")
    st.rerun()

user_input       = st.chat_input("Ask about fraud, failures, amounts, trends, comparisons…")
query_to_process = st.session_state.pop("pending_query", None) or user_input

if query_to_process:
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.spinner("⚡ Analysing…"):
        context_hint = st.session_state.memory.to_prompt_context()
        parsed_query = parse_query(query_to_process, context_hint=context_hint)
        st.session_state.memory.push(parsed_query)
        exec_result  = execute_query(parsed_query, df)
        ai_response  = generate_response(exec_result, parsed_query.confidence)
    st.session_state.messages.append({
        "role": "assistant", "content": query_to_process,
        "result": exec_result, "response": ai_response, "query": parsed_query,
    })
    st.rerun()

"""
insightx/app.py
───────────────
InsightX — Conversational Business Intelligence for UPI Payments
Streamlit entry point.

Run with:
    streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Path setup (MUST be before local imports) ─────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Streamlit (MUST be before any st.* usage) ─────────────────────────────
import streamlit as st

# ── Third-party ───────────────────────────────────────────────────────────
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────
from src.data_loader        import load_data, get_schema_info
from src.nlp_engine         import parse_query
from src.analytics_engine   import execute_query, METRIC_AGG
from src.response_generator import generate_response
from src.visualizer         import (
    bar_chart, line_chart, gauge_chart,
    heatmap_hourly, anomaly_chart, donut_chart,
    overview_kpi_bars, comparison_bar_chart, pick_chart,
)
from src.context_memory     import ContextMemory, parse_llm_response
from src.system_prompt      import SYSTEM_PROMPT
from ui_components          import (
    inject_custom_css, render_chat_message,
    render_comparison_header, render_context_trail, render_kpi_row,
)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first st.* call)
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title            = "InsightX BI",
    page_icon             = "⚡",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════
#  INJECT UI CSS
# ══════════════════════════════════════════════════════════════════════════
inject_custom_css()

# ══════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS (legacy styles kept for backwards compat)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #0F172A; }

[data-testid="stSidebar"] { background: #020817 !important; border-right: 1px solid #1E293B; }
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] h1,h2,h3 { color: #E2E8F0 !important; }

[data-testid="stChatInput"] textarea {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans' !important;
}

.kpi-card {
    background: rgba(30, 41, 59, 0.8);                     /* Softer, premium card look */
    border: 1px solid #334155;
    border-radius: 14px;

    padding: 22px 20px;                                   /* More breathing room */
    min-height: 120px;                                    /* Uniform height */
    
    display: flex;
    flex-direction: column;
    justify-content: center;                              /* Vertical centering */
    align-items: center;                                  /* Horizontal centering */
    text-align: center;

    backdrop-filter: blur(6px);                           /* Subtle glass effect */
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);

    transition: transform .2s, border-color .2s;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: #00D4FF;
}
.kpi-value { font-size: 28px; font-weight: 800; color: #00D4FF; font-family: 'Space Mono'; }
.kpi-label { font-size: 12px; color: #64748B; margin-top: 4px; letter-spacing: .8px; text-transform: uppercase; }

.user-bubble {
    background: linear-gradient(135deg, #00D4FF22, #7C3AED22);
    border: 1px solid #00D4FF44;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    margin: 8px 0;
    color: #E2E8F0;
    max-width: 80%;
    margin-left: auto;
}
.ai-bubble {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #E2E8F0;
}
.headline  { font-size: 17px; font-weight: 700; color: #E2E8F0; margin-bottom: 8px; }
.narrative { font-size: 14px; color: #94A3B8; line-height: 1.7; }

.risk-flag {
    background: #EF444420;
    border-left: 3px solid #EF4444;
    border-radius: 0 8px 8px 0;
    padding: 8px 12px; margin: 4px 0;
    font-size: 13px; color: #FCA5A5;
}
.rec-flag {
    background: #10B98120;
    border-left: 3px solid #10B981;
    border-radius: 0 8px 8px 0;
    padding: 8px 12px; margin: 4px 0;
    font-size: 13px; color: #6EE7B7;
}

.conf-bar-bg   { background: #1E293B; border-radius: 6px; height: 8px; overflow: hidden; margin: 6px 0; }
.conf-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #00D4FF, #7C3AED); }

.pill { display:inline-block; background:#1E293B; border:1px solid #334155; border-radius:20px;
        padding:3px 12px; font-size:11px; color:#64748B; margin:2px; cursor:pointer; }
.pill:hover { border-color:#00D4FF; color:#00D4FF; }

.json-block { background:#020817; border:1px solid #1E293B; border-radius:8px; padding:14px;
              font-family:'Space Mono',monospace; font-size:11px; color:#64748B;
              overflow-x:auto; white-space:pre; }

.section-header { font-size:11px; text-transform:uppercase; letter-spacing:1.5px;
                  color:#475569; margin:12px 0 6px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def cached_load():
    return load_data()

df     = cached_load()
schema = get_schema_info(df)


# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
if "messages"  not in st.session_state:
    st.session_state.messages  = []
if "show_json" not in st.session_state:
    st.session_state.show_json = {}
if "memory"    not in st.session_state:
    st.session_state.memory    = ContextMemory()


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ InsightX BI")
    st.markdown("*Conversational analytics for UPI payments*")
    st.markdown("---")

    st.markdown("### 📊 Dataset")
    st.markdown(f"""
- **Rows:** {schema['total_rows']:,}
- **Date range:** {schema['date_range'][0]} → {schema['date_range'][1]}
- **States:** {len(schema['states'])}
- **Banks:** {len(schema['banks'])}
- **Categories:** {len(schema['categories'])}
""")

    st.markdown("---")
    st.markdown("### 🎯 Sample Queries")
    sample_queries = [
        "Overall summary",
        "Compare fraud rate by device",
        "Which bank has highest failure rate?",
        "Avg amount by category",
        "Fraud trend by day",
        "Top 5 states by volume",
        "Peak transaction hour",
        "Failure rate on 3G vs 5G",
        "Fraud rate for 18-25 age group",
        "Compare P2P vs P2M count",
        "Evening transactions fraud rate",
        "Maharashtra transaction analysis",
        "Anomaly detection in fraud rate",
        "HDFC vs SBI comparison",
        "Education category avg amount",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state["trigger_query"] = sq

    st.markdown("---")

    # ── Query history trail ───────────────────────────────────────────────
    st.markdown("### 🕓 Query History")
    render_context_trail(st.session_state.memory.get_history_display())

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.show_json = {}
        st.session_state.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    show_json_global = st.toggle("Show Query JSON",     value=False)
    show_anomalies   = st.toggle("Highlight Anomalies", value=True)


# ══════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════
col_logo, col_title, col_stats = st.columns([1, 5, 4])
with col_logo:
    st.markdown("# ⚡")
with col_title:
    st.markdown("## InsightX — UPI Payment Intelligence")
    st.markdown(
        f"<span style='color:#64748B;font-size:13px'>"
        f"2,50,000 transactions · {schema['date_range'][0]} to {schema['date_range'][1]}"
        f"</span>",
        unsafe_allow_html=True,
    )
with col_stats:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="kpi-card"><div class="kpi-value">2.5L</div><div class="kpi-label">Transactions</div></div>',  unsafe_allow_html=True)
    c2.markdown('<div class="kpi-card"><div class="kpi-value">4.95%</div><div class="kpi-label">Fail Rate</div></div>',   unsafe_allow_html=True)
    c3.markdown('<div class="kpi-card"><div class="kpi-value">0.19%</div><div class="kpi-label">Fraud Rate</div></div>',  unsafe_allow_html=True)
    c4.markdown('<div class="kpi-card"><div class="kpi-value">₹1,312</div><div class="kpi-label">Avg Amount</div></div>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
#  HELPER: RENDER ONE AI MESSAGE
# ══════════════════════════════════════════════════════════════════════════

def render_ai_message(msg: dict, idx: int):
    result        = msg["result"]
    response      = msg["response"]
    query         = msg["query"]
    is_followup   = getattr(query, "followup", False)
    compare_vals  = getattr(query, "compare",  [])

    with st.container():
        # ── Comparison header (shows "3G vs 5G" pills) ────────────────────
        if compare_vals and len(compare_vals) >= 2:
            render_comparison_header(compare_vals, result.metric_label)

        # ── Headline + confidence ─────────────────────────────────────────
        col_h, col_c = st.columns([6, 2])
        with col_h:
            followup_tag = " <span style='font-size:11px;background:#7C3AED22;color:#7C3AED;border:1px solid #7C3AED44;border-radius:4px;padding:2px 7px;vertical-align:middle'>↩ follow-up</span>" if is_followup else ""
            st.markdown(
                f"<div class='headline'>{response.headline}{followup_tag}</div>",
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                f"<div class='section-header'>Confidence: {response.confidence_label}</div>",
                unsafe_allow_html=True,
            )
            conf_pct = int(response.confidence_bar * 100)
            st.markdown(f"""
<div class="conf-bar-bg">
  <div class="conf-bar-fill" style="width:{conf_pct}%"></div>
</div>
<div style="font-size:10px;color:#475569;text-align:right">{conf_pct}%</div>
""", unsafe_allow_html=True)

        # ── Narrative ─────────────────────────────────────────────────────
        st.markdown(
            f"<div class='narrative'>{response.narrative}</div>",
            unsafe_allow_html=True,
        )

        # ── Chart ─────────────────────────────────────────────────────────
        if result.result_df is not None and not result.result_df.empty:
            chart_col, insight_col = st.columns([6, 4])

            with chart_col:
                # Use comparison chart when compare values are present
                if compare_vals and len(compare_vals) >= 2 and response.chart_type != "line":
                    fig = comparison_bar_chart(result, compare_vals)
                elif response.chart_type == "line":
                    fig = line_chart(result)
                elif response.chart_type == "gauge" and result.scalar_value is not None:
                    fig = gauge_chart(result.scalar_value, result.metric, result.metric_label)
                elif response.chart_type == "anomaly":
                    fig = anomaly_chart(result)
                elif response.chart_type == "donut":
                    fig = donut_chart(result.result_df, result.group_by, result.metric)
                else:
                    fig = bar_chart(result)

                st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")

            with insight_col:
                if response.bullets:
                    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
                    for b in response.bullets[:4]:
                        st.markdown(
                            f"<div style='font-size:13px;color:#CBD5E1;padding:4px 0;"
                            f"border-bottom:1px solid #1E293B'>{b}</div>",
                            unsafe_allow_html=True,
                        )

                if response.risk_flags and show_anomalies:
                    st.markdown("<div class='section-header' style='margin-top:12px'>Risk Flags</div>", unsafe_allow_html=True)
                    for f in response.risk_flags[:3]:
                        st.markdown(f"<div class='risk-flag'>{f}</div>", unsafe_allow_html=True)

                if response.recommendations:
                    st.markdown("<div class='section-header' style='margin-top:12px'>Recommendations</div>", unsafe_allow_html=True)
                    for r in response.recommendations[:2]:
                        st.markdown(f"<div class='rec-flag'>{r}</div>", unsafe_allow_html=True)

        elif result.scalar_value is not None:
            g_col, t_col = st.columns([4, 6])
            with g_col:
                st.plotly_chart(
                    gauge_chart(result.scalar_value, result.metric, result.metric_label),
                    use_container_width=True, key=f"sg_{idx}",
                )
            with t_col:
                if response.bullets:
                    st.markdown("<div class='section-header'>Highlights</div>", unsafe_allow_html=True)
                    for b in response.bullets:
                        st.markdown(
                            f"<div style='font-size:13px;color:#CBD5E1;padding:4px 0'>{b}</div>",
                            unsafe_allow_html=True,
                        )

        # ── Data table toggle ─────────────────────────────────────────────
        if result.result_df is not None:
            with st.expander("📋 View Data Table", expanded=False):
                display_df = result.result_df.copy()
                if "value" in display_df.columns:
                    fmt = METRIC_AGG.get(result.metric, {}).get("fmt", "{:.2f}")
                    display_df["value"] = display_df["value"].apply(lambda v: fmt.format(v))
                    display_df = display_df.rename(columns={"value": result.metric_label})
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── JSON panel ────────────────────────────────────────────────────
        if show_json_global:
            with st.expander("🔧 Structured Query JSON", expanded=False):
                st.markdown(
                    f"<div class='json-block'>{result.query_json}</div>",
                    unsafe_allow_html=True,
                )

        # ── Anomaly detail ────────────────────────────────────────────────
        if result.anomalies and show_anomalies:
            with st.expander(f"⚠️ {len(result.anomalies)} Anomalies Detected", expanded=False):
                for a in result.anomalies:
                    icon = "🔴" if a["direction"] == "high" else "🔵"
                    st.markdown(
                        f"{icon} **{a['group']}** → {a['formatted']} "
                        f"&nbsp; (z-score: `{a['z_score']:+.2f}`)"
                    )

        # ── Follow-up suggestions ─────────────────────────────────────────
        if response.follow_ups:
            st.markdown(
                "<div class='section-header' style='margin-top:12px'>💡 Follow-up Questions</div>",
                unsafe_allow_html=True,
            )
            fu_cols = st.columns(min(len(response.follow_ups[:4]), 4))
            for fi, fq in enumerate(response.follow_ups[:4]):
                with fu_cols[fi]:
                    if st.button(fq, key=f"fu_{idx}_{fi}", use_container_width=True):
                        st.session_state["trigger_query"] = fq


# ══════════════════════════════════════════════════════════════════════════
#  RENDER CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.markdown("""
<div style="text-align:center;padding:40px 20px">
  <div style="font-size:52px;margin-bottom:16px">⚡</div>
  <div style="font-size:22px;font-weight:700;color:#E2E8F0;margin-bottom:8px">Ask anything about your UPI data</div>
  <div style="color:#64748B;font-size:14px">Type a question below or pick a sample from the sidebar</div>
</div>
""", unsafe_allow_html=True)

    quick = [
        "📊 Overall summary",        "🔴 Fraud by device",
        "❌ Failure rate by bank",    "💰 Avg amount by category",
        "📈 Transaction trend by day","🗺️ Top states by volume",
    ]
    cols = st.columns(3)
    for qi, q in enumerate(quick):
        with cols[qi % 3]:
            if st.button(q, key=f"qs_{qi}", use_container_width=True):
                st.session_state["trigger_query"] = q.split(" ", 1)[1]

else:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
<div style="display:flex;justify-content:flex-end;margin:12px 0">
  <div class="user-bubble">💬 {msg["content"]}</div>
</div>""", unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:8px 0 4px">
  <div style="width:28px;height:28px;border-radius:8px;
              background:linear-gradient(135deg,#00D4FF,#7C3AED);
              display:flex;align-items:center;justify-content:center;font-size:14px">⚡</div>
  <div style="font-size:12px;color:#475569;font-weight:600">InsightX</div>
</div>""", unsafe_allow_html=True)
                render_ai_message(msg, i)
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
#  CHAT INPUT + QUERY PROCESSING
# ══════════════════════════════════════════════════════════════════════════

# Handle sidebar / follow-up button triggers
if "trigger_query" in st.session_state:
    triggered = st.session_state.pop("trigger_query")
    st.session_state["pending_query"] = triggered
    st.rerun()

user_input = st.chat_input("Ask about fraud, failures, amounts, trends, comparisons…")

query_to_process = st.session_state.pop("pending_query", None) or user_input

if query_to_process:
    st.session_state.messages.append({"role": "user", "content": query_to_process})

    with st.spinner("⚡ Analysing…"):
        # ── Build context-aware prompt ────────────────────────────────────
        context_hint  = st.session_state.memory.to_prompt_context()
        parsed_query  = parse_query(query_to_process, context_hint=context_hint)

        # ── Push to memory ────────────────────────────────────────────────
        st.session_state.memory.push(parsed_query)

        # ── Execute + respond ─────────────────────────────────────────────
        exec_result   = execute_query(parsed_query, df)
        ai_response   = generate_response(exec_result, parsed_query.confidence)

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  query_to_process,
        "result":   exec_result,
        "response": ai_response,
        "query":    parsed_query,
    })

    st.rerun()
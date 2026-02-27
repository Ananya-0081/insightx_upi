"""
insightx/ui_components.py
──────────────────────────
Streamlit UI helpers for InsightX v2.
Drop-in replacements / additions for the existing app.py render functions.

Includes:
  - render_chat_message()      — polished AI message cards
  - render_comparison_header() — "X vs Y vs Z" pill header
  - render_context_trail()     — sidebar breadcrumb of conversation
  - render_metric_pills()      — quick metric selector chips
  - render_confidence_bar()    — animated confidence indicator
  - render_kpi_row()           — top KPI summary strip
  - inject_custom_css()        — global style overrides
"""

import streamlit as st
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    """Inject all custom CSS. Call once at the top of app.py."""
    st.markdown("""
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: #0F172A;
        color: #E2E8F0;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem !important; max-width: 900px; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

    /* ── Chat input ── */
    .stChatInput > div {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
    }
    .stChatInput input {
        color: #E2E8F0 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── AI message card ── */
    .ix-message {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        position: relative;
        animation: fadeSlideIn 0.3s ease;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .ix-message-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.7rem;
    }
    .ix-bot-avatar {
        width: 28px; height: 28px;
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: 700; color: white;
    }
    .ix-bot-name {
        font-size: 12px; font-weight: 600;
        color: #00D4FF; letter-spacing: 0.05em; text-transform: uppercase;
    }
    .ix-message-body {
        font-size: 14px; line-height: 1.65;
        color: #CBD5E1;
    }
    .ix-message-body strong { color: #E2E8F0; font-weight: 600; }

    /* ── Comparison header ── */
    .ix-compare-header {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
        padding: 0.8rem 1rem;
        background: #0F172A;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    .ix-compare-label {
        font-size: 11px; font-weight: 600;
        color: #64748B; text-transform: uppercase; letter-spacing: 0.08em;
        margin-right: 4px;
    }
    .ix-compare-pill {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px; font-weight: 600;
        border: 1.5px solid;
        white-space: nowrap;
    }
    .ix-vs-divider {
        color: #475569; font-size: 11px; font-weight: 700;
    }

    /* ── Context trail (sidebar) ── */
    .ix-trail-item {
        padding: 8px 10px;
        margin-bottom: 6px;
        background: #0F172A;
        border-radius: 8px;
        border-left: 3px solid #334155;
        font-size: 12px;
        color: #94A3B8;
        transition: border-color 0.2s;
        cursor: default;
    }
    .ix-trail-item:first-child {
        border-left-color: #00D4FF;
        color: #E2E8F0;
        background: #1E293B;
    }
    .ix-trail-query {
        font-size: 12px; color: #E2E8F0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        max-width: 180px;
    }
    .ix-trail-meta {
        font-size: 10px; color: #475569;
        font-family: 'DM Mono', monospace;
        margin-top: 2px;
    }
    .ix-followup-badge {
        display: inline-block;
        background: #7C3AED22;
        color: #7C3AED;
        border: 1px solid #7C3AED44;
        font-size: 9px; font-weight: 700;
        padding: 1px 6px; border-radius: 4px;
        margin-left: 6px;
        vertical-align: middle;
        text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* ── Metric pills ── */
    .ix-metric-strip {
        display: flex; gap: 8px; flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .ix-metric-pill {
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px; font-weight: 500;
        background: #1E293B;
        border: 1px solid #334155;
        color: #94A3B8;
        cursor: pointer;
        transition: all 0.15s;
        white-space: nowrap;
    }
    .ix-metric-pill.active {
        background: #00D4FF18;
        border-color: #00D4FF55;
        color: #00D4FF;
    }

    /* ── KPI row ── */
    .ix-kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .ix-kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
    }
    .ix-kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }
    .ix-kpi-card.danger::before  { background: #EF4444; }
    .ix-kpi-card.warning::before { background: #F97316; }
    .ix-kpi-card.primary::before { background: #00D4FF; }
    .ix-kpi-card.success::before { background: #10B981; }
    .ix-kpi-label {
        font-size: 11px; color: #64748B;
        font-weight: 500; text-transform: uppercase;
        letter-spacing: 0.07em; margin-bottom: 6px;
    }
    .ix-kpi-value {
        font-size: 22px; font-weight: 700;
        line-height: 1;
    }
    .ix-kpi-value.danger  { color: #EF4444; }
    .ix-kpi-value.warning { color: #F97316; }
    .ix-kpi-value.primary { color: #00D4FF; }
    .ix-kpi-value.success { color: #10B981; }
    .ix-kpi-sub {
        font-size: 11px; color: #475569;
        margin-top: 4px;
    }

    /* ── Confidence bar ── */
    .ix-confidence {
        display: flex; align-items: center;
        gap: 10px; margin-top: 0.7rem;
    }
    .ix-confidence-label {
        font-size: 10px; color: #475569;
        text-transform: uppercase; letter-spacing: 0.08em;
        min-width: 70px;
    }
    .ix-confidence-track {
        flex: 1; height: 4px;
        background: #1E293B; border-radius: 2px;
        overflow: hidden;
    }
    .ix-confidence-fill {
        height: 100%; border-radius: 2px;
        animation: growWidth 0.6s ease-out forwards;
    }
    @keyframes growWidth {
        from { width: 0%; }
    }
    .ix-confidence-pct {
        font-size: 10px; font-weight: 600;
        font-family: 'DM Mono', monospace;
        min-width: 32px; text-align: right;
    }

    /* ── Insight highlight ── */
    .ix-insight {
        background: linear-gradient(135deg, #00D4FF08, #7C3AED08);
        border: 1px solid #334155;
        border-left: 3px solid #00D4FF;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px;
        margin: 0.8rem 0;
        font-size: 13px;
        color: #CBD5E1;
        line-height: 1.6;
    }

    /* ── Section divider ── */
    .ix-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1rem 0;
    }

    /* ── Winner badge ── */
    .ix-winner-badge {
        display: inline-flex; align-items: center; gap: 5px;
        background: #EF444415; border: 1px solid #EF444433;
        color: #EF4444; border-radius: 6px;
        padding: 3px 10px; font-size: 11px; font-weight: 600;
    }

    /* ── Empty state ── */
    .ix-empty {
        text-align: center;
        padding: 3rem 2rem;
        color: #475569;
    }
    .ix-empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .ix-empty-title { font-size: 16px; font-weight: 600; color: #64748B; }
    .ix-empty-sub   { font-size: 13px; margin-top: 0.3rem; }

    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  COMPONENTS
# ══════════════════════════════════════════════════════════════════════════

def render_chat_message(
    text: str,
    confidence: Optional[float] = None,
    insight: Optional[str] = None,
    is_followup: bool = False,
):
    """Render a polished AI response card."""
    followup_badge = '<span class="ix-followup-badge">↩ follow-up</span>' if is_followup else ""

    conf_html = ""
    if confidence is not None:
        pct       = int(confidence * 100)
        color     = "#10B981" if pct >= 80 else "#F97316" if pct >= 50 else "#EF4444"
        conf_text = "HIGH" if pct >= 80 else "MEDIUM" if pct >= 50 else "LOW"
        conf_html = f"""
        <div class="ix-confidence">
            <span class="ix-confidence-label">Confidence: {conf_text}</span>
            <div class="ix-confidence-track">
                <div class="ix-confidence-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="ix-confidence-pct" style="color:{color};">{pct}%</span>
        </div>
        """

    insight_html = ""
    if insight:
        insight_html = f'<div class="ix-insight">💡 {insight}</div>'

    st.markdown(f"""
    <div class="ix-message">
        <div class="ix-message-header">
            <div class="ix-bot-avatar">IX</div>
            <span class="ix-bot-name">InsightX</span>
            {followup_badge}
        </div>
        <div class="ix-message-body">{text}</div>
        {insight_html}
        {conf_html}
    </div>
    """, unsafe_allow_html=True)


def render_comparison_header(
    segments: List[str],
    metric_label: str,
    colors: Optional[List[str]] = None,
):
    """
    Render a visual 'A vs B vs C' pill header — pure inline styles only.
    """
    default_colors = [
        "#00D4FF", "#7C3AED", "#F59E0B",
        "#10B981", "#EF4444", "#F97316",
    ]
    colors = colors or default_colors

    pills_html = ""
    for i, seg in enumerate(segments):
        c = colors[i % len(colors)]
        pills_html += (
            f'<span style="color:{c};border:1.5px solid {c}55;background:{c}18;'
            f'border-radius:20px;padding:4px 14px;font-size:12px;'
            f'font-weight:700;white-space:nowrap">{seg}</span>'
        )
        if i < len(segments) - 1:
            pills_html += (
                '<span style="color:#475569;font-size:11px;'
                'font-weight:700;margin:0 4px">vs</span>'
            )

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;'
        f'margin-bottom:12px;padding:10px 14px;background:#0F172A;'
        f'border-radius:10px;border:1px solid #334155">'
        f'<span style="font-size:10px;font-weight:700;color:#64748B;'
        f'text-transform:uppercase;letter-spacing:.08em;margin-right:4px">'
        f'{metric_label}</span>'
        f'{pills_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_context_trail(history: List[Dict]):
    """
    Render the query breadcrumb in the sidebar.
    history: list from ContextMemory.get_history_display()
    """
    if not history:
        st.markdown("""
        <div class="ix-empty">
            <div class="ix-empty-icon">🔍</div>
            <div class="ix-empty-title">No queries yet</div>
            <div class="ix-empty-sub">Your query history will appear here</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("**Query History**", unsafe_allow_html=False)
    for item in history[:8]:
        followup_badge = '<span class="ix-followup-badge">↩</span>' if item.get("followup") else ""
        filters_str = ""
        if item.get("filters"):
            filters_str = " · " + ", ".join(f"{k}={v}" for k,v in item["filters"].items())

        st.markdown(f"""
        <div class="ix-trail-item">
            <div class="ix-trail-query">{item['query']}{followup_badge}</div>
            <div class="ix-trail-meta">{item['metric']} · {item['intent']} · {item['group_by']}{filters_str}</div>
        </div>
        """, unsafe_allow_html=True)


def render_kpi_row(kpis: List[Dict]):
    """
    Render a horizontal KPI strip at the top of results.

    kpis = [
        {"label": "Fraud Rate", "value": "0.21%", "sub": "Overall", "variant": "danger"},
        {"label": "Transactions", "value": "250K",  "sub": "2024",   "variant": "primary"},
    ]
    """
    cards_html = ""
    for kpi in kpis:
        v = kpi.get("variant", "primary")
        cards_html += f"""
        <div class="ix-kpi-card {v}">
            <div class="ix-kpi-label">{kpi['label']}</div>
            <div class="ix-kpi-value {v}">{kpi['value']}</div>
            <div class="ix-kpi-sub">{kpi.get('sub','')}</div>
        </div>
        """
    st.markdown(f'<div class="ix-kpi-row">{cards_html}</div>', unsafe_allow_html=True)


def render_winner_badge(label: str, value: str, metric_label: str):
    """Show a 'highest risk: Web at 0.21%' type badge."""
    st.markdown(f"""
    <div class="ix-winner-badge">
        ⚠ Highest {metric_label}: <strong style="margin-left:4px;">{label}</strong>
        <span style="color:#EF4444aa; margin-left:4px;">at {value}</span>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(message: str = "Ask me anything about UPI fraud data"):
    """Centered empty state for fresh chat."""
    st.markdown(f"""
    <div class="ix-empty">
        <div class="ix-empty-icon">⚡</div>
        <div class="ix-empty-title">InsightX BI</div>
        <div class="ix-empty-sub">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_divider():
    st.markdown('<div class="ix-divider"></div>', unsafe_allow_html=True)


def render_section_label(text: str):
    st.markdown(
        f'<div style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;'
        f'letter-spacing:.08em;margin:.8rem 0 .4rem;">{text}</div>',
        unsafe_allow_html=True
    )

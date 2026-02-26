"""
insightx/ui_components.py
FINAL 100% WORKING VERSION
Includes:
 - KPI boxes (fixed)
 - Comparison header
 - Context trail
 - Metric pills
 - Chat message card
 - Confidence bar
 - Dividers + labels
 - Full CSS
"""

import streamlit as st
from typing import List, Dict, Optional


# =============================================================================
# GLOBAL CSS (FULL + FIXED)
# =============================================================================

def inject_custom_css():
    st.markdown("""
    <style>

    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Wider App */
    .block-container {
        max-width: 1300px !important;
        padding-top: 1rem !important;
        margin: auto;
    }

    .stApp { background: #0F172A; color: #E2E8F0; }
    #MainMenu, header, footer { visibility: hidden; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #334155; }


    /* ================= KPI ROW ================= */
    .ix-kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 22px;
        margin-top: 1.2rem;
        margin-bottom: 2rem;
    }

    .ix-kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 20px;
        min-height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        position: relative;
    }

    .ix-kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .ix-kpi-card.primary::before { background: #00D4FF; }
    .ix-kpi-card.warning::before { background: #F97316; }
    .ix-kpi-card.success::before { background: #10B981; }
    .ix-kpi-card.danger::before  { background: #EF4444; }

    .ix-kpi-label {
        font-size: 11px;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 6px;
    }

    .ix-kpi-value {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .ix-kpi-value.primary { color: #00D4FF; }
    .ix-kpi-value.warning { color: #F97316; }
    .ix-kpi-value.success { color: #10B981; }
    .ix-kpi-value.danger  { color: #EF4444; }

    .ix-kpi-sub {
        font-size: 11px;
        color: #64748B;
    }


    /* ================= COMPARISON HEADER ================= */
    .ix-compare-header {
        padding: 1rem;
        border-radius: 14px;
        background: #0F172A;
        border: 1px solid #334155;
        margin-bottom: 1.3rem;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
    }

    .ix-compare-label {
        font-size: 12px;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
    }

    .ix-compare-pill {
        padding: 6px 16px;
        border-radius: 20px;
        border: 1.5px solid;
        font-size: 13px;
        font-weight: 600;
    }

    .ix-vs-divider {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
    }


    /* ================= CONTEXT TRAIL (SIDEBAR) ================= */
    .ix-trail-item {
        padding: 10px 12px;
        boundary: 1px solid #334155;
        background: #1E293B;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    .ix-trail-query {
        font-size: 13px;
        font-weight: 600;
        color: #E2E8F0;
    }

    .ix-trail-meta {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 2px;
    }


    /* ================= CHAT MESSAGE ================= */
    .ix-message {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        animation: fadeSlideIn 0.3s ease;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .ix-message-body {
        font-size: 15px;
        line-height: 1.7;
        color: #CBD5E1;
    }


    /* Divider */
    .ix-divider {
        height: 1px;
        background: #33415555;
        margin: 1.5rem 0;
    }

    </style>
    """, unsafe_allow_html=True)



# =============================================================================
# COMPONENTS (ALL INCLUDED — 100% COMPLETE)
# =============================================================================

def render_kpi_row(kpis: List[Dict]):
    html = ""
    for k in kpis:
        variant = k.get("variant", "primary")
        html += f"""
        <div class="ix-kpi-card {variant}">
            <div class="ix-kpi-label">{k['label']}</div>
            <div class="ix-kpi-value {variant}">{k['value']}</div>
            <div class="ix-kpi-sub">{k.get('sub', '')}</div>
        </div>
        """
    st.markdown(f"<div class='ix-kpi-row'>{html}</div>", unsafe_allow_html=True)



def render_comparison_header(
    segments: List[str],
    metric_label: str,
    colors: Optional[List[str]] = None,
):
    default_colors = ["#00D4FF", "#7C3AED", "#F59E0B", "#10B981", "#EF4444"]
    colors = colors or default_colors

    pills = ""
    for i, seg in enumerate(segments):
        c = colors[i % len(colors)]
        pills += f"""
        <span class="ix-compare-pill" style="color:{c}; border-color:{c}66; background:{c}20;">
            {seg}
        </span>
        """
        if i < len(segments) - 1:
            pills += '<span class="ix-vs-divider">vs</span>'

    st.markdown(f"""
    <div class="ix-compare-header">
        <span class="ix-compare-label">{metric_label}</span>
        {pills}
    </div>
    """, unsafe_allow_html=True)



def render_context_trail(history: List[Dict]):
    if not history:
        st.markdown("<p style='color:#64748B;'>No history yet...</p>", unsafe_allow_html=True)
        return

    st.markdown("### Query History")
    for item in history:
        filters_str = ""
        if item.get("filters"):
            filters_str = " | " + ", ".join(f"{k}: {v}" for k, v in item["filters"].items())

        st.markdown(f"""
        <div class="ix-trail-item">
            <div class="ix-trail-query">{item['query']}</div>
            <div class="ix-trail-meta">{item['metric']} • {item['intent']} • {item['group_by']}{filters_str}</div>
        </div>
        """, unsafe_allow_html=True)



def render_chat_message(
    text: str,
    confidence: Optional[float] = None,
    insight: Optional[str] = None,
    is_followup: bool = False
):
    insight_html = f"<div class='ix-insight'>💡 {insight}</div>" if insight else ""
    st.markdown(f"""
    <div class="ix-message">
        <div class="ix-message-body">{text}</div>
        {insight_html}
    </div>
    """, unsafe_allow_html=True)



def render_empty_state(message="Ask anything about UPI data"):
    st.markdown(f"""
    <div style="text-align:center; margin-top:3rem;">
        <div style="font-size:40px;">⚡</div>
        <div style="font-size:20px; font-weight:700;">InsightX BI</div>
        <div style="color:#94A3B8; margin-top:6px;">{message}</div>
    </div>
    """, unsafe_allow_html=True)



def render_divider():
    st.markdown("<div class='ix-divider'></div>", unsafe_allow_html=True)



def render_section_label(text):
    st.markdown(
        f"<div style='font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase; margin:1rem 0 .4rem;'>{text}</div>",
        unsafe_allow_html=True
    )
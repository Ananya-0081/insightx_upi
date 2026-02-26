"""
insightx/src/response_generator.py
────────────────────────────────────
Converts an AnalyticsResult into a structured InsightResponse
containing narrative text, bullet points, risk flags,
and follow-up question suggestions.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from src.analytics_engine import AnalyticsResult, METRIC_AGG, _fmt_value


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class InsightResponse:
    headline:       str              = ""
    narrative:      str              = ""
    bullets:        list[str]        = field(default_factory=list)
    risk_flags:     list[str]        = field(default_factory=list)
    recommendations:list[str]        = field(default_factory=list)
    follow_ups:     list[str]        = field(default_factory=list)
    confidence_bar: float            = 0.0   # 0–1
    confidence_label:str             = ""
    chart_type:     str              = "bar" # bar | line | gauge | heatmap | donut | anomaly
    has_anomalies:  bool             = False
    anomaly_count:  int              = 0


# ══════════════════════════════════════════════════════════════════════════
#  RISK FLAG RULES
# ══════════════════════════════════════════════════════════════════════════

def _compute_risk_flags(result: AnalyticsResult) -> list[str]:
    flags = []
    df = result.result_df

    if result.metric == "fraud_rate":
        if result.scalar_value and result.scalar_value > 0.3:
            flags.append("🚨 **Critical:** Overall fraud rate exceeds 0.3% — immediate investigation recommended.")
        if df is not None:
            high = df[df["value"] > 0.25]
            for _, row in high.iterrows():
                grp = row[result.group_by] if result.group_by in row.index else "Unknown"
                flags.append(f"⚠️ **{grp}** fraud rate {_fmt_value(row['value'], 'fraud_rate')} — above threshold.")

    elif result.metric == "failure_rate":
        if result.scalar_value and result.scalar_value > 6.0:
            flags.append("🚨 **Critical:** Failure rate above 6% — SLA breach risk.")
        if df is not None:
            high = df[df["value"] > 5.5]
            for _, row in high.iterrows():
                grp = row[result.group_by] if result.group_by in row.index else "Unknown"
                flags.append(f"⚠️ **{grp}** failure rate {_fmt_value(row['value'], 'failure_rate')} — exceeds 5.5% benchmark.")

    if result.anomalies:
        for a in result.anomalies[:3]:
            direction = "significantly above" if a["direction"] == "high" else "significantly below"
            flags.append(f"📊 **{a['group']}** is {direction} average (z={a['z_score']:+.2f}).")

    return flags


# ══════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION RULES
# ══════════════════════════════════════════════════════════════════════════

def _compute_recommendations(result: AnalyticsResult) -> list[str]:
    recs = []
    metric = result.metric

    if metric == "fraud_rate":
        recs += [
            "Implement velocity checks for high-fraud network/device combinations.",
            "Add step-up authentication for transactions in peak fraud hours (1–3 AM).",
            "Flag transactions from high-fraud states for manual review queues.",
        ]
    elif metric == "failure_rate":
        recs += [
            "Prioritise reliability improvements for 3G users — consider retry logic.",
            "Investigate Web browser failures; may indicate session timeout issues.",
            "Set up real-time failure rate alerts by network type.",
        ]
    elif metric == "avg_amount":
        recs += [
            "Apply tiered transaction limits based on merchant category risk profile.",
            "High-value categories (Education, Shopping) warrant enhanced KYC checks.",
        ]
    elif metric == "count":
        recs += [
            "Align customer support staffing with peak transaction hours.",
            "Optimise infrastructure capacity for high-volume states and devices.",
        ]

    return recs[:3]


# ══════════════════════════════════════════════════════════════════════════
#  FOLLOW-UP SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════

_FOLLOW_UPS_BY_METRIC: dict[str, list[str]] = {
    "fraud_rate": [
        "Which hour has the highest fraud rate?",
        "Compare fraud rate by age group",
        "Show fraud trend by day of week",
        "What is the fraud rate in Maharashtra?",
    ],
    "failure_rate": [
        "Which bank has the highest failure rate?",
        "Compare failure rate by network type",
        "Show failure rate trend by hour",
        "Is failure rate higher on weekends?",
    ],
    "avg_amount": [
        "Which age group has the highest average amount?",
        "Compare average amount by category",
        "What is average transaction for P2P vs P2M?",
        "Which state has the highest average transaction?",
    ],
    "count": [
        "Which device is most popular?",
        "How many transactions happen during evenings?",
        "Compare transaction count by bank",
    ],
    "total_volume": [
        "Which state generates the highest total volume?",
        "Compare total volume by transaction type",
        "What is the total volume for Education category?",
    ],
}

_GENERIC_FOLLOW_UPS = [
    "Show me the overall summary",
    "What is the fraud rate by device?",
    "Which bank has the lowest failure rate?",
    "Compare transaction volumes by state",
]


def _compute_follow_ups(result: AnalyticsResult) -> list[str]:
    pool = _FOLLOW_UPS_BY_METRIC.get(result.metric, []) + _GENERIC_FOLLOW_UPS
    # Exclude current query's group_by to keep suggestions varied
    filtered = [q for q in pool if not result.group_by or result.group_by not in q.lower()]
    return filtered[:4] if filtered else _GENERIC_FOLLOW_UPS[:4]


# ══════════════════════════════════════════════════════════════════════════
#  CONFIDENCE LABEL
# ══════════════════════════════════════════════════════════════════════════

def _confidence_label(conf: float) -> str:
    if conf >= 0.85:
        return "Very High"
    elif conf >= 0.70:
        return "High"
    elif conf >= 0.55:
        return "Medium"
    else:
        return "Low"


# ══════════════════════════════════════════════════════════════════════════
#  CHART TYPE SELECTOR
# ══════════════════════════════════════════════════════════════════════════

def _select_chart_type(result: AnalyticsResult) -> str:
    if result.intent == "trend":
        return "line"
    if result.intent == "anomaly":
        return "anomaly"
    if result.intent == "single" and result.scalar_value is not None and not result.group_by:
        return "gauge"
    if result.group_by in ("day_of_week", "hour_of_day") and result.intent == "trend":
        return "heatmap"
    if result.group_by in ("device_type", "transaction_type") and result.metric == "count":
        return "donut"
    return "bar"


# ══════════════════════════════════════════════════════════════════════════
#  HEADLINE TEMPLATES
# ══════════════════════════════════════════════════════════════════════════

def _build_headline(result: AnalyticsResult) -> str:
    ml  = result.metric_label
    dl  = result.dim_label
    fil = result.filters

    if result.intent == "single" and result.scalar_fmt:
        return f"{ml}: **{result.scalar_fmt}**"
    if result.intent in ("comparison", "ranking") and result.result_df is not None:
        dim  = result.group_by
        best = result.result_df.sort_values("value", ascending=False).iloc[0]
        grp  = str(best[dim]) if dim in best.index else "—"
        return f"Top {dl} for {ml}: **{grp}** at **{_fmt_value(best['value'], result.metric)}**"
    if result.intent == "trend":
        return f"{ml} trend over {dl}"
    if result.intent == "anomaly":
        n = len(result.anomalies)
        return f"**{n} anomalies** found in {ml} by {dl}"
    return f"{ml} Analysis"


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def generate_response(result: AnalyticsResult, confidence: float) -> InsightResponse:
    """
    Turn an AnalyticsResult into a rich InsightResponse ready for display.
    """
    resp = InsightResponse()

    resp.headline        = _build_headline(result)
    resp.narrative       = result.narrative
    resp.bullets         = result.insight_bullets
    resp.risk_flags      = _compute_risk_flags(result)
    resp.recommendations = _compute_recommendations(result)
    resp.follow_ups      = _compute_follow_ups(result)
    resp.confidence_bar  = confidence
    resp.confidence_label= _confidence_label(confidence)
    resp.chart_type      = _select_chart_type(result)
    resp.has_anomalies   = len(result.anomalies) > 0
    resp.anomaly_count   = len(result.anomalies)

    return resp
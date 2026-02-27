"""
insightx/src/response_generator.py
────────────────────────────────────
Converts an AnalyticsResult into a structured InsightResponse.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from src.analytics_engine import AnalyticsResult, METRIC_AGG, _fmt_value, DIM_LABELS


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class InsightResponse:
    headline:        str       = ""
    narrative:       str       = ""
    bullets:         list[str] = field(default_factory=list)
    risk_flags:      list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    follow_ups:      list[str] = field(default_factory=list)
    confidence_bar:  float     = 0.0
    confidence_label:str       = ""
    chart_type:      str       = "bar"
    has_anomalies:   bool      = False
    anomaly_count:   int       = 0


# ══════════════════════════════════════════════════════════════════════════
#  RISK FLAGS
# ══════════════════════════════════════════════════════════════════════════

def _compute_risk_flags(result: AnalyticsResult) -> list[str]:
    flags = []
    df    = result.result_df

    if result.metric == "fraud_rate":
        if result.scalar_value and result.scalar_value > 0.3:
            flags.append("🚨 Critical: Overall fraud rate exceeds 0.3% — immediate investigation recommended.")
        if df is not None and result.group_by and result.group_by in df.columns:
            for _, row in df[df["value"] > 0.25].iterrows():
                grp = row[result.group_by]
                flags.append(f"⚠ {grp} fraud rate {_fmt_value(row['value'], 'fraud_rate')} — above threshold.")

    elif result.metric == "failure_rate":
        if result.scalar_value and result.scalar_value > 6.0:
            flags.append("🚨 Critical: Failure rate above 6% — SLA breach risk.")
        if df is not None and result.group_by and result.group_by in df.columns:
            for _, row in df[df["value"] > 5.5].iterrows():
                grp = row[result.group_by]
                flags.append(f"⚠ {grp} failure rate {_fmt_value(row['value'], 'failure_rate')} — exceeds 5.5% benchmark.")

    if result.anomalies:
        for a in result.anomalies[:3]:
            direction = "significantly above" if a["direction"] == "high" else "significantly below"
            flags.append(f"📊 {a['group']} is {direction} average (z={a['z_score']:+.2f}).")

    return flags


# ══════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════

def _compute_recommendations(result: AnalyticsResult) -> list[str]:
    recs = {
        "fraud_rate": [
            "Implement velocity checks for high-fraud network/device combinations.",
            "Add step-up authentication for transactions in peak fraud hours (1–3 AM).",
            "Flag transactions from high-fraud states for manual review queues.",
        ],
        "failure_rate": [
            "Prioritise reliability improvements for 3G users — consider retry logic.",
            "Investigate Web browser failures — may indicate session timeout issues.",
            "Set up real-time failure rate alerts by network type.",
        ],
        "avg_amount": [
            "Apply tiered transaction limits based on merchant category risk profile.",
            "High-value categories (Education, Shopping) warrant enhanced KYC checks.",
        ],
        "count": [
            "Align customer support staffing with peak transaction hours.",
            "Optimise infrastructure capacity for high-volume states and devices.",
        ],
        "total_volume": [
            "Prioritise payment infrastructure in high-volume states.",
            "Review settlement processes for top-volume merchant categories.",
        ],
    }
    return recs.get(result.metric, [])[:2]


# ══════════════════════════════════════════════════════════════════════════
#  CONTEXT-AWARE FOLLOW-UP SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════

# States and banks for drill-down suggestions
_STATES = ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Gujarat",
           "Rajasthan", "West Bengal", "Telangana", "Andhra Pradesh", "Uttar Pradesh"]
_BANKS  = ["HDFC", "SBI", "ICICI", "Axis", "Kotak", "PNB", "Yes Bank", "IndusInd"]

def _compute_follow_ups(result: AnalyticsResult) -> list[str]:
    metric  = result.metric
    group   = result.group_by
    filters = result.filters
    df      = result.result_df
    compare = result.compare

    suggestions = []

    # ── 1. Drill down into top result ─────────────────────────────────────
    if df is not None and not df.empty and group and group in df.columns:
        top_val = str(df.sort_values("value", ascending=False).iloc[0][group])

        # Suggest a different state if already filtered by one state
        if group == "sender_state" or "sender_state" in filters:
            current_state = filters.get("sender_state", top_val)
            others = [s for s in _STATES if s != current_state][:2]
            for s in others:
                suggestions.append(f"What about {s}?")

        # Suggest a different bank
        elif group == "sender_bank" or "sender_bank" in filters:
            current_bank = filters.get("sender_bank", top_val)
            others = [b for b in _BANKS if b != current_bank][:2]
            for b in others:
                suggestions.append(f"What about {b}?")

        # Suggest filtering by top segment
        elif group in ("device_type", "network_type", "transaction_type"):
            suggestions.append(f"Show only {top_val}")

    # ── 2. Cross-dimension drill ──────────────────────────────────────────
    dim_suggestions = {
        "device_type":       [f"Compare {metric.replace('_',' ')} by network type",
                              f"Show {metric.replace('_',' ')} trend by hour"],
        "network_type":      [f"Compare {metric.replace('_',' ')} by device",
                              "Which state has the highest failure rate?"],
        "sender_state":      [f"Compare {metric.replace('_',' ')} by bank",
                              f"Compare {metric.replace('_',' ')} by device"],
        "sender_bank":       [f"Compare {metric.replace('_',' ')} by state",
                              f"Compare {metric.replace('_',' ')} by network type"],
        "merchant_category": [f"Which state has the highest {metric.replace('_',' ')}?",
                              f"Show {metric.replace('_',' ')} trend by day"],
        "sender_age_group":  [f"Compare {metric.replace('_',' ')} by device",
                              f"Compare {metric.replace('_',' ')} by state"],
        "hour_of_day":       ["Compare failure rate by network type",
                              "Show fraud rate by day of week"],
        "day_of_week":       ["Show fraud trend by hour",
                              "Compare failure rate by bank"],
    }
    if group in dim_suggestions:
        suggestions += dim_suggestions[group]

    # ── 3. Metric switch suggestions ──────────────────────────────────────
    metric_alts = {
        "fraud_rate":   "failure rate",
        "failure_rate": "fraud rate",
        "avg_amount":   "transaction count",
        "count":        "total volume",
        "total_volume": "average amount",
    }
    alt = metric_alts.get(metric)
    if alt and group:
        dim_lbl = DIM_LABELS.get(group, group)
        suggestions.append(f"Show {alt} by {dim_lbl.lower()}")

    # ── 4. Always include at least one safe generic ───────────────────────
    generics = [
        "Show me the overall summary",
        "Which bank has the highest failure rate?",
        "Compare fraud rate by device",
        "Top 5 states by transaction volume",
    ]
    for g in generics:
        if len(suggestions) >= 4:
            break
        if g not in suggestions:
            suggestions.append(g)

    return suggestions[:4]


# ══════════════════════════════════════════════════════════════════════════
#  CONFIDENCE LABEL
# ══════════════════════════════════════════════════════════════════════════

def _confidence_label(conf: float) -> str:
    if conf >= 0.85: return "Very High"
    if conf >= 0.70: return "High"
    if conf >= 0.55: return "Medium"
    return "Low"


# ══════════════════════════════════════════════════════════════════════════
#  CHART TYPE SELECTOR
# ══════════════════════════════════════════════════════════════════════════

def _select_chart_type(result: AnalyticsResult) -> str:
    compare = result.compare
    if compare and len(compare) >= 2:
        return "comparison"
    if result.intent == "trend":
        return "line"
    if result.intent == "anomaly":
        return "anomaly"
    if result.intent == "single" and result.scalar_value is not None and not result.group_by:
        return "gauge"
    if result.group_by in ("device_type", "transaction_type") and result.metric == "count":
        return "donut"
    return "bar"


# ══════════════════════════════════════════════════════════════════════════
#  HEADLINE (clean — no raw asterisks)
# ══════════════════════════════════════════════════════════════════════════

def _build_headline(result: AnalyticsResult) -> str:
    ml  = result.metric_label
    dl  = result.dim_label
    compare = result.compare

    # Comparison: "HDFC vs SBI — Fraud Rate"
    if compare and len(compare) >= 2:
        vs_str = " vs ".join(compare)
        return f"{vs_str} — {ml}"

    if result.intent == "single" and result.scalar_fmt:
        return f"{ml}: {result.scalar_fmt}"

    if result.intent in ("comparison", "ranking") and result.result_df is not None:
        dim  = result.group_by
        best = result.result_df.sort_values("value", ascending=False).iloc[0]
        grp  = str(best[dim]) if dim and dim in best.index else "—"
        val  = _fmt_value(best["value"], result.metric)
        return f"Top {dl} for {ml}: {grp} at {val}"

    if result.intent == "trend":
        return f"{ml} Trend over {dl}"

    if result.intent == "anomaly":
        return f"{len(result.anomalies)} Anomalies in {ml} by {dl}"

    return f"{ml} Analysis"


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def generate_response(result: AnalyticsResult, confidence: float) -> InsightResponse:
    resp = InsightResponse()
    resp.headline         = _build_headline(result)
    resp.narrative        = result.narrative
    resp.bullets          = result.insight_bullets
    resp.risk_flags       = _compute_risk_flags(result)
    resp.recommendations  = _compute_recommendations(result)
    resp.follow_ups       = _compute_follow_ups(result)
    resp.confidence_bar   = confidence
    resp.confidence_label = _confidence_label(confidence)
    resp.chart_type       = _select_chart_type(result)
    resp.has_anomalies    = len(result.anomalies) > 0
    resp.anomaly_count    = len(result.anomalies)
    return resp

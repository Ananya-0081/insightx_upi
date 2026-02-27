"""
insightx/src/analytics_engine.py
─────────────────────────────────
Receives an AnalyticalQuery and a pandas DataFrame,
executes the correct aggregation, and returns an AnalyticsResult.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from src.nlp_engine import AnalyticalQuery

# ══════════════════════════════════════════════════════════════════════════
#  METRIC CONFIG
# ══════════════════════════════════════════════════════════════════════════

METRIC_AGG: dict[str, dict] = {
    "fraud_rate":   {"col": "is_fraud",  "agg": "mean", "scale": 100, "fmt": "{:.2f}%",  "label": "Fraud Rate (%)"},
    "failure_rate": {"col": "is_failed", "agg": "mean", "scale": 100, "fmt": "{:.2f}%",  "label": "Failure Rate (%)"},
    "avg_amount":   {"col": "amount_inr","agg": "mean", "scale": 1,   "fmt": "₹{:,.2f}", "label": "Avg Amount (₹)"},
    "count":        {"col": None,        "agg": "count","scale": 1,   "fmt": "{:,.0f}",  "label": "Transaction Count"},
    "total_volume": {"col": "amount_inr","agg": "sum",  "scale": 1,   "fmt": "₹{:,.0f}","label": "Total Volume (₹)"},
}

DIM_LABELS: dict[str, str] = {
    "device_type":       "Device",
    "network_type":      "Network",
    "sender_state":      "State",
    "sender_bank":       "Bank",
    "merchant_category": "Category",
    "sender_age_group":  "Age Group",
    "day_of_week":       "Day of Week",
    "hour_of_day":       "Hour of Day",
    "transaction_type":  "Transaction Type",
    "month":             "Month",
    "quarter":           "Quarter",
}


# ══════════════════════════════════════════════════════════════════════════
#  RESULT DATA CLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalyticsResult:
    success:         bool                  = True
    error:           Optional[str]         = None
    intent:          str                   = ""
    metric:          str                   = ""
    metric_label:    str                   = ""
    group_by:        Optional[str]         = None
    dim_label:       str                   = ""
    filters:         dict                  = field(default_factory=dict)
    compare:         list                  = field(default_factory=list)

    scalar_value:    Optional[float]       = None
    scalar_fmt:      Optional[str]         = None
    result_df:       Optional[pd.DataFrame]= None

    total_rows:      int                   = 0
    filtered_rows:   int                   = 0

    narrative:       str                   = ""
    insight_bullets: list[str]             = field(default_factory=list)
    anomalies:       list[dict]            = field(default_factory=list)
    query_json:      str                   = ""


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _fmt_value(val: float, metric: str) -> str:
    return METRIC_AGG[metric]["fmt"].format(val)


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col == "month_num":
            mask &= (df["timestamp"].dt.month == val)
        elif col == "is_weekend":
            mask &= (df["is_weekend"] == val)
        elif col in df.columns:
            mask &= (df[col] == val)
    return df[mask].copy()


def _apply_compare_filter(df: pd.DataFrame, group_col: str, compare: list) -> pd.DataFrame:
    """When compare values are set, restrict df to only those segments."""
    if not compare or not group_col or group_col not in df.columns:
        return df
    return df[df[group_col].astype(str).isin([str(v) for v in compare])].copy()


def _compute_metric(df: pd.DataFrame, metric: str, group_col: Optional[str] = None) -> pd.DataFrame:
    cfg      = METRIC_AGG[metric]
    agg_col  = cfg["col"]
    agg_type = cfg["agg"]
    scale    = cfg["scale"]

    if group_col:
        grp    = df.groupby(group_col, observed=True)
        counts = grp.size().rename("count")
        if agg_type == "count":
            values = counts.rename("value")
        elif agg_type == "mean":
            values = (grp[agg_col].mean() * scale).rename("value")
        elif agg_type == "sum":
            values = grp[agg_col].sum().rename("value")
        result = pd.concat([values, counts], axis=1).reset_index()
        result.columns = [group_col, "value", "count"]
        return result.dropna(subset=["value"])
    else:
        if agg_type == "count":
            return pd.DataFrame({"value": [len(df)], "count": [len(df)]})
        elif agg_type == "mean":
            return pd.DataFrame({"value": [df[agg_col].mean() * scale], "count": [len(df)]})
        elif agg_type == "sum":
            return pd.DataFrame({"value": [df[agg_col].sum()], "count": [len(df)]})


def _build_narrative(
    result_df: pd.DataFrame,
    query: AnalyticalQuery,
    metric_label: str,
    dim_label: str,
) -> tuple[str, list[str]]:
    metric = query.metric
    group  = query.group_by

    if result_df is None or result_df.empty:
        return "No data matched the filters.", []

    sorted_df  = result_df.sort_values("value", ascending=False)
    top        = sorted_df.iloc[0]
    bottom     = sorted_df.iloc[-1]
    top_fmt    = _fmt_value(top["value"],    metric)
    bottom_fmt = _fmt_value(bottom["value"], metric)
    top_name   = str(top[group])    if group and group in top.index    else "Overall"
    bottom_name= str(bottom[group]) if group and group in bottom.index else "Overall"

    bullets = []

    if metric == "fraud_rate":
        bullets.append(f"{top_name} has the highest fraud rate at {top_fmt} — a risk priority.")
        bullets.append(f"{bottom_name} is the safest at {bottom_fmt}.")
        if group == "network_type":
            bullets.append("Network-level fraud variance may indicate device/authentication differences.")
        elif group == "sender_state":
            bullets.append("Regional fraud variance may reflect UPI adoption maturity.")
        elif group == "hour_of_day":
            bullets.append("Late-night hours (1–3 AM) typically show elevated fraud patterns.")
    elif metric == "failure_rate":
        bullets.append(f"{top_name} has the highest failure rate at {top_fmt} — indicating reliability issues.")
        bullets.append(f"{bottom_name} is the most reliable at {bottom_fmt}.")
        if group == "network_type":
            bullets.append("3G shows structurally higher failure rates — users on slow networks face worse UX.")
    elif metric == "avg_amount":
        bullets.append(f"{top_name} records the highest average transaction of {top_fmt}.")
        bullets.append(f"{bottom_name} has the lowest average at {bottom_fmt}.")
        diff = top["value"] - bottom["value"]
        bullets.append(f"Spread between highest and lowest: {_fmt_value(diff, metric)}.")
    elif metric == "count":
        total = result_df["value"].sum()
        share = top["value"] / total * 100 if total > 0 else 0
        bullets.append(f"{top_name} dominates with {_fmt_value(top['value'], metric)} transactions ({share:.1f}% share).")
        bullets.append(f"{bottom_name} is the least used with {_fmt_value(bottom['value'], metric)} transactions.")
    elif metric == "total_volume":
        bullets.append(f"{top_name} generates the highest total volume at {top_fmt}.")
        bullets.append(f"{bottom_name} contributes the least at {bottom_fmt}.")

    # Build filter context string for narrative
    filter_ctx = ""
    if query.filters:
        parts = [f"{k.replace('sender_','').replace('_',' ').title()}={v}" for k, v in query.filters.items()]
        filter_ctx = f" (filtered: {', '.join(parts)})"

    narrative = (
        f"{metric_label} analysis by {dim_label} across {len(result_df)} groups{filter_ctx}. "
        + " ".join(bullets)
    )
    return narrative, bullets


def _detect_anomalies(result_df: pd.DataFrame, group_col: str, metric: str, z_thresh: float = 2.0) -> list[dict]:
    if result_df is None or len(result_df) < 4:
        return []
    vals  = result_df["value"].values.astype(float)
    mean  = np.mean(vals)
    std   = np.std(vals)
    if std == 0:
        return []
    zscores = (vals - mean) / std
    anomalies = []
    for i, (_, row) in enumerate(result_df.iterrows()):
        z = zscores[i]
        if abs(z) >= z_thresh:
            anomalies.append({
                "group":     str(row[group_col]),
                "value":     float(row["value"]),
                "z_score":   round(float(z), 2),
                "direction": "high" if z > 0 else "low",
                "formatted": _fmt_value(row["value"], metric),
            })
    return sorted(anomalies, key=lambda x: abs(x["z_score"]), reverse=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

def execute_query(query: AnalyticalQuery, df: pd.DataFrame) -> AnalyticsResult:
    result = AnalyticsResult(
        intent       = query.intent,
        metric       = query.metric,
        metric_label = METRIC_AGG.get(query.metric, {}).get("label", query.metric),
        group_by     = query.group_by,
        dim_label    = DIM_LABELS.get(query.group_by or "", query.group_by or ""),
        filters      = query.filters,
        compare      = getattr(query, "compare", []),
        total_rows   = len(df),
        query_json   = query.to_json(),
    )

    if query.metric not in METRIC_AGG:
        result.success = False
        result.error   = f"Unknown metric '{query.metric}'."
        return result

    # ── Apply regular filters ─────────────────────────────────────────────
    try:
        fdf = _apply_filters(df, query.filters)
        result.filtered_rows = len(fdf)
    except Exception as e:
        result.success = False
        result.error   = f"Filter error: {e}"
        return result

    if len(fdf) == 0:
        result.success = False
        result.error   = "No transactions matched your filters. Try broadening the query."
        return result

    # ── Time-window filter ────────────────────────────────────────────────
    if query.time_window:
        tw = query.time_window
        if tw["type"] == "hour_range":
            fdf = fdf[(fdf["hour_of_day"] >= tw["min"]) & (fdf["hour_of_day"] <= tw["max"])]
        elif tw["type"] == "weekend":
            fdf = fdf[fdf["is_weekend"] == 1]
        elif tw["type"] == "weekday":
            fdf = fdf[fdf["is_weekend"] == 0]

    # ── Resolve group column ──────────────────────────────────────────────
    group_col = None
    if query.group_by:
        group_col = query.group_by if query.group_by in fdf.columns else None

    # ── Apply compare filter (restrict to named segments ONLY) ───────────
    compare_vals = getattr(query, "compare", [])
    if compare_vals and group_col:
        fdf = _apply_compare_filter(fdf, group_col, compare_vals)
        result.filtered_rows = len(fdf)
        if len(fdf) == 0:
            result.success = False
            result.error   = f"No data found for comparison segments: {compare_vals}"
            return result

    # ══════════════════════════════════════════════════════════════════════
    #  INTENT DISPATCH
    # ══════════════════════════════════════════════════════════════════════
    try:
        # ── SINGLE (scalar, no grouping) ──────────────────────────────────
        if query.intent == "single" and not group_col:
            scalar_df = _compute_metric(fdf, query.metric)
            val = float(scalar_df["value"].iloc[0])
            result.scalar_value = val
            result.scalar_fmt   = _fmt_value(val, query.metric)

            # Supporting breakdown
            top_dim = "merchant_category" if "merchant_category" in fdf.columns else "sender_state"
            ctx_df  = _compute_metric(fdf, query.metric, top_dim)
            ctx_df  = ctx_df.sort_values("value", ascending=False).head(5)
            result.result_df = ctx_df

            filter_ctx = ""
            if query.filters:
                parts = [f"{k.replace('sender_','').replace('_',' ').title()}={v}"
                         for k, v in query.filters.items()]
                filter_ctx = f" (filtered: {', '.join(parts)})"

            result.narrative = (
                f"Overall {result.metric_label} is {result.scalar_fmt} "
                f"across {result.filtered_rows:,} transactions{filter_ctx}."
            )
            result.insight_bullets = [
                f"Dataset covers {result.total_rows:,} transactions.",
                f"Filtered subset: {result.filtered_rows:,} transactions.",
                f"Computed {result.metric_label}: {result.scalar_fmt}",
            ]

        # ── COMPARISON / RANKING ──────────────────────────────────────────
        elif query.intent in ("comparison", "ranking", "single") and group_col:
            agg_df = _compute_metric(fdf, query.metric, group_col)
            agg_df = agg_df.sort_values(
                "value", ascending=query.sort_ascending
            ).head(query.top_n)

            result.result_df = agg_df
            narrative, bullets = _build_narrative(
                agg_df, query, result.metric_label, result.dim_label
            )
            result.narrative       = narrative
            result.insight_bullets = bullets
            result.anomalies       = _detect_anomalies(agg_df, group_col, query.metric)

        # ── TREND ─────────────────────────────────────────────────────────
        elif query.intent == "trend":
            time_col = group_col or "month"
            if time_col not in fdf.columns:
                time_col = "day_of_week"

            agg_df = _compute_metric(fdf, query.metric, time_col)

            if time_col == "day_of_week":
                day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                agg_df[time_col] = pd.Categorical(agg_df[time_col], categories=day_order, ordered=True)
                agg_df = agg_df.sort_values(time_col)
            elif time_col in ("month", "quarter", "hour_of_day"):
                agg_df = agg_df.sort_values(time_col)

            result.result_df = agg_df

            peak   = agg_df.loc[agg_df["value"].idxmax()]
            trough = agg_df.loc[agg_df["value"].idxmin()]
            result.narrative = (
                f"{result.metric_label} trend over {DIM_LABELS.get(time_col, time_col)}. "
                f"Peak: {peak[time_col]} at {_fmt_value(peak['value'], query.metric)}. "
                f"Trough: {trough[time_col]} at {_fmt_value(trough['value'], query.metric)}."
            )
            result.insight_bullets = [
                f"Peak: {peak[time_col]} ({_fmt_value(peak['value'], query.metric)})",
                f"Quietest: {trough[time_col]} ({_fmt_value(trough['value'], query.metric)})",
                f"Std deviation: {_fmt_value(float(agg_df['value'].std()), query.metric)}",
            ]

        # ── ANOMALY ───────────────────────────────────────────────────────
        elif query.intent == "anomaly":
            dim    = group_col or "merchant_category"
            agg_df = _compute_metric(fdf, query.metric, dim)
            anomalies = _detect_anomalies(agg_df, dim, query.metric, z_thresh=1.5)
            result.result_df = agg_df
            result.anomalies = anomalies
            if anomalies:
                top_a = anomalies[0]
                result.narrative = (
                    f"{len(anomalies)} anomalies detected in {result.metric_label} "
                    f"by {result.dim_label}. "
                    f"Strongest: {top_a['group']} at {top_a['formatted']} "
                    f"(z={top_a['z_score']:+.2f})."
                )
                result.insight_bullets = [
                    f"{a['group']}: {a['formatted']} (z={a['z_score']:+.2f}, {a['direction']})"
                    for a in anomalies[:5]
                ]
            else:
                result.narrative = "No significant anomalies detected — distribution appears normal."

        # ── FALLBACK ──────────────────────────────────────────────────────
        else:
            result.narrative = "Query parsed successfully but no dimension detected. Showing overall stats."
            scalar_df = _compute_metric(fdf, query.metric)
            val = float(scalar_df["value"].iloc[0])
            result.scalar_value = val
            result.scalar_fmt   = _fmt_value(val, query.metric)

    except Exception as e:
        result.success = False
        result.error   = f"Computation error: {str(e)}"
        return result

    return result

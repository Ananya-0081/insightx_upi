"""
insightx/src/analytics_engine.py
─────────────────────────────────
Receives an AnalyticalQuery and a pandas DataFrame,
executes the correct aggregation, and returns an AnalyticsResult.

Supported metrics
-----------------
  fraud_rate    – % of is_fraud == 1
  failure_rate  – % of is_failed == 1
  avg_amount    – mean of amount_inr
  count         – row count
  total_volume  – sum of amount_inr

Supported intents
-----------------
  single      → scalar result + narrative
  comparison  → grouped bar chart data
  trend       → time-series data
  ranking     → sorted top-N table
  anomaly     → z-score outlier detection
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any

from src.nlp_engine import AnalyticalQuery

# ══════════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

METRIC_AGG: dict[str, dict] = {
    "fraud_rate":   {"col": "is_fraud",  "agg": "mean", "scale": 100, "fmt": "{:.2f}%",  "label": "Fraud Rate (%)"},
    "failure_rate": {"col": "is_failed", "agg": "mean", "scale": 100, "fmt": "{:.2f}%",  "label": "Failure Rate (%)"},
    "avg_amount":   {"col": "amount_inr","agg": "mean", "scale": 1,   "fmt": "₹{:,.2f}", "label": "Avg Amount (₹)"},
    "count":        {"col": None,        "agg": "count","scale": 1,   "fmt": "{:,.0f}",  "label": "Transaction Count"},
    "total_volume": {"col": "amount_inr","agg": "sum",  "scale": 1,   "fmt": "₹{:,.0f}","label": "Total Volume (₹)"},
}

# Human-readable dimension labels
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

# ── column alias resolution (handles CSV quirks) ──────────────────────────
COL_ALIAS: dict[str, str] = {
    "transaction_type": "transaction_type",
}


# ══════════════════════════════════════════════════════════════════════════
#  RESULT DATA CLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalyticsResult:
    success:      bool             = True
    error:        Optional[str]    = None
    intent:       str              = ""
    metric:       str              = ""
    metric_label: str              = ""
    group_by:     Optional[str]    = None
    dim_label:    str              = ""
    filters:      dict             = field(default_factory=dict)

    # Scalar (single intent)
    scalar_value: Optional[float]  = None
    scalar_fmt:   Optional[str]    = None

    # Tabular (comparison / ranking / trend)
    result_df:    Optional[pd.DataFrame] = None

    # Overall stats for context
    total_rows:   int              = 0
    filtered_rows:int              = 0

    # Narrative explanation
    narrative:    str              = ""
    insight_bullets: list[str]     = field(default_factory=list)

    # Anomalies
    anomalies:    list[dict]       = field(default_factory=list)

    # Structured JSON representation of the query executed
    query_json:   str              = ""


# ══════════════════════════════════════════════════════════════════════════
#  FILTER APPLICATION
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def _compute_metric(df: pd.DataFrame, metric: str, group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Compute the requested metric, optionally grouped by group_col.
    Returns a DataFrame with columns [group_col (if any), 'value', 'count'].
    """
    cfg = METRIC_AGG[metric]
    agg_col  = cfg["col"]
    agg_type = cfg["agg"]
    scale    = cfg["scale"]

    if group_col:
        grp = df.groupby(group_col, observed=True)
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

    else:  # scalar
        if agg_type == "count":
            return pd.DataFrame({"value": [len(df)], "count": [len(df)]})
        elif agg_type == "mean":
            return pd.DataFrame({"value": [df[agg_col].mean() * scale], "count": [len(df)]})
        elif agg_type == "sum":
            return pd.DataFrame({"value": [df[agg_col].sum()], "count": [len(df)]})


def _fmt_value(val: float, metric: str) -> str:
    return METRIC_AGG[metric]["fmt"].format(val)


# ══════════════════════════════════════════════════════════════════════════
#  NARRATIVE GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def _build_narrative(result_df: pd.DataFrame, query: AnalyticalQuery, metric_label: str, dim_label: str) -> tuple[str, list[str]]:
    metric = query.metric
    group  = query.group_by

    if result_df is None or result_df.empty:
        return "No data matched the filters.", []

    sorted_df = result_df.sort_values("value", ascending=False)
    top    = sorted_df.iloc[0]
    bottom = sorted_df.iloc[-1]

    top_fmt    = _fmt_value(top["value"],    metric)
    bottom_fmt = _fmt_value(bottom["value"], metric)
    top_name   = top[group] if group else "Overall"
    bottom_name= bottom[group] if group else "Overall"

    bullets = []

    if metric == "fraud_rate":
        bullets.append(f"**{top_name}** has the highest fraud rate at **{top_fmt}** — a risk priority.")
        bullets.append(f"**{bottom_name}** is the safest at **{bottom_fmt}**.")
        if group == "sender_state":
            bullets.append("Regional fraud variance may reflect UPI adoption maturity and user awareness levels.")
        if group == "hour_of_day":
            bullets.append("Late-night hours (1–3 AM) show elevated fraud — consider transaction velocity limits.")

    elif metric == "failure_rate":
        bullets.append(f"**{top_name}** has the highest failure rate at **{top_fmt}** — indicating reliability issues.")
        bullets.append(f"**{bottom_name}** is the most reliable at **{bottom_fmt}**.")
        if group == "network_type":
            bullets.append("3G shows structurally higher failure rates — users on slow networks face worse UX.")

    elif metric == "avg_amount":
        bullets.append(f"**{top_name}** records the highest average transaction of **{top_fmt}**.")
        bullets.append(f"**{bottom_name}** has the lowest average at **{bottom_fmt}**.")
        diff = top["value"] - bottom["value"]
        bullets.append(f"The spread between highest and lowest is **{_fmt_value(diff, metric)}**.")

    elif metric == "count":
        total = result_df["value"].sum()
        share = top["value"] / total * 100 if total > 0 else 0
        bullets.append(f"**{top_name}** dominates with **{_fmt_value(top['value'], metric)} transactions** ({share:.1f}% share).")
        bullets.append(f"**{bottom_name}** is the least used with **{_fmt_value(bottom['value'], metric)} transactions**.")

    elif metric == "total_volume":
        bullets.append(f"**{top_name}** generates the highest total volume at **{top_fmt}**.")
        bullets.append(f"**{bottom_name}** contributes the least at **{bottom_fmt}**.")

    narrative = f"{metric_label} analysis by **{dim_label}** across {len(result_df)} groups. " + " ".join(bullets)
    return narrative, bullets


# ══════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════

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
                "group":    str(row[group_col]),
                "value":    float(row["value"]),
                "z_score":  round(float(z), 2),
                "direction":"high" if z > 0 else "low",
                "formatted":_fmt_value(row["value"], metric),
            })
    return sorted(anomalies, key=lambda x: abs(x["z_score"]), reverse=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def execute_query(query: AnalyticalQuery, df: pd.DataFrame) -> AnalyticsResult:
    """
    Execute the analytical query against the DataFrame.
    Returns a fully populated AnalyticsResult.
    """
    result = AnalyticsResult(
        intent       = query.intent,
        metric       = query.metric,
        metric_label = METRIC_AGG.get(query.metric, {}).get("label", query.metric),
        group_by     = query.group_by,
        dim_label    = DIM_LABELS.get(query.group_by or "", query.group_by or ""),
        filters      = query.filters,
        total_rows   = len(df),
        query_json   = query.to_json(),
    )

    # ── validate metric ───────────────────────────────────────────────────
    if query.metric not in METRIC_AGG:
        result.success = False
        result.error   = f"Unknown metric '{query.metric}'. Supported: {list(METRIC_AGG.keys())}"
        return result

    # ── apply filters ─────────────────────────────────────────────────────
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

    # ── time-window filter ────────────────────────────────────────────────
    if query.time_window:
        tw = query.time_window
        if tw["type"] == "hour_range":
            fdf = fdf[(fdf["hour_of_day"] >= tw["min"]) & (fdf["hour_of_day"] <= tw["max"])]
        elif tw["type"] == "weekend":
            fdf = fdf[fdf["is_weekend"] == 1]
        elif tw["type"] == "weekday":
            fdf = fdf[fdf["is_weekend"] == 0]

    # ── resolve group column ──────────────────────────────────────────────
    group_col = None
    if query.group_by:
        # map logical name to actual df column
        col_map = {
            "transaction_type": "transaction_type",
            **{c: c for c in df.columns}
        }
        group_col = col_map.get(query.group_by, query.group_by)
        if group_col not in fdf.columns:
            group_col = None

    # ══════════════════════════════════════════════════════════════════════
    #  INTENT DISPATCH
    # ══════════════════════════════════════════════════════════════════════

    try:
        # ── SINGLE ────────────────────────────────────────────────────────
        if query.intent == "single" and not group_col:
            scalar_df = _compute_metric(fdf, query.metric)
            val = float(scalar_df["value"].iloc[0])
            result.scalar_value = val
            result.scalar_fmt   = _fmt_value(val, query.metric)

            # Supporting context breakdown by top dimension
            top_dim = "merchant_category" if "merchant_category" in fdf.columns else "sender_state"
            ctx_df  = _compute_metric(fdf, query.metric, top_dim)
            ctx_df  = ctx_df.sort_values("value", ascending=False).head(5)
            result.result_df = ctx_df.rename(columns={top_dim: "group_by"})

            result.narrative = (
                f"Overall **{result.metric_label}** is **{result.scalar_fmt}** "
                f"across {result.filtered_rows:,} transactions"
                + (f" (filtered from {result.total_rows:,})" if result.filtered_rows < result.total_rows else "")
                + "."
            )
            result.insight_bullets = [
                f"Dataset covers {result.total_rows:,} transactions.",
                f"Filtered subset: {result.filtered_rows:,} transactions.",
                f"Computed {result.metric_label}: **{result.scalar_fmt}**",
            ]

        # ── COMPARISON / RANKING ──────────────────────────────────────────
        elif query.intent in ("comparison", "ranking", "single") and group_col:
            agg_df = _compute_metric(fdf, query.metric, group_col)
            agg_df = agg_df.sort_values(
                "value", ascending=query.sort_ascending
            ).head(query.top_n)

            result.result_df = agg_df
            narrative, bullets = _build_narrative(agg_df, query, result.metric_label, result.dim_label)
            result.narrative       = narrative
            result.insight_bullets = bullets
            result.anomalies       = _detect_anomalies(agg_df, group_col, query.metric)

        # ── TREND ─────────────────────────────────────────────────────────
        elif query.intent == "trend":
            time_col = group_col or "month"
            if time_col not in fdf.columns:
                time_col = "day_of_week"

            agg_df = _compute_metric(fdf, query.metric, time_col)

            # Sort trends chronologically where possible
            if time_col == "day_of_week":
                day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                agg_df[time_col] = pd.Categorical(agg_df[time_col], categories=day_order, ordered=True)
                agg_df = agg_df.sort_values(time_col)
            elif time_col in ("month", "quarter"):
                agg_df = agg_df.sort_values(time_col)
            elif time_col == "hour_of_day":
                agg_df = agg_df.sort_values(time_col)

            result.result_df = agg_df

            # Find peak
            peak_row    = agg_df.loc[agg_df["value"].idxmax()]
            trough_row  = agg_df.loc[agg_df["value"].idxmin()]
            result.narrative = (
                f"**{result.metric_label}** trend over **{DIM_LABELS.get(time_col, time_col)}**. "
                f"Peak: **{peak_row[time_col]}** at **{_fmt_value(peak_row['value'], query.metric)}**. "
                f"Trough: **{trough_row[time_col]}** at **{_fmt_value(trough_row['value'], query.metric)}**."
            )
            result.insight_bullets = [
                f"Peak period: **{peak_row[time_col]}** ({_fmt_value(peak_row['value'], query.metric)})",
                f"Quietest period: **{trough_row[time_col]}** ({_fmt_value(trough_row['value'], query.metric)})",
                f"Variance across periods: {_fmt_value(float(agg_df['value'].std()), query.metric)} std dev",
            ]

        # ── ANOMALY ───────────────────────────────────────────────────────
        elif query.intent == "anomaly":
            dim = group_col or "merchant_category"
            agg_df = _compute_metric(fdf, query.metric, dim)
            anomalies = _detect_anomalies(agg_df, dim, query.metric, z_thresh=1.5)
            result.result_df = agg_df
            result.anomalies = anomalies
            if anomalies:
                top_a = anomalies[0]
                result.narrative = (
                    f"**{len(anomalies)} anomalies** detected in {result.metric_label} by {result.dim_label}. "
                    f"Strongest: **{top_a['group']}** at **{top_a['formatted']}** "
                    f"(z-score: {top_a['z_score']:+.2f})."
                )
                result.insight_bullets = [
                    f"**{a['group']}**: {a['formatted']} (z={a['z_score']:+.2f}, {a['direction']} outlier)"
                    for a in anomalies[:5]
                ]
            else:
                result.narrative = "No significant anomalies detected — distribution appears normal."

        else:
            # Fallback: comparison without group
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
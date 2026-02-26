"""
insightx/src/visualizer.py
──────────────────────────
Builds Plotly figures from AnalyticsResult objects.
All charts use a consistent dark theme matching the Streamlit UI.

v2 — Added:
  • comparison_bar_chart()  — grouped/side-by-side bars for vs. queries
  • network_comparison()    — specialized 3G/4G/5G radar + bar
  • _apply_base() no longer conflicts with xaxis/yaxis overrides
"""

from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional, List

from src.analytics_engine import AnalyticsResult, METRIC_AGG, DIM_LABELS

# ══════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════

PALETTE = {
    "primary":   "#00D4FF",
    "secondary": "#7C3AED",
    "accent":    "#F59E0B",
    "success":   "#10B981",
    "danger":    "#EF4444",
    "warning":   "#F97316",
    "bg":        "#0F172A",
    "surface":   "#1E293B",
    "border":    "#334155",
    "text":      "#E2E8F0",
    "muted":     "#64748B",
}

# Multi-series color palette for comparison charts
COMPARE_COLORS = [
    "#00D4FF", "#7C3AED", "#F59E0B", "#10B981",
    "#EF4444", "#F97316", "#06B6D4", "#8B5CF6",
]

GRADIENT_SCALE = [
    [0.0,  "#1E293B"],
    [0.4,  "#0EA5E9"],
    [0.7,  "#7C3AED"],
    [1.0,  "#F59E0B"],
]

RISK_SCALE = [
    [0.0, "#10B981"],
    [0.5, "#F97316"],
    [1.0, "#EF4444"],
]

_BASE_LAYOUT = dict(
    paper_bgcolor = PALETTE["bg"],
    plot_bgcolor  = PALETTE["surface"],
    font          = dict(family="DM Sans, sans-serif", color=PALETTE["text"], size=13),
    margin        = dict(l=40, r=20, t=50, b=40),
    xaxis         = dict(gridcolor=PALETTE["border"], showgrid=True, zeroline=False),
    yaxis         = dict(gridcolor=PALETTE["border"], showgrid=True, zeroline=False),
    hoverlabel    = dict(bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"], font_size=13),
    legend        = dict(bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"]),
)

# Base layout WITHOUT xaxis/yaxis/legend — safe to use when you override them yourself
_BASE_LAYOUT_NO_AXES = {k: v for k, v in _BASE_LAYOUT.items() if k not in ("xaxis", "yaxis", "legend")}


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    layout = dict(**_BASE_LAYOUT)
    if title:
        layout["title"] = dict(text=title, font=dict(size=16, color=PALETTE["text"]), x=0)
    fig.update_layout(**layout)
    return fig


def _metric_color(metric: str) -> str:
    return {
        "fraud_rate":   PALETTE["danger"],
        "failure_rate": PALETTE["warning"],
        "avg_amount":   PALETTE["primary"],
        "count":        PALETTE["secondary"],
        "total_volume": PALETTE["success"],
    }.get(metric, PALETTE["primary"])


# ══════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def bar_chart(result: AnalyticsResult) -> go.Figure:
    """Horizontal bar chart for comparison / ranking intents."""
    df  = result.result_df
    dim = result.group_by

    if df is None or df.empty:
        return go.Figure()

    x_col     = dim if dim in df.columns else df.columns[0]
    df_sorted = df.sort_values("value", ascending=True)

    norm_vals = (df_sorted["value"] - df_sorted["value"].min()) / (
        (df_sorted["value"].max() - df_sorted["value"].min()) or 1
    )
    colors = [
        f"rgba({int(0 + 239*v)}, {int(212 - 212*v)}, {int(255 - 255*v)}, 0.85)"
        for v in norm_vals
    ]

    fig = go.Figure(go.Bar(
        x            = df_sorted["value"],
        y            = df_sorted[x_col].astype(str),
        orientation  = "h",
        marker_color = colors,
        text         = df_sorted["value"].apply(
            lambda v: METRIC_AGG[result.metric]["fmt"].format(v)
        ),
        textposition = "outside",
        textfont     = dict(color=PALETTE["text"], size=11),
        hovertemplate= f"<b>%{{y}}</b><br>{result.metric_label}: %{{x:.2f}}<extra></extra>",
    ))

    # ✅ FIX: use _BASE_LAYOUT_NO_AXES so xaxis/yaxis are never duplicated
    fig.update_layout(
        **_BASE_LAYOUT_NO_AXES,
        title  = dict(text=f"{result.metric_label} by {result.dim_label}", font=dict(size=15, color=PALETTE["text"])),
        xaxis  = dict(title=result.metric_label, gridcolor=PALETTE["border"], showgrid=True, zeroline=False),
        yaxis  = dict(title="", gridcolor="rgba(0,0,0,0)", showgrid=True, zeroline=False, tickfont=dict(size=11)),
        height = max(320, len(df_sorted) * 40 + 80),
    )
    return fig


def line_chart(result: AnalyticsResult) -> go.Figure:
    """Line / area chart for trend intent."""
    df  = result.result_df
    dim = result.group_by or "month"

    if df is None or df.empty:
        return go.Figure()

    x_col = dim if dim in df.columns else df.columns[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x            = df[x_col].astype(str),
        y            = df["value"],
        mode         = "lines+markers",
        line         = dict(color=PALETTE["primary"], width=2.5, shape="spline"),
        marker       = dict(size=7, color=PALETTE["primary"], line=dict(color=PALETTE["bg"], width=1.5)),
        fill         = "tozeroy",
        fillcolor    = "rgba(0,212,255,0.12)",
        name         = result.metric_label,
        hovertemplate= f"<b>%{{x}}</b><br>{result.metric_label}: %{{y:.2f}}<extra></extra>",
    ))

    if result.anomalies:
        anom_groups = {a["group"] for a in result.anomalies}
        anom_df = df[df[x_col].astype(str).isin(anom_groups)]
        if not anom_df.empty:
            fig.add_trace(go.Scatter(
                x          = anom_df[x_col].astype(str),
                y          = anom_df["value"],
                mode       = "markers",
                marker     = dict(size=13, color=PALETTE["danger"], symbol="diamond",
                                  line=dict(color="#fff", width=1.5)),
                name       = "Anomaly",
                hovertemplate="<b>%{x}</b> ⚠ Anomaly<extra></extra>",
            ))

    # ✅ FIX: use _BASE_LAYOUT_NO_AXES
    fig.update_layout(
        **_BASE_LAYOUT_NO_AXES,
        title  = dict(text=f"{result.metric_label} Trend over {DIM_LABELS.get(dim, dim)}",
                      font=dict(size=15, color=PALETTE["text"])),
        xaxis  = dict(title=DIM_LABELS.get(dim, dim), gridcolor=PALETTE["border"], showgrid=True, zeroline=False, tickangle=-30),
        yaxis  = dict(title=result.metric_label, gridcolor=PALETTE["border"], showgrid=True, zeroline=False),
        height = 380,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  NEW: COMPARISON CHARTS
# ══════════════════════════════════════════════════════════════════════════

def comparison_bar_chart(
    result: AnalyticsResult,
    compare_values: List[str],
) -> go.Figure:
    """
    Side-by-side vertical bar chart for explicit vs. comparisons.
    e.g. "3G vs 5G", "iOS vs Android vs Web"

    result.result_df must have columns: [group_by, 'value']
    compare_values: list of segment values to highlight
    """
    df  = result.result_df
    dim = result.group_by

    if df is None or df.empty:
        return go.Figure()

    x_col = dim if dim in df.columns else df.columns[0]

    # Filter to only the requested comparison groups if specified
    if compare_values:
        mask = df[x_col].astype(str).isin([str(v) for v in compare_values])
        df   = df[mask]

    df = df.sort_values("value", ascending=False)

    fig = go.Figure()

    for i, (_, row) in enumerate(df.iterrows()):
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        val   = row["value"]
        label = str(row[x_col])
        fmt   = METRIC_AGG[result.metric]["fmt"].format(val)

        fig.add_trace(go.Bar(
            name         = label,
            x            = [label],
            y            = [val],
            marker_color = color,
            marker_line  = dict(color=PALETTE["bg"], width=2),
            text         = [fmt],
            textposition = "outside",
            textfont     = dict(color=color, size=13, family="DM Sans"),
            hovertemplate= f"<b>{label}</b><br>{result.metric_label}: {fmt}<extra></extra>",
        ))

    # Rank badge annotation — highlight winner
    if not df.empty:
        winner_row = df.iloc[0]
        winner_val = winner_row["value"]
        winner_lbl = str(winner_row[x_col])
        winner_fmt = METRIC_AGG[result.metric]["fmt"].format(winner_val)

    title_text = f"{result.metric_label}: {' vs '.join(df[x_col].astype(str).tolist())}"

    fig.update_layout(
        **_BASE_LAYOUT_NO_AXES,
        title       = dict(text=title_text, font=dict(size=15, color=PALETTE["text"])),
        xaxis       = dict(title="", gridcolor=PALETTE["border"], showgrid=False, zeroline=False),
        yaxis       = dict(title=result.metric_label, gridcolor=PALETTE["border"], showgrid=True, zeroline=False),
        barmode     = "group",
        showlegend  = True,
        legend      = dict(
            bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"],
            orientation="h", y=-0.15, x=0.5, xanchor="center",
        ),
        height      = 420,
        bargap      = 0.25,
        bargroupgap = 0.1,
    )
    return fig


def comparison_radar_chart(
    result: AnalyticsResult,
    all_metrics_data: dict,  # {segment: {metric: value}}
    metrics_to_show: Optional[List[str]] = None,
) -> go.Figure:
    """
    Radar / spider chart for multi-metric comparison across segments.
    e.g. compare 3G vs 4G vs 5G across fraud_rate, failure_rate, avg_amount

    all_metrics_data: {
        "3G": {"fraud_rate": 0.21, "failure_rate": 3.5, "avg_amount": 1200},
        "5G": {"fraud_rate": 0.15, "failure_rate": 2.1, "avg_amount": 1800},
    }
    """
    if not all_metrics_data:
        return go.Figure()

    default_metrics = ["fraud_rate", "failure_rate", "avg_amount", "count"]
    metrics = metrics_to_show or default_metrics
    metric_labels = [METRIC_AGG.get(m, {}).get("label", m) for m in metrics]

    fig = go.Figure()

    for i, (segment, values) in enumerate(all_metrics_data.items()):
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]

        # Normalize values to 0–1 for radar (min-max across segments per metric)
        raw = [values.get(m, 0) for m in metrics]

        # Simple normalization per metric across all segments
        all_vals_per_metric = {
            m: [v.get(m, 0) for v in all_metrics_data.values()]
            for m in metrics
        }
        normed = []
        for m, rv in zip(metrics, raw):
            mn = min(all_vals_per_metric[m])
            mx = max(all_vals_per_metric[m])
            normed.append((rv - mn) / (mx - mn + 1e-9))

        fig.add_trace(go.Scatterpolar(
            r     = normed + [normed[0]],  # close the polygon
            theta = metric_labels + [metric_labels[0]],
            fill  = "toself",
            fillcolor = f"rgba{tuple(list(_hex_to_rgb(color)) + [0.15])}",
            line  = dict(color=color, width=2),
            name  = segment,
            hovertemplate="<b>" + segment + "</b><br>%{theta}: %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT_NO_AXES,
        polar = dict(
            bgcolor    = PALETTE["surface"],
            radialaxis = dict(
                visible    = True,
                range      = [0, 1],
                gridcolor  = PALETTE["border"],
                linecolor  = PALETTE["border"],
                tickfont   = dict(color=PALETTE["muted"], size=10),
            ),
            angularaxis = dict(
                gridcolor = PALETTE["border"],
                linecolor = PALETTE["border"],
                tickfont  = dict(color=PALETTE["text"], size=12),
            ),
        ),
        title  = dict(
            text = f"Multi-Metric Comparison — {' vs '.join(all_metrics_data.keys())}",
            font = dict(size=15, color=PALETTE["text"]),
        ),
        legend = dict(
            bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"],
            orientation="h", y=-0.1, x=0.5, xanchor="center",
        ),
        height = 460,
    )
    return fig


def comparison_delta_chart(
    result: AnalyticsResult,
    compare_values: List[str],
) -> go.Figure:
    """
    Delta / waterfall-style chart showing difference from average.
    Great for 'which is better/worse than average?' queries.
    """
    df  = result.result_df
    dim = result.group_by

    if df is None or df.empty:
        return go.Figure()

    x_col = dim if dim in df.columns else df.columns[0]

    if compare_values:
        df = df[df[x_col].astype(str).isin([str(v) for v in compare_values])]

    df       = df.copy()
    avg      = df["value"].mean()
    df["delta"]   = df["value"] - avg
    df["is_above"] = df["delta"] >= 0
    df = df.sort_values("delta", ascending=True)

    colors = [PALETTE["danger"] if above else PALETTE["success"] for above in df["is_above"]]

    fig = go.Figure(go.Bar(
        x            = df["delta"],
        y            = df[x_col].astype(str),
        orientation  = "h",
        marker_color = colors,
        marker_line  = dict(color=PALETTE["bg"], width=1),
        text         = df["delta"].apply(lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"),
        textposition = "outside",
        textfont     = dict(color=PALETTE["text"], size=11),
        hovertemplate= f"<b>%{{y}}</b><br>Δ from avg: %{{x:+.2f}}<extra></extra>",
    ))

    # Average reference line
    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color=PALETTE["muted"],
        annotation_text=f"Avg: {METRIC_AGG[result.metric]['fmt'].format(avg)}",
        annotation_font_color=PALETTE["muted"],
        annotation_position="top",
    )

    fig.update_layout(
        **_BASE_LAYOUT_NO_AXES,
        title  = dict(
            text = f"{result.metric_label} — Delta from Average",
            font = dict(size=15, color=PALETTE["text"]),
        ),
        xaxis  = dict(title=f"Δ {result.metric_label}", gridcolor=PALETTE["border"], showgrid=True, zeroline=True, zerolinecolor=PALETTE["border"]),
        yaxis  = dict(title="", gridcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False, tickfont=dict(size=11)),
        height = max(320, len(df) * 44 + 80),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  EXISTING CHARTS (unchanged, but fixed _BASE_LAYOUT usage)
# ══════════════════════════════════════════════════════════════════════════

def gauge_chart(value: float, metric: str, label: str) -> go.Figure:
    """Single KPI gauge for scalar results."""
    cfg = METRIC_AGG.get(metric, {})
    fmt = cfg.get("fmt", "{:.2f}")

    max_val = {"fraud_rate": 5, "failure_rate": 20, "avg_amount": 5000,
               "count": 250000, "total_volume": 500000000}.get(metric, value * 2 or 1)

    color_steps = (
        [{"range": [0, max_val * 0.33], "color": "#10B981"},
         {"range": [max_val * 0.33, max_val * 0.66], "color": "#F97316"},
         {"range": [max_val * 0.66, max_val], "color": "#EF4444"}]
        if metric in ("fraud_rate", "failure_rate") else
        [{"range": [0, max_val], "color": PALETTE["secondary"]}]
    )

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = value,
        title  = {"text": label, "font": {"color": PALETTE["text"], "size": 15}},
        number = {"suffix": "%" if "rate" in metric else "", "font": {"color": PALETTE["primary"], "size": 36}},
        gauge  = {
            "axis":        {"range": [0, max_val], "tickcolor": PALETTE["muted"]},
            "bar":         {"color": _metric_color(metric)},
            "bgcolor":     PALETTE["surface"],
            "bordercolor": PALETTE["border"],
            "steps":       color_steps,
            "threshold":   {"line": {"color": PALETTE["accent"], "width": 3}, "value": value},
        },
    ))

    fig.update_layout(
        paper_bgcolor = PALETTE["bg"],
        font          = dict(family="DM Sans", color=PALETTE["text"]),
        height        = 280,
        margin        = dict(l=30, r=30, t=40, b=20),
    )
    return fig


def heatmap_hourly(df_hour: pd.DataFrame, metric: str) -> go.Figure:
    """Heatmap of metric across hours × days."""
    days  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hours = list(range(24))

    z = [[0.0] * 24 for _ in range(7)]
    for _, row in df_hour.iterrows():
        try:
            d_idx = days.index(str(row.get("day_of_week", "")))
            h_idx = int(row.get("hour_of_day", 0))
            z[d_idx][h_idx] = float(row.get("value", 0))
        except (ValueError, TypeError):
            pass

    scale = RISK_SCALE if metric in ("fraud_rate", "failure_rate") else GRADIENT_SCALE

    fig = go.Figure(go.Heatmap(
        z           = z,
        x           = [f"{h:02d}:00" for h in hours],
        y           = days,
        colorscale  = scale,
        hoverongaps = False,
        hovertemplate="<b>%{y}</b> %{x}<br>Value: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title  = dict(text=f"{METRIC_AGG.get(metric,{}).get('label','Metric')} — Hourly Heatmap",
                      font=dict(size=15, color=PALETTE["text"])),
        height = 350,
    )
    return fig


def anomaly_chart(result: AnalyticsResult) -> go.Figure:
    """Bar chart highlighting anomalous groups."""
    df          = result.result_df
    dim         = result.group_by or "group_by"
    anom_groups = {a["group"] for a in result.anomalies}

    if df is None or df.empty:
        return go.Figure()

    x_col = dim if dim in df.columns else df.columns[0]

    colors = [
        PALETTE["danger"] if str(row[x_col]) in anom_groups else PALETTE["surface"]
        for _, row in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x            = df[x_col].astype(str),
        y            = df["value"],
        marker_color = colors,
        marker_line  = dict(color=PALETTE["border"], width=1),
        text         = df["value"].apply(lambda v: METRIC_AGG[result.metric]["fmt"].format(v)),
        textposition = "outside",
        hovertemplate= f"<b>%{{x}}</b><br>{result.metric_label}: %{{y:.2f}}<extra></extra>",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title  = dict(text=f"Anomaly Detection — {result.metric_label} by {result.dim_label}",
                      font=dict(size=15, color=PALETTE["text"])),
        height = 380,
    )
    return fig


def donut_chart(df: pd.DataFrame, dim: str, metric: str = "count") -> go.Figure:
    """Donut for share/composition view."""
    if df is None or df.empty:
        return go.Figure()

    x_col  = dim if dim in df.columns else df.columns[0]
    colors = [
        "#00D4FF","#7C3AED","#F59E0B","#10B981","#EF4444",
        "#F97316","#06B6D4","#8B5CF6","#EC4899","#14B8A6",
    ]
    fig = go.Figure(go.Pie(
        labels        = df[x_col].astype(str),
        values        = df["value"],
        hole          = 0.55,
        marker_colors = colors[:len(df)],
        textinfo      = "label+percent",
        textfont      = dict(size=11, color=PALETTE["text"]),
        hovertemplate = "<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor = PALETTE["bg"],
        font          = dict(family="DM Sans", color=PALETTE["text"]),
        legend        = dict(bgcolor=PALETTE["surface"]),
        height        = 340,
        margin        = dict(l=10, r=10, t=30, b=10),
    )
    return fig


def overview_kpi_bars(df: pd.DataFrame) -> go.Figure:
    """Compact multi-metric bar for the dashboard overview."""
    categories = df["merchant_category"].value_counts().head(8)
    fig = go.Figure(go.Bar(
        x            = categories.values,
        y            = categories.index,
        orientation  = "h",
        marker_color = PALETTE["primary"],
        hovertemplate= "<b>%{y}</b><br>Transactions: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title  = dict(text="Top Categories by Transaction Volume", font=dict(size=14, color=PALETTE["text"])),
        height = 300,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str):
    """Convert #RRGGBB to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def pick_chart(result: AnalyticsResult, compare_values: Optional[List[str]] = None) -> go.Figure:
    """
    Smart chart picker — call this from app.py instead of hardcoding chart types.
    Routes to the best visualization based on intent + compare context.
    """
    intent = result.intent

    if intent == "comparison":
        if compare_values and len(compare_values) >= 2:
            return comparison_bar_chart(result, compare_values)
        return bar_chart(result)

    if intent == "trend":
        return line_chart(result)

    if intent == "anomaly":
        return anomaly_chart(result)

    if intent == "distribution":
        return donut_chart(result.result_df, result.group_by, result.metric)

    if intent == "summary":
        if result.scalar_value is not None:
            return gauge_chart(result.scalar_value, result.metric, result.metric_label)
        return bar_chart(result)

    # fallback
    return bar_chart(result)

"""
insightx/src/nlp_engine.py
──────────────────────────
Converts a free-text business question into a structured
AnalyticalQuery object — fully rule-based with fuzzy matching
for entity resolution.

Pipeline
--------
1. tokenise & lowercase
2. detect INTENT   (single | comparison | trend | ranking | anomaly)
3. detect METRIC   (fraud_rate | failure_rate | avg_amount | count | total_volume)
4. detect GROUP_BY dimension
5. resolve FILTERS (state / bank / category / device / network / age / txn_type)
6. detect TIME range keywords
7. detect SORT direction (top / bottom / highest / lowest)
8. apply context_hint from ContextMemory for follow-up queries
9. return AnalyticalQuery dataclass
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# optional fuzzy matching — falls back gracefully if library absent
try:
    from rapidfuzz import process as fuzz_process, fuzz
    _FUZZY = True
except ImportError:
    _FUZZY = False

# Stub for streamlit cache when running outside streamlit
try:
    import streamlit as st
except ImportError:
    class _st:
        @staticmethod
        def cache_data(*args, **kwargs):
            def decorator(fn): return fn
            return decorator
    st = _st()


# ══════════════════════════════════════════════════════════════════════════
#  ENTITY VOCABULARY
# ══════════════════════════════════════════════════════════════════════════

STATES = [
    "Andhra Pradesh", "Delhi", "Gujarat", "Karnataka", "Maharashtra",
    "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal",
]
BANKS      = ["Axis", "HDFC", "ICICI", "IndusInd", "Kotak", "PNB", "SBI", "Yes Bank"]
CATEGORIES = [
    "Education", "Entertainment", "Food", "Fuel", "Grocery",
    "Healthcare", "Other", "Shopping", "Transport", "Utilities",
]
AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]
DEVICES    = ["Android", "iOS", "Web"]
NETWORKS   = ["3G", "4G", "5G", "WiFi"]
TXN_TYPES  = ["P2P", "P2M", "Bill Payment", "Recharge"]

# ── keyword → canonical dimension name ──────────────────────────────────
DIM_KEYWORDS: dict[str, list[str]] = {
    "device_type":       ["device", "android", "ios", "web", "mobile", "browser", "platform"],
    "network_type":      ["network", "5g", "4g", "3g", "wifi", "wi-fi", "connectivity", "connection"],
    "sender_state":      ["state", "region", "city", "location", "geography"],
    "sender_bank":       ["bank", "lender", "provider", "financial institution"],
    "merchant_category": ["category", "sector", "merchant", "industry", "type of purchase", "spend category"],
    "sender_age_group":  ["age", "age group", "demographic", "generation", "young", "senior", "millennial"],
    "day_of_week":       ["day", "weekday", "weekend", "monday", "tuesday", "wednesday",
                          "thursday", "friday", "saturday", "sunday"],
    "hour_of_day":       ["hour", "time", "peak hour", "morning", "afternoon", "evening",
                          "night", "midnight", "clock"],
    "transaction_type":  ["transaction type", "p2p", "p2m", "bill payment", "recharge",
                          "mode", "payment type"],
    "month":             ["month", "monthly", "january", "february", "march", "april", "may",
                          "june", "july", "august", "september", "october", "november", "december"],
}

# ── keyword → canonical metric name ─────────────────────────────────────
METRIC_KEYWORDS: dict[str, list[str]] = {
    "fraud_rate":   ["fraud", "fraudulent", "scam", "flag", "flagged", "suspicious", "risk"],
    "failure_rate": ["fail", "failure", "failed", "decline", "declined", "unstable",
                     "unsuccessful", "error", "drop", "dropout", "bounce"],
    "avg_amount":   ["average", "avg", "mean", "typical", "usual", "standard amount", "transaction size"],
    "count":        ["count", "volume", "number", "how many", "total transaction",
                     "frequency", "most used", "popular", "busiest"],
    "total_volume": ["total amount", "total value", "revenue", "sum", "aggregate", "cumulative"],
}

# ── intent keywords ──────────────────────────────────────────────────────
INTENT_KEYWORDS: dict[str, list[str]] = {
    "trend":      ["trend", "over time", "monthly", "weekly", "daily", "by month", "by week",
                   "by day", "by hour", "time series", "timeline", "pattern", "seasonal",
                   "growth", "decline", "change"],
    "ranking":    ["top", "bottom", "best", "worst", "highest", "lowest", "rank", "ranking",
                   "most", "least", "which.*most", "which.*least", "who has", "who is",
                   "maximum", "minimum"],
    "comparison": ["compare", "vs", "versus", "against", "difference between", "contrast",
                   "between", "across", "each", "per", "by device", "by bank",
                   "by state", "by category"],
    "anomaly":    ["anomal", "outlier", "unusual", "spike", "abnormal", "sudden", "unexpected"],
    "single":     ["what is", "what's", "show", "tell me", "give me", "display",
                   "overall", "total", "summary", "overview"],
}

# ── sort direction ───────────────────────────────────────────────────────
ASCENDING_KW  = ["lowest", "least", "bottom", "worst", "minimum", "min"]
DESCENDING_KW = ["highest", "most", "top", "best", "maximum", "max"]

# ── time-window keywords ─────────────────────────────────────────────────
TIME_KEYWORDS = {
    "morning":   (6,  11),
    "afternoon": (12, 16),
    "evening":   (17, 20),
    "night":     (21, 23),
    "midnight":  (0,   5),
    "weekend":   "weekend",
    "weekday":   "weekday",
}

MONTH_MAP = {
    "jan": 1,  "feb": 2,  "mar": 3,  "apr": 4,  "may": 5,  "jun": 6,
    "jul": 7,  "aug": 8,  "sep": 9,  "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3,     "april": 4,
    "june": 6,    "july": 7,    "august": 8,     "september": 9,
    "october": 10,"november": 11,"december": 12,
}


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalyticalQuery:
    intent:               str            = "single"
    metric:               str            = "count"
    group_by:             Optional[str]  = None
    filters:              dict           = field(default_factory=dict)
    time_window:          Optional[dict] = None
    sort_ascending:       bool           = False
    top_n:                int            = 10
    confidence:           float          = 0.0
    confidence_reasoning: str            = ""
    raw_query:            str            = ""
    followup:             bool           = False   # True when merged from context memory
    compare:              list           = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _tok(text: str) -> str:
    """Lower-case and strip punctuation except hyphens/+."""
    return re.sub(r"[^\w\s\-\+]", " ", text.lower()).strip()


def _any_kw(text: str, kws: list[str]) -> bool:
    return any(kw in text for kw in kws)


def _fuzzy_resolve(query_token: str, candidates: list[str], threshold: int = 72) -> Optional[str]:
    if not _FUZZY or not query_token:
        return None
    result = fuzz_process.extractOne(query_token, candidates, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def _extract_top_n(text: str) -> int:
    m = re.search(r"\b(?:top|bottom|best|worst)\s+(\d+)\b", text)
    if m:
        return int(m.group(1))
    return 10


# ══════════════════════════════════════════════════════════════════════════
#  CORE PARSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _detect_intent(text: str) -> tuple[str, float]:
    scores: dict[str, int] = {}
    for intent, kws in INTENT_KEYWORDS.items():
        hit = sum(1 for kw in kws if kw in text)
        if hit:
            scores[intent] = hit
    if not scores:
        return "single", 0.5
    best = max(scores, key=scores.get)
    conf = min(0.95, 0.6 + scores[best] * 0.1)
    return best, conf


def _detect_metric(text: str) -> tuple[str, float]:
    scores: dict[str, int] = {}
    for metric, kws in METRIC_KEYWORDS.items():
        hit = sum(1 for kw in kws if kw in text)
        if hit:
            scores[metric] = hit
    if not scores:
        return "count", 0.4
    best = max(scores, key=scores.get)
    conf = min(0.95, 0.65 + scores[best] * 0.1)
    return best, conf


def _detect_groupby(text: str) -> tuple[Optional[str], float]:
    scores: dict[str, int] = {}
    for dim, kws in DIM_KEYWORDS.items():
        hit = sum(1 for kw in kws if kw in text)
        if hit:
            scores[dim] = hit
    if not scores:
        return None, 0.0
    best = max(scores, key=scores.get)
    conf = min(0.95, 0.6 + scores[best] * 0.1)
    return best, conf


def _detect_filters(raw: str, tok: str) -> dict:
    filters: dict = {}

    # State
    for s in STATES:
        if s.lower() in tok:
            filters["sender_state"] = s
            break
    if "sender_state" not in filters and _FUZZY:
        for word in tok.split():
            match = _fuzzy_resolve(word, [s.lower() for s in STATES], 80)
            if match:
                filters["sender_state"] = STATES[[s.lower() for s in STATES].index(match)]
                break

    # Bank
    for b in BANKS:
        if b.lower() in tok:
            filters["sender_bank"] = b
            break

    # Category
    for c in CATEGORIES:
        if c.lower() in tok:
            filters["merchant_category"] = c
            break

    # Device
    for d in DEVICES:
        if d.lower() in tok:
            filters["device_type"] = d
            break

    # Network
    for n in NETWORKS:
        if n.lower() in tok:
            filters["network_type"] = n
            break

    # Age group
    age_match = re.search(r"\b(18-25|26-35|36-45|46-55|56\+)\b", raw)
    if age_match:
        filters["sender_age_group"] = age_match.group(1)
    elif "young" in tok or "youth" in tok:
        filters["sender_age_group"] = "18-25"
    elif "senior" in tok or "elder" in tok or "old" in tok:
        filters["sender_age_group"] = "56+"
    elif "millennial" in tok:
        filters["sender_age_group"] = "26-35"

    # Transaction type
    if "p2p" in tok:
        filters["transaction_type"] = "P2P"
    elif "p2m" in tok:
        filters["transaction_type"] = "P2M"
    elif "bill payment" in tok or "bill pay" in tok:
        filters["transaction_type"] = "Bill Payment"
    elif "recharge" in tok:
        filters["transaction_type"] = "Recharge"

    # Weekend / weekday
    if "weekend" in tok:
        filters["is_weekend"] = 1
    elif "weekday" in tok:
        filters["is_weekend"] = 0

    # Month
    for mname, mnum in MONTH_MAP.items():
        if mname in tok:
            filters["month_num"] = mnum
            break

    return filters


def _detect_time_window(tok: str) -> Optional[dict]:
    for label, val in TIME_KEYWORDS.items():
        if label in tok:
            if isinstance(val, tuple):
                return {"type": "hour_range", "label": label, "min": val[0], "max": val[1]}
            else:
                return {"type": val}
    return None


def _detect_sort(tok: str) -> bool:
    """True = ascending (lowest/worst queries)."""
    return _any_kw(tok, ASCENDING_KW)


def _detect_compare_values(tok: str) -> list[str]:
    """Extract explicit comparison values like '3G vs 5G' or 'HDFC vs SBI'."""
    compare = []
    # Network types
    for n in NETWORKS:
        if n.lower() in tok:
            compare.append(n)
    # Banks (only when multiple detected)
    bank_hits = [b for b in BANKS if b.lower() in tok]
    if len(bank_hits) >= 2:
        compare = bank_hits
    # Devices (only when multiple detected)
    device_hits = [d for d in DEVICES if d.lower() in tok]
    if len(device_hits) >= 2:
        compare = device_hits
    return compare


def _is_followup(tok: str) -> bool:
    """Detect if the query is a follow-up on previous context."""
    followup_signals = [
        "what about", "only ", "and in ", "for ", "same but",
        "now for", "show me only", "just ", "in that case",
        "how about", "what if",
    ]
    return any(sig in tok for sig in followup_signals)


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def parse_query(raw_query: str, context_hint: str = "") -> AnalyticalQuery:
    """
    Main entry point. Takes a natural-language question and an optional
    context_hint string from ContextMemory, and returns a fully
    populated AnalyticalQuery.

    Parameters
    ----------
    raw_query    : The user's natural-language question.
    context_hint : Optional string injected from ContextMemory.to_prompt_context().
                   Used to detect follow-up intent and inherit previous dimensions.
    """
    tok = _tok(raw_query)

    intent, intent_conf  = _detect_intent(tok)
    metric, metric_conf  = _detect_metric(tok)
    group_by, group_conf = _detect_groupby(tok)
    filters              = _detect_filters(raw_query, tok)
    time_window          = _detect_time_window(tok)
    sort_asc             = _detect_sort(tok)
    top_n                = _extract_top_n(tok)
    compare              = _detect_compare_values(tok)
    is_followup          = _is_followup(tok) and bool(context_hint)

    # ── Comparison promotion ──────────────────────────────────────────────
    # When multiple entities of the same type are detected (e.g. HDFC vs SBI,
    # P2P vs P2M, 3G vs 5G), we must:
    #   1. Set group_by to the correct dimension
    #   2. Remove those entities from filters (they're segments, not filters)
    #   3. Set intent = comparison

    bank_hits   = [b for b in BANKS    if b.lower() in tok]
    device_hits = [d for d in DEVICES  if d.lower() in tok]
    net_hits    = [n for n in NETWORKS if n.lower() in tok]
    txn_hits    = [t for t in TXN_TYPES if t.lower() in tok]

    if len(bank_hits) >= 2:
        group_by   = "sender_bank"
        group_conf = 0.95
        compare    = bank_hits
        filters.pop("sender_bank", None)   # remove single-bank filter

    elif len(device_hits) >= 2:
        group_by   = "device_type"
        group_conf = 0.95
        compare    = device_hits
        filters.pop("device_type", None)

    elif len(net_hits) >= 2:
        group_by   = "network_type"
        group_conf = 0.95
        compare    = net_hits
        filters.pop("network_type", None)

    elif len(txn_hits) >= 2:
        group_by   = "transaction_type"
        group_conf = 0.95
        compare    = txn_hits
        filters.pop("transaction_type", None)

    # Single network hit still needs group_by (e.g. "3G failure rate" = filter,
    # but "failure rate on 3G vs 5G" already handled above)
    if len(net_hits) == 1 and not group_by:
        group_by   = "network_type"
        group_conf = 0.85

    # ── Intent override heuristics ────────────────────────────────────────
    if group_by in ("month", "day_of_week", "hour_of_day") and intent == "single":
        intent = "trend"
    if group_by and intent == "single":
        intent = "comparison"
    if _any_kw(tok, ["top", "bottom", "rank", "highest", "lowest", "best", "worst",
                     "which", "who has", "most", "least"]):
        if intent not in ("trend",):
            intent = "ranking"
    if "vs" in tok or "versus" in tok or "compare" in tok:
        intent = "comparison"
    if compare and len(compare) >= 2:
        intent = "comparison"

    # ── Confidence aggregation ────────────────────────────────────────────
    conf = round((intent_conf + metric_conf + (group_conf if group_by else 0.5)) / 3, 2)

    # ── Reasoning string ──────────────────────────────────────────────────
    reasoning_parts = [
        f"Intent='{intent}' (conf {intent_conf:.0%})",
        f"Metric='{metric}' (conf {metric_conf:.0%})",
        f"GroupBy='{group_by}' (conf {group_conf:.0%})" if group_by else "No group-by detected",
    ]
    if filters:
        reasoning_parts.append(f"Filters={filters}")
    if compare:
        reasoning_parts.append(f"Compare={compare}")
    if time_window:
        reasoning_parts.append(f"TimeWindow={time_window}")
    if is_followup:
        reasoning_parts.append("Follow-up query detected")

    return AnalyticalQuery(
        intent               = intent,
        metric               = metric,
        group_by             = group_by,
        filters              = filters,
        time_window          = time_window,
        sort_ascending       = sort_asc,
        top_n                = top_n,
        confidence           = conf,
        confidence_reasoning = " | ".join(reasoning_parts),
        raw_query            = raw_query,
        followup             = is_followup,
        compare              = compare,
    )

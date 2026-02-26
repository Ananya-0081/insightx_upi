# InsightX — System Architecture & Data Analysis Methodology

## 1. Overview

InsightX is a rule-based Conversational BI system that translates
natural-language business questions into structured pandas queries,
executes them against 250,000 UPI transaction records, and returns
rich, explainable responses through a Streamlit chat interface.

No large language model or cloud service is required — the system runs
fully offline and deterministically.

---

## 2. Component Architecture

### 2.1 Data Layer (`src/data_loader.py`)

**Responsibilities:**
- Load the 250,000-row CSV once and cache via `@st.cache_data`
- Normalise column names (handle spaces, case variations)
- Cast types: timestamps → datetime, amounts → float, flags → int8
- Derive helper columns: `is_failed`, `is_fraud`, `month`, `quarter`
- Return schema metadata (unique values per dimension) for the NLP layer

**Key design decision:** Streamlit's `@st.cache_data` decorator ensures
the 250k-row parse happens once per user session, keeping response
latency under 200ms for subsequent queries.

---

### 2.2 NLP Engine (`src/nlp_engine.py`)

**Responsibilities:**
- Parse raw query string → `AnalyticalQuery` dataclass
- Detect: intent, metric, group-by dimension, entity filters, time window, sort direction

**Pipeline:**

```
Raw text
  │
  ├─ tokenise + lowercase
  ├─ INTENT detection    (keyword matching over 5 intent classes)
  ├─ METRIC detection    (keyword matching over 5 metric classes)
  ├─ GROUP_BY detection  (keyword matching over 11 dimension classes)
  ├─ FILTER extraction   (exact + fuzzy entity matching)
  ├─ TIME WINDOW         (morning/evening/night/weekend keywords)
  ├─ SORT DIRECTION      (top/bottom/highest/lowest keywords)
  └─ CONFIDENCE scoring  (weighted average of component confidences)
```

**Entity resolution strategy:**

| Entity type | Method |
|-------------|--------|
| States, banks, categories | Exact substring match on lowercase |
| States (typos) | `rapidfuzz.token_sort_ratio` ≥ 72% |
| Age groups | Regex `\b(18-25|26-35|…)\b` |
| Time-of-day | Keyword map → hour range tuples |

**Output:** `AnalyticalQuery` dataclass with JSON serialization

---

### 2.3 Analytics Engine (`src/analytics_engine.py`)

**Responsibilities:**
- Apply filters to the DataFrame
- Compute the requested metric (grouped or scalar)
- Detect anomalies (z-score method)
- Build narrative text and insight bullets

**Metric computation:**

| Metric | pandas operation |
|--------|-----------------|
| `fraud_rate` | `groupby.mean('is_fraud') * 100` |
| `failure_rate` | `groupby.mean('is_failed') * 100` |
| `avg_amount` | `groupby.mean('amount_inr')` |
| `count` | `groupby.size()` |
| `total_volume` | `groupby.sum('amount_inr')` |

**Intent dispatch:**

| Intent | Execution |
|--------|-----------|
| `single` | Scalar + breakdown by top dimension |
| `comparison` | Grouped aggregation, sorted by value |
| `trend` | Grouped aggregation, sorted chronologically |
| `ranking` | Grouped aggregation, top-N |
| `anomaly` | Grouped aggregation + z-score filtering |

**Anomaly detection:**
```
z_score = (value - mean) / std_dev
threshold = ±2.0σ (configurable)
```

---

### 2.4 Response Generator (`src/response_generator.py`)

**Responsibilities:**
- Build headline (concise summary of top finding)
- Select chart type based on intent + dimension
- Generate risk flags (rule-based thresholds)
- Generate action recommendations (per metric)
- Suggest follow-up questions (per metric, avoiding current group-by)
- Score confidence label: Very High / High / Medium / Low

**Risk thresholds:**

| Metric | Flag threshold |
|--------|----------------|
| `fraud_rate` | > 0.25% per group, > 0.3% overall |
| `failure_rate` | > 5.5% per group, > 6.0% overall |

---

### 2.5 Visualizer (`src/visualizer.py`)

All charts use a consistent dark theme (bg: `#0F172A`, surface: `#1E293B`).

| Chart type | When used |
|------------|-----------|
| Horizontal bar | comparison / ranking |
| Line + area fill | trend over time |
| Gauge | single scalar result |
| Heatmap | hour × day matrix |
| Donut | composition / share |
| Anomaly bar | anomaly intent (red = outlier) |

---

### 2.6 Streamlit UI (`app.py`)

**Features:**
- Persistent chat history in `st.session_state`
- Sidebar: 15 sample query buttons, schema info, options
- Global KPI header bar (always visible)
- Per-message: chart, insight panel, risk flags, data table, JSON panel
- Follow-up question buttons drive next query
- Dark CSS theme, DM Sans + Space Mono typography

---

## 3. Data Analysis Methodology

### 3.1 Dataset Profile

| Attribute | Value |
|-----------|-------|
| Total records | 250,000 |
| Date range | 2024 (full year) |
| States | 10 Indian states |
| Banks | 8 (SBI, HDFC, ICICI, Axis, Kotak, PNB, IndusInd, Yes Bank) |
| Categories | 10 merchant categories |
| Device types | Android, iOS, Web |
| Network types | 5G, 4G, 3G, WiFi |
| Transaction types | P2P, P2M, Bill Payment, Recharge |

### 3.2 Key Statistical Findings

**Fraud Analysis:**
- Overall fraud rate: 0.19% (480 transactions)
- Peak fraud hour: 3 AM (0.30%)
- Highest risk bank: Kotak (0.25%)
- Highest risk state: Karnataka, Rajasthan (0.23%)
- Most vulnerable age: 18–25 (0.23%)

**Failure Analysis:**
- Overall failure rate: 4.95% (12,376 transactions)
- Worst network: 3G (5.22%)
- Worst device: Web (5.15%)
- Worst state: Uttar Pradesh (5.22%)
- Worst day: Sunday (5.10%)

**Transaction Amounts:**
- Overall average: ₹1,311.76
- Highest category: Education (₹5,094)
- Lowest category: Transport (₹308)
- Highest age group: 36–45 (₹1,424)

**Volume Patterns:**
- Peak hour: 7 PM (21,232 txns)
- Dominant device: Android (75.1%)
- Largest state: Maharashtra (37,427 txns)
- Largest bank: SBI (62,693 txns)

---

## 4. Confidence Scoring

Each query is scored 0–1 based on:

```
confidence = (intent_conf + metric_conf + groupby_conf) / 3

where:
  intent_conf  = 0.6 + (keyword_hits * 0.1), capped at 0.95
  metric_conf  = 0.65 + (keyword_hits * 0.1), capped at 0.95
  groupby_conf = 0.6 + (keyword_hits * 0.1), capped at 0.95
               = 0.5 if no dimension detected
```

Labels: Very High (≥85%) · High (≥70%) · Medium (≥55%) · Low (<55%)

---

## 5. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Rule-based NLP over LLM | Deterministic, offline, zero latency, no API costs |
| pandas over SQL | Single-file deployment, no database setup required |
| Plotly over matplotlib | Interactive charts, dark theme, Streamlit native |
| dataclass AnalyticalQuery | Structured JSON output for explainability |
| Z-score anomaly detection | Simple, interpretable, no model training needed |
| Streamlit caching | 250k row CSV loads once per session |

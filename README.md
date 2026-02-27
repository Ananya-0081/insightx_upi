# ⚡ InsightX BI — Conversational UPI Payment Intelligence

> Ask anything about your UPI data in plain English. Get instant charts, insights, and risk flags — no SQL, no code.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://insightxupi-hzd9ybappvibnqhrd5tyign.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)

---

## 🚀 Live Demo

**[https://insightxupi-hzd9ybappvibnqhrd5tyign.streamlit.app/](https://insightxupi-hzd9ybappvibnqhrd5tyign.streamlit.app/)**

No installation required. Works in any modern browser.

---

## 📌 What is InsightX BI?

InsightX BI is a conversational business intelligence platform for UPI payment analytics. It analyses **250,000 real-world UPI transactions** from 2024 and lets anyone — not just data scientists — ask business questions in natural language.

### Key Features

| Feature | Description |
|---|---|
| 🧠 **Natural Language Queries** | Rule-based NLP parser — no LLM API required |
| 📊 **6 Chart Types** | Auto-selected: bar, line, donut, gauge, comparison, anomaly |
| 🔁 **Context Memory** | "What about Karnataka?" inherits previous metric automatically |
| ⚔️ **Comparison Mode** | "HDFC vs SBI", "3G vs 5G", "P2P vs P2M" — side-by-side |
| 🚨 **Risk Flags** | Automatic thresholds for fraud (>0.25%) and failure (>5.5%) |
| 📈 **Anomaly Detection** | Z-score based outlier detection across all grouped results |
| 💡 **Context-Aware Follow-ups** | Suggested next questions based on current query |

---

## 🏃 Run Locally

### Prerequisites
- Python 3.9+
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/insightx.git
cd insightx

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Requirements
```
streamlit
pandas
numpy
plotly
rapidfuzz
```

---

## 💬 Sample Queries to Try

```
# Basic metrics
"Overall summary"
"What is the total fraud rate?"

# Comparisons
"Compare fraud rate by device"
"Failure rate on 3G vs 5G"
"HDFC vs SBI comparison"
"Compare P2P vs P2M count"

# Rankings
"Which bank has the highest failure rate?"
"Top 5 states by total volume"

# Trends
"Fraud trend by day of week"
"Peak transaction hour"

# Filtered
"Evening transactions fraud rate"
"Maharashtra transaction analysis"
"Fraud rate for 18-25 age group"

# Context chains (try in sequence!)
"Compare fraud rate by device"   →   "What about only weekends?"
"Which bank has highest failure rate?"   →   "What about Karnataka?"

# Anomalies
"Anomaly detection in fraud rate by category"
```

---

## 🏗️ System Architecture

```
User Query (Natural Language)
         │
         ▼
┌─────────────────────┐
│   NLP Engine        │  Intent + Metric + Dimension + Filters + Compare
│   nlp_engine.py     │  Rule-based, no external API
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Context Memory     │  Merges follow-up queries with conversation history
│  context_memory.py  │  Sliding window of last 10 queries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Analytics Engine   │  Pandas aggregations on 250K row DataFrame
│  analytics_engine.py│  Filter → Compare → Group → Aggregate
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Response Generator  │  Narrative, bullets, risk flags, recommendations
│ response_generator.py│  Context-aware follow-up suggestions
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Visualizer        │  Plotly charts auto-selected by intent
│   visualizer.py     │  bar / line / donut / gauge / comparison / anomaly
└─────────┬───────────┘
          │
          ▼
   Streamlit UI (app.py)
```

---

## 📁 Project Structure

```
insightx/
├── app.py                    # Main Streamlit entry point
├── ui_components.py          # UI helper functions
├── requirements.txt
├── .gitignore
├── data/
│   └── upi_transactions_2024.csv   # 250,000 UPI transactions
├── docs/
│   └── architecture.md
└── src/
    ├── analytics_engine.py   # Query execution + aggregations
    ├── context_memory.py     # Multi-turn conversation memory
    ├── data_loader.py        # CSV loading + schema extraction
    ├── nlp_engine.py         # Natural language query parser
    ├── response_generator.py # Narrative + insight generation
    ├── system_prompt.py      # NLP rules and prompt config
    └── visualizer.py         # Plotly chart functions
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 250,000 |
| Date Range | 2024-01-01 to 2024-12-30 |
| States | 10 Indian states |
| Banks | 8 (HDFC, SBI, ICICI, Axis, Kotak, PNB, Yes Bank, IndusInd) |
| Categories | 10 merchant categories |
| Transaction Types | P2P, P2M, Bill Payment, Recharge |
| Networks | 3G, 4G, 5G, WiFi |
| Devices | Android, iOS, Web |

---

## 🧩 Supported Query Types

| Intent | Example | Chart |
|---|---|---|
| `single` | "Overall fraud rate" | Gauge |
| `comparison` | "Fraud rate by device" | Bar |
| `comparison` | "3G vs 5G failure rate" | Comparison Bar |
| `ranking` | "Top 5 banks by failure" | Bar |
| `trend` | "Fraud by day of week" | Line |
| `anomaly` | "Anomaly in fraud by category" | Anomaly Bar |

---

## 🔧 Deployment

Deployed on **Streamlit Cloud** (free tier):

1. Push code to GitHub (without `venv/`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → set `app.py` as main file → Deploy

Auto-redeploys on every `git push`.

---

## 👥 Team
Lannisters:

Aryan Singh
Ananya Sharma

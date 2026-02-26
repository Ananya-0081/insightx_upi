"""
insightx/tests/test_queries.py
────────────────────────────────
Runs all 15 sample queries through the full pipeline
(NLP → Analytics → Response) and prints a formatted report.

Run with:
    python tests/test_queries.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from pathlib import Path

# Standalone data loader (no streamlit) for test runner
def load_data():
    path = Path(__file__).parent.parent / "data" / "upi_transactions_2024.csv"
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={"transaction id":"transaction_id","transaction type":"transaction_type","amount (inr)":"amount_inr"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"]  = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["amount_inr"]  = pd.to_numeric(df["amount_inr"], errors="coerce")
    df["hour_of_day"] = pd.to_numeric(df["hour_of_day"], errors="coerce").astype("Int16")
    df["fraud_flag"]  = pd.to_numeric(df["fraud_flag"],  errors="coerce").astype("Int8")
    df["is_weekend"]  = pd.to_numeric(df["is_weekend"],  errors="coerce").astype("Int8")
    df["is_failed"] = (df["transaction_status"] == "FAILED").astype("Int8")
    df["is_fraud"]  = df["fraud_flag"]
    return df

from src.nlp_engine         import parse_query
from src.analytics_engine   import execute_query
from src.response_generator import generate_response

# ══════════════════════════════════════════════════════════════════════════
#  SAMPLE QUERY SET (15 diverse queries)
# ══════════════════════════════════════════════════════════════════════════

SAMPLE_QUERIES = [
    # 1 — Overall summary
    "Give me an overall summary of the dataset",

    # 2 — Single metric
    "What is the overall fraud rate?",

    # 3 — Comparison by device
    "Compare fraud rate by device type",

    # 4 — Comparison by bank
    "Which bank has the highest failure rate?",

    # 5 — Average amount by category
    "Show average transaction amount by merchant category",

    # 6 — Trend by day
    "How does fraud rate vary across days of the week?",

    # 7 — Trend by hour
    "What is the peak hour for transactions?",

    # 8 — State filter
    "What is the failure rate in Maharashtra?",

    # 9 — Network comparison
    "Compare failure rate between 5G, 4G, 3G, and WiFi",

    # 10 — Age group analysis
    "Which age group has the highest average transaction amount?",

    # 11 — Transaction type comparison
    "Compare total volume for P2P vs P2M vs Bill Payment",

    # 12 — Time-of-day filter
    "What is the fraud rate during late night hours?",

    # 13 — Ranking / top-N
    "Top 5 states by transaction count",

    # 14 — Anomaly detection
    "Detect anomalies in fraud rate by merchant category",

    # 15 — Multi-entity filter
    "What is the fraud rate for HDFC bank users on iOS?",
]


# ══════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_tests(df):
    results = []
    print("\n" + "="*80)
    print("  InsightX — Sample Query Test Suite (15 Queries)")
    print("="*80)

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n[{i:02d}] Query: \"{query}\"")
        print("-" * 60)

        # Step 1: Parse
        parsed = parse_query(query)
        print(f"  Intent  : {parsed.intent}")
        print(f"  Metric  : {parsed.metric}")
        print(f"  Group By: {parsed.group_by}")
        print(f"  Filters : {parsed.filters}")
        print(f"  Conf    : {parsed.confidence:.0%} — {parsed.confidence_reasoning[:80]}…")

        # Step 2: Execute
        result = execute_query(parsed, df)
        if not result.success:
            print(f"  ❌ ERROR: {result.error}")
            results.append({"query": query, "status": "ERROR", "error": result.error})
            continue

        # Step 3: Generate response
        response = generate_response(result, parsed.confidence)

        # Print result
        print(f"  Headline: {response.headline}")
        print(f"  Narrative (excerpt): {result.narrative[:120]}…")
        if result.scalar_fmt:
            print(f"  Value   : {result.scalar_fmt}")
        if result.result_df is not None:
            print(f"  Rows returned: {len(result.result_df)}")
        if result.anomalies:
            print(f"  Anomalies: {len(result.anomalies)}")
        print(f"  Chart type: {response.chart_type}")
        print(f"  Risk flags: {len(response.risk_flags)}")
        print(f"  Status  : ✅ SUCCESS")

        results.append({
            "query":      query,
            "status":     "SUCCESS",
            "intent":     parsed.intent,
            "metric":     parsed.metric,
            "group_by":   parsed.group_by,
            "confidence": parsed.confidence,
            "headline":   response.headline,
        })

    # Summary
    passed = sum(1 for r in results if r["status"] == "SUCCESS")
    print("\n" + "="*80)
    print(f"  Results: {passed}/{len(SAMPLE_QUERIES)} queries passed")
    print("="*80 + "\n")
    return results


if __name__ == "__main__":
    print("Loading dataset…")
    df = load_data()
    print(f"Loaded {len(df):,} rows.\n")
    run_tests(df)

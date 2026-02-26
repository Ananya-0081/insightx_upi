"""
insightx/src/data_loader.py
───────────────────────────
Loads and preprocesses the UPI transaction CSV.
Caches the processed DataFrame in Streamlit session state
so the 250 k rows are parsed only once per session.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "upi_transactions_2024.csv"

# ── column name normalisation map ─────────────────────────────────────────
_COL_RENAME = {
    "transaction id":   "transaction_id",
    "transaction type": "transaction_type",
    "amount (inr)":     "amount_inr",
    "amount_(inr)":     "amount_inr",
}

# ── canonical dtype definitions ───────────────────────────────────────────
_CATEGORICALS = [
    "transaction_type", "merchant_category", "transaction_status",
    "sender_age_group",  "receiver_age_group",
    "sender_state",      "sender_bank", "receiver_bank",
    "device_type",       "network_type", "day_of_week",
]

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


@st.cache_data(show_spinner="Loading 250 000 transactions…")
def load_data(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """
    Read the CSV, normalise column names, cast dtypes, and derive
    helper columns.  Result is cached for the Streamlit session.
    """
    df = pd.read_csv(path, low_memory=False)

    # ── normalise column names ────────────────────────────────────────────
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns=_COL_RENAME, inplace=True)

    # ── parse timestamp ───────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"]      = df["timestamp"].dt.date
    df["month"]     = df["timestamp"].dt.to_period("M").astype(str)
    df["quarter"]   = df["timestamp"].dt.to_period("Q").astype(str)

    # ── numeric columns ───────────────────────────────────────────────────
    df["amount_inr"]   = pd.to_numeric(df["amount_inr"],   errors="coerce")
    df["hour_of_day"]  = pd.to_numeric(df["hour_of_day"],  errors="coerce").astype("Int16")
    df["fraud_flag"]   = pd.to_numeric(df["fraud_flag"],   errors="coerce").astype("Int8")
    df["is_weekend"]   = pd.to_numeric(df["is_weekend"],   errors="coerce").astype("Int8")

    # ── boolean helpers ───────────────────────────────────────────────────
    df["is_failed"] = (df["transaction_status"] == "FAILED").astype("Int8")
    df["is_fraud"]  = df["fraud_flag"]

    # ── categorical columns ───────────────────────────────────────────────
    for col in _CATEGORICALS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # ── ordered weekday ───────────────────────────────────────────────────
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.Categorical(
            df["day_of_week"], categories=DAY_ORDER, ordered=True
        )

    return df


def get_schema_info(df: pd.DataFrame) -> dict:
    """Return a human-readable schema summary used by the NLP engine."""
    return {
        "total_rows":   len(df),
        "columns":      list(df.columns),
        "states":       sorted(df["sender_state"].dropna().unique().tolist()),
        "banks":        sorted(df["sender_bank"].dropna().unique().tolist()),
        "categories":   sorted(df["merchant_category"].dropna().unique().tolist()),
        "age_groups":   sorted(df["sender_age_group"].dropna().unique().tolist()),
        "devices":      sorted(df["device_type"].dropna().unique().tolist()),
        "networks":     sorted(df["network_type"].dropna().unique().tolist()),
        "txn_types":    sorted(df["transaction_type"].dropna().unique().tolist()),
        "days":         DAY_ORDER,
        "date_range":   (str(df["timestamp"].min().date()), str(df["timestamp"].max().date())),
    }

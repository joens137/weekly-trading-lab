"""Publish reproducible, non-personal research results to the connected GitHub repository.

This script runs in GitHub Actions, not inside Streamlit. It does NOT read Streamlit
session state, portfolio credentials, private logs, or user-specific settings.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from engine import (
    TICKERS, features_for, get_predictions, normalize_data, performance,
    run_backtest, score_log, weights_for,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "public_results"
WEEKS = 104
STARTING_DOLLARS = 100.0
COST_BPS = 5
FULL_ML = True

JOURNAL_COLUMNS = [
    "as_of", "strategy", "weights", "holding", "predicted",
    "starting_value", "forecast_value", "status", "actual",
    "actual_value", "error_pp", "direction_correct",
]


def get_market_data():
    raw = yf.download(
        list(TICKERS), period="10y", interval="1d", auto_adjust=True,
        progress=False, group_by="ticker", threads=True,
    )
    return normalize_data(raw)


def get_snapshot(closes, opens):
    features = features_for(opens, closes)
    idx = len(closes) - 1
    ridge, forest = get_predictions(features, idx, use_ml=FULL_ML)
    choices = weights_for(closes, idx, ridge, forest)
    observed = str(closes.index[idx].date())
    rows = []
    for strategy, (weights, prediction) in choices.items():
        rows.append({
            "as_of": observed,
            "strategy": strategy,
            "weights": repr(weights),
            "holding": ", ".join(f"{ticker}: {weight:.0%}" for ticker, weight in weights.items()) or "CASH",
            "predicted": float(prediction),
            "starting_value": STARTING_DOLLARS,
            "forecast_value": STARTING_DOLLARS * (1 + float(prediction)),
        })
    return pd.DataFrame(rows)


def update_forecast_journal(opens, closes, forecast):
    path = OUT / "weekly_forecasts.csv"
    if path.exists():
        journal = pd.read_csv(path)
    else:
        journal = pd.DataFrame(columns=JOURNAL_COLUMNS)
    if not journal.empty:
        required = {"as_of", "strategy", "weights", "predicted", "starting_value"}
        if not required.issubset(journal.columns):
            raise ValueError("Existing public forecast journal has an unexpected schema; refusing to overwrite.")
    existing = set()
    if not journal.empty:
        existing = set(zip(journal["as_of"].astype(str), journal["strategy"].astype(str)))
    incoming = forecast.loc[
        [ (str(row["as_of"]), str(row["strategy"])) not in existing for _, row in forecast.iterrows() ]
    ]
    if not incoming.empty:
        journal = pd.concat([journal, incoming], ignore_index=True)
    journal = score_log(opens, closes, journal)
    # Fail rather than silently delete old observations or duplicate them.
    if journal.duplicated(subset=["as_of", "strategy"]).any():
        raise ValueError("Duplicate journal snapshots detected; not publishing conflicting results.")
    journal.sort_values(["as_of", "strategy"], inplace=True)
    journal.to_csv(path, index=False, columns=JOURNAL_COLUMNS)
    return int((journal["status"] == "Scored").sum()), int((journal["status"] == "Pending").sum())


def main():
    opens, closes = get_market_data()
    bt = run_backtest(opens, closes, weeks=WEEKS, bps=COST_BPS, model_enabled=FULL_ML)
    if bt.empty:
        raise RuntimeError("Backtest produced no results; no snapshot will be published.")
    leaderboard, _ = performance(bt, STARTING_DOLLARS)
    latest = get_snapshot(closes, opens)

    OUT.mkdir(exist_ok=True)
    leaderboard.to_csv(OUT / "strategy_leaderboard.csv", index=False)
    bt.to_csv(OUT / "historical_backtest.csv", index=False)
    bt[bt["strategy"] == "Dual momentum"].to_csv(OUT / "dual_momentum_trades.csv", index=False)
    scored, pending = update_forecast_journal(opens, closes, latest)
    status = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_market_date": str(closes.index[-1].date()),
        "historical_weeks_requested": WEEKS,
        "historical_weeks_evaluated": int(bt["week"].nunique()),
        "starting_dollars_per_strategy": STARTING_DOLLARS,
        "cost_basis_points": COST_BPS,
        "full_ml_enabled": FULL_ML,
        "forecasts_scored": scored,
        "forecasts_pending": pending,
        "source": "GitHub Actions scheduled independent research replay",
        "disclaimer": "Public virtual research results, not actual Streamlit sessions or real trades.",
    }
    (OUT / "sync_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Portfolio construction for the Betting-Against-Beta (BAB) strategy.

This module loads pre-computed price/return artifacts, forms beta quintiles
using 60-month rolling betas, and generates the monthly BAB portfolio returns
and diagnostic statistics.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

DATA_DIR = "data"
RETURNS_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")
BETAS_FILE = os.path.join(DATA_DIR, "rolling_betas.csv")
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "bab_portfolios.csv")
MSCI_ETF_TICKER = "URTH"
QUINTILES = 5


def load_artifacts() -> Dict[str, pd.DataFrame]:
    returns = pd.read_csv(RETURNS_FILE, index_col="Date", parse_dates=True)
    betas = pd.read_csv(BETAS_FILE, index_col="Date", parse_dates=True)
    tickers_df = pd.read_csv(TICKERS_FILE)

    stock_tickers: List[str] = tickers_df["Ticker"].tolist()
    return {
        "returns": returns,
        "betas": betas,
        "stock_tickers": stock_tickers,
    }


def assign_quintiles(beta_series: pd.Series) -> pd.Series:
    labels = list(range(1, QUINTILES + 1))
    try:
        return pd.qcut(beta_series, QUINTILES, labels=labels)
    except ValueError:
        # When there are too few unique beta values, fall back to rank-based bins
        ranked = beta_series.rank(method="first")
        quantiles = np.linspace(0, 1, QUINTILES + 1)
        bins = ranked.quantile(quantiles).to_list()
        bins[0] = bins[0] - 1e-6
        return pd.cut(ranked, bins=bins, labels=labels, include_lowest=True)


def construct_bab_portfolios() -> pd.DataFrame:
    artifacts = load_artifacts()
    returns = artifacts["returns"]
    betas = artifacts["betas"]
    stock_tickers = artifacts["stock_tickers"]

    results = []
    common_index = returns.index.intersection(betas.index)

    for date in common_index:
        beta_slice = betas.loc[date, stock_tickers]
        return_slice = returns.loc[date, stock_tickers]
        joined = pd.DataFrame({"beta": beta_slice, "ret": return_slice}).dropna()

        if len(joined) < QUINTILES:
            continue

        joined["quintile"] = assign_quintiles(joined["beta"])

        q1 = joined[joined["quintile"] == 1]
        q5 = joined[joined["quintile"] == QUINTILES]

        if q1.empty or q5.empty:
            continue

        bab_return = q1["ret"].mean() - q5["ret"].mean()

        results.append(
            {
                "Date": date,
                "BAB_Return": bab_return,
                "Q1_Mean_Beta": q1["beta"].mean(),
                "Q5_Mean_Beta": q5["beta"].mean(),
                "Q1_Mean_Return": q1["ret"].mean(),
                "Q5_Mean_Return": q5["ret"].mean(),
                "N_Q1": len(q1),
                "N_Q5": len(q5),
            }
        )

    bab_df = pd.DataFrame(results).sort_values("Date")
    bab_df.set_index("Date", inplace=True)
    return bab_df


def main() -> None:
    bab_portfolios = construct_bab_portfolios()
    bab_portfolios.to_csv(OUTPUT_FILE, index_label="Date")
    print(f"Saved BAB portfolios to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

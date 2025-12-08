#!/usr/bin/env python3
"""
Backtesting module for the Betting-Against-Beta (BAB) strategy.

Loads BAB portfolio returns and MSCI World benchmark returns, computes monthly
and annualized performance statistics, and saves results to CSV files.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

DATA_DIR = "data"
BAB_PORTFOLIO_FILE = os.path.join(DATA_DIR, "bab_portfolios.csv")
RETURNS_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")
MONTHLY_OUTPUT_FILE = os.path.join(DATA_DIR, "bab_backtest_monthly.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "bab_backtest_summary.csv")
MSCI_ETF_TICKER = "URTH"


def load_inputs() -> Dict[str, pd.DataFrame]:
    bab = pd.read_csv(BAB_PORTFOLIO_FILE, index_col="Date", parse_dates=True)
    returns = pd.read_csv(RETURNS_FILE, index_col="Date", parse_dates=True)
    return {"bab": bab, "returns": returns}


def compute_drawdown(series: pd.Series) -> pd.Series:
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown


def summarize_performance(returns: pd.Series) -> Dict[str, float]:
    monthly_mean = returns.mean()
    monthly_std = returns.std()

    annual_return = monthly_mean * 12
    annual_vol = monthly_std * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    max_dd = compute_drawdown(returns).min()

    return {
        "Annualized_Return": annual_return,
        "Annualized_Volatility": annual_vol,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_dd,
    }


def run_backtest() -> None:
    inputs = load_inputs()
    bab = inputs["bab"]
    returns = inputs["returns"]

    bab_returns = bab["BAB_Return"].dropna()
    benchmark_returns = returns[MSCI_ETF_TICKER].reindex(bab_returns.index).dropna()

    aligned_index = bab_returns.index.intersection(benchmark_returns.index)
    bab_returns = bab_returns.loc[aligned_index]
    benchmark_returns = benchmark_returns.loc[aligned_index]

    bab_drawdown = compute_drawdown(bab_returns)
    bench_drawdown = compute_drawdown(benchmark_returns)

    monthly_perf = pd.DataFrame(
        {
            "BAB_Return": bab_returns,
            "Benchmark_Return": benchmark_returns,
            "Cumulative_BAB": (1 + bab_returns).cumprod(),
            "Cumulative_Benchmark": (1 + benchmark_returns).cumprod(),
            "Drawdown_BAB": bab_drawdown,
            "Drawdown_Benchmark": bench_drawdown,
        }
    )

    summary = {
        "BAB": summarize_performance(bab_returns),
        "Benchmark": summarize_performance(benchmark_returns),
    }

    monthly_perf.to_csv(MONTHLY_OUTPUT_FILE, index_label="Date")
    pd.DataFrame(summary).to_csv(SUMMARY_FILE)

    print(f"Saved monthly backtest results to {MONTHLY_OUTPUT_FILE}")
    print(f"Saved summary statistics to {SUMMARY_FILE}")


if __name__ == "__main__":
    run_backtest()

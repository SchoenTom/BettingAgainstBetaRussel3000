#!/usr/bin/env python3
"""
Visualization utilities for the Betting-Against-Beta (BAB) strategy.

Generates cumulative performance plots, rolling risk metrics, and beta spread
charts based on previously computed backtest results.
"""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data"
MONTHLY_FILE = os.path.join(DATA_DIR, "bab_backtest_monthly.csv")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "bab_portfolios.csv")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
ROLLING_WINDOW = 12


plt.style.use("ggplot")


def ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(MONTHLY_FILE, index_col="Date", parse_dates=True)
    portfolios = pd.read_csv(PORTFOLIO_FILE, index_col="Date", parse_dates=True)
    return monthly, portfolios


def plot_cumulative(monthly: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly[["Cumulative_BAB", "Cumulative_Benchmark"]].plot(ax=ax)
    ax.set_title("Cumulative Performance: BAB vs MSCI World")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    path = os.path.join(FIGURES_DIR, "cumulative_performance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rolling_sharpe(monthly: pd.DataFrame) -> str:
    bab_returns = monthly["BAB_Return"]
    rolling_mean = bab_returns.rolling(ROLLING_WINDOW).mean()
    rolling_std = bab_returns.rolling(ROLLING_WINDOW).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    rolling_sharpe.plot(ax=ax, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling {ROLLING_WINDOW}-Month Sharpe Ratio (BAB)")
    ax.set_ylabel("Sharpe Ratio")
    path = os.path.join(FIGURES_DIR, "rolling_sharpe.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_beta_spread(portfolios: pd.DataFrame) -> str:
    beta_spread = portfolios["Q5_Mean_Beta"] - portfolios["Q1_Mean_Beta"]
    fig, ax = plt.subplots(figsize=(10, 6))
    beta_spread.plot(ax=ax, color="darkred")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Beta Spread: Q5 - Q1")
    ax.set_ylabel("Beta Difference")
    path = os.path.join(FIGURES_DIR, "beta_spread.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ensure_figures_dir()
    monthly, portfolios = load_results()

    cumulative_path = plot_cumulative(monthly)
    sharpe_path = plot_rolling_sharpe(monthly)
    beta_spread_path = plot_beta_spread(portfolios)

    print("Saved plots:")
    for p in (cumulative_path, sharpe_path, beta_spread_path):
        print(f" - {p}")


if __name__ == "__main__":
    main()

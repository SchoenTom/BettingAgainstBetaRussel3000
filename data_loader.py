#!/usr/bin/env python3
"""
Betting-Against-Beta (BAB) data preparation script.

This module downloads the current MSCI World ETF constituents from Yahoo Finance,
cleans ticker symbols, fetches price history, computes returns, excess returns,
and 60-month rolling betas versus the MSCI World benchmark. All artifacts are
saved as CSV files under the local data directory for downstream modules to
consume.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf

# Configuration
DATA_DIR = "data"
START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ROLLING_WINDOW = 60
MSCI_ETF_TICKER = "URTH"  # iShares MSCI World ETF
RISK_FREE_TICKER = "^IRX"

# Output paths
PRICES_FILE = os.path.join(DATA_DIR, "monthly_prices.csv")
RETURNS_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")
EXCESS_RETURNS_FILE = os.path.join(DATA_DIR, "monthly_excess_returns.csv")
BETAS_FILE = os.path.join(DATA_DIR, "rolling_betas.csv")
RF_RATE_FILE = os.path.join(DATA_DIR, "risk_free_rate.csv")
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")


def ensure_data_dir() -> None:
    """Create the data directory if it does not already exist."""

    os.makedirs(DATA_DIR, exist_ok=True)


def _extract_ticker_column(df: pd.DataFrame) -> pd.Series:
    """Attempt to extract a ticker column from a holdings DataFrame."""

    for col in ("symbol", "Symbol", "ticker", "Ticker"):
        if col in df.columns:
            return df[col]
    # Fallback: if the index looks like tickers, return it
    return df.index.to_series()


def fetch_msci_world_constituents(etf_ticker: str = MSCI_ETF_TICKER) -> List[str]:
    """Retrieve the current MSCI World ETF constituents from Yahoo Finance."""

    etf = yf.Ticker(etf_ticker)
    holdings_sources = [
        getattr(etf, "fund_holdings", None),
        getattr(etf, "holdings", None),
    ]

    holdings_df: pd.DataFrame | None = None
    for source in holdings_sources:
        if isinstance(source, pd.DataFrame) and not source.empty:
            holdings_df = source.copy()
            break

    if holdings_df is None:
        # Try API-style accessor if available
        try:
            info = etf.get_fund_holding_info()
            raw_holdings = info.get("holdings") if isinstance(info, dict) else None
            if raw_holdings is not None:
                holdings_df = pd.DataFrame(raw_holdings)
        except Exception:
            holdings_df = None

    if holdings_df is None or holdings_df.empty:
        raise RuntimeError("Unable to retrieve MSCI World constituents from Yahoo Finance.")

    tickers = _extract_ticker_column(holdings_df)
    return clean_tickers(tickers)


def clean_tickers(raw_tickers: Iterable[str]) -> List[str]:
    """Filter raw ticker symbols down to standard equity tickers."""

    cleaned: List[str] = []
    pattern = re.compile(r"^[A-Z0-9][A-Z0-9\.-]{0,8}$")
    for ticker in raw_tickers:
        if not isinstance(ticker, str):
            continue
        symbol = ticker.strip().upper()
        if pattern.match(symbol):
            cleaned.append(symbol)
    return sorted(pd.unique(cleaned).tolist())


def download_monthly_prices(tickers: List[str]) -> pd.DataFrame:
    """Download monthly adjusted close prices for the provided tickers."""

    all_tickers = sorted(set(tickers + [MSCI_ETF_TICKER]))
    raw = yf.download(
        tickers=all_tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    if raw.empty:
        raise RuntimeError("No price data downloaded from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Adj Close", level=1, axis=1)
    else:
        prices = raw[["Adj Close"]].rename(columns={"Adj Close": all_tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def download_risk_free_rate() -> pd.Series:
    """Download the ^IRX series and convert to monthly decimal returns."""

    rf_daily = yf.download(
        tickers=RISK_FREE_TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if rf_daily.empty:
        raise RuntimeError("No risk-free rate data downloaded from Yahoo Finance.")

    rf_daily = rf_daily["Adj Close"].dropna()
    rf_daily.index = pd.to_datetime(rf_daily.index)
    rf_monthly = rf_daily.resample("M").last()
    monthly_decimal = np.power(1 + rf_monthly / 100.0, 1 / 12.0) - 1
    monthly_decimal.name = "RF"
    return monthly_decimal


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percentage monthly returns from adjusted close prices."""

    returns = prices.pct_change().dropna(how="all")
    returns.index.name = "Date"
    return returns


def compute_excess_returns(returns: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    """Subtract the risk-free rate from each month's returns."""

    aligned_rf = risk_free.reindex(returns.index).fillna(method="ffill")
    excess = returns.sub(aligned_rf, axis=0)
    excess.index.name = "Date"
    return excess


def compute_rolling_betas(excess_returns: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Estimate rolling betas using a 60-month window with no look-ahead bias."""

    if benchmark not in excess_returns.columns:
        raise KeyError(f"Benchmark column {benchmark} not found in excess returns.")

    benchmark_series = excess_returns[benchmark]
    benchmark_var = benchmark_series.rolling(ROLLING_WINDOW).var()
    betas = pd.DataFrame(index=excess_returns.index, columns=excess_returns.columns, dtype=float)

    for ticker in excess_returns.columns:
        cov = excess_returns[ticker].rolling(ROLLING_WINDOW).cov(benchmark_series)
        betas[ticker] = cov / benchmark_var

    # Shift forward one month to ensure betas are based solely on prior information
    betas = betas.shift(1)
    betas.index.name = "Date"
    return betas


def save_series(series: pd.Series, path: str) -> None:
    series.to_csv(path, header=True, index_label="Date")


def save_frame(frame: pd.DataFrame, path: str) -> None:
    frame.to_csv(path, index_label="Date")


def main() -> None:
    ensure_data_dir()

    tickers = fetch_msci_world_constituents()
    save_frame(pd.DataFrame({"Ticker": tickers}), TICKERS_FILE)

    prices = download_monthly_prices(tickers)
    returns = compute_simple_returns(prices)

    risk_free = download_risk_free_rate()
    save_series(risk_free, RF_RATE_FILE)

    excess_returns = compute_excess_returns(returns, risk_free)
    betas = compute_rolling_betas(excess_returns, MSCI_ETF_TICKER)

    save_frame(prices, PRICES_FILE)
    save_frame(returns, RETURNS_FILE)
    save_frame(excess_returns, EXCESS_RETURNS_FILE)
    save_frame(betas, BETAS_FILE)

    print(f"Saved tickers to {TICKERS_FILE}")
    print(f"Saved monthly prices to {PRICES_FILE}")
    print(f"Saved monthly returns to {RETURNS_FILE}")
    print(f"Saved excess returns to {EXCESS_RETURNS_FILE}")
    print(f"Saved rolling betas to {BETAS_FILE}")
    print(f"Saved risk-free rate to {RF_RATE_FILE}")


if __name__ == "__main__":
    main()

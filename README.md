# Betting-Against-Beta (BAB) Strategy

A modular Python implementation of the Betting-Against-Beta (BAB) factor on the MSCI World universe. The project downloads the current MSCI World ETF constituents from Yahoo Finance, builds monthly beta-quintile portfolios, and reports performance statistics and plots.

## Overview

- **Universe**: Current MSCI World constituents (proxied by iShares MSCI World ETF holdings) treated as a fixed set across the backtest to avoid changing membership.
- **Benchmark**: MSCI World ETF (URTH).
- **Period**: 2000-01-01 to present (limited by data availability).
- **Rebalancing**: Monthly.
- **Beta Estimation**: 60-month rolling window of excess returns, shifted forward one month to avoid look-ahead.
- **Portfolio Construction**: Equal-weight long Q1 (lowest beta quintile) and short Q5 (highest beta quintile) without leverage adjustments.

## Project Structure

```
Betting_Against_Beta/
├── data_loader.py            # Downloads holdings, prices, risk-free rate, returns, betas
├── portfolio_construction.py # Forms monthly beta quintiles and BAB returns
├── backtest.py               # Computes BAB vs benchmark performance statistics
├── illustrations.py          # Creates plots from the backtest output
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── data/                     # Generated artifacts (created by the scripts)
    ├── tickers.csv
    ├── monthly_prices.csv
    ├── monthly_returns.csv
    ├── monthly_excess_returns.csv
    ├── rolling_betas.csv
    ├── risk_free_rate.csv
    ├── bab_portfolios.csv
    ├── bab_backtest_monthly.csv
    └── bab_backtest_summary.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in order to reproduce the full workflow:

1. **Data Loading**
   ```bash
   python data_loader.py
   ```
   - Downloads current MSCI World holdings from Yahoo Finance.
   - Cleans ticker symbols and fetches monthly adjusted close prices for all constituents, the MSCI World ETF, and ^IRX.
   - Converts ^IRX to monthly decimal risk-free rates, computes monthly returns, excess returns, and 60-month rolling betas (shifted one month forward to remove look-ahead), and saves all CSV artifacts to `data/`.

2. **Portfolio Construction**
   ```bash
   python portfolio_construction.py
   ```
   - Forms monthly cross-sections, drops missing values, sorts stocks into beta quintiles, and computes the BAB return (Q1 average return minus Q5 average return).
   - Saves monthly BAB returns and quintile diagnostics to `data/bab_portfolios.csv`.

3. **Backtesting**
   ```bash
   python backtest.py
   ```
   - Merges BAB returns with benchmark MSCI World returns, computes annualized return, volatility, Sharpe ratio (rf = 0), and max drawdown.
   - Saves monthly performance to `data/bab_backtest_monthly.csv` and summary metrics to `data/bab_backtest_summary.csv`.

4. **Illustrations**
   ```bash
   python illustrations.py
   ```
   - Generates plots for cumulative performance, rolling 12-month Sharpe ratio, and the beta spread (Q5 minus Q1), saved under `data/figures/`.

## Methodology Notes

- **Risk-Free Rate**: ^IRX daily values are resampled to month-end and converted from annual percent to monthly decimal rates using `(1 + rate/100)^(1/12) - 1`.
- **Rolling Betas**: Calculated as the 60-month rolling covariance of stock excess returns with MSCI World excess returns divided by the benchmark variance, then shifted one month forward to ensure portfolios rely only on prior information.
- **Survivorship Bias Choice**: The current MSCI World constituents are treated as a fixed universe throughout the backtest to simplify data handling while avoiding historical membership changes.
- **Weights**: Portfolios are equal-weighted within quintiles with no leverage or dollar-neutral scaling.

## License

MIT License

# Betting-Against-Beta (BAB) Strategy

A Python implementation of the Betting-Against-Beta (BAB) factor strategy on the Russell 3000 universe.

## Overview

The Betting-Against-Beta (BAB) strategy is based on the empirical finding that low-beta stocks tend to outperform high-beta stocks on a risk-adjusted basis. This project implements a full BAB strategy with:

- **Universe**: Russell 3000 (proxied by iShares IWV ETF constituents)
- **Period**: January 2000 to present
- **Rebalancing**: Monthly
- **Beta Estimation**: 60-month rolling window
- **Portfolio Construction**: Long Q1 (lowest beta quintile), Short Q5 (highest beta quintile)

## Project Structure

```
Betting_Against_Beta/
├── data_loader.py           # Downloads and prepares all data
├── portfolio_construction.py # Forms BAB portfolios
├── backtest.py              # Computes performance statistics
├── illustrations.py         # Generates visualization plots
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Generated data files (created by data_loader.py)
│   ├── tickers.csv
│   ├── monthly_prices.csv
│   ├── monthly_returns.csv
│   ├── monthly_excess_returns.csv
│   ├── rolling_betas.csv
│   ├── risk_free_rate.csv
│   └── iwv_returns.csv
└── output/                 # Generated outputs (created by other scripts)
    ├── bab_returns.csv
    ├── quintile_statistics.csv
    ├── bab_backtest_summary.csv
    ├── bab_monthly_performance.csv
    └── figures/
        ├── cumulative_returns.png
        ├── rolling_sharpe.png
        ├── beta_spread.png
        └── ...
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Betting_Against_Beta
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in order:

### 1. Data Loading
```bash
python data_loader.py
```
Downloads Russell 3000 tickers, fetches price data, and computes:
- Monthly prices and returns
- Monthly excess returns (vs risk-free rate)
- Rolling 60-month betas

### 2. Portfolio Construction
```bash
python portfolio_construction.py
```
Loads saved data and constructs:
- Monthly beta quintile portfolios
- BAB returns (Q1 - Q5 equal-weighted)
- Quintile statistics

### 3. Backtesting
```bash
python backtest.py
```
Computes performance metrics:
- Annualized return and volatility
- Sharpe ratio (rf = 0)
- Maximum drawdown
- Calmar ratio, Sortino ratio
- Alpha and Information ratio vs IWV

### 4. Visualizations
```bash
python illustrations.py
```
Generates plots:
- Cumulative equity curves (BAB vs IWV)
- Rolling 12-month Sharpe ratio
- Rolling excess return over benchmark
- Beta spread over time
- Drawdown comparison
- Annual returns comparison
- And more...

## Methodology

### Beta Estimation
Betas are estimated using a 60-month rolling window:
```
Beta = Cov(R_stock - R_f, R_market - R_f) / Var(R_market - R_f)
```

Where:
- R_stock = Monthly stock return
- R_market = Monthly IWV (Russell 3000) return
- R_f = Monthly risk-free rate (from ^IRX, converted from annual % to monthly decimal)

### Portfolio Construction
Each month:
1. Sort stocks into 5 equal-sized quintiles based on prior month's beta
2. Q1 = lowest beta stocks, Q5 = highest beta stocks
3. BAB_Return = Equal-weight mean(Q1 returns) - Equal-weight mean(Q5 returns)

### Notes on Implementation
- **Survivorship Bias**: This implementation uses current IWV constituents as a fixed universe throughout the backtest period. This introduces survivorship bias but simplifies data collection.
- **Look-Ahead Bias**: Portfolios are formed using betas computed at month t-1 to predict returns at month t.
- **No Leverage Adjustment**: Unlike the original Frazzini-Pedersen paper, this implementation does not apply leverage/de-leverage adjustments to make the strategy dollar-neutral at each beta level.

## Output Files

### Data Files (`data/`)
| File | Description |
|------|-------------|
| `tickers.csv` | List of Russell 3000 tickers |
| `monthly_prices.csv` | Monthly adjusted close prices |
| `monthly_returns.csv` | Simple monthly returns |
| `monthly_excess_returns.csv` | Returns minus risk-free rate |
| `rolling_betas.csv` | 60-month rolling betas |
| `risk_free_rate.csv` | Monthly risk-free rate |
| `iwv_returns.csv` | IWV benchmark returns |

### Output Files (`output/`)
| File | Description |
|------|-------------|
| `bab_returns.csv` | Monthly BAB strategy returns with quintile statistics |
| `quintile_statistics.csv` | Detailed quintile-level statistics |
| `bab_backtest_summary.csv` | Summary performance metrics |
| `bab_monthly_performance.csv` | Full monthly performance data |

## Performance Metrics

The backtest computes:
- **Annualized Return**: Geometric average annual return
- **Annualized Volatility**: Standard deviation * sqrt(12)
- **Sharpe Ratio**: Annualized return / Annualized volatility (rf = 0)
- **Sortino Ratio**: Annualized return / Downside deviation
- **Max Drawdown**: Maximum peak-to-trough decline
- **Calmar Ratio**: Annualized return / |Max Drawdown|
- **Information Ratio**: Active return / Tracking error vs IWV
- **Win Rate**: Percentage of positive monthly returns

## References

- Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.
- iShares Russell 3000 ETF (IWV): https://www.ishares.com/us/products/239714/

## License

MIT License

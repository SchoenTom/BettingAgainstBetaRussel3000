# Betting-Against-Beta (BAB) Strategy

Frazzini & Pedersen (2014) BAB implementation for Russell 3000.

## Overview

The BAB strategy exploits the finding that low-beta stocks earn higher risk-adjusted returns than predicted by CAPM. This implementation follows the methodology from F&P (2014) Table 4:

- **Beta calculation**: Correlation (12m) x Volatility ratio (60m) with shrinkage
- **Portfolio construction**: Beta-rank weighted, rescaled to beta=1 per leg
- **Market neutrality**: $2 gross exposure, ex-ante beta ~0

## Quick Start

```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard.py
```

## Files

```
├── config.py              # Central configuration
├── data_loader.py         # Data download + F&P beta calculation
├── portfolio_construction.py  # BAB portfolio construction
├── backtest.py            # Performance statistics
├── illustrations.py       # PNG visualizations
├── dashboard.py           # Streamlit dashboard
├── main.py                # Pipeline orchestrator
└── requirements.txt
```

## Methodology

### Beta Calculation (Frazzini-Pedersen)

```python
# Correlation over 12 months
rolling_corr = stock.rolling(12).corr(market)

# Volatility over 60 months
vol_ratio = stock.rolling(60).std() / market.rolling(60).std()

# Time-series beta
beta_ts = rolling_corr * vol_ratio

# Shrinkage toward 1
beta = 0.6 * beta_ts + 0.4 * 1.0
```

### BAB Construction (Table 4)

Each month:
1. Split stocks at median beta into low-beta and high-beta groups
2. Weight stocks by beta rank within each group
3. Rescale both portfolios to beta=1
4. BAB = Long low-beta - Short high-beta

```python
# Raw weights to rescale to beta=1
w_L_raw = 1.0 / beta_L
w_H_raw = 1.0 / beta_H

# Normalize to $2 gross ($1 per leg)
gross = w_L_raw + w_H_raw
w_L = 2.0 * w_L_raw / gross
w_H = 2.0 * w_H_raw / gross

# BAB return
bab = w_L * r_L - w_H * r_H
```

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| START_DATE | 1995-01-01 | Beta estimation start |
| END_DATE | 2014-12-31 | Pre-publication |
| CORRELATION_WINDOW | 12 | Months for correlation |
| VOLATILITY_WINDOW | 60 | Months for volatility |
| SHRINKAGE_FACTOR | 0.6 | Beta shrinkage toward 1 |
| NUM_DECILES | 10 | Decile portfolios |

## Usage

### Full Pipeline
```bash
python main.py
```

### Skip Download (use existing data)
```bash
python main.py --skip-download
```

### Individual Steps
```bash
python data_loader.py
python portfolio_construction.py
python backtest.py
python illustrations.py
```

### Dashboard
```bash
streamlit run dashboard.py
```

## Expected Results

With correct implementation (accounting for survivorship bias):

| Metric | Expected |
|--------|----------|
| Annualized Return | 1-5% |
| Sharpe Ratio | 0.1-0.5 |
| Max Drawdown | -50% to -80% |
| Ex-Ante Beta | ~0 |

## Important Notes

**Survivorship Bias**: Uses current Russell 3000 constituents historically. Results are indicative only.

**Time Period**: Ends 2014 (pre-F&P publication) to avoid look-ahead bias.

**Data Source**: Yahoo Finance, Ken French Data Library.

## References

- Frazzini, A. & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.

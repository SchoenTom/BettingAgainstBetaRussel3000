# Betting-Against-Beta (BAB) Strategy

A Python implementation replicating the Betting-Against-Beta (BAB) factor strategy from Frazzini and Pedersen (2014) on the Russell 3000 universe.

## Overview

The Betting-Against-Beta (BAB) strategy exploits the empirical finding that the security market line is too flat—low-beta stocks earn higher risk-adjusted returns than high-beta stocks. This implementation follows the methodology of Frazzini and Pedersen (2014), including the critical **beta scaling** that creates a market-neutral factor.

### Key Features

- **Universe**: Russell 3000 (proxied by iShares IWV ETF constituents)
- **Period**: January 2000 to present
- **Rebalancing**: Monthly
- **Beta Estimation**: 60-month rolling window with 36-month minimum
- **Beta Scaling**: Long and short legs scaled to beta ≈ 1 each (Frazzini-Pedersen methodology)
- **Market Neutrality**: Combined portfolio targets beta ≈ 0
- **Factor Regressions**: FF3, Carhart 4-factor, and FF5 model analysis

## Important: Survivorship Bias Declaration

**This implementation uses current IWV constituents as a fixed universe throughout the entire backtest period (2000-present). This introduces significant survivorship bias:**

1. **Selection Bias**: Companies that survived to today are included, while those that went bankrupt, were acquired, or delisted are excluded.

2. **Performance Inflation**: Survivorship bias typically inflates backtest performance by 1-2% annually, as failed companies (often high-beta) are excluded from the analysis.

3. **Beta Distribution Bias**: The beta distribution of surviving companies may differ systematically from the historical cross-section. High-beta companies that failed are excluded, potentially understating the true performance differential.

4. **Academic Benchmark**: Frazzini and Pedersen (2014) used point-in-time constituent data, which this simplified implementation cannot replicate without expensive historical index membership data.

**Interpretation**: Results should be viewed as indicative of the BAB premium's existence rather than as precise estimates of realizable returns. The survivorship bias caveat applies to all performance statistics, factor regressions, and visualizations.

## Project Structure

```
Betting_Against_Beta/
├── data_loader.py           # Downloads stock data, computes betas, fetches FF factors
├── portfolio_construction.py # Forms BAB portfolios with beta scaling
├── factor_regressions.py    # Runs FF3, Carhart, FF5 regressions
├── backtest.py              # Computes performance stats & market neutrality
├── illustrations.py         # Generates all visualization plots
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Generated data files
│   ├── tickers.csv
│   ├── monthly_prices.csv
│   ├── monthly_returns.csv
│   ├── monthly_excess_returns.csv
│   ├── rolling_betas.csv
│   ├── risk_free_rate.csv
│   ├── iwv_returns.csv
│   ├── iwv_excess_returns.csv
│   └── ff_factors.csv       # Fama-French + Momentum factors
└── output/                 # Generated outputs
    ├── bab_returns.csv
    ├── bab_returns_unscaled.csv
    ├── quintile_statistics.csv
    ├── leverage_history.csv
    ├── bab_backtest_summary.csv
    ├── bab_monthly_performance.csv
    ├── rolling_portfolio_beta.csv
    ├── market_neutrality_analysis.csv
    ├── factor_regression_results.csv
    └── figures/
        ├── cumulative_returns.png
        ├── rolling_sharpe.png
        ├── beta_spread.png
        ├── rolling_portfolio_beta.png
        ├── scaled_vs_unscaled.png
        ├── leverage_over_time.png
        ├── factor_regression_summary.png
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
- Rolling 60-month betas (36-month minimum)
- IWV benchmark returns (raw and excess)
- Fama-French factors (FF5 + Momentum)

### 2. Portfolio Construction
```bash
python portfolio_construction.py
```
Constructs BAB portfolios with Frazzini-Pedersen beta scaling:
- Monthly beta quintile sorting
- **Beta-scaled returns** for market neutrality
- Both scaled (BAB) and unscaled versions for comparison
- Leverage tracking (1/β for each leg)

### 3. Factor Regressions
```bash
python factor_regressions.py
```
Runs time-series regressions against standard asset pricing models:
- CAPM (market factor only)
- Fama-French 3-factor (Mkt-RF, SMB, HML)
- Carhart 4-factor (+ Momentum)
- Fama-French 5-factor (+ RMW, CMA)
- FF5 + Momentum (6-factor)

### 4. Backtesting
```bash
python backtest.py
```
Computes performance metrics and verifies market neutrality:
- Annualized return and volatility
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown analysis
- **Rolling portfolio beta** (should be ≈ 0 if properly scaled)
- Market neutrality statistics

### 5. Visualizations
```bash
python illustrations.py
```
Generates comprehensive plots including:
- Cumulative equity curves (BAB vs IWV)
- Rolling 12-month Sharpe ratio
- Beta spread over time (Q5 - Q1)
- **Rolling portfolio beta** (market neutrality verification)
- Scaled vs. unscaled BAB comparison
- Leverage over time (1/β_L and 1/β_H)
- Factor regression summary (alphas across models)
- Drawdown comparison
- Annual returns comparison

## Methodology

### Beta Estimation

Betas are estimated using a 60-month rolling window with a 36-month minimum requirement:

```
β_i,t = Cov(r_i - r_f, r_m - r_f) / Var(r_m - r_f)
```

Where:
- r_i = Monthly stock return
- r_m = Monthly IWV (Russell 3000) return
- r_f = Monthly risk-free rate (from 13-week T-bill)

The 36-month minimum ensures statistically meaningful beta estimates while allowing newer stocks to enter the universe.

### Portfolio Construction with Beta Scaling

This is the **critical innovation** from Frazzini and Pedersen (2014). Each month:

1. **Sort**: Rank stocks by prior month's beta into 5 equal-sized quintiles
   - Q1 = Lowest beta stocks (defensive)
   - Q5 = Highest beta stocks (aggressive)

2. **Compute Portfolio Returns**: Equal-weighted returns for each quintile
   ```
   r_L,t = mean(Q1 stock returns)  [Long leg]
   r_H,t = mean(Q5 stock returns)  [Short leg]
   ```

3. **Compute Portfolio Betas**: Average beta of stocks in each quintile
   ```
   β_L = mean(Q1 stock betas)
   β_H = mean(Q5 stock betas)
   ```

4. **Beta Scaling**: Scale each leg to achieve beta ≈ 1
   ```
   Scaled Long Return  = (1/β_L) × r_L,t
   Scaled Short Return = (1/β_H) × r_H,t
   ```

5. **BAB Return**: Combine scaled legs (market neutral)
   ```
   BAB_t = (1/β_L) × r_L,t - (1/β_H) × r_H,t
   ```

### Why Beta Scaling Matters

Without scaling, a simple Q1 - Q5 spread has negative market beta (since β_L < 1 < β_H). This makes the strategy:
- Long market in bull markets (hurting short leg performance)
- Short market in bear markets (hurting long leg performance)

Beta scaling creates a **market-neutral** portfolio with β ≈ 0, isolating the pure BAB premium from market timing effects.

### Leverage Interpretation

The scaling factors (1/β_L and 1/β_H) represent leverage:
- If β_L = 0.6, the long leg is levered 1.67x
- If β_H = 1.4, the short leg is levered 0.71x

This leverage is tracked in `leverage_history.csv` for transparency.

### Market Neutrality Verification

The implementation verifies market neutrality by:
1. Computing rolling 36-month portfolio beta of BAB returns vs. market
2. Checking that average portfolio beta is close to zero
3. Flagging periods where |β_portfolio| > 0.1

Results are saved to `rolling_portfolio_beta.csv` and `market_neutrality_analysis.csv`.

### Factor Regressions

Time-series regressions test whether BAB alpha survives after controlling for known factors:

| Model | Factors |
|-------|---------|
| CAPM | Mkt-RF |
| FF3 | Mkt-RF, SMB, HML |
| Carhart | Mkt-RF, SMB, HML, Mom |
| FF5 | Mkt-RF, SMB, HML, RMW, CMA |
| FF5+Mom | Mkt-RF, SMB, HML, RMW, CMA, Mom |

For each model, we report:
- **Alpha (α)**: Monthly abnormal return, annualized
- **t-statistic**: Statistical significance of alpha
- **Factor Loadings**: Exposure to each factor
- **R²**: Variance explained by the model

Frazzini and Pedersen (2014) find significant positive alpha across all models, consistent with BAB representing a distinct anomaly.

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
| `iwv_excess_returns.csv` | IWV excess returns |
| `ff_factors.csv` | Fama-French 5 factors + Momentum |

### Output Files (`output/`)

| File | Description |
|------|-------------|
| `bab_returns.csv` | Monthly BAB returns (beta-scaled) |
| `bab_returns_unscaled.csv` | Monthly BAB returns (naive Q1-Q5) |
| `quintile_statistics.csv` | Detailed quintile-level statistics |
| `leverage_history.csv` | Monthly leverage (1/β) for each leg |
| `bab_backtest_summary.csv` | Summary performance metrics |
| `bab_monthly_performance.csv` | Full monthly performance data |
| `rolling_portfolio_beta.csv` | Rolling 36-month portfolio beta |
| `market_neutrality_analysis.csv` | Market neutrality statistics |
| `factor_regression_results.csv` | Alpha, t-stats, loadings, R² |

## Performance Metrics

The backtest computes:
- **Annualized Return**: Geometric average annual return
- **Annualized Volatility**: Standard deviation × √12
- **Sharpe Ratio**: Annualized excess return / Annualized volatility
- **Sortino Ratio**: Annualized return / Downside deviation
- **Max Drawdown**: Maximum peak-to-trough decline
- **Calmar Ratio**: Annualized return / |Max Drawdown|
- **Information Ratio**: Active return / Tracking error vs IWV
- **Win Rate**: Percentage of positive monthly returns
- **Average Portfolio Beta**: Should be ≈ 0 for scaled BAB

## Expected Results

Based on Frazzini and Pedersen (2014) and similar implementations:

1. **Positive BAB Premium**: Low-beta stocks outperform high-beta stocks on a risk-adjusted basis

2. **Market Neutrality**: Rolling portfolio beta should hover around zero (±0.1)

3. **Significant Alpha**: Alpha should be positive and statistically significant (t > 2) across factor models

4. **Low Factor Exposure**: BAB should have low correlation with Mkt-RF, SMB, HML after scaling

**Caveat**: Due to survivorship bias, actual results may overstate the premium. See survivorship bias declaration above.

## Comparison: Scaled vs. Unscaled BAB

The implementation provides both versions for comparison:

| Metric | Scaled BAB | Unscaled (Q1-Q5) |
|--------|------------|------------------|
| Market Beta | ≈ 0 | Negative |
| Bull Market | Pure alpha | Underperforms (short market) |
| Bear Market | Pure alpha | Outperforms (short market) |
| Interpretation | BAB premium | BAB + market timing |

## References

- Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25. https://doi.org/10.1016/j.jfineco.2013.10.005

- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.

- Carhart, M. M. (1997). On persistence in mutual fund performance. *The Journal of Finance*, 52(1), 57-82.

- Fama-French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

- iShares Russell 3000 ETF (IWV): https://www.ishares.com/us/products/239714/

## License

MIT License

"""
backtest.py - Compute performance statistics for BAB strategy

Computes:
- Annualized return, volatility, Sharpe ratio
- Maximum drawdown, Sortino ratio, Calmar ratio
- CAPM alpha and beta
- Rolling performance metrics
"""

import pandas as pd
import numpy as np
import os
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import DATA_DIR, OUTPUT_DIR, PERIODS_PER_YEAR, ensure_directories


def load_data():
    """Load BAB portfolio and benchmark data."""
    logger.info("Loading data...")

    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
    excess_path = os.path.join(DATA_DIR, 'monthly_excess_returns.csv')

    if not os.path.exists(bab_path):
        raise FileNotFoundError("Run portfolio_construction.py first.")

    bab_df = pd.read_csv(bab_path, index_col=0, parse_dates=True)
    excess_returns = pd.read_csv(excess_path, index_col=0, parse_dates=True)

    benchmark = excess_returns['Benchmark'] if 'Benchmark' in excess_returns.columns else None

    logger.info(f"BAB: {len(bab_df)} months, Benchmark: {len(benchmark) if benchmark is not None else 0}")
    return bab_df, benchmark


def cumulative_returns(returns):
    """Growth of $1."""
    return (1 + returns).cumprod()


def annualized_return(returns):
    """Annualized return from monthly returns."""
    return returns.mean() * PERIODS_PER_YEAR


def annualized_volatility(returns):
    """Annualized volatility from monthly returns."""
    return returns.std() * np.sqrt(PERIODS_PER_YEAR)


def sharpe_ratio(returns):
    """Sharpe ratio (assumes excess returns)."""
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    return ann_ret / ann_vol if ann_vol > 0 else 0


def max_drawdown(returns):
    """Maximum drawdown."""
    cum = cumulative_returns(returns)
    running_max = cum.expanding().max()
    drawdowns = cum / running_max - 1
    return abs(drawdowns.min())


def sortino_ratio(returns):
    """Sortino ratio (downside deviation)."""
    ann_ret = annualized_return(returns)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    downside_vol = downside.std() * np.sqrt(PERIODS_PER_YEAR)
    return ann_ret / downside_vol if downside_vol > 0 else 0


def calmar_ratio(returns):
    """Calmar ratio (return / max drawdown)."""
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(returns)
    return ann_ret / mdd if mdd > 0 else 0


def run_capm(strategy, benchmark):
    """CAPM regression: r_s = alpha + beta * r_m."""
    common = strategy.index.intersection(benchmark.index)
    y = strategy.loc[common].dropna()
    x = benchmark.loc[common].dropna()
    common = y.index.intersection(x.index)
    y = y.loc[common]
    x = x.loc[common]

    n = len(y)
    if n < 12:
        return {'alpha': np.nan, 'beta': np.nan, 'alpha_t': np.nan, 'beta_t': np.nan, 'r2': np.nan}

    X = np.column_stack([np.ones(n), x.values])
    try:
        beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y.values)
    except np.linalg.LinAlgError:
        return {'alpha': np.nan, 'beta': np.nan, 'alpha_t': np.nan, 'beta_t': np.nan, 'r2': np.nan}

    residuals = y.values - X @ beta_hat
    sigma2 = (residuals @ residuals) / (n - 2)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta_hat / se_beta

    y_hat = X @ beta_hat
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y.values - y_hat) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'alpha': beta_hat[0],
        'beta': beta_hat[1],
        'alpha_t': t_stats[0],
        'beta_t': t_stats[1],
        'r2': r2,
        'n': n
    }


def compute_statistics(bab_returns, benchmark_returns):
    """Compute all performance statistics."""
    logger.info("Computing statistics...")

    common = bab_returns.index.intersection(benchmark_returns.index)
    bab = bab_returns.loc[common]
    bench = benchmark_returns.loc[common]

    capm = run_capm(bab, bench)

    stats = {
        'Start_Date': bab.index.min().strftime('%Y-%m-%d'),
        'End_Date': bab.index.max().strftime('%Y-%m-%d'),
        'N_Months': len(bab),
        'Total_Return': (1 + bab).prod() - 1,
        'Annualized_Return': annualized_return(bab),
        'Annualized_Volatility': annualized_volatility(bab),
        'Sharpe_Ratio': sharpe_ratio(bab),
        'Sortino_Ratio': sortino_ratio(bab),
        'Max_Drawdown': max_drawdown(bab),
        'Calmar_Ratio': calmar_ratio(bab),
        'Win_Rate': (bab > 0).mean(),
        'Best_Month': bab.max(),
        'Worst_Month': bab.min(),
        'Skewness': bab.skew(),
        'Kurtosis': bab.kurtosis(),
        'Beta_to_Benchmark': capm['beta'],
        'Alpha_Monthly': capm['alpha'],
        'Alpha_Annualized': capm['alpha'] * 12 if not pd.isna(capm['alpha']) else np.nan,
        'Alpha_t': capm['alpha_t'],
        'Beta_t': capm['beta_t'],
        'CAPM_R2': capm['r2'],
        'Benchmark_Ann_Return': annualized_return(bench),
        'Benchmark_Ann_Vol': annualized_volatility(bench),
        'Benchmark_Sharpe': sharpe_ratio(bench),
        'Benchmark_Max_DD': max_drawdown(bench),
    }

    return stats


def create_monthly_performance(bab_df, benchmark):
    """Create monthly performance DataFrame."""
    logger.info("Creating monthly performance...")

    perf = bab_df.copy()

    common = perf.index.intersection(benchmark.index)
    perf = perf.loc[common]
    perf['Benchmark_Return'] = benchmark.loc[common]

    perf['BAB_Cumulative'] = cumulative_returns(perf['BAB_Return'])
    perf['Benchmark_Cumulative'] = cumulative_returns(perf['Benchmark_Return'])

    perf['BAB_Drawdown'] = perf['BAB_Cumulative'] / perf['BAB_Cumulative'].expanding().max() - 1
    perf['Benchmark_Drawdown'] = perf['Benchmark_Cumulative'] / perf['Benchmark_Cumulative'].expanding().max() - 1

    # Rolling Sharpe (12-month)
    perf['Rolling_12M_BAB_Sharpe'] = perf['BAB_Return'].rolling(12).apply(sharpe_ratio, raw=False)
    perf['Rolling_12M_Benchmark_Sharpe'] = perf['Benchmark_Return'].rolling(12).apply(sharpe_ratio, raw=False)

    return perf


def save_outputs(stats, monthly_perf):
    """Save outputs."""
    ensure_directories()

    summary_df = pd.DataFrame([stats])
    summary_path = os.path.join(OUTPUT_DIR, 'bab_backtest_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved: {summary_path}")

    monthly_path = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    monthly_perf.to_csv(monthly_path)
    logger.info(f"Saved: {monthly_path}")


def print_summary(stats):
    """Print summary."""
    print("\n" + "=" * 60)
    print("BAB Backtest Results")
    print("=" * 60)

    print(f"\nPeriod: {stats['Start_Date']} to {stats['End_Date']}")
    print(f"Months: {stats['N_Months']}")

    def stars(t):
        if pd.isna(t): return ""
        if abs(t) > 2.58: return "***"
        if abs(t) > 1.96: return "**"
        if abs(t) > 1.65: return "*"
        return ""

    print("\n--- Performance ---")
    print(f"Annualized Return:  {stats['Annualized_Return']*100:>8.2f}%")
    print(f"Annualized Vol:     {stats['Annualized_Volatility']*100:>8.2f}%")
    print(f"Sharpe Ratio:       {stats['Sharpe_Ratio']:>8.3f}")
    print(f"Max Drawdown:       {stats['Max_Drawdown']*100:>8.2f}%")

    print("\n--- CAPM ---")
    print(f"Beta:               {stats['Beta_to_Benchmark']:>8.4f} (t={stats['Beta_t']:.2f})")
    print(f"Alpha (ann.):       {stats['Alpha_Annualized']*100:>8.2f}%{stars(stats['Alpha_t'])}")

    print("\n--- Benchmark (S&P 500) ---")
    print(f"Ann. Return:        {stats['Benchmark_Ann_Return']*100:>8.2f}%")
    print(f"Sharpe Ratio:       {stats['Benchmark_Sharpe']:>8.3f}")

    print("\n" + "=" * 60)


def main():
    """Main pipeline."""
    logger.info("=" * 60)
    logger.info("Backtest Analysis")
    logger.info("=" * 60)

    ensure_directories()

    bab_df, benchmark = load_data()

    if benchmark is None:
        raise ValueError("Benchmark data not found")

    stats = compute_statistics(bab_df['BAB_Return'], benchmark)
    monthly_perf = create_monthly_performance(bab_df, benchmark)
    save_outputs(stats, monthly_perf)
    print_summary(stats)

    logger.info("Backtest complete!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
backtest.py - Betting-Against-Beta (BAB) Strategy Backtesting

Replication of Frazzini and Pedersen (2014) "Betting Against Beta"
Journal of Financial Economics, 111(1), 1-25.

================================================================================
PERFORMANCE DIAGNOSTICS
================================================================================

This script computes comprehensive performance statistics:

1. RETURN METRICS:
   - Annualized return (geometric)
   - Monthly mean and median
   - Best/worst months

2. RISK METRICS:
   - Annualized volatility
   - Maximum drawdown
   - Skewness and kurtosis

3. RISK-ADJUSTED METRICS:
   - Sharpe ratio (rf=0 for BAB since already market-neutral)
   - Sortino ratio (downside risk)
   - Calmar ratio (return/max drawdown)

4. MARKET NEUTRALITY VERIFICATION:
   - Rolling portfolio beta (should stay near 0)
   - Correlation with market
   - Time-varying beta analysis

5. COMPARISON WITH BENCHMARK:
   - Alpha vs IWV
   - Information ratio
   - Beta to market

================================================================================

Author: BAB Strategy Implementation (Frazzini-Pedersen Replication)
Date: 2024
"""

import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "output"
DATA_DIR = "data"

# Input file paths
BAB_RETURNS_FILE = os.path.join(OUTPUT_DIR, "bab_returns.csv")
IWV_RETURNS_FILE = os.path.join(DATA_DIR, "iwv_returns.csv")

# Output file paths
BACKTEST_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "bab_backtest_summary.csv")
MONTHLY_PERFORMANCE_FILE = os.path.join(OUTPUT_DIR, "bab_monthly_performance.csv")
ROLLING_BETA_FILE = os.path.join(OUTPUT_DIR, "rolling_portfolio_beta.csv")


def load_data() -> tuple:
    """
    Load BAB returns and IWV returns data.

    Returns:
        Tuple of (BAB returns DataFrame, IWV returns DataFrame)
    """
    print("Loading data...")

    bab_returns = pd.read_csv(BAB_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  BAB Returns: {len(bab_returns)} months")

    iwv_returns = pd.read_csv(IWV_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  IWV Returns: {len(iwv_returns)} months")

    return bab_returns, iwv_returns


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from a series of simple returns.

    Args:
        returns: Series of simple returns

    Returns:
        Series of cumulative returns (wealth index starting at 1)
    """
    return (1 + returns).cumprod()


def compute_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series from cumulative returns.

    Args:
        cumulative_returns: Series of cumulative returns (wealth index)

    Returns:
        Series of drawdowns (negative values)
    """
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from returns series.

    Args:
        returns: Series of simple returns

    Returns:
        Maximum drawdown as a negative decimal
    """
    cum_returns = compute_cumulative_returns(returns)
    drawdown = compute_drawdown(cum_returns)
    return drawdown.min()


def compute_annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Compute annualized return from periodic returns.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Annualized return as decimal
    """
    total_return = (1 + returns).prod()
    n_periods = len(returns)
    n_years = n_periods / periods_per_year
    return total_return ** (1 / n_years) - 1


def compute_annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Compute annualized volatility from periodic returns.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Annualized volatility as decimal
    """
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe_ratio(returns: pd.Series, rf_rate: float = 0,
                         periods_per_year: int = 12) -> float:
    """
    Compute Sharpe ratio.

    For BAB strategy, rf_rate should be 0 since returns are already
    excess returns from a market-neutral portfolio.

    Args:
        returns: Series of periodic returns
        rf_rate: Risk-free rate (annualized, default 0)
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Sharpe ratio
    """
    excess_return = compute_annualized_return(returns, periods_per_year) - rf_rate
    volatility = compute_annualized_volatility(returns, periods_per_year)

    if volatility == 0:
        return np.nan

    return excess_return / volatility


def compute_sortino_ratio(returns: pd.Series, rf_rate: float = 0,
                          periods_per_year: int = 12) -> float:
    """
    Compute Sortino ratio (using downside deviation).

    Args:
        returns: Series of periodic returns
        rf_rate: Risk-free rate (annualized, default 0)
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Sortino ratio
    """
    excess_return = compute_annualized_return(returns, periods_per_year) - rf_rate
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)

    if downside_std == 0 or len(downside_returns) == 0:
        return np.nan

    return excess_return / downside_std


def compute_calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Calmar ratio
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    max_dd = compute_max_drawdown(returns)

    if max_dd == 0:
        return np.nan

    return ann_return / abs(max_dd)


def compute_rolling_beta(strategy_returns: pd.Series, market_returns: pd.Series,
                         window: int = 36) -> pd.Series:
    """
    Compute rolling beta of strategy returns vs market.

    This is critical for verifying market neutrality of the BAB strategy.
    Beta should remain near 0 throughout the sample period.

    Args:
        strategy_returns: Series of strategy returns
        market_returns: Series of market returns
        window: Rolling window in months (default 36)

    Returns:
        Series of rolling betas
    """
    # Align data
    common_idx = strategy_returns.index.intersection(market_returns.index)
    strat = strategy_returns.loc[common_idx]
    mkt = market_returns.loc[common_idx]

    # Rolling covariance and variance
    rolling_cov = strat.rolling(window).cov(mkt)
    rolling_var = mkt.rolling(window).var()

    # Beta = Cov / Var
    rolling_beta = rolling_cov / rolling_var

    return rolling_beta


def compute_performance_metrics(returns: pd.Series, name: str,
                                benchmark_returns: pd.Series = None) -> Dict:
    """
    Compute comprehensive performance metrics for a return series.

    Args:
        returns: Series of monthly returns
        name: Name of the strategy
        benchmark_returns: Optional benchmark returns for relative metrics

    Returns:
        Dictionary of performance metrics
    """
    metrics = {
        'Strategy': name,
        'Start_Date': returns.index.min().strftime('%Y-%m-%d'),
        'End_Date': returns.index.max().strftime('%Y-%m-%d'),
        'N_Months': len(returns),
        'N_Years': len(returns) / 12,
        'Total_Return': (1 + returns).prod() - 1,
        'Annualized_Return': compute_annualized_return(returns),
        'Annualized_Volatility': compute_annualized_volatility(returns),
        'Sharpe_Ratio': compute_sharpe_ratio(returns, rf_rate=0),
        'Sortino_Ratio': compute_sortino_ratio(returns, rf_rate=0),
        'Max_Drawdown': compute_max_drawdown(returns),
        'Calmar_Ratio': compute_calmar_ratio(returns),
        'Mean_Monthly_Return': returns.mean(),
        'Median_Monthly_Return': returns.median(),
        'Monthly_Std': returns.std(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Win_Rate': (returns > 0).sum() / len(returns),
        'Best_Month': returns.max(),
        'Worst_Month': returns.min(),
        'Positive_Months': (returns > 0).sum(),
        'Negative_Months': (returns <= 0).sum(),
    }

    # Add relative metrics if benchmark provided
    if benchmark_returns is not None:
        common_idx = returns.index.intersection(benchmark_returns.index)
        strat = returns.loc[common_idx]
        bench = benchmark_returns.loc[common_idx]

        # Beta to market
        cov = strat.cov(bench)
        var = bench.var()
        market_beta = cov / var if var > 0 else np.nan

        # Alpha (CAPM)
        alpha_monthly = strat.mean() - market_beta * bench.mean()

        # Information ratio
        active_returns = strat - bench
        tracking_error = active_returns.std() * np.sqrt(12)
        info_ratio = active_returns.mean() * 12 / tracking_error if tracking_error > 0 else np.nan

        metrics['Market_Beta'] = market_beta
        metrics['Alpha_Monthly'] = alpha_monthly
        metrics['Alpha_Annualized'] = alpha_monthly * 12
        metrics['Information_Ratio'] = info_ratio
        metrics['Correlation_to_Benchmark'] = strat.corr(bench)

    return metrics


def compute_rolling_metrics(returns: pd.Series, market_returns: pd.Series = None,
                            window: int = 12) -> pd.DataFrame:
    """
    Compute rolling performance metrics.

    Args:
        returns: Series of monthly returns
        market_returns: Optional market returns for rolling beta
        window: Rolling window in months

    Returns:
        DataFrame with rolling metrics
    """
    rolling = pd.DataFrame(index=returns.index)

    # Rolling return (annualized)
    rolling['Rolling_Return_Ann'] = returns.rolling(window).apply(
        lambda x: compute_annualized_return(x) if len(x) == window else np.nan
    )

    # Rolling volatility (annualized)
    rolling['Rolling_Vol_Ann'] = returns.rolling(window).std() * np.sqrt(12)

    # Rolling Sharpe
    rolling['Rolling_Sharpe'] = rolling['Rolling_Return_Ann'] / rolling['Rolling_Vol_Ann']

    # Rolling beta to market
    if market_returns is not None:
        rolling['Rolling_Market_Beta'] = compute_rolling_beta(returns, market_returns, window)

    return rolling


def analyze_market_neutrality(bab_returns: pd.Series, iwv_returns: pd.Series) -> Dict:
    """
    Comprehensive analysis of market neutrality.

    Args:
        bab_returns: Series of BAB returns
        iwv_returns: Series of IWV (market) returns

    Returns:
        Dictionary with neutrality statistics
    """
    print("\nAnalyzing Market Neutrality...")

    common_idx = bab_returns.index.intersection(iwv_returns.index)
    bab = bab_returns.loc[common_idx]
    iwv = iwv_returns.loc[common_idx]

    # Full-sample beta
    cov = bab.cov(iwv)
    var = iwv.var()
    full_beta = cov / var if var > 0 else np.nan

    # Rolling beta statistics
    rolling_beta = compute_rolling_beta(bab, iwv, window=36)

    stats = {
        'Full_Sample_Beta': full_beta,
        'Correlation': bab.corr(iwv),
        'Rolling_Beta_Mean': rolling_beta.mean(),
        'Rolling_Beta_Std': rolling_beta.std(),
        'Rolling_Beta_Min': rolling_beta.min(),
        'Rolling_Beta_Max': rolling_beta.max(),
        'Pct_Periods_Beta_Below_0.2': (abs(rolling_beta) < 0.2).sum() / len(rolling_beta.dropna()) * 100,
        'Pct_Periods_Beta_Below_0.3': (abs(rolling_beta) < 0.3).sum() / len(rolling_beta.dropna()) * 100,
    }

    print(f"  Full-sample market beta: {full_beta:.4f}")
    print(f"  Correlation with market: {bab.corr(iwv):.4f}")
    print(f"  Rolling beta mean: {rolling_beta.mean():.4f}")
    print(f"  Rolling beta std: {rolling_beta.std():.4f}")
    print(f"  % periods with |beta| < 0.2: {stats['Pct_Periods_Beta_Below_0.2']:.1f}%")

    if abs(full_beta) < 0.2:
        print("  ✓ Strategy maintains approximate market neutrality")
    else:
        print("  ⚠ Strategy has non-trivial market exposure")

    return stats, rolling_beta


def generate_monthly_performance(bab_data: pd.DataFrame, iwv_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Generate detailed monthly performance report.

    Args:
        bab_data: DataFrame with BAB returns and statistics
        iwv_returns: DataFrame with IWV returns

    Returns:
        DataFrame with monthly performance details
    """
    perf = pd.DataFrame(index=bab_data.index)

    # Monthly returns
    perf['BAB_Return'] = bab_data['BAB_Return']
    perf['BAB_Return_Unscaled'] = bab_data['BAB_Return_Unscaled']

    # Add IWV returns
    common_dates = bab_data.index.intersection(iwv_returns.index)
    perf['IWV_Return'] = iwv_returns.loc[common_dates, 'IWV']

    # Excess return over market
    perf['Excess_Return'] = perf['BAB_Return'] - perf['IWV_Return']

    # Cumulative returns
    perf['BAB_Cumulative'] = compute_cumulative_returns(bab_data['BAB_Return'])
    perf['IWV_Cumulative'] = compute_cumulative_returns(iwv_returns.loc[common_dates, 'IWV'])

    # Drawdowns
    perf['BAB_Drawdown'] = compute_drawdown(perf['BAB_Cumulative'])
    perf['IWV_Drawdown'] = compute_drawdown(perf['IWV_Cumulative'])

    # Beta statistics from portfolio construction
    perf['Q1_Mean_Beta'] = bab_data['Q1_Mean_Beta']
    perf['Q5_Mean_Beta'] = bab_data['Q5_Mean_Beta']
    perf['Beta_Spread'] = bab_data['Beta_Spread']

    # Leverage
    perf['Long_Leverage'] = bab_data['Long_Leverage']
    perf['Short_Leverage'] = bab_data['Short_Leverage']

    # Quintile returns
    perf['Q1_Return'] = bab_data['Q1_Mean_Return']
    perf['Q5_Return'] = bab_data['Q5_Mean_Return']
    perf['Q1_Scaled_Return'] = bab_data['Q1_Scaled_Return']
    perf['Q5_Scaled_Return'] = bab_data['Q5_Scaled_Return']

    # Portfolio sizes
    perf['N_Q1'] = bab_data['N_Q1']
    perf['N_Q5'] = bab_data['N_Q5']
    perf['N_Total'] = bab_data['N_Total']

    # Rolling metrics (12-month)
    iwv_series = iwv_returns.loc[common_dates, 'IWV']
    rolling = compute_rolling_metrics(bab_data['BAB_Return'], iwv_series, window=12)
    perf = perf.join(rolling)

    # Rolling 36-month beta
    rolling_beta_36 = compute_rolling_beta(bab_data['BAB_Return'], iwv_series, window=36)
    perf['Rolling_Beta_36M'] = rolling_beta_36

    return perf


def print_backtest_summary(metrics: Dict, neutrality_stats: Dict = None) -> None:
    """Print formatted backtest summary."""
    print("\n" + "=" * 70)
    print(f"Backtest Summary: {metrics['Strategy']}")
    print("=" * 70)

    print(f"\nPeriod: {metrics['Start_Date']} to {metrics['End_Date']}")
    print(f"Duration: {metrics['N_Months']} months ({metrics['N_Years']:.1f} years)")

    print(f"\n--- Return Metrics ---")
    print(f"Total Return:        {metrics['Total_Return']*100:>10.2f}%")
    print(f"Annualized Return:   {metrics['Annualized_Return']*100:>10.2f}%")
    print(f"Mean Monthly:        {metrics['Mean_Monthly_Return']*100:>10.3f}%")
    print(f"Median Monthly:      {metrics['Median_Monthly_Return']*100:>10.3f}%")

    print(f"\n--- Risk Metrics ---")
    print(f"Annualized Vol:      {metrics['Annualized_Volatility']*100:>10.2f}%")
    print(f"Max Drawdown:        {metrics['Max_Drawdown']*100:>10.2f}%")
    print(f"Skewness:            {metrics['Skewness']:>10.3f}")
    print(f"Kurtosis:            {metrics['Kurtosis']:>10.3f}")

    print(f"\n--- Risk-Adjusted Metrics ---")
    print(f"Sharpe Ratio:        {metrics['Sharpe_Ratio']:>10.3f}")
    print(f"Sortino Ratio:       {metrics['Sortino_Ratio']:>10.3f}")
    print(f"Calmar Ratio:        {metrics['Calmar_Ratio']:>10.3f}")

    print(f"\n--- Win/Loss Statistics ---")
    print(f"Win Rate:            {metrics['Win_Rate']*100:>10.1f}%")
    print(f"Best Month:          {metrics['Best_Month']*100:>10.2f}%")
    print(f"Worst Month:         {metrics['Worst_Month']*100:>10.2f}%")
    print(f"Positive Months:     {metrics['Positive_Months']:>10}")
    print(f"Negative Months:     {metrics['Negative_Months']:>10}")

    if 'Market_Beta' in metrics:
        print(f"\n--- Market Neutrality ---")
        print(f"Market Beta:         {metrics['Market_Beta']:>10.4f}")
        print(f"Correlation:         {metrics['Correlation_to_Benchmark']:>10.4f}")
        print(f"Alpha (Monthly):     {metrics['Alpha_Monthly']*100:>10.3f}%")
        print(f"Alpha (Annualized):  {metrics['Alpha_Annualized']*100:>10.2f}%")
        print(f"Information Ratio:   {metrics['Information_Ratio']:>10.3f}")

    if neutrality_stats:
        print(f"\n--- Rolling Beta Analysis ---")
        print(f"Rolling Beta Mean:   {neutrality_stats['Rolling_Beta_Mean']:>10.4f}")
        print(f"Rolling Beta Std:    {neutrality_stats['Rolling_Beta_Std']:>10.4f}")
        print(f"Rolling Beta Range:  [{neutrality_stats['Rolling_Beta_Min']:.3f}, {neutrality_stats['Rolling_Beta_Max']:.3f}]")
        print(f"% with |β| < 0.2:    {neutrality_stats['Pct_Periods_Beta_Below_0.2']:>10.1f}%")


def main():
    """Main function to execute backtesting."""
    print("=" * 70)
    print("BAB Strategy Backtesting")
    print("Frazzini-Pedersen (2014) Replication")
    print("=" * 70)

    # Load data
    bab_data, iwv_returns = load_data()

    # Check for required columns
    if 'BAB_Return' not in bab_data.columns:
        raise ValueError("Missing BAB_Return column")

    # Align data
    common_dates = bab_data.index.intersection(iwv_returns.index)
    iwv_series = iwv_returns.loc[common_dates, 'IWV']

    # Compute performance metrics for BAB
    bab_metrics = compute_performance_metrics(
        bab_data.loc[common_dates, 'BAB_Return'],
        name='BAB Strategy (Scaled)',
        benchmark_returns=iwv_series
    )

    # Compute performance metrics for unscaled BAB
    bab_unscaled_metrics = compute_performance_metrics(
        bab_data.loc[common_dates, 'BAB_Return_Unscaled'],
        name='BAB Strategy (Unscaled)',
        benchmark_returns=iwv_series
    )

    # Compute performance metrics for IWV (benchmark)
    iwv_metrics = compute_performance_metrics(
        iwv_series,
        name='IWV (Russell 3000)'
    )

    # Analyze market neutrality
    neutrality_stats, rolling_beta = analyze_market_neutrality(
        bab_data.loc[common_dates, 'BAB_Return'],
        iwv_series
    )

    # Print summaries
    print_backtest_summary(bab_metrics, neutrality_stats)
    print_backtest_summary(bab_unscaled_metrics)
    print_backtest_summary(iwv_metrics)

    # Generate monthly performance report
    monthly_perf = generate_monthly_performance(bab_data, iwv_returns)

    # Create summary DataFrame
    summary_df = pd.DataFrame([bab_metrics, bab_unscaled_metrics, iwv_metrics])

    # Add neutrality stats to summary
    for key, value in neutrality_stats.items():
        summary_df.loc[summary_df['Strategy'] == 'BAB Strategy (Scaled)', key] = value

    # Save outputs
    print("\nSaving outputs...")

    summary_df.to_csv(BACKTEST_SUMMARY_FILE, index=False)
    print(f"  Saved: {BACKTEST_SUMMARY_FILE}")

    monthly_perf.to_csv(MONTHLY_PERFORMANCE_FILE)
    print(f"  Saved: {MONTHLY_PERFORMANCE_FILE}")

    # Save rolling beta separately for detailed analysis
    rolling_beta_df = pd.DataFrame({'Rolling_Beta_36M': rolling_beta})
    rolling_beta_df.to_csv(ROLLING_BETA_FILE)
    print(f"  Saved: {ROLLING_BETA_FILE}")

    print("\n" + "=" * 70)
    print("Backtesting complete!")
    print("=" * 70)

    return summary_df, monthly_perf


if __name__ == "__main__":
    main()

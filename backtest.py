#!/usr/bin/env python3
"""
backtest.py - Betting-Against-Beta (BAB) Strategy Backtesting

This script computes performance statistics for the BAB strategy:
1. Loads BAB returns from portfolio_construction.py output
2. Computes performance metrics:
   - Annualized return
   - Annualized volatility
   - Sharpe ratio (assuming risk-free = 0)
   - Maximum drawdown
   - Calmar ratio
   - Win rate and other statistics
3. Saves summary CSV and full monthly performance file

Author: BAB Strategy Implementation
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

# Input file path (from portfolio_construction.py)
BAB_RETURNS_FILE = os.path.join(OUTPUT_DIR, "bab_returns.csv")

# Output file paths
BACKTEST_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "bab_backtest_summary.csv")
MONTHLY_PERFORMANCE_FILE = os.path.join(OUTPUT_DIR, "bab_monthly_performance.csv")


def load_bab_returns() -> pd.DataFrame:
    """
    Load BAB returns data.

    Returns:
        DataFrame with BAB and IWV returns
    """
    print("Loading BAB returns data...")

    bab_returns = pd.read_csv(BAB_RETURNS_FILE, index_col=0, parse_dates=True)

    print(f"  Loaded {len(bab_returns)} monthly observations")
    print(f"  Date range: {bab_returns.index.min()} to {bab_returns.index.max()}")

    return bab_returns


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

    if downside_std == 0:
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


def compute_information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                              periods_per_year: int = 12) -> float:
    """
    Compute Information Ratio relative to benchmark.

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Information ratio
    """
    active_returns = strategy_returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    active_return_ann = active_returns.mean() * periods_per_year

    if tracking_error == 0:
        return np.nan

    return active_return_ann / tracking_error


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

        metrics['Alpha_Monthly'] = strat.mean() - bench.mean()
        metrics['Alpha_Annualized'] = metrics['Alpha_Monthly'] * 12
        metrics['Information_Ratio'] = compute_information_ratio(strat, bench)
        metrics['Correlation_to_Benchmark'] = strat.corr(bench)
        metrics['Beta_to_Benchmark'] = strat.cov(bench) / bench.var() if bench.var() != 0 else np.nan

    return metrics


def compute_rolling_metrics(returns: pd.Series, window: int = 12) -> pd.DataFrame:
    """
    Compute rolling performance metrics.

    Args:
        returns: Series of monthly returns
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

    return rolling


def generate_monthly_performance(bab_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate detailed monthly performance report.

    Args:
        bab_data: DataFrame with BAB returns and statistics

    Returns:
        DataFrame with monthly performance details
    """
    perf = pd.DataFrame(index=bab_data.index)

    # Monthly returns
    perf['BAB_Return'] = bab_data['BAB_Return']
    perf['IWV_Return'] = bab_data['IWV_Return']
    perf['Excess_Return'] = bab_data['BAB_Return'] - bab_data['IWV_Return']

    # Cumulative returns
    perf['BAB_Cumulative'] = compute_cumulative_returns(bab_data['BAB_Return'])
    perf['IWV_Cumulative'] = compute_cumulative_returns(bab_data['IWV_Return'])

    # Drawdowns
    perf['BAB_Drawdown'] = compute_drawdown(perf['BAB_Cumulative'])
    perf['IWV_Drawdown'] = compute_drawdown(perf['IWV_Cumulative'])

    # Beta spread
    perf['Beta_Spread'] = bab_data['Q5_Mean_Beta'] - bab_data['Q1_Mean_Beta']
    perf['Q1_Mean_Beta'] = bab_data['Q1_Mean_Beta']
    perf['Q5_Mean_Beta'] = bab_data['Q5_Mean_Beta']

    # Quintile returns
    perf['Q1_Return'] = bab_data['Q1_Mean_Return']
    perf['Q5_Return'] = bab_data['Q5_Mean_Return']

    # Portfolio sizes
    perf['N_Q1'] = bab_data['N_Q1']
    perf['N_Q5'] = bab_data['N_Q5']

    # Rolling metrics (12-month)
    rolling = compute_rolling_metrics(bab_data['BAB_Return'], window=12)
    perf = perf.join(rolling)

    return perf


def print_backtest_summary(metrics: Dict) -> None:
    """Print formatted backtest summary."""
    print("\n" + "=" * 60)
    print(f"Backtest Summary: {metrics['Strategy']}")
    print("=" * 60)

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

    if 'Alpha_Annualized' in metrics:
        print(f"\n--- Relative to Benchmark (IWV) ---")
        print(f"Alpha (Annualized):  {metrics['Alpha_Annualized']*100:>10.2f}%")
        print(f"Information Ratio:   {metrics['Information_Ratio']:>10.3f}")
        print(f"Correlation:         {metrics['Correlation_to_Benchmark']:>10.3f}")
        print(f"Beta to Benchmark:   {metrics['Beta_to_Benchmark']:>10.3f}")


def main():
    """Main function to execute backtesting."""
    print("=" * 60)
    print("BAB Strategy Backtesting")
    print("=" * 60)

    # Load data
    bab_data = load_bab_returns()

    # Check for required columns
    required_cols = ['BAB_Return', 'IWV_Return']
    for col in required_cols:
        if col not in bab_data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute performance metrics for BAB
    bab_metrics = compute_performance_metrics(
        bab_data['BAB_Return'],
        name='BAB Strategy',
        benchmark_returns=bab_data['IWV_Return']
    )

    # Compute performance metrics for IWV (benchmark)
    iwv_metrics = compute_performance_metrics(
        bab_data['IWV_Return'],
        name='IWV (Russell 3000)'
    )

    # Print summaries
    print_backtest_summary(bab_metrics)
    print_backtest_summary(iwv_metrics)

    # Generate monthly performance report
    monthly_perf = generate_monthly_performance(bab_data)

    # Create summary DataFrame
    summary_df = pd.DataFrame([bab_metrics, iwv_metrics])

    # Save outputs
    print("\nSaving outputs...")

    summary_df.to_csv(BACKTEST_SUMMARY_FILE, index=False)
    print(f"  Saved: {BACKTEST_SUMMARY_FILE}")

    monthly_perf.to_csv(MONTHLY_PERFORMANCE_FILE)
    print(f"  Saved: {MONTHLY_PERFORMANCE_FILE}")

    print("\n" + "=" * 60)
    print("Backtesting complete!")
    print("=" * 60)

    return summary_df, monthly_perf


if __name__ == "__main__":
    main()

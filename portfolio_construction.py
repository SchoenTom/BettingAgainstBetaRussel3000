#!/usr/bin/env python3
"""
portfolio_construction.py - Betting-Against-Beta (BAB) Portfolio Construction

This script constructs the BAB portfolio by:
1. Loading saved CSV files (returns, betas)
2. Forming monthly cross-sections of stocks
3. Sorting stocks into five equal-sized beta quintiles each month
4. Computing BAB return as Q1 (low beta) minus Q5 (high beta) equal-weight average returns

Output DataFrame columns:
- Date: Month-end date
- BAB_Return: Monthly BAB strategy return
- Q1_Mean_Beta: Average beta of low-beta quintile
- Q5_Mean_Beta: Average beta of high-beta quintile
- Q1_Mean_Return: Average return of low-beta quintile
- Q5_Mean_Return: Average return of high-beta quintile
- N_Q1: Number of stocks in Q1
- N_Q5: Number of stocks in Q5

Author: BAB Strategy Implementation
Date: 2024
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "output"

# Input file paths (from data_loader.py)
RETURNS_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")
BETAS_FILE = os.path.join(DATA_DIR, "rolling_betas.csv")
IWV_RETURNS_FILE = os.path.join(DATA_DIR, "iwv_returns.csv")

# Output file paths
BAB_RETURNS_FILE = os.path.join(OUTPUT_DIR, "bab_returns.csv")
QUINTILE_STATS_FILE = os.path.join(OUTPUT_DIR, "quintile_statistics.csv")


def ensure_output_dir() -> None:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")


def load_data() -> tuple:
    """
    Load required data from CSV files.

    Returns:
        Tuple of (returns DataFrame, betas DataFrame, IWV returns DataFrame)
    """
    print("Loading data files...")

    # Load monthly returns
    returns = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  Returns: {returns.shape[0]} months, {returns.shape[1]} tickers")

    # Load rolling betas
    betas = pd.read_csv(BETAS_FILE, index_col=0, parse_dates=True)
    print(f"  Betas: {betas.shape[0]} months, {betas.shape[1]} tickers")

    # Load IWV returns
    iwv_returns = pd.read_csv(IWV_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  IWV Returns: {iwv_returns.shape[0]} months")

    return returns, betas, iwv_returns


def assign_quintiles(betas_row: pd.Series, n_quantiles: int = 5) -> pd.Series:
    """
    Assign stocks to quintiles based on beta values.

    Args:
        betas_row: Series of beta values for a single month
        n_quantiles: Number of quantile groups (default 5 for quintiles)

    Returns:
        Series of quintile assignments (1 = lowest beta, 5 = highest beta)
    """
    # Drop NaN values
    valid_betas = betas_row.dropna()

    if len(valid_betas) < n_quantiles:
        return pd.Series(dtype=float)

    # Use qcut to create equal-sized quintiles
    try:
        quintiles = pd.qcut(valid_betas, q=n_quantiles, labels=False, duplicates='drop') + 1
        return quintiles
    except ValueError:
        # Handle case where there aren't enough unique values
        return pd.Series(dtype=float)


def construct_bab_portfolio(returns: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    """
    Construct BAB portfolio by sorting stocks into beta quintiles each month.

    The BAB return is computed as:
    BAB_Return = Mean(Q1 Returns) - Mean(Q5 Returns)

    where Q1 is the lowest beta quintile and Q5 is the highest.

    Args:
        returns: DataFrame of monthly stock returns
        betas: DataFrame of rolling betas

    Returns:
        DataFrame with BAB portfolio returns and statistics
    """
    print("\nConstructing BAB portfolio...")

    # Get common dates and tickers
    common_dates = returns.index.intersection(betas.index)
    common_tickers = returns.columns.intersection(betas.columns)

    print(f"  Common dates: {len(common_dates)}")
    print(f"  Common tickers: {len(common_tickers)}")

    returns = returns.loc[common_dates, common_tickers]
    betas = betas.loc[common_dates, common_tickers]

    # Results storage
    results = []

    # Process each month
    for date in common_dates:
        # Get beta values for this month (lagged - use previous month's beta)
        # This avoids look-ahead bias: we form portfolios at the start of month t
        # using betas computed at the end of month t-1
        date_idx = common_dates.get_loc(date)

        if date_idx == 0:
            continue  # Skip first month - no prior beta available

        prev_date = common_dates[date_idx - 1]
        month_betas = betas.loc[prev_date]

        # Get returns for current month
        month_returns = returns.loc[date]

        # Get stocks with both valid beta and return
        valid_mask = month_betas.notna() & month_returns.notna()
        valid_tickers = valid_mask[valid_mask].index

        if len(valid_tickers) < 10:  # Need enough stocks for meaningful quintiles
            continue

        valid_betas = month_betas[valid_tickers]
        valid_returns = month_returns[valid_tickers]

        # Assign quintiles based on beta (1 = lowest, 5 = highest)
        quintiles = assign_quintiles(valid_betas, n_quantiles=5)

        if len(quintiles) == 0:
            continue

        # Calculate quintile statistics
        q1_mask = quintiles == 1
        q5_mask = quintiles == 5

        if q1_mask.sum() == 0 or q5_mask.sum() == 0:
            continue

        # Get tickers in each quintile
        q1_tickers = quintiles[q1_mask].index
        q5_tickers = quintiles[q5_mask].index

        # Calculate mean returns for Q1 and Q5
        q1_mean_return = valid_returns[q1_tickers].mean()
        q5_mean_return = valid_returns[q5_tickers].mean()

        # Calculate mean betas for Q1 and Q5
        q1_mean_beta = valid_betas[q1_tickers].mean()
        q5_mean_beta = valid_betas[q5_tickers].mean()

        # BAB return (long low beta, short high beta)
        bab_return = q1_mean_return - q5_mean_return

        # Store results
        results.append({
            'Date': date,
            'BAB_Return': bab_return,
            'Q1_Mean_Beta': q1_mean_beta,
            'Q5_Mean_Beta': q5_mean_beta,
            'Q1_Mean_Return': q1_mean_return,
            'Q5_Mean_Return': q5_mean_return,
            'N_Q1': len(q1_tickers),
            'N_Q5': len(q5_tickers),
            'N_Total': len(valid_tickers)
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    results_df = results_df.set_index('Date')

    print(f"\nBAB portfolio constructed:")
    print(f"  Months with valid portfolios: {len(results_df)}")
    print(f"  Date range: {results_df.index.min()} to {results_df.index.max()}")

    return results_df


def compute_quintile_statistics(returns: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed statistics for each quintile across all months.

    Args:
        returns: DataFrame of monthly stock returns
        betas: DataFrame of rolling betas

    Returns:
        DataFrame with quintile-level statistics
    """
    print("\nComputing quintile statistics...")

    common_dates = returns.index.intersection(betas.index)
    common_tickers = returns.columns.intersection(betas.columns)

    returns = returns.loc[common_dates, common_tickers]
    betas = betas.loc[common_dates, common_tickers]

    all_quintile_stats = []

    for date in common_dates:
        date_idx = common_dates.get_loc(date)

        if date_idx == 0:
            continue

        prev_date = common_dates[date_idx - 1]
        month_betas = betas.loc[prev_date]
        month_returns = returns.loc[date]

        valid_mask = month_betas.notna() & month_returns.notna()
        valid_tickers = valid_mask[valid_mask].index

        if len(valid_tickers) < 10:
            continue

        valid_betas = month_betas[valid_tickers]
        valid_returns = month_returns[valid_tickers]

        quintiles = assign_quintiles(valid_betas, n_quantiles=5)

        if len(quintiles) == 0:
            continue

        # Compute stats for each quintile
        for q in range(1, 6):
            q_mask = quintiles == q
            if q_mask.sum() == 0:
                continue

            q_tickers = quintiles[q_mask].index
            q_betas = valid_betas[q_tickers]
            q_returns = valid_returns[q_tickers]

            all_quintile_stats.append({
                'Date': date,
                'Quintile': q,
                'N_Stocks': len(q_tickers),
                'Mean_Beta': q_betas.mean(),
                'Median_Beta': q_betas.median(),
                'Min_Beta': q_betas.min(),
                'Max_Beta': q_betas.max(),
                'Mean_Return': q_returns.mean(),
                'Median_Return': q_returns.median(),
                'Std_Return': q_returns.std()
            })

    stats_df = pd.DataFrame(all_quintile_stats)
    stats_df['Date'] = pd.to_datetime(stats_df['Date'])

    print(f"  Generated {len(stats_df)} quintile-month observations")

    return stats_df


def print_summary_statistics(bab_returns: pd.DataFrame) -> None:
    """Print summary statistics of BAB portfolio."""
    print("\n" + "=" * 60)
    print("BAB Portfolio Summary Statistics")
    print("=" * 60)

    bab = bab_returns['BAB_Return']

    print(f"\nReturn Statistics:")
    print(f"  Mean Monthly Return: {bab.mean()*100:.3f}%")
    print(f"  Median Monthly Return: {bab.median()*100:.3f}%")
    print(f"  Std Dev (Monthly): {bab.std()*100:.3f}%")
    print(f"  Min Monthly Return: {bab.min()*100:.3f}%")
    print(f"  Max Monthly Return: {bab.max()*100:.3f}%")
    print(f"  Skewness: {bab.skew():.3f}")
    print(f"  Kurtosis: {bab.kurtosis():.3f}")

    print(f"\nBeta Spread Statistics:")
    print(f"  Mean Q1 Beta: {bab_returns['Q1_Mean_Beta'].mean():.3f}")
    print(f"  Mean Q5 Beta: {bab_returns['Q5_Mean_Beta'].mean():.3f}")
    print(f"  Mean Beta Spread (Q5-Q1): {(bab_returns['Q5_Mean_Beta'] - bab_returns['Q1_Mean_Beta']).mean():.3f}")

    print(f"\nPortfolio Composition:")
    print(f"  Avg Stocks in Q1: {bab_returns['N_Q1'].mean():.1f}")
    print(f"  Avg Stocks in Q5: {bab_returns['N_Q5'].mean():.1f}")
    print(f"  Avg Total Stocks: {bab_returns['N_Total'].mean():.1f}")

    # Win rate
    win_rate = (bab > 0).sum() / len(bab) * 100
    print(f"\nPerformance:")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Positive Months: {(bab > 0).sum()}")
    print(f"  Negative Months: {(bab <= 0).sum()}")


def main():
    """Main function to execute portfolio construction."""
    print("=" * 60)
    print("BAB Portfolio Construction")
    print("=" * 60)

    # Create output directory
    ensure_output_dir()

    # Load data
    returns, betas, iwv_returns = load_data()

    # Construct BAB portfolio
    bab_returns = construct_bab_portfolio(returns, betas)

    # Compute quintile statistics
    quintile_stats = compute_quintile_statistics(returns, betas)

    # Add IWV returns to BAB DataFrame for comparison
    common_dates = bab_returns.index.intersection(iwv_returns.index)
    bab_returns = bab_returns.loc[common_dates]
    bab_returns['IWV_Return'] = iwv_returns.loc[common_dates, 'IWV']

    # Save outputs
    print("\nSaving outputs...")

    bab_returns.to_csv(BAB_RETURNS_FILE)
    print(f"  Saved: {BAB_RETURNS_FILE}")

    quintile_stats.to_csv(QUINTILE_STATS_FILE, index=False)
    print(f"  Saved: {QUINTILE_STATS_FILE}")

    # Print summary statistics
    print_summary_statistics(bab_returns)

    print("\n" + "=" * 60)
    print("Portfolio construction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

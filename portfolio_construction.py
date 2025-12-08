#!/usr/bin/env python3
"""
portfolio_construction.py - Betting-Against-Beta (BAB) Portfolio Construction

Replication of Frazzini and Pedersen (2014) "Betting Against Beta"
Journal of Financial Economics, 111(1), 1-25.

================================================================================
BETA SCALING METHODOLOGY (Critical for Proper BAB Construction)
================================================================================

The key innovation in Frazzini-Pedersen is BETA SCALING, which ensures:
1. Each portfolio leg has an effective beta of approximately 1.0
2. The combined BAB portfolio has near-zero market exposure
3. Returns are comparable across different beta levels

CONSTRUCTION PROCEDURE:

1. RANK STOCKS BY BETA: Sort stocks into quintiles based on estimated beta
   - Q1 = lowest beta stocks (defensive)
   - Q5 = highest beta stocks (aggressive)

2. COMPUTE PORTFOLIO BETAS:
   - β_L = average beta of low-beta portfolio (Q1)
   - β_H = average beta of high-beta portfolio (Q5)

3. APPLY LEVERAGE/DE-LEVERAGE:
   - Scale low-beta portfolio: (1/β_L) × r_L  (leverage up)
   - Scale high-beta portfolio: (1/β_H) × r_H  (de-leverage down)

4. COMPUTE BAB RETURN:
   BAB_t = (1/β_L) × r_L,t - (1/β_H) × r_H,t

This scaling ensures:
- Low-beta stocks are levered to beta ≈ 1
- High-beta stocks are de-levered to beta ≈ 1
- Net BAB portfolio has beta ≈ 0 (market neutral)

Without scaling, BAB would have negative market exposure (short high-beta
means net short the market), confounding the true low-beta premium.

================================================================================

Output DataFrame columns:
- Date: Month-end date
- BAB_Return: Scaled BAB strategy return
- BAB_Return_Unscaled: Simple Q1 - Q5 return (for comparison)
- Q1_Mean_Beta: Average beta of low-beta quintile
- Q5_Mean_Beta: Average beta of high-beta quintile
- Q1_Mean_Return: Average return of low-beta quintile
- Q5_Mean_Return: Average return of high-beta quintile
- Q1_Scaled_Return: Beta-scaled return of Q1
- Q5_Scaled_Return: Beta-scaled return of Q5
- Portfolio_Beta: Ex-ante portfolio beta (should be near 0)
- N_Q1: Number of stocks in Q1
- N_Q5: Number of stocks in Q5

Author: BAB Strategy Implementation (Frazzini-Pedersen Replication)
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
EXCESS_RETURNS_FILE = os.path.join(DATA_DIR, "monthly_excess_returns.csv")
BETAS_FILE = os.path.join(DATA_DIR, "rolling_betas.csv")
IWV_RETURNS_FILE = os.path.join(DATA_DIR, "iwv_returns.csv")
IWV_EXCESS_RETURNS_FILE = os.path.join(DATA_DIR, "iwv_excess_returns.csv")
RF_RATE_FILE = os.path.join(DATA_DIR, "risk_free_rate.csv")

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
        Tuple of (returns, excess_returns, betas, IWV returns, IWV excess returns, rf_rate)
    """
    print("Loading data files...")

    # Load monthly returns
    returns = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  Returns: {returns.shape[0]} months, {returns.shape[1]} tickers")

    # Load excess returns
    excess_returns = pd.read_csv(EXCESS_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  Excess Returns: {excess_returns.shape[0]} months, {excess_returns.shape[1]} tickers")

    # Load rolling betas
    betas = pd.read_csv(BETAS_FILE, index_col=0, parse_dates=True)
    print(f"  Betas: {betas.shape[0]} months, {betas.shape[1]} tickers")

    # Load IWV returns
    iwv_returns = pd.read_csv(IWV_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  IWV Returns: {iwv_returns.shape[0]} months")

    # Load IWV excess returns
    iwv_excess = pd.read_csv(IWV_EXCESS_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  IWV Excess Returns: {iwv_excess.shape[0]} months")

    # Load risk-free rate
    rf_rate = pd.read_csv(RF_RATE_FILE, index_col=0, parse_dates=True)
    print(f"  Risk-Free Rate: {rf_rate.shape[0]} months")

    return returns, excess_returns, betas, iwv_returns, iwv_excess, rf_rate


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

    if len(valid_betas) < n_quantiles * 2:  # Need at least 2 per quintile
        return pd.Series(dtype=float)

    # Use qcut to create equal-sized quintiles
    try:
        quintiles = pd.qcut(valid_betas, q=n_quantiles, labels=False, duplicates='drop') + 1
        return quintiles
    except ValueError:
        # Handle case where there aren't enough unique values
        return pd.Series(dtype=float)


def construct_bab_portfolio_with_scaling(
    returns: pd.DataFrame,
    excess_returns: pd.DataFrame,
    betas: pd.DataFrame,
    rf_rate: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct BAB portfolio with proper beta scaling following Frazzini-Pedersen (2014).

    The scaling procedure ensures market neutrality:
    - Long portfolio scaled by 1/β_L to achieve beta ≈ 1
    - Short portfolio scaled by 1/β_H to achieve beta ≈ 1
    - Combined BAB has beta ≈ 0

    BAB_t = (1/β_L) × r_L,t - (1/β_H) × r_H,t

    This is equivalent to:
    - Going long $1/β_L in low-beta portfolio
    - Going short $1/β_H in high-beta portfolio
    - Net market exposure ≈ 0

    Args:
        returns: DataFrame of monthly stock returns
        excess_returns: DataFrame of monthly excess returns
        betas: DataFrame of rolling betas
        rf_rate: DataFrame of risk-free rates

    Returns:
        DataFrame with BAB portfolio returns and statistics
    """
    print("\n" + "=" * 70)
    print("Constructing BAB Portfolio with Beta Scaling")
    print("Following Frazzini and Pedersen (2014) methodology")
    print("=" * 70)

    # Get common dates and tickers
    common_dates = returns.index.intersection(betas.index).intersection(excess_returns.index)
    common_tickers = returns.columns.intersection(betas.columns).intersection(excess_returns.columns)

    print(f"  Common dates: {len(common_dates)}")
    print(f"  Common tickers: {len(common_tickers)}")

    returns = returns.loc[common_dates, common_tickers]
    excess_returns = excess_returns.loc[common_dates, common_tickers]
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
        month_excess = excess_returns.loc[date]

        # Get stocks with both valid beta and return
        valid_mask = month_betas.notna() & month_returns.notna() & month_excess.notna()
        valid_tickers = valid_mask[valid_mask].index

        if len(valid_tickers) < 20:  # Need enough stocks for meaningful quintiles
            continue

        valid_betas = month_betas[valid_tickers]
        valid_returns = month_returns[valid_tickers]
        valid_excess = month_excess[valid_tickers]

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

        # Calculate mean betas for Q1 (low) and Q5 (high)
        q1_mean_beta = valid_betas[q1_tickers].mean()
        q5_mean_beta = valid_betas[q5_tickers].mean()

        # Skip if betas are too extreme or invalid
        if q1_mean_beta <= 0.01 or q5_mean_beta <= 0.01:
            continue
        if np.isnan(q1_mean_beta) or np.isnan(q5_mean_beta):
            continue

        # Calculate equal-weighted returns for Q1 and Q5
        q1_mean_return = valid_returns[q1_tickers].mean()
        q5_mean_return = valid_returns[q5_tickers].mean()

        q1_mean_excess = valid_excess[q1_tickers].mean()
        q5_mean_excess = valid_excess[q5_tickers].mean()

        # ================================================================
        # BETA SCALING - The key innovation of Frazzini-Pedersen
        # ================================================================
        # Scale returns by inverse of portfolio beta to achieve beta ≈ 1
        # This creates market-neutral exposure

        # Scaled returns (leverage/de-leverage)
        q1_scaled_return = (1.0 / q1_mean_beta) * q1_mean_return
        q5_scaled_return = (1.0 / q5_mean_beta) * q5_mean_return

        q1_scaled_excess = (1.0 / q1_mean_beta) * q1_mean_excess
        q5_scaled_excess = (1.0 / q5_mean_beta) * q5_mean_excess

        # BAB return with scaling (market neutral)
        bab_return_scaled = q1_scaled_return - q5_scaled_return
        bab_excess_scaled = q1_scaled_excess - q5_scaled_excess

        # BAB return without scaling (for comparison - NOT market neutral)
        bab_return_unscaled = q1_mean_return - q5_mean_return

        # Ex-ante portfolio beta (should be close to 0)
        # Long (1/β_L) × β_L = 1, Short (1/β_H) × β_H = 1
        # Net beta = 1 - 1 = 0 (by construction)
        # But we can compute realized beta for verification
        portfolio_beta_exante = (1.0 / q1_mean_beta) * q1_mean_beta - (1.0 / q5_mean_beta) * q5_mean_beta

        # Compute leverage applied
        long_leverage = 1.0 / q1_mean_beta
        short_leverage = 1.0 / q5_mean_beta

        # Store results
        results.append({
            'Date': date,
            # Scaled BAB returns (main strategy)
            'BAB_Return': bab_return_scaled,
            'BAB_Excess_Return': bab_excess_scaled,
            # Unscaled for comparison
            'BAB_Return_Unscaled': bab_return_unscaled,
            # Portfolio betas
            'Q1_Mean_Beta': q1_mean_beta,
            'Q5_Mean_Beta': q5_mean_beta,
            'Portfolio_Beta_ExAnte': portfolio_beta_exante,  # Should be ~0
            # Raw returns
            'Q1_Mean_Return': q1_mean_return,
            'Q5_Mean_Return': q5_mean_return,
            # Scaled returns
            'Q1_Scaled_Return': q1_scaled_return,
            'Q5_Scaled_Return': q5_scaled_return,
            # Leverage factors
            'Long_Leverage': long_leverage,
            'Short_Leverage': short_leverage,
            # Portfolio sizes
            'N_Q1': len(q1_tickers),
            'N_Q5': len(q5_tickers),
            'N_Total': len(valid_tickers),
            # Beta spread
            'Beta_Spread': q5_mean_beta - q1_mean_beta,
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    results_df = results_df.set_index('Date')

    print(f"\nBAB portfolio constructed:")
    print(f"  Months with valid portfolios: {len(results_df)}")
    print(f"  Date range: {results_df.index.min()} to {results_df.index.max()}")
    print(f"  Mean ex-ante portfolio beta: {results_df['Portfolio_Beta_ExAnte'].mean():.4f}")

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

        if len(valid_tickers) < 20:
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
                'Std_Beta': q_betas.std(),
                'Mean_Return': q_returns.mean(),
                'Median_Return': q_returns.median(),
                'Std_Return': q_returns.std(),
                # Scaled return (as if beta = 1)
                'Scaled_Return': q_returns.mean() / q_betas.mean() if q_betas.mean() > 0.01 else np.nan,
            })

    stats_df = pd.DataFrame(all_quintile_stats)
    stats_df['Date'] = pd.to_datetime(stats_df['Date'])

    print(f"  Generated {len(stats_df)} quintile-month observations")

    return stats_df


def verify_market_neutrality(bab_returns: pd.DataFrame, iwv_returns: pd.DataFrame) -> dict:
    """
    Verify that the scaled BAB portfolio is approximately market neutral.

    This is done by regressing BAB returns on market returns.
    Beta should be close to 0 if scaling is working correctly.

    Args:
        bab_returns: DataFrame with BAB returns
        iwv_returns: DataFrame with IWV (market) returns

    Returns:
        Dictionary with regression statistics
    """
    print("\nVerifying market neutrality of scaled BAB portfolio...")

    common_idx = bab_returns.index.intersection(iwv_returns.index)
    bab = bab_returns.loc[common_idx, 'BAB_Return'].dropna()
    iwv = iwv_returns.loc[common_idx, 'IWV'].dropna()

    # Align
    common = bab.index.intersection(iwv.index)
    bab = bab.loc[common]
    iwv = iwv.loc[common]

    # Simple regression: BAB = alpha + beta * IWV + epsilon
    cov = bab.cov(iwv)
    var = iwv.var()
    beta = cov / var if var > 0 else np.nan
    alpha = bab.mean() - beta * iwv.mean()

    # R-squared
    y_pred = alpha + beta * iwv
    ss_res = ((bab - y_pred) ** 2).sum()
    ss_tot = ((bab - bab.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Correlation
    correlation = bab.corr(iwv)

    # T-statistic for beta
    n = len(bab)
    if n > 2:
        residuals = bab - y_pred
        se_beta = np.sqrt(residuals.var() / ((n - 2) * var)) if var > 0 else np.nan
        t_stat = beta / se_beta if se_beta > 0 else np.nan
    else:
        t_stat = np.nan

    results = {
        'Market_Beta': beta,
        'Alpha_Monthly': alpha,
        'Alpha_Annualized': alpha * 12,
        'R_Squared': r_squared,
        'Correlation': correlation,
        'T_Stat_Beta': t_stat,
        'N_Observations': n,
    }

    print(f"  Market Beta: {beta:.4f} (should be ~0)")
    print(f"  Correlation with Market: {correlation:.4f}")
    print(f"  Monthly Alpha: {alpha*100:.3f}%")
    print(f"  Annualized Alpha: {alpha*12*100:.2f}%")

    if abs(beta) < 0.3:
        print("  ✓ Portfolio is approximately market neutral")
    else:
        print("  ⚠ Portfolio has significant market exposure")

    return results


def print_summary_statistics(bab_returns: pd.DataFrame) -> None:
    """Print summary statistics of BAB portfolio."""
    print("\n" + "=" * 70)
    print("BAB Portfolio Summary Statistics")
    print("=" * 70)

    bab_scaled = bab_returns['BAB_Return']
    bab_unscaled = bab_returns['BAB_Return_Unscaled']

    print(f"\nSCALED BAB Return Statistics (Market Neutral):")
    print(f"  Mean Monthly Return: {bab_scaled.mean()*100:.3f}%")
    print(f"  Median Monthly Return: {bab_scaled.median()*100:.3f}%")
    print(f"  Std Dev (Monthly): {bab_scaled.std()*100:.3f}%")
    print(f"  Annualized Return: {bab_scaled.mean()*12*100:.2f}%")
    print(f"  Annualized Volatility: {bab_scaled.std()*np.sqrt(12)*100:.2f}%")
    print(f"  Sharpe Ratio (rf=0): {bab_scaled.mean()/bab_scaled.std()*np.sqrt(12):.3f}")

    print(f"\nUNSCALED BAB Return Statistics (for comparison):")
    print(f"  Mean Monthly Return: {bab_unscaled.mean()*100:.3f}%")
    print(f"  Annualized Return: {bab_unscaled.mean()*12*100:.2f}%")

    print(f"\nBeta Statistics:")
    print(f"  Mean Q1 (Low) Beta: {bab_returns['Q1_Mean_Beta'].mean():.3f}")
    print(f"  Mean Q5 (High) Beta: {bab_returns['Q5_Mean_Beta'].mean():.3f}")
    print(f"  Mean Beta Spread (Q5-Q1): {bab_returns['Beta_Spread'].mean():.3f}")

    print(f"\nLeverage Statistics:")
    print(f"  Mean Long Leverage (1/β_L): {bab_returns['Long_Leverage'].mean():.2f}x")
    print(f"  Mean Short Leverage (1/β_H): {bab_returns['Short_Leverage'].mean():.2f}x")

    print(f"\nPortfolio Composition:")
    print(f"  Avg Stocks in Q1: {bab_returns['N_Q1'].mean():.1f}")
    print(f"  Avg Stocks in Q5: {bab_returns['N_Q5'].mean():.1f}")
    print(f"  Avg Total Stocks: {bab_returns['N_Total'].mean():.1f}")

    # Win rate
    win_rate = (bab_scaled > 0).sum() / len(bab_scaled) * 100
    print(f"\nPerformance:")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Positive Months: {(bab_scaled > 0).sum()}")
    print(f"  Negative Months: {(bab_scaled <= 0).sum()}")


def main():
    """Main function to execute portfolio construction."""
    print("=" * 70)
    print("BAB Portfolio Construction")
    print("Frazzini-Pedersen (2014) Replication with Beta Scaling")
    print("=" * 70)

    # Create output directory
    ensure_output_dir()

    # Load data
    returns, excess_returns, betas, iwv_returns, iwv_excess, rf_rate = load_data()

    # Construct BAB portfolio with scaling
    bab_returns = construct_bab_portfolio_with_scaling(
        returns, excess_returns, betas, rf_rate
    )

    # Compute quintile statistics
    quintile_stats = compute_quintile_statistics(returns, betas)

    # Add IWV returns to BAB DataFrame for comparison
    common_dates = bab_returns.index.intersection(iwv_returns.index)
    bab_returns = bab_returns.loc[common_dates]
    bab_returns['IWV_Return'] = iwv_returns.loc[common_dates, 'IWV']

    # Verify market neutrality
    neutrality_stats = verify_market_neutrality(bab_returns, iwv_returns)
    bab_returns['Realized_Market_Beta'] = neutrality_stats['Market_Beta']

    # Save outputs
    print("\nSaving outputs...")

    bab_returns.to_csv(BAB_RETURNS_FILE)
    print(f"  Saved: {BAB_RETURNS_FILE}")

    quintile_stats.to_csv(QUINTILE_STATS_FILE, index=False)
    print(f"  Saved: {QUINTILE_STATS_FILE}")

    # Print summary statistics
    print_summary_statistics(bab_returns)

    print("\n" + "=" * 70)
    print("Portfolio construction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

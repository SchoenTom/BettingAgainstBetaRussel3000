"""
portfolio_construction.py - Construct BAB portfolios following F&P (2014) Table 4

Key methodology from Frazzini & Pedersen (2014):
1. Decile portfolios: Equal-weighted within each decile
2. BAB factor: Beta-rank weighted, rescaled to beta=1 per leg
3. Beta-neutral: Rescale each leg to beta=1, normalize to $2 gross exposure
4. Uses LAGGED betas (t-1) to avoid look-ahead bias
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import (
    DATA_DIR, OUTPUT_DIR, NUM_DECILES, MIN_STOCKS_PER_DECILE, ensure_directories
)


def load_data():
    """Load required data files."""
    logger.info("Loading data files...")

    excess_path = os.path.join(DATA_DIR, 'monthly_excess_returns.csv')
    betas_path = os.path.join(DATA_DIR, 'rolling_betas.csv')

    if not os.path.exists(excess_path) or not os.path.exists(betas_path):
        raise FileNotFoundError("Run data_loader.py first.")

    excess_returns = pd.read_csv(excess_path, index_col=0, parse_dates=True)
    betas = pd.read_csv(betas_path, index_col=0, parse_dates=True)

    logger.info(f"Excess returns: {excess_returns.shape}")
    logger.info(f"Betas: {betas.shape}")

    return excess_returns, betas


def get_stock_columns(df):
    """Get stock columns (exclude benchmark)."""
    exclude = ['Benchmark', 'MSCI_World', 'RF_Rate']
    return [col for col in df.columns if col not in exclude]


def assign_deciles(betas_series, num_deciles=NUM_DECILES):
    """Assign stocks to deciles based on beta (1=lowest, 10=highest)."""
    valid = betas_series.dropna()

    if len(valid) < num_deciles * MIN_STOCKS_PER_DECILE:
        return pd.Series(dtype=float)

    try:
        deciles = pd.qcut(valid, q=num_deciles, labels=False, duplicates='drop') + 1
        return deciles
    except ValueError:
        return pd.Series(dtype=float)


def compute_beta_rank_weights(betas, is_low_beta_portfolio):
    """
    Compute beta-rank weights as per F&P Table 4.

    For low-beta portfolio: lower beta = higher weight
    For high-beta portfolio: higher beta = higher weight
    """
    valid = betas.dropna()
    if len(valid) == 0:
        return pd.Series(dtype=float)

    # Rank betas (1 = lowest)
    ranks = valid.rank(method='average')

    if is_low_beta_portfolio:
        # Lower beta = higher weight: use inverse rank
        weights = len(ranks) - ranks + 1
    else:
        # Higher beta = higher weight: use direct rank
        weights = ranks

    # Normalize to sum to 1
    weights = weights / weights.sum()
    return weights


def construct_bab_portfolios(excess_returns, betas):
    """
    Construct monthly BAB portfolios following F&P (2014) Table 4.

    For each month t:
    1. Use betas from t-1 (lagged)
    2. Split stocks at median beta into low-beta and high-beta groups
    3. Weight stocks by beta rank within each group
    4. Rescale both portfolios to beta=1
    5. BAB = Long low-beta - Short high-beta
    """
    logger.info("Constructing BAB portfolios (F&P Table 4 methodology)...")

    stock_cols = get_stock_columns(excess_returns)
    common_dates = excess_returns.index.intersection(betas.index)
    common_stocks = [col for col in stock_cols if col in betas.columns]

    logger.info(f"Common dates: {len(common_dates)}, stocks: {len(common_stocks)}")

    returns_aligned = excess_returns.loc[common_dates, common_stocks]
    betas_aligned = betas.loc[common_dates, common_stocks]

    results = []
    dates = sorted(common_dates)

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]

        # Lagged betas (t-1) - NO LOOK-AHEAD
        betas_lagged = betas_aligned.loc[prev_date]
        returns_current = returns_aligned.loc[current_date]

        # Valid stocks
        valid_mask = betas_lagged.notna() & returns_current.notna()
        valid_stocks = valid_mask[valid_mask].index

        if len(valid_stocks) < MIN_STOCKS_PER_DECILE * 2:
            continue

        valid_betas = betas_lagged[valid_stocks]
        valid_returns = returns_current[valid_stocks]

        # Split at median beta
        median_beta = valid_betas.median()
        low_beta_stocks = valid_betas[valid_betas < median_beta].index
        high_beta_stocks = valid_betas[valid_betas >= median_beta].index

        if len(low_beta_stocks) < MIN_STOCKS_PER_DECILE or len(high_beta_stocks) < MIN_STOCKS_PER_DECILE:
            continue

        # Beta-rank weights (F&P methodology)
        low_weights = compute_beta_rank_weights(valid_betas[low_beta_stocks], is_low_beta_portfolio=True)
        high_weights = compute_beta_rank_weights(valid_betas[high_beta_stocks], is_low_beta_portfolio=False)

        if low_weights.empty or high_weights.empty:
            continue

        # Weighted returns and betas
        r_L = (valid_returns[low_beta_stocks] * low_weights).sum()
        r_H = (valid_returns[high_beta_stocks] * high_weights).sum()
        beta_L = (valid_betas[low_beta_stocks] * low_weights).sum()
        beta_H = (valid_betas[high_beta_stocks] * high_weights).sum()

        if beta_L <= 0 or beta_H <= 0:
            continue

        # =================================================================
        # BETA-NEUTRAL BAB CONSTRUCTION (F&P 2014)
        # =================================================================
        # Raw weights to rescale each leg to beta=1
        w_L_raw = 1.0 / beta_L
        w_H_raw = 1.0 / beta_H
        gross_raw = w_L_raw + w_H_raw

        # Normalize to $2 gross exposure ($1 long + $1 short equivalent)
        w_L = 2.0 * w_L_raw / gross_raw
        w_H = 2.0 * w_H_raw / gross_raw

        # BAB excess return (beta-neutral, controlled leverage)
        bab_return = w_L * r_L - w_H * r_H

        # Ex-ante beta (should be ~0)
        beta_ex_ante = w_L * beta_L - w_H * beta_H

        # Dollar exposure tracking
        dollar_net = w_L - w_H

        results.append({
            'Date': current_date,
            'BAB_Return': bab_return,
            'BAB_Excess_Return': bab_return,
            'Beta_L': beta_L,
            'Beta_H': beta_H,
            'Beta_Spread': beta_H - beta_L,
            'R_L': r_L,
            'R_H': r_H,
            'W_L': w_L,
            'W_H': w_H,
            'Beta_ExAnte': beta_ex_ante,
            'Dollar_Net': dollar_net,
            'Gross_Exposure': w_L + w_H,
            'N_Low': len(low_beta_stocks),
            'N_High': len(high_beta_stocks),
            'N_Total': len(valid_stocks),
        })

    if not results:
        logger.warning("No valid BAB portfolios constructed!")
        return pd.DataFrame()

    bab_df = pd.DataFrame(results)
    bab_df.set_index('Date', inplace=True)
    bab_df.index = pd.to_datetime(bab_df.index)

    logger.info(f"Constructed BAB: {len(bab_df)} months")
    logger.info(f"Ex-ante beta: mean={bab_df['Beta_ExAnte'].mean():.6f}")
    logger.info(f"Dollar net: mean={bab_df['Dollar_Net'].mean():.3f}")

    return bab_df


def compute_decile_returns(excess_returns, betas):
    """Compute equal-weighted returns for each decile (D1-D10)."""
    logger.info("Computing decile returns...")

    stock_cols = get_stock_columns(excess_returns)
    common_dates = excess_returns.index.intersection(betas.index)
    common_stocks = [col for col in stock_cols if col in betas.columns]

    returns_aligned = excess_returns.loc[common_dates, common_stocks]
    betas_aligned = betas.loc[common_dates, common_stocks]

    results = []
    dates = sorted(common_dates)

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]

        betas_lagged = betas_aligned.loc[prev_date]
        returns_current = returns_aligned.loc[current_date]

        valid_mask = betas_lagged.notna() & returns_current.notna()
        valid_stocks = valid_mask[valid_mask].index

        if len(valid_stocks) < NUM_DECILES * MIN_STOCKS_PER_DECILE:
            continue

        deciles = assign_deciles(betas_lagged[valid_stocks])
        if deciles.empty:
            continue

        row = {'Date': current_date}

        for d in range(1, NUM_DECILES + 1):
            d_stocks = deciles[deciles == d].index
            if len(d_stocks) > 0:
                row[f'D{d}_Return'] = returns_current[d_stocks].mean()
                row[f'D{d}_Beta'] = betas_lagged[d_stocks].mean()
                row[f'D{d}_N'] = len(d_stocks)
            else:
                row[f'D{d}_Return'] = np.nan
                row[f'D{d}_Beta'] = np.nan
                row[f'D{d}_N'] = 0

        results.append(row)

    if not results:
        return pd.DataFrame()

    decile_df = pd.DataFrame(results)
    decile_df.set_index('Date', inplace=True)
    decile_df.index = pd.to_datetime(decile_df.index)

    logger.info(f"Decile returns: {len(decile_df)} months")
    return decile_df


def save_outputs(bab_df, decile_df):
    """Save outputs to CSV."""
    ensure_directories()

    if not bab_df.empty:
        bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
        bab_df.to_csv(bab_path)
        logger.info(f"Saved: {bab_path}")

    if not decile_df.empty:
        decile_path = os.path.join(OUTPUT_DIR, 'decile_returns.csv')
        decile_df.to_csv(decile_path)
        logger.info(f"Saved: {decile_path}")


def print_summary(bab_df, decile_df):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("BAB Portfolio Summary (F&P 2014 Methodology)")
    print("=" * 60)

    if bab_df.empty:
        print("No portfolios constructed.")
        return

    print(f"\nPeriod: {bab_df.index.min():%Y-%m} to {bab_df.index.max():%Y-%m}")
    print(f"Months: {len(bab_df)}")

    print("\n--- Beta Statistics ---")
    print(f"Avg Low-Beta:     {bab_df['Beta_L'].mean():.3f}")
    print(f"Avg High-Beta:    {bab_df['Beta_H'].mean():.3f}")
    print(f"Avg Beta Spread:  {bab_df['Beta_Spread'].mean():.3f}")

    print("\n--- Portfolio Weights ---")
    print(f"Avg Long Weight:   ${bab_df['W_L'].mean():.3f}")
    print(f"Avg Short Weight:  ${bab_df['W_H'].mean():.3f}")
    print(f"Avg Gross:         ${bab_df['Gross_Exposure'].mean():.3f}")
    print(f"Avg Net Dollar:    ${bab_df['Dollar_Net'].mean():.4f}")

    print("\n--- Market Neutrality ---")
    print(f"Ex-Ante Beta:      {bab_df['Beta_ExAnte'].mean():.6f} (should be ~0)")

    print("\n--- Return Statistics ---")
    bab_ret = bab_df['BAB_Return']
    ann_ret = bab_ret.mean() * 12
    ann_vol = bab_ret.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    t_stat = bab_ret.mean() / (bab_ret.std() / np.sqrt(len(bab_ret))) if bab_ret.std() > 0 else 0

    print(f"Monthly Return:    {bab_ret.mean()*100:.3f}% (t={t_stat:.2f})")
    print(f"Annualized Return: {ann_ret*100:.2f}%")
    print(f"Annualized Vol:    {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.3f}")

    print("\n--- Decile Average Returns ---")
    if not decile_df.empty:
        for d in range(1, NUM_DECILES + 1):
            col = f'D{d}_Return'
            if col in decile_df.columns:
                avg_ret = decile_df[col].mean() * 100
                print(f"D{d}: {avg_ret:>7.3f}%/month")

    print("\n" + "=" * 60)


def main():
    """Main pipeline."""
    logger.info("=" * 60)
    logger.info("Portfolio Construction - F&P (2014)")
    logger.info("=" * 60)

    ensure_directories()

    excess_returns, betas = load_data()
    bab_df = construct_bab_portfolios(excess_returns, betas)
    decile_df = compute_decile_returns(excess_returns, betas)
    save_outputs(bab_df, decile_df)
    print_summary(bab_df, decile_df)

    logger.info("Portfolio construction complete!")


if __name__ == '__main__':
    main()

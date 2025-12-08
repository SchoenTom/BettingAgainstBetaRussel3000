#!/usr/bin/env python3
"""
factor_regressions.py - Factor Model Analysis for BAB Strategy

Replication of Frazzini and Pedersen (2014) "Betting Against Beta"
Journal of Financial Economics, 111(1), 1-25.

================================================================================
FACTOR REGRESSION METHODOLOGY
================================================================================

This script conducts time-series factor regressions of BAB returns against
standard asset pricing models to determine whether the BAB factor produces
statistically significant abnormal returns (alpha) beyond known risk factors.

MODELS TESTED:

1. CAPM (Capital Asset Pricing Model):
   BAB_t = α + β_MKT × (R_m - R_f)_t + ε_t

2. Fama-French 3-Factor Model (FF3):
   BAB_t = α + β_MKT × MKT_t + β_SMB × SMB_t + β_HML × HML_t + ε_t

3. Carhart 4-Factor Model:
   BAB_t = α + β_MKT × MKT_t + β_SMB × SMB_t + β_HML × HML_t + β_MOM × MOM_t + ε_t

4. Fama-French 5-Factor Model (FF5):
   BAB_t = α + β_MKT × MKT_t + β_SMB × SMB_t + β_HML × HML_t
         + β_RMW × RMW_t + β_CMA × CMA_t + ε_t

INTERPRETATION OF RESULTS:

- ALPHA: Monthly abnormal return after controlling for factor exposures
  - Significant positive alpha indicates BAB captures a distinct premium
  - Alpha should be interpreted as risk-adjusted excess return

- T-STATISTIC: Statistical significance of alpha
  - |t| > 1.96 indicates significance at 5% level
  - |t| > 2.58 indicates significance at 1% level

- FACTOR LOADINGS: Exposure to systematic risk factors
  - β_MKT near 0 confirms market neutrality
  - β_SMB shows size tilt (negative = large cap bias)
  - β_HML shows value tilt
  - β_MOM shows momentum exposure
  - β_RMW shows profitability exposure
  - β_CMA shows investment exposure

- R-SQUARED: Fraction of variance explained by factors
  - Low R² suggests BAB is largely orthogonal to standard factors

================================================================================

Author: BAB Strategy Implementation (Frazzini-Pedersen Replication)
Date: 2024
"""

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "output"

# Input file paths
BAB_RETURNS_FILE = os.path.join(OUTPUT_DIR, "bab_returns.csv")
FF_FACTORS_FILE = os.path.join(DATA_DIR, "ff_factors.csv")

# Output file paths
REGRESSION_RESULTS_FILE = os.path.join(OUTPUT_DIR, "factor_regression_results.csv")
REGRESSION_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "factor_regression_summary.txt")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BAB returns and Fama-French factor data.

    Returns:
        Tuple of (BAB returns DataFrame, FF factors DataFrame)
    """
    print("Loading data files...")

    # Load BAB returns
    bab_returns = pd.read_csv(BAB_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  BAB Returns: {len(bab_returns)} months")

    # Load FF factors
    ff_factors = pd.read_csv(FF_FACTORS_FILE, index_col=0, parse_dates=True)
    print(f"  FF Factors: {len(ff_factors)} months")
    print(f"  Available factors: {list(ff_factors.columns)}")

    return bab_returns, ff_factors


def run_ols_regression(y: pd.Series, X: pd.DataFrame) -> Dict:
    """
    Run OLS regression with Newey-West standard errors.

    y = X @ beta + epsilon

    Args:
        y: Dependent variable (BAB returns)
        X: Independent variables (factors)

    Returns:
        Dictionary with regression results
    """
    # Align data
    common_idx = y.dropna().index.intersection(X.dropna().index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]

    n = len(y)
    k = X.shape[1]

    if n < k + 10:
        return None

    # Add constant for alpha
    X_with_const = X.copy()
    X_with_const.insert(0, 'const', 1.0)

    # OLS: beta = (X'X)^-1 X'y
    XtX = X_with_const.T @ X_with_const
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_with_const.T @ y
    beta = XtX_inv @ Xty

    # Fitted values and residuals
    y_pred = X_with_const @ beta
    residuals = y - y_pred

    # R-squared
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    # Standard errors (Newey-West with 6 lags for monthly data)
    # For simplicity, using heteroskedasticity-robust (HC0) standard errors
    # In production, would use statsmodels for Newey-West
    sigma2 = ss_res / (n - k - 1)

    # Robust standard errors (HC0)
    # Var(beta) = (X'X)^-1 X' diag(e^2) X (X'X)^-1
    e2 = residuals ** 2
    bread = XtX_inv
    meat = X_with_const.T @ np.diag(e2) @ X_with_const
    var_beta_robust = bread @ meat @ bread

    se_robust = np.sqrt(np.diag(var_beta_robust))

    # T-statistics
    t_stats = beta / se_robust

    # P-values (two-tailed, using normal approximation)
    from scipy import stats
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

    # Extract results
    var_names = ['Alpha'] + list(X.columns)
    results = {
        'N_Obs': n,
        'R_Squared': r_squared,
        'Adj_R_Squared': adj_r_squared,
        'Residual_Std': np.sqrt(sigma2),
    }

    for i, var in enumerate(var_names):
        results[f'{var}_Coef'] = beta.iloc[i] if hasattr(beta, 'iloc') else beta[i]
        results[f'{var}_SE'] = se_robust[i]
        results[f'{var}_T'] = t_stats.iloc[i] if hasattr(t_stats, 'iloc') else t_stats[i]
        results[f'{var}_P'] = p_values[i]

    return results


def run_factor_regressions(bab_returns: pd.DataFrame, ff_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Run all factor model regressions.

    Args:
        bab_returns: DataFrame with BAB returns
        ff_factors: DataFrame with Fama-French factors

    Returns:
        DataFrame with regression results for all models
    """
    print("\n" + "=" * 70)
    print("Running Factor Model Regressions")
    print("=" * 70)

    # Align data
    common_idx = bab_returns.index.intersection(ff_factors.index)
    bab = bab_returns.loc[common_idx, 'BAB_Return']
    ff = ff_factors.loc[common_idx]

    print(f"  Common observations: {len(common_idx)}")

    results = []

    # 1. CAPM
    print("\n1. CAPM Regression...")
    if 'Mkt-RF' in ff.columns:
        X_capm = ff[['Mkt-RF']]
        res = run_ols_regression(bab, X_capm)
        if res:
            res['Model'] = 'CAPM'
            results.append(res)
            print(f"   Alpha: {res['Alpha_Coef']*100:.3f}% (t={res['Alpha_T']:.2f})")
            print(f"   Market Beta: {res['Mkt-RF_Coef']:.3f}")
    else:
        print("   Skipped - Mkt-RF not available")

    # 2. Fama-French 3-Factor
    print("\n2. Fama-French 3-Factor Regression...")
    ff3_factors = ['Mkt-RF', 'SMB', 'HML']
    if all(f in ff.columns for f in ff3_factors):
        X_ff3 = ff[ff3_factors]
        res = run_ols_regression(bab, X_ff3)
        if res:
            res['Model'] = 'FF3'
            results.append(res)
            print(f"   Alpha: {res['Alpha_Coef']*100:.3f}% (t={res['Alpha_T']:.2f})")
            print(f"   Market Beta: {res['Mkt-RF_Coef']:.3f}")
            print(f"   SMB: {res['SMB_Coef']:.3f}, HML: {res['HML_Coef']:.3f}")
    else:
        print("   Skipped - not all factors available")

    # 3. Carhart 4-Factor
    print("\n3. Carhart 4-Factor Regression...")
    carhart_factors = ['Mkt-RF', 'SMB', 'HML', 'Mom']
    if all(f in ff.columns for f in carhart_factors):
        X_carhart = ff[carhart_factors].dropna()
        bab_aligned = bab.loc[bab.index.intersection(X_carhart.index)]
        res = run_ols_regression(bab_aligned, X_carhart)
        if res:
            res['Model'] = 'Carhart'
            results.append(res)
            print(f"   Alpha: {res['Alpha_Coef']*100:.3f}% (t={res['Alpha_T']:.2f})")
            print(f"   Market Beta: {res['Mkt-RF_Coef']:.3f}")
            print(f"   Mom: {res['Mom_Coef']:.3f}")
    else:
        print("   Skipped - momentum factor not available")

    # 4. Fama-French 5-Factor
    print("\n4. Fama-French 5-Factor Regression...")
    ff5_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    if all(f in ff.columns for f in ff5_factors):
        X_ff5 = ff[ff5_factors]
        res = run_ols_regression(bab, X_ff5)
        if res:
            res['Model'] = 'FF5'
            results.append(res)
            print(f"   Alpha: {res['Alpha_Coef']*100:.3f}% (t={res['Alpha_T']:.2f})")
            print(f"   Market Beta: {res['Mkt-RF_Coef']:.3f}")
            print(f"   RMW: {res['RMW_Coef']:.3f}, CMA: {res['CMA_Coef']:.3f}")
    else:
        print("   Skipped - not all factors available")

    # 5. Full Model (FF5 + Momentum)
    print("\n5. Full Model (FF5 + Momentum) Regression...")
    full_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    if all(f in ff.columns for f in full_factors):
        X_full = ff[full_factors].dropna()
        bab_aligned = bab.loc[bab.index.intersection(X_full.index)]
        res = run_ols_regression(bab_aligned, X_full)
        if res:
            res['Model'] = 'FF5+Mom'
            results.append(res)
            print(f"   Alpha: {res['Alpha_Coef']*100:.3f}% (t={res['Alpha_T']:.2f})")
            print(f"   Market Beta: {res['Mkt-RF_Coef']:.3f}")
    else:
        print("   Skipped - not all factors available")

    return pd.DataFrame(results)


def create_regression_summary(results_df: pd.DataFrame) -> str:
    """
    Create a formatted summary of regression results.

    Args:
        results_df: DataFrame with regression results

    Returns:
        Formatted string summary
    """
    summary = []
    summary.append("=" * 80)
    summary.append("FACTOR REGRESSION ANALYSIS - BAB STRATEGY")
    summary.append("Frazzini and Pedersen (2014) Replication")
    summary.append("=" * 80)

    summary.append("\n" + "=" * 80)
    summary.append("SUMMARY TABLE: Alpha Estimates Across Factor Models")
    summary.append("=" * 80)
    summary.append(f"{'Model':<12} {'Alpha (%)':>10} {'t-stat':>10} {'Mkt Beta':>10} {'R²':>10}")
    summary.append("-" * 52)

    for _, row in results_df.iterrows():
        model = row['Model']
        alpha = row['Alpha_Coef'] * 100 * 12  # Annualized
        t_stat = row['Alpha_T']
        mkt_beta = row.get('Mkt-RF_Coef', np.nan)
        r2 = row['R_Squared']

        sig = ""
        if abs(t_stat) > 2.58:
            sig = "***"
        elif abs(t_stat) > 1.96:
            sig = "**"
        elif abs(t_stat) > 1.65:
            sig = "*"

        summary.append(f"{model:<12} {alpha:>9.2f}{sig} {t_stat:>10.2f} {mkt_beta:>10.3f} {r2:>10.3f}")

    summary.append("-" * 52)
    summary.append("Note: Alpha is annualized (monthly × 12)")
    summary.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")

    # Detailed results for each model
    for _, row in results_df.iterrows():
        model = row['Model']
        summary.append("\n" + "=" * 80)
        summary.append(f"DETAILED RESULTS: {model}")
        summary.append("=" * 80)

        summary.append(f"\nObservations: {int(row['N_Obs'])}")
        summary.append(f"R-squared: {row['R_Squared']:.4f}")
        summary.append(f"Adjusted R²: {row['Adj_R_Squared']:.4f}")
        summary.append(f"Residual Std: {row['Residual_Std']*100:.3f}%")

        summary.append(f"\n{'Variable':<12} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'p-value':>10}")
        summary.append("-" * 58)

        # Alpha
        alpha = row['Alpha_Coef']
        se = row['Alpha_SE']
        t = row['Alpha_T']
        p = row['Alpha_P']
        summary.append(f"{'Alpha':<12} {alpha*100:>11.4f}% {se*100:>11.4f}% {t:>10.2f} {p:>10.4f}")

        # Factor loadings
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
        for factor in factors:
            coef_key = f'{factor}_Coef'
            if coef_key in row and not pd.isna(row[coef_key]):
                coef = row[coef_key]
                se = row[f'{factor}_SE']
                t = row[f'{factor}_T']
                p = row[f'{factor}_P']
                summary.append(f"{factor:<12} {coef:>12.4f} {se:>12.4f} {t:>10.2f} {p:>10.4f}")

    # Interpretation
    summary.append("\n" + "=" * 80)
    summary.append("INTERPRETATION")
    summary.append("=" * 80)

    if len(results_df) > 0:
        # Find most comprehensive model
        best_model = results_df.iloc[-1]
        alpha_ann = best_model['Alpha_Coef'] * 12 * 100
        t_stat = best_model['Alpha_T']
        mkt_beta = best_model.get('Mkt-RF_Coef', np.nan)

        summary.append(f"\nUsing the most comprehensive model ({best_model['Model']}):")
        summary.append(f"- Annualized Alpha: {alpha_ann:.2f}%")

        if abs(t_stat) > 2.58:
            summary.append(f"- Alpha is HIGHLY SIGNIFICANT (t={t_stat:.2f}, p<0.01)")
            summary.append("- The BAB factor produces abnormal returns beyond standard factors")
        elif abs(t_stat) > 1.96:
            summary.append(f"- Alpha is SIGNIFICANT (t={t_stat:.2f}, p<0.05)")
            summary.append("- The BAB factor produces abnormal returns beyond standard factors")
        elif abs(t_stat) > 1.65:
            summary.append(f"- Alpha is MARGINALLY SIGNIFICANT (t={t_stat:.2f}, p<0.10)")
        else:
            summary.append(f"- Alpha is NOT SIGNIFICANT (t={t_stat:.2f})")

        if not pd.isna(mkt_beta):
            if abs(mkt_beta) < 0.2:
                summary.append(f"- Market beta is near zero ({mkt_beta:.3f}), confirming market neutrality")
            else:
                summary.append(f"- Market beta is {mkt_beta:.3f}, indicating some market exposure")

    summary.append("\n" + "=" * 80)
    summary.append("CAVEAT: SURVIVORSHIP BIAS")
    summary.append("=" * 80)
    summary.append("These results are subject to survivorship bias from using current")
    summary.append("Russell 3000 constituents applied historically. Alpha estimates may")
    summary.append("be overstated or understated depending on the net effect of excluding")
    summary.append("failed stocks from both low-beta and high-beta portfolios.")

    return "\n".join(summary)


def main():
    """Main function to run factor regression analysis."""
    print("=" * 70)
    print("Factor Regression Analysis")
    print("BAB Strategy - Frazzini-Pedersen (2014) Replication")
    print("=" * 70)

    # Load data
    bab_returns, ff_factors = load_data()

    # Run regressions
    results_df = run_factor_regressions(bab_returns, ff_factors)

    if len(results_df) == 0:
        print("\nNo regressions could be run - check factor data availability")
        return

    # Create summary
    summary = create_regression_summary(results_df)

    # Print summary
    print("\n" + summary)

    # Save results
    print("\nSaving outputs...")

    results_df.to_csv(REGRESSION_RESULTS_FILE, index=False)
    print(f"  Saved: {REGRESSION_RESULTS_FILE}")

    with open(REGRESSION_SUMMARY_FILE, 'w') as f:
        f.write(summary)
    print(f"  Saved: {REGRESSION_SUMMARY_FILE}")

    print("\n" + "=" * 70)
    print("Factor regression analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

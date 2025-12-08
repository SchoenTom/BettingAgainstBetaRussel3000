#!/usr/bin/env python3
"""
illustrations.py - Betting-Against-Beta (BAB) Strategy Visualizations

Replication of Frazzini and Pedersen (2014) "Betting Against Beta"
Journal of Financial Economics, 111(1), 1-25.

================================================================================
VISUALIZATION OUTPUTS
================================================================================

This script generates comprehensive visualizations for BAB strategy analysis:

1. PERFORMANCE PLOTS:
   - Cumulative equity curves (BAB vs IWV)
   - Log-scale cumulative returns
   - Annual returns comparison

2. ROLLING METRICS:
   - Rolling 12-month Sharpe ratio
   - Rolling excess returns over IWV

3. MARKET NEUTRALITY:
   - Rolling portfolio beta (critical for verifying market neutrality)
   - BAB vs IWV scatter plot with regression

4. BETA ANALYSIS:
   - Beta spread over time (Q5 - Q1)
   - Quintile betas over time
   - Average returns by beta quintile

5. RISK ANALYSIS:
   - Drawdown comparison
   - Return distributions

All plots are saved as PNG files in the output/figures directory.

================================================================================

Author: BAB Strategy Implementation (Frazzini-Pedersen Replication)
Date: 2024
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Input file paths
MONTHLY_PERFORMANCE_FILE = os.path.join(OUTPUT_DIR, "bab_monthly_performance.csv")
BAB_RETURNS_FILE = os.path.join(OUTPUT_DIR, "bab_returns.csv")
QUINTILE_STATS_FILE = os.path.join(OUTPUT_DIR, "quintile_statistics.csv")
ROLLING_BETA_FILE = os.path.join(OUTPUT_DIR, "rolling_portfolio_beta.csv")
REGRESSION_RESULTS_FILE = os.path.join(OUTPUT_DIR, "factor_regression_results.csv")

# Plot settings
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

FIGSIZE = (12, 6)
FIGSIZE_LARGE = (14, 8)
DPI = 150
COLORS = {
    'bab': '#2E86AB',      # Blue
    'iwv': '#A23B72',      # Purple/Pink
    'q1': '#28A745',       # Green
    'q5': '#DC3545',       # Red
    'spread': '#FFC107',   # Yellow/Gold
    'neutral': '#6C757D',  # Gray
    'scaled': '#17A2B8',   # Cyan
    'unscaled': '#FD7E14', # Orange
}


def ensure_figures_dir() -> None:
    """Create figures directory if it doesn't exist."""
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        print(f"Created figures directory: {FIGURES_DIR}")


def load_data() -> tuple:
    """
    Load required data for visualizations.

    Returns:
        Tuple of DataFrames
    """
    print("Loading data files...")

    monthly_perf = pd.read_csv(MONTHLY_PERFORMANCE_FILE, index_col=0, parse_dates=True)
    print(f"  Monthly performance: {len(monthly_perf)} rows")

    bab_returns = pd.read_csv(BAB_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"  BAB returns: {len(bab_returns)} rows")

    quintile_stats = None
    if os.path.exists(QUINTILE_STATS_FILE):
        quintile_stats = pd.read_csv(QUINTILE_STATS_FILE, parse_dates=['Date'])
        print(f"  Quintile statistics: {len(quintile_stats)} rows")

    rolling_beta = None
    if os.path.exists(ROLLING_BETA_FILE):
        rolling_beta = pd.read_csv(ROLLING_BETA_FILE, index_col=0, parse_dates=True)
        print(f"  Rolling beta: {len(rolling_beta)} rows")

    regression_results = None
    if os.path.exists(REGRESSION_RESULTS_FILE):
        regression_results = pd.read_csv(REGRESSION_RESULTS_FILE)
        print(f"  Regression results: {len(regression_results)} models")

    return monthly_perf, bab_returns, quintile_stats, rolling_beta, regression_results


def plot_cumulative_returns(monthly_perf: pd.DataFrame) -> None:
    """Plot cumulative equity curves for BAB and IWV."""
    print("Generating cumulative returns plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(monthly_perf.index, monthly_perf['BAB_Cumulative'],
            color=COLORS['bab'], linewidth=2, label='BAB Strategy (Scaled)')
    ax.plot(monthly_perf.index, monthly_perf['IWV_Cumulative'],
            color=COLORS['iwv'], linewidth=2, label='IWV (Russell 3000)')

    ax.axhline(y=1, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Growth of $1)', fontsize=12)
    ax.set_title('Betting-Against-Beta vs Russell 3000 (IWV)\nCumulative Returns',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # Add final values annotation
    final_bab = monthly_perf['BAB_Cumulative'].iloc[-1]
    final_iwv = monthly_perf['IWV_Cumulative'].iloc[-1]
    ax.annotate(f'BAB: ${final_bab:.2f}', xy=(monthly_perf.index[-1], final_bab),
                xytext=(10, 0), textcoords='offset points', fontsize=10,
                color=COLORS['bab'], fontweight='bold')
    ax.annotate(f'IWV: ${final_iwv:.2f}', xy=(monthly_perf.index[-1], final_iwv),
                xytext=(10, 0), textcoords='offset points', fontsize=10,
                color=COLORS['iwv'], fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'cumulative_returns.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_rolling_portfolio_beta(monthly_perf: pd.DataFrame, rolling_beta: pd.DataFrame = None) -> None:
    """
    Plot rolling portfolio beta to verify market neutrality.

    This is a critical diagnostic for the BAB strategy - beta should stay near 0.
    """
    print("Generating rolling portfolio beta plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Use rolling beta from separate file if available, otherwise compute
    if rolling_beta is not None and 'Rolling_Beta_36M' in rolling_beta.columns:
        beta_series = rolling_beta['Rolling_Beta_36M'].dropna()
    elif 'Rolling_Beta_36M' in monthly_perf.columns:
        beta_series = monthly_perf['Rolling_Beta_36M'].dropna()
    else:
        print("  Skipping - rolling beta data not available")
        return

    ax.plot(beta_series.index, beta_series, color=COLORS['bab'], linewidth=1.5,
            label='Rolling 36-Month Market Beta')

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='Target (β=0)')
    ax.axhline(y=0.2, color=COLORS['q5'], linestyle='--', alpha=0.7, label='±0.2 Bounds')
    ax.axhline(y=-0.2, color=COLORS['q5'], linestyle='--', alpha=0.7)

    # Shade region
    ax.fill_between(beta_series.index, -0.2, 0.2, alpha=0.1, color=COLORS['q1'])

    # Add mean beta annotation
    mean_beta = beta_series.mean()
    ax.axhline(y=mean_beta, color=COLORS['spread'], linestyle=':', linewidth=1.5,
               label=f'Mean β: {mean_beta:.3f}')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Beta', fontsize=12)
    ax.set_title('BAB Strategy Rolling 36-Month Market Beta\n(Verifying Market Neutrality)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # Add text box with statistics
    pct_in_bounds = (abs(beta_series) < 0.2).sum() / len(beta_series) * 100
    textstr = f'Mean β: {mean_beta:.3f}\nStd β: {beta_series.std():.3f}\n% within ±0.2: {pct_in_bounds:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'rolling_portfolio_beta.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_scaled_vs_unscaled(bab_returns: pd.DataFrame) -> None:
    """Plot comparison of scaled vs unscaled BAB returns."""
    print("Generating scaled vs unscaled comparison plot...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Cumulative returns
    scaled_cum = (1 + bab_returns['BAB_Return']).cumprod()
    unscaled_cum = (1 + bab_returns['BAB_Return_Unscaled']).cumprod()

    axes[0].plot(bab_returns.index, scaled_cum, color=COLORS['scaled'],
                 linewidth=2, label='Scaled (Market Neutral)')
    axes[0].plot(bab_returns.index, unscaled_cum, color=COLORS['unscaled'],
                 linewidth=2, label='Unscaled (Q1 - Q5)')
    axes[0].axhline(y=1, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].set_title('BAB Returns: Scaled vs Unscaled\n(Beta Scaling Creates Market Neutrality)',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=11)

    # Monthly returns comparison
    axes[1].bar(bab_returns.index, bab_returns['BAB_Return'] * 100,
                color=COLORS['scaled'], alpha=0.7, label='Scaled', width=20)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Monthly Return (%)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'scaled_vs_unscaled.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_leverage_over_time(bab_returns: pd.DataFrame) -> None:
    """Plot leverage applied to long and short legs over time."""
    print("Generating leverage over time plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(bab_returns.index, bab_returns['Long_Leverage'],
            color=COLORS['q1'], linewidth=1.5, label='Long Leverage (1/β_L)')
    ax.plot(bab_returns.index, bab_returns['Short_Leverage'],
            color=COLORS['q5'], linewidth=1.5, label='Short Leverage (1/β_H)')

    ax.axhline(y=1, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='No Leverage (1x)')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Leverage Multiple', fontsize=12)
    ax.set_title('Leverage Applied to Long and Short Portfolios\n(To Achieve Beta = 1 on Each Leg)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # Add statistics
    mean_long = bab_returns['Long_Leverage'].mean()
    mean_short = bab_returns['Short_Leverage'].mean()
    textstr = f'Mean Long Lev: {mean_long:.2f}x\nMean Short Lev: {mean_short:.2f}x'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'leverage_over_time.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_rolling_sharpe(monthly_perf: pd.DataFrame) -> None:
    """Plot rolling 12-month Sharpe ratio for BAB."""
    print("Generating rolling Sharpe ratio plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    rolling_sharpe = monthly_perf['Rolling_Sharpe'].dropna()

    ax.plot(rolling_sharpe.index, rolling_sharpe,
            color=COLORS['bab'], linewidth=1.5, label='Rolling 12-Month Sharpe')
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=1)

    # Add mean line
    mean_sharpe = rolling_sharpe.mean()
    ax.axhline(y=mean_sharpe, color=COLORS['q1'], linestyle=':', linewidth=1.5,
               label=f'Mean Sharpe: {mean_sharpe:.2f}')

    # Shade positive/negative regions
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                    where=rolling_sharpe > 0, alpha=0.3, color=COLORS['q1'])
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                    where=rolling_sharpe < 0, alpha=0.3, color=COLORS['q5'])

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('BAB Strategy Rolling 12-Month Sharpe Ratio',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'rolling_sharpe.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_beta_spread(monthly_perf: pd.DataFrame) -> None:
    """Plot beta spread (Q5 - Q1) over time."""
    print("Generating beta spread plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    beta_spread = monthly_perf['Beta_Spread']

    ax.plot(monthly_perf.index, beta_spread,
            color=COLORS['spread'], linewidth=1.5, label='Beta Spread (Q5 - Q1)')

    mean_spread = beta_spread.mean()
    ax.axhline(y=mean_spread, color=COLORS['q5'], linestyle='--', linewidth=1.5,
               label=f'Mean Spread: {mean_spread:.2f}')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Beta Spread', fontsize=12)
    ax.set_title('Beta Spread: High Beta (Q5) minus Low Beta (Q1) Quintile',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'beta_spread.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_quintile_betas(monthly_perf: pd.DataFrame) -> None:
    """Plot Q1 and Q5 mean betas over time."""
    print("Generating quintile betas plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(monthly_perf.index, monthly_perf['Q1_Mean_Beta'],
            color=COLORS['q1'], linewidth=1.5, label='Q1 (Low Beta)')
    ax.plot(monthly_perf.index, monthly_perf['Q5_Mean_Beta'],
            color=COLORS['q5'], linewidth=1.5, label='Q5 (High Beta)')

    ax.axhline(y=1, color=COLORS['neutral'], linestyle='--', alpha=0.5,
               label='Market Beta (1.0)')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Beta', fontsize=12)
    ax.set_title('Average Beta by Quintile Over Time',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'quintile_betas.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_drawdowns(monthly_perf: pd.DataFrame) -> None:
    """Plot drawdown comparison for BAB and IWV."""
    print("Generating drawdowns plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.fill_between(monthly_perf.index, monthly_perf['BAB_Drawdown'] * 100, 0,
                    color=COLORS['bab'], alpha=0.5, label='BAB')
    ax.fill_between(monthly_perf.index, monthly_perf['IWV_Drawdown'] * 100, 0,
                    color=COLORS['iwv'], alpha=0.5, label='IWV')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Underwater Chart: BAB vs IWV Drawdowns',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='lower right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    max_bab_dd = monthly_perf['BAB_Drawdown'].min() * 100
    max_iwv_dd = monthly_perf['IWV_Drawdown'].min() * 100
    ax.text(0.02, 0.05, f'Max BAB DD: {max_bab_dd:.1f}%\nMax IWV DD: {max_iwv_dd:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'drawdowns.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_scatter_bab_vs_iwv(monthly_perf: pd.DataFrame) -> None:
    """Plot scatter of BAB returns vs IWV returns with regression line."""
    print("Generating BAB vs IWV scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    bab = monthly_perf['BAB_Return'] * 100
    iwv = monthly_perf['IWV_Return'] * 100

    # Remove NaNs
    mask = bab.notna() & iwv.notna()
    bab = bab[mask]
    iwv = iwv[mask]

    ax.scatter(iwv, bab, alpha=0.5, color=COLORS['bab'], s=30)

    # Add regression line
    z = np.polyfit(iwv, bab, 1)
    p = np.poly1d(z)
    x_line = np.linspace(iwv.min(), iwv.max(), 100)
    ax.plot(x_line, p(x_line), color='red', linestyle='--',
            linewidth=2, label=f'β = {z[0]:.3f}')

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    correlation = bab.corr(iwv)
    textstr = f'Correlation: {correlation:.3f}\nMarket β: {z[0]:.3f}\nAlpha: {z[1]:.3f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.set_xlabel('IWV Monthly Return (%)', fontsize=12)
    ax.set_ylabel('BAB Monthly Return (%)', fontsize=12)
    ax.set_title('BAB vs IWV Monthly Returns\n(Low Beta Confirms Market Neutrality)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'scatter_bab_vs_iwv.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_quintile_returns_bar(quintile_stats: pd.DataFrame) -> None:
    """Plot average returns by quintile (both raw and scaled)."""
    if quintile_stats is None:
        print("Skipping quintile returns plot (no data)...")
        return

    print("Generating quintile returns bar plot...")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # Raw returns
    avg_returns = quintile_stats.groupby('Quintile')['Mean_Return'].mean() * 100
    colors = [COLORS['q1'], '#5CB85C', COLORS['neutral'], '#E67E22', COLORS['q5']]

    axes[0].bar(avg_returns.index, avg_returns.values, color=colors, edgecolor='white')
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_xlabel('Beta Quintile', fontsize=12)
    axes[0].set_ylabel('Average Monthly Return (%)', fontsize=12)
    axes[0].set_title('Raw Returns by Beta Quintile', fontsize=13, fontweight='bold')
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].set_xticklabels(['Q1\n(Low β)', 'Q2', 'Q3', 'Q4', 'Q5\n(High β)'])

    # Add value labels
    for i, v in enumerate(avg_returns.values):
        axes[0].annotate(f'{v:.2f}%', xy=(i+1, v), ha='center',
                         va='bottom' if v >= 0 else 'top', fontsize=10)

    # Scaled returns (beta-adjusted)
    if 'Scaled_Return' in quintile_stats.columns:
        avg_scaled = quintile_stats.groupby('Quintile')['Scaled_Return'].mean() * 100
        axes[1].bar(avg_scaled.index, avg_scaled.values, color=colors, edgecolor='white')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_xlabel('Beta Quintile', fontsize=12)
        axes[1].set_ylabel('Average Monthly Return (%)', fontsize=12)
        axes[1].set_title('Beta-Scaled Returns by Quintile\n(Return per Unit Beta)',
                          fontsize=13, fontweight='bold')
        axes[1].set_xticks([1, 2, 3, 4, 5])
        axes[1].set_xticklabels(['Q1\n(Low β)', 'Q2', 'Q3', 'Q4', 'Q5\n(High β)'])

        for i, v in enumerate(avg_scaled.values):
            if not np.isnan(v):
                axes[1].annotate(f'{v:.2f}%', xy=(i+1, v), ha='center',
                                 va='bottom' if v >= 0 else 'top', fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'quintile_returns.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_factor_regression_summary(regression_results: pd.DataFrame) -> None:
    """Plot factor regression results summary."""
    if regression_results is None or len(regression_results) == 0:
        print("Skipping factor regression plot (no data)...")
        return

    print("Generating factor regression summary plot...")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # Alpha comparison across models
    models = regression_results['Model'].values
    alphas = regression_results['Alpha_Coef'].values * 12 * 100  # Annualized %
    t_stats = regression_results['Alpha_T'].values

    x = np.arange(len(models))
    bars = axes[0].bar(x, alphas, color=COLORS['bab'], edgecolor='white')

    # Color bars by significance
    for i, (bar, t) in enumerate(zip(bars, t_stats)):
        if abs(t) > 2.58:
            bar.set_color(COLORS['q1'])  # Highly significant
        elif abs(t) > 1.96:
            bar.set_color(COLORS['scaled'])  # Significant
        elif abs(t) > 1.65:
            bar.set_color(COLORS['spread'])  # Marginally significant

    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_xlabel('Factor Model', fontsize=12)
    axes[0].set_ylabel('Annualized Alpha (%)', fontsize=12)
    axes[0].set_title('Alpha Across Factor Models', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)

    # Add t-stat labels
    for i, (alpha, t) in enumerate(zip(alphas, t_stats)):
        sig = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.65 else ''))
        axes[0].annotate(f'{alpha:.1f}%{sig}\n(t={t:.1f})', xy=(i, alpha),
                         ha='center', va='bottom' if alpha >= 0 else 'top', fontsize=9)

    # Market beta across models
    if 'Mkt-RF_Coef' in regression_results.columns:
        betas = regression_results['Mkt-RF_Coef'].values
        axes[1].bar(x, betas, color=COLORS['iwv'], edgecolor='white')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=2)
        axes[1].axhline(y=0.2, color=COLORS['q5'], linestyle='--', alpha=0.5)
        axes[1].axhline(y=-0.2, color=COLORS['q5'], linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Factor Model', fontsize=12)
        axes[1].set_ylabel('Market Beta', fontsize=12)
        axes[1].set_title('Market Beta Across Models\n(Should be Near Zero)',
                          fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45)

        for i, beta in enumerate(betas):
            axes[1].annotate(f'{beta:.3f}', xy=(i, beta), ha='center',
                             va='bottom' if beta >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'factor_regression_summary.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_annual_returns(monthly_perf: pd.DataFrame) -> None:
    """Plot annual returns comparison."""
    print("Generating annual returns plot...")

    annual_bab = monthly_perf['BAB_Return'].resample('YE').apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    annual_iwv = monthly_perf['IWV_Return'].resample('YE').apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    # Remove NaN years
    annual_bab = annual_bab.dropna()
    annual_iwv = annual_iwv.loc[annual_bab.index]

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    x = np.arange(len(annual_bab))
    width = 0.35

    ax.bar(x - width/2, annual_bab.values, width, label='BAB Strategy',
           color=COLORS['bab'], edgecolor='white')
    ax.bar(x + width/2, annual_iwv.values, width, label='IWV (Russell 3000)',
           color=COLORS['iwv'], edgecolor='white')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.set_title('Annual Returns: BAB Strategy vs IWV',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([d.year for d in annual_bab.index], rotation=45)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'annual_returns.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    """Main function to generate all visualizations."""
    print("=" * 70)
    print("BAB Strategy Visualizations")
    print("Frazzini-Pedersen (2014) Replication")
    print("=" * 70)

    # Create figures directory
    ensure_figures_dir()

    # Load data
    monthly_perf, bab_returns, quintile_stats, rolling_beta, regression_results = load_data()

    # Generate all plots
    print("\nGenerating plots...")

    # Core performance plots
    plot_cumulative_returns(monthly_perf)
    plot_scaled_vs_unscaled(bab_returns)
    plot_annual_returns(monthly_perf)

    # Market neutrality verification (CRITICAL)
    plot_rolling_portfolio_beta(monthly_perf, rolling_beta)
    plot_scatter_bab_vs_iwv(monthly_perf)

    # Beta analysis
    plot_beta_spread(monthly_perf)
    plot_quintile_betas(monthly_perf)
    plot_leverage_over_time(bab_returns)

    # Risk metrics
    plot_rolling_sharpe(monthly_perf)
    plot_drawdowns(monthly_perf)

    # Quintile analysis
    plot_quintile_returns_bar(quintile_stats)

    # Factor regression results
    plot_factor_regression_summary(regression_results)

    print("\n" + "=" * 70)
    print("Visualization generation complete!")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 70)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
illustrations.py - Betting-Against-Beta (BAB) Strategy Visualizations

This script generates visualizations for the BAB strategy analysis:
1. Cumulative equity curves (BAB vs IWV)
2. Rolling 12-month Sharpe ratio and rolling excess returns
3. Beta spread plot (Q5 mean beta - Q1 mean beta)
4. Additional diagnostic plots

All plots are saved as PNG files in the output directory.

Author: BAB Strategy Implementation
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

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
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
        Tuple of DataFrames (monthly_perf, bab_returns, quintile_stats)
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

    return monthly_perf, bab_returns, quintile_stats


def plot_cumulative_returns(monthly_perf: pd.DataFrame) -> None:
    """
    Plot cumulative equity curves for BAB and IWV.

    Args:
        monthly_perf: DataFrame with cumulative returns columns
    """
    print("Generating cumulative returns plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(monthly_perf.index, monthly_perf['BAB_Cumulative'],
            color=COLORS['bab'], linewidth=2, label='BAB Strategy')
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


def plot_cumulative_returns_log(monthly_perf: pd.DataFrame) -> None:
    """
    Plot cumulative equity curves on log scale.

    Args:
        monthly_perf: DataFrame with cumulative returns columns
    """
    print("Generating cumulative returns (log scale) plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.semilogy(monthly_perf.index, monthly_perf['BAB_Cumulative'],
                color=COLORS['bab'], linewidth=2, label='BAB Strategy')
    ax.semilogy(monthly_perf.index, monthly_perf['IWV_Cumulative'],
                color=COLORS['iwv'], linewidth=2, label='IWV (Russell 3000)')

    ax.axhline(y=1, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
    ax.set_title('Betting-Against-Beta vs Russell 3000 (IWV)\nCumulative Returns (Log Scale)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'cumulative_returns_log.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_rolling_sharpe(monthly_perf: pd.DataFrame) -> None:
    """
    Plot rolling 12-month Sharpe ratio for BAB.

    Args:
        monthly_perf: DataFrame with rolling Sharpe column
    """
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


def plot_rolling_excess_return(monthly_perf: pd.DataFrame) -> None:
    """
    Plot rolling 12-month BAB excess return over IWV.

    Args:
        monthly_perf: DataFrame with monthly returns
    """
    print("Generating rolling excess return plot...")

    # Compute rolling 12-month excess return
    bab_returns = monthly_perf['BAB_Return']
    iwv_returns = monthly_perf['IWV_Return']
    excess = bab_returns - iwv_returns
    rolling_excess = excess.rolling(12).mean() * 12  # Annualized

    rolling_excess = rolling_excess.dropna()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(rolling_excess.index, rolling_excess * 100,
            color=COLORS['bab'], linewidth=1.5, label='Rolling 12-Month Excess Return')
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=1)

    # Add mean line
    mean_excess = rolling_excess.mean() * 100
    ax.axhline(y=mean_excess, color=COLORS['q1'], linestyle=':', linewidth=1.5,
               label=f'Mean: {mean_excess:.1f}%')

    # Shade positive/negative regions
    ax.fill_between(rolling_excess.index, 0, rolling_excess * 100,
                    where=rolling_excess > 0, alpha=0.3, color=COLORS['q1'])
    ax.fill_between(rolling_excess.index, 0, rolling_excess * 100,
                    where=rolling_excess < 0, alpha=0.3, color=COLORS['q5'])

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Excess Return (%)', fontsize=12)
    ax.set_title('BAB Strategy Rolling 12-Month Excess Return over IWV',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'rolling_excess_return.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_beta_spread(monthly_perf: pd.DataFrame) -> None:
    """
    Plot beta spread (Q5 mean beta - Q1 mean beta) over time.

    Args:
        monthly_perf: DataFrame with Q1 and Q5 beta columns
    """
    print("Generating beta spread plot...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    beta_spread = monthly_perf['Beta_Spread']

    ax.plot(monthly_perf.index, beta_spread,
            color=COLORS['spread'], linewidth=1.5, label='Beta Spread (Q5 - Q1)')

    # Add mean line
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
    """
    Plot Q1 and Q5 mean betas over time.

    Args:
        monthly_perf: DataFrame with Q1 and Q5 beta columns
    """
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
    """
    Plot drawdown comparison for BAB and IWV.

    Args:
        monthly_perf: DataFrame with drawdown columns
    """
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

    # Add max drawdown annotations
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


def plot_monthly_returns_distribution(monthly_perf: pd.DataFrame) -> None:
    """
    Plot histogram of monthly returns for BAB and IWV.

    Args:
        monthly_perf: DataFrame with monthly returns
    """
    print("Generating monthly returns distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # BAB distribution
    bab_returns = monthly_perf['BAB_Return'] * 100
    axes[0].hist(bab_returns, bins=50, color=COLORS['bab'], alpha=0.7, edgecolor='white')
    axes[0].axvline(x=bab_returns.mean(), color='red', linestyle='--',
                    label=f'Mean: {bab_returns.mean():.2f}%')
    axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[0].set_xlabel('Monthly Return (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('BAB Strategy Monthly Returns', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)

    # IWV distribution
    iwv_returns = monthly_perf['IWV_Return'] * 100
    axes[1].hist(iwv_returns, bins=50, color=COLORS['iwv'], alpha=0.7, edgecolor='white')
    axes[1].axvline(x=iwv_returns.mean(), color='red', linestyle='--',
                    label=f'Mean: {iwv_returns.mean():.2f}%')
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Monthly Return (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('IWV (Russell 3000) Monthly Returns', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'monthly_returns_distribution.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_quintile_returns(quintile_stats: pd.DataFrame) -> None:
    """
    Plot average returns by quintile.

    Args:
        quintile_stats: DataFrame with quintile-level statistics
    """
    if quintile_stats is None:
        print("Skipping quintile returns plot (no data)...")
        return

    print("Generating quintile returns bar plot...")

    # Calculate average monthly return by quintile
    avg_returns = quintile_stats.groupby('Quintile')['Mean_Return'].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['q1'], '#5CB85C', COLORS['neutral'], '#E67E22', COLORS['q5']]
    bars = ax.bar(avg_returns.index, avg_returns.values, color=colors, edgecolor='white')

    ax.set_xlabel('Beta Quintile', fontsize=12)
    ax.set_ylabel('Average Monthly Return (%)', fontsize=12)
    ax.set_title('Average Monthly Returns by Beta Quintile\n(Q1 = Low Beta, Q5 = High Beta)',
                 fontsize=14, fontweight='bold')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['Q1\n(Low Beta)', 'Q2', 'Q3', 'Q4', 'Q5\n(High Beta)'])

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, avg_returns.values):
        ax.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3 if val >= 0 else -12),
                    textcoords='offset points',
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'quintile_returns.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_quintile_betas_bar(quintile_stats: pd.DataFrame) -> None:
    """
    Plot average beta by quintile.

    Args:
        quintile_stats: DataFrame with quintile-level statistics
    """
    if quintile_stats is None:
        print("Skipping quintile betas bar plot (no data)...")
        return

    print("Generating quintile betas bar plot...")

    # Calculate average beta by quintile
    avg_betas = quintile_stats.groupby('Quintile')['Mean_Beta'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['q1'], '#5CB85C', COLORS['neutral'], '#E67E22', COLORS['q5']]
    bars = ax.bar(avg_betas.index, avg_betas.values, color=colors, edgecolor='white')

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Market Beta (1.0)')

    ax.set_xlabel('Beta Quintile', fontsize=12)
    ax.set_ylabel('Average Beta', fontsize=12)
    ax.set_title('Average Beta by Quintile\n(Q1 = Low Beta, Q5 = High Beta)',
                 fontsize=14, fontweight='bold')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['Q1\n(Low Beta)', 'Q2', 'Q3', 'Q4', 'Q5\n(High Beta)'])

    ax.legend(loc='upper left', fontsize=11)

    # Add value labels on bars
    for bar, val in zip(bars, avg_betas.values):
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'quintile_betas_bar.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_annual_returns(monthly_perf: pd.DataFrame) -> None:
    """
    Plot annual returns comparison for BAB and IWV.

    Args:
        monthly_perf: DataFrame with monthly returns
    """
    print("Generating annual returns plot...")

    # Calculate annual returns
    annual_bab = monthly_perf['BAB_Return'].resample('YE').apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    annual_iwv = monthly_perf['IWV_Return'].resample('YE').apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    x = np.arange(len(annual_bab))
    width = 0.35

    bars1 = ax.bar(x - width/2, annual_bab.values, width, label='BAB Strategy',
                   color=COLORS['bab'], edgecolor='white')
    bars2 = ax.bar(x + width/2, annual_iwv.values, width, label='IWV (Russell 3000)',
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


def plot_scatter_bab_vs_iwv(monthly_perf: pd.DataFrame) -> None:
    """
    Plot scatter of BAB returns vs IWV returns.

    Args:
        monthly_perf: DataFrame with monthly returns
    """
    print("Generating BAB vs IWV scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    bab = monthly_perf['BAB_Return'] * 100
    iwv = monthly_perf['IWV_Return'] * 100

    ax.scatter(iwv, bab, alpha=0.5, color=COLORS['bab'], s=30)

    # Add regression line
    z = np.polyfit(iwv, bab, 1)
    p = np.poly1d(z)
    x_line = np.linspace(iwv.min(), iwv.max(), 100)
    ax.plot(x_line, p(x_line), color='red', linestyle='--',
            label=f'Beta: {z[0]:.3f}')

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    correlation = bab.corr(iwv)
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('IWV Monthly Return (%)', fontsize=12)
    ax.set_ylabel('BAB Monthly Return (%)', fontsize=12)
    ax.set_title('BAB vs IWV Monthly Returns Scatter Plot',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'scatter_bab_vs_iwv.png')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("BAB Strategy Visualizations")
    print("=" * 60)

    # Create figures directory
    ensure_figures_dir()

    # Load data
    monthly_perf, bab_returns, quintile_stats = load_data()

    # Generate all plots
    print("\nGenerating plots...")

    # Core plots
    plot_cumulative_returns(monthly_perf)
    plot_cumulative_returns_log(monthly_perf)
    plot_rolling_sharpe(monthly_perf)
    plot_rolling_excess_return(monthly_perf)
    plot_beta_spread(monthly_perf)

    # Additional diagnostic plots
    plot_quintile_betas(monthly_perf)
    plot_drawdowns(monthly_perf)
    plot_monthly_returns_distribution(monthly_perf)
    plot_quintile_returns(quintile_stats)
    plot_quintile_betas_bar(quintile_stats)
    plot_annual_returns(monthly_perf)
    plot_scatter_bab_vs_iwv(monthly_perf)

    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()

"""
illustrations.py - Generate visualizations for BAB strategy

Creates PNG plots:
- Cumulative returns (BAB vs Benchmark)
- Drawdown analysis
- Rolling Sharpe ratio
- Beta spread over time
- Decile returns bar chart
- Annual returns comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import OUTPUT_DIR, NUM_DECILES, FIGURE_SIZE, FIGURE_DPI, COLORS, ensure_directories

FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')


def ensure_figures_dir():
    """Create figures directory."""
    ensure_directories()
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load required data."""
    logger.info("Loading data...")

    perf_path = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    decile_path = os.path.join(OUTPUT_DIR, 'decile_returns.csv')
    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')

    monthly_perf = pd.read_csv(perf_path, index_col=0, parse_dates=True) if os.path.exists(perf_path) else None
    decile_returns = pd.read_csv(decile_path, index_col=0, parse_dates=True) if os.path.exists(decile_path) else None
    bab_df = pd.read_csv(bab_path, index_col=0, parse_dates=True) if os.path.exists(bab_path) else None

    return monthly_perf, decile_returns, bab_df


def plot_cumulative_returns(perf):
    """Plot cumulative returns."""
    if perf is None:
        return

    logger.info("Generating cumulative returns plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(perf.index, perf['BAB_Cumulative'], color=COLORS['bab'], lw=2, label='BAB Strategy')
    ax.plot(perf.index, perf['Benchmark_Cumulative'], color=COLORS['benchmark'], lw=2, label='S&P 500')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.set_title('Cumulative Returns: BAB vs S&P 500')
    ax.legend(loc='upper left')
    ax.set_yscale('log')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cumulative_returns.png')
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_drawdowns(perf):
    """Plot drawdowns."""
    if perf is None:
        return

    logger.info("Generating drawdown plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.fill_between(perf.index, perf['BAB_Drawdown'] * 100, 0,
                    color=COLORS['bab'], alpha=0.5, label='BAB')
    ax.fill_between(perf.index, perf['Benchmark_Drawdown'] * 100, 0,
                    color=COLORS['benchmark'], alpha=0.5, label='S&P 500')

    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Analysis')
    ax.legend(loc='lower right')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'drawdowns.png')
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_rolling_sharpe(perf):
    """Plot rolling Sharpe ratio."""
    if perf is None or 'Rolling_12M_BAB_Sharpe' not in perf.columns:
        return

    logger.info("Generating rolling Sharpe plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    sharpe = perf['Rolling_12M_BAB_Sharpe'].dropna()
    ax.plot(sharpe.index, sharpe, color=COLORS['bab'], lw=1.5)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.fill_between(sharpe.index, 0, sharpe, where=sharpe > 0, alpha=0.3, color=COLORS['low_beta'])
    ax.fill_between(sharpe.index, 0, sharpe, where=sharpe < 0, alpha=0.3, color=COLORS['high_beta'])

    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Rolling 12-Month Sharpe Ratio')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'rolling_sharpe.png')
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_beta_spread(bab_df):
    """Plot beta spread over time."""
    if bab_df is None or 'Beta_Spread' not in bab_df.columns:
        return

    logger.info("Generating beta spread plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    spread = bab_df['Beta_Spread']
    ax.plot(spread.index, spread, color=COLORS['spread'], lw=1.5)

    mean_spread = spread.mean()
    ax.axhline(y=mean_spread, color='darkred', linestyle='--', lw=1.5, label=f'Mean: {mean_spread:.2f}')

    ax.set_xlabel('Date')
    ax.set_ylabel('Beta Spread (High - Low)')
    ax.set_title('Beta Spread Over Time')
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'beta_spread.png')
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_decile_returns(decile_returns):
    """Plot average returns by decile."""
    if decile_returns is None:
        return

    logger.info("Generating decile returns plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    avg_returns = []
    for d in range(1, NUM_DECILES + 1):
        col = f'D{d}_Return'
        if col in decile_returns.columns:
            avg_returns.append(decile_returns[col].mean() * 100)
        else:
            avg_returns.append(0)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, NUM_DECILES))
    bars = ax.bar(range(1, NUM_DECILES + 1), avg_returns, color=colors, edgecolor='white')

    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.set_xlabel('Beta Decile (1=Low, 10=High)')
    ax.set_ylabel('Avg Monthly Return (%)')
    ax.set_title('Average Monthly Returns by Beta Decile')
    ax.set_xticks(range(1, NUM_DECILES + 1))

    for i, v in enumerate(avg_returns):
        ax.text(i + 1, v + 0.02, f'{v:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'decile_returns.png')
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_annual_returns(perf):
    """Plot annual returns comparison."""
    if perf is None:
        return

    logger.info("Generating annual returns plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    annual_bab = (1 + perf['BAB_Return']).resample('YE').prod() - 1
    annual_bench = (1 + perf['Benchmark_Return']).resample('YE').prod() - 1

    common = annual_bab.index.intersection(annual_bench.index)
    annual_bab = annual_bab.loc[common].dropna()
    annual_bench = annual_bench.loc[common].dropna()

    x = np.arange(len(annual_bab))
    width = 0.35

    ax.bar(x - width/2, annual_bab.values * 100, width, label='BAB', color=COLORS['bab'])
    ax.bar(x + width/2, annual_bench.values * 100, width, label='S&P 500', color=COLORS['benchmark'])

    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Annual Returns: BAB vs S&P 500')
    ax.set_xticks(x)
    ax.set_xticklabels([d.year for d in annual_bab.index], rotation=45)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'annual_returns.png')
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def main():
    """Main pipeline."""
    logger.info("=" * 60)
    logger.info("Generating Visualizations")
    logger.info("=" * 60)

    ensure_figures_dir()

    monthly_perf, decile_returns, bab_df = load_data()

    if monthly_perf is None:
        logger.warning("No performance data found. Run backtest.py first.")
        return

    plot_cumulative_returns(monthly_perf)
    plot_drawdowns(monthly_perf)
    plot_rolling_sharpe(monthly_perf)
    plot_beta_spread(bab_df)
    plot_decile_returns(decile_returns)
    plot_annual_returns(monthly_perf)

    logger.info("=" * 60)
    logger.info(f"All figures saved to: {FIGURES_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
main.py - Run the complete BAB strategy pipeline

Pipeline steps:
1. data_loader.py - Download data, compute F&P betas
2. portfolio_construction.py - Build BAB portfolios
3. backtest.py - Compute performance statistics
4. illustrations.py - Generate visualizations

Usage:
    python main.py                    # Full pipeline
    python main.py --skip-download    # Use existing data
    python main.py --skip-plots       # Skip visualizations
    python main.py --only-download    # Only download data
"""

import argparse
import sys
import time
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import DATA_DIR, OUTPUT_DIR, ensure_directories


def check_data_exists():
    """Check if data files exist."""
    required = [
        os.path.join(DATA_DIR, 'monthly_excess_returns.csv'),
        os.path.join(DATA_DIR, 'rolling_betas.csv'),
    ]
    return all(os.path.exists(f) for f in required)


def check_portfolio_exists():
    """Check if portfolio files exist."""
    return os.path.exists(os.path.join(OUTPUT_DIR, 'bab_portfolio.csv'))


def check_backtest_exists():
    """Check if backtest files exist."""
    return os.path.exists(os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv'))


def run_data_loader():
    """Run data loader."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Loading")
    logger.info("=" * 60)

    from data_loader import main as data_main
    data_main()

    if not check_data_exists():
        raise RuntimeError("Data loading failed!")

    logger.info("Data loading complete.")


def run_portfolio_construction():
    """Run portfolio construction."""
    logger.info("=" * 60)
    logger.info("STEP 2: Portfolio Construction")
    logger.info("=" * 60)

    if not check_data_exists():
        raise RuntimeError("Data files missing. Run data_loader.py first.")

    from portfolio_construction import main as portfolio_main
    portfolio_main()

    if not check_portfolio_exists():
        raise RuntimeError("Portfolio construction failed!")

    logger.info("Portfolio construction complete.")


def run_backtest():
    """Run backtest."""
    logger.info("=" * 60)
    logger.info("STEP 3: Backtesting")
    logger.info("=" * 60)

    if not check_portfolio_exists():
        raise RuntimeError("Portfolio files missing.")

    from backtest import main as backtest_main
    backtest_main()

    if not check_backtest_exists():
        raise RuntimeError("Backtest failed!")

    logger.info("Backtesting complete.")


def run_illustrations():
    """Run illustrations."""
    logger.info("=" * 60)
    logger.info("STEP 4: Visualizations")
    logger.info("=" * 60)

    if not check_backtest_exists():
        raise RuntimeError("Backtest files missing.")

    from illustrations import main as illustrations_main
    illustrations_main()

    logger.info("Visualizations complete.")


def print_banner(mode="full"):
    """Print startup banner."""
    print("\n" + "=" * 60)
    print("  BETTING-AGAINST-BETA (BAB) STRATEGY")
    print("  Frazzini & Pedersen (2014) Implementation")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {mode}")
    print("=" * 60 + "\n")


def print_summary(elapsed, steps):
    """Print completion summary."""
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETED")
    print("=" * 60)
    print(f"  Steps: {', '.join(steps)}")
    print(f"  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print("Run dashboard: streamlit run dashboard.py\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run BAB strategy pipeline')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download')
    parser.add_argument('--skip-plots', action='store_true', help='Skip visualizations')
    parser.add_argument('--only-download', action='store_true', help='Only download data')
    parser.add_argument('--only-backtest', action='store_true', help='Skip plots')

    args = parser.parse_args()

    if args.only_download:
        mode = "Data Only"
    elif args.only_backtest:
        mode = "Through Backtest"
    elif args.skip_download and args.skip_plots:
        mode = "Portfolio + Backtest"
    elif args.skip_download:
        mode = "Skip Download"
    elif args.skip_plots:
        mode = "Skip Plots"
    else:
        mode = "Full Pipeline"

    ensure_directories()
    print_banner(mode)

    start = time.time()
    steps = []

    try:
        # Step 1: Data
        if args.only_download:
            run_data_loader()
            steps.append("Data")
            print_summary(time.time() - start, steps)
            return

        if not args.skip_download:
            run_data_loader()
            steps.append("Data")
        else:
            if not check_data_exists():
                logger.error("Data files missing!")
                sys.exit(1)
            logger.info("Using existing data files.")

        # Step 2: Portfolio
        run_portfolio_construction()
        steps.append("Portfolio")

        # Step 3: Backtest
        run_backtest()
        steps.append("Backtest")

        # Step 4: Plots
        if not args.only_backtest and not args.skip_plots:
            run_illustrations()
            steps.append("Plots")
        else:
            logger.info("Skipping visualizations.")

        print_summary(time.time() - start, steps)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
data_loader.py - Download and prepare data for Betting-Against-Beta strategy

Implements Frazzini & Pedersen (2014) beta calculation:
- Correlation: 12-month rolling window
- Volatility: 60-month rolling window
- Beta = correlation * (vol_stock / vol_market)
- Shrinkage: 0.6 * beta_TS + 0.4 * 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import io
import zipfile
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import (
    START_DATE, END_DATE, DATA_DIR, BENCHMARK_TICKER, RUSSELL_3000_TICKERS,
    DOWNLOAD_BATCH_SIZE, ensure_directories, KEN_FRENCH_URL,
    MIN_DATA_COVERAGE, MAX_GAP_MONTHS, REQUIRE_FULL_HISTORY,
    CORRELATION_WINDOW, VOLATILITY_WINDOW, MIN_PERIODS_CORR, MIN_PERIODS_VOL,
    SHRINKAGE_FACTOR, PRIOR_BETA, WINSORIZE_PERCENTILE
)


def download_ken_french_rf():
    """
    Download 1-month T-bill rate from Ken French Data Library.

    Returns:
        pd.Series: Monthly risk-free rate as decimal
    """
    logger.info("Downloading risk-free rate from Ken French Data Library...")

    try:
        import urllib.request
        response = urllib.request.urlopen(KEN_FRENCH_URL, timeout=30)
        zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_name = [n for n in z.namelist() if n.endswith('.csv') or n.endswith('.CSV')][0]
            with z.open(csv_name) as f:
                lines = f.read().decode('utf-8').split('\n')

        # Find data section (skip header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                data_start = i
                break

        # Parse monthly data
        rf_data = []
        for line in lines[data_start:]:
            parts = line.strip().split(',')
            if len(parts) >= 5 and len(parts[0]) == 6:
                try:
                    date_str = parts[0]
                    rf_value = float(parts[4]) / 100  # Convert from percentage
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                    rf_data.append({'Date': date, 'RF': rf_value})
                except (ValueError, IndexError):
                    continue

        rf_df = pd.DataFrame(rf_data)
        rf_df.set_index('Date', inplace=True)
        rf_series = rf_df['RF']
        rf_series.name = 'RF_Rate'

        logger.info(f"Ken French RF: {len(rf_series)} months, avg={rf_series.mean()*100:.3f}%/month")
        return rf_series

    except Exception as e:
        logger.warning(f"Ken French download failed: {e}")
        logger.warning("Falling back to Yahoo ^IRX")
        return download_yahoo_rf()


def download_yahoo_rf():
    """Fallback: Download 3-month T-bill rate from Yahoo Finance."""
    try:
        data = yf.download('^IRX', start=START_DATE, end=END_DATE, interval='1d', progress=False)
        if data.empty:
            raise ValueError("No ^IRX data")

        rf_daily = data['Close'].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data['Close']
        rf_daily.index = pd.to_datetime(rf_daily.index)
        rf_monthly = rf_daily.resample('ME').last()
        rf_monthly_decimal = (1 + rf_monthly / 100) ** (1/12) - 1
        rf_monthly_decimal.name = 'RF_Rate'

        if hasattr(rf_monthly_decimal.index, 'tz'):
            rf_monthly_decimal.index = rf_monthly_decimal.index.tz_localize(None)

        logger.info(f"Yahoo RF: {len(rf_monthly_decimal)} months")
        return rf_monthly_decimal

    except Exception as e:
        logger.warning(f"Yahoo RF failed: {e}, using 2% annual fallback")
        date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='ME')
        monthly_rate = (1 + 0.02) ** (1/12) - 1
        return pd.Series(monthly_rate, index=date_range, name='RF_Rate')


def download_monthly_prices(tickers, start_date, end_date):
    """Download monthly adjusted close prices for all tickers."""
    logger.info(f"Downloading monthly prices for {len(tickers)} tickers...")

    all_data = pd.DataFrame()
    failed_tickers = []

    for i in range(0, len(tickers), DOWNLOAD_BATCH_SIZE):
        batch = tickers[i:i+DOWNLOAD_BATCH_SIZE]
        batch_num = i // DOWNLOAD_BATCH_SIZE + 1
        total_batches = (len(tickers) - 1) // DOWNLOAD_BATCH_SIZE + 1
        logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} tickers)")

        try:
            data = yf.download(
                batch, start=start_date, end=end_date,
                interval='1mo', auto_adjust=True, progress=False, threads=True
            )

            if data.empty:
                failed_tickers.extend(batch)
                continue

            if len(batch) == 1:
                batch_df = data[['Close']].copy() if 'Close' in data.columns else pd.DataFrame()
                batch_df.columns = [batch[0]]
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    batch_df = data['Close'].copy() if 'Close' in data.columns.get_level_values(0) else pd.DataFrame()
                else:
                    batch_df = data[['Close']].copy() if 'Close' in data.columns else pd.DataFrame()

            if not batch_df.empty:
                all_data = batch_df if all_data.empty else all_data.join(batch_df, how='outer')

        except Exception as e:
            logger.warning(f"Batch {batch_num} failed: {e}")
            failed_tickers.extend(batch)

    if all_data.empty:
        raise RuntimeError("No price data downloaded!")

    all_data.index = pd.to_datetime(all_data.index)
    all_data = all_data.resample('ME').last()

    # Data quality filtering
    logger.info("Applying data quality filters...")
    initial_count = len(all_data.columns)

    # 1. Coverage filter
    coverage = all_data.notna().sum() / len(all_data)
    all_data = all_data[coverage[coverage >= MIN_DATA_COVERAGE].index]
    logger.info(f"Coverage filter: {len(all_data.columns)}/{initial_count} tickers")

    # 2. Full history filter
    if REQUIRE_FULL_HISTORY:
        first_valid = all_data.apply(lambda x: x.first_valid_index())
        start_threshold = pd.Timestamp(START_DATE) + pd.DateOffset(months=3)
        all_data = all_data[first_valid[first_valid <= start_threshold].index]
        logger.info(f"History filter: {len(all_data.columns)} tickers")

    # 3. Gap filter
    def has_large_gaps(series, max_gap=MAX_GAP_MONTHS):
        is_na = series.isna()
        if not is_na.any():
            return False
        gaps = is_na.astype(int).groupby((~is_na).cumsum()).sum()
        return gaps.max() > max_gap

    no_gaps = [col for col in all_data.columns if not has_large_gaps(all_data[col])]
    all_data = all_data[no_gaps]
    logger.info(f"Gap filter: {len(all_data.columns)} tickers")

    all_data = all_data.ffill(limit=2)

    logger.info(f"Final: {all_data.shape[1]} stocks, {all_data.shape[0]} months")
    if failed_tickers:
        logger.info(f"Failed: {len(failed_tickers)} tickers")

    return all_data


def download_benchmark(start_date, end_date):
    """Download market benchmark monthly prices."""
    logger.info(f"Downloading benchmark ({BENCHMARK_TICKER})...")

    data = yf.download(
        BENCHMARK_TICKER, start=start_date, end=end_date,
        interval='1mo', auto_adjust=True, progress=False
    )

    if data.empty:
        raise ValueError(f"No data for {BENCHMARK_TICKER}")

    benchmark = data['Close'].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data['Close']
    benchmark.index = pd.to_datetime(benchmark.index)
    benchmark = benchmark.resample('ME').last()
    benchmark.name = 'Benchmark'

    logger.info(f"Benchmark: {len(benchmark)} months")
    return benchmark


def compute_returns(prices):
    """Compute simple monthly returns."""
    returns = prices.pct_change()
    return returns.iloc[1:]


def winsorize(series, percentile=WINSORIZE_PERCENTILE):
    """Winsorize series at given percentile."""
    lower = series.quantile(percentile)
    upper = series.quantile(1 - percentile)
    return series.clip(lower=lower, upper=upper)


def compute_fp_betas(stock_returns, market_returns):
    """
    Compute Frazzini-Pedersen betas with shrinkage.

    Beta_FP = correlation(12m) * (vol_stock(60m) / vol_market(60m))
    Beta_shrunk = 0.6 * Beta_FP + 0.4 * 1.0
    """
    logger.info("Computing F&P betas (correlation 12m, volatility 60m)...")

    common_dates = stock_returns.index.intersection(market_returns.index)
    stocks = stock_returns.loc[common_dates]
    market = market_returns.loc[common_dates]

    betas = pd.DataFrame(index=stocks.index, columns=stocks.columns, dtype=float)

    # Pre-compute market volatility
    market_vol = market.rolling(window=VOLATILITY_WINDOW, min_periods=MIN_PERIODS_VOL).std()

    for col in stocks.columns:
        stock = stocks[col]

        # Correlation over 12 months
        rolling_corr = stock.rolling(window=CORRELATION_WINDOW, min_periods=MIN_PERIODS_CORR).corr(market)

        # Volatility over 60 months
        stock_vol = stock.rolling(window=VOLATILITY_WINDOW, min_periods=MIN_PERIODS_VOL).std()

        # Time-series beta
        beta_ts = rolling_corr * (stock_vol / market_vol)

        # Shrinkage toward prior
        beta_shrunk = SHRINKAGE_FACTOR * beta_ts + (1 - SHRINKAGE_FACTOR) * PRIOR_BETA

        betas[col] = beta_shrunk

    betas = betas.replace([np.inf, -np.inf], np.nan)

    # Winsorize betas
    for date in betas.index:
        row = betas.loc[date].dropna()
        if len(row) > 10:
            betas.loc[date, row.index] = winsorize(row)

    valid_betas = betas.notna().sum().sum()
    total_cells = betas.shape[0] * betas.shape[1]
    logger.info(f"F&P Betas: {valid_betas:,}/{total_cells:,} valid ({100*valid_betas/total_cells:.1f}%)")

    return betas


def save_data(data, filename):
    """Save DataFrame to CSV."""
    ensure_directories()
    filepath = os.path.join(DATA_DIR, filename)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.to_csv(filepath)
    logger.info(f"Saved {filename}: {data.shape}")


def main():
    """Main data loading pipeline."""
    logger.info("=" * 60)
    logger.info("BAB Data Loader - Russell 3000")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info("=" * 60)

    ensure_directories()

    # Get unique tickers
    tickers = list(dict.fromkeys(RUSSELL_3000_TICKERS))
    logger.info(f"Total tickers: {len(tickers)}")

    # Download data
    stock_prices = download_monthly_prices(tickers, START_DATE, END_DATE)
    benchmark_prices = download_benchmark(START_DATE, END_DATE)
    rf_rate = download_ken_french_rf()

    # Compute returns
    stock_returns = compute_returns(stock_prices)
    benchmark_returns = compute_returns(benchmark_prices.to_frame()).iloc[:, 0]

    # Align dates
    common_dates = stock_returns.index.intersection(benchmark_returns.index).intersection(rf_rate.index)
    stock_returns = stock_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    rf_rate = rf_rate.loc[common_dates]

    # Compute excess returns
    stock_excess = stock_returns.subtract(rf_rate, axis=0)
    benchmark_excess = benchmark_returns - rf_rate

    # Compute F&P betas
    rolling_betas = compute_fp_betas(stock_excess, benchmark_excess)

    # Save outputs
    save_data(stock_prices, 'monthly_prices.csv')
    save_data(stock_returns, 'monthly_returns.csv')
    stock_excess['Benchmark'] = benchmark_excess
    save_data(stock_excess, 'monthly_excess_returns.csv')
    save_data(rf_rate, 'risk_free_rate.csv')
    save_data(rolling_betas, 'rolling_betas.csv')

    # Save ticker list
    pd.DataFrame({'Ticker': list(stock_prices.columns)}).to_csv(
        os.path.join(DATA_DIR, 'ticker_list.csv'), index=False
    )

    logger.info("=" * 60)
    logger.info("Data loading complete!")
    logger.info("=" * 60)

    print(f"\n=== Data Summary ===")
    print(f"Stocks: {len(stock_prices.columns)}")
    print(f"Months: {len(stock_prices)}")
    print(f"Date range: {stock_prices.index.min()} to {stock_prices.index.max()}")
    print(f"Avg valid betas (last 12m): {rolling_betas.tail(12).notna().sum(axis=1).mean():.0f}")


if __name__ == '__main__':
    main()

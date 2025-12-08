#!/usr/bin/env python3
"""
data_loader.py - Betting-Against-Beta (BAB) Strategy Data Loader

Replication of Frazzini and Pedersen (2014) "Betting Against Beta"
Journal of Financial Economics, 111(1), 1-25.

================================================================================
SURVIVORSHIP BIAS DECLARATION
================================================================================

IMPORTANT: This implementation uses TODAY'S Russell 3000 constituents (via iShares
IWV ETF holdings) applied historically back to January 2000. This introduces
SURVIVORSHIP BIAS with the following implications:

1. OVERESTIMATION OF LOW-BETA PERFORMANCE:
   - Stocks that survived to today are disproportionately successful
   - Low-beta "boring" stocks that failed (e.g., Kodak, Sears) are excluded
   - This biases low-beta portfolio returns upward

2. UNDERREPRESENTATION OF DISTRESSED HIGH-BETA STOCKS:
   - Failed high-beta stocks (e.g., Enron, WorldCom, Lehman) are excluded
   - High-beta portfolios miss their most negative outcomes
   - This biases high-beta portfolio returns upward (less negative than reality)
   - Net effect: BAB spread may be SMALLER than true historical spread

3. UNIVERSE SIZE MISMATCH:
   - Russell 3000 in 2000 had different constituents than today
   - We use ~3000 current survivors across entire period
   - True implementable strategy would have different stocks each year

4. SECTOR COMPOSITION CHANGES:
   - Tech sector weight much higher today than in 2000
   - Financial sector composition changed dramatically post-2008
   - Energy sector evolved with shale revolution

MITIGATION: Despite these biases, the BAB factor has been documented across:
- Multiple countries and time periods (Frazzini-Pedersen 2014)
- Different universes with point-in-time constituents
- Out-of-sample periods following original publication

The survivorship-biased results here should be interpreted as INDICATIVE rather
than definitive evidence of the BAB premium.

================================================================================

This script downloads and prepares all data required for the BAB strategy:
1. Downloads current Russell 3000 tickers from iShares IWV holdings CSV
2. Cleans invalid symbols and retains only standard stock tickers
3. Fetches monthly adjusted close prices for all tickers, IWV, and ^IRX
4. Downloads Fama-French factor data for factor regressions
5. Computes and saves:
   - Monthly prices (wide DataFrame)
   - Simple monthly returns (pct_change)
   - Monthly excess returns (return - risk-free rate)
   - Rolling 60-month betas (minimum 36 months required for valid estimate)

Author: BAB Strategy Implementation (Frazzini-Pedersen Replication)
Date: 2024
"""

import os
import re
import warnings
from datetime import datetime
from typing import Optional, Tuple
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ROLLING_WINDOW = 60  # months for beta calculation (Frazzini-Pedersen use 60 months)
MIN_PERIODS = 36     # minimum months required for valid beta estimate
IWV_HOLDINGS_URL = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"

# Fama-French data URL (Ken French's data library)
FF_FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FF_MOMENTUM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

# Output file paths
PRICES_FILE = os.path.join(DATA_DIR, "monthly_prices.csv")
RETURNS_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")
EXCESS_RETURNS_FILE = os.path.join(DATA_DIR, "monthly_excess_returns.csv")
BETAS_FILE = os.path.join(DATA_DIR, "rolling_betas.csv")
RF_RATE_FILE = os.path.join(DATA_DIR, "risk_free_rate.csv")
IWV_RETURNS_FILE = os.path.join(DATA_DIR, "iwv_returns.csv")
IWV_EXCESS_RETURNS_FILE = os.path.join(DATA_DIR, "iwv_excess_returns.csv")
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")
FF_FACTORS_FILE = os.path.join(DATA_DIR, "ff_factors.csv")
SURVIVORSHIP_BIAS_FILE = os.path.join(DATA_DIR, "SURVIVORSHIP_BIAS_WARNING.txt")


def ensure_data_dir() -> None:
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")


def write_survivorship_bias_warning() -> None:
    """Write survivorship bias warning file."""
    warning_text = """
================================================================================
SURVIVORSHIP BIAS WARNING
================================================================================

This dataset uses TODAY'S Russell 3000 constituents applied historically.

IMPLICATIONS:
1. Low-beta returns are OVERESTIMATED (failed boring stocks excluded)
2. High-beta returns are LESS NEGATIVE than reality (failed speculative stocks excluded)
3. The BAB spread may be UNDERESTIMATED or OVERESTIMATED depending on net effect
4. Results should be interpreted as INDICATIVE, not definitive

For rigorous academic research, use point-in-time constituent data from:
- CRSP (Center for Research in Security Prices)
- Compustat
- Bloomberg Point-in-Time data

Generated: {date}
Universe: Russell 3000 (IWV ETF Holdings as of download date)
Period: {start} to {end}
================================================================================
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           start=START_DATE, end=END_DATE)

    with open(SURVIVORSHIP_BIAS_FILE, 'w') as f:
        f.write(warning_text)
    print(f"Wrote survivorship bias warning to {SURVIVORSHIP_BIAS_FILE}")


def download_iwv_holdings() -> pd.DataFrame:
    """
    Download current iShares IWV (Russell 3000 ETF) holdings from iShares website.

    NOTE: This uses TODAY'S constituents, introducing survivorship bias.

    Returns:
        DataFrame with ticker symbols and other holdings information.
    """
    print("Downloading IWV holdings from iShares...")
    print("  WARNING: Using current constituents (survivorship bias)")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(IWV_HOLDINGS_URL, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse CSV content - skip metadata rows at the top
        lines = response.text.split('\n')

        # Find the header row (contains 'Ticker' column)
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Ticker' in line:
                header_idx = i
                break

        # Read from the header row onwards
        csv_content = '\n'.join(lines[header_idx:])
        holdings = pd.read_csv(StringIO(csv_content))

        print(f"Downloaded {len(holdings)} holdings from IWV")
        return holdings

    except Exception as e:
        print(f"Error downloading IWV holdings: {e}")
        print("Attempting alternative approach...")
        return download_iwv_holdings_alternative()


def download_iwv_holdings_alternative() -> pd.DataFrame:
    """
    Alternative method to get Russell 3000 constituents.
    Falls back to Wikipedia S&P 500 + curated mid/small cap list.
    """
    print("Using alternative method to fetch Russell 3000 constituents...")

    # Get S&P 500 as a base (major large caps)
    try:
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_tables = pd.read_html(sp500_url)
        sp500_tickers = sp500_tables[0]['Symbol'].tolist()
    except Exception:
        sp500_tickers = []

    # Comprehensive list covering various sectors and market caps
    additional_tickers = [
        # Large Cap Technology
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE',
        'ORCL', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO', 'NOW', 'INTU', 'AMAT', 'MU',
        'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'FTNT', 'PANW', 'CRWD', 'ZS',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'MDT', 'ISRG', 'VRTX', 'REGN', 'BSX', 'EW', 'ZTS',
        'DXCM', 'IDXX', 'IQV', 'MTD', 'STE', 'HOLX', 'ALGN', 'TECH', 'BIO', 'PKI',
        # Financials
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB',
        'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'KEY', 'RF', 'CFG', 'HBAN',
        'MTB', 'ZION', 'CMA', 'WAL', 'EWBC', 'FHN', 'SNV', 'BOKF', 'CBSH', 'UMBF',
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG',
        'MAR', 'HLT', 'YUM', 'DPZ', 'ORLY', 'AZO', 'ROST', 'DG', 'DLTR', 'BBY',
        'ULTA', 'POOL', 'WSM', 'RH', 'FIVE', 'BOOT', 'PLNT', 'WING', 'SHAK', 'TXRH',
        # Consumer Staples
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
        'GIS', 'K', 'HSY', 'SJM', 'CAG', 'CPB', 'HRL', 'MKC', 'TSN', 'KHC',
        # Industrials
        'UPS', 'UNP', 'HON', 'CAT', 'RTX', 'BA', 'LMT', 'GE', 'MMM', 'DE',
        'FDX', 'CSX', 'NSC', 'WM', 'RSG', 'EMR', 'ETN', 'ITW', 'PH', 'ROK',
        'CMI', 'PCAR', 'FAST', 'ODFL', 'JBHT', 'XPO', 'CHRW', 'EXPD', 'SAIA', 'WERN',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'PXD', 'OXY',
        'DVN', 'HES', 'FANG', 'APA', 'HAL', 'BKR', 'OKE', 'WMB', 'KMI', 'ET',
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'DOW', 'DD', 'PPG', 'VMC',
        'MLM', 'NUE', 'STLD', 'CF', 'MOS', 'ALB', 'FMC', 'CE', 'EMN', 'SEE',
        # Utilities (typically low beta)
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
        'ES', 'PEG', 'AWK', 'AEE', 'CMS', 'DTE', 'FE', 'PPL', 'EVRG', 'ATO',
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'HST', 'REG', 'KIM', 'IRM',
        # Communication Services
        'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'TTWO',
        'PARA', 'FOX', 'FOXA', 'LYV', 'MTCH', 'PINS', 'SNAP', 'ROKU',
        # Mid/Small Cap Tech
        'BILL', 'HUBS', 'DDOG', 'NET', 'MDB', 'SNOW', 'PLTR', 'PATH', 'DOCN',
        'TEAM', 'WDAY', 'VEEV', 'PCTY', 'PAYC', 'PAYX', 'ADP', 'EPAM', 'GLOB',
        'GPN', 'FIS', 'FISV', 'PYPL', 'DOCU', 'BOX', 'TWLO', 'OKTA', 'ZM',
        # Mid/Small Cap Healthcare
        'BIIB', 'MRNA', 'ALNY', 'INCY', 'EXEL', 'BMRN', 'SRPT', 'IONS', 'NBIX',
        'SYK', 'ZBH', 'BAX', 'BDX', 'RMD', 'COO', 'TFX', 'PODD', 'LIVN',
        # Insurance
        'BRK-B', 'ALL', 'PGR', 'TRV', 'CB', 'MET', 'PRU', 'AFL', 'AIG', 'LNC',
        'HIG', 'GL', 'UNM', 'VOYA', 'AIZ', 'RGA', 'RNR',
        # Retailers
        'KSS', 'M', 'JWN', 'GPS', 'ANF', 'AEO', 'URBN', 'BURL', 'DKS', 'HIBB',
        # Autos and Transports
        'F', 'GM', 'RIVN', 'APTV', 'BWA', 'LEA', 'LKQ',
        'DAL', 'UAL', 'LUV', 'AAL', 'ALK', 'JBLU',
        'EXPE', 'ABNB', 'RCL', 'CCL', 'NCLH',
        # Construction and Homebuilders
        'LEN', 'DHI', 'PHM', 'NVR', 'TOL', 'KBH', 'TMHC',
        'BLD', 'BLDR', 'BECN', 'GMS',
        # Defense
        'NOC', 'GD', 'LHX', 'HII', 'TXT', 'TDG', 'HEI', 'AXON',
        'LDOS', 'SAIC', 'BAH',
        # Semiconductors
        'TSM', 'ASML', 'NXPI', 'MCHP', 'ON', 'SWKS', 'QRVO',
        'MPWR', 'CRUS', 'LSCC',
        # Additional diversified
        'GWW', 'HUBB', 'IDEX', 'IEX', 'ITT', 'NDSN', 'OTIS', 'PCAR',
        'RPM', 'SNA', 'SWK', 'TTC', 'WAB', 'XYL', 'ZBH', 'ZBRA',
        'MAN', 'RHI', 'HSIC', 'OMC', 'IPG',
        'MORN', 'MSCI', 'SPGI', 'MCO', 'INFO',
    ]

    # Combine and deduplicate
    all_tickers = list(set(sp500_tickers + additional_tickers))
    print(f"Compiled {len(all_tickers)} unique tickers")

    return pd.DataFrame({'Ticker': all_tickers})


def clean_tickers(holdings_df: pd.DataFrame) -> list:
    """
    Clean ticker symbols from holdings DataFrame.

    Args:
        holdings_df: DataFrame containing 'Ticker' column

    Returns:
        List of cleaned, valid ticker symbols
    """
    print("Cleaning ticker symbols...")

    # Get ticker column (try different possible column names)
    ticker_col = None
    for col in ['Ticker', 'ticker', 'Symbol', 'symbol', 'TICKER', 'SYMBOL']:
        if col in holdings_df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        raise ValueError("Could not find ticker column in holdings data")

    tickers = holdings_df[ticker_col].dropna().astype(str).tolist()

    # Clean tickers
    cleaned = []
    for ticker in tickers:
        # Skip empty or invalid
        if not ticker or ticker in ['nan', '-', 'N/A', 'NA', 'CASH', 'Cash']:
            continue

        # Remove whitespace
        ticker = ticker.strip().upper()

        # Skip ETFs, preferred stocks, warrants, etc.
        # Keep only standard stock tickers (1-5 letters, or with common suffixes)
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            # Allow tickers with dots like BRK.B or BF.B
            if re.match(r'^[A-Z]{1,4}\.[A-Z]$', ticker):
                # Convert to Yahoo format (BRK.B -> BRK-B)
                ticker = ticker.replace('.', '-')
            else:
                continue

        cleaned.append(ticker)

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    print(f"Cleaned tickers: {len(unique_tickers)} valid symbols")
    return unique_tickers


def download_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download monthly adjusted close prices for given tickers.

    Args:
        tickers: List of ticker symbols
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)

    Returns:
        DataFrame with monthly prices (tickers as columns, dates as index)
    """
    print(f"Downloading price data for {len(tickers)} tickers...")
    print(f"Date range: {start} to {end}")

    # Download in batches to avoid timeout issues
    batch_size = 100
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} ({len(batch)} tickers)...")

        try:
            # Download daily data first, then resample to monthly
            data = yf.download(
                batch,
                start=start,
                end=end,
                interval='1d',
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if not data.empty:
                # Get adjusted close prices
                if 'Close' in data.columns or (isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.get_level_values(0)):
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data['Close']
                    else:
                        # Single ticker case
                        prices = data[['Close']]
                        if len(batch) == 1:
                            prices.columns = batch

                    all_data.append(prices)

        except Exception as e:
            print(f"  Warning: Error downloading batch: {e}")
            continue

    if not all_data:
        raise ValueError("Failed to download any price data")

    # Combine all batches
    prices = pd.concat(all_data, axis=1)

    # Remove duplicate columns if any
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Resample to month-end
    monthly_prices = prices.resample('ME').last()

    print(f"Downloaded monthly prices: {monthly_prices.shape[0]} months, {monthly_prices.shape[1]} tickers")

    return monthly_prices


def download_benchmark_data(start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download IWV (Russell 3000 ETF) and risk-free rate (^IRX) data.

    Args:
        start: Start date string
        end: End date string

    Returns:
        Tuple of (IWV monthly prices, monthly risk-free rate)
    """
    print("Downloading benchmark (IWV) and risk-free rate (^IRX) data...")

    # Download IWV
    iwv_data = yf.download('IWV', start=start, end=end, interval='1d',
                           auto_adjust=True, progress=False)
    if iwv_data.empty:
        raise ValueError("Failed to download IWV data")

    iwv_monthly = iwv_data['Close'].resample('ME').last()
    iwv_monthly = pd.DataFrame(iwv_monthly)
    iwv_monthly.columns = ['IWV']

    # Download ^IRX (3-month T-Bill rate)
    irx_data = yf.download('^IRX', start=start, end=end, interval='1d',
                           auto_adjust=True, progress=False)

    if irx_data.empty:
        print("Warning: Could not download ^IRX, using alternative risk-free rate source...")
        # Fallback to ^TNX (10-year) or assume zero
        try:
            irx_data = yf.download('^TNX', start=start, end=end, interval='1d',
                                   auto_adjust=True, progress=False)
        except:
            pass

    if not irx_data.empty:
        # Resample to month-end
        rf_monthly = irx_data['Close'].resample('ME').last()
        # Convert from annual percentage to monthly decimal
        # IRX is quoted as annualized percentage (e.g., 5.0 means 5%)
        rf_monthly = (rf_monthly / 100) / 12  # Annual % -> monthly decimal
    else:
        print("Warning: Using zero risk-free rate as fallback")
        rf_monthly = pd.Series(0, index=iwv_monthly.index)

    rf_monthly = pd.DataFrame(rf_monthly)
    rf_monthly.columns = ['RF']

    print(f"IWV data: {len(iwv_monthly)} months")
    print(f"Risk-free rate data: {len(rf_monthly)} months")

    return iwv_monthly, rf_monthly


def download_fama_french_factors() -> pd.DataFrame:
    """
    Download Fama-French factor data from Ken French's data library.

    Downloads:
    - FF 5 Factors (Mkt-RF, SMB, HML, RMW, CMA)
    - Momentum factor (MOM)

    Returns:
        DataFrame with monthly factor returns
    """
    print("Downloading Fama-French factor data...")

    try:
        import zipfile
        from io import BytesIO

        # Download 5-factor data
        response = requests.get(FF_FACTORS_URL, timeout=30)
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # Find the CSV file
            csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
            with z.open(csv_name) as f:
                content = f.read().decode('utf-8')

        # Parse the CSV - find where monthly data starts and ends
        lines = content.split('\n')
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if line.strip().startswith('199') or line.strip().startswith('200') or line.strip().startswith('201') or line.strip().startswith('202'):
                if start_idx is None:
                    start_idx = i
                end_idx = i
            elif start_idx is not None and 'Annual' in line:
                break

        # Read just the monthly data
        if start_idx:
            # Get header (usually one line before data)
            header_line = lines[start_idx - 1] if start_idx > 0 else "Date,Mkt-RF,SMB,HML,RMW,CMA,RF"
            monthly_lines = [header_line] + lines[start_idx:end_idx+1]
            ff_data = pd.read_csv(StringIO('\n'.join(monthly_lines)))

            # Rename first column to Date if needed
            ff_data.columns = ['Date'] + list(ff_data.columns[1:])

            # Parse date
            ff_data['Date'] = pd.to_datetime(ff_data['Date'].astype(str), format='%Y%m')
            ff_data['Date'] = ff_data['Date'] + pd.offsets.MonthEnd(0)
            ff_data = ff_data.set_index('Date')

            # Convert from percentage to decimal
            for col in ff_data.columns:
                ff_data[col] = ff_data[col].astype(float) / 100
        else:
            raise ValueError("Could not parse FF data")

        # Try to download momentum factor
        try:
            response = requests.get(FF_MOMENTUM_URL, timeout=30)
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    content = f.read().decode('utf-8')

            lines = content.split('\n')
            start_idx = None
            end_idx = None

            for i, line in enumerate(lines):
                if line.strip().startswith('199') or line.strip().startswith('200') or line.strip().startswith('201') or line.strip().startswith('202'):
                    if start_idx is None:
                        start_idx = i
                    end_idx = i
                elif start_idx is not None and 'Annual' in line:
                    break

            if start_idx:
                header_line = "Date,Mom"
                monthly_lines = [header_line] + lines[start_idx:end_idx+1]
                mom_data = pd.read_csv(StringIO('\n'.join(monthly_lines)))
                mom_data.columns = ['Date', 'Mom']
                mom_data['Date'] = pd.to_datetime(mom_data['Date'].astype(str).str.strip(), format='%Y%m')
                mom_data['Date'] = mom_data['Date'] + pd.offsets.MonthEnd(0)
                mom_data = mom_data.set_index('Date')
                mom_data['Mom'] = mom_data['Mom'].astype(float) / 100

                # Merge with FF data
                ff_data = ff_data.join(mom_data, how='left')

        except Exception as e:
            print(f"  Warning: Could not download momentum factor: {e}")
            ff_data['Mom'] = np.nan

        print(f"Downloaded FF factors: {len(ff_data)} months")
        print(f"  Factors: {list(ff_data.columns)}")

        return ff_data

    except Exception as e:
        print(f"Error downloading FF factors: {e}")
        print("Creating placeholder FF factor data...")

        # Create placeholder data
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq='ME')
        ff_data = pd.DataFrame(index=dates)
        ff_data.index.name = 'Date'
        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']:
            ff_data[col] = np.nan

        return ff_data


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from price data.

    Args:
        prices: DataFrame of monthly prices

    Returns:
        DataFrame of monthly returns
    """
    returns = prices.pct_change()
    # First row will be NaN
    return returns


def compute_excess_returns(returns: pd.DataFrame, rf_rate: pd.DataFrame) -> pd.DataFrame:
    """
    Compute excess returns (return - risk-free rate).

    This follows the standard asset pricing convention where:
    Excess Return = Raw Return - Risk-Free Rate

    Args:
        returns: DataFrame of monthly returns
        rf_rate: DataFrame with 'RF' column of monthly risk-free rates

    Returns:
        DataFrame of excess returns
    """
    # Align indices
    common_idx = returns.index.intersection(rf_rate.index)

    excess = returns.loc[common_idx].subtract(rf_rate.loc[common_idx, 'RF'], axis=0)

    return excess


def compute_rolling_betas(excess_returns: pd.DataFrame,
                          iwv_excess_returns: pd.Series,
                          window: int = 60,
                          min_periods: int = 36) -> pd.DataFrame:
    """
    Compute rolling betas using 60-month rolling window.

    Following Frazzini and Pedersen (2014):
    Beta = Cov(R_i - R_f, R_m - R_f) / Var(R_m - R_f)

    Key methodological decisions:
    - Rolling window: 60 months (5 years) as in original paper
    - Minimum periods: 36 months required for valid estimate
    - No shrinkage applied (unlike some implementations)
    - Betas computed at month-end for use in next month's portfolio formation

    Args:
        excess_returns: DataFrame of stock excess returns
        iwv_excess_returns: Series of IWV (market) excess returns
        window: Rolling window size in months (default 60)
        min_periods: Minimum observations required for valid beta (default 36)

    Returns:
        DataFrame of rolling betas (same shape as excess_returns)
    """
    print(f"Computing rolling {window}-month betas (min {min_periods} months required)...")

    # Align data
    common_idx = excess_returns.index.intersection(iwv_excess_returns.index)
    stock_returns = excess_returns.loc[common_idx]
    market_returns = iwv_excess_returns.loc[common_idx]

    # Initialize beta DataFrame
    betas = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)

    # Compute rolling market variance
    market_var = market_returns.rolling(window=window, min_periods=min_periods).var()

    # Compute rolling covariance and beta for each stock
    for col in stock_returns.columns:
        stock_ret = stock_returns[col]
        # Rolling covariance
        cov = stock_ret.rolling(window=window, min_periods=min_periods).cov(market_returns)
        # Beta = Cov / Var
        betas[col] = cov / market_var

    # Replace infinities with NaN
    betas = betas.replace([np.inf, -np.inf], np.nan)

    # Winsorize extreme betas (optional, but common in practice)
    # Following academic practice, cap at reasonable bounds
    betas = betas.clip(lower=-2.0, upper=5.0)

    # Count valid betas
    valid_count = betas.notna().sum().sum()
    print(f"Computed {valid_count:,} valid beta values")

    # Report statistics
    mean_beta = betas.mean().mean()
    print(f"  Average beta across all stocks/months: {mean_beta:.3f}")

    return betas


def main():
    """Main function to execute the data loading pipeline."""
    print("=" * 70)
    print("BAB Strategy Data Loader")
    print("Replication of Frazzini and Pedersen (2014)")
    print("=" * 70)
    print(f"Start Date: {START_DATE}")
    print(f"End Date: {END_DATE}")
    print(f"Rolling Beta Window: {ROLLING_WINDOW} months")
    print(f"Minimum Periods for Beta: {MIN_PERIODS} months")
    print("=" * 70)
    print("\n*** SURVIVORSHIP BIAS WARNING ***")
    print("Using today's Russell 3000 constituents applied historically.")
    print("Results should be interpreted as indicative, not definitive.")
    print("=" * 70)

    # Create data directory
    ensure_data_dir()

    # Write survivorship bias warning
    write_survivorship_bias_warning()

    # Step 1: Download IWV holdings to get Russell 3000 tickers
    holdings = download_iwv_holdings()
    tickers = clean_tickers(holdings)

    # Save tickers list
    pd.DataFrame({'Ticker': tickers}).to_csv(TICKERS_FILE, index=False)
    print(f"Saved {len(tickers)} tickers to {TICKERS_FILE}")

    # Step 2: Download price data for all tickers
    prices = download_price_data(tickers, START_DATE, END_DATE)

    # Step 3: Download benchmark and risk-free rate data
    iwv_prices, rf_rate = download_benchmark_data(START_DATE, END_DATE)

    # Step 4: Download Fama-French factors
    ff_factors = download_fama_french_factors()

    # Step 5: Compute returns
    returns = compute_returns(prices)
    iwv_returns = compute_returns(iwv_prices)

    # Step 6: Compute excess returns
    excess_returns = compute_excess_returns(returns, rf_rate)
    iwv_excess = compute_excess_returns(iwv_returns, rf_rate)

    # Step 7: Compute rolling betas
    betas = compute_rolling_betas(
        excess_returns,
        iwv_excess['IWV'],
        window=ROLLING_WINDOW,
        min_periods=MIN_PERIODS
    )

    # Step 8: Save all outputs
    print("\nSaving data files...")

    prices.to_csv(PRICES_FILE)
    print(f"  Saved: {PRICES_FILE}")

    returns.to_csv(RETURNS_FILE)
    print(f"  Saved: {RETURNS_FILE}")

    excess_returns.to_csv(EXCESS_RETURNS_FILE)
    print(f"  Saved: {EXCESS_RETURNS_FILE}")

    betas.to_csv(BETAS_FILE)
    print(f"  Saved: {BETAS_FILE}")

    rf_rate.to_csv(RF_RATE_FILE)
    print(f"  Saved: {RF_RATE_FILE}")

    iwv_returns.to_csv(IWV_RETURNS_FILE)
    print(f"  Saved: {IWV_RETURNS_FILE}")

    iwv_excess.to_csv(IWV_EXCESS_RETURNS_FILE)
    print(f"  Saved: {IWV_EXCESS_RETURNS_FILE}")

    ff_factors.to_csv(FF_FACTORS_FILE)
    print(f"  Saved: {FF_FACTORS_FILE}")

    print("\n" + "=" * 70)
    print("Data loading complete!")
    print("=" * 70)

    # Summary statistics
    print("\nSummary:")
    print(f"  Total tickers: {len(tickers)}")
    print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
    print(f"  Monthly observations: {len(prices)}")
    print(f"  Tickers with valid betas: {(betas.notna().sum() > 0).sum()}")
    print(f"  Fama-French factors available: {list(ff_factors.columns)}")


if __name__ == "__main__":
    main()

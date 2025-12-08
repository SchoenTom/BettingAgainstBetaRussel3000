#!/usr/bin/env python3
"""
data_loader.py - Betting-Against-Beta (BAB) Strategy Data Loader

This script downloads and prepares all data required for the BAB strategy:
1. Downloads current Russell 3000 tickers from iShares IWV holdings CSV
2. Cleans invalid symbols and retains only standard stock tickers
3. Fetches monthly adjusted close prices for all tickers, IWV, and ^IRX
4. Computes and saves:
   - Monthly prices (wide DataFrame)
   - Simple monthly returns (pct_change)
   - Monthly excess returns (return - risk-free rate)
   - Rolling 60-month betas (no look-ahead bias)

Author: BAB Strategy Implementation
Date: 2024
"""

import os
import re
import warnings
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ROLLING_WINDOW = 60  # months for beta calculation
IWV_HOLDINGS_URL = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"

# Output file paths
PRICES_FILE = os.path.join(DATA_DIR, "monthly_prices.csv")
RETURNS_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")
EXCESS_RETURNS_FILE = os.path.join(DATA_DIR, "monthly_excess_returns.csv")
BETAS_FILE = os.path.join(DATA_DIR, "rolling_betas.csv")
RF_RATE_FILE = os.path.join(DATA_DIR, "risk_free_rate.csv")
IWV_RETURNS_FILE = os.path.join(DATA_DIR, "iwv_returns.csv")
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")


def ensure_data_dir() -> None:
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")


def download_iwv_holdings() -> pd.DataFrame:
    """
    Download current iShares IWV (Russell 3000 ETF) holdings from iShares website.

    Returns:
        DataFrame with ticker symbols and other holdings information.
    """
    print("Downloading IWV holdings from iShares...")

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
        from io import StringIO
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
    Alternative method to get Russell 3000 constituents using yfinance.
    Falls back to a curated list of major Russell 3000 stocks.
    """
    print("Using alternative method to fetch Russell 3000 constituents...")

    # Try to get holdings via yfinance
    try:
        iwv = yf.Ticker("IWV")
        # Some ETFs expose holdings - try multiple attributes
        if hasattr(iwv, 'holdings') and iwv.holdings is not None:
            holdings = iwv.holdings
            if 'Symbol' in holdings.columns:
                return holdings
    except Exception:
        pass

    # Fallback: Use a comprehensive list of Russell 3000 stocks
    # This list includes major components across market caps
    print("Fetching constituent list from publicly available sources...")

    # Get S&P 500 as a base (major large caps)
    try:
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_tables = pd.read_html(sp500_url)
        sp500_tickers = sp500_tables[0]['Symbol'].tolist()
    except Exception:
        sp500_tickers = []

    # Combine with additional Russell 3000 components
    # This is a representative sample of mid and small cap stocks
    additional_tickers = [
        # Technology
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
        'MTB', 'ZION', 'CMA', 'FRC', 'SIVB', 'WAL', 'PACW', 'EWBC', 'FHN', 'SNV',
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
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
        'ES', 'PEG', 'AWK', 'AEE', 'CMS', 'DTE', 'FE', 'PPL', 'EVRG', 'ATO',
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'PEAK', 'HST', 'REG', 'KIM',
        # Communication Services
        'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'ATVI', 'EA', 'TTWO',
        'WBD', 'PARA', 'FOX', 'FOXA', 'LYV', 'MTCH', 'ZG', 'PINS', 'SNAP', 'ROKU',
        # Small/Mid Caps - Various sectors
        'ETSY', 'CHWY', 'W', 'PTON', 'FVRR', 'UPWK', 'CVNA', 'OPEN', 'RDFN', 'ZI',
        'BILL', 'HUBS', 'DDOG', 'NET', 'MDB', 'SNOW', 'PLTR', 'PATH', 'DOCN', 'S',
        'CFLT', 'ESTC', 'GTLB', 'SUMO', 'NEWR', 'DT', 'FSLY', 'TWLO', 'OKTA', 'ZM',
        'RNG', 'FIVN', 'TEAM', 'WDAY', 'VEEV', 'COUP', 'PLAN', 'PCTY', 'PAYC', 'PAYX',
        'ADP', 'CDAY', 'EPAM', 'GLOB', 'EXLS', 'GPN', 'FIS', 'FISV', 'SQ', 'PYPL',
        # More Mid/Small Caps
        'SMAR', 'DOCU', 'BOX', 'ASAN', 'MNDY', 'AI', 'UPST', 'SOFI', 'AFRM', 'HOOD',
        'COIN', 'MARA', 'RIOT', 'HUT', 'BITF', 'SMCI', 'DELL', 'HPQ', 'HPE', 'NTAP',
        'STX', 'WDC', 'PSTG', 'CIEN', 'JNPR', 'AKAM', 'FFIV', 'ANET', 'KEYS', 'ZBRA',
        # Banks and Regional Banks
        'NYCB', 'OZK', 'BOKF', 'CBSH', 'UMBF', 'PNFP', 'IBKR', 'LPLA', 'RJF', 'SEIC',
        'TROW', 'BEN', 'IVZ', 'JHG', 'AMG', 'VRTS', 'APAM', 'WDR', 'CNS', 'EV',
        # Insurance
        'BRK-B', 'ALL', 'PGR', 'TRV', 'CB', 'MET', 'PRU', 'AFL', 'AIG', 'LNC',
        'HIG', 'L', 'GL', 'UNM', 'CNO', 'VOYA', 'AIZ', 'RGA', 'PRI', 'RNR',
        # REITs
        'IRM', 'CUBE', 'EXR', 'LSI', 'NSA', 'REXR', 'EGP', 'FR', 'PLD', 'STAG',
        'TRNO', 'COLD', 'IIPR', 'SBAC', 'UNIT', 'LUMN', 'FRT', 'BXP', 'SLG', 'VNO',
        # Biotech
        'BIIB', 'MRNA', 'BNTX', 'SGEN', 'ALNY', 'INCY', 'EXEL', 'BMRN', 'RARE', 'SRPT',
        'IONS', 'NBIX', 'PTCT', 'HALO', 'BLUE', 'SAGE', 'ARWR', 'FOLD', 'VCEL', 'XNCR',
        # Medical Devices
        'SYK', 'ZBH', 'BAX', 'BDX', 'RMD', 'COO', 'TFX', 'NUVA', 'GMED', 'IRTC',
        'SWAV', 'PODD', 'TNDM', 'LIVN', 'PEN', 'NVST', 'XRAY', 'HSIC', 'PDCO', 'OMI',
        # Retailers
        'KSS', 'M', 'JWN', 'GPS', 'ANF', 'AEO', 'URBN', 'EXPR', 'BURL', 'CATO',
        'DKS', 'HIBB', 'BGFV', 'GCO', 'SCVL', 'CAL', 'BOOT', 'TLYS', 'ZUMZ', 'PLCE',
        # Autos
        'F', 'GM', 'STLA', 'RIVN', 'LCID', 'FSR', 'NKLA', 'GOEV', 'RIDE', 'WKHS',
        'APTV', 'BWA', 'LEA', 'ALV', 'THRM', 'VC', 'DAN', 'MTOR', 'MOD', 'LKQ',
        # Restaurants and Leisure
        'DRI', 'CAKE', 'EAT', 'RRGB', 'BJRI', 'BLMN', 'CBRL', 'JACK', 'DENN', 'PLAY',
        'SIX', 'FUN', 'SEAS', 'HLT', 'WH', 'IHG', 'MGM', 'WYNN', 'LVS', 'CZR',
        # Airlines and Travel
        'DAL', 'UAL', 'LUV', 'AAL', 'ALK', 'JBLU', 'SAVE', 'HA', 'SKYW', 'MESA',
        'EXPE', 'TRIP', 'ABNB', 'RCL', 'CCL', 'NCLH', 'VAC', 'TNL', 'STAY', 'HGV',
        # Construction and Engineering
        'LEN', 'DHI', 'PHM', 'NVR', 'TOL', 'KBH', 'MDC', 'TMHC', 'MTH', 'MHO',
        'BLD', 'BLDR', 'BECN', 'GMS', 'SITE', 'FLR', 'J', 'MTZ', 'PWR', 'PRIM',
        # Defense and Aerospace
        'NOC', 'GD', 'LHX', 'HII', 'TXT', 'CW', 'TDG', 'HEI', 'AXON', 'OSK',
        'SPR', 'HWM', 'KTOS', 'MRCY', 'CACI', 'LDOS', 'SAIC', 'BAH', 'MANT', 'PSN',
        # Chemicals
        'CTVA', 'IFF', 'AVTR', 'GRA', 'HUN', 'OLN', 'CC', 'AXTA', 'ASH', 'CBT',
        'KWR', 'KRA', 'HWKN', 'IOSP', 'TROX', 'VNTR', 'NEU', 'ESI', 'BCPC', 'CMP',
        # Metals and Mining
        'CLF', 'X', 'AA', 'CENX', 'ATI', 'CMC', 'RS', 'WOR', 'ZEUS', 'KALU',
        'RYI', 'USAP', 'CRS', 'HAYN', 'MP', 'LAC', 'PLL', 'LTHM', 'SQM', 'ALB',
        # Paper and Packaging
        'IP', 'PKG', 'WRK', 'SLGN', 'CCK', 'BLL', 'SON', 'BERY', 'ATR', 'AMBP',
        'GEF', 'TRS', 'UFP', 'UFPI', 'ROCK', 'CARR', 'TT', 'JCI', 'LII', 'WSO',
        # Software
        'MSFT', 'ORCL', 'SAP', 'ADSK', 'ANSS', 'PTC', 'TYL', 'BSY', 'APPN', 'NCNO',
        'APPF', 'MANH', 'QTWO', 'ALTR', 'TENB', 'QLYS', 'VRNS', 'SAIL', 'CYBR', 'RPD',
        # Semiconductors
        'TSM', 'ASML', 'NXPI', 'MCHP', 'ON', 'SWKS', 'QRVO', 'WOLF', 'SLAB', 'DIOD',
        'MPWR', 'ALGM', 'RMBS', 'CRUS', 'SITM', 'POWI', 'AMBA', 'LSCC', 'FORM', 'ACLS',
        # Telecom
        'LUMN', 'ATUS', 'CABO', 'SHEN', 'GOGO', 'BAND', 'CCOI', 'EGHT', 'LOGI', 'OSIS',
        'VG', 'COMM', 'INFN', 'VIAV', 'LITE', 'IIVI', 'COHR', 'PI', 'CEVA', 'SYNA',
        # Internet and E-commerce
        'EBAY', 'MELI', 'SE', 'PDD', 'JD', 'BABA', 'BIDU', 'TCOM', 'VIPS', 'TME',
        'WB', 'IQ', 'BILI', 'YY', 'HUYA', 'DOYU', 'TAL', 'EDU', 'GOTU', 'DAO',
        # Additional Mid/Small Caps
        'CIEN', 'CALX', 'ADTN', 'INFN', 'CMBM', 'CRNT', 'CLFD', 'AUDC', 'NTWK', 'NTGR',
        'UI', 'UBNT', 'CASA', 'DGII', 'DIGI', 'SLAB', 'IDCC', 'INSG', 'ITRI', 'JAKK',
        # More diversified names
        'AMWD', 'AWI', 'CBZ', 'CCS', 'CHE', 'CLH', 'CRAI', 'CRL', 'CWT', 'ENS',
        'EPC', 'EXPO', 'FCN', 'FDS', 'FHI', 'FELE', 'GATX', 'GGG', 'GNTX', 'GTLS',
        'GWW', 'HE', 'HELE', 'HI', 'HIW', 'HNI', 'HSII', 'HUBB', 'ICFI', 'IDEX',
        'IEX', 'INGR', 'ITT', 'JLL', 'JOUT', 'KAI', 'KAR', 'KFY', 'KNL', 'KNX',
        'LANC', 'LEG', 'LECO', 'LFUS', 'LH', 'LHX', 'LIN', 'LKQ', 'LNT', 'LSTR',
        'MAN', 'MASI', 'MAS', 'MATX', 'MC', 'MGRC', 'MIDD', 'MKL', 'MKSI', 'MLI',
        'MMSI', 'MMS', 'MORN', 'MSA', 'MSM', 'MTDR', 'MTN', 'NDSN', 'NJR', 'NNN',
        'NPO', 'NVT', 'NXST', 'OGE', 'OGS', 'OHI', 'OI', 'OII', 'OLN', 'OMC',
        'ORI', 'OSK', 'OTIS', 'PCAR', 'PCH', 'PDM', 'PII', 'PINC', 'PNR', 'PNW',
        'POL', 'POST', 'PRGO', 'PRI', 'PRIM', 'PVH', 'R', 'RBC', 'REXR', 'RGA',
        'RHI', 'RLI', 'RNR', 'RPM', 'RRC', 'RS', 'SABR', 'SAIC', 'SAM', 'SBRA',
        'SCI', 'SEB', 'SF', 'SFM', 'SGMS', 'SHC', 'SIG', 'SKX', 'SLM', 'SMG',
        'SNA', 'SNV', 'SON', 'SRC', 'SSD', 'ST', 'STE', 'STL', 'SUI', 'SWK',
        'SWKS', 'SWN', 'SXT', 'SYF', 'SYY', 'TAP', 'TDC', 'TDY', 'TER', 'TEX',
        'TFX', 'THC', 'THO', 'THS', 'TKR', 'TNC', 'TNET', 'TPL', 'TPX', 'TR',
        'TRN', 'TRNO', 'TRU', 'TSCO', 'TTC', 'TUP', 'TW', 'TXT', 'UFPI', 'UIL',
        'UNM', 'UNP', 'UPS', 'URBN', 'USPH', 'UTHR', 'VFC', 'VIAV', 'VICR', 'VMI',
        'VNT', 'VOYA', 'VVV', 'WAB', 'WAT', 'WBS', 'WEN', 'WEX', 'WGO', 'WHR',
        'WLK', 'WRB', 'WRK', 'WSM', 'WSO', 'WST', 'WTFC', 'WWD', 'WWW', 'XPO',
        'XYL', 'Y', 'ZBH', 'ZBRA', 'ZEN', 'ZTO', 'ZWS',
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
                          window: int = 60) -> pd.DataFrame:
    """
    Compute rolling betas using 60-month rolling window.

    Beta = Cov(stock_excess, market_excess) / Var(market_excess)

    Uses historical data only (no look-ahead bias).

    Args:
        excess_returns: DataFrame of stock excess returns
        iwv_excess_returns: Series of IWV (market) excess returns
        window: Rolling window size in months

    Returns:
        DataFrame of rolling betas (same shape as excess_returns)
    """
    print(f"Computing rolling {window}-month betas...")

    # Align data
    common_idx = excess_returns.index.intersection(iwv_excess_returns.index)
    stock_returns = excess_returns.loc[common_idx]
    market_returns = iwv_excess_returns.loc[common_idx]

    # Initialize beta DataFrame
    betas = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)

    # Compute rolling market variance
    market_var = market_returns.rolling(window=window, min_periods=window).var()

    # Compute rolling covariance and beta for each stock
    for col in stock_returns.columns:
        stock_ret = stock_returns[col]
        # Rolling covariance
        cov = stock_ret.rolling(window=window, min_periods=window).cov(market_returns)
        # Beta = Cov / Var
        betas[col] = cov / market_var

    # Replace infinities with NaN
    betas = betas.replace([np.inf, -np.inf], np.nan)

    # Count valid betas
    valid_count = betas.notna().sum().sum()
    print(f"Computed {valid_count:,} valid beta values")

    return betas


def main():
    """Main function to execute the data loading pipeline."""
    print("=" * 60)
    print("BAB Strategy Data Loader")
    print("=" * 60)
    print(f"Start Date: {START_DATE}")
    print(f"End Date: {END_DATE}")
    print(f"Rolling Beta Window: {ROLLING_WINDOW} months")
    print("=" * 60)

    # Create data directory
    ensure_data_dir()

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

    # Step 4: Compute returns
    returns = compute_returns(prices)
    iwv_returns = compute_returns(iwv_prices)

    # Step 5: Compute excess returns
    excess_returns = compute_excess_returns(returns, rf_rate)
    iwv_excess = compute_excess_returns(iwv_returns, rf_rate)

    # Step 6: Compute rolling betas
    betas = compute_rolling_betas(excess_returns, iwv_excess['IWV'], window=ROLLING_WINDOW)

    # Step 7: Save all outputs
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

    print("\n" + "=" * 60)
    print("Data loading complete!")
    print("=" * 60)

    # Summary statistics
    print("\nSummary:")
    print(f"  Total tickers: {len(tickers)}")
    print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
    print(f"  Monthly observations: {len(prices)}")
    print(f"  Tickers with valid betas: {(betas.notna().sum() > 0).sum()}")


if __name__ == "__main__":
    main()

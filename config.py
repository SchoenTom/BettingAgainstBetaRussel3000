"""
config.py - Centralized configuration for BAB (Betting Against Beta) strategy

Russell 3000 implementation following Frazzini & Pedersen (2014) methodology.
"""

import os

# Directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

# Date Range (academic correctness: end before F&P 2014 publication)
START_DATE = '1995-01-01'  # Start of beta estimation
END_DATE = '2014-12-31'    # End of analysis (pre-publication)

# Beta Calculation (Frazzini-Pedersen methodology)
CORRELATION_WINDOW = 12    # Months for correlation
VOLATILITY_WINDOW = 60     # Months for volatility
MIN_PERIODS_CORR = 9       # Min periods for correlation
MIN_PERIODS_VOL = 36       # Min periods for volatility
SHRINKAGE_FACTOR = 0.6     # Beta shrinkage toward prior
PRIOR_BETA = 1.0           # Prior for shrinkage

# Portfolio Construction
NUM_DECILES = 10           # Deciles (like F&P)
MIN_STOCKS_PER_DECILE = 10 # Minimum stocks per group
WINSORIZE_PERCENTILE = 0.005  # 0.5% tails

# Data Quality
MIN_DATA_COVERAGE = 0.95   # Require 95% non-missing
MAX_GAP_MONTHS = 2         # Max consecutive missing months
REQUIRE_FULL_HISTORY = True

# Benchmark
BENCHMARK_TICKER = '^GSPC'  # S&P 500 for beta calculation

# Ken French Data Library URL for risk-free rate
KEN_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

# Download Configuration
DOWNLOAD_BATCH_SIZE = 50
PERIODS_PER_YEAR = 12

# Visualization
FIGURE_SIZE = (12, 6)
FIGURE_DPI = 150
COLORS = {
    'bab': '#1f77b4',
    'benchmark': '#ff7f0e',
    'low_beta': '#2ca02c',
    'high_beta': '#d62728',
    'spread': '#9467bd',
}

# Russell 3000 Curated Ticker List (~300 stocks with IPO before 1995)
# Includes Large, Mid, and Small cap across all sectors
RUSSELL_3000_TICKERS = [
    # Technology (Large Cap)
    'AAPL', 'MSFT', 'INTC', 'IBM', 'ORCL', 'TXN', 'HPQ', 'CSCO', 'AMD', 'MU',
    'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP', 'XLNX', 'NTAP', 'CA', 'ADBE', 'SNPS',
    'CDNS', 'ANSS', 'CTXS', 'FFIV', 'JNPR', 'AKAM', 'ADSK', 'INTU', 'PAYX',

    # Financials (Large Cap Banks)
    'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'STT', 'BK', 'KEY', 'FITB',
    'RF', 'HBAN', 'CFG', 'ZION', 'CMA', 'PBCT', 'FHN', 'MTB', 'NTRS', 'SCHW',

    # Financials (Investment Banks & Asset Managers)
    'GS', 'MS', 'AXP', 'BLK', 'BEN', 'TROW', 'IVZ', 'SEIC', 'AMG', 'LM',

    # Insurance
    'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'HIG', 'LNC', 'UNM', 'PFG',
    'L', 'CINF', 'CB', 'MMC', 'AON', 'AJG', 'WRB', 'GL', 'PGR', 'RE',

    # Healthcare (Pharma)
    'JNJ', 'PFE', 'MRK', 'ABT', 'LLY', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN',
    'VRTX', 'ALXN', 'ILMN', 'ISRG', 'IQV', 'ZTS', 'AGN', 'MYL', 'PRGO', 'ENDP',

    # Healthcare (Devices & Services)
    'MDT', 'BDX', 'BAX', 'SYK', 'ZBH', 'BSX', 'EW', 'HOLX', 'WAT', 'A',
    'TMO', 'DHR', 'MCK', 'ABC', 'CAH', 'HSIC', 'OMI', 'PDCO', 'HCA', 'UHS',

    # Consumer Staples (Food & Beverage)
    'KO', 'PEP', 'GIS', 'K', 'HSY', 'SJM', 'CPB', 'CAG', 'HRL', 'TSN',
    'MKC', 'KHC', 'MDLZ', 'MNST', 'STZ', 'TAP', 'BF.B', 'DEO', 'SAM', 'POST',

    # Consumer Staples (Household & Personal)
    'PG', 'CL', 'KMB', 'CHD', 'CLX', 'SPB', 'NWL', 'EL', 'COTY', 'IPAR',

    # Consumer Staples (Retail)
    'WMT', 'COST', 'KR', 'SYY', 'WBA', 'CVS', 'RAD', 'UNFI', 'SFM', 'USFD',

    # Consumer Discretionary (Retail)
    'HD', 'LOW', 'TGT', 'TJX', 'ROST', 'GPS', 'KSS', 'M', 'JWN', 'DDS',
    'JCP', 'BBBY', 'BBY', 'GME', 'ORLY', 'AZO', 'AAP', 'GPC', 'TSCO', 'WSM',

    # Consumer Discretionary (Apparel & Footwear)
    'NKE', 'VFC', 'PVH', 'RL', 'TPR', 'HBI', 'GIII', 'OXM', 'CRI', 'DECK',
    'SKX', 'WWW', 'FL', 'FINL', 'BOOT', 'GCO', 'CAL', 'SCVL', 'DSW', 'HIBB',

    # Consumer Discretionary (Autos)
    'F', 'GM', 'HOG', 'LEA', 'BWA', 'DLPH', 'VC', 'APTV', 'ALV', 'MGA',
    'TEN', 'THRM', 'AXL', 'DAN', 'MOD', 'LCII', 'SMP', 'MTOR', 'CPS', 'GNTX',

    # Consumer Discretionary (Homebuilding)
    'DHI', 'LEN', 'PHM', 'NVR', 'TOL', 'KBH', 'MDC', 'MHO', 'TMHC', 'MTH',

    # Consumer Discretionary (Hotels & Leisure)
    'MAR', 'HLT', 'H', 'CCL', 'RCL', 'NCLH', 'WYN', 'CHH', 'MGM', 'LVS',
    'WYNN', 'CZR', 'PENN', 'BYD', 'PLYA', 'VAC', 'HGV', 'TNL', 'STAY', 'RHP',

    # Consumer Discretionary (Media & Entertainment)
    'DIS', 'CMCSA', 'TWX', 'FOXA', 'CBS', 'VIAB', 'DISCA', 'SNI', 'NWSA', 'NYT',
    'GCI', 'TGNA', 'SSP', 'GTN', 'MNI', 'TRCO', 'MSGN', 'SIRI', 'LBRDA', 'LBTYA',

    # Industrials (Aerospace & Defense)
    'BA', 'LMT', 'NOC', 'GD', 'RTX', 'TXT', 'HII', 'LHX', 'TDG', 'HEI',
    'AXON', 'KTOS', 'MRCY', 'FLIR', 'CW', 'AJRD', 'DCO', 'MANT', 'LDOS', 'SAIC',

    # Industrials (Machinery)
    'CAT', 'DE', 'EMR', 'ETN', 'PH', 'ROK', 'ITW', 'DOV', 'IR', 'PNR',
    'XYL', 'AME', 'ROP', 'SWK', 'SNHY', 'TTC', 'GGG', 'FSS', 'GNRC', 'AWI',

    # Industrials (Conglomerates)
    'GE', 'MMM', 'HON', 'DHR', 'ITT', 'RHI', 'MAS', 'FLS', 'SPW', 'JCI',

    # Industrials (Transportation)
    'UNP', 'NSC', 'CSX', 'KSU', 'FDX', 'UPS', 'CHRW', 'EXPD', 'JBHT', 'XPO',
    'ODFL', 'SAIA', 'WERN', 'KNX', 'LSTR', 'HUBG', 'ARCB', 'HTLD', 'MRTN', 'GWW',

    # Industrials (Waste Management)
    'WM', 'RSG', 'WCN', 'CLH', 'ECOL', 'SRCL', 'HCCI', 'ADSW', 'CVA', 'GFL',

    # Energy (Integrated)
    'XOM', 'CVX', 'COP', 'OXY', 'MRO', 'DVN', 'EOG', 'PXD', 'APA', 'HES',
    'MUR', 'NFX', 'NBL', 'CLR', 'CPE', 'WLL', 'SWN', 'RRC', 'AR', 'EQT',

    # Energy (Refining & Marketing)
    'VLO', 'MPC', 'PSX', 'ANDV', 'HFC', 'PBF', 'DK', 'CVI', 'PARR', 'CLMT',

    # Energy (Equipment & Services)
    'SLB', 'HAL', 'BKR', 'NOV', 'FTI', 'HP', 'PTEN', 'NBR', 'RIG', 'DO',
    'ESV', 'NE', 'RDC', 'SDRL', 'ATW', 'OII', 'CLB', 'USAC', 'GPP', 'AROC',

    # Materials (Chemicals)
    'DOW', 'DD', 'LYB', 'PPG', 'SHW', 'APD', 'ECL', 'ALB', 'EMN', 'CE',
    'FMC', 'HUN', 'OLN', 'WLK', 'TROX', 'KOP', 'ASH', 'CBT', 'AXTA', 'RPM',

    # Materials (Metals & Mining)
    'NEM', 'FCX', 'NUE', 'STLD', 'CLF', 'X', 'AKS', 'ATI', 'CMC', 'RS',
    'AA', 'CENX', 'KGC', 'AUY', 'HL', 'CDE', 'EGO', 'AG', 'FSM', 'PAAS',

    # Materials (Packaging & Paper)
    'IP', 'PKG', 'WRK', 'AVY', 'SEE', 'BLL', 'CCK', 'OI', 'BERY', 'ATR',
    'SON', 'GEF', 'GPK', 'SLGN', 'UFPI', 'PPC', 'SWM', 'CLW', 'MERC', 'NTIC',

    # Utilities (Electric)
    'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'ED', 'XEL', 'WEC',
    'ES', 'DTE', 'AEE', 'CMS', 'PPL', 'FE', 'EIX', 'ETR', 'CNP', 'NI',
    'PNW', 'OGE', 'IDA', 'AVA', 'NWE', 'MDU', 'POR', 'BKH', 'MGEE', 'SJI',

    # Utilities (Gas)
    'NRG', 'VST', 'OKE', 'WMB', 'KMI', 'TRP', 'ENB', 'ATO', 'NJR', 'SWX',
    'SR', 'CPK', 'RGCO', 'SPKE', 'UTL', 'GAS', 'NWN', 'NFG', 'LNT', 'BIP',

    # Real Estate
    'SPG', 'PLD', 'EQIX', 'PSA', 'DLR', 'AVB', 'EQR', 'VTR', 'WELL', 'HCP',
    'O', 'NNN', 'STOR', 'EPR', 'OHI', 'LTC', 'SBRA', 'DOC', 'HR', 'CTRE',

    # Communication Services
    'T', 'VZ', 'TMUS', 'CTL', 'FTR', 'USM', 'SHEN', 'CNSL', 'LUMN', 'WIN',

    # Additional Mid-Cap Technology
    'CTSH', 'CRM', 'NOW', 'WDAY', 'SPLK', 'PANW', 'FTNT', 'ZS', 'CRWD', 'NET',

    # Additional Small-Cap Industrial
    'PATK', 'SXI', 'TRS', 'AIMC', 'TWIN', 'KBAL', 'NHC', 'AMWD', 'TILE', 'SCHL',

    # Additional Small-Cap Consumer
    'CATO', 'PIR', 'RCII', 'PLCE', 'ANF', 'AEO', 'URBN', 'EXPR', 'CHICO', 'SMRT',
]


def ensure_directories():
    """Create data and output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == '__main__':
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Number of Tickers: {len(RUSSELL_3000_TICKERS)}")
    print(f"Beta Methodology: F&P (Correlation {CORRELATION_WINDOW}m, Volatility {VOLATILITY_WINDOW}m)")
    print(f"Shrinkage: {SHRINKAGE_FACTOR} * beta + {1-SHRINKAGE_FACTOR} * {PRIOR_BETA}")

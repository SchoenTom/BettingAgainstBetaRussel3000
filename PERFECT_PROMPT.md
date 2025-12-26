# PERFEKTER PROMPT FÜR BAB-STRATEGIE NEUIMPLEMENTIERUNG

## Kontext
Implementiere die Betting-Against-Beta (BAB) Strategie nach Frazzini & Pedersen (2014) für den Russell 3000 Index. Die Implementierung soll akademisch korrekt sein und der Struktur des BettingAgainstBetaMSCIWRLD Repos entsprechen.

---

## PROMPT

```
Implementiere eine akademisch korrekte Betting-Against-Beta (BAB) Strategie für ca. 200 Russell 3000 Aktien mit folgenden exakten Spezifikationen:

## 1. DATEISTRUKTUR (8 Dateien, keine mehr)

```
/
├── config.py              # Zentrale Konfiguration
├── data_loader.py         # Daten-Download und -Aufbereitung
├── portfolio_construction.py  # Portfolio-Konstruktion
├── backtest.py            # Performance-Statistiken
├── illustrations.py       # Visualisierungen (PNG)
├── dashboard.py           # Streamlit Dashboard
├── main.py                # Pipeline-Orchestrator
└── requirements.txt       # Dependencies
```

## 2. CONFIG.PY

Zentrale Konstanten:
- START_DATE = '1995-01-01' (für Beta-Schätzung)
- END_DATE = '2014-12-31' (VOR F&P Publikation!)
- ROLLING_WINDOW = 60 (Monate für Volatilität)
- CORRELATION_WINDOW = 12 (Monate für Korrelation)
- SHRINKAGE_FACTOR = 0.6
- PRIOR_BETA = 1.0
- NUM_QUANTILES = 10 (Decile wie F&P)
- MIN_STOCKS_PER_QUANTILE = 10
- WINSORIZE_PERCENTILE = 0.005
- BENCHMARK_TICKER = '^GSPC'
- ~200 kuratierte Russell 3000 Ticker mit IPO vor 1995

## 3. BETA-BERECHNUNG (Frazzini-Pedersen Methodik)

```python
def compute_fp_beta(stock_returns, market_returns):
    # 1. Korrelation über 12 Monate
    rolling_corr = stock.rolling(12).corr(market)

    # 2. Volatilitäten über 60 Monate
    vol_stock = stock.rolling(60).std()
    vol_market = market.rolling(60).std()

    # 3. Time-Series Beta
    beta_ts = rolling_corr * (vol_stock / vol_market)

    # 4. Shrinkage toward 1
    beta_shrunk = 0.6 * beta_ts + 0.4 * 1.0

    return beta_shrunk
```

## 4. PORTFOLIO-KONSTRUKTION (DOLLAR-NEUTRAL!)

```python
def construct_bab(returns, betas, date):
    # Lagged betas (t-1) - NO LOOK-AHEAD
    betas_t1 = betas.shift(1).loc[date]

    # Sort into deciles
    deciles = pd.qcut(betas_t1, q=10, labels=range(1,11))

    # Low-beta (D1) and High-beta (D10)
    low_stocks = deciles[deciles == 1].index
    high_stocks = deciles[deciles == 10].index

    # Equal-weighted returns
    r_L = returns.loc[date, low_stocks].mean()
    r_H = returns.loc[date, high_stocks].mean()

    # Portfolio betas
    beta_L = betas_t1[low_stocks].mean()
    beta_H = betas_t1[high_stocks].mean()

    # DOLLAR-NEUTRAL BAB
    # Raw inverse-beta weights
    w_L_raw = 1.0 / beta_L
    w_H_raw = 1.0 / beta_H

    # Normalize to $2 gross ($1 long + $1 short)
    gross = w_L_raw + w_H_raw
    w_L = 2.0 * w_L_raw / gross
    w_H = 2.0 * w_H_raw / gross

    # Beta-neutral, dollar-neutral return
    bab_return = w_L * r_L - w_H * r_H

    return bab_return, beta_L, beta_H
```

## 5. RISK-FREE RATE

Ken French Data Library (1-Monats T-Bill):
```python
KEN_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
```

Fallback: Yahoo ^IRX konvertiert zu monatlich.

## 6. DATENQUALITÄT

- MIN_DATA_COVERAGE = 0.95 (95% der Datenpunkte)
- Keine Gaps > 2 Monate
- Daten müssen ab START_DATE vorhanden sein
- Winsorization bei 0.5% Tails

## 7. DASHBOARD (Streamlit)

Interaktive Visualisierungen:
- Cumulative Returns (BAB vs Benchmark)
- Rolling 12-Month Sharpe
- Drawdown Analysis
- Beta Spread über Zeit
- Decile Returns Bar Chart
- Yearly Returns Comparison
- Date Range Filter
- Key Metrics Cards

## 8. MAIN.PY PIPELINE

```python
def main():
    # Step 1: Data Loading
    run_data_loader()

    # Step 2: Portfolio Construction
    run_portfolio_construction()

    # Step 3: Backtesting
    run_backtest()

    # Step 4: Visualizations
    run_illustrations()
```

Mit CLI-Argumenten: --skip-download, --skip-plots, --only-backtest

## 9. ERWARTETE ERGEBNISSE

Mit korrekter Implementierung (Survivorship Bias bleibt):
- Annualized Return: ~1-5% (NICHT 40%!)
- Sharpe Ratio: ~0.1-0.5
- Max Drawdown: ~-50% bis -80%
- Ex-Ante Beta: ~0 (marktneutral)
- Net Dollar Position: ~$0 (dollar-neutral)

## 10. CODE-QUALITÄT

Minimalistisch:
- Docstrings max 5 Zeilen
- Keine redundanten Kommentare
- Jede Zeile muss notwendig sein
- Keine Disclaimer-Dateien (in README)
- Type Hints nur wo nötig

## 11. TICKER-LISTE (~200)

Kuratierte Russell 3000 Aktien mit:
- IPO vor 1995
- Kontinuierliche Daten
- Verschiedene Sektoren
- Mix aus Large/Mid/Small Cap
```

---

## QUALITÄTSKRITERIEN

Ein perfekter Code ist einer, bei dem man **nichts mehr weglassen kann**, nicht einer, bei dem man nichts mehr hinzufügen kann.

### ✅ JA:
- Jede Funktion hat einen klaren, einzigen Zweck
- Variablennamen sind selbsterklärend
- Keine magischen Zahlen (alles in config.py)
- Fehlerbehandlung nur wo notwendig

### ❌ NEIN:
- Keine 60-Zeilen Docstrings
- Keine redundanten Kommentare ("# compute returns" vor compute_returns())
- Keine Disclaimer-Dateien
- Keine doppelte Logik
- Keine "defensive" Validierung für unmögliche Fälle

---

## VALIDIERUNG

Nach Implementierung prüfen:

1. **Dollar-Neutralität**: `net_dollars.mean() ≈ 0`
2. **Beta-Neutralität**: `ex_ante_beta.mean() ≈ 0`
3. **Realistische Returns**: Annualized < 20%
4. **Keine Look-Ahead**: Nur Daten bis 2014
5. **Alle Tests laufen**: `python main.py`
6. **Dashboard startet**: `streamlit run dashboard.py`

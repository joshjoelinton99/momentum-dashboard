"""Futu OpenD connection and analysis settings."""

OPEND_HOST = "127.0.0.1"
OPEND_PORT = 11111

# Futu watchlist group that holds your screener output.
WATCHLIST_GROUP = "Screener"

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
CACHE_DIR = "data/cache"
HISTORY_DIR = "data/history"
DAILY_BREADTH_CSV = "data/history/daily_breadth.csv"
WEEKLY_SCREENER_CSV = "data/history/weekly_screener.csv"
# Legacy file kept for migration reference
LEGACY_WEEKLY_BREADTH_CSV = "data/history/weekly_breadth.csv"

# Weekly sector analysis (Futu)
WEEKLY_LOOKBACK_DAYS = 5
TOP_SECTORS_COUNT = 10
MIN_SECTOR_STOCKS = 2
TOP_CONCENTRATION_SECTORS = 3

# Momentum score (Sector Rankings tab — Futu screener only)
MOMENTUM_MEDIAN_WEIGHT = 0.6
MOMENTUM_MEAN_WEIGHT = 0.4
MOMENTUM_BREADTH_BASELINE = 0.50

# Screener health thresholds
SCREENER_BULLISH_MIN = 600
SCREENER_SELECTIVE_MIN = 400
SCREENER_HISTORY_WEEKS = 8

# Market breadth indices (Yahoo Finance — full constituent lists)
INDEX_DEFINITIONS = [
    {
        "name": "S&P 500",
        "ticker_source": "sp500",
        "history_prefix": "sandp_500",
        "yahoo_index": "^GSPC",
        "symbol_label": "SP500.I",
        "sma_prefix": "sp500",
    },
    {
        "name": "Nasdaq 100",
        "ticker_source": "ndx100",
        "history_prefix": "nasdaq_100",
        "yahoo_index": "^NDX",
        "symbol_label": "NDX.I",
        "sma_prefix": "ndx100",
    },
    {
        "name": "Russell 2000",
        "ticker_source": "russell2000",
        "history_prefix": "russell_2000",
        "yahoo_index": "^RUT",
        "symbol_label": "RUT.I",
        "sma_prefix": None,
    },
]

# Constituent list files (cached locally after first fetch)
SP500_CONSTITUENTS_CSV = "data/sp500_constituents.csv"
NDX100_CONSTITUENTS_JSON = "data/ndx100_constituents.json"
RUSSELL2000_CONSTITUENTS_JSON = "data/russell2000_constituents.json"

# Yahoo Finance settings
YAHOO_BENCHMARK = "^GSPC"
SMA_PERIODS = (5, 50, 200)
YAHOO_DOWNLOAD_CHUNK = 100
YAHOO_SMA_PERIOD = "1y"
BREADTH_BACKFILL_DAYS = 252          # ~1 year of trading days to backfill
DAILY_BREADTH_CHART_DAYS = 90        # default window on dashboard
AD_SMOOTH_DAYS = 5                   # MA window for % Advancing on dashboard

# Market volatility (dashboard — stop-loss / regime context)
VIX_TICKER = "^VIX"
VOL_LOOKBACK_DAYS = 400              # ~1y+ for ATR percentile and VIX SMA
RV_WINDOW = 20                       # realized vol window (trading days)
ATR_PERIOD = 14
VIX_SMA_DAYS = 50
VIX_CALM_MAX = 12                    # below: calm / tight stops more viable
VIX_NORMAL_MAX = 20                  # 12–20: typical equity-vol regime
VIX_ELEVATED_MAX = 30                # 20–30: elevated; >30 stress

# Strongest screener stocks (weekly report + dashboard tables)
STRONG_WEEKLY_MIN_PCT = 5.0
TOP_RS_COUNT = 30
DASHBOARD_STRONG_ROWS = 20
LATEST_WEEKLY_RISERS_CSV = "data/history/latest_weekly_risers.csv"
LATEST_RELATIVE_STRENGTH_CSV = "data/history/latest_relative_strength.csv"
LATEST_SECTOR_CONCENTRATION_CSV = "data/history/latest_sector_concentration.csv"

# Momentum pullback backtest + screener (Yahoo history).
# Strategy knobs live in momentum_pullback.PullbackParams, not here.
MOMENTUM_PULLBACK_HTML = "output/momentum_pullback.html"
MOMENTUM_PULLBACK_JSON = "output/momentum_pullback_summary.json"
MOMENTUM_PULLBACK_TRADES_CSV = "output/momentum_pullback_trades.csv"
MOMENTUM_PULLBACK_CANDIDATES_CSV = "output/momentum_pullback_candidates.csv"
MOMENTUM_LONG_HTML = "output/momentum_long.html"
MOMENTUM_LONG_JSON = "output/momentum_long_summary.json"
MOMENTUM_LONG_TRADES_CSV = "output/momentum_long_trades.csv"
MOMENTUM_LONG_CANDIDATES_CSV = "output/momentum_long_candidates.csv"

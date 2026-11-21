"""Load index constituent lists and fetch market data from Yahoo Finance."""

from __future__ import annotations

import json
import ssl
import subprocess
import time
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import certifi
import numpy as np
import pandas as pd
import yfinance as yf

import config

USER_AGENT = "Mozilla/5.0 (compatible; futu-stock-analysis/1.0)"


def _fetch_url(url: str) -> str:
    try:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=60, context=context) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        result = subprocess.run(
            ["curl", "-fsSL", "-A", USER_AGENT, url],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


def _normalize_ticker(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def futu_code_to_yahoo(code: str) -> str:
    return _normalize_ticker(code.replace("US.", "", 1))


def _load_cached_tickers(path: str) -> list[str] | None:
    cache = Path(path)
    if not cache.exists():
        return None
    data = json.loads(cache.read_text(encoding="utf-8"))
    tickers = data.get("tickers")
    return tickers if tickers else None


def _save_cached_tickers(path: str, tickers: list[str], source: str) -> None:
    cache = Path(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"source": source, "tickers": tickers, "count": len(tickers)}, indent=2),
        encoding="utf-8",
    )


def load_sp500_tickers() -> list[str]:
    csv_path = Path(config.SP500_CONSTITUENTS_CSV)
    if not csv_path.exists():
        raise RuntimeError(
            f"Missing {csv_path}. Run:\n"
            "  curl -fsSL https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv "
            f"-o {csv_path}"
        )
    tickers = [_normalize_ticker(s) for s in pd.read_csv(csv_path)["Symbol"].astype(str).tolist()]
    return tickers


def load_ndx100_tickers() -> list[str]:
    cached = _load_cached_tickers(config.NDX100_CONSTITUENTS_JSON)
    if cached:
        return cached

    html = _fetch_url("https://stockanalysis.com/list/nasdaq-100-stocks/")
    tables = pd.read_html(StringIO(html))
    table = tables[0]
    symbol_col = next(c for c in table.columns if str(c).lower() == "symbol")
    tickers = [_normalize_ticker(s) for s in table[symbol_col].astype(str).tolist()]
    tickers = [t for t in tickers if t and t != "NAN"]
    _save_cached_tickers(config.NDX100_CONSTITUENTS_JSON, tickers, "stockanalysis:nasdaq-100-stocks")
    return tickers


def load_russell2000_tickers() -> list[str]:
    cached = _load_cached_tickers(config.RUSSELL2000_CONSTITUENTS_JSON)
    if cached:
        return cached

    import re

    tickers: list[str] = []
    seen: set[str] = set()
    row = 1
    while row <= 2001:
        url = f"https://finviz.com/screener.ashx?v=111&f=idx_rut&r={row}"
        html = _fetch_url(url)
        page_tickers = re.findall(r'href="stock\?t=([A-Z0-9\.\-]+)', html)
        page_tickers = [_normalize_ticker(t) for t in page_tickers]
        page_tickers = [t for t in page_tickers if t and t != "NAN"]

        if not page_tickers:
            break

        new_count = 0
        for ticker in page_tickers:
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
                new_count += 1

        if new_count == 0:
            break
        row += 20
        time.sleep(0.3)

    if len(tickers) < 100:
        raise RuntimeError("Could not load enough Russell 2000 tickers from Finviz.")

    _save_cached_tickers(config.RUSSELL2000_CONSTITUENTS_JSON, tickers, "finviz:idx_rut")
    return tickers


def load_index_tickers(source: str) -> list[str]:
    loaders = {
        "sp500": load_sp500_tickers,
        "ndx100": load_ndx100_tickers,
        "russell2000": load_russell2000_tickers,
    }
    loader = loaders.get(source)
    if loader is None:
        raise ValueError(f"Unknown ticker source: {source}")
    return loader()


def fetch_benchmark_weekly_change(symbol: str | None = None) -> float | None:
    symbol = symbol or config.YAHOO_BENCHMARK
    hist = yf.download(symbol, period=f"{config.WEEKLY_LOOKBACK_DAYS + 2}d", progress=False, auto_adjust=True)
    closes = _extract_close_series(hist, symbol)
    if closes is None or len(closes) < 2:
        return None
    start_close = float(closes.iloc[0])
    end_close = float(closes.iloc[-1])
    if not start_close:
        return None
    return ((end_close - start_close) / start_close) * 100


def fetch_index_price_history(days: int | None = None) -> dict[str, pd.Series]:
    """Download Yahoo index closes keyed by INDEX_DEFINITIONS ticker_source."""
    days = days or max(config.DAILY_BREADTH_CHART_DAYS + 30, 120)
    period = _period_for_days(days)
    tickers = [idx["yahoo_index"] for idx in config.INDEX_DEFINITIONS if idx.get("yahoo_index")]
    if not tickers:
        return {}

    closes_by_ticker = _download_closes(tickers, period=period)
    by_source: dict[str, pd.Series] = {}
    for index in config.INDEX_DEFINITIONS:
        yahoo = index.get("yahoo_index")
        source = index["ticker_source"]
        if not yahoo:
            continue
        series = closes_by_ticker.get(yahoo)
        if series is None or series.empty:
            continue
        series = series.copy()
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        by_source[source] = series.sort_index()
    return by_source


def _extract_ohlc_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return Date-indexed High/Low/Close for one ticker from a yfinance download."""
    if data is None or data.empty:
        return pd.DataFrame()

    frame = data
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)
        if ticker in level0:
            frame = data[ticker]
        elif ticker in level1:
            frame = data.xs(ticker, axis=1, level=1)
        else:
            try:
                frame = data["Close"].to_frame("Close") if "Close" in level0 else data
            except Exception:
                return pd.DataFrame()

    cols = {str(c).title(): c for c in frame.columns}
    needed = {}
    for name in ("High", "Low", "Close"):
        src = cols.get(name)
        if src is None:
            continue
        needed[name] = frame[src]
    if "Close" not in needed:
        return pd.DataFrame()
    out = pd.DataFrame(needed)
    if "High" not in out.columns:
        out["High"] = out["Close"]
    if "Low" not in out.columns:
        out["Low"] = out["Close"]
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out.sort_index()


def fetch_volatility_history(days: int | None = None) -> pd.DataFrame:
    """VIX, 20-day realized vol, and SPX ATR% for dashboard regime / stop-loss context."""
    days = days or config.VOL_LOOKBACK_DAYS
    period = _period_for_days(days)
    vix_ticker = config.VIX_TICKER
    spx_ticker = config.YAHOO_BENCHMARK
    print(f"  Yahoo download: {vix_ticker} + {spx_ticker} ({period})")
    try:
        data = yf.download(
            [vix_ticker, spx_ticker],
            period=period,
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        return pd.DataFrame()

    vix_ohlc = _extract_ohlc_frame(data, vix_ticker).dropna(subset=["Close"])
    spx_ohlc = _extract_ohlc_frame(data, spx_ticker).dropna(subset=["Close", "High", "Low"])
    if vix_ohlc.empty or spx_ohlc.empty:
        return pd.DataFrame()

    vix = vix_ohlc["Close"].rename("vix")
    close = spx_ohlc["Close"]
    high = spx_ohlc["High"]
    low = spx_ohlc["Low"]
    prev = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / config.ATR_PERIOD, min_periods=config.ATR_PERIOD, adjust=False).mean()
    ratio = close / prev
    log_ret = np.log(ratio.where(ratio > 0))
    rv = log_ret.rolling(config.RV_WINDOW, min_periods=max(10, config.RV_WINDOW // 2)).std() * np.sqrt(252) * 100

    out = pd.DataFrame(
        {
            "vix": vix,
            "vix_sma": vix.rolling(config.VIX_SMA_DAYS, min_periods=config.VIX_SMA_DAYS).mean(),
            "rv20": rv,
            "atr_pct": atr / close * 100,
            "spx_close": close,
        }
    )
    out.index.name = "report_date"
    return out.dropna(how="all")


def _extract_close_series(data: pd.DataFrame, ticker: str | None = None) -> pd.Series | None:
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if ticker and ticker in data.columns.get_level_values(0):
            closes = data[ticker]["Close"].dropna()
            return closes if not closes.empty else None
        # Single ticker downloads sometimes still use MultiIndex
        try:
            closes = data["Close"].dropna()
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            return closes if not closes.empty else None
        except Exception:
            return None
    closes = data["Close"].dropna()
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    return closes if not closes.empty else None


def _download_closes(tickers: list[str], period: str) -> dict[str, pd.Series]:
    closes_by_ticker: dict[str, pd.Series] = {}
    chunk_size = config.YAHOO_DOWNLOAD_CHUNK

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"  Yahoo download: {min(i + chunk_size, len(tickers))}/{len(tickers)} ({period})")
        try:
            data = yf.download(
                chunk,
                period=period,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            continue

        if len(chunk) == 1:
            series = _extract_close_series(data, chunk[0])
            if series is not None:
                closes_by_ticker[chunk[0]] = series
        else:
            for ticker in chunk:
                series = _extract_close_series(data, ticker)
                if series is not None:
                    closes_by_ticker[ticker] = series

        time.sleep(0.2)

    return closes_by_ticker


def compute_daily_changes(tickers: list[str]) -> pd.Series:
    closes_by_ticker = _download_closes(tickers, period="5d")
    changes: list[float] = []

    for closes in closes_by_ticker.values():
        if len(closes) < 2:
            continue
        prev_close = float(closes.iloc[-2])
        last_close = float(closes.iloc[-1])
        if prev_close:
            changes.append(((last_close - prev_close) / prev_close) * 100)

    return pd.Series(changes)


def fetch_weekly_changes_for_futu_codes(futu_codes: list[str]) -> dict[str, float | None]:
    """Weekly return % keyed by Futu code (e.g. US.AAPL), via Yahoo Finance."""
    lookback = config.WEEKLY_LOOKBACK_DAYS
    need_bars = lookback + 1
    yahoo_by_code = {code: futu_code_to_yahoo(code) for code in futu_codes}
    yahoo_tickers = sorted(set(yahoo_by_code.values()))
    closes_by_ticker = _download_closes(yahoo_tickers, period=f"{lookback + 10}d")

    weekly: dict[str, float | None] = {}
    for code, yahoo_ticker in yahoo_by_code.items():
        closes = closes_by_ticker.get(yahoo_ticker)
        if closes is None or len(closes) < need_bars:
            weekly[code] = None
            continue
        start_close = float(closes.iloc[-need_bars])
        end_close = float(closes.iloc[-1])
        weekly[code] = ((end_close - start_close) / start_close) * 100 if start_close else None
    return weekly


def compute_ad_stats(tickers: list[str]) -> tuple[int, int, int, int]:
    changes = compute_daily_changes(tickers)
    if changes.empty:
        return 0, 0, 0, 0
    advancers = int((changes > 0).sum())
    decliners = int((changes < 0).sum())
    unchanged = int((changes == 0).sum())
    return advancers, decliners, unchanged, len(changes)


def compute_sma_participation(tickers: list[str]) -> tuple[float | None, float | None, float | None, int, int]:
    """Return (% above 5, % above 50, % above 200, analyzed, skipped)."""
    closes_by_ticker = _download_closes(tickers, period=config.YAHOO_SMA_PERIOD)
    sma5, sma50, sma200 = config.SMA_PERIODS
    above_5 = above_50 = above_200 = analyzed = 0
    skipped = len(tickers) - len(closes_by_ticker)

    for closes in closes_by_ticker.values():
        if len(closes) < sma200:
            skipped += 1
            continue
        last_close = float(closes.iloc[-1])
        analyzed += 1
        if last_close > float(closes.tail(sma5).mean()):
            above_5 += 1
        if last_close > float(closes.tail(sma50).mean()):
            above_50 += 1
        if last_close > float(closes.tail(sma200).mean()):
            above_200 += 1

    pct_5 = (above_5 / analyzed * 100) if analyzed else None
    pct_50 = (above_50 / analyzed * 100) if analyzed else None
    pct_200 = (above_200 / analyzed * 100) if analyzed else None
    return pct_5, pct_50, pct_200, analyzed, skipped


def _cache_path(ticker_source: str) -> Path:
    return Path(config.CACHE_DIR) / f"{ticker_source}_closes.parquet"


def _period_for_days(days: int) -> str:
    if days <= 5:
        return "5d"
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    return "2y"


def build_close_matrix(tickers: list[str], period: str) -> pd.DataFrame:
    """Return a date-indexed DataFrame of adjusted close prices (columns = tickers)."""
    closes_by_ticker = _download_closes(tickers, period=period)
    if not closes_by_ticker:
        return pd.DataFrame()

    matrix = pd.DataFrame(closes_by_ticker)
    matrix.index = pd.to_datetime(matrix.index).normalize()
    matrix = matrix.sort_index()
    return matrix


def load_or_update_close_matrix(tickers: list[str], ticker_source: str, days: int = 252) -> pd.DataFrame:
    """Load cached close matrix or download/update from Yahoo Finance."""
    cache = _cache_path(ticker_source)
    cache.parent.mkdir(parents=True, exist_ok=True)
    period = _period_for_days(days + max(config.SMA_PERIODS))

    existing = pd.DataFrame()
    if cache.exists():
        existing = pd.read_parquet(cache)
        existing.index = pd.to_datetime(existing.index).normalize()

    if existing.empty:
        print(f"  No cache for {ticker_source}; downloading {period} history...")
        matrix = build_close_matrix(tickers, period=period)
        if not matrix.empty:
            matrix.to_parquet(cache)
        return matrix

    # Incremental update: fetch recent bars and merge (~1mo covers skipped weeks)
    print(f"  Refreshing {ticker_source} cache ({len(tickers)} tickers)...")
    recent = build_close_matrix(tickers, period="1mo")
    if recent.empty:
        return existing

    merged = existing.combine_first(recent)
    for ticker in tickers:
        if ticker not in merged.columns and ticker in recent.columns:
            merged[ticker] = recent[ticker]
    merged = merged.sort_index()
    merged.to_parquet(cache)
    return merged


def compute_ad_history_from_matrix(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily advance/decline stats from a close price matrix."""
    empty_cols = [
        "report_date",
        "advancers",
        "decliners",
        "unchanged",
        "analyzed",
        "ad_ratio",
        "advance_pct",
    ]
    if closes.empty or len(closes) < 2:
        return pd.DataFrame(columns=empty_cols)

    daily_returns = closes.pct_change()
    rows = []
    for date, row in daily_returns.iloc[1:].iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        advancers = int((valid > 0).sum())
        decliners = int((valid < 0).sum())
        unchanged = int((valid == 0).sum())
        analyzed = len(valid)
        rows.append(
            {
                "report_date": pd.Timestamp(date).normalize(),
                "advancers": advancers,
                "decliners": decliners,
                "unchanged": unchanged,
                "analyzed": analyzed,
                "ad_ratio": round(advancers / max(decliners, 1), 4),
                "advance_pct": round(advancers / analyzed * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_sma_participation_history(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily % of stocks above 5/50/200 SMA from a close price matrix."""
    empty_cols = [
        "report_date",
        "pct_above_5sma",
        "pct_above_50sma",
        "pct_above_200sma",
        "analyzed",
    ]
    if closes.empty:
        return pd.DataFrame(columns=empty_cols)

    sma5_n, sma50_n, sma200_n = config.SMA_PERIODS
    sma5 = closes.rolling(sma5_n, min_periods=sma5_n).mean()
    sma50 = closes.rolling(sma50_n, min_periods=sma50_n).mean()
    sma200 = closes.rolling(sma200_n, min_periods=sma200_n).mean()

    above5 = closes.gt(sma5)
    above50 = closes.gt(sma50)
    above200 = closes.gt(sma200)
    valid200 = closes.notna() & sma200.notna()

    pct5 = above5.where(valid200).mean(axis=1, skipna=True) * 100
    pct50 = above50.where(valid200).mean(axis=1, skipna=True) * 100
    pct200 = above200.where(valid200).mean(axis=1, skipna=True) * 100
    analyzed = valid200.sum(axis=1)

    hist = pd.DataFrame(
        {
            "report_date": pd.to_datetime(closes.index).normalize(),
            "pct_above_5sma": pct5.round(2),
            "pct_above_50sma": pct50.round(2),
            "pct_above_200sma": pct200.round(2),
            "analyzed": analyzed.astype(int),
        }
    )
    return hist[hist["analyzed"] > 0].reset_index(drop=True)


def compute_new_high_low_history(closes: pd.DataFrame, lookback: int = 252, ma: int = 10) -> pd.DataFrame:
    """Daily 52-week new highs minus new lows, plus a short moving average.

    A new high is a close at or above the prior `lookback` sessions' high
    (today excluded). Same idea for new lows. This matches the MarketInOut
    New Highs − New Lows definition (52-week highs vs lows, MA of the net).
    """
    empty_cols = ["report_date", "new_highs", "new_lows", "nhl", "nhl_ma"]
    if closes.empty or len(closes) <= lookback:
        return pd.DataFrame(columns=empty_cols)

    prior_high = closes.rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low = closes.rolling(lookback, min_periods=lookback).min().shift(1)
    valid = closes.notna() & prior_high.notna() & prior_low.notna()
    new_highs = ((closes >= prior_high) & valid).sum(axis=1)
    new_lows = ((closes <= prior_low) & valid).sum(axis=1)
    nhl = (new_highs - new_lows).astype(float)
    hist = pd.DataFrame(
        {
            "report_date": pd.to_datetime(closes.index).normalize(),
            "new_highs": new_highs.astype(int),
            "new_lows": new_lows.astype(int),
            "nhl": nhl,
            "nhl_ma": nhl.rolling(ma, min_periods=ma).mean().round(2),
        }
    )
    return hist[hist["new_highs"].notna()].reset_index(drop=True)

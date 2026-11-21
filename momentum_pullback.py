#!/usr/bin/env python3
"""Martin Luk-style momentum pullback backtest and screener.

Yahoo Finance supplies the history. The project's Futu OpenD connection is
left untouched so it can still back live features later.

Rules, and where they live:
  Universe  load_universe / evaluate_bar "universe"
            US names from the S&P 500 and Nasdaq 100 (all above the $100M cap).
            Each signal day also needs 20-day dollar volume and 20-day ADR%.
  Trend     evaluate_bar "trend"
            Long:  9 EMA > 21 EMA > 50 EMA, and the EMA you buy is rising.
            Short: 9 EMA < 21 EMA < 50 EMA, and the EMA you short is falling.
  Setup     evaluate_bar "setup"
            Long:  close within ~1× ADR above that rising 9 or 21 EMA.
            Short: close within ~1× ADR below that falling 9 or 21 EMA.
  Entry     _simulate
            Next session's open, if it has not chased out of the zone or
            gapped through the stop.
  Stop      stop_price
            Long:  under the recent low or the trigger EMA.
            Short: above the recent high or the trigger EMA.
            Either way, hard-capped at 0.5× ADR%.
  Risk      apply_equity
            0.40% of equity per trade (the middle of the 0.3–0.5% band).
  Exit      _walk_exit
            Long:  close below the EMA that triggered the entry.
            Short: close above the EMA that triggered the entry.
            A stop hit intraday comes first. Scored in R.

Change the rules in PullbackParams below, or pass the flags listed in --help.
Then run:

  python3 momentum_pullback.py
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

import config
from yahoo_data import load_ndx100_tickers, load_sp500_tickers

REASON_LABEL = {
    "stop": "Stop",
    "gap_stop": "Gap through stop",
    "ema_trail": "Close below exit EMA",
    "max_hold": "Max hold",
}

R_BIN_LABELS = ("< -1R", "-1 to 0R", "0 to 1R", "1 to 2R", "2 to 3R", "> 3R")


@dataclass(frozen=True)
class PullbackParams:
    """Every knob for the rules above. Defaults match the brief."""

    # Universe: market cap, liquidity, and range.
    min_market_cap: float = 100_000_000.0
    min_dollar_volume: float = 50_000_000.0
    dollar_vol_window: int = 20
    min_adr_pct: float = 5.0
    adr_window: int = 20

    # Trend. "short" mirrors every directional test: falling stack, rally
    # into the EMA, stop above, trail on a close back through the EMA.
    side: str = "short"
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    ema_rise_lookback: int = 5
    # Long: close must be above the slow EMA. Short: close must be below it.
    require_close_above_slow: bool = True

    # Setup: price near a rising 9 or 21 EMA, within ~0.5–1× ADR.
    # proximity_adr_mult is the band above the EMA. 1.0 is the wide end of
    # that range; set 0.5 for a tighter pullback.
    proximity_adr_mult: float = 1.0
    undercut_adr_mult: float = 0.25
    setup_on_fast: bool = True
    setup_on_mid: bool = True

    # Entry approximation. Signal on the close, fill at the next open.
    # Skip the fill if the open has run more than this many ADR above the EMA.
    max_chase_adr: float = 1.5

    # Stop: under the recent low or the trigger EMA, hard-capped at 0.5× ADR%.
    stop_lookback: int = 5
    stop_adr_cap: float = 0.5
    stop_buffer: float = 0.0
    # False = tighten a wide structural stop to the cap (the "hard cap").
    # True  = skip the trade when structure is already wider than the cap.
    skip_if_wider_than_cap: bool = False

    # Risk: 0.3–0.5% of equity. 0.40% is the middle of that band.
    risk_pct: float = 0.004
    account_equity: float = 100_000.0
    # No leverage. A very tight stop risks less than risk_pct instead of
    # borrowing to reach it. R-multiples themselves are unchanged.
    max_position_pct: float = 1.0

    # Exit: trail the EMA that triggered the trade. Set 9 or 21 to force that
    # trail. A long 21-EMA pullback is usually already under the 9, and a
    # short 21-EMA rally is usually already above the 9, so a forced 9-EMA
    # trail would exit those trades on the entry day.
    exit_ema: int | None = None
    max_hold_days: int = 0  # 0 disables the time stop; only stop and EMA trail

    # Data. "sp500", "ndx100", or "sp500+ndx100".
    # period is used when start is empty. start is the first entry date
    # (YYYY-MM-DD); earlier bars are downloaded only to warm up the EMAs.
    period: str = "5y"
    start: str | None = None
    universe: str = "sp500+ndx100"

    # Optional calendar block. Empty trades every month.
    skip_months: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise ValueError("side must be long or short.")
        if self.exit_ema is not None and self.exit_ema not in (self.ema_fast, self.ema_mid):
            raise ValueError(
                f"exit_ema must be {self.ema_fast}, {self.ema_mid}, or None "
                "to trail whichever EMA triggered the entry"
            )
        if not (self.setup_on_fast or self.setup_on_mid):
            raise ValueError("Turn on at least one of the 9 EMA or 21 EMA setups.")
        if not 0 < self.risk_pct < 0.05:
            raise ValueError("risk_pct must be a fraction such as 0.004 for 0.4%.")
        if self.min_adr_pct <= 0 or self.stop_adr_cap <= 0:
            raise ValueError("ADR threshold and stop cap must be positive.")
        if self.proximity_adr_mult <= 0:
            raise ValueError("proximity_adr_mult must be positive.")
        if self.start:
            pd.Timestamp(self.start)
        if any(month < 1 or month > 12 for month in self.skip_months):
            raise ValueError("skip_months must be calendar months, 1 through 12.")


_MONTH_NAMES = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _skip_label(p: PullbackParams) -> str:
    names = [_MONTH_NAMES[month] for month in sorted(set(p.skip_months))]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _month_blocked(when, p: PullbackParams) -> bool:
    """True when a new entry on this date falls in a stand-down month."""
    if not p.skip_months:
        return False
    return pd.Timestamp(when).month in set(p.skip_months)


def _is_short(p: PullbackParams) -> bool:
    return p.side == "short"


def _book_name(p: PullbackParams) -> str:
    return "Short momentum rally" if _is_short(p) else "Long momentum rally"


def _page_href(side: str) -> str:
    return "long.html" if side == "long" else "short.html"


def _section_id(side: str) -> str:
    # The short anchor stays momentum-pullback so older dashboard links still land.
    return "momentum-long" if side == "long" else "momentum-pullback"


def _zone_word(p: PullbackParams) -> str:
    return "rally zone" if _is_short(p) else "pullback zone"


def _risk_per_share(entry: float, stop: float, p: PullbackParams) -> float:
    return (stop - entry) if _is_short(p) else (entry - stop)


def _pnl_per_share(entry: float, exit_px: float, p: PullbackParams) -> float:
    return (entry - exit_px) if _is_short(p) else (exit_px - entry)


def _ema_choices(p: PullbackParams) -> str:
    parts: list[str] = []
    if p.setup_on_fast:
        parts.append(str(p.ema_fast))
    if p.setup_on_mid:
        parts.append(str(p.ema_mid))
    if len(parts) == 2:
        return f"{parts[0]} or {parts[1]}"
    return parts[0]


def rules_text(p: PullbackParams, universe_label: str) -> list[tuple[str, str]]:
    """Plain-language copy of the rules, filled in with the active parameters."""
    if p.skip_if_wider_than_cap:
        cap_line = (
            f"If that distance is wider than {p.stop_adr_cap:g}× ADR%, the trade is skipped."
        )
    else:
        cap_line = (
            f"If that distance is wider than {p.stop_adr_cap:g}× ADR%, the stop is tightened "
            "to the cap so the risk stays inside the rule."
        )
    if p.exit_ema is None:
        trail = (
            f"the {p.ema_fast} EMA when that was the trigger, and the {p.ema_mid} EMA "
            "when that was the trigger"
        )
    else:
        trail = f"the {p.exit_ema} EMA on every trade"
    if _is_short(p):
        exit_line = (
            f"Cover when the close finishes above {trail}. "
            "This test fills that exit at the close. A high through the stop exits at the stop "
            "first; an open that gaps through the stop exits at the open."
        )
        r_line = " R = (entry − exit) / (stop − entry)."
    else:
        exit_line = (
            f"Trail out when the close finishes below {trail}. "
            "This test fills that exit at the close. A low through the stop exits at the stop "
            "first; an open that gaps through the stop exits at the open."
        )
        r_line = " R = (exit − entry) / (entry − stop)."
    if p.max_hold_days:
        exit_line += f" Anything still open after {p.max_hold_days} sessions is closed at that close."
    exit_line += r_line
    if _is_short(p):
        above = (
            f" The close also has to be below the {p.ema_slow} EMA."
            if p.require_close_above_slow
            else ""
        )
        trend_line = (
            f"{p.ema_fast} EMA below {p.ema_mid} EMA below {p.ema_slow} EMA. "
            f"The EMA you short against has to be lower than it was {p.ema_rise_lookback} sessions ago."
            f"{above}"
        )
        setup_line = (
            f"The close is within {p.proximity_adr_mult:g}× ADR below the falling {_ema_choices(p)} EMA, "
            f"or not more than {p.undercut_adr_mult:g}× ADR above it. "
            "When both qualify, the closer one is the trigger and anchors the stop."
        )
        entry_line = (
            "The signal is the close. The test shorts the next session's open. "
            f"If that open is more than {p.max_chase_adr:g}× ADR below the trigger EMA, it has already "
            "extended and is skipped. If the open is already through the stop, there is no fill and it is not a loss."
        )
        stop_line = (
            f"Resistance is the higher of the {p.stop_lookback}-session high and the trigger EMA. "
            f"The stop sits at that level{'' if not p.stop_buffer else f' plus a {p.stop_buffer:.2%} buffer'}. "
            f"{cap_line}"
        )
    else:
        above = (
            f" The close also has to be above the {p.ema_slow} EMA."
            if p.require_close_above_slow
            else ""
        )
        trend_line = (
            f"{p.ema_fast} EMA above {p.ema_mid} EMA above {p.ema_slow} EMA. "
            f"The EMA you buy against has to be higher than it was {p.ema_rise_lookback} sessions ago."
            f"{above}"
        )
        setup_line = (
            f"The close is within {p.proximity_adr_mult:g}× ADR above the rising {_ema_choices(p)} EMA, "
            f"or not more than {p.undercut_adr_mult:g}× ADR below it. "
            "When both qualify, the closer one is the trigger and anchors the stop."
        )
        entry_line = (
            "The signal is the close. The test buys the next session's open. "
            f"If that open is more than {p.max_chase_adr:g}× ADR above the trigger EMA, it is a chase and "
            "is skipped. If the open is already through the stop, there is no fill and it is not a loss."
        )
        stop_line = (
            f"Support is the lower of the {p.stop_lookback}-session low and the trigger EMA. "
            f"The stop sits at that level{'' if not p.stop_buffer else f' minus a {p.stop_buffer:.2%} buffer'}. "
            f"{cap_line}"
        )
    return [
        (
            "Universe",
            f"Current {universe_label} names. Index membership is the historical stand-in "
            f"for market cap above ${p.min_market_cap / 1e6:.0f}M; the live screen still drops a name "
            f"when Yahoo reports a cap under that. On the signal day, {p.dollar_vol_window}-day average "
            f"dollar volume must exceed ${p.min_dollar_volume / 1e6:.0f}M and {p.adr_window}-day ADR% "
            f"must exceed {p.min_adr_pct:g}%. Delisted names are absent from a current constituent list.",
        ),
        ("Trend", trend_line),
        ("Setup", setup_line),
        ("Entry", entry_line),
        (
            "Stand down",
            (
                f"No new entries in {_skip_label(p)}."
                if p.skip_months
                else "Every month is traded."
            ),
        ),
        ("Stop", stop_line),
        (
            "Risk",
            f"{p.risk_pct * 100:.2f}% of current equity per trade. Shares = risk dollars / stop distance. "
            f"Notional is capped at {p.max_position_pct * 100:.0f}% of equity, so a very tight stop "
            "risks less instead of using leverage. Reported R is the price result either way.",
        ),
        ("Exit", exit_line),
    ]


def stop_price(
    entry: float,
    structure: float,
    adr_pct: float,
    p: PullbackParams,
) -> tuple[float, bool] | None:
    """Stop beyond `structure`, then enforce the 0.5× ADR% hard cap.

    Long: structure is the lower of the recent low and the trigger EMA, and
    the stop sits under the entry. Short: structure is the higher of the
    recent high and the trigger EMA, and the stop sits above the entry.
    Returns (stop, capped). None means the structure is on the wrong side of
    the entry, or the trade is skipped because structure is wider than the cap.
    """
    if (
        entry <= 0
        or not np.isfinite(entry)
        or not np.isfinite(structure)
        or not np.isfinite(adr_pct)
        or adr_pct <= 0
    ):
        return None
    if _is_short(p):
        beyond = structure * (1.0 + p.stop_buffer)
        if beyond <= entry:
            return None
        natural_frac = (beyond - entry) / entry
    else:
        beyond = structure * (1.0 - p.stop_buffer)
        if beyond >= entry:
            return None
        natural_frac = (entry - beyond) / entry
    cap_frac = p.stop_adr_cap * (adr_pct / 100.0)
    if cap_frac <= 0:
        return None
    if natural_frac > cap_frac and p.skip_if_wider_than_cap:
        return None
    capped = natural_frac > cap_frac
    stop_frac = min(natural_frac, cap_frac)
    if _is_short(p):
        return entry * (1.0 + stop_frac), capped
    return entry * (1.0 - stop_frac), capped


def _norm(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def load_name_meta() -> dict[str, dict[str, str]]:
    """S&P 500 names and sectors. Nasdaq-only names simply show the ticker."""
    path = Path(config.SP500_CONSTITUENTS_CSV)
    meta: dict[str, dict[str, str]] = {}
    if not path.exists():
        return meta
    frame = pd.read_csv(path)
    for rec in frame.to_dict("records"):
        meta[_norm(rec["Symbol"])] = {
            "name": str(rec.get("Security") or ""),
            "sector": str(rec.get("GICS Sector") or ""),
        }
    return meta


def load_universe(
    p: PullbackParams,
    tickers: list[str] | None = None,
    tickers_file: str | None = None,
    limit: int | None = None,
) -> tuple[list[str], str, bool]:
    """Return (tickers, label, is_partial).

    Partial runs (an explicit list or --limit) are labeled as samples so the
    page does not pretend they are the full index universe.
    """
    partial = False
    if tickers:
        chosen = [_norm(t) for t in tickers if str(t).strip()]
        label = "Custom list"
        partial = True
    elif tickers_file:
        lines = Path(tickers_file).read_text(encoding="utf-8").splitlines()
        chosen = []
        for line in lines:
            raw = line.split("#", 1)[0].strip()
            if raw:
                chosen.append(_norm(raw))
        label = f"List file {tickers_file}"
        partial = True
    else:
        chosen = []
        parts: list[str] = []
        source = p.universe
        if "sp500" in source:
            chosen.extend(load_sp500_tickers())
            parts.append("S&P 500")
        if "ndx100" in source:
            chosen.extend(load_ndx100_tickers())
            parts.append("Nasdaq 100")
        if not parts:
            raise SystemExit(f"Unknown universe {source!r}. Use sp500, ndx100, or sp500+ndx100.")
        label = " + ".join(parts)
    seen: set[str] = set()
    unique: list[str] = []
    for ticker in chosen:
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    if limit is not None and limit > 0:
        unique = unique[:limit]
        label += f", first {limit}"
        partial = True
    if not unique:
        raise SystemExit("The universe is empty.")
    return unique, label, partial


def _naive_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _frame_for_ticker(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one symbol's OHLCV out of a yfinance download."""
    if data is None or data.empty:
        return None
    frame: pd.DataFrame | None
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        level1 = set(data.columns.get_level_values(1))
        if ticker in level0:
            frame = data[ticker]
        elif ticker in level1:
            frame = data.xs(ticker, axis=1, level=1)
        else:
            return None
    else:
        frame = data
    if frame is None or frame.empty:
        return None
    lookup = {str(col).lower(): col for col in frame.columns}

    def series(*names: str) -> pd.Series | None:
        for name in names:
            src = lookup.get(name)
            if src is not None:
                return pd.to_numeric(frame[src], errors="coerce")
        return None

    close = series("close", "adj close")
    if close is None:
        return None
    out = pd.DataFrame(
        {
            "open": series("open"),
            "high": series("high"),
            "low": series("low"),
            "close": close,
            "volume": series("volume"),
        }
    )
    if out["open"] is None:
        out["open"] = out["close"]
    if out["high"] is None:
        out["high"] = out["close"]
    if out["low"] is None:
        out["low"] = out["close"]
    if out["volume"] is None:
        out["volume"] = np.nan
    out.index = _naive_index(frame.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["close"])
    return out if not out.empty else None


def history_label(p: PullbackParams) -> str:
    """How the page should describe the window that was tested."""
    window = f"entries from {p.start}" if p.start else f"{p.period} of history"
    if p.skip_months:
        return f"{window}, no new entries in {_skip_label(p)}"
    return window


def _cache_id(p: PullbackParams) -> str:
    return f"start_{p.start}" if p.start else p.period


def _download_start(p: PullbackParams) -> str | None:
    """First bar to request. A year before `start` so the 50 EMA is valid on day one."""
    if not p.start:
        return None
    return (pd.Timestamp(p.start) - pd.Timedelta(days=400)).date().isoformat()


def _cache_paths(cache_id: str) -> tuple[Path, Path]:
    path = Path(config.CACHE_DIR) / f"momentum_pullback_ohlcv_{cache_id}.parquet"
    return path, path.with_suffix(".meta.json")


def _read_cache(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    prices: dict[str, pd.DataFrame] = {}
    for ticker, group in frame.groupby("ticker", sort=False):
        part = group.set_index("date").sort_index()
        part = part[~part.index.duplicated(keep="last")]
        prices[str(ticker)] = part[["open", "high", "low", "close", "volume"]].astype(float)
    return prices


def _write_cache(
    path: Path,
    meta_path: Path,
    frames: dict[str, pd.DataFrame],
    period: str,
    unavailable: list[str] | None = None,
) -> None:
    merged = _read_cache(path) if path.exists() else {}
    merged.update(frames)
    rows = []
    latest = None
    for ticker, frame in merged.items():
        part = frame.copy()
        part["ticker"] = ticker
        part["date"] = part.index
        rows.append(part.reset_index(drop=True))
        last = pd.Timestamp(frame.index.max())
        latest = last if latest is None or last > latest else latest
    if not rows or latest is None:
        return
    failed: set[str] = set()
    if meta_path.exists():
        try:
            failed.update(json.loads(meta_path.read_text(encoding="utf-8")).get("unavailable") or [])
        except (OSError, json.JSONDecodeError):
            failed = set()
    failed.update(unavailable or [])
    failed -= set(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "period": period,
                "latest": pd.Timestamp(latest).date().isoformat(),
                "tickers": sorted(merged),
                "count": len(merged),
                "unavailable": sorted(failed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _cache_is_fresh(meta: dict, period: str) -> bool:
    if meta.get("period") != period or not meta.get("latest"):
        return False
    latest = pd.Timestamp(meta["latest"])
    # US cash equities. A weekday's bar is "due" only after the US close,
    # which is 21:00 UTC during daylight time and 22:00 UTC in winter.
    now_utc = pd.Timestamp.now(tz="UTC")
    today = now_utc.tz_localize(None).normalize()
    due = today
    if now_utc.hour < 21:
        due = today - pd.Timedelta(days=1)
    while due.weekday() >= 5:
        due -= pd.Timedelta(days=1)
    return latest.normalize() >= due


def _download_chunk(chunk: list[str], period: str | None, start: str | None) -> pd.DataFrame | None:
    last_error: Exception | None = None
    for threads in (True, False):
        try:
            kwargs = dict(
                group_by="ticker",
                auto_adjust=True,
                threads=threads,
                progress=False,
            )
            if start:
                kwargs["start"] = start
            else:
                kwargs["period"] = period or "5y"
            data = yf.download(chunk, **kwargs)
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        print(f"  chunk failed ({chunk[0]}…{chunk[-1]}): {last_error}")
    return None


def _download(tickers: list[str], period: str | None, start: str | None) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    chunk_size = min(60, config.YAHOO_DOWNLOAD_CHUNK)
    label = f"from {start}" if start else (period or "5y")
    for offset in range(0, len(tickers), chunk_size):
        chunk = tickers[offset : offset + chunk_size]
        print(f"  Yahoo download {min(offset + chunk_size, len(tickers))}/{len(tickers)} ({label})")
        data = _download_chunk(chunk, period, start)
        if data is None:
            continue
        for ticker in chunk:
            frame = _frame_for_ticker(data, ticker)
            if frame is not None and len(frame) >= 30:
                frames[ticker] = frame
    return frames


def load_prices(
    tickers: list[str],
    p: PullbackParams,
    *,
    refresh: bool,
    write_cache: bool,
) -> dict[str, pd.DataFrame]:
    """OHLCV by ticker. A fresh cache is reused; missing names are filled in."""
    period = _cache_id(p)
    download_start = _download_start(p)
    path, meta_path = _cache_paths(period)
    cached: dict[str, pd.DataFrame] = {}
    meta: dict = {}
    if path.exists() and meta_path.exists() and not refresh:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        if _cache_is_fresh(meta, period):
            cached = _read_cache(path)
    # Names Yahoo already failed are skipped until --refresh, so a tweak-and-rerun
    # does not re-request delisted or bad symbols every time.
    known_missing = set() if refresh else set(meta.get("unavailable") or [])
    missing = [ticker for ticker in tickers if ticker not in cached and ticker not in known_missing]
    if cached and not missing:
        print(f"Using cached Yahoo prices through {meta.get('latest')} ({len(cached)} symbols).")
        return {ticker: cached[ticker] for ticker in tickers if ticker in cached}
    if cached and len(missing) <= 40:
        print(f"Cache is fresh. Downloading {len(missing)} symbols it does not have yet.")
        downloaded = _download(missing, None if download_start else p.period, download_start)
        cached.update(downloaded)
        if write_cache:
            failed = [ticker for ticker in missing if ticker not in downloaded]
            _write_cache(path, meta_path, cached, period, unavailable=failed)
        return {ticker: cached[ticker] for ticker in tickers if ticker in cached}

    window = f"from {download_start}" if download_start else p.period
    print(f"Downloading {len(tickers)} symbols from Yahoo Finance ({window}).")
    frames = _download(tickers, None if download_start else p.period, download_start)
    if not frames:
        raise SystemExit("Yahoo returned no prices. Check the network and run again.")
    if write_cache:
        failed = [ticker for ticker in tickers if ticker not in frames]
        _write_cache(path, meta_path, frames, period, unavailable=failed)
    elif cached:
        frames = {**cached, **frames}
    got = sum(1 for ticker in tickers if ticker in frames)
    print(f"  prices for {got}/{len(tickers)} symbols")
    return {ticker: frames[ticker] for ticker in tickers if ticker in frames}


def _ema(series: pd.Series, span: int) -> pd.Series:
    # span=N is the usual trading EMA, alpha = 2/(N+1). It only looks backward.
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def add_indicators(df: pd.DataFrame, p: PullbackParams) -> pd.DataFrame:
    """Causal indicators. Nothing here reads a future bar."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.sort_index().copy()
    out = out[~out.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[out["close"] > 0]
    if out.empty:
        return out

    # ADR% is the mean of (high−low)/close. ADR in dollars is the mean of
    # (high−low). The zone uses dollars ("within 1× ADR"); the stop cap uses
    # the percent ("0.5 × ADR%").
    range_pct = (out["high"] - out["low"]) / out["close"] * 100.0
    out["adr_pct"] = range_pct.rolling(p.adr_window, min_periods=p.adr_window).mean()
    out["adr_dollar"] = (out["high"] - out["low"]).rolling(p.adr_window, min_periods=p.adr_window).mean()
    # 20-day average dollar volume. Adjusted close × reported volume.
    out["dollar_vol"] = (out["close"] * out["volume"]).rolling(
        p.dollar_vol_window, min_periods=p.dollar_vol_window
    ).mean()
    out["ema_fast"] = _ema(out["close"], p.ema_fast)
    out["ema_mid"] = _ema(out["close"], p.ema_mid)
    out["ema_slow"] = _ema(out["close"], p.ema_slow)
    # Slope uses only bars already closed. Flat EMAs qualify for neither side.
    shifted = p.ema_rise_lookback
    out["fast_rising"] = (out["ema_fast"] > out["ema_fast"].shift(shifted)).fillna(False)
    out["mid_rising"] = (out["ema_mid"] > out["ema_mid"].shift(shifted)).fillna(False)
    out["fast_falling"] = (out["ema_fast"] < out["ema_fast"].shift(shifted)).fillna(False)
    out["mid_falling"] = (out["ema_mid"] < out["ema_mid"].shift(shifted)).fillna(False)
    # Recent extreme includes the signal day, which is known before the next open.
    out["recent_low"] = out["low"].rolling(p.stop_lookback, min_periods=p.stop_lookback).min()
    out["recent_high"] = out["high"].rolling(p.stop_lookback, min_periods=p.stop_lookback).max()
    return out


def _in_zone(dist_adr: float, p: PullbackParams) -> bool:
    """True when price is inside the band around an EMA.

    dist_adr is (close − ema) / ADR. Positive means the close is above the EMA.
    Long keeps a wide band above the EMA and a small undercut below it.
    Short mirrors that: a wide band below the EMA and a small poke above it.
    """
    if _is_short(p):
        return -p.proximity_adr_mult <= dist_adr <= p.undercut_adr_mult
    return -p.undercut_adr_mult <= dist_adr <= p.proximity_adr_mult


def _arrays(df: pd.DataFrame) -> dict:
    return {
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
        "ema_fast": df["ema_fast"].to_numpy(dtype=float),
        "ema_mid": df["ema_mid"].to_numpy(dtype=float),
        "ema_slow": df["ema_slow"].to_numpy(dtype=float),
        "adr_pct": df["adr_pct"].to_numpy(dtype=float),
        "adr_dollar": df["adr_dollar"].to_numpy(dtype=float),
        "dollar_vol": df["dollar_vol"].to_numpy(dtype=float),
        "recent_low": df["recent_low"].to_numpy(dtype=float),
        "recent_high": df["recent_high"].to_numpy(dtype=float),
        "fast_rising": df["fast_rising"].fillna(False).to_numpy(dtype=bool),
        "mid_rising": df["mid_rising"].fillna(False).to_numpy(dtype=bool),
        "fast_falling": df["fast_falling"].fillna(False).to_numpy(dtype=bool),
        "mid_falling": df["mid_falling"].fillna(False).to_numpy(dtype=bool),
        "dates": df.index,
    }


def _val(x) -> float | None:
    try:
        number = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def evaluate_bar(i: int, a: dict, p: PullbackParams) -> dict:
    """Classify one session. Status is how far it got through the rules.

    warmup    indicators are not ready
    universe  fails the dollar-volume or ADR test (market cap is applied later)
    trend     liquidity passed, the EMA stack or the rising test did not
    extended  trend is valid, price is outside the pullback zone
    setup     in the zone; the caller still has to price the stop and the fill
    """
    close = _val(a["close"][i])
    high = _val(a["high"][i])
    low = _val(a["low"][i])
    ema_fast = _val(a["ema_fast"][i])
    ema_mid = _val(a["ema_mid"][i])
    ema_slow = _val(a["ema_slow"][i])
    adr_pct = _val(a["adr_pct"][i])
    adr_dollar = _val(a["adr_dollar"][i])
    dollar_vol = _val(a["dollar_vol"][i])
    recent_low = _val(a["recent_low"][i])
    recent_high = _val(a["recent_high"][i])

    def distance(ema: float | None) -> tuple[float | None, float | None]:
        if close is None or ema is None or not adr_dollar or close == 0:
            return None, None
        return (close - ema) / adr_dollar, (close - ema) / close * 100.0

    dist_fast, pct_fast = distance(ema_fast)
    dist_mid, pct_mid = distance(ema_mid)
    row = {
        "status": "warmup",
        "close": close,
        "ema_fast": ema_fast,
        "ema_mid": ema_mid,
        "ema_slow": ema_slow,
        "adr_pct": adr_pct,
        "adr_dollar": adr_dollar,
        "dollar_vol": dollar_vol,
        "recent_low": recent_low,
        "recent_high": recent_high,
        "dist_fast_adr": dist_fast,
        "dist_mid_adr": dist_mid,
        "dist_fast_pct": pct_fast,
        "dist_mid_pct": pct_mid,
        "trigger": None,
        "trigger_ema": None,
        "trigger_dist_adr": None,
        "nearest_ema": None,
        "nearest_dist_adr": None,
    }
    needed = (close, high, low, ema_fast, ema_mid, ema_slow, adr_pct, adr_dollar, dollar_vol, recent_low, recent_high)
    if any(v is None for v in needed) or high < low:
        return row
    # Universe rule, aside from market cap (that needs a separate quote).
    if adr_pct < p.min_adr_pct or dollar_vol < p.min_dollar_volume:
        row["status"] = "universe"
        return row

    if _is_short(p):
        fast_slope = bool(a["fast_falling"][i])
        mid_slope = bool(a["mid_falling"][i])
        stack = ema_fast < ema_mid < ema_slow
        side_ok = close < ema_slow if p.require_close_above_slow else True
    else:
        fast_slope = bool(a["fast_rising"][i])
        mid_slope = bool(a["mid_rising"][i])
        stack = ema_fast > ema_mid > ema_slow
        side_ok = close > ema_slow if p.require_close_above_slow else True
    slope_ok = (p.setup_on_fast and fast_slope) or (p.setup_on_mid and mid_slope)
    nearest = []
    if p.setup_on_fast and fast_slope and dist_fast is not None:
        nearest.append((abs(dist_fast), str(p.ema_fast), dist_fast, ema_fast))
    if p.setup_on_mid and mid_slope and dist_mid is not None:
        nearest.append((abs(dist_mid), str(p.ema_mid), dist_mid, ema_mid))
    if nearest:
        nearest.sort(key=lambda item: item[0])
        row["nearest_ema"] = nearest[0][1]
        row["nearest_dist_adr"] = nearest[0][2]
    if not (stack and side_ok and slope_ok):
        row["status"] = "trend"
        return row

    # Setup rule: only an EMA sloping with the trend, and only inside the band.
    hits = [item for item in nearest if _in_zone(item[2], p)]
    if not hits:
        row["status"] = "extended"
        return row
    hits.sort(key=lambda item: item[0])
    _, label, dist_adr, ema = hits[0]
    row["status"] = "setup"
    row["trigger"] = label
    row["trigger_ema"] = ema
    row["trigger_dist_adr"] = dist_adr
    return row


def _date_str(dates, i: int) -> str:
    return pd.Timestamp(dates[i]).date().isoformat()


def _exit_series_key(trigger: str, p: PullbackParams) -> str:
    """Which EMA the trail watches. None follows the EMA that triggered the buy."""
    forced = p.exit_ema if p.exit_ema is not None else int(trigger)
    return "ema_fast" if forced == p.ema_fast else "ema_mid"


def _walk_exit(a: dict, entry_i: int, stop: float, p: PullbackParams, exit_key: str):
    """Walk forward from the fill.

    Stop first (it is live during the session). Then the EMA trail, which
    uses the close. The time stop is only a backtest lid and only fires when
    the EMA has not broken.
    """
    opens = a["open"]
    highs = a["high"]
    lows = a["low"]
    closes = a["close"]
    exit_ema = a[exit_key]
    n = len(closes)
    short = _is_short(p)
    for j in range(entry_i, n):
        if short:
            if j > entry_i and np.isfinite(opens[j]) and opens[j] >= stop:
                return j, float(opens[j]), "gap_stop"
            if np.isfinite(highs[j]) and highs[j] >= stop:
                return j, float(stop), "stop"
            level = exit_ema[j]
            if np.isfinite(level) and np.isfinite(closes[j]) and closes[j] > level:
                return j, float(closes[j]), "ema_trail"
        else:
            if j > entry_i and np.isfinite(opens[j]) and opens[j] <= stop:
                return j, float(opens[j]), "gap_stop"
            if np.isfinite(lows[j]) and lows[j] <= stop:
                return j, float(stop), "stop"
            level = exit_ema[j]
            if np.isfinite(level) and np.isfinite(closes[j]) and closes[j] < level:
                return j, float(closes[j]), "ema_trail"
        if p.max_hold_days and (j - entry_i + 1) >= p.max_hold_days:
            return j, float(closes[j]), "max_hold"
    return None


def _simulate(a: dict, ticker: str, p: PullbackParams) -> tuple[list[dict], list[dict]]:
    """One symbol, one position at a time.

    Entry rule: the signal is bar i's close; the fill is bar i+1's open.
    The next signal is allowed only after the exit, so a name cannot pyramid.
    """
    n = len(a["close"])
    closed: list[dict] = []
    still_open: list[dict] = []
    i = 0
    while i < n - 1:
        ev = evaluate_bar(i, a, p)
        if ev["status"] != "setup":
            i += 1
            continue
        entry_i = i + 1
        if _month_blocked(a["dates"][entry_i], p):
            i += 1
            continue
        entry = _val(a["open"][entry_i])
        if entry is None or entry <= 0:
            i += 1
            continue
        # Chase filter: the open left the zone. Not a loss, just no trade.
        chase_band = p.max_chase_adr * ev["adr_dollar"]
        if _is_short(p):
            chased = entry < ev["trigger_ema"] - chase_band
        else:
            chased = entry > ev["trigger_ema"] + chase_band
        if chased:
            i += 1
            continue
        if _is_short(p):
            structure = max(ev["recent_high"], ev["trigger_ema"])
        else:
            structure = min(ev["recent_low"], ev["trigger_ema"])
        priced = stop_price(entry, structure, ev["adr_pct"], p)
        if priced is None:
            # Open is already through the stop, or structure is on the wrong side.
            i += 1
            continue
        stop, capped = priced
        risk_per_share = _risk_per_share(entry, stop, p)
        if risk_per_share <= 0:
            # Open is already through the stop. No fill, and it is not a loss.
            i += 1
            continue
        walked = _walk_exit(a, entry_i, stop, p, _exit_series_key(ev["trigger"], p))
        if walked is None:
            last = n - 1
            last_close = float(a["close"][last])
            still_open.append(
                {
                    "ticker": ticker,
                    "signal_date": _date_str(a["dates"], i),
                    "entry_date": _date_str(a["dates"], entry_i),
                    "entry": entry,
                    "stop": stop,
                    "last_close": last_close,
                    "current_r": _pnl_per_share(entry, last_close, p) / risk_per_share,
                    "hold_sessions": last - entry_i + 1,
                    "trigger": ev["trigger"],
                    "adr_pct": ev["adr_pct"],
                    "stop_capped": bool(capped),
                }
            )
            break
        j, exit_px, reason = walked
        # An exact stop is -1R. Float dust was pushing some into the gap bucket.
        r_mult = -1.0 if reason == "stop" else _pnl_per_share(entry, exit_px, p) / risk_per_share
        closed.append(
            {
                "ticker": ticker,
                "signal_date": _date_str(a["dates"], i),
                "entry_date": _date_str(a["dates"], entry_i),
                "exit_date": _date_str(a["dates"], j),
                "entry": entry,
                "stop": stop,
                "exit": exit_px,
                "r": float(r_mult),
                "stop_capped": bool(capped),
                "exit_reason": reason,
                "hold_sessions": int(j - entry_i + 1),
                "adr_pct": ev["adr_pct"],
                "trigger": ev["trigger"],
                "dollar_vol": ev["dollar_vol"],
            }
        )
        i = j + 1
    return closed, still_open


def run_symbol(raw: pd.DataFrame, ticker: str, p: PullbackParams) -> tuple[list[dict], list[dict], dict]:
    prepared = add_indicators(raw, p)
    if prepared.empty:
        return [], [], {"ticker": ticker, "status": "warmup", "date": None}
    arrays = _arrays(prepared)
    closed, still_open = _simulate(arrays, ticker, p)
    snap = evaluate_bar(len(prepared) - 1, arrays, p)
    snap["ticker"] = ticker
    snap["date"] = pd.Timestamp(prepared.index[-1]).date().isoformat()
    return closed, still_open, snap


def _bin_label(r: float) -> str:
    if r < -1:
        return "< -1R"
    if r <= 0:
        return "-1 to 0R"
    if r < 1:
        return "0 to 1R"
    if r < 2:
        return "1 to 2R"
    if r < 3:
        return "2 to 3R"
    return "> 3R"


def _max_consec(rs: list[float]) -> int:
    """Longest run of R <= 0. Order is the caller's (exit order)."""
    best = current = 0
    for r in rs:
        if r <= 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_concurrent(closed: list[dict], still_open: list[dict]) -> int:
    """Peak overlapping positions. Same-day entries count before that day's exits."""
    events: list[tuple[str, int]] = []
    for trade in closed:
        events.append((trade["entry_date"], 1))
        events.append((trade["exit_date"], -1))
    for trade in still_open:
        events.append((trade["entry_date"], 1))
    events.sort(key=lambda item: (item[0], 0 if item[1] > 0 else 1))
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


def r_bins(rs: list[float]) -> list[dict]:
    counts = {label: 0 for label in R_BIN_LABELS}
    for r in rs:
        counts[_bin_label(r)] += 1
    return [{"label": label, "count": counts[label]} for label in R_BIN_LABELS]


def apply_equity(trades: list[dict], p: PullbackParams) -> tuple[list[dict], list[dict]]:
    """Compound risk_pct of equity at each exit.

    Trades are ordered by exit date. Overlapping names are all kept — see
    max concurrent — but the curve sizes each trade as if the risk is taken
    from equity after the previous exit. That is the single-stream reading
    of a 0.4% risk rule. The notional cap refuses leverage when the stop is
    tighter than the risk budget.
    """
    ordered = sorted(trades, key=lambda t: (t["exit_date"], t["entry_date"], t["ticker"]))
    equity = float(p.account_equity)
    curve: list[dict] = []
    if ordered:
        curve.append(
            {
                "date": ordered[0]["entry_date"],
                "equity": equity,
                "ticker": "",
                "r": None,
            }
        )
    enriched: list[dict] = []
    for trade in ordered:
        row = dict(trade)
        entry = float(trade["entry"])
        stop = float(trade["stop"])
        exit_px = float(trade["exit"])
        risk_per_share = _risk_per_share(entry, stop, p)
        if equity <= 0 or entry <= 0 or risk_per_share <= 0:
            shares = 0.0
        else:
            shares = (equity * p.risk_pct) / risk_per_share
            max_shares = (equity * p.max_position_pct) / entry
            if shares > max_shares:
                shares = max_shares
        actual_risk = shares * risk_per_share
        row["risk_frac"] = (actual_risk / equity) if equity > 0 else 0.0
        row["sized_to_target"] = bool(equity > 0 and actual_risk >= equity * p.risk_pct * 0.999)
        equity = equity + shares * _pnl_per_share(entry, exit_px, p)
        if equity < 0:
            equity = 0.0
        row["equity_after"] = equity
        curve.append(
            {
                "date": trade["exit_date"],
                "equity": equity,
                "ticker": trade["ticker"],
                "r": float(trade["r"]),
            }
        )
        enriched.append(row)
    return enriched, curve


def summarize(trades: list[dict], still_open: list[dict], curve: list[dict], p: PullbackParams) -> dict:
    """Win rate, average R, expectancy, drawdown, and the loss streak."""
    rs = [float(t["r"]) for t in trades]
    holds = [int(t["hold_sessions"]) for t in trades]
    empty = {
        "trades": 0,
        "win_rate": None,
        "avg_r": None,
        "median_r": None,
        "expectancy_r": None,
        "expectancy_frac": None,
        "max_consec_losses": 0,
        "profit_factor": None,
        "profit_factor_infinite": False,
        "avg_winner_r": None,
        "avg_loser_r": None,
        "max_r": None,
        "min_r": None,
        "avg_hold_days": None,
        "max_concurrent": _max_concurrent(trades, still_open),
        "stop_capped_frac": None,
        "full_risk_frac": None,
        "sized_below_target": 0,
        "start_equity": p.account_equity,
        "end_equity": p.account_equity,
        "return_frac": 0.0,
        "max_dd_frac": 0.0,
        "tickers_traded": 0,
    }
    if not rs:
        return empty
    winners = [r for r in rs if r > 0]
    losers = [r for r in rs if r < 0]
    gross_loss = -sum(losers)
    avg_r = float(np.mean(rs))
    equities = [p.account_equity] + [float(point["equity"]) for point in curve]
    series = pd.Series(equities, dtype=float)
    peak = series.cummax()
    drawdown = series / peak - 1.0
    end_equity = float(equities[-1])
    return {
        "trades": len(rs),
        "win_rate": float(np.mean([r > 0 for r in rs])),
        "avg_r": avg_r,
        "median_r": float(np.median(rs)),
        "expectancy_r": avg_r,
        "expectancy_frac": avg_r * p.risk_pct,
        "max_consec_losses": _max_consec(rs),
        "profit_factor": (sum(winners) / gross_loss) if gross_loss > 0 else None,
        "profit_factor_infinite": gross_loss == 0 and bool(winners),
        "avg_winner_r": float(np.mean(winners)) if winners else None,
        "avg_loser_r": float(np.mean(losers)) if losers else None,
        "max_r": float(np.max(rs)),
        "min_r": float(np.min(rs)),
        "avg_hold_days": float(np.mean(holds)) if holds else None,
        "max_concurrent": _max_concurrent(trades, still_open),
        "stop_capped_frac": float(np.mean([bool(t["stop_capped"]) for t in trades])),
        "full_risk_frac": float(np.mean([bool(t.get("sized_to_target")) for t in trades])),
        "sized_below_target": int(sum(not t.get("sized_to_target") for t in trades)),
        "start_equity": p.account_equity,
        "end_equity": end_equity,
        "return_frac": end_equity / p.account_equity - 1.0,
        "max_dd_frac": float(drawdown.min()),
        "tickers_traded": len({t["ticker"] for t in trades}),
    }


def exit_breakdown(trades: list[dict]) -> list[dict]:
    groups: dict[str, list[float]] = {}
    for trade in trades:
        groups.setdefault(trade["exit_reason"], []).append(float(trade["r"]))
    rows = []
    for reason, rs in groups.items():
        rows.append(
            {
                "reason": REASON_LABEL.get(reason, reason),
                "trades": len(rs),
                "avg_r": float(np.mean(rs)),
                "win_rate": float(np.mean([r > 0 for r in rs])),
            }
        )
    rows.sort(key=lambda row: (-row["trades"], row["reason"]))
    return rows


def _fetch_caps(tickers: list[str]) -> dict[str, float]:
    def one(ticker: str) -> tuple[str, float | None]:
        try:
            cap = yf.Ticker(ticker).fast_info.market_cap
            return ticker, float(cap) if cap else None
        except Exception:
            return ticker, None

    found: dict[str, float] = {}
    workers = min(8, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ticker, cap in pool.map(one, tickers):
            if cap and cap > 0:
                found[ticker] = cap
    return found


def market_caps(tickers: list[str]) -> dict[str, float]:
    """Yahoo market caps, cached for a week. Used only on the live screen."""
    if not tickers:
        return {}
    path = Path(config.CACHE_DIR) / "momentum_market_caps.json"
    cached: dict = {}
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
    fresh: dict[str, float] = {}
    stale: list[str] = []
    now = datetime.now()
    for ticker in dict.fromkeys(tickers):
        rec = cached.get(ticker) or {}
        cap = rec.get("cap")
        as_of = rec.get("as_of")
        if cap and as_of:
            age = now - datetime.fromisoformat(as_of)
            if age.days < 7:
                fresh[ticker] = float(cap)
                continue
        stale.append(ticker)
    if stale:
        print(f"  Market caps for {len(stale)} screened names…")
        fetched = _fetch_caps(stale)
        for ticker, cap in fetched.items():
            cached[ticker] = {"cap": cap, "as_of": now.isoformat(timespec="seconds")}
            fresh[ticker] = cap
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached), encoding="utf-8")
    return fresh


def _zone(ema: float, adr_dollar: float, p: PullbackParams) -> tuple[float, float]:
    if _is_short(p):
        return ema - p.proximity_adr_mult * adr_dollar, ema + p.undercut_adr_mult * adr_dollar
    return ema - p.undercut_adr_mult * adr_dollar, ema + p.proximity_adr_mult * adr_dollar


def build_screens(
    snaps: list[dict],
    still_open: list[dict],
    caps: dict[str, float],
    meta: dict[str, dict[str, str]],
    p: PullbackParams,
) -> tuple[list[dict], list[dict], dict]:
    """Current candidates, plus names in trend that have not reached the zone."""
    in_trade = {trade["ticker"] for trade in still_open}
    candidates: list[dict] = []
    removed_cap = 0
    removed_stop = 0
    for snap in snaps:
        if snap.get("status") != "setup":
            continue
        entry = snap["close"]
        if _is_short(p):
            structure = max(snap["recent_high"], snap["trigger_ema"])
        else:
            structure = min(snap["recent_low"], snap["trigger_ema"])
        priced = stop_price(entry, structure, snap["adr_pct"], p)
        if priced is None:
            removed_stop += 1
            continue
        stop, capped = priced
        cap = caps.get(snap["ticker"])
        if cap is not None and cap < p.min_market_cap:
            removed_cap += 1
            continue
        risk_per_share = _risk_per_share(entry, stop, p)
        shares = 0.0
        risk_dollars = 0.0
        if risk_per_share > 0 and entry > 0:
            shares = (p.account_equity * p.risk_pct) / risk_per_share
            max_shares = (p.account_equity * p.max_position_pct) / entry
            shares = min(shares, max_shares)
            risk_dollars = shares * risk_per_share
        zone_low, zone_high = _zone(snap["trigger_ema"], snap["adr_dollar"], p)
        info = meta.get(snap["ticker"], {})
        candidates.append(
            {
                "ticker": snap["ticker"],
                "name": info.get("name") or "",
                "sector": info.get("sector") or "",
                "date": snap.get("date"),
                "close": entry,
                "adr_pct": snap["adr_pct"],
                "dollar_vol": snap["dollar_vol"],
                "market_cap": cap,
                "dist_fast_pct": snap["dist_fast_pct"],
                "dist_fast_adr": snap["dist_fast_adr"],
                "dist_mid_pct": snap["dist_mid_pct"],
                "dist_mid_adr": snap["dist_mid_adr"],
                "trigger": snap["trigger"],
                "trigger_dist_adr": snap["trigger_dist_adr"],
                "zone_low": zone_low,
                "zone_high": zone_high,
                "stop": stop,
                "stop_pct": abs(entry - stop) / entry * 100.0,
                "stop_capped": bool(capped),
                "shares": shares,
                "risk_dollars": risk_dollars,
                "in_trade": snap["ticker"] in in_trade,
            }
        )
    # Tightest pullback first, then the wider-range names.
    candidates.sort(key=lambda row: (abs(row["trigger_dist_adr"]), -row["adr_pct"], row["ticker"]))
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank

    extended_src = [snap for snap in snaps if snap.get("status") == "extended"]
    extended_src.sort(
        key=lambda snap: (
            abs(snap["nearest_dist_adr"]) if snap.get("nearest_dist_adr") is not None else 99,
            snap["ticker"],
        )
    )
    extended: list[dict] = []
    for snap in extended_src:
        if snap.get("nearest_dist_adr") is None or not snap.get("adr_dollar"):
            continue
        cap = caps.get(snap["ticker"])
        if cap is not None and cap < p.min_market_cap:
            continue
        ema = snap["close"] - snap["nearest_dist_adr"] * snap["adr_dollar"]
        zone_low, zone_high = _zone(ema, snap["adr_dollar"], p)
        info = meta.get(snap["ticker"], {})
        extended.append(
            {
                "ticker": snap["ticker"],
                "name": info.get("name") or "",
                "sector": info.get("sector") or "",
                "date": snap.get("date"),
                "close": snap["close"],
                "adr_pct": snap["adr_pct"],
                "dollar_vol": snap["dollar_vol"],
                "market_cap": cap,
                "nearest_ema": snap["nearest_ema"],
                "nearest_dist_adr": snap["nearest_dist_adr"],
                "zone_low": zone_low,
                "zone_high": zone_high,
            }
        )
        if len(extended) >= 15:
            break
    extra = {"removed_market_cap": removed_cap, "removed_no_stop": removed_stop}
    return candidates, extended, extra


def _chart_fragments(
    curve: list[dict],
    bins: list[dict],
    *,
    equity_id: str,
    hist_id: str,
    start_equity: float,
    equity_color: str = "#4a7ab5",
) -> tuple[str, str]:
    if len(curve) < 2:
        equity_html = "<p class='muted'>No closed trades yet, so there is no equity curve.</p>"
    else:
        hover = []
        for point in curve:
            if point.get("r") is None:
                hover.append("Start")
            else:
                hover.append(f"{point.get('ticker') or ''} {point['r']:+.2f}R".strip())
        fig = go.Figure(
            go.Scatter(
                x=[point["date"] for point in curve],
                y=[point["equity"] for point in curve],
                mode="lines",
                line=dict(color=equity_color, width=1.8),
                text=hover,
                hovertemplate="%{x}<br>%{text}<br>Equity $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_hline(y=start_equity, line_dash="dot", line_color="#9ca3af", line_width=1)
        fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=16, r=16, t=36, b=32),
            title=dict(text="Equity curve", font=dict(size=13, color="#1f2937")),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=11, color="#6b7280"),
            showlegend=False,
            yaxis=dict(tickprefix="$", separatethousands=True, gridcolor="#e5e7eb"),
            xaxis=dict(gridcolor="#e5e7eb"),
        )
        equity_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=equity_id,
            config={"displayModeBar": False, "responsive": True},
        )

    total = sum(b["count"] for b in bins)
    if total == 0:
        hist_html = "<p class='muted'>No closed trades yet, so there is no R distribution.</p>"
    else:
        colors = ["#b91c1c" if b["label"].startswith(("<", "-")) else "#15803d" for b in bins]
        fig = go.Figure(
            go.Bar(
                x=[b["label"] for b in bins],
                y=[b["count"] for b in bins],
                marker_color=colors,
                text=[b["count"] for b in bins],
                textposition="auto",
                hovertemplate="%{x}: %{y} trades<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=16, r=16, t=36, b=32),
            title=dict(text="R-multiple distribution", font=dict(size=13, color="#1f2937")),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=11, color="#6b7280"),
            showlegend=False,
            yaxis=dict(title="Trades", gridcolor="#e5e7eb", rangemode="tozero"),
            xaxis=dict(gridcolor="#e5e7eb"),
        )
        hist_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=hist_id,
            config={"displayModeBar": False, "responsive": True},
        )
    return equity_html, hist_html


def _fmt_r(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:+.2f}R"


def _fmt_px(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _fmt_dist(pct, adr) -> str:
    if pct is None or adr is None:
        return "—"
    return f"{pct:+.1f}% · {adr:+.2f}×"


def _fmt_cap(value) -> str:
    if not value:
        return "—"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    return f"${value / 1e6:.0f}M"


def _r_html(value: float) -> str:
    css = "up" if value > 0 else "down" if value < 0 else ""
    return f"<span class='{css}'>{value:+.2f}R</span>"


def html_table(headers: list[str], rows: list[tuple[str, list[str]]], numeric: set[int] | None = None, empty: str = "None.") -> str:
    """Render a table. Cell strings are already HTML; headers are escaped."""
    if not rows:
        return f"<p class='muted'>{escape(empty)}</p>"
    numeric = numeric or set()
    head = "".join(
        f"<th class='{'num' if i in numeric else ''}'>{escape(header)}</th>"
        for i, header in enumerate(headers)
    )
    body = []
    for css, cells in rows:
        tds = "".join(
            f"<td class='{'num' if i in numeric else ''}'>{cell}</td>"
            for i, cell in enumerate(cells)
        )
        body.append(f"<tr class='{css}'>{tds}</tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def stat_pills(stats: dict) -> str:
    def pill(label: str, value: str, detail: str, tone: str = "normal") -> str:
        return (
            f"<div class='vol-pill {tone}'>"
            f"<div class='k'>{escape(label)}</div>"
            f"<div class='v'>{value}</div>"
            f"<div class='d'>{escape(detail)}</div>"
            "</div>"
        )

    trades = stats["trades"]
    if stats["profit_factor_infinite"]:
        pf = "∞"
    elif stats["profit_factor"] is None:
        pf = "—"
    else:
        pf = f"{stats['profit_factor']:.2f}"
    win = "—" if stats["win_rate"] is None else f"{stats['win_rate'] * 100:.1f}%"
    exp = "—" if stats["expectancy_frac"] is None else f"{stats['expectancy_frac'] * 100:+.2f}%"
    avg_tone = "stress" if (stats["avg_r"] or 0) < 0 else "calm"
    streak = stats["max_consec_losses"]
    streak_tone = "stress" if streak >= 8 else "elevated" if streak >= 5 else "normal"
    pills = [
        pill("Trades", f"{trades:,}", "Closed trades only", "normal"),
        pill("Win rate", win, "Share of trades with R > 0", "normal"),
        pill("Average R", _fmt_r(stats["avg_r"]), "Expectancy in R", avg_tone),
        pill("Expectancy", exp, "Average R × risk per trade", avg_tone),
        pill("Max loss streak", str(streak), "R ≤ 0, in exit order", streak_tone),
        pill("Profit factor", pf, "Gross winning R / gross losing R", "normal"),
    ]
    return "<div class='vol-stats'>" + "".join(pills) + "</div>"


def accounting_sentence(stats: dict, p: PullbackParams) -> str:
    if not stats["trades"]:
        return "No closed trades for these rules. The screen below still uses the latest bar."
    hold = stats["avg_hold_days"]
    hold_txt = f"{hold:.1f}" if hold is not None else "—"
    return (
        f"Closed trades only, ordered by exit. Equity starts at ${p.account_equity:,.0f} and compounds "
        f"{p.risk_pct * 100:.2f}% risk per trade, with notional capped at {p.max_position_pct * 100:.0f}% of equity. "
        f"Ending equity ${stats['end_equity']:,.0f} ({stats['return_frac'] * 100:+.1f}%). "
        f"Max drawdown {stats['max_dd_frac'] * 100:.1f}%. "
        f"Median {_fmt_r(stats['median_r'])}, best {_fmt_r(stats['max_r'])}, worst {_fmt_r(stats['min_r'])}. "
        f"Average winner {_fmt_r(stats['avg_winner_r'])}, average loser {_fmt_r(stats['avg_loser_r'])}. "
        f"Average hold {hold_txt} sessions. Max concurrent positions {stats['max_concurrent']}. "
        f"{stats['sized_below_target']} trades were sized below the risk budget so the position would not use leverage. "
        f"Stops tightened by the ADR cap: "
        f"{'—' if stats['stop_capped_frac'] is None else f'{stats['stop_capped_frac'] * 100:.0f}%'} of trades."
    )


def _candidate_rows(candidates: list[dict], p: PullbackParams, *, detailed: bool) -> list[tuple[str, list[str]]]:
    rows = []
    for row in candidates:
        zone = f"{row['zone_low']:.2f} – {row['zone_high']:.2f}"
        if detailed:
            zone += f" · {escape(str(row['trigger']))} EMA"
        stop = _fmt_px(row["stop"])
        if row["stop_capped"]:
            stop += " <span class='warn'>capped</span>"
        cells = [
            str(row["rank"]),
            f"<strong>{escape(row['ticker'])}</strong>",
        ]
        if detailed:
            cells.append(escape(row["name"]))
        cells.extend(
            [
                _fmt_px(row["close"]),
                f"{row['adr_pct']:.1f}%",
            ]
        )
        if detailed:
            cells.append(f"{row['dollar_vol'] / 1e6:,.0f}")
            cells.append(_fmt_cap(row["market_cap"]))
        cells.extend(
            [
                _fmt_dist(row["dist_fast_pct"], row["dist_fast_adr"]),
                _fmt_dist(row["dist_mid_pct"], row["dist_mid_adr"]),
                zone,
                stop,
                f"{row['stop_pct']:.1f}%",
            ]
        )
        if detailed:
            shares = int(round(row["shares"])) if row["shares"] else 0
            cells.append(f"{shares:,}" if shares else "—")
            cells.append(f"${row['risk_dollars']:,.0f}")
            cells.append("Yes" if row["in_trade"] else "")
        rows.append(("", cells))
    return rows


def candidate_headers(p: PullbackParams, *, detailed: bool) -> list[str]:
    headers = ["Rank", "Ticker"]
    if detailed:
        headers.append("Name")
    headers.extend(["Close", "ADR%"])
    if detailed:
        headers.extend(["$ Vol M", "Mkt cap"])
    headers.extend(
        [
            f"vs {p.ema_fast} EMA",
            f"vs {p.ema_mid} EMA",
            "Entry zone",
            "Stop",
            "Stop %",
        ]
    )
    if detailed:
        headers.extend(["Shares", "Risk $", "Held"])
    return headers


def _numeric_indexes(headers: list[str]) -> set[int]:
    text_headers = {"Ticker", "Name", "Entry zone", "Held"}
    return {i for i, header in enumerate(headers) if header not in text_headers}


def funnel_sentence(funnel: dict) -> str:
    text = (
        f"{funnel['listed']} listed → {funnel['downloaded']} priced → "
        f"{funnel['pass_liquidity']} pass dollar volume and ADR → "
        f"{funnel['pass_trend']} in trend → {funnel['candidates']} in the setup zone"
    )
    if funnel.get("removed_market_cap"):
        text += f" ({funnel['removed_market_cap']} removed under the market-cap rule)"
    return text


def _side_book_path(side: str) -> Path:
    return Path(config.CACHE_DIR) / f"momentum_{side}_trades.parquet"


def _save_side_book(trades: list[dict], side: str) -> None:
    """Keep each side on disk so the comparison can show long and short together."""
    frame = pd.DataFrame(trades)
    if frame.empty:
        return
    frame = frame.copy()
    frame["side"] = side
    path = _side_book_path(side)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _load_side_book(side: str) -> pd.DataFrame | None:
    path = _side_book_path(side)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty or "r" not in frame.columns:
        return None
    frame = frame.copy()
    frame["side"] = side
    for col in ("entry_date", "exit_date", "signal_date"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col])
    frame["r"] = frame["r"].astype(float)
    for col in ("entry", "stop", "exit"):
        frame[col] = frame[col].astype(float)
    return frame


def _mixed_equity(frame: pd.DataFrame, p: PullbackParams) -> tuple[list[dict], float, float]:
    """Compound 0.40% risk in exit order. Each row uses its own side for the P&L."""
    ordered = frame.sort_values(["exit_date", "entry_date", "ticker"])
    equity = float(p.account_equity)
    curve: list[dict] = []
    if not ordered.empty:
        curve.append({"date": pd.Timestamp(ordered.iloc[0]["entry_date"]).date().isoformat(), "equity": equity})
    peak = equity
    max_dd = 0.0
    for row in ordered.itertuples(index=False):
        entry = float(row.entry)
        stop = float(row.stop)
        exit_px = float(row.exit)
        short = row.side == "short"
        risk = (stop - entry) if short else (entry - stop)
        pnl = (entry - exit_px) if short else (exit_px - entry)
        if equity <= 0 or risk <= 0 or entry <= 0:
            continue
        shares = min((equity * p.risk_pct) / risk, (equity * p.max_position_pct) / entry)
        equity = max(0.0, equity + shares * pnl)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
        curve.append(
            {
                "date": pd.Timestamp(row.exit_date).date().isoformat(),
                "equity": equity,
                "ticker": row.ticker,
                "r": float(row.r),
            }
        )
    return curve, equity, max_dd


def _monthly_matrix(frame: pd.DataFrame) -> tuple[list[int], list[list[float | None]], list[list[str]]]:
    stamps = pd.to_datetime(frame["entry_date"])
    period = stamps.dt.to_period("M")
    sums = frame.groupby(period, observed=True)["r"].sum()
    years = list(range(int(stamps.dt.year.min()), int(stamps.dt.year.max()) + 1))
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for year in years:
        row_z: list[float | None] = []
        row_t: list[str] = []
        for month in range(1, 13):
            key = pd.Period(year=year, month=month, freq="M")
            if key in sums.index:
                value = float(sums.loc[key])
                row_z.append(value)
                row_t.append(f"{value:+.0f}")
            else:
                row_z.append(None)
                row_t.append("")
        z.append(row_z)
        text.append(row_t)
    return years, z, text


def _heatmap_html(frame: pd.DataFrame, *, title: str, div_id: str, clip: float) -> str:
    years, z, text = _monthly_matrix(frame)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=[str(year) for year in years],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#1f2937"),
            colorscale=[[0.0, "#b91c1c"], [0.5, "#f8fafc"], [1.0, "#15803d"]],
            zmid=0,
            zmin=-clip,
            zmax=clip,
            xgap=3,
            ygap=3,
            hovertemplate="%{y} %{x}<br>Sum of R %{z:+.1f}<extra></extra>",
            colorbar=dict(title="Sum of R", thickness=12, len=0.8),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=48, r=12, t=36, b=28),
        title=dict(text=title, font=dict(size=13, color="#1f2937")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=11, color="#6b7280"),
        yaxis=dict(autorange="reversed", title="Entry year"),
        xaxis=dict(title="Entry month", side="top"),
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def _equity_panel(curve: list[dict], *, title: str, div_id: str, color: str, start: float) -> str:
    if len(curve) < 2:
        return "<p class='muted'>No closed trades.</p>"
    fig = go.Figure(
        go.Scatter(
            x=[point["date"] for point in curve],
            y=[point["equity"] for point in curve],
            mode="lines",
            line=dict(color=color, width=1.8),
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=start, line_dash="dot", line_color="#9ca3af", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=12, r=12, t=36, b=28),
        title=dict(text=title, font=dict(size=13, color="#1f2937")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=11, color="#6b7280"),
        showlegend=False,
        yaxis=dict(tickprefix="$", separatethousands=True, gridcolor="#e5e7eb"),
        xaxis=dict(gridcolor="#e5e7eb"),
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def _comparison_card(name: str, frame: pd.DataFrame, end: float, max_dd: float, tone: str) -> str:
    win = float((frame["r"] > 0).mean() * 100.0)
    avg = float(frame["r"].mean())
    total = float(frame["r"].sum())
    return (
        f"<div class='vol-pill {tone}'>"
        f"<div class='k'>{escape(name)}</div>"
        f"<div class='v'>${end:,.0f}</div>"
        f"<div class='d'>{len(frame):,} trades · win {win:.1f}% · avg {avg:+.2f}R · "
        f"sum {total:+,.0f}R · max DD {max_dd * 100:.0f}%</div>"
        "</div>"
    )


def comparison_section_html(p: PullbackParams, *, id_prefix: str) -> str:
    """Long, short, and both, with monthly gain and loss highlights.

    A week-by-week grid is about 450 cells across this window, so the color
    washes out. Months stay readable: green is a gain, red is a loss.
    """
    long_book = _load_side_book("long")
    short_book = _load_side_book("short")
    if long_book is None or short_book is None:
        return (
            "<div class='section-title' id='long-short-both'>Long, short, and both</div>"
            "<p class='section-desc'>Run the long book and the short book once each. "
            "The comparison appears here after both result files exist.</p>"
        )
    both = pd.concat([long_book, short_book], ignore_index=True)
    long_curve, long_end, long_dd = _mixed_equity(long_book, p)
    short_curve, short_end, short_dd = _mixed_equity(short_book, p)
    both_curve, both_end, both_dd = _mixed_equity(both, p)
    weeks = pd.to_datetime(both["entry_date"]).dt.to_period("W").nunique()
    cards = (
        "<div class='cmp-grid'>"
        + _comparison_card("Long", long_book, long_end, long_dd, "calm")
        + _comparison_card("Short", short_book, short_end, short_dd, "stress")
        + _comparison_card("Both together", both, both_end, both_dd, "elevated")
        + "</div>"
    )
    equities = (
        "<div class='cmp-grid'>"
        f"<div class='panel'>{_equity_panel(long_curve, title='Long equity', div_id=id_prefix + '-long-eq', color='#15803d', start=p.account_equity)}</div>"
        f"<div class='panel'>{_equity_panel(short_curve, title='Short equity', div_id=id_prefix + '-short-eq', color='#b91c1c', start=p.account_equity)}</div>"
        f"<div class='panel'>{_equity_panel(both_curve, title='Both together', div_id=id_prefix + '-both-eq', color='#4a7ab5', start=p.account_equity)}</div>"
        "</div>"
    )
    clip = 60.0
    heats = (
        f"<div class='panel'>{_heatmap_html(long_book, title='Long, sum of R by entry month', div_id=id_prefix + '-long-heat', clip=clip)}</div>"
        f"<div class='panel'>{_heatmap_html(short_book, title='Short, sum of R by entry month', div_id=id_prefix + '-short-heat', clip=clip)}</div>"
        f"<div class='panel'>{_heatmap_html(both, title='Both together, sum of R by entry month', div_id=id_prefix + '-both-heat', clip=clip)}</div>"
    )
    return f"""
    <div class="section-title" id="long-short-both">Long, short, and both</div>
    <p class="section-desc">
      Same names and the same rules, one book long and one book short, entries from 2018.
      Both together takes every long and every short. Green months made R. Red months lost R.
      A blank cell had no entries. Color saturates at ±{clip:.0f}R so a crash month does not wash out the rest;
      the number in the cell is the real sum. Weekly would be {weeks} columns and the color would disappear, so this is monthly.
      Equity compounds {p.risk_pct * 100:.2f}% of equity per closed trade in exit order, one book at a time.
    </p>
    {cards}
    {equities}
    {heats}
    """


def render_dashboard_section(ctx: dict) -> str:
    """Fragment dropped into the breadth dashboard. Uses that page's classes."""
    p: PullbackParams = ctx["params"]
    stats = ctx["stats"]
    headers = candidate_headers(p, detailed=False)
    preview = ctx["candidates"][:8]
    table = html_table(
        headers,
        _candidate_rows(preview, p, detailed=False),
        _numeric_indexes(headers),
        empty="No names are in the setup zone on the latest bar.",
    )
    sample = ""
    if ctx["sample"]:
        sample = "<p class='section-desc warn-text'>This run used a partial ticker list, so treat the stats as a sample.</p>"
    as_of = ctx["as_of"] or "the latest bar"
    if _is_short(p):
        exit_blurb = (
            "exit on a close above the EMA that was shorted"
            if p.exit_ema is None
            else f"exit on a close above the {p.exit_ema} EMA"
        )
        stack_blurb = f"{p.ema_fast} below {p.ema_mid} below {p.ema_slow} EMA, rally within {p.proximity_adr_mult:g}× ADR"
    else:
        exit_blurb = (
            "exit on a close below the EMA that was bought"
            if p.exit_ema is None
            else f"exit on a close below the {p.exit_ema} EMA"
        )
        stack_blurb = f"{p.ema_fast}/{p.ema_mid}/{p.ema_slow} EMA stack, pullback within {p.proximity_adr_mult:g}× ADR"
    blurb = (
        f"{_book_name(p)} as of {as_of}. "
        f"{stack_blurb}, "
        f"stop capped at {p.stop_adr_cap:g}× ADR%, {p.risk_pct * 100:.2f}% risk, "
        f"{exit_blurb}. {funnel_sentence(ctx['funnel'])}."
    )
    if p.skip_months:
        blurb += f" No new entries in {_skip_label(p)}."
    href = _page_href(p.side)
    which = "short" if _is_short(p) else "long"
    return f"""
    <div class="section-title" id="{_section_id(p.side)}">{escape(_book_name(p))}</div>
    <p class="section-desc">{escape(blurb)}</p>
    {sample}
    {stat_pills(stats)}
    <p class="section-desc">{escape(accounting_sentence(stats, p))}</p>
    <div class="mom-charts">
      <div class="panel">{ctx["equity_section_html"]}</div>
      <div class="panel">{ctx["hist_section_html"]}</div>
    </div>
    <div class="table-card">
      <h3>Current candidates</h3>
      <p class="muted">Ranked by how close the close is to the trigger EMA. Entry zone is that EMA ± the setup band. Stop uses the last close as the estimated entry. {len(ctx["candidates"])} names in the zone; the full table is on the {which} page.</p>
      {table}
    </div>
    <a class="mom-link" href="{href}">Open the full {which} backtest, distribution, and ranked screen</a>
    """


def _side_section_path(side: str) -> Path:
    return Path(config.OUTPUT_DIR) / f"momentum_{side}_section.html"


def _missing_side_section(side: str) -> str:
    name = "Long momentum rally" if side == "long" else "Short momentum rally"
    flag = "long" if side == "long" else "short"
    return (
        f'<div class="section-title" id="{_section_id(side)}">{name}</div>'
        f'<p class="section-desc">This breakdown is empty until '
        f'<code>python3 momentum_pullback.py --start 2018-01-01 --side {flag} --skip-months none</code> '
        f"has been run.</p>"
    )


def combined_dashboard_html(p: PullbackParams, fresh: dict[str, str]) -> str:
    """Long breakdown, short breakdown, then the monthly comparison. One of each."""
    parts: list[str] = []
    for side in ("long", "short"):
        html = fresh.get(side)
        if html is None:
            path = _side_section_path(side)
            html = path.read_text(encoding="utf-8") if path.exists() else _missing_side_section(side)
        parts.append(html)
    parts.append(comparison_section_html(p, id_prefix="dash-cmp"))
    return "\n".join(parts)


def render_page(ctx: dict) -> str:
    p: PullbackParams = ctx["params"]
    stats = ctx["stats"]
    rules = "".join(
        f"<li><strong>{escape(title)}.</strong> {escape(text)}</li>"
        for title, text in rules_text(p, ctx["universe_label"])
    )
    cand_headers = candidate_headers(p, detailed=True)
    candidates = html_table(
        cand_headers,
        _candidate_rows(ctx["candidates"], p, detailed=True),
        _numeric_indexes(cand_headers),
        empty="No names are in the setup zone on the latest bar.",
    )
    ext_rows = []
    for row in ctx["extended"]:
        dist = row["nearest_dist_adr"]
        dist_txt = "—" if dist is None else f"{dist:+.2f}× {row['nearest_ema']} EMA"
        ext_rows.append(
            (
                "",
                [
                    f"<strong>{escape(row['ticker'])}</strong>",
                    escape(row["name"]),
                    _fmt_px(row["close"]),
                    f"{row['adr_pct']:.1f}%",
                    f"{row['dollar_vol'] / 1e6:,.0f}",
                    dist_txt,
                    f"{row['zone_low']:.2f} – {row['zone_high']:.2f}",
                ],
            )
        )
    extended = html_table(
        ["Ticker", "Name", "Close", "ADR%", "$ Vol M", "Distance", "Zone if it pulls back"],
        ext_rows,
        {2, 3, 4, 5, 6},
        empty="No names are in trend outside the zone.",
    )
    open_rows = []
    for trade in sorted(ctx["opens"], key=lambda t: -t["current_r"]):
        open_rows.append(
            (
                "win" if trade["current_r"] > 0 else "loss",
                [
                    escape(trade["entry_date"]),
                    f"<strong>{escape(trade['ticker'])}</strong>",
                    escape(str(trade["trigger"])),
                    _fmt_px(trade["entry"]),
                    _fmt_px(trade["stop"]),
                    _fmt_px(trade["last_close"]),
                    _r_html(trade["current_r"]),
                    str(trade["hold_sessions"]),
                ],
            )
        )
    opens = html_table(
        ["Entry", "Ticker", "EMA", "Entry px", "Stop", "Last", "Open R", "Sessions"],
        open_rows,
        {0, 2, 3, 4, 5, 6, 7},
        empty="No simulated positions are still open.",
    )
    recent_rows = []
    recent = list(reversed(ctx["trades"]))[:40]
    for trade in recent:
        recent_rows.append(
            (
                "win" if trade["r"] > 0 else "loss",
                [
                    escape(trade["exit_date"]),
                    f"<strong>{escape(trade['ticker'])}</strong>",
                    escape(str(trade["trigger"])),
                    _fmt_px(trade["entry"]),
                    _fmt_px(trade["stop"]),
                    _fmt_px(trade["exit"]),
                    _r_html(trade["r"]),
                    escape(REASON_LABEL.get(trade["exit_reason"], trade["exit_reason"])),
                    str(trade["hold_sessions"]),
                ],
            )
        )
    trades_table = html_table(
        ["Exit", "Ticker", "EMA", "Entry", "Stop", "Exit px", "R", "Reason", "Sessions"],
        recent_rows,
        {0, 2, 3, 4, 5, 6, 8},
        empty="No closed trades.",
    )
    reason_rows = []
    for row in ctx["exit_reasons"]:
        reason_rows.append(
            (
                "",
                [
                    escape(row["reason"]),
                    f"{row['trades']:,}",
                    f"{row['win_rate'] * 100:.1f}%",
                    _r_html(row["avg_r"]),
                ],
            )
        )
    reasons = html_table(
        ["Exit", "Trades", "Win rate", "Average R"],
        reason_rows,
        {1, 2, 3},
        empty="No closed trades.",
    )
    param_rows = []
    for key, value in asdict(p).items():
        shown = f"{value:.6g}" if isinstance(value, float) else str(value)
        param_rows.append(("", [f"<code>{escape(key)}</code>", escape(shown)]))
    params_html = html_table(["Parameter", "Value"], param_rows, {1})
    sample = ""
    if ctx["sample"]:
        sample = "<p class='sample'>Partial ticker list. These stats are a sample, not the full index universe.</p>"
    as_of = ctx["as_of"] or "—"
    command = reproduce_command(p)
    long_active = "" if _is_short(p) else "active"
    short_active = "active" if _is_short(p) else ""
    trades_file = "momentum_long_trades.csv" if not _is_short(p) else "momentum_pullback_trades.csv"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(_book_name(p))}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
{PAGE_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="topnav">
      <a href="index.html">Dashboard</a>
      <a class="{long_active}" href="long.html">Long momentum rally</a>
      <a class="{short_active}" href="short.html">Short momentum rally</a>
    </div>
    <h1>{escape(_book_name(p))}</h1>
    <p class="subtitle">
      As of {escape(as_of)}. {escape(ctx["universe_label"])}.
      Yahoo Finance adjusted prices, {escape(history_label(p))}.
      Generated {escape(ctx["generated_at"])}.
    </p>
    {sample}

    {comparison_section_html(p, id_prefix="page-cmp")}

    <div class="section-title">Rules</div>
    <p class="section-desc">
      Each line is the rule this run actually used. Edit <code>PullbackParams</code> in
      <code>momentum_pullback.py</code>, or pass the flags in the command at the bottom.
    </p>
    <ol class="rules">{rules}</ol>

    <div class="section-title">Backtest</div>
    <p class="section-desc">{escape(funnel_sentence(ctx["funnel"]))}{"" if not ctx["errors"] else f". {ctx['errors']} symbols failed to simulate."}</p>
    {stat_pills(stats)}
    <p class="section-desc">{escape(accounting_sentence(stats, p))}</p>
    <div class="mom-charts">
      <div class="panel">{ctx["equity_page_html"]}</div>
      <div class="panel">{ctx["hist_page_html"]}</div>
    </div>

    <div class="section-title">How trades ended</div>
    <div class="table-card">{reasons}</div>

    <div class="section-title">Still open</div>
    <p class="section-desc">Marked to the last close. These are not in the win rate, average R, or equity curve.</p>
    <div class="table-card">{opens}</div>

    <div class="section-title">Recent closed trades</div>
    <p class="section-desc">Latest 40 by exit date. The full list is <code>output/{trades_file}</code>.</p>
    <div class="table-card">{trades_table}</div>

    <div class="section-title">Current candidates</div>
    <p class="section-desc">
      Latest bar only. Ranked by distance to the trigger EMA, tightest pullback first.
      The entry zone is that EMA minus {p.undercut_adr_mult:g}× ADR through that EMA plus {p.proximity_adr_mult:g}× ADR.
      Estimated entry is the last close; a live order would be the next open. The stop is calculated from that estimate.
      Shares assume a ${p.account_equity:,.0f} account risking {p.risk_pct * 100:.2f}%.
      {"No new entries in " + _skip_label(p) + "." if p.skip_months else "Every month is eligible."}
      Held means the backtest is already in the name, so it is not a fresh entry.
    </p>
    <div class="table-card">{candidates}</div>

    <div class="section-title">In trend, not in the zone yet</div>
    <p class="section-desc">
      Same universe and EMA stack, but the close is outside the setup band.
      Sorted by distance to the nearest setup EMA that is sloping with the trend. These are not signals.
    </p>
    <div class="table-card">{extended}</div>

    <div class="section-title">Parameters</div>
    <details>
      <summary>Values used for this run</summary>
      {params_html}
    </details>
    <p class="section-desc">Rebuild with</p>
    <p><code class="cmd">{escape(command)}</code></p>
    <p class="subtitle">Yahoo Finance for history. Futu OpenD is not called here; it stays available for live account work.</p>
  </div>
</body>
</html>
"""


PAGE_CSS = """
    :root {
      --bg: #f3f4f6;
      --card: #ffffff;
      --border: #d1d5db;
      --text: #1f2937;
      --muted: #6b7280;
      --up: #15803d;
      --down: #b91c1c;
      --accent: #a83a3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .page { max-width: 1400px; margin: 0 auto; padding: 20px 16px 48px; }
    h1 { margin: 0 0 4px; font-size: 1.55rem; font-weight: 650; }
    .subtitle { margin: 0 0 18px; color: var(--muted); font-size: 0.92rem; }
    .section-title {
      margin: 22px 0 10px;
      font-size: 1.05rem;
      font-weight: 650;
      border-bottom: 2px solid var(--accent);
      display: inline-block;
      padding-bottom: 2px;
    }
    .section-desc {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.86rem;
      max-width: 980px;
      line-height: 1.45;
    }
    .topnav { display: flex; gap: 8px; margin: 0 0 14px; flex-wrap: wrap; }
    .topnav a {
      text-decoration: none;
      color: var(--muted);
      background: #fff;
      border: 1px solid var(--border);
      padding: 5px 10px;
      font-size: 0.82rem;
      font-weight: 600;
    }
    .topnav a.active { color: #fff; background: #1f2937; border-color: #1f2937; }
    .rules { margin: 0 0 8px; padding-left: 1.2rem; max-width: 980px; }
    .rules li { margin: 0 0 8px; font-size: 0.9rem; line-height: 1.45; }
    .sample {
      background: #fff7ed;
      border: 1px solid #fdba74;
      padding: 8px 12px;
      font-size: 0.88rem;
      margin: 0 0 12px;
    }
    .vol-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 8px 0 12px;
    }
    @media (max-width: 700px) {
      .vol-stats { grid-template-columns: 1fr 1fr; }
    }
    .vol-pill {
      background: var(--card);
      border: 1px solid var(--border);
      padding: 10px 12px;
    }
    .vol-pill .k { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
    .vol-pill .v { font-size: 1.35rem; font-weight: 700; margin: 4px 0 2px; }
    .vol-pill .d { font-size: 0.78rem; color: var(--muted); line-height: 1.35; }
    .vol-pill.calm { border-left: 4px solid #15803d; }
    .vol-pill.normal { border-left: 4px solid #4a7ab5; }
    .vol-pill.elevated { border-left: 4px solid #b45309; }
    .vol-pill.stress { border-left: 4px solid #b91c1c; }
    .mom-charts {
      display: grid;
      grid-template-columns: 1.35fr 0.85fr;
      gap: 12px;
      margin-top: 4px;
    }
    .mom-charts > * { min-width: 0; }
    .cmp-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 12px;
    }
    .cmp-grid > * { min-width: 0; }
    @media (max-width: 900px) {
      .mom-charts, .cmp-grid { grid-template-columns: 1fr; }
    }
    .panel, .table-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 2px;
      padding: 10px 10px 8px;
      margin: 0 0 12px;
      max-width: 100%;
      overflow: hidden;
    }
    .table-card h3 { margin: 4px 8px 8px; font-size: 0.98rem; }
    .table-wrap { display: block; width: 100%; max-width: 100%; overflow: auto; max-height: 640px; }
    table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th, td {
      text-align: left;
      padding: 5px 6px;
      border-bottom: 1px solid #e5e7eb;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: #1f2937;
      color: #fff;
      font-weight: 600;
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr:nth-child(even) td { background: #f8fafc; }
    tr.win td { background: #f0fdf4; }
    tr.loss td { background: #fef2f2; }
    .up { color: var(--up); font-weight: 650; }
    .down { color: var(--down); font-weight: 650; }
    .warn { color: #b45309; font-size: 0.75rem; }
    .muted { color: var(--muted); padding: 8px; margin: 0; }
    a { color: #1e4d8c; }
    code { font-size: 0.84em; background: #eef2f7; padding: 1px 4px; }
    code.cmd { display: inline-block; padding: 8px 10px; }
    details { margin: 8px 0 12px; }
    summary { cursor: pointer; font-size: 0.9rem; }
"""


def reproduce_command(p: PullbackParams) -> str:
    return (
        "python3 momentum_pullback.py"
        f" --period {p.period}"
        f"{' --start ' + p.start if p.start else ''}"
        f" --universe {p.universe}"
        f" --min-adr {p.min_adr_pct:g}"
        f" --min-dollar-vol {p.min_dollar_volume / 1e6:g}"
        f" --proximity {p.proximity_adr_mult:g}"
        f" --stop-cap {p.stop_adr_cap:g}"
        f" --risk-pct {p.risk_pct * 100:g}"
        f" --side {p.side}"
        f" --exit-ema {'trigger' if p.exit_ema is None else p.exit_ema}"
        f" --skip-months {','.join(str(month) for month in p.skip_months) if p.skip_months else 'none'}"
        f" --account {p.account_equity:.0f}"
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _output_paths(side: str) -> dict[str, Path]:
    if side == "long":
        return {
            "html": Path(config.MOMENTUM_LONG_HTML),
            "json": Path(config.MOMENTUM_LONG_JSON),
            "trades": Path(config.MOMENTUM_LONG_TRADES_CSV),
            "candidates": Path(config.MOMENTUM_LONG_CANDIDATES_CSV),
        }
    return {
        "html": Path(config.MOMENTUM_PULLBACK_HTML),
        "json": Path(config.MOMENTUM_PULLBACK_JSON),
        "trades": Path(config.MOMENTUM_PULLBACK_TRADES_CSV),
        "candidates": Path(config.MOMENTUM_PULLBACK_CANDIDATES_CSV),
    }


def write_outputs(ctx: dict) -> None:
    p: PullbackParams = ctx["params"]
    paths = _output_paths(p.side)
    paths["html"].parent.mkdir(parents=True, exist_ok=True)

    page = render_page(ctx)
    paths["html"].write_text(page, encoding="utf-8")

    trade_cols = [
        "ticker", "signal_date", "entry_date", "exit_date", "entry", "stop", "exit",
        "r", "stop_capped", "exit_reason", "hold_sessions", "adr_pct", "trigger",
        "dollar_vol", "risk_frac", "equity_after",
    ]
    trades = pd.DataFrame(ctx["trades"])
    if trades.empty:
        trades = pd.DataFrame(columns=trade_cols)
    else:
        trades = trades.reindex(columns=trade_cols)
    trades.to_csv(paths["trades"], index=False)

    cand_cols = [
        "rank", "ticker", "name", "sector", "date", "close", "adr_pct", "dollar_vol",
        "market_cap", "trigger", "trigger_dist_adr", "dist_fast_pct", "dist_fast_adr",
        "dist_mid_pct", "dist_mid_adr", "zone_low", "zone_high", "stop", "stop_pct",
        "stop_capped", "shares", "risk_dollars", "in_trade",
    ]
    cands = pd.DataFrame(ctx["candidates"])
    if cands.empty:
        cands = pd.DataFrame(columns=cand_cols)
    else:
        cands = cands.reindex(columns=cand_cols)
    cands.to_csv(paths["candidates"], index=False)

    side_section = ctx["section_html"]
    _side_section_path(p.side).write_text(side_section, encoding="utf-8")
    # The breadth page reads one fragment. Long and short each keep their own page.
    dashboard_section = combined_dashboard_html(p, {p.side: side_section})
    summary = {
        "as_of": ctx["as_of"],
        "generated_at": ctx["generated_at"],
        "sample": ctx["sample"],
        "universe_label": ctx["universe_label"],
        "funnel": ctx["funnel"],
        "stats": ctx["stats"],
        "equity_curve": ctx["curve"],
        "r_bins": ctx["bins"],
        "candidates": ctx["candidates"],
        "extended": ctx["extended"],
        "open_positions": ctx["opens"],
        "exit_reasons": ctx["exit_reasons"],
        "params": asdict(ctx["params"]),
        "section_html": side_section,
    }
    paths["json"].write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    if p.side == "long":
        # Breadth dashboard injects this file. Keep the combined long + short fragment here.
        dash = dict(summary)
        dash["section_html"] = dashboard_section
        Path(config.MOMENTUM_PULLBACK_JSON).write_text(
            json.dumps(_json_ready(dash), indent=2), encoding="utf-8"
        )
    else:
        # Short summary is the file the dashboard reads. Swap in both breakdowns.
        summary["section_html"] = dashboard_section
        paths["json"].write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(f"Wrote {paths['html']}")
    print(f"Wrote {paths['trades']}")
    print(f"Wrote {paths['candidates']}")


def _check_stop_caps(trades: list[dict], p: PullbackParams) -> None:
    for trade in trades:
        cap = p.stop_adr_cap * (float(trade["adr_pct"]) / 100.0) + 1e-6
        risk_frac = abs(float(trade["entry"]) - float(trade["stop"])) / float(trade["entry"])
        if risk_frac > cap:
            raise RuntimeError(
                f"Stop cap breached on {trade['ticker']} {trade['entry_date']}: "
                f"{risk_frac:.4%} vs cap {cap:.4%}"
            )


def run(p: PullbackParams, tickers: list[str], label: str, partial: bool, *, refresh: bool, write_cache: bool) -> dict:
    prices = load_prices(tickers, p, refresh=refresh, write_cache=write_cache)
    closed: list[dict] = []
    still_open: list[dict] = []
    snaps: list[dict] = []
    errors = 0
    symbols = list(prices.items())
    for n, (ticker, frame) in enumerate(symbols, start=1):
        try:
            symbol_closed, symbol_open, snap = run_symbol(frame, ticker, p)
        except Exception as exc:
            errors += 1
            print(f"  skip {ticker}: {exc}")
            snaps.append({"ticker": ticker, "status": "warmup", "date": None})
            continue
        closed.extend(symbol_closed)
        still_open.extend(symbol_open)
        snaps.append(snap)
        if n % 100 == 0 or n == len(symbols):
            print(f"  simulated {n}/{len(symbols)}")
    if p.start:
        # Warmup bars stay in the indicators. Only entries inside the window count.
        closed = [trade for trade in closed if trade["entry_date"] >= p.start]
        still_open = [trade for trade in still_open if trade["entry_date"] >= p.start]
        print(f"  kept trades entered on or after {p.start}")
    _check_stop_caps(closed, p)
    _save_side_book(closed, p.side)
    enriched, curve = apply_equity(closed, p)
    stats = summarize(enriched, still_open, curve, p)
    bins = r_bins([float(t["r"]) for t in enriched])
    meta = load_name_meta()
    setup_names = [s["ticker"] for s in snaps if s.get("status") == "setup"]
    extended_names = sorted(
        (s for s in snaps if s.get("status") == "extended" and s.get("nearest_dist_adr") is not None),
        key=lambda s: abs(s["nearest_dist_adr"]),
    )
    cap_names = setup_names + [s["ticker"] for s in extended_names[:15]]
    caps = market_caps(cap_names)
    candidates, extended, extra = build_screens(snaps, still_open, caps, meta, p)
    dates = [s["date"] for s in snaps if s.get("date")]
    funnel = {
        "listed": len(tickers),
        "downloaded": len(prices),
        "pass_liquidity": sum(1 for s in snaps if s.get("status") in {"trend", "extended", "setup"}),
        "pass_trend": sum(1 for s in snaps if s.get("status") in {"extended", "setup"}),
        "in_zone": sum(1 for s in snaps if s.get("status") == "setup"),
        "candidates": len(candidates),
        "removed_market_cap": extra["removed_market_cap"],
        "removed_no_stop": extra["removed_no_stop"],
    }
    equity_color = "#b91c1c" if _is_short(p) else "#15803d"
    equity_section, hist_section = _chart_fragments(
        curve,
        bins,
        equity_id=f"mp-{p.side}-equity",
        hist_id=f"mp-{p.side}-rdist",
        start_equity=p.account_equity,
        equity_color=equity_color,
    )
    equity_page, hist_page = _chart_fragments(
        curve,
        bins,
        equity_id=f"mp-{p.side}-page-equity",
        hist_id=f"mp-{p.side}-page-rdist",
        start_equity=p.account_equity,
        equity_color=equity_color,
    )
    ctx = {
        "as_of": max(dates) if dates else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample": partial,
        "universe_label": label,
        "params": p,
        "funnel": funnel,
        "stats": stats,
        "curve": curve,
        "bins": bins,
        "candidates": candidates,
        "extended": extended,
        "opens": still_open,
        "trades": enriched,
        "exit_reasons": exit_breakdown(enriched),
        "errors": errors,
        "equity_section_html": equity_section,
        "hist_section_html": hist_section,
        "equity_page_html": equity_page,
        "hist_page_html": hist_page,
    }
    ctx["section_html"] = render_dashboard_section(ctx)
    write_outputs(ctx)
    _print_headline(ctx)
    return ctx


def _print_headline(ctx: dict) -> None:
    stats = ctx["stats"]
    print(funnel_sentence(ctx["funnel"]))
    if not stats["trades"]:
        print("No closed trades.")
        return
    print(
        f"Trades {stats['trades']}  win {stats['win_rate'] * 100:.1f}%  "
        f"avg R {stats['avg_r']:+.2f}  expectancy {stats['expectancy_frac'] * 100:+.2f}%  "
        f"max streak {stats['max_consec_losses']}  "
        f"equity ${stats['end_equity']:,.0f} ({stats['return_frac'] * 100:+.1f}%)"
    )
    print(f"Candidates in the zone: {len(ctx['candidates'])}")


def self_test() -> None:
    """Stop math, loss streak, and a render of an empty result. No network."""
    p = PullbackParams()
    long_p = replace(p, side="long")
    wide = stop_price(100, 90, 8, long_p)
    tight = stop_price(100, 98, 8, long_p)
    assert wide is not None and abs(wide[0] - 96.0) < 1e-9 and wide[1] is True
    assert tight is not None and abs(tight[0] - 98.0) < 1e-9 and tight[1] is False
    assert stop_price(100, 100, 8, long_p) is None
    assert stop_price(100, 105, 8, long_p) is None
    skipped = replace(long_p, skip_if_wider_than_cap=True)
    assert stop_price(100, 90, 8, skipped) is None
    assert stop_price(100, 98, 8, skipped) is not None
    # Short mirror: cap is 0.5 × 8% = 4 points above 100. Structure 110 tightens to 104.
    short_wide = stop_price(100, 110, 8, p)
    short_tight = stop_price(100, 102, 8, p)
    assert short_wide is not None and abs(short_wide[0] - 104.0) < 1e-9 and short_wide[1] is True
    assert short_tight is not None and abs(short_tight[0] - 102.0) < 1e-9 and short_tight[1] is False
    assert stop_price(100, 100, 8, p) is None
    assert stop_price(100, 95, 8, p) is None
    assert _pnl_per_share(100, 96, p) == 4
    assert _risk_per_share(100, 104, p) == 4
    try:
        replace(p, exit_ema=50)
        raise AssertionError("exit EMA 50 should be rejected")
    except ValueError:
        pass
    assert _max_consec([1, -1, -0.2, -1, 0.5]) == 3
    assert _max_consec([1, 1]) == 0
    assert _bin_label(-1.2) == "< -1R"
    assert _bin_label(-1) == "-1 to 0R"
    assert _bin_label(0) == "-1 to 0R"
    assert _bin_label(0.2) == "0 to 1R"
    assert _bin_label(3) == "> 3R"
    assert _max_concurrent(
        [
            {"entry_date": "2024-01-02", "exit_date": "2024-01-05"},
            {"entry_date": "2024-01-04", "exit_date": "2024-01-08"},
        ],
        [],
    ) == 2
    assert len(rules_text(p, "S&P 500 + Nasdaq 100")) == 8
    assert _book_name(long_p) == "Long momentum rally"
    assert _book_name(p) == "Short momentum rally"
    blocked = replace(p, skip_months=(3, 8))
    assert _month_blocked(pd.Timestamp("2024-03-15"), blocked)
    assert _month_blocked(pd.Timestamp("2024-08-01"), blocked)
    assert not _month_blocked(pd.Timestamp("2024-05-02"), blocked)
    assert not _month_blocked(pd.Timestamp("2024-03-15"), p)
    assert _in_zone(0.4, long_p) and _in_zone(-0.2, long_p) and not _in_zone(1.4, long_p)
    assert _in_zone(-0.4, p) and _in_zone(0.2, p) and not _in_zone(-1.4, p)

    # A smooth climb with a last-bar dip should at least run, and any trade
    # it does take has to respect the stop cap.
    n = 160
    close = 100 * np.exp(np.linspace(0, 0.35, n))
    high = close * 1.035
    low = close * 0.965
    opened = np.r_[close[0], close[:-1]]
    volume = np.full(n, 3_000_000.0)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().to_numpy()
    close = close.copy()
    close[-6] = float(ema9[-7]) * 1.003
    high[-6] = close[-6] * 1.02
    low[-6] = min(close[-6] * 0.985, float(ema9[-7]) * 0.995)
    idx = pd.bdate_range("2023-01-02", periods=n)
    frame = pd.DataFrame(
        {"open": opened, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    synthetic, _, snap = run_symbol(frame, "TEST", p)
    _check_stop_caps(synthetic, p)
    if snap["status"] not in {"setup", "extended", "trend", "universe", "warmup"}:
        raise AssertionError(snap["status"])

    empty_stats = summarize([], [], [], p)
    ctx = {
        "as_of": "2024-01-02",
        "generated_at": "2024-01-02T00:00:00",
        "sample": False,
        "universe_label": "S&P 500 + Nasdaq 100",
        "params": p,
        "funnel": {
            "listed": 0,
            "downloaded": 0,
            "pass_liquidity": 0,
            "pass_trend": 0,
            "in_zone": 0,
            "candidates": 0,
            "removed_market_cap": 0,
            "removed_no_stop": 0,
        },
        "stats": empty_stats,
        "curve": [],
        "bins": r_bins([]),
        "candidates": [],
        "extended": [],
        "opens": [],
        "trades": [],
        "exit_reasons": [],
        "errors": 0,
        "equity_section_html": "<p class='muted'>No closed trades yet, so there is no equity curve.</p>",
        "hist_section_html": "<p class='muted'>No closed trades yet, so there is no R distribution.</p>",
        "equity_page_html": "<p class='muted'>No closed trades yet, so there is no equity curve.</p>",
        "hist_page_html": "<p class='muted'>No closed trades yet, so there is no R distribution.</p>",
    }
    ctx["section_html"] = render_dashboard_section(ctx)
    page = render_page(ctx)
    if _book_name(p) not in page or "id=\"momentum-pullback\"" not in ctx["section_html"]:
        raise AssertionError("page render missed its title or dashboard anchor")
    print("Self-test passed.")


def _dollars_from_millions(value: float) -> float:
    # 50 means $50 million. A figure already in dollars (50000000) is kept.
    if value >= 1000:
        return value
    return value * 1_000_000.0


def params_from_args(args: argparse.Namespace) -> PullbackParams:
    p = PullbackParams()
    updates: dict = {}
    if args.period:
        updates["period"] = args.period
    if args.start:
        updates["start"] = args.start
    if args.side:
        updates["side"] = args.side
    if args.universe:
        updates["universe"] = args.universe
    if args.min_adr is not None:
        updates["min_adr_pct"] = args.min_adr
    if args.min_dollar_vol is not None:
        updates["min_dollar_volume"] = _dollars_from_millions(args.min_dollar_vol)
    if args.proximity is not None:
        updates["proximity_adr_mult"] = args.proximity
    if args.stop_cap is not None:
        updates["stop_adr_cap"] = args.stop_cap
    if args.risk_pct is not None:
        updates["risk_pct"] = args.risk_pct / 100.0
    if args.account is not None:
        updates["account_equity"] = args.account
    if args.max_hold is not None:
        updates["max_hold_days"] = args.max_hold
    if args.skip_months is not None:
        token = args.skip_months.strip().lower()
        if token in {"", "none", "off"}:
            updates["skip_months"] = ()
        else:
            try:
                updates["skip_months"] = tuple(int(part) for part in token.split(",") if part.strip())
            except ValueError as exc:
                raise SystemExit("--skip-months expects numbers such as 3,8, or none") from exc
    if args.exit_ema is not None:
        token = str(args.exit_ema).lower()
        if token in {"trigger", "same", "auto"}:
            updates["exit_ema"] = None
        elif token in {str(p.ema_fast), "fast"}:
            updates["exit_ema"] = p.ema_fast
        elif token in {str(p.ema_mid), "mid"}:
            updates["exit_ema"] = p.ema_mid
        else:
            raise SystemExit(f"--exit-ema must be trigger, {p.ema_fast}, or {p.ema_mid}")
    if args.setup_ema:
        token = args.setup_ema.lower()
        if token in {str(p.ema_fast), "fast"}:
            updates["setup_on_fast"] = True
            updates["setup_on_mid"] = False
        elif token in {str(p.ema_mid), "mid"}:
            updates["setup_on_fast"] = False
            updates["setup_on_mid"] = True
        elif token == "both":
            updates["setup_on_fast"] = True
            updates["setup_on_mid"] = True
        else:
            raise SystemExit("--setup-ema must be 9, 21, or both")
    return replace(p, **updates) if updates else p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Momentum pullback backtest and screener (Yahoo Finance)")
    parser.add_argument("--period", default=None, help="Yahoo history window, default 5y. Ignored for the download when --start is set.")
    parser.add_argument("--start", default=None, help="First entry date, YYYY-MM-DD. Example: 2018-01-01")
    parser.add_argument("--side", default=None, choices=["long", "short"], help="long pullback or the mirrored short (default short)")
    parser.add_argument("--universe", default=None, choices=["sp500", "ndx100", "sp500+ndx100"])
    parser.add_argument("--tickers", default=None, help="Comma-separated symbols instead of the index universe")
    parser.add_argument("--tickers-file", default=None, help="Text file of symbols, one per line")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N names (sample run)")
    parser.add_argument("--min-adr", type=float, default=None, help="Minimum 20-day ADR%% (default 5)")
    parser.add_argument("--min-dollar-vol", type=float, default=None, help="Minimum 20-day dollar volume in millions (default 50)")
    parser.add_argument("--proximity", type=float, default=None, help="Max distance to the EMA, in ADR (default 1)")
    parser.add_argument("--stop-cap", type=float, default=None, help="Stop cap as a multiple of ADR%% (default 0.5)")
    parser.add_argument("--risk-pct", type=float, default=None, help="Equity risk per trade in percent (0.4 means 0.4%%)")
    parser.add_argument("--exit-ema", default=None, help="Trail EMA: trigger (default), 9, or 21")
    parser.add_argument(
        "--skip-months",
        default=None,
        help="Months with no new entries, e.g. 3,8. Default is every month.",
    )
    parser.add_argument("--setup-ema", default=None, help="Which EMA can trigger: 9, 21, or both")
    parser.add_argument("--max-hold", type=int, default=None, help="Time stop in sessions; 0 disables it")
    parser.add_argument("--account", type=float, default=None, help="Starting equity for the curve and share counts")
    parser.add_argument("--refresh", action="store_true", help="Ignore the price cache and download again")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not rebuild the breadth dashboard")
    parser.add_argument("--open", action="store_true", help="Open the HTML page when finished")
    parser.add_argument("--self-test", action="store_true", help="Run the offline checks and exit")
    return parser.parse_args()


def refresh_breadth_dashboard() -> None:
    try:
        from dashboard import build_dashboard

        path = build_dashboard()
        print(f"Updated {path}")
    except Exception as exc:
        print(f"Breadth dashboard was not rebuilt ({exc}). The pullback page is still written.")


def main() -> None:
    args = parse_args()
    self_test()
    if args.self_test:
        return
    params = params_from_args(args)
    explicit = [part.strip() for part in args.tickers.split(",")] if args.tickers else None
    if explicit and args.tickers_file:
        raise SystemExit("Pass either --tickers or --tickers-file, not both.")
    tickers, label, partial = load_universe(params, explicit, args.tickers_file, args.limit)
    write_cache = not partial
    print(f"Universe: {label} ({len(tickers)} symbols)")
    run(params, tickers, label, partial, refresh=args.refresh, write_cache=write_cache)
    if not args.no_dashboard:
        refresh_breadth_dashboard()
    if args.open:
        page = config.MOMENTUM_LONG_HTML if params.side == "long" else config.MOMENTUM_PULLBACK_HTML
        webbrowser.open(Path(page).resolve().as_uri())


if __name__ == "__main__":
    main()

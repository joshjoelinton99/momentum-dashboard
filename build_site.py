"""Build the phone site: momentum books, breadth, and the research pages.

The Futu screener is left out. Breadth here is the Yahoo advance/decline,
moving-average, new-high, and volatility charts.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
LOCAL_OUTPUT = ROOT.parent / "output"

MOMENTUM_LINKS = {
    "market_breadth_dashboard.html": "breadth.html",
    "momentum_long.html": "long.html",
    "momentum_pullback.html": "short.html",
    "analysis_library.html": "library.html",
}


def rewrite_names(text: str) -> str:
    for old, new in MOMENTUM_LINKS.items():
        text = text.replace(old, new)
    return text


def drop_local_files(text: str) -> str:
    return re.sub(
        r'<a href="file://[^"]*"([^>]*)>(.*?)</a>',
        r"<span\1>\2</span>",
        text,
        flags=re.DOTALL,
    )


def as_of(page: str) -> str:
    match = re.search(r"As of ([0-9]{4}-[0-9]{2}-[0-9]{2})", page)
    return match.group(1) if match else ""


def breadth_for_phone(text: str) -> str:
    text = rewrite_names(text)
    text = text.replace(
        '<div class="topnav"><a class="active" href="breadth.html">Breadth</a><a href="long.html">Long momentum rally</a><a href="short.html">Short momentum rally</a><a href="library.html">Analysis library</a></div>',
        '<div class="topnav"><a href="index.html">Home</a><a class="active" href="breadth.html">Breadth</a><a href="long.html">Long</a><a href="short.html">Short</a><a href="library.html">Research</a></div>',
    )
    text = re.sub(
        r"Weekly screener: \d+ runs\. Strongest Screener Stocks \(as of [^)]+\)\.",
        "The Futu screener is not on this page.",
        text,
    )
    start = text.find('<div class="section-title">Participation')
    if start != -1:
        text = (
            text[:start]
            + '<p class="section-desc">Long and short books, and the rest of the research, are on the home page. The Futu screener stays on the computer.</p>\n'
            + "  </div>\n</body>\n</html>\n"
        )
    return text


def index_html(date: str) -> str:
    when = date or "the latest session"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Research dashboard</title>
  <style>
    :root {{
      --bg: #f3f4f6;
      --card: #ffffff;
      --border: #d1d5db;
      --text: #1f2937;
      --muted: #6b7280;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{ max-width: 720px; margin: 0 auto; padding: 24px 16px 48px; }}
    h1 {{ margin: 0 0 6px; font-size: 1.6rem; }}
    p {{ margin: 0 0 18px; color: var(--muted); line-height: 1.45; }}
    a.card {{
      display: block;
      background: var(--card);
      border: 1px solid var(--border);
      text-decoration: none;
      color: inherit;
      padding: 16px 16px 14px;
      margin: 0 0 12px;
    }}
    a.card strong {{ display: block; font-size: 1.15rem; margin-bottom: 4px; }}
    a.card span {{ color: var(--muted); font-size: 0.92rem; }}
    a.long {{ border-left: 4px solid #15803d; }}
    a.short {{ border-left: 4px solid #b91c1c; }}
    a.breadth {{ border-left: 4px solid #4a7ab5; }}
    a.library {{ border-left: 4px solid #1f2937; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Research dashboard</h1>
    <p>Momentum prices through {when}. Breadth and the research pages are the latest copies from this computer. The Futu screener is not here.</p>
    <a class="card long" href="long.html"><strong>Long momentum rally</strong><span>Equity curve, monthly map, and candidates</span></a>
    <a class="card short" href="short.html"><strong>Short momentum rally</strong><span>Equity curve, monthly map, and candidates</span></a>
    <a class="card breadth" href="breadth.html"><strong>Market breadth</strong><span>Advance/decline, moving averages, new highs, and volatility</span></a>
    <a class="card library" href="library.html"><strong>Research library</strong><span>Studies and the Hong Kong dividend pages</span></a>
    <a class="card library" href="reports/crash-rebound-double-fade.html"><strong>Crash → rebound → double fade</strong><span>Four-day path after crash, rip, then two down days</span></a>
    <a class="card library" href="reports/crash-rebound-double-fade.html"><strong>Crash → rebound → double fade</strong><span>Four-day path after crash, rip, then two down days</span></a>
    <a class="card library" href="reports/crash-rebound-fade.html"><strong>Crash → rebound → fade</strong><span>What usually follows a crash, rip, then next-day drop</span></a>
    <a class="card library" href="reports/crash-then-rebound.html"><strong>Crash then huge rebound</strong><span>What usually follows a one-day crash and next-day rip</span></a>
    <a class="card library" href="reports/ma-extension.html"><strong>MA extension — how wide / what next</strong><span>NDX, SPX, SOXX distance below moving averages and forward returns</span></a>
    <a class="card library" href="breakout_dashboard.html"><strong>Tight breakout journal</strong><span>Logged breakouts and how they followed through</span></a>
  </div>
</body>
</html>
"""


def write_momentum(output: Path) -> str:
    long_page = rewrite_names((output / "momentum_long.html").read_text(encoding="utf-8"))
    short_page = rewrite_names((output / "momentum_pullback.html").read_text(encoding="utf-8"))
    (SITE / "long.html").write_text(long_page, encoding="utf-8")
    (SITE / "short.html").write_text(short_page, encoding="utf-8")
    return as_of(long_page)


def copy_static() -> None:
    if not LOCAL_OUTPUT.is_dir():
        return
    library = LOCAL_OUTPUT / "analysis_library.html"
    if library.exists():
        text = drop_local_files(library.read_text(encoding="utf-8"))
        text = text.replace("market_breadth_dashboard.html", "breadth.html")
        text = text.replace("momentum_pullback.html", "short.html")
        text = text.replace("momentum_long.html", "long.html")
        (SITE / "library.html").write_text(text, encoding="utf-8")
    breadth = LOCAL_OUTPUT / "market_breadth_dashboard.html"
    if breadth.exists():
        (SITE / "breadth.html").write_text(
            breadth_for_phone(breadth.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    breakout = LOCAL_OUTPUT / "breakout_dashboard.html"
    if breakout.exists():
        (SITE / "breakout_dashboard.html").write_text(
            breakout.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    reports = LOCAL_OUTPUT / "reports"
    if reports.is_dir():
        dest = SITE / "reports"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(reports, dest)
        for html in dest.glob("*.html"):
            text = html.read_text(encoding="utf-8")
            text = text.replace("../analysis_library.html", "../library.html")
            html.write_text(text, encoding="utf-8")


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    if (LOCAL_OUTPUT / "momentum_long.html").exists():
        output = LOCAL_OUTPUT
    else:
        output = ROOT / "output"
    date = ""
    if (output / "momentum_long.html").exists():
        date = write_momentum(output)
    copy_static()
    (SITE / "index.html").write_text(index_html(date), encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'}")


if __name__ == "__main__":
    main()

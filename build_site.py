"""Turn the backtest pages into the phone site: index, long, and short."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
REPLACEMENTS = {
    "market_breadth_dashboard.html": "index.html",
    "momentum_long.html": "long.html",
    "momentum_pullback.html": "short.html",
    "analysis_library.html": "index.html",
}


def rewrite(name: str) -> str:
    text = (ROOT / "output" / name).read_text(encoding="utf-8")
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def as_of(page: str) -> str:
    match = re.search(r"As of ([0-9]{4}-[0-9]{2}-[0-9]{2})", page)
    return match.group(1) if match else ""


def index_html(date: str) -> str:
    when = date or "the latest session"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Momentum dashboard</title>
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
  </style>
</head>
<body>
  <div class="page">
    <h1>Momentum dashboard</h1>
    <p>Prices through {when}. Each page has the equity curve, the monthly gain and loss map, and the current names. This updates after the US close.</p>
    <a class="card long" href="long.html"><strong>Long momentum rally</strong><span>Pullback book, equity curve, and candidates</span></a>
    <a class="card short" href="short.html"><strong>Short momentum rally</strong><span>Rally book, equity curve, and candidates</span></a>
  </div>
</body>
</html>
"""


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    long_page = rewrite("momentum_long.html")
    short_page = rewrite("momentum_pullback.html")
    (SITE / "long.html").write_text(long_page, encoding="utf-8")
    (SITE / "short.html").write_text(short_page, encoding="utf-8")
    (SITE / "index.html").write_text(index_html(as_of(long_page)), encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'}")


if __name__ == "__main__":
    main()

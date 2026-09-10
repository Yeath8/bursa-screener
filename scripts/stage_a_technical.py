"""
stage_a_technical.py

Milestone 4 — Stage A of the screening funnel: checks criteria 2 (volume
surge) and 3 (momentum) for every ticker in data/tickers_universe.csv,
using ONE batched price/volume download (not one call per ticker — this
is what keeps 900 tickers fast and avoids the rate-limiting we already
hit once in Milestone 3).

Locked formulas (confirmed earlier in this project):
    Criterion 2: today's volume >= 2 x (20-trading-day average volume,
                 the 20 days BEFORE today, not including today)
    Criterion 3: today's close > close 7 TRADING days ago

Only tickers passing BOTH get written to data/stage_a_shortlist.csv —
that shortlist is what Stage B (fundamentals) will run against, so we
never re-fetch fundamentals for tickers that were never going to
qualify anyway.

Usage:
    python scripts/stage_a_technical.py
    python scripts/stage_a_technical.py --tickers 1155.KL,1295.KL
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKERS_FILE = REPO_ROOT / "data" / "tickers_universe.csv"
OUTPUT_FILE = REPO_ROOT / "data" / "stage_a_shortlist.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage_a_technical")

VOLUME_SMA_DAYS = 20
MOMENTUM_LOOKBACK_TRADING_DAYS = 7
VOLUME_MULTIPLIER = 2.0

# Need at least 20 (SMA) + 7 (momentum lookback) + 1 (today) trading days
# of history. "3mo" comfortably covers that even across public holidays.
DOWNLOAD_PERIOD = "3mo"


def load_tickers() -> list[str]:
    if not TICKERS_FILE.exists():
        logger.error(
            "%s not found. Run scripts/update_ticker_universe.py first — "
            "Stage A depends on that file.", TICKERS_FILE,
        )
        return []
    df = pd.read_csv(TICKERS_FILE)
    return df["ticker"].dropna().tolist()


def download_batch(tickers: list[str]) -> pd.DataFrame:
    logger.info("Downloading %d days of price/volume history for %d ticker(s)...",
                90, len(tickers))
    return yf.download(
        tickers=tickers,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=True,
    )


def evaluate_ticker(data: pd.DataFrame, ticker: str, batch_width: int) -> dict | None:
    """
    Returns a result dict, or None if there isn't enough history to evaluate
    this ticker at all (too new a listing, long trading halt, etc.).
    """
    try:
        series = data if batch_width == 1 else data[ticker]
    except KeyError:
        return None

    closes = series["Close"].dropna()
    volumes = series["Volume"].dropna()

    min_rows_needed = VOLUME_SMA_DAYS + MOMENTUM_LOOKBACK_TRADING_DAYS + 1
    if len(closes) < min_rows_needed or len(volumes) < min_rows_needed:
        return None

    today_close = float(closes.iloc[-1])
    close_7d_ago = float(closes.iloc[-(MOMENTUM_LOOKBACK_TRADING_DAYS + 1)])

    today_volume = float(volumes.iloc[-1])
    # 20 trading days BEFORE today — excludes today itself.
    prior_20_volumes = volumes.iloc[-(VOLUME_SMA_DAYS + 1):-1]
    sma20_volume = float(prior_20_volumes.mean())

    passed_momentum = today_close > close_7d_ago
    passed_volume = sma20_volume > 0 and today_volume >= VOLUME_MULTIPLIER * sma20_volume

    return {
        "ticker": ticker,
        "today_close": round(today_close, 4),
        "close_7d_ago": round(close_7d_ago, 4),
        "momentum_pct": round((today_close / close_7d_ago - 1) * 100, 2) if close_7d_ago else None,
        "passed_momentum": passed_momentum,
        "today_volume": int(today_volume),
        "sma20_volume": round(sma20_volume, 1),
        "volume_ratio": round(today_volume / sma20_volume, 2) if sma20_volume else None,
        "passed_volume": passed_volume,
        "passed_stage_a": passed_momentum and passed_volume,
    }


def run_stage_a(tickers: list[str]) -> list[dict]:
    data = download_batch(tickers)
    results = []
    for ticker in tickers:
        result = evaluate_ticker(data, ticker, len(tickers))
        if result is None:
            logger.info("%s: skipped (not enough trading history)", ticker)
            continue
        results.append(result)
    return results


def write_results(results: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checked_on = datetime.now(timezone.utc).isoformat(timespec="seconds")

    shortlisted = [r for r in results if r["passed_stage_a"]]

    fieldnames = [
        "ticker", "today_close", "close_7d_ago", "momentum_pct", "passed_momentum",
        "today_volume", "sma20_volume", "volume_ratio", "passed_volume",
        "passed_stage_a", "checked_on",
    ]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "checked_on": checked_on})

    logger.info(
        "Evaluated %d ticker(s), %d passed both Stage A criteria. Written to %s",
        len(results), len(shortlisted), OUTPUT_FILE,
    )


def print_summary(results: list[dict]) -> None:
    shortlisted = [r for r in results if r["passed_stage_a"]]
    print(f"\n{len(shortlisted)} of {len(results)} ticker(s) passed Stage A (volume surge + momentum):\n")
    for r in shortlisted:
        print(f"  {r['ticker']:<10} momentum {r['momentum_pct']:+.2f}%   "
              f"volume {r['volume_ratio']}x the 20-day average")
    print(f"\nFull detail for every ticker (including ones that didn't pass) is in:\n  {OUTPUT_FILE}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated tickers to check instead of the full universe file, e.g. 1155.KL,1295.KL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else load_tickers()

    if not tickers:
        return 1

    results = run_stage_a(tickers)
    write_results(results)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

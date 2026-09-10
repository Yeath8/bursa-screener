"""
data_quality_spike.py

Milestone 3: cross-checks yfinance fundamentals (ROE, market cap) against
KLSE Screener for a small, diverse sample of tickers, so we know how much
to trust yfinance before Stage B depends on it for all ~900 tickers.

DATA SOURCE NOTE: KLSE Screener exposes a form-POST endpoint at
    https://www.klsescreener.com/v2/screener/quote_results
that returns an HTML table with price, EPS, PE, DY, ROE, NTA, PTBV, and
market cap for the whole Bursa universe in one request. This isn't a
guess — it's the same endpoint an open-source Go scraper
(github.com/kokweikhong/klsescreener-scraper) and a commercial Apify
actor both use in production, and the table's exact column order below
mirrors that Go client's parser. It has NOT been exercised against the
live site from this environment (no network access to klsescreener.com
from the sandbox this was written in) — run it for real in your
Codespace and see the troubleshooting notes at the bottom of this file
if the request comes back empty or blocked.

Usage:
    python scripts/data_quality_spike.py
    python scripts/data_quality_spike.py --tickers 1155.KL,1295.KL,7106.KL
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message

# Cloud IPs (like Codespaces') get rate-limited by Yahoo faster than a
# residential IP would — confirmed in testing (real "Too Many Requests"
# errors, not a one-off). Space requests out and retry with backoff
# instead of hammering the endpoint.
YFINANCE_DELAY_SECONDS = 3

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "data" / "run_logs" / "data_quality_spike_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_quality_spike")

# A deliberately mixed sample: large-cap banks, a plantation/industrial
# name, and a couple of smaller/thinner-traded counters — the kind of
# spread where fundamentals coverage is most likely to break down.
DEFAULT_SAMPLE = [
    "1155.KL",  # Malayan Banking (large-cap, should be well-covered everywhere)
    "1295.KL",  # Public Bank (large-cap)
    "5347.KL",  # Tenaga Nasional (utility, large-cap)
    "1961.KL",  # IOI Corporation (plantation)
    "7106.KL",  # VS Industry (mid-cap industrial)
    "0138.KL",  # smaller/thinner-traded name — good stress test for gaps
]

KLSE_QUOTE_URL = "https://www.klsescreener.com/v2/screener/quote_results"

# Column order confirmed against the LIVE site via --debug-columns on
# 2026-09-10 (18 raw cells; last one is always empty and ignored). This
# has 3 more fields than the Go reference implementation assumed — a
# separate point-change column, a combined category string, and a
# trailing stock-tags column — which shifted "dy"/"roe"/"ptbv" one
# position later than originally mapped. Confirmed against known-good
# values: Maybank ROE ~11.25% (not the DY figure at the old position)
# and market cap ~RM126,520M (not the PTBV figure at the old position).
KLSE_COLUMNS = [
    "short_name", "code", "market_category", "price", "change_abs",
    "changes_pct", "52w_range", "volume", "eps", "dps", "nta", "pe", "dy",
    "roe", "ptbv", "market_cap_rm_millions", "tags",
]


@dataclass
class ComparisonRow:
    ticker: str
    company_name: str = ""
    yfinance_roe_pct: float | None = None
    klse_roe_pct: float | None = None
    yfinance_market_cap_rm: float | None = None
    klse_market_cap_rm: float | None = None
    yfinance_latest_q_net_profit: float | None = None
    yfinance_prior_year_q_net_profit: float | None = None
    notes: str = ""


def _bare_code(ticker: str) -> str:
    """'1155.KL' -> '1155'"""
    return ticker.split(".")[0]


def debug_print_raw_columns(codes: list[str]) -> None:
    """Prints every raw cell in the first matching row, indexed, so we can
    verify (or fix) KLSE_COLUMNS against what the live site actually returns."""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
    }
    payload = {"getquote": "1", "stock_tags": ",".join(codes)}
    resp = requests.post(KLSE_QUOTE_URL, data=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tbody tr.list")

    if not rows:
        print("No rows found at all — the request itself may be failing silently.")
        return

    print(f"Found {len(rows)} row(s). Raw cells for the first row:\n")
    cells = rows[0].find_all("td")
    for i, cell in enumerate(cells):
        print(f"  [{i}] {cell.get_text(strip=True)!r}")
    print(f"\nCurrent KLSE_COLUMNS assumes {len(KLSE_COLUMNS)} columns in this order:")
    print(" ", KLSE_COLUMNS)


def fetch_klse_screener_quotes(codes: list[str]) -> dict[str, dict]:
    """
    POSTs to KLSE Screener's quote_results endpoint, filtered to the given
    bare stock codes (e.g. ['1155', '1295']), and parses the returned HTML
    table into a dict keyed by code.
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
    }
    payload = {
        "getquote": "1",
        "stock_tags": ",".join(codes),
    }

    try:
        resp = requests.post(KLSE_QUOTE_URL, data=payload, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("KLSE Screener request failed.")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tbody tr.list")

    if not rows:
        logger.warning(
            "KLSE Screener returned no rows matching 'tbody tr.list'. "
            "Either the tickers weren't found, or the page structure has "
            "changed since this script was written — see troubleshooting "
            "notes at the bottom of this file."
        )
        return {}

    results: dict[str, dict] = {}
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < len(KLSE_COLUMNS):
            continue

        record = {}
        for i, col_name in enumerate(KLSE_COLUMNS):
            record[col_name] = cells[i].get_text(strip=True)

        # "[s]" marks Shariah-compliant counters in the raw table — strip it
        # from the display name (mirrors the reference Go scraper's parsing).
        record["short_name"] = record["short_name"].replace("[s]", "").strip()

        code = record["code"]
        results[code] = record

    return results


def parse_klse_roe(raw: str) -> float | None:
    try:
        return float(raw.replace("%", "").replace(",", ""))
    except (ValueError, AttributeError):
        return None


def parse_klse_market_cap_rm(raw: str) -> float | None:
    """KLSE Screener reports market cap in RM millions — convert to RM."""
    try:
        return float(raw.replace(",", "")) * 1_000_000
    except (ValueError, AttributeError):
        return None


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=3, min=3, max=30),
    retry=retry_if_exception_message(match=r".*(Too Many Requests|429|rate limit).*"),
    reraise=True,
)
def _get_info(t: "yf.Ticker") -> dict:
    return t.info


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=3, min=3, max=30),
    retry=retry_if_exception_message(match=r".*(Too Many Requests|429|rate limit).*"),
    reraise=True,
)
def _get_quarterly_income(t: "yf.Ticker"):
    return t.quarterly_income_stmt


def fetch_yfinance_fundamentals(ticker: str) -> dict:
    """
    Best-effort pull of ROE, market cap, and quarterly net profit via yfinance.
    Retries with exponential backoff specifically on rate-limit errors —
    confirmed necessary: cloud IPs (Codespaces included) get throttled by
    Yahoo faster than this script originally accounted for.
    """
    t = yf.Ticker(ticker)
    out = {
        "roe_pct": None,
        "market_cap_rm": None,
        "latest_q_net_profit": None,
        "prior_year_q_net_profit": None,
        "notes": "",
    }

    try:
        info = _get_info(t)
        roe = info.get("returnOnEquity")
        out["roe_pct"] = round(roe * 100, 2) if roe is not None else None
        out["market_cap_rm"] = info.get("marketCap")
    except Exception as exc:
        out["notes"] += f"info fetch failed after retries: {exc}; "

    time.sleep(YFINANCE_DELAY_SECONDS)

    try:
        q_income = _get_quarterly_income(t)
        if q_income is not None and "Net Income" in q_income.index and q_income.shape[1] > 0:
            out["latest_q_net_profit"] = float(q_income.loc["Net Income"].iloc[0])
            if q_income.shape[1] > 3:
                out["prior_year_q_net_profit"] = float(q_income.loc["Net Income"].iloc[3])
            else:
                out["notes"] += "fewer than 4 quarters of history available; "
        else:
            out["notes"] += "no quarterly Net Income row returned; "
    except Exception as exc:
        out["notes"] += f"quarterly income fetch failed after retries: {exc}; "

    return out


def run_spike(tickers: list[str]) -> list[ComparisonRow]:
    codes = [_bare_code(t) for t in tickers]
    logger.info("Fetching KLSE Screener data for %d ticker(s)...", len(codes))
    klse_data = fetch_klse_screener_quotes(codes)

    rows: list[ComparisonRow] = []
    for i, ticker in enumerate(tickers):
        code = _bare_code(ticker)
        logger.info("Fetching yfinance data for %s...", ticker)
        yf_data = fetch_yfinance_fundamentals(ticker)
        if i < len(tickers) - 1:
            time.sleep(YFINANCE_DELAY_SECONDS)
        klse_record = klse_data.get(code)

        row = ComparisonRow(
            ticker=ticker,
            yfinance_roe_pct=yf_data["roe_pct"],
            yfinance_market_cap_rm=yf_data["market_cap_rm"],
            yfinance_latest_q_net_profit=yf_data["latest_q_net_profit"],
            yfinance_prior_year_q_net_profit=yf_data["prior_year_q_net_profit"],
            notes=yf_data["notes"],
        )

        if klse_record:
            row.company_name = klse_record.get("short_name", "")
            row.klse_roe_pct = parse_klse_roe(klse_record.get("roe", ""))
            row.klse_market_cap_rm = parse_klse_market_cap_rm(
                klse_record.get("market_cap_rm_millions", "")
            )
        else:
            row.notes += "no matching KLSE Screener row found; "

        rows.append(row)

    return rows


def write_csv(rows: list[ComparisonRow]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(ComparisonRow)]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames + ["checked_on"])
        checked_on = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for row in rows:
            writer.writerow([getattr(row, name) for name in fieldnames] + [checked_on])

    logger.info("Wrote %d comparison row(s) to %s", len(rows), OUTPUT_FILE)


def print_summary(rows: list[ComparisonRow]) -> None:
    print("\n" + "=" * 78)
    print(f"{'Ticker':<10} {'yf ROE%':>9} {'klse ROE%':>10} {'yf MCap (RM)':>16} {'klse MCap (RM)':>16}")
    print("-" * 78)
    for r in rows:
        yf_mcap = f"{r.yfinance_market_cap_rm:,.0f}" if r.yfinance_market_cap_rm else "—"
        klse_mcap = f"{r.klse_market_cap_rm:,.0f}" if r.klse_market_cap_rm else "—"
        yf_roe = f"{r.yfinance_roe_pct:.2f}" if r.yfinance_roe_pct is not None else "—"
        klse_roe = f"{r.klse_roe_pct:.2f}" if r.klse_roe_pct is not None else "—"
        print(f"{r.ticker:<10} {yf_roe:>9} {klse_roe:>10} {yf_mcap:>16} {klse_mcap:>16}")
    print("=" * 78)
    print(f"Full detail (including quarterly net profit + notes) written to:\n  {OUTPUT_FILE}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to check, e.g. 1155.KL,1295.KL. Defaults to a built-in mixed sample.",
    )
    parser.add_argument(
        "--debug-columns",
        action="store_true",
        help="Print the raw KLSE Screener table cells for one ticker instead of running the full comparison — use this to fix column mapping if results look wrong.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.debug_columns:
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else DEFAULT_SAMPLE
        )
        debug_print_raw_columns([_bare_code(t) for t in tickers[:1]])
        return 0
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else DEFAULT_SAMPLE
    )

    rows = run_spike(tickers)
    write_csv(rows)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ── Troubleshooting (read this if the KLSE Screener side comes back empty) ──
#
# 1. Zero rows / "no rows matching" warning:
#    - Open https://www.klsescreener.com/v2/ in a real browser, open dev
#      tools -> Network tab, use the on-page screener form, and find the
#      actual request to quote_results. Compare its form fields and headers
#      against what this script sends — the site may have added a required
#      field (e.g. a CSRF/session token) since the Go scraper this was
#      based on was last updated.
#    - If so, you'll likely need requests.Session() to first GET the
#      screener page (to pick up cookies) before POSTing.
#
# 2. HTTP error / connection refused:
#    - Check whether GitHub Codespaces' outbound IP is being rate-limited
#      or blocked outright. Try the same request from your own laptop's
#      browser first to confirm the endpoint itself still exists.
#
# 3. Rows come back but columns look shifted:
#    - The site may have added/reordered a column in its results table
#      since this was written. Print len(cells) and the raw text of each
#      cell for one row to re-map KLSE_COLUMNS by hand.

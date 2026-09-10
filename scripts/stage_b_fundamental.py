"""
stage_b_fundamental.py

Milestone 5 — Stage B of the screening funnel: checks criteria 1, 4, 5
(market cap, earnings growth, ROE) for every ticker that already passed
Stage A. Runs on the SHORTLIST only, not the full universe — that's the
whole point of the funnel, and it's what keeps this fast and avoids the
rate-limiting we hit in Milestone 3.

Locked formulas (confirmed earlier in this project):
    Criterion 1: market cap >= RM500,000,000
    Criterion 4: (latest quarter net profit - same quarter prior year)
                 > 0.002 x market cap
    Criterion 5: ROE >= 10%

Data sourcing:
    - yfinance is tried first for all three criteria.
    - If yfinance is missing market cap or ROE specifically, KLSE
      Screener (the endpoint proven working in Milestone 3, with the
      column mapping confirmed against live data) is used as a fallback.
    - Earnings growth (criterion 4) has NO fallback — KLSE Screener's
      table doesn't expose quarterly net profit. If yfinance can't
      supply it, that ticker is marked "insufficient data", never
      silently assumed to pass or fail.

Usage:
    python scripts/stage_b_fundamental.py
    python scripts/stage_b_fundamental.py --tickers 1155.KL,7113.KL
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_A_FILE = REPO_ROOT / "data" / "stage_a_shortlist.csv"
OUTPUT_FILE = REPO_ROOT / "data" / "stage_b_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage_b_fundamental")

MIN_MARKET_CAP_RM = 500_000_000
EARNINGS_GROWTH_THRESHOLD_PCT_OF_MCAP = 0.002
MIN_ROE_PCT = 10.0

YFINANCE_DELAY_SECONDS = 3

KLSE_QUOTE_URL = "https://www.klsescreener.com/v2/screener/quote_results"
# Column order confirmed against the LIVE site in Milestone 3 (2026-09-10).
KLSE_COLUMNS = [
    "short_name", "code", "market_category", "price", "change_abs",
    "changes_pct", "52w_range", "volume", "eps", "dps", "nta", "pe", "dy",
    "roe", "ptbv", "market_cap_rm_millions", "tags",
]


def _bare_code(ticker: str) -> str:
    return ticker.split(".")[0]


def load_shortlist() -> list[str]:
    if not STAGE_A_FILE.exists():
        logger.error("%s not found. Run scripts/stage_a_technical.py first.", STAGE_A_FILE)
        return []
    df = pd.read_csv(STAGE_A_FILE)
    passed = df[df["passed_stage_a"] == True]  # noqa: E712
    return passed["ticker"].tolist()


def fetch_klse_screener_quotes(codes: list[str]) -> dict[str, dict]:
    """Same proven approach from Milestone 3 — one batched request for all codes."""
    if not codes:
        return {}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
    }
    payload = {"getquote": "1", "stock_tags": ",".join(codes)}

    try:
        resp = requests.post(KLSE_QUOTE_URL, data=payload, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("KLSE Screener request failed.")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tbody tr.list")
    results: dict[str, dict] = {}
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < len(KLSE_COLUMNS):
            continue
        record = {KLSE_COLUMNS[i]: cells[i].get_text(strip=True) for i in range(len(KLSE_COLUMNS))}
        record["short_name"] = record["short_name"].replace("[s]", "").strip()
        results[record["code"]] = record
    return results


def parse_klse_roe(raw: str) -> float | None:
    try:
        return float(raw.replace("%", "").replace(",", ""))
    except (ValueError, AttributeError, TypeError):
        return None


def parse_klse_market_cap_rm(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "")) * 1_000_000
    except (ValueError, AttributeError, TypeError):
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
    t = yf.Ticker(ticker)
    out = {
        "roe_pct": None, "market_cap_rm": None,
        "latest_q_net_profit": None, "prior_year_q_net_profit": None,
        "notes": "",
    }
    try:
        info = _get_info(t)
        roe = info.get("returnOnEquity")
        out["roe_pct"] = round(roe * 100, 2) if roe is not None else None
        out["market_cap_rm"] = info.get("marketCap")
    except Exception as exc:
        out["notes"] += f"yfinance info failed: {exc}; "

    time.sleep(YFINANCE_DELAY_SECONDS)

    try:
        q_income = _get_quarterly_income(t)
        if q_income is not None and "Net Income" in q_income.index and q_income.shape[1] > 0:
            out["latest_q_net_profit"] = float(q_income.loc["Net Income"].iloc[0])
            if q_income.shape[1] > 3:
                out["prior_year_q_net_profit"] = float(q_income.loc["Net Income"].iloc[3])
            else:
                out["notes"] += "fewer than 4 quarters of history; "
        else:
            out["notes"] += "no quarterly Net Income row; "
    except Exception as exc:
        out["notes"] += f"yfinance quarterly income failed: {exc}; "

    return out


def evaluate_ticker(ticker: str, klse_data: dict[str, dict]) -> dict:
    code = _bare_code(ticker)
    yf_data = fetch_yfinance_fundamentals(ticker)

    market_cap = yf_data["market_cap_rm"]
    roe_pct = yf_data["roe_pct"]
    source_market_cap = "yfinance"
    source_roe = "yfinance"

    klse_record = klse_data.get(code)

    if market_cap is None and klse_record:
        market_cap = parse_klse_market_cap_rm(klse_record.get("market_cap_rm_millions", ""))
        source_market_cap = "klse_screener_fallback"

    if roe_pct is None and klse_record:
        roe_pct = parse_klse_roe(klse_record.get("roe", ""))
        source_roe = "klse_screener_fallback"

    latest_q = yf_data["latest_q_net_profit"]
    prior_q = yf_data["prior_year_q_net_profit"]

    result = {
        "ticker": ticker,
        "market_cap_rm": market_cap,
        "market_cap_source": source_market_cap if market_cap is not None else "unavailable",
        "roe_pct": roe_pct,
        "roe_source": source_roe if roe_pct is not None else "unavailable",
        "latest_q_net_profit": latest_q,
        "prior_year_q_net_profit": prior_q,
        "earnings_growth_rm": None,
        "notes": yf_data["notes"],
    }

    # Criterion 1: market cap
    result["passed_market_cap"] = (
        market_cap is not None and market_cap >= MIN_MARKET_CAP_RM
    )

    # Criterion 5: ROE
    result["passed_roe"] = roe_pct is not None and roe_pct >= MIN_ROE_PCT

    # Criterion 4: earnings growth — needs market_cap AND both quarterly
    # figures. No fallback source exists for quarterly net profit, so a
    # missing value here means "insufficient data", not "failed".
    if market_cap is not None and latest_q is not None and prior_q is not None:
        growth = latest_q - prior_q
        result["earnings_growth_rm"] = round(growth, 0)
        result["passed_earnings_growth"] = growth > EARNINGS_GROWTH_THRESHOLD_PCT_OF_MCAP * market_cap
    else:
        result["passed_earnings_growth"] = None  # insufficient data — not a pass, not a fail
        result["notes"] += "insufficient data for earnings growth check; "

    criteria = [result["passed_market_cap"], result["passed_roe"], result["passed_earnings_growth"]]
    if None in criteria:
        result["qualified"] = None  # can't determine — missing data somewhere
    else:
        result["qualified"] = all(criteria)

    return result


def run_stage_b(tickers: list[str]) -> list[dict]:
    codes = [_bare_code(t) for t in tickers]
    logger.info("Fetching KLSE Screener fallback data for %d ticker(s)...", len(codes))
    klse_data = fetch_klse_screener_quotes(codes)

    results = []
    for i, ticker in enumerate(tickers):
        logger.info("Evaluating %s...", ticker)
        results.append(evaluate_ticker(ticker, klse_data))
        if i < len(tickers) - 1:
            time.sleep(YFINANCE_DELAY_SECONDS)
    return results


def write_results(results: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checked_on = datetime.now(timezone.utc).isoformat(timespec="seconds")

    fieldnames = [
        "ticker", "market_cap_rm", "market_cap_source", "passed_market_cap",
        "roe_pct", "roe_source", "passed_roe",
        "latest_q_net_profit", "prior_year_q_net_profit", "earnings_growth_rm",
        "passed_earnings_growth", "qualified", "notes", "checked_on",
    ]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "checked_on": checked_on})

    logger.info("Wrote %d result(s) to %s", len(results), OUTPUT_FILE)


def print_summary(results: list[dict]) -> None:
    qualified = [r for r in results if r["qualified"] is True]
    unclear = [r for r in results if r["qualified"] is None]

    print(f"\n{len(qualified)} of {len(results)} ticker(s) FULLY QUALIFY (all 5 criteria):\n")
    for r in qualified:
        print(f"  {r['ticker']:<10} mcap RM{r['market_cap_rm']:,.0f}   "
              f"ROE {r['roe_pct']}%   earnings growth RM{r['earnings_growth_rm']:,.0f}")

    if unclear:
        print(f"\n{len(unclear)} ticker(s) couldn't be fully evaluated (missing data) — see notes column:")
        for r in unclear:
            print(f"  {r['ticker']:<10} {r['notes']}")

    print(f"\nFull detail written to:\n  {OUTPUT_FILE}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated tickers to test instead of reading the Stage A shortlist, e.g. 1155.KL,7113.KL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else load_shortlist()

    if not tickers:
        print("No tickers to evaluate — Stage A shortlist is empty (or --tickers wasn't given).")
        print("Try: python scripts/stage_b_fundamental.py --tickers 1155.KL,7113.KL")
        return 0

    results = run_stage_b(tickers)
    write_results(results)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

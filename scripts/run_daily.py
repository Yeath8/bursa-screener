"""
run_daily.py

Milestone 6 — the full funnel in one command: Stage A (technical) filters
the whole ticker universe down to a shortlist, Stage B (fundamentals)
filters that shortlist down to the final qualifying stocks, and the
result is written to data/results.json — the exact file the Streamlit
dashboard will read (locked storage decision from the project plan).

This deliberately REUSES stage_a_technical.py and stage_b_fundamental.py
rather than reimplementing their logic — both were already tested against
synthetic data with known-correct answers, and against real market data.
Importing them means this script inherits that correctness instead of
risking a second, untested copy of the same formulas.

Usage:
    python scripts/run_daily.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from stage_a_technical import load_tickers, run_stage_a
from stage_b_fundamental import run_stage_b

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "data" / "results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily")


def build_qualified_record(stage_a_row: dict, stage_b_row: dict) -> dict:
    """Combines a ticker's Stage A + Stage B data into one record for the dashboard,
    including plain-language reasons — this is what the UI will show as badges."""
    return {
        "ticker": stage_a_row["ticker"],
        "today_close": stage_a_row["today_close"],
        "momentum_pct": stage_a_row["momentum_pct"],
        "volume_ratio": stage_a_row["volume_ratio"],
        "market_cap_rm": stage_b_row["market_cap_rm"],
        "roe_pct": stage_b_row["roe_pct"],
        "earnings_growth_rm": stage_b_row["earnings_growth_rm"],
        "reasons": [
            "Big, established company",
            "More people are buying than usual",
            "Price has been rising",
            "Company is profitable and growing",
            "Good return on shareholders' money",
        ],
    }


def main() -> int:
    started_at = datetime.now(timezone.utc)
    logger.info("=== Daily screening run started ===")

    universe = load_tickers()
    if not universe:
        logger.error("Ticker universe is empty — nothing to screen. Run update_ticker_universe.py first.")
        return 1

    logger.info("Stage A: checking %d ticker(s) for volume surge + momentum...", len(universe))
    stage_a_results = run_stage_a(universe)
    shortlist = [r for r in stage_a_results if r["passed_stage_a"]]
    logger.info("Stage A shortlist: %d ticker(s) passed.", len(shortlist))

    stage_b_results_by_ticker: dict[str, dict] = {}
    if shortlist:
        shortlist_tickers = [r["ticker"] for r in shortlist]
        logger.info("Stage B: checking %d shortlisted ticker(s) for fundamentals...", len(shortlist_tickers))
        stage_b_results = run_stage_b(shortlist_tickers)
        stage_b_results_by_ticker = {r["ticker"]: r for r in stage_b_results}
    else:
        logger.info("Stage A shortlist is empty — skipping Stage B entirely (nothing to check).")

    qualified_records = []
    for stage_a_row in shortlist:
        ticker = stage_a_row["ticker"]
        stage_b_row = stage_b_results_by_ticker.get(ticker)
        if stage_b_row is not None and stage_b_row["qualified"] is True:
            record = build_qualified_record(stage_a_row, stage_b_row)
            qualified_records.append(record)

    output = {
        "generated_at": started_at.isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "stage_a_shortlist_size": len(shortlist),
        "qualified_count": len(qualified_records),
        "qualified_stocks": qualified_records,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "=== Run complete: %d of %d ticker(s) fully qualify. Written to %s ===",
        len(qualified_records), len(universe), RESULTS_FILE,
    )

    print(f"\n{len(qualified_records)} stock(s) qualify today:\n")
    for r in qualified_records:
        print(f"  {r['ticker']:<10} momentum {r['momentum_pct']:+.2f}%   ROE {r['roe_pct']}%")
    print(f"\nFull results (including counts at each funnel stage) written to:\n  {RESULTS_FILE}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

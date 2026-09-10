"""
update_ticker_universe.py

Builds/refreshes data/tickers_universe.csv — the master list of Bursa
Malaysia (.KL) tickers the daily screener will evaluate.

WHY THIS SCRIPT DOESN'T "JUST SCRAPE BURSA": Bursa Malaysia's own listed-
companies directory blocks bot/scraper traffic, and no other free source
currently offers a clean, reliably-structured, up-to-date bulk export of
all ~900 tickers. Rather than build a scraper against a moving target,
this script uses a SEED + VALIDATE pattern:

    1. Start from a small built-in STARTER_SEED (well-known Main Market
       names) so the script is runnable out of the box.
    2. Optionally merge in a --seed-file you export yourself, once, from
       a browser session on a site like KLSE Screener or a Malaysia stock
       screener (a human browsing a page is not a scraping problem — this
       is the most reliable way to get a genuinely current, complete list).
    3. Validate every candidate ticker against yfinance in small batches.
       Anything that doesn't return real price data is dropped from the
       universe and logged separately — this is what protects Stage A/B
       from silently running on a dead or mistyped ticker.

Usage:
    python scripts/update_ticker_universe.py
    python scripts/update_ticker_universe.py --seed-file data/manual_ticker_export.csv
    python scripts/update_ticker_universe.py --skip-validation   # fast, no network calls
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "tickers_universe.csv"
ISSUES_FILE = DATA_DIR / "run_logs" / "ticker_validation_issues.csv"

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update_ticker_universe")

# ── Batch/validation tuning ──────────────────────────────────────────────
BATCH_SIZE = 40          # tickers per yfinance batch download
BATCH_PAUSE_SECONDS = 2  # polite pause between batches


@dataclass
class TickerCandidate:
    ticker: str   # e.g. "1155.KL"
    name: str
    source: str   # "starter_seed" | "seed_file"


# ── Built-in starter seed ────────────────────────────────────────────────
# NOT exhaustive, NOT guaranteed 100% accurate — treat this as a working
# example so the script runs end-to-end before you supply a real,
# manually-exported seed file covering the full ~900-ticker universe.
# Validation below will drop anything that doesn't actually resolve.
STARTER_SEED: list[tuple[str, str]] = [
    ("1155.KL", "Malayan Banking Berhad"),
    ("1295.KL", "Public Bank Berhad"),
    ("1023.KL", "CIMB Group Holdings Berhad"),
    ("5347.KL", "Tenaga Nasional Berhad"),
    ("5225.KL", "IHH Healthcare Berhad"),
    ("6888.KL", "Axiata Group Berhad"),
    ("6012.KL", "Maxis Berhad"),
    ("4197.KL", "Sime Darby Berhad"),
    ("1961.KL", "IOI Corporation Berhad"),
    ("2445.KL", "Kuala Lumpur Kepong Berhad"),
    ("4707.KL", "Nestle (Malaysia) Berhad"),
    ("7084.KL", "QL Resources Berhad"),
    ("7113.KL", "Top Glove Corporation Berhad"),
    ("3182.KL", "Genting Berhad"),
    ("4715.KL", "Genting Malaysia Berhad"),
    ("5819.KL", "Hong Leong Bank Berhad"),
    ("1082.KL", "Hong Leong Financial Group Berhad"),
    ("1066.KL", "RHB Bank Berhad"),
    ("1015.KL", "AMMB Holdings Berhad"),
    ("4065.KL", "PPB Group Berhad"),
    ("7277.KL", "Dialog Group Berhad"),
    ("5398.KL", "Gamuda Berhad"),
    ("3336.KL", "IJM Corporation Berhad"),
    ("4677.KL", "YTL Corporation Berhad"),
    ("6742.KL", "YTL Power International Berhad"),
    ("4863.KL", "Telekom Malaysia Berhad"),
    ("6947.KL", "CelcomDigi Berhad"),
    ("3816.KL", "MISC Berhad"),
    ("6033.KL", "Petronas Gas Berhad"),
    ("5681.KL", "Petronas Dagangan Berhad"),
    ("8869.KL", "S P Setia Berhad"),
    ("6399.KL", "Astro Malaysia Holdings Berhad"),
]
# A handful of entries were deliberately trimmed here after a test run
# surfaced duplicate stock codes under different company names (e.g. two
# different Petronas entities recalled under the same code). That's a
# real memory-recall error, not a hypothetical one — treat every code in
# this list as unverified until your own --seed-file + validation run
# confirms it, and don't extend this list from memory without checking it.


def load_starter_seed() -> list[TickerCandidate]:
    return [TickerCandidate(t, n, "starter_seed") for t, n in STARTER_SEED]


def load_seed_file(path: Path) -> list[TickerCandidate]:
    """
    Load a manually-exported CSV. Accepts either:
      - a 'ticker' column already in yfinance format (e.g. '1155.KL'), or
      - a 'code' column with the bare Bursa numeric code (e.g. '1155'),
        which gets '.KL' appended automatically.
    An optional 'name' column is used if present.
    """
    if not path.exists():
        logger.warning("Seed file not found at %s — skipping.", path)
        return []

    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    candidates: list[TickerCandidate] = []
    for _, row in df.iterrows():
        if "ticker" in df.columns and row["ticker"].strip():
            ticker = row["ticker"].strip().upper()
            if not ticker.endswith(".KL"):
                ticker = f"{ticker}.KL"
        elif "code" in df.columns and row["code"].strip():
            code = row["code"].strip().upper()
            ticker = code if code.endswith(".KL") else f"{code}.KL"
        else:
            continue

        name = row.get("name", "").strip() or ticker
        candidates.append(TickerCandidate(ticker, name, "seed_file"))

    logger.info("Loaded %d candidates from seed file %s", len(candidates), path)
    return candidates


def dedupe(candidates: list[TickerCandidate]) -> list[TickerCandidate]:
    seen: dict[str, TickerCandidate] = {}
    for c in candidates:
        # Prefer a seed_file entry's name over the starter seed's if both exist
        if c.ticker not in seen or seen[c.ticker].source == "starter_seed":
            seen[c.ticker] = c
    return sorted(seen.values(), key=lambda c: c.ticker)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _download_batch(tickers: list[str]) -> pd.DataFrame:
    """Batched price pull — one network round-trip per batch, not per ticker."""
    return yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=True,
    )


def validate_candidates(
    candidates: list[TickerCandidate],
) -> tuple[list[TickerCandidate], list[TickerCandidate]]:
    """
    Confirms each candidate actually resolves to real price data via
    yfinance. Returns (valid, invalid).
    """
    valid: list[TickerCandidate] = []
    invalid: list[TickerCandidate] = []

    batches = [
        candidates[i : i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)
    ]
    logger.info(
        "Validating %d tickers across %d batches of up to %d...",
        len(candidates), len(batches), BATCH_SIZE,
    )

    for batch_num, batch in enumerate(batches, start=1):
        tickers = [c.ticker for c in batch]
        logger.info("Batch %d/%d: %s", batch_num, len(batches), ", ".join(tickers))

        try:
            data = _download_batch(tickers)
        except Exception:
            logger.exception("Batch %d failed after retries — marking all as invalid.", batch_num)
            invalid.extend(batch)
            continue

        for candidate in batch:
            has_data = _batch_result_has_data(data, candidate.ticker, len(tickers))
            (valid if has_data else invalid).append(candidate)

        if batch_num < len(batches):
            time.sleep(BATCH_PAUSE_SECONDS)

    return valid, invalid


def _batch_result_has_data(data: pd.DataFrame, ticker: str, batch_width: int) -> bool:
    """Handle both yfinance shapes: single-ticker (flat) vs multi-ticker (MultiIndex columns)."""
    try:
        if batch_width == 1:
            close = data["Close"] if "Close" in data.columns else data.get(ticker, {}).get("Close")
        else:
            close = data[ticker]["Close"]
        return close is not None and close.notna().any()
    except (KeyError, AttributeError, TypeError):
        return False


def write_universe_csv(candidates: list[TickerCandidate]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checked_on = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "name", "source", "validated_on"])
        for c in candidates:
            writer.writerow([c.ticker, c.name, c.source, checked_on])

    logger.info("Wrote %d validated tickers to %s", len(candidates), OUTPUT_FILE)


def write_issues_csv(candidates: list[TickerCandidate]) -> None:
    ISSUES_FILE.parent.mkdir(parents=True, exist_ok=True)
    checked_on = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with ISSUES_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "name", "source", "checked_on", "reason"])
        for c in candidates:
            writer.writerow([c.ticker, c.name, c.source, checked_on, "no_price_data_returned"])

    if candidates:
        logger.warning(
            "%d ticker(s) failed validation — see %s for the list.",
            len(candidates), ISSUES_FILE,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Path to a manually-exported CSV with 'ticker' or 'code' (+ optional 'name') columns.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the yfinance validation pass (fast, no network calls — useful for a quick local check).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    candidates = load_starter_seed()
    if args.seed_file:
        candidates += load_seed_file(args.seed_file)
    candidates = dedupe(candidates)

    if not candidates:
        logger.error("No candidate tickers to process. Provide --seed-file or check STARTER_SEED.")
        return 1

    logger.info("%d unique candidate ticker(s) before validation.", len(candidates))

    if args.skip_validation:
        logger.warning("--skip-validation set: writing candidates WITHOUT confirming they resolve.")
        write_universe_csv(candidates)
        return 0

    valid, invalid = validate_candidates(candidates)
    write_universe_csv(valid)
    write_issues_csv(invalid)

    logger.info("Done. %d valid / %d invalid out of %d candidates.", len(valid), len(invalid), len(candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
streamlit_app/app.py

Milestone 8 — the MVP dashboard. Reads data/results.json (the file
run_daily.py writes, and the exact file GitHub Actions now commits back
to the repo automatically every weekday) and displays it.

Deliberately plain for now — no high-contrast/elderly-friendly styling
yet, that's Milestone 9. This milestone only needs to prove the data
flow works: pipeline -> results.json -> dashboard, all the way through.

Run locally with:
    streamlit run streamlit_app/app.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "data" / "results.json"


def load_results() -> dict | None:
    if not RESULTS_FILE.exists():
        return None
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_market_cap(rm: float | None) -> str:
    if rm is None:
        return "—"
    if rm >= 1_000_000_000:
        return f"RM{rm / 1_000_000_000:.2f}B"
    return f"RM{rm / 1_000_000:.1f}M"


def format_generated_at(iso_string: str) -> str:
    dt = datetime.fromisoformat(iso_string)
    now = datetime.now(timezone.utc)
    age = now - dt
    hours = age.total_seconds() / 3600
    freshness = (
        f" ({hours:.0f}h ago)"
        if hours < 48
        else " (more than 2 days ago — may be stale)"
    )
    return dt.strftime("%Y-%m-%d %H:%M UTC") + freshness


def main() -> None:
    st.set_page_config(page_title="Bursa Malaysia Daily Screener", layout="centered")
    st.title("Bursa Malaysia Daily Screener")

    data = load_results()

    if data is None:
        st.warning(
            "No results file found yet. This means the daily pipeline "
            "hasn't run successfully at least once — check the GitHub "
            "Actions tab for the 'Daily Bursa Screener' workflow."
        )
        return

    st.caption(f"Last updated: {format_generated_at(data['generated_at'])}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Stocks screened", data["universe_size"])
    col2.metric("Passed initial filter", data["stage_a_shortlist_size"])
    col3.metric("Fully qualify today", data["qualified_count"])

    st.divider()

    qualified = data["qualified_stocks"]

    if not qualified:
        st.info(
            "No stocks fully qualify today. This is a normal outcome — "
            "the criteria are intentionally strict, so it's common to "
            "have zero qualifying stocks on any given day, especially "
            "while the ticker list is still small."
        )
        return

    for stock in qualified:
        with st.container(border=True):
            st.subheader(stock["ticker"])
            st.write(
                f"**RM {stock['today_close']}** &nbsp;&nbsp; "
                f"momentum: **{stock['momentum_pct']:+.2f}%** &nbsp;&nbsp; "
                f"volume: **{stock['volume_ratio']}x** the 20-day average"
            )
            st.write(
                f"Market cap: {format_market_cap(stock['market_cap_rm'])} &nbsp;&nbsp; "
                f"ROE: **{stock['roe_pct']}%**"
            )
            for reason in stock["reasons"]:
                st.write(f"✅ {reason}")

    st.divider()
    st.caption(
        "This tool is for informational purposes only and is not financial "
        "advice. Always do your own research before buying or selling any stock."
    )


if __name__ == "__main__":
    main()

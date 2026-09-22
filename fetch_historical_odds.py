"""
fetch_historical_odds.py -- Pulls REAL point-in-time historical odds (not
closing lines) for every week of the 2024 and 2025 NFL regular seasons, via
The Odds API's historical odds endpoint. This is what makes a genuine
profitability backtest possible: game_lines_model.py's existing validation
used CLOSING lines (nflverse), which proves the model is calibrated but
CANNOT prove it would have been profitable, since a closing line already
bakes in everything that moved the market by kickoff -- you could never
actually have bet at that price. This script pulls the line that was
ACTUALLY LIVE a few days before kickoff instead.

SNAPSHOT TIMING: one snapshot per week, taken at the Tuesday following the
prior week's games (the start of a new NFL "week" by convention -- lines
for that week's slate are typically posted by then). This is a real,
principled choice, not arbitrary: it's roughly when a bettor could first
realistically shop that week's numbers, well before the sharp
line-movement that happens Thursday-Sunday.

COST -- REAL, NOT HYPOTHETICAL: the historical odds endpoint costs 10x a
live call (10 x markets x regions per snapshot, per The Odds API's own
documented pricing). Fetching h2h+spreads+totals (3 markets, us region) for
36 weeks (18 weeks x 2 seasons) = 36 x 10 x 3 x 1 = 1,080 credits. Check
your plan's quota before running this -- it is not reversible, and this
script does not retry on a budget-exhausted 401/402, it just fails and
tells you.

NOT YET SMOKE-TESTED AGAINST A LIVE KEY (same caveat every other odds
integration in this repo shipped with) -- built directly against The Odds
API's documented historical-odds response shape, not tested against a real
response, since this sandbox has no network path to api.the-odds-api.com.
Worth watching the first real run closely.

Usage:
    ODDS_API_KEY=... python fetch_historical_odds.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
MARKETS = "h2h,spreads,totals"
SEASONS = [2024, 2025]
OUT_PATH = "../data/historical_odds_2024_2025.jsonl"


def week_snapshot_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    """For each (season, week), the Tuesday snapshot timestamp -- the
    Tuesday on or immediately before that week's earliest game. NFL weeks
    always start on Tuesday by convention, so this is a real anchor point,
    not an approximation."""
    schedule = schedule.copy()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    weeks = schedule.groupby(["season", "week"], as_index=False)["gameday"].min()
    weeks = weeks.rename(columns={"gameday": "first_game"})
    # back up to the preceding Tuesday (weekday()==1); if first_game IS a
    # Tuesday, use it as-is
    weeks["days_since_tuesday"] = (weeks["first_game"].dt.weekday - 1) % 7
    weeks["snapshot_date"] = weeks["first_game"] - pd.to_timedelta(weeks["days_since_tuesday"], unit="D")
    # 14:00 UTC (~10am ET) -- arbitrary but fixed and reasonable time of day
    weeks["snapshot_iso"] = weeks["snapshot_date"].dt.strftime("%Y-%m-%dT14:00:00Z")
    return weeks[["season", "week", "snapshot_iso"]]


def fetch_snapshot(snapshot_iso: str, api_key: str) -> list[dict] | None:
    try:
        resp = requests.get(
            f"{BASE_URL}/historical/sports/{SPORT}/odds",
            params={"apiKey": api_key, "regions": "us", "markets": MARKETS,
                    "oddsFormat": "american", "date": snapshot_iso},
            timeout=30,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        print(f"    [odds api] snapshot {snapshot_iso}: used={used} remaining={remaining}")
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        print(f"    WARNING: snapshot {snapshot_iso} failed ({e}) -- skipping this week.")
        return None


def main():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set -- nothing to do.")
        sys.exit(1)

    df = pd.read_csv("games_full.csv", low_memory=False) if os.path.exists("games_full.csv") \
        else pd.read_csv("../games_full.csv", low_memory=False)
    schedule = df[(df["season"].isin(SEASONS)) & (df["game_type"] == "REG")]
    snapshots = week_snapshot_dates(schedule)

    est_cost = len(snapshots) * 10 * len(MARKETS.split(",")) * 1
    print(f"About to fetch {len(snapshots)} weekly snapshots across {SEASONS}.")
    print(f"Estimated cost: {len(snapshots)} x 10 x {len(MARKETS.split(','))} markets x 1 region "
          f"= ~{est_cost} credits. This is NOT reversible once run.")

    out_rows = []
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    for _, row in snapshots.iterrows():
        print(f"  Season {row['season']} Week {row['week']}: requesting snapshot {row['snapshot_iso']}...")
        events = fetch_snapshot(row["snapshot_iso"], api_key)
        if events is None:
            continue
        for ev in events:
            out_rows.append({
                "season": int(row["season"]), "week": int(row["week"]),
                "snapshot_iso": row["snapshot_iso"],
                "event_id": ev.get("id"), "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"), "commence_time": ev.get("commence_time"),
                "bookmakers": ev.get("bookmakers", []),
            })
        time.sleep(0.3)  # light self-throttle

    with open(OUT_PATH, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_rows)} event snapshots to {OUT_PATH}")
    print("Next: run backtest_profitability.py to grade these against real outcomes.")


if __name__ == "__main__":
    main()

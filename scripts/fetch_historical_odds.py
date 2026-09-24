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


def load_schedule() -> pd.DataFrame:
    """Fetches the real nflverse schedule (with final scores) directly --
    this repo doesn't already have a local copy of this file; earlier
    testing of this script used one that existed only in the sandbox it was
    built in, which would have failed the first time this actually ran here."""
    resp = requests.get("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", timeout=30)
    resp.raise_for_status()
    import io
    return pd.read_csv(io.StringIO(resp.text), low_memory=False)


def get_deduplicated_real_games() -> pd.DataFrame:
    """Backward-compatible wrapper -- the original 2024-2025 case. See
    get_deduplicated_real_games_for()'s docstring for what this actually does."""
    return get_deduplicated_real_games_for(OUT_PATH, SEASONS)


def get_deduplicated_real_games_for(hist_path: str, seasons: list[int]) -> pd.DataFrame:
    """
    Same dedup/true-week-recovery logic as backtest_profitability.py's
    load_historical_odds() (see that function's docstring for the full
    story of the bug this fixes -- the rolling-window historical odds
    endpoint duplicating games under multiple weeks), but keeps `event_id`
    and each game's own correct target `snapshot_iso`, which that function
    drops. Needed here so a player-props fetch (or, later, the props MODEL's
    implied-total feature) can reuse the SAME real event_ids already paid
    for in a game-lines fetch, rather than paying for a second events
    lookup. Generalized to take a path + season list so it works for
    EITHER the 2024-2025 file or a training-years file (e.g. 2021-2023),
    not just the original hardcoded case.
    """
    rows = [json.loads(l) for l in open(hist_path)]
    raw = pd.DataFrame(rows)
    raw = raw.drop(columns=["season", "week"])  # the original fetch-loop tags -- known buggy, see backtest_profitability.py's load_historical_odds docstring; the schedule merge below supplies the correct ones
    raw["commence_time"] = pd.to_datetime(raw["commence_time"])
    raw["game_date"] = raw["commence_time"].dt.date
    raw["home_team"] = raw["home_team"].map(lambda n: TEAM_NAME_TO_ABBR.get(n, n))
    raw["away_team"] = raw["away_team"].map(lambda n: TEAM_NAME_TO_ABBR.get(n, n))

    sched = load_schedule()
    sched = sched[(sched["season"].isin(seasons)) & (sched["game_type"] == "REG")].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"]).dt.date
    sched_key = sched[["season", "week", "gameday", "home_team", "away_team"]].rename(columns={"gameday": "game_date"})
    df = raw.merge(sched_key, on=["game_date", "home_team", "away_team"], how="inner")

    weeks = week_snapshot_dates(sched)
    df = df.merge(weeks, on=["season", "week"], how="left", suffixes=("", "_target"))
    df["target_dt"] = pd.to_datetime(df["snapshot_iso_target"])
    df["fetched_dt"] = pd.to_datetime(df["snapshot_iso"])
    df["is_on_or_before_target"] = df["fetched_dt"] <= df["target_dt"]
    df = df.sort_values(["season", "week", "home_team", "is_on_or_before_target", "fetched_dt"],
                         ascending=[True, True, True, False, True])
    df = df.drop_duplicates(subset=["season", "week", "home_team", "away_team"], keep="first")
    return df[["season", "week", "home_team", "away_team", "event_id", "snapshot_iso_target"]].reset_index(drop=True)


TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=SEASONS,
                         help="Seasons to fetch, e.g. --seasons 2021 2022 2023. Defaults to 2024/2025.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output path. Defaults to ../data/historical_odds_{first}_{last}.jsonl")
    args = parser.parse_args()
    seasons = args.seasons
    out_path = args.out or f"../data/historical_odds_{min(seasons)}_{max(seasons)}.jsonl"

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set -- nothing to do.")
        sys.exit(1)

    df = load_schedule()
    schedule = df[(df["season"].isin(seasons)) & (df["game_type"] == "REG")]
    snapshots = week_snapshot_dates(schedule)

    est_cost = len(snapshots) * 10 * len(MARKETS.split(",")) * 1
    print(f"About to fetch {len(snapshots)} weekly snapshots across {seasons}.")
    print(f"Estimated cost: {len(snapshots)} x 10 x {len(MARKETS.split(','))} markets x 1 region "
          f"= ~{est_cost} credits. This is NOT reversible once run.")

    out_rows = []
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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

    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_rows)} event snapshots to {out_path}")
    print("Next: run backtest_profitability.py to grade these against real outcomes.")


if __name__ == "__main__":
    main()

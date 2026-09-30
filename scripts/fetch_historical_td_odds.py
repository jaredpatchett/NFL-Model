"""
fetch_historical_td_odds.py -- Real point-in-time anytime-TD odds for the
same 438 real games already recovered by fetch_historical_odds.py +
get_deduplicated_real_games(). Same principle as
fetch_historical_player_props.py, one market instead of three.

REUSES the event_ids already paid for in the game-lines fetch.

COST: per-event historical endpoint, 10 credits x 1 market x 1 region PER
EVENT = 10 credits/game. For 438 games = ~4,380 credits.

NOT YET SMOKE-TESTED AGAINST A LIVE KEY -- built directly against the same
documented historical-event-odds response shape already confirmed working
for fetch_historical_player_props.py.

Usage:
    ODDS_API_KEY=... python fetch_historical_td_odds.py
"""
from __future__ import annotations
import json
import os
import sys
import time

import requests

from fetch_historical_odds import get_deduplicated_real_games

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
MARKETS = "player_anytime_td"
OUT_PATH = "../data/historical_td_odds_2024_2025.jsonl"


def fetch_event_td_odds(event_id: str, snapshot_iso: str, api_key: str) -> dict | None:
    try:
        resp = requests.get(
            f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds",
            params={"apiKey": api_key, "regions": "us", "markets": MARKETS,
                    "oddsFormat": "american", "date": snapshot_iso},
            timeout=30,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        print(f"    [odds api] event {event_id}: used={used} remaining={remaining}")
        data = resp.json()
        return data.get("data", data)
    except Exception as e:
        print(f"    WARNING: event {event_id} failed ({e}) -- skipping this game's TD odds.")
        return None


def main():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set -- nothing to do.")
        sys.exit(1)

    games = get_deduplicated_real_games()
    est_cost = len(games) * 10 * len(MARKETS.split(","))
    print(f"About to fetch anytime-TD odds for {len(games)} real games.")
    print(f"Estimated cost: {len(games)} x 10 x {len(MARKETS.split(','))} market x 1 region "
          f"= ~{est_cost} credits. This is NOT reversible once run.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_rows = []
    for i, row in games.iterrows():
        print(f"  [{i+1}/{len(games)}] {row['away_team']} @ {row['home_team']} "
              f"({row['season']} wk{row['week']}): event {row['event_id']}...")
        ev = fetch_event_td_odds(row["event_id"], row["snapshot_iso_target"], api_key)
        if ev is None:
            continue
        out_rows.append({
            "season": int(row["season"]), "week": int(row["week"]),
            "home_team": row["home_team"], "away_team": row["away_team"],
            "event_id": row["event_id"], "snapshot_iso": row["snapshot_iso_target"],
            "bookmakers": ev.get("bookmakers", []),
        })
        time.sleep(0.2)

    with open(OUT_PATH, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_rows)} games' TD odds to {OUT_PATH}")
    print("Next: run backtest_td_profitability.py to grade these against real outcomes.")


if __name__ == "__main__":
    main()

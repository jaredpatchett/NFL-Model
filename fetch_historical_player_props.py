"""
fetch_historical_player_props.py -- Real point-in-time player prop odds
(rec yards, receptions, rush yards) for the same 438 real games already
recovered by fetch_historical_odds.py + get_deduplicated_real_games(). This
is what makes a genuine Track C (fantasy props) profitability backtest
possible, the same way fetch_historical_odds.py did for Track B.

REUSES the event_ids already paid for in the game-lines fetch instead of
re-querying the events endpoint -- no reason to pay twice for the same
event lookup.

COST -- REAL, NOT HYPOTHETICAL: player props require the PER-EVENT
historical odds endpoint (bulk historical odds only covers featured
markets -- h2h/spreads/totals -- not props), at 10 credits x markets x
regions PER EVENT (not per week). For 438 real games x 3 markets
(player_rush_yds, player_reception_yds, player_receptions) x 1 region =
438 x 30 = ~13,140 credits. Not reversible once run.

NOT YET SMOKE-TESTED AGAINST A LIVE KEY -- built directly against The Odds
API's documented historical-event-odds response shape (same shape as the
live fetch_fantasy_prop_odds() in odds_api.py, which HAS been confirmed
against real live data), but the historical variant itself hasn't been
run for real yet.

Usage:
    ODDS_API_KEY=... python fetch_historical_player_props.py
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
MARKETS = "player_rush_yds,player_reception_yds,player_receptions"
OUT_PATH = "../data/historical_player_props_2024_2025.jsonl"


def fetch_event_props(event_id: str, snapshot_iso: str, api_key: str) -> dict | None:
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
        # historical per-event responses wrap the event under "data"
        return data.get("data", data)
    except Exception as e:
        print(f"    WARNING: event {event_id} failed ({e}) -- skipping this game's props.")
        return None


def main():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set -- nothing to do.")
        sys.exit(1)

    games = get_deduplicated_real_games()
    est_cost = len(games) * 10 * len(MARKETS.split(","))
    print(f"About to fetch player props for {len(games)} real games.")
    print(f"Estimated cost: {len(games)} x 10 x {len(MARKETS.split(','))} markets x 1 region "
          f"= ~{est_cost} credits. This is NOT reversible once run.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_rows = []
    for i, row in games.iterrows():
        print(f"  [{i+1}/{len(games)}] {row['away_team']} @ {row['home_team']} "
              f"({row['season']} wk{row['week']}): event {row['event_id']}...")
        ev = fetch_event_props(row["event_id"], row["snapshot_iso_target"], api_key)
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
    print(f"\nWrote {len(out_rows)} games' player props to {OUT_PATH}")
    print("Next: run backtest_player_props_profitability.py to grade these against real outcomes.")


if __name__ == "__main__":
    main()

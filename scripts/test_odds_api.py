"""
test_odds_api.py — Validates odds_api.py's PARSING logic against realistic
mock responses matching The Odds API's documented schema (see docstrings in
odds_api.py for the source). Does NOT hit the real API -- there's no key
available in this environment to test against. Run this again against the
real API once a key is added (e.g. via `python odds_api.py` manually) to
confirm the live response actually matches what's mocked here; documented
schemas can drift from reality.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from odds_api import fetch_game_odds, fetch_player_td_odds, normalize_name, american_to_implied_prob


def _mock_response(json_data, headers=None):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.headers = headers or {"x-requests-remaining": "487", "x-requests-used": "13"}
    resp.raise_for_status = MagicMock()
    return resp


SAMPLE_GAME_ODDS = [
    {
        "id": "abc123", "sport_key": "americanfootball_nfl", "commence_time": "2026-09-10T00:20:00Z",
        "home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
        "bookmakers": [
            {
                "key": "draftkings", "title": "DraftKings", "last_update": "2026-09-08T12:00:00Z",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -150},
                        {"name": "Baltimore Ravens", "price": 130},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -110, "point": -3.5},
                        {"name": "Baltimore Ravens", "price": -110, "point": 3.5},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": -108, "point": 47.5},
                        {"name": "Under", "price": -112, "point": 47.5},
                    ]},
                ],
            },
            {
                "key": "fanduel", "title": "FanDuel", "last_update": "2026-09-08T12:01:00Z",
                "markets": [
                    # Better price on the home moneyline and away spread -- tests best-price selection.
                    {"key": "h2h", "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -140},
                        {"name": "Baltimore Ravens", "price": 125},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -105, "point": -3.5},
                        {"name": "Baltimore Ravens", "price": -115, "point": 3.5},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": -105, "point": 47.5},
                        {"name": "Under", "price": -115, "point": 47.5},
                    ]},
                ],
            },
        ],
    },
]

SAMPLE_EVENT_ODDS = {
    "id": "abc123", "home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
    "bookmakers": [
        {
            "key": "draftkings", "markets": [
                {"key": "player_anytime_td", "outcomes": [
                    {"name": "Yes", "description": "Patrick Mahomes", "price": 145},
                    {"name": "No", "description": "Patrick Mahomes", "price": -180},
                    {"name": "Yes", "description": "Derrick Henry", "price": -120},
                    {"name": "No", "description": "Derrick Henry", "price": 100},
                ]},
            ],
        },
        {
            "key": "fanduel", "markets": [
                {"key": "player_anytime_td", "outcomes": [
                    # Better price on Mahomes -- tests best-price selection across books.
                    {"name": "Yes", "description": "Patrick Mahomes", "price": 160},
                    {"name": "No", "description": "Patrick Mahomes", "price": -190},
                    {"name": "Yes", "description": "Derrick Henry", "price": -125},
                    {"name": "No", "description": "Derrick Henry", "price": 105},
                ]},
            ],
        },
    ],
}


def main():
    checks = []

    # ---- normalize_name ----
    checks.append(("normalize_name strips suffix", normalize_name("Odell Beckham Jr."), "odell beckham"))
    checks.append(("normalize_name lowercases + strips apostrophe", normalize_name("Ja'Marr Chase"), "jamarr chase"))

    # ---- american_to_implied_prob ----
    checks.append(("implied prob, positive odds", round(american_to_implied_prob(150), 4), round(100/250, 4)))
    checks.append(("implied prob, negative odds", round(american_to_implied_prob(-150), 4), round(150/250, 4)))

    # ---- fetch_game_odds: best price selection + team mapping ----
    with patch("odds_api.requests.get", return_value=_mock_response(SAMPLE_GAME_ODDS)):
        df = fetch_game_odds(api_key="fake_key_for_test")
    checks.append(("fetch_game_odds returns 1 row", len(df) if df is not None else -1, 1))
    row = df.iloc[0]
    checks.append(("team name mapped: home", row["home_team"], "KC"))
    checks.append(("team name mapped: away", row["away_team"], "BAL"))
    checks.append(("best home moneyline (FD -140 beats DK -150)", row["live_home_moneyline"], -140))
    checks.append(("best away moneyline (DK +130 beats FD +125)", row["live_away_moneyline"], 130))
    checks.append(("best away spread price (DK -110 beats FD -115)", row["live_away_spread_price"], -110))

    # ---- fetch_player_td_odds: best price selection + name normalization ----
    events = pd.DataFrame([{"event_id": "abc123", "home_team": "KC", "away_team": "BAL"}])
    with patch("odds_api.requests.get", return_value=_mock_response(SAMPLE_EVENT_ODDS)):
        pdf = fetch_player_td_odds(events, api_key="fake_key_for_test")
    checks.append(("fetch_player_td_odds returns 2 players", len(pdf) if pdf is not None else -1, 2))
    mahomes = pdf[pdf["player_name_raw"] == "Patrick Mahomes"].iloc[0]
    checks.append(("best price for Mahomes (FD +160 beats DK +145)", mahomes["live_anytime_td_price"], 160))
    checks.append(("normalized name for matching", mahomes["player_name_norm"], "patrick mahomes"))

    # ---- Graceful degradation: no key ----
    with patch.dict("os.environ", {}, clear=True):
        result = fetch_game_odds(api_key=None)
    checks.append(("no key -> returns None, does not crash", result, None))

    print(f"{'check':55s} {'got':>20s} {'want':>20s}  ok")
    print("-" * 105)
    all_ok = True
    for name, got, want in checks:
        ok = (got == want)
        all_ok &= ok
        print(f"{name:55s} {str(got):>20s} {str(want):>20s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 105)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

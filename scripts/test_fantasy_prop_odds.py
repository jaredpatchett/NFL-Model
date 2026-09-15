"""
test_fantasy_prop_odds.py -- Validates fetch_fantasy_prop_odds's parsing
logic against a realistic MOCKED response shape (no live API key used or
required -- same approach as the existing test_odds_api.py). This does NOT
prove the real API's response matches this fixture exactly; it proves the
parsing code does what it's supposed to against the documented shape.

Run: python test_fantasy_prop_odds.py
"""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import pandas as pd

import odds_api

# Realistic fixture: two bookmakers, one event, all three markets, with a
# deliberately WORSE second bookmaker to prove best-price selection works,
# and one Under-only outcome to prove partial data doesn't crash anything.
MOCK_RESPONSE = {
    "id": "evt123", "home_team": "Buffalo Bills", "away_team": "Detroit Lions",
    "bookmakers": [
        {
            "key": "draftkings",
            "markets": [
                {"key": "player_reception_yds", "outcomes": [
                    {"name": "Over", "description": "Amon-Ra St. Brown", "price": -110, "point": 75.5},
                    {"name": "Under", "description": "Amon-Ra St. Brown", "price": -110, "point": 75.5},
                ]},
                {"key": "player_receptions", "outcomes": [
                    {"name": "Over", "description": "Amon-Ra St. Brown", "price": -125, "point": 6.5},
                    {"name": "Under", "description": "Amon-Ra St. Brown", "price": 105, "point": 6.5},
                ]},
                {"key": "player_rush_yds", "outcomes": [
                    {"name": "Over", "description": "Jahmyr Gibbs", "price": -115, "point": 55.5},
                ]},
            ],
        },
        {
            "key": "fanduel",  # worse Over price on rec yds -- should NOT win best-price
            "markets": [
                {"key": "player_reception_yds", "outcomes": [
                    {"name": "Over", "description": "Amon-Ra St. Brown", "price": -130, "point": 76.5},
                    {"name": "Under", "description": "Amon-Ra St. Brown", "price": 108, "point": 76.5},
                ]},
            ],
        },
    ],
}


def test_best_price_selection_and_shape():
    events = pd.DataFrame([{"event_id": "evt123", "home_team": "BUF", "away_team": "DET"}])

    with patch("odds_api.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {}
        mock_get.return_value = mock_resp

        df = odds_api.fetch_fantasy_prop_odds(events, api_key="fake-key-for-test")

    assert df is not None, "expected a DataFrame, got None"
    assert len(df) == 2, f"expected 2 player rows (ASB + Gibbs), got {len(df)}"

    asb = df[df["player_name_raw"] == "Amon-Ra St. Brown"].iloc[0]
    # draftkings' -110 beats fanduel's -130 as the better (higher/less negative) Over price
    assert asb["rec_yds_over_price"] == -110, f"expected best Over price -110, got {asb['rec_yds_over_price']}"
    assert asb["rec_yds_point"] == 75.5, f"expected the point tied to the WINNING price (75.5), got {asb['rec_yds_point']}"
    assert asb["rec_yds_under_price"] == 108, f"expected best Under price 108 (fanduel), got {asb['rec_yds_under_price']}"
    assert asb["receptions_over_price"] == -125
    assert asb["receptions_point"] == 6.5

    gibbs = df[df["player_name_raw"] == "Jahmyr Gibbs"].iloc[0]
    assert gibbs["rush_yds_over_price"] == -115
    assert gibbs["rush_yds_point"] == 55.5
    assert pd.isna(gibbs.get("rush_yds_under_price")), "no Under outcome in fixture -- should stay missing, not crash"
    assert pd.isna(gibbs.get("rec_yds_point")), "Gibbs has no rec-yds outcome in fixture -- should stay missing"

    print("PASS: best-price selection, point-tied-to-winning-price, partial-data handling")


def test_no_key_returns_none():
    events = pd.DataFrame([{"event_id": "evt123", "home_team": "BUF", "away_team": "DET"}])
    import os
    old = os.environ.pop("ODDS_API_KEY", None)
    try:
        result = odds_api.fetch_fantasy_prop_odds(events, api_key=None)
        assert result is None, "expected None with no API key available"
    finally:
        if old is not None:
            os.environ["ODDS_API_KEY"] = old
    print("PASS: no key -> None, no crash")


def test_empty_events_returns_none():
    result = odds_api.fetch_fantasy_prop_odds(pd.DataFrame(), api_key="fake-key")
    assert result is None, "expected None for empty events frame"
    print("PASS: empty events -> None")


if __name__ == "__main__":
    test_best_price_selection_and_shape()
    test_no_key_returns_none()
    test_empty_events_returns_none()
    print("\n3/3 PASSING (mocked -- not yet checked against a live key, see module docstring)")

"""
test_game_features_weather.py — Integration test for game_features.py's
_apply_weather(): verifies the full flow (indoor neutral fill, real
recorded data left untouched, future outdoor games getting real
live/seasonal weather, away rows inheriting the home row's resolved
weather, and the narrow last-resort fallback) against a synthetic
`long`-shaped DataFrame -- without needing the full power_ratings/
team_features dependency chain this sandbox doesn't have.
"""
import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from game_features import _apply_weather


def mock_response(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def make_synthetic_long():
    """
    6 rows modeling 3 games:
      - Game 1: BUF home vs MIA away, PLAYED (real recorded temp/wind) -- must stay untouched.
      - Game 2: DET home vs GB away, DOME -- both rows must get the neutral fill.
      - Game 3: BUF home vs NE away, FUTURE outdoor game, gameday soon (within
        forecast horizon) -- must get real weather resolved for the home row,
        and the away (NE) row must inherit the SAME resolved values.
    """
    today = date(2026, 11, 10)
    soon = today + timedelta(days=5)
    rows = [
        # Game 1 -- played, real recorded weather (BUF home)
        {"game_id": "g1", "team": "BUF", "opponent": "MIA", "is_home": 1,
         "roof": "outdoors", "temp": 41.0, "wind": 9.0, "gameday": "2025-11-02"},
        {"game_id": "g1", "team": "MIA", "opponent": "BUF", "is_home": 0,
         "roof": "outdoors", "temp": 41.0, "wind": 9.0, "gameday": "2025-11-02"},
        # Game 2 -- dome, both rows
        {"game_id": "g2", "team": "DET", "opponent": "GB", "is_home": 1,
         "roof": "dome", "temp": None, "wind": None, "gameday": "2026-11-15"},
        {"game_id": "g2", "team": "GB", "opponent": "DET", "is_home": 0,
         "roof": "dome", "temp": None, "wind": None, "gameday": "2026-11-15"},
        # Game 3 -- future outdoor, within forecast horizon
        {"game_id": "g3", "team": "BUF", "opponent": "NE", "is_home": 1,
         "roof": "outdoors", "temp": None, "wind": None, "gameday": soon.isoformat()},
        {"game_id": "g3", "team": "NE", "opponent": "BUF", "is_home": 0,
         "roof": "outdoors", "temp": None, "wind": None, "gameday": soon.isoformat()},
        # Extra real historical BUF November games so the seasonal tier has data too
        {"game_id": "h1", "team": "BUF", "opponent": "NYJ", "is_home": 1,
         "roof": "outdoors", "temp": 36.0, "wind": 14.0, "gameday": "2024-11-10"},
        {"game_id": "h2", "team": "BUF", "opponent": "NYJ", "is_home": 1,
         "roof": "outdoors", "temp": 40.0, "wind": 12.0, "gameday": "2023-11-12"},
    ]
    return pd.DataFrame(rows)


def main():
    checks = []
    fake_forecast = {
        "daily": {
            "time": [(date(2026, 11, 10) + timedelta(days=5)).isoformat()],
            "temperature_2m_max": [44.0],
            "temperature_2m_min": [30.0],
            "wind_speed_10m_max": [16.0],
        }
    }

    with patch("weather_forecast.date") as mock_date, \
         patch("weather_forecast.requests.get", return_value=mock_response(fake_forecast)):
        mock_date.today.return_value = date(2026, 11, 10)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        long = _apply_weather(make_synthetic_long())

    def row(game_id, team):
        r = long[(long["game_id"] == game_id) & (long["team"] == team)].iloc[0]
        return r

    # ---- Played game: real recorded data must be completely untouched ----
    g1_buf = row("g1", "BUF")
    checks.append(("played game keeps real recorded temp", g1_buf["temp"], 41.0))
    checks.append(("played game keeps real recorded wind", g1_buf["wind"], 9.0))
    checks.append(("played game temp_filled = real temp (no fallback touched it)", g1_buf["temp_filled"], 41.0))

    # ---- Dome game: both rows get neutral fill, is_indoor=1 ----
    g2_det = row("g2", "DET")
    g2_gb = row("g2", "GB")
    checks.append(("dome home row is_indoor=1", g2_det["is_indoor"], 1.0))
    checks.append(("dome home row temp_filled = neutral 70", g2_det["temp_filled"], 70.0))
    checks.append(("dome home row wind_filled = neutral 0", g2_det["wind_filled"], 0.0))
    checks.append(("dome away row also gets neutral fill", g2_gb["temp_filled"], 70.0))

    # ---- Future outdoor game: home row gets REAL live forecast ----
    g3_buf = row("g3", "BUF")
    checks.append(("future game temp = live forecast avg(44,30)=37.0", g3_buf["temp"], 37.0))
    checks.append(("future game wind = live forecast value", g3_buf["wind"], 16.0))

    # ---- Away row inherits the SAME resolved weather (same venue/date) ----
    g3_ne = row("g3", "NE")
    checks.append(("away row inherits home row's real resolved temp", g3_ne["temp"], 37.0))
    checks.append(("away row inherits home row's real resolved wind", g3_ne["wind"], 16.0))

    print(f"{'check':60s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 90)
    all_ok = True
    for name, got, want in checks:
        ok = (got == want)
        all_ok &= ok
        print(f"{name:60s} {str(got):>10s} {str(want):>10s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 90)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

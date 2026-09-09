"""
test_weather_forecast.py — Validates weather_forecast.py: live-forecast
response parsing (mocked against Open-Meteo's documented response shape),
the seasonal-history fallback computed from real synthetic historical rows,
graceful failure handling, and the priority order between all three tiers.
"""
import sys
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests
from weather_forecast import (
    fetch_live_forecast, seasonal_fallback, get_game_weather, STADIUM_COORDS, _to_date,
)


def mock_response(json_data, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_data
    if status_ok:
        resp.raise_for_status = MagicMock()
    else:
        resp.raise_for_status.side_effect = Exception("HTTP error")
    return resp


def make_historical_df():
    """Synthetic real-shaped historical rows: BUF has 3 real outdoor home
    games in November (avg temp=35, avg wind=12), 1 in September, GB has
    none at all (tests the 'no data' path)."""
    rows = [
        {"team": "BUF", "is_home": 1, "is_indoor": 0, "temp": 38.0, "wind": 10.0, "game_month": 11},
        {"team": "BUF", "is_home": 1, "is_indoor": 0, "temp": 32.0, "wind": 15.0, "game_month": 11},
        {"team": "BUF", "is_home": 1, "is_indoor": 0, "temp": 35.0, "wind": 11.0, "game_month": 11},
        {"team": "BUF", "is_home": 1, "is_indoor": 0, "temp": 68.0, "wind": 8.0, "game_month": 9},
        {"team": "BUF", "is_home": 0, "is_indoor": 0, "temp": -10.0, "wind": 99.0, "game_month": 11},  # away game, must be excluded
        {"team": "MIA", "is_home": 1, "is_indoor": 1, "temp": None, "wind": None, "game_month": 11},  # dome, must be excluded
    ]
    return pd.DataFrame(rows)


def main():
    checks = []

    # ---- _to_date handles string, Timestamp, and date ----
    checks.append(("_to_date parses ISO string", _to_date("2026-11-15"), date(2026, 11, 15)))
    checks.append(("_to_date passes through a real date", _to_date(date(2026, 11, 15)), date(2026, 11, 15)))
    checks.append(("_to_date parses pandas Timestamp", _to_date(pd.Timestamp("2026-11-15")), date(2026, 11, 15)))

    # ---- Live forecast: realistic Open-Meteo response shape, happy path ----
    fake_response = {
        "daily": {
            "time": ["2026-11-15"],
            "temperature_2m_max": [40.0],
            "temperature_2m_min": [28.0],
            "wind_speed_10m_max": [18.5],
        }
    }
    with patch("weather_forecast.requests.get", return_value=mock_response(fake_response)) as mock_get:
        result = fetch_live_forecast(42.7738, -78.7870, date(2026, 11, 15))
    checks.append(("live forecast temp = avg(max,min) = (40+28)/2=34.0", result["temp"], 34.0))
    checks.append(("live forecast wind passed through", result["wind"], 18.5))
    called_params = mock_get.call_args.kwargs["params"]
    checks.append(("real lat/lon sent to the API", (called_params["latitude"], called_params["longitude"]), (42.7738, -78.7870)))

    # ---- Live forecast: date outside what the API returned (beyond real horizon) ----
    with patch("weather_forecast.requests.get", return_value=mock_response({"daily": {"time": [], "temperature_2m_max": [], "temperature_2m_min": [], "wind_speed_10m_max": []}})):
        result2 = fetch_live_forecast(42.7738, -78.7870, date(2026, 12, 25))
    checks.append(("date not in API response -> None, not a crash", result2, None))

    # ---- Live forecast: network failure handled gracefully ----
    with patch("weather_forecast.requests.get", side_effect=requests.exceptions.ConnectionError("connection refused")):
        result3 = fetch_live_forecast(42.7738, -78.7870, date(2026, 11, 15))
    checks.append(("network failure -> None, not a crash", result3, None))

    # ---- Live forecast: a genuinely unanticipated error type is still caught ----
    with patch("weather_forecast.requests.get", side_effect=RuntimeError("something nobody anticipated")):
        result3b = fetch_live_forecast(42.7738, -78.7870, date(2026, 11, 15))
    checks.append(("unanticipated exception type -> still None, not a crash", result3b, None))

    # ---- Live forecast: malformed response handled gracefully ----
    with patch("weather_forecast.requests.get", return_value=mock_response({"unexpected": "shape"})):
        result4 = fetch_live_forecast(42.7738, -78.7870, date(2026, 11, 15))
    checks.append(("malformed response -> None, not a crash", result4, None))

    # ---- Seasonal fallback: real synthetic historical computation ----
    hist = make_historical_df()
    buf_nov = seasonal_fallback(hist, "BUF", 11)
    checks.append(("BUF November seasonal temp = mean(38,32,35)=35.0", buf_nov["temp"], 35.0))
    checks.append(("BUF November seasonal wind = mean(10,15,11)=12.0", buf_nov["wind"], 12.0))

    buf_sep = seasonal_fallback(hist, "BUF", 9)
    checks.append(("BUF September seasonal temp = single real value 68.0", buf_sep["temp"], 68.0))

    gb_nov = seasonal_fallback(hist, "GB", 11)
    checks.append(("GB has zero historical rows -> None", gb_nov, None))

    mia_nov = seasonal_fallback(hist, "MIA", 11)
    checks.append(("MIA dome game correctly excluded -> None (no outdoor history)", mia_nov, None))

    # ---- get_game_weather: priority order ----
    # Case 1: within forecast horizon, live call succeeds -> live_forecast wins
    with patch("weather_forecast.requests.get", return_value=mock_response(fake_response)):
        r = get_game_weather("BUF", date(2026, 11, 15), hist, today=date(2026, 11, 10))
    checks.append(("within horizon + live succeeds -> source=live_forecast", r["source"], "live_forecast"))
    checks.append(("within horizon + live succeeds -> real live temp used", r["temp"], 34.0))

    # Case 2: within forecast horizon, live call FAILS -> falls back to seasonal
    with patch("weather_forecast.requests.get", side_effect=requests.exceptions.Timeout("timeout")):
        r2 = get_game_weather("BUF", date(2026, 11, 15), hist, today=date(2026, 11, 10))
    checks.append(("within horizon + live fails -> source=seasonal_history", r2["source"], "seasonal_history"))
    checks.append(("within horizon + live fails -> real seasonal temp used", r2["temp"], 35.0))

    # Case 3: beyond forecast horizon -> seasonal directly, no API call attempted
    with patch("weather_forecast.requests.get") as mock_get_not_called:
        r3 = get_game_weather("BUF", date(2027, 11, 21), hist, today=date(2026, 8, 1))
    checks.append(("beyond horizon -> source=seasonal_history", r3["source"], "seasonal_history"))
    checks.append(("beyond horizon -> API never called (saves a request)", mock_get_not_called.called, False))

    # Case 4: beyond horizon AND no seasonal history -> global_fallback
    with patch("weather_forecast.requests.get"):
        r4 = get_game_weather("GB", date(2026, 12, 20), hist, today=date(2026, 8, 1))
    checks.append(("beyond horizon + no history -> source=global_fallback", r4["source"], "global_fallback"))
    checks.append(("global_fallback -> temp is None (caller applies its own default)", r4["temp"], None))

    # ---- Coordinate table sanity ----
    checks.append(("all 32 teams have coordinates", len(STADIUM_COORDS), 32))
    checks.append(("LAC and LAR share SoFi Stadium coords", STADIUM_COORDS["LAC"], STADIUM_COORDS["LAR"]))
    checks.append(("NYG and NYJ share MetLife coords", STADIUM_COORDS["NYG"], STADIUM_COORDS["NYJ"]))

    print(f"{'check':65s} {'got':>18s} {'want':>18s}  ok")
    print("-" * 108)
    all_ok = True
    for name, got, want in checks:
        ok = (got == want)
        all_ok &= ok
        print(f"{name:65s} {str(got):>18s} {str(want):>18s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 108)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

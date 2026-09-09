"""
weather_forecast.py — Real weather for upcoming outdoor NFL games, replacing
the crude "single global blended median" fallback that used to fill
temp/wind for every future game regardless of location or season.

TWO-TIER APPROACH:
  1. LIVE FORECAST (within ~15 days of kickoff): calls Open-Meteo's free,
     keyless forecast API for the specific stadium's coordinates and game
     date. Real, location- and date-specific.
  2. HISTORICAL SEASONAL FALLBACK (beyond the forecast horizon, or if the
     live call fails for any reason): the mean of that team's own ACTUAL
     recorded outdoor home-game temp/wind in the SAME CALENDAR MONTH,
     computed from real historical data already loaded -- no new API
     dependency for this tier. Location- and season-aware, a real
     improvement over the old single-global-median fallback -- though still
     an estimate, not a forecast, and explicitly labeled as such via the
     "source" field in the return value.

WHY OPEN-METEO: free, no API key (one less secret for a user who manages
everything through GitHub's web UI already), documented and stable daily-
forecast endpoint, global coverage (handles the occasional London game
correctly, unlike a US-only weather source).

HONESTY NOTE ON VERIFICATION: this module was built against Open-Meteo's
publicly documented API contract and tested with mocked HTTP responses
matching that documented shape (see test_weather_forecast.py) -- the live
network call itself has NOT been exercised against the real API, because
this was built in a sandboxed environment with no route to
api.open-meteo.com (confirmed blocked, not just untested). Worth an
explicit check on the first real GitHub Actions run: look for a
"[weather]" line in the log and confirm no unexpected warning about a
failed forecast call for every outdoor game.

Public API:
    get_game_weather(team, game_date, historical_df) -> {temp, wind, source}
        source is one of "live_forecast", "seasonal_history", "global_fallback"
"""

from __future__ import annotations
from datetime import datetime, date

import pandas as pd
import requests

# Open-Meteo's standard forecast is documented as reliable up to ~16 days
# out; staying one day inside that avoids relying on the least-reliable
# edge of its range.
FORECAST_HORIZON_DAYS = 15

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Home stadium coordinates for all 32 teams. Two pairs of teams share a
# stadium (LAC/LAR at SoFi, NYG/NYJ at MetLife) -- correct, not a typo.
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3392),
    "LAR": (33.9535, -118.3392), "MIA": (25.9580, -80.2389), "MIN": (44.9737, -93.2577),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SF": (37.4032, -121.9698), "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}


def _to_date(d) -> date:
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    return d


def fetch_live_forecast(lat: float, lon: float, target_date) -> dict | None:
    """
    Real Open-Meteo daily forecast for one date. Returns {"temp": F,
    "wind": mph} or None on any failure (network, unexpected response
    shape, date outside what the API actually returned) -- callers fall
    back to the seasonal estimate rather than crash the whole pipeline
    over one bad weather call, matching this project's established
    graceful-fallback pattern (see odds_api.py).

    Daily-level, not hourly: this averages that day's forecast high/low as
    an approximation of game-time conditions, not the specific kickoff
    hour, since kickoff time isn't reliably available in the loaded
    schedule columns. A coarse in-model signal, not a minute-by-minute
    simulation -- worth knowing when interpreting a specific number.
    """
    target_date = _to_date(target_date)
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        resp = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "start_date": date_str, "end_date": date_str, "timezone": "auto",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        if date_str not in times:
            return None  # date outside what the API actually returned
        idx = times.index(date_str)
        tmax = daily["temperature_2m_max"][idx]
        tmin = daily["temperature_2m_min"][idx]
        wind = daily["wind_speed_10m_max"][idx]
        if tmax is None or tmin is None or wind is None:
            return None
        return {"temp": round((tmax + tmin) / 2, 1), "wind": round(wind, 1)}
    except Exception as e:
        # Deliberately broad: this is a best-effort forecast call feeding a
        # fallback chain, not a critical path -- ANY failure here (network,
        # malformed JSON, an nflverse-style surprise in the response shape,
        # something not anticipated by the narrower exception types) should
        # degrade to the seasonal estimate, never take down the whole
        # prediction run over one weather API call.
        print(f"WARNING: [weather] live forecast call failed ({e}) -- falling back to seasonal estimate.")
        return None


def seasonal_fallback(historical_df: pd.DataFrame, team: str, target_month: int) -> dict | None:
    """
    Mean of TEAM's own real recorded outdoor home-game temp/wind in the
    same calendar MONTH, from historical_df. Expects columns: team,
    is_home, is_indoor, temp, wind, game_month. Returns None if there's no
    real historical data to compute from (e.g. zero prior outdoor home
    games for that team in that month across the loaded seasons) -- caller
    falls back further in that case.
    """
    subset = historical_df[
        (historical_df["team"] == team)
        & (historical_df["is_home"] == 1)
        & (historical_df["is_indoor"] == 0)
        & (historical_df["game_month"] == target_month)
        & historical_df["temp"].notna()
    ]
    if len(subset) == 0:
        return None
    return {
        "temp": round(float(subset["temp"].mean()), 1),
        "wind": round(float(subset["wind"].mean()), 1),
    }


def get_game_weather(team: str, game_date, historical_df: pd.DataFrame,
                      today: date | None = None) -> dict:
    """
    Real weather estimate for one team's outdoor home game, in priority order:
      1. Live forecast, if the game is within FORECAST_HORIZON_DAYS of today
      2. Seasonal historical average (that team, same calendar month, real
         past games) -- used both when the game is too far out to forecast
         and as the fallback if the live call fails
      3. Global fallback (temp/wind = None) -- only if both above fail,
         meaning a team has zero outdoor home-game history in that month
         AND the live forecast call also failed. Caller should apply its
         own last-resort default in this case.

    `today` is injectable for testing; defaults to the real current date.

    Returns {"temp": float|None, "wind": float|None, "source": str} --
    source is "live_forecast", "seasonal_history", or "global_fallback",
    so the output can show which tier actually produced the number.
    """
    today = today or date.today()
    game_date = _to_date(game_date)

    days_out = (game_date - today).days
    if 0 <= days_out <= FORECAST_HORIZON_DAYS and team in STADIUM_COORDS:
        lat, lon = STADIUM_COORDS[team]
        live = fetch_live_forecast(lat, lon, game_date)
        if live is not None:
            return {**live, "source": "live_forecast"}

    seasonal = seasonal_fallback(historical_df, team, game_date.month)
    if seasonal is not None:
        return {**seasonal, "source": "seasonal_history"}

    return {"temp": None, "wind": None, "source": "global_fallback"}

"""
odds_api.py — Shared client for The Odds API (the-odds-api.com), used by
both production scripts to fetch real market odds.

CORRECTION to earlier notes in this project: current (non-historical) player
prop odds ARE included on The Odds API's free tier (500 credits/month) --
only HISTORICAL player-prop data requires a paid plan. Verified against the
live docs at the-odds-api.com as of this session. Earlier README language
saying player props "need a paid plan" was imprecise -- fixed.

Requires an API key, set via the ODDS_API_KEY environment variable (GitHub
Actions: repo secret of the same name). If unset, both fetch functions
return None and callers fall back to model-only output -- this integration
degrades gracefully rather than breaking the pipeline if the key is missing
or the API errors.

CREDIT BUDGET (free tier: 500 credits/month):
  - Track B (game lines): 1 bulk call = markets x regions = 3 x 1 = 3 credits
    per run, regardless of game count.
  - Track A (player TD props): 1 event-odds call PER GAME (no bulk endpoint
    for additional markets) = 1 credit x N games per run.
  - At 4 runs/week x ~16 games x 18 weeks: ~1,368 credits/season combined --
    fits the free tier's ~2,250/season budget (500/mo x ~4.5 months) with
    room to spare. Re-check this if cadence or game count assumptions change.

NAME MATCHING: The Odds API returns player names as free text (e.g.
"Saquon Barkley"), not our GSIS player_id. Matched via nfl_data_py's ID
crosswalk `merge_name` field -- built for exactly this cross-provider
matching problem -- normalized the same way on both sides as a defensive
double-check.

Public API:
    fetch_game_odds(api_key) -> DataFrame keyed by (home_team, away_team) with
        best-price h2h/spreads/totals across bookmakers, or None
    fetch_player_td_odds(api_key, events) -> DataFrame of anytime-TD best
        prices per player per event, or None
"""

from __future__ import annotations
import os
import re
import time
import requests
import pandas as pd

BASE_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

# Odds API uses full team names; our pipeline uses nflverse abbreviations.
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

REQUEST_TIMEOUT = 15


def _get_api_key(api_key: str | None) -> str | None:
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY not set -- skipping live odds, model-only output.")
        return None
    return key


def _log_usage(resp: requests.Response, label: str):
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        print(f"  [odds api] {label}: {used} used this billing period, {remaining} remaining")


def normalize_name(name: str) -> str:
    """Same normalization on both sides of a match: lowercase, strip
    punctuation/suffixes, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    n = re.sub(r"[.'\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def fetch_game_odds(api_key: str | None = None) -> pd.DataFrame | None:
    """
    Best-price h2h/spreads/totals across bookmakers for all upcoming NFL
    games, in ONE bulk API call (3 credits: 3 markets x 1 region).
    Returns None on any failure (missing key, API error, empty response) --
    callers should fall back to model-only output, never crash the pipeline
    over a live-odds hiccup.
    """
    key = _get_api_key(api_key)
    if key is None:
        return None

    try:
        resp = requests.get(
            f"{BASE_URL}/odds",
            params={"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american", "apiKey": key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        _log_usage(resp, "game odds")
        games = resp.json()
    except Exception as e:
        print(f"WARNING: fetch_game_odds failed ({e}) -- falling back to model-only output.")
        return None

    rows = []
    for g in games:
        home = TEAM_NAME_TO_ABBR.get(g["home_team"])
        away = TEAM_NAME_TO_ABBR.get(g["away_team"])
        if home is None or away is None:
            continue

        best_home_ml, best_away_ml = None, None
        best_home_spread_price, best_home_spread_point = None, None
        best_away_spread_price, best_away_spread_point = None, None
        best_over_price, best_over_point, best_under_price = None, None, None

        for bk in g.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if o["name"] == g["home_team"]:
                            if best_home_ml is None or o["price"] > best_home_ml:
                                best_home_ml = o["price"]
                        elif o["name"] == g["away_team"]:
                            if best_away_ml is None or o["price"] > best_away_ml:
                                best_away_ml = o["price"]
                elif mkt["key"] == "spreads":
                    for o in mkt["outcomes"]:
                        if o["name"] == g["home_team"]:
                            if best_home_spread_price is None or o["price"] > best_home_spread_price:
                                best_home_spread_price = o["price"]
                                best_home_spread_point = o["point"]
                        elif o["name"] == g["away_team"]:
                            if best_away_spread_price is None or o["price"] > best_away_spread_price:
                                best_away_spread_price = o["price"]
                                best_away_spread_point = o["point"]
                elif mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            if best_over_price is None or o["price"] > best_over_price:
                                best_over_price = o["price"]
                                best_over_point = o["point"]
                        elif o["name"] == "Under":
                            if best_under_price is None or o["price"] > best_under_price:
                                best_under_price = o["price"]

        rows.append({
            "event_id": g["id"], "home_team": home, "away_team": away,
            "commence_time": g["commence_time"],
            "live_home_moneyline": best_home_ml, "live_away_moneyline": best_away_ml,
            "live_home_spread_point": best_home_spread_point, "live_home_spread_price": best_home_spread_price,
            "live_away_spread_point": best_away_spread_point, "live_away_spread_price": best_away_spread_price,
            "live_total_point": best_over_point,
            "live_over_price": best_over_price, "live_under_price": best_under_price,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def fetch_player_td_odds(events: pd.DataFrame, api_key: str | None = None) -> pd.DataFrame | None:
    """
    Best-price anytime-TD odds per player, across all events in `events`
    (expects an event_id column, e.g. from fetch_game_odds's output). One
    API call PER EVENT (no bulk endpoint for additional markets) -- costs
    len(events) credits total. Returns None if no key or all calls fail.
    """
    key = _get_api_key(api_key)
    if key is None or events is None or events.empty:
        return None

    rows = []
    for _, ev in events.iterrows():
        try:
            resp = requests.get(
                f"{BASE_URL}/events/{ev['event_id']}/odds",
                params={"regions": "us", "markets": "player_anytime_td", "oddsFormat": "american", "apiKey": key},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            _log_usage(resp, f"TD props {ev['home_team']}v{ev['away_team']}")
            data = resp.json()
        except Exception as e:
            print(f"WARNING: fetch_player_td_odds failed for event {ev['event_id']} "
                  f"({ev['home_team']} v {ev['away_team']}): {e} -- skipping this game's props.")
            continue

        best_price_by_player: dict[str, int] = {}
        for bk in data.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] != "player_anytime_td":
                    continue
                for o in mkt["outcomes"]:
                    if o["name"] != "Yes":
                        continue
                    player = o.get("description")
                    if not player:
                        continue
                    if player not in best_price_by_player or o["price"] > best_price_by_player[player]:
                        best_price_by_player[player] = o["price"]

        for player, price in best_price_by_player.items():
            rows.append({
                "event_id": ev["event_id"], "home_team": ev["home_team"], "away_team": ev["away_team"],
                "player_name_raw": player, "player_name_norm": normalize_name(player),
                "live_anytime_td_price": price,
            })

        time.sleep(0.15)  # light self-throttle, polite to the API

    if not rows:
        return None
    return pd.DataFrame(rows)


def fetch_fantasy_prop_odds(events: pd.DataFrame, api_key: str | None = None) -> pd.DataFrame | None:
    """
    Best-price rec yards / rush yards / receptions O/U lines per player,
    across all events in `events` -- same shape as fetch_player_td_odds, one
    call PER EVENT, but all three markets requested together in a single
    call (comma-separated `markets` param) rather than three separate calls.
    Per the API's own documented cost formula (unique markets present in the
    response x regions), this costs UP TO 3 credits per event (up from Track
    A's 1 credit/event for player_anytime_td alone) -- re-check the combined
    Track A + Track C credit budget against the free tier's ~2,250/season
    if running both at the existing Tue-Fri cadence (see this module's
    CREDIT BUDGET docstring above, written for Track A/B only -- Track C
    adds to it, not yet re-totaled here).

    For each player/market, keeps the point value tied to whichever
    bookmaker gave the single BEST (highest) Over price -- same
    "best-price across bookmakers" convention fetch_player_td_odds already
    uses, not a new one invented for this function.

    NOT YET SMOKE-TESTED AGAINST A LIVE KEY (same caveat fetch_player_td_odds
    originally shipped with) -- built and parses against the API's
    documented response shape, validated with a mocked fixture
    (test_fantasy_prop_odds.py), but real bookmaker responses can drift from
    docs. Worth a real run with ODDS_API_KEY set and checking the Action log
    for '[odds api] ... credits remaining' lines before trusting it blind.

    Returns None if no key, no events, or every call fails.
    """
    key = _get_api_key(api_key)
    if key is None or events is None or events.empty:
        return None

    MARKETS = "player_rush_yds,player_reception_yds,player_receptions"
    rows_by_player: dict[tuple, dict] = {}

    for _, ev in events.iterrows():
        try:
            resp = requests.get(
                f"{BASE_URL}/events/{ev['event_id']}/odds",
                params={"regions": "us", "markets": MARKETS, "oddsFormat": "american", "apiKey": key},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            _log_usage(resp, f"fantasy props {ev['home_team']}v{ev['away_team']}")
            data = resp.json()
        except Exception as e:
            print(f"WARNING: fetch_fantasy_prop_odds failed for event {ev['event_id']} "
                  f"({ev['home_team']} v {ev['away_team']}): {e} -- skipping this game's props.")
            continue

        MARKET_FIELD = {
            "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds", "player_receptions": "receptions",
        }
        for bk in data.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                field = MARKET_FIELD.get(mkt["key"])
                if field is None:
                    continue
                for o in mkt.get("outcomes", []):
                    player = o.get("description")
                    if not player:
                        continue
                    key_tuple = (ev["event_id"], player)
                    row = rows_by_player.setdefault(key_tuple, {
                        "event_id": ev["event_id"], "home_team": ev["home_team"], "away_team": ev["away_team"],
                        "player_name_raw": player, "player_name_norm": normalize_name(player),
                    })
                    if o["name"] == "Over":
                        price_field, point_field = f"{field}_over_price", f"{field}_point"
                        if row.get(price_field) is None or o["price"] > row[price_field]:
                            row[price_field] = o["price"]
                            row[point_field] = o.get("point")
                    elif o["name"] == "Under":
                        price_field = f"{field}_under_price"
                        if row.get(price_field) is None or o["price"] > row[price_field]:
                            row[price_field] = o["price"]

        time.sleep(0.15)  # light self-throttle, polite to the API

    if not rows_by_player:
        return None
    return pd.DataFrame(list(rows_by_player.values()))


def american_to_implied_prob(price: float) -> float:
    if price > 0:
        return 100 / (price + 100)
    return -price / (-price + 100)


def json_default(obj):
    """
    Pass as `default=` to json.dump/json.dumps anywhere output might contain
    values pulled from a pandas DataFrame/Series. numpy.float64 happens to
    subclass Python's float (so it silently serializes fine), but
    numpy.int64 does NOT subclass int -- it raises TypeError with no
    indication of WHERE the offending value came from. This bit the live-
    odds integration immediately: American odds prices (moneylines, spread
    prices) come back as int64 once they've passed through a DataFrame,
    even if they started life as plain Python ints in odds_api.py's own
    row-construction code -- a DataFrame column is always numpy-backed
    regardless of what Python type went in, so casting at construction time
    doesn't help; the fix has to be at serialization time.
    """
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

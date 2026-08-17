"""
manual_odds.py — Loads hand-maintained anytime-TD prices from
data/manual_odds.csv, as a free (no API key) alternative or supplement to
The Odds API.

WHY THIS EXISTS: current player-prop odds ARE on The Odds API's free tier
(confirmed earlier this project -- only historical props need a paid plan),
but the user doesn't want to sign up for any key at all right now. This is
a legitimate alternative, not just a stopgap: real prices, typed in by
hand from DraftKings (or any book), matched the same way and logged the
same way live-API odds would be -- still usable for the real backtest
later, not a throwaway.

FILE FORMAT (data/manual_odds.csv), edited directly on GitHub like every
other file in this project:

    player_name,price
    Saquon Barkley,-145
    Jonathan Taylor,180

Player names can be full names ("Saquon Barkley") or the dashboard's
abbreviated form ("S.Barkley") -- matched via the same normalize_name() +
merge_name mechanism odds_api.py already uses for live-API names, so
either format works. Price is American odds (a bare "180" and "+180" both
parse as +180).

STALENESS: this file is NOT automatically refreshed -- unlike the live API,
which pulls current prices every cron run, manual_odds.csv only has
whatever was last typed into it. The output's model_notes flag this
explicitly (a timestamp comparison isn't reliable across git checkouts, so
this is a documentation reminder, not an automated staleness check) --
update the file before a run if the lines have moved.

Public API:
    load_manual_odds(path) -> DataFrame with player_name_norm, manual_price,
        or None if the file doesn't exist or has no valid rows
"""

from __future__ import annotations
import os
import pandas as pd

from odds_api import normalize_name

DEFAULT_PATH = "../data/manual_odds.csv"


def _parse_price(raw) -> int | None:
    try:
        s = str(raw).strip().replace("+", "")
        if s == "" or s.lower() == "nan":
            return None
        return int(float(s))
    except (ValueError, TypeError):
        return None


def load_manual_odds(path: str = DEFAULT_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"WARNING: manual_odds.csv exists but couldn't be read ({e}) -- skipping.")
        return None

    if "player_name" not in df.columns or "price" not in df.columns:
        print("WARNING: manual_odds.csv missing required columns (player_name, price) -- skipping.")
        return None

    df["price"] = df["price"].apply(_parse_price)
    df = df.dropna(subset=["player_name", "price"]).copy()
    if df.empty:
        return None

    df["price"] = df["price"].astype(int)
    df["player_name_norm"] = df["player_name"].apply(normalize_name)
    df = df.drop_duplicates(subset=["player_name_norm"], keep="last")
    return df[["player_name_norm", "price"]].rename(columns={"price": "manual_price"})

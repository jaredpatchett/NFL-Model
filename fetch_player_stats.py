"""
fetch_player_stats.py

Pulls weekly player stats + snap counts from nflverse-data (GitHub releases),
filters to the current season, and writes them into data/ for the fantasy
projection track (Track C) and downstream features (defense-vs-position
ranks, player percentiles, snap share chart).

Source: https://github.com/nflverse/nflverse-data/releases
No API key required.

Usage:
    python fetch_player_stats.py                # defaults to current season
    python fetch_player_stats.py --season 2025   # explicit season
"""

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
PLAYER_STATS_URL = f"{NFLVERSE_BASE}/stats_player/stats_player_week_{{season}}.csv"
SNAP_COUNTS_URL = f"{NFLVERSE_BASE}/snap_counts/snap_counts_{{season}}.csv"

DATA_DIR = Path("data")

# Keep the columns Track C / the fantasy card actually needs — trims a wide
# nflverse export down to what we store, same spirit as the lean feature
# sets used in Track A/B.
PLAYER_STATS_COLUMNS = [
    "player_id", "player_name", "player_display_name", "position",
    "position_group", "headshot_url", "season", "week", "season_type",
    "game_id", "team", "opponent_team",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles_lost", "rushing_fumbles_lost",
]


def fetch_csv(url: str, season: int) -> pd.DataFrame:
    resolved_url = url.format(season=season)
    resp = requests.get(resolved_url, timeout=60)
    if resp.status_code == 404:
        raise FileNotFoundError(
            f"No nflverse release file for season {season} at {resolved_url}. "
            "It may not have published yet for this season/week."
        )
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), low_memory=False)


def fetch_player_stats(season: int) -> pd.DataFrame:
    df = fetch_csv(PLAYER_STATS_URL, season)
    # regular season only, matching Track A/B convention
    df = df[df["season_type"] == "REG"].copy()
    keep = [c for c in PLAYER_STATS_COLUMNS if c in df.columns]
    df = df[keep]

    df["fantasy_points_ppr"] = (
        df.get("receiving_yards", 0) * 0.1
        + df.get("receptions", 0) * 1.0
        + df.get("rushing_yards", 0) * 0.1
        + df.get("rushing_tds", 0) * 6
        + df.get("receiving_tds", 0) * 6
        - df.get("receiving_fumbles_lost", 0) * 2
        - df.get("rushing_fumbles_lost", 0) * 2
    )
    return df


def fetch_snap_counts(season: int) -> pd.DataFrame:
    df = fetch_csv(SNAP_COUNTS_URL, season)
    keep = [
        "game_id", "season", "week", "player", "position", "team",
        "opponent", "offense_snaps", "offense_pct",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None,
                         help="NFL season year, e.g. 2025. Defaults to current year.")
    args = parser.parse_args()

    from datetime import date
    season = args.season or date.today().year

    DATA_DIR.mkdir(exist_ok=True)

    print(f"Fetching player stats for {season}...")
    try:
        player_stats = fetch_player_stats(season)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}", file=sys.stderr)
        player_stats = None

    print(f"Fetching snap counts for {season}...")
    try:
        snap_counts = fetch_snap_counts(season)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}", file=sys.stderr)
        snap_counts = None

    if player_stats is not None:
        out_path = DATA_DIR / f"player_stats_{season}.csv"
        player_stats.to_csv(out_path, index=False)
        print(f"  Wrote {len(player_stats)} rows -> {out_path}")

    if snap_counts is not None:
        out_path = DATA_DIR / f"snap_counts_{season}.csv"
        snap_counts.to_csv(out_path, index=False)
        print(f"  Wrote {len(snap_counts)} rows -> {out_path}")

    if player_stats is None and snap_counts is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
generate_fantasy_projections.py -- Production entrypoint for Track C
(fantasy point projections). Same pattern as generate_predictions.py (Track B)
and generate_player_predictions.py (Track A): trains/derives from all played
history, projects the upcoming week, writes a current snapshot (overwritten)
+ appends to a persistent log (never overwritten).

INPUT: data/player_stats_{season}.csv and data/snap_counts_{season}.csv,
written by fetch_player_stats.py (nflverse-data, pulled earlier in the same
workflow run -- see .github/workflows/update-predictions.yml).

METHOD (first cut -- a trailing-average baseline, not yet a trained
regression like Track A/B):
  1. Leakage-safe trailing fantasy points: for every player, shift(1) before
     taking a rolling mean over their last 4 played games -- same
     shift-before-aggregate principle as rolling_features.py, so a game's
     own result can never enter its own projection.
  2. Opponent-adjustment: trailing PPR fantasy points allowed BY THE
     UPCOMING OPPONENT to that player's position_group, normalized against
     the league-average allowed to that position_group THAT SAME WEEK --
     same "normalize against the same-week league average" approach as
     Track A's matchup_rating (defense_features.py), not a fixed historical
     constant. 1.00 = league average; above 1.00 = softer than average.
  3. Final projection = trailing_avg_pts * matchup_factor.

Honest limitation, stated plainly (same spirit as Track B's total-points
caveat): this hasn't been validated against held-out data the way Track A/B
were (AUC 0.70, Ridge MAE beats baseline). It's a reasonable, leakage-safe
baseline to ship and start logging -- a real accuracy check happens once
predictions_log-style entries can be joined against real outcomes.

Usage:
    python generate_fantasy_projections.py
"""

from __future__ import annotations
import json
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from nfl_data import load_schedules

SEASON = date.today().year
STATS_CSV = f"../data/player_stats_{SEASON}.csv"
SNAPS_CSV = f"../data/snap_counts_{SEASON}.csv"

OUT_JSON = "../data/fantasy_projections.json"
OUT_JS = "../data/fantasy_projections.js"
OUT_LOG = "../data/fantasy_projections_log.jsonl"

TRAILING_WINDOW = 4
POSITIONS = ["QB", "RB", "WR", "TE"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(
            f"{STATS_CSV} not found -- run fetch_player_stats.py first "
            "(it runs earlier in the same workflow)."
        )
    stats = pd.read_csv(STATS_CSV)
    stats = stats[stats["position"].isin(POSITIONS)].copy()

    snaps = pd.DataFrame()
    if os.path.exists(SNAPS_CSV):
        snaps = pd.read_csv(SNAPS_CSV)
    return stats, snaps


def build_trailing_player_avg(stats: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe trailing mean fantasy_points_ppr per player, shift(1)
    before rolling so a week's own result never enters its own average."""
    stats = stats.sort_values(["player_id", "week"]).reset_index(drop=True)
    shifted = stats.groupby("player_id")["fantasy_points_ppr"].shift(1)
    stats["trailing_avg_pts"] = shifted.groupby(stats["player_id"]).transform(
        lambda s: s.rolling(TRAILING_WINDOW, min_periods=1).mean()
    )
    stats["trailing_games"] = shifted.groupby(stats["player_id"]).cumcount()
    return stats


def build_defense_vs_position(stats: pd.DataFrame) -> pd.DataFrame:
    """Trailing fantasy points allowed by each team to each position_group,
    normalized against that same week's league-average allowed to that
    position -- same approach as Track A's matchup_rating."""
    team_pos_week = (
        stats.groupby(["opponent_team", "position", "week"], as_index=False)
        .agg(pts_allowed=("fantasy_points_ppr", "sum"),
             players=("player_id", "nunique"))
    )
    # per-team-per-position trailing (leakage-safe: shift before roll)
    team_pos_week = team_pos_week.sort_values(["opponent_team", "position", "week"])
    shifted = team_pos_week.groupby(["opponent_team", "position"])["pts_allowed"].shift(1)
    team_pos_week["trailing_allowed"] = shifted.groupby(
        [team_pos_week["opponent_team"], team_pos_week["position"]]
    ).transform(lambda s: s.rolling(TRAILING_WINDOW, min_periods=1).mean())

    # league-average allowed to that position, same week, same trailing basis
    league_avg = (
        team_pos_week.groupby(["position", "week"])["trailing_allowed"]
        .transform("mean")
    )
    team_pos_week["matchup_factor"] = np.where(
        league_avg > 0, team_pos_week["trailing_allowed"] / league_avg, 1.0
    )
    return team_pos_week


def rank_defense_vs_position(team_pos_week: pd.DataFrame, latest_week: dict) -> pd.DataFrame:
    """1 = allows the MOST fantasy points to that position (softest matchup),
    32 = allows the fewest (toughest) -- as of each team's latest played week."""
    rows = []
    for pos in POSITIONS:
        pos_df = team_pos_week[team_pos_week["position"] == pos]
        latest = (
            pos_df.sort_values("week")
            .groupby("opponent_team")
            .tail(1)
            .copy()
        )
        latest = latest.dropna(subset=["trailing_allowed"])
        if latest.empty:
            continue
        latest["def_rank"] = latest["trailing_allowed"].rank(ascending=False, method="min").astype(int)
        latest["def_rank_of"] = latest["opponent_team"].nunique()
        rows.append(latest[["opponent_team", "def_rank", "def_rank_of", "matchup_factor"]].assign(position=pos))
    if not rows:
        return pd.DataFrame(columns=["opponent_team", "def_rank", "def_rank_of", "matchup_factor", "position"])
    return pd.concat(rows, ignore_index=True)


def build_snap_pct(snaps: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        return pd.DataFrame(columns=["player", "team", "latest_snap_pct"])
    snaps = snaps.sort_values(["player", "week"])
    latest = snaps.groupby("player").tail(1)
    return latest[["player", "offense_pct"]].rename(columns={"offense_pct": "latest_snap_pct"})


def find_upcoming_matchups(season: int) -> pd.DataFrame:
    """Earliest unplayed week per team, with opponent -- same
    find_upcoming_week pattern as generate_predictions.py, but keyed by team
    since every team needs its own next opponent, not one shared week."""
    sched = load_schedules([season])
    unplayed = sched[sched["home_score"].isna()].copy()
    if unplayed.empty:
        return pd.DataFrame(columns=["team", "opponent", "week"])

    home = unplayed[["week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = unplayed[["week", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    both = pd.concat([home, away], ignore_index=True)
    return both.sort_values("week").groupby("team").head(1)


def main():
    print(f"Building fantasy projections for {SEASON}...")
    stats, snaps = load_inputs()
    if stats.empty:
        print("No player stats available yet -- nothing to project.")
        return

    stats = build_trailing_player_avg(stats)
    team_pos_week = build_defense_vs_position(stats)
    def_ranks = rank_defense_vs_position(team_pos_week, {})
    snap_pct = build_snap_pct(snaps)

    latest_row = (
        stats.sort_values("week")
        .groupby("player_id")
        .tail(1)
        .copy()
    )

    upcoming = find_upcoming_matchups(SEASON)
    latest_row = latest_row.merge(upcoming, left_on="team", right_on="team", how="left")

    latest_row = latest_row.merge(
        def_ranks, left_on=["opponent", "position"], right_on=["opponent_team", "position"], how="left"
    )
    latest_row = latest_row.merge(snap_pct, left_on="player_display_name", right_on="player", how="left")

    latest_row["matchup_factor"] = latest_row["matchup_factor"].fillna(1.0)
    latest_row["proj_fantasy_pts"] = (
        latest_row["trailing_avg_pts"].fillna(0) * latest_row["matchup_factor"]
    ).round(2)

    players = []
    for _, r in latest_row.iterrows():
        if pd.isna(r.get("opponent")):
            continue  # no upcoming game found for this team (bye week or season over)
        players.append({
            "player_id": r["player_id"],
            "player_name": r["player_display_name"],
            "position": r["position"],
            "team": r["team"],
            "opponent": r["opponent"],
            "headshot_url": r.get("headshot_url"),
            "trailing_avg_pts": None if pd.isna(r["trailing_avg_pts"]) else round(float(r["trailing_avg_pts"]), 2),
            "trailing_games": int(r["trailing_games"]),
            "proj_fantasy_pts": None if pd.isna(r["proj_fantasy_pts"]) else float(r["proj_fantasy_pts"]),
            "def_rank": None if pd.isna(r.get("def_rank")) else int(r["def_rank"]),
            "def_rank_of": None if pd.isna(r.get("def_rank_of")) else int(r["def_rank_of"]),
            "matchup_factor": round(float(r["matchup_factor"]), 3),
            "latest_snap_pct": None if pd.isna(r.get("latest_snap_pct")) else round(float(r["latest_snap_pct"]), 3),
        })

    players.sort(key=lambda p: (p["proj_fantasy_pts"] or 0), reverse=True)

    output = {
        "season": SEASON,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "trailing_L4_avg x opponent_matchup_factor (baseline, not yet validated against held-out data)",
        "players": players,
    }

    out_dir = os.path.dirname(os.path.abspath(OUT_JSON))
    os.makedirs(out_dir, exist_ok=True)

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    with open(OUT_JS, "w") as f:
        f.write("// Auto-generated by scripts/generate_fantasy_projections.py -- do not edit by hand.\n")
        f.write(f"const FANTASY_DATA = {json.dumps(output, indent=2)};\n")

    with open(OUT_LOG, "a") as f:
        for p in players:
            f.write(json.dumps({**p, "season": SEASON, "logged_at": output["generated_at"]}) + "\n")

    print(f"Wrote {len(players)} player projections to {OUT_JSON} and {OUT_JS}")


if __name__ == "__main__":
    main()

"""
team_features.py — Team-week modeling table for the Layer 2 team-TD model.

Combines:
  - Leakage-safe trailing team opportunity stats (rolling_features.build_asof_team_week)
  - Game context from the schedule: home/away, rest days, roof/surface, and —
    the single most useful Layer 2 signal — the Vegas-implied team total.

IMPLIED TEAM TOTAL, sign convention (verified against 2024 results):
    spread_line here = projected (home_score - away_score). Positive means the
    home team is favored. This is a "home margin" convention, not the
    traditional negative-favorite American-odds convention — verified by
    checking that home blowout wins in 2024 have large POSITIVE spread_line.

    implied_home_total = (total_line + spread_line) / 2
    implied_away_total = (total_line - spread_line) / 2

Public API:
    build_team_week_model_table(seasons) -> one row per (season, week, team)
        with asof_* trailing features + game context, ready to model.
"""

from __future__ import annotations
import pandas as pd

from nfl_data import load_pbp, load_snaps, load_id_crosswalk, load_schedules
from features import build_player_week_features
from rolling_features import build_asof_team_week


def _schedule_long(sched: pd.DataFrame) -> pd.DataFrame:
    """Melt the schedule to one row per (season, week, team) with that team's
    game context, computed from both the home and away perspective."""
    keep = [
        "season", "week", "game_id", "home_team", "away_team",
        "spread_line", "total_line", "home_rest", "away_rest",
        "roof", "surface", "temp", "wind", "div_game",
    ]
    s = sched[[c for c in keep if c in sched.columns]].copy()
    s["implied_home_total"] = (s["total_line"] + s["spread_line"]) / 2
    s["implied_away_total"] = (s["total_line"] - s["spread_line"]) / 2

    home = s.rename(columns={
        "home_team": "posteam", "away_team": "opponent",
        "implied_home_total": "implied_team_total",
        "implied_away_total": "implied_opp_total",
        "home_rest": "rest_days", "away_rest": "opp_rest_days",
    })
    home["is_home"] = 1

    away = s.rename(columns={
        "away_team": "posteam", "home_team": "opponent",
        "implied_away_total": "implied_team_total",
        "implied_home_total": "implied_opp_total",
        "away_rest": "rest_days", "home_rest": "opp_rest_days",
    })
    away["is_home"] = 0

    cols = [
        "season", "week", "posteam", "opponent", "is_home",
        "implied_team_total", "implied_opp_total", "rest_days", "opp_rest_days",
        "roof", "surface", "temp", "wind", "div_game",
    ]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def build_team_week_model_table(seasons: list[int]) -> pd.DataFrame:
    """
    Full Layer-2-ready team-week table: trailing opportunity features (leakage-
    safe) + this week's Vegas-implied total and game context (NOT leaky — the
    line is set before kickoff, so unlike trailing stats it's legitimately
    available at prediction time and used AS-IS, un-lagged).
    """
    pbp = load_pbp(seasons)
    snaps = load_snaps(seasons)
    xwalk = load_id_crosswalk()
    sched = load_schedules(seasons)

    player_feat = build_player_week_features(pbp, snaps, id_crosswalk=xwalk)
    team_asof = build_asof_team_week(player_feat)

    game_ctx = _schedule_long(sched)

    out = team_asof.merge(game_ctx, on=["season", "week", "posteam"], how="left")
    return out


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    df = build_team_week_model_table([yr])
    print(f"team-week rows: {len(df):,}  cols: {len(df.columns)}")
    print(df[["season", "week", "posteam", "opponent", "is_home",
              "implied_team_total", "team_actual_tds",
              "asof_roll4_team_rush_opp", "asof_roll4_team_pass_opp"]].head(10))

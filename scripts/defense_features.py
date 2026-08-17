"""
defense_features.py — Position-specific, leakage-safe opponent matchup
ratings, built from real play-by-play (not a single blended points-allowed
number).

WHY THIS EXISTS: the blueprint PDF's "matchup rating >= 1.10" threshold is
named without a formula. Rather than guess with one blended number (e.g.
just points allowed), this builds what the PDF's own data-requirements
section actually asks for: "Touchdowns allowed by position and play type,
adjusted for opponent strength" and "Red-zone touchdown percentage
allowed." A RB's matchup should be judged against how many RUSHING TDs a
defense allows; a WR/TE's against PASSING TDs allowed -- not the same
number for both.

WHAT'S COMPUTED, per team-week (as the DEFENSE that week):
  - rush_td_allowed, rush_opp_faced -> rushing-TD rate allowed
  - pass_td_allowed, pass_opp_faced -> passing-TD rate allowed
  - rz_td_allowed, rz_plays_faced   -> red-zone-play TD rate allowed
All leakage-safe (asof_roll4/asof_cum via the same shift-before-aggregate
mechanism as rolling_features.py), including for a genuinely future/
unplayed week -- same stub-row-free approach as game_features.py's pace
fix: raw defensive counts (from played weeks only) are merged onto the
FULL schedule-based team-week table (which has rows for future weeks too),
then trailed. A future week's own defensive stats are unknown (NaN,
correctly), but its trailing value is still computed correctly from real
prior games.

MATCHUP RATING = opponent's trailing TD-rate-allowed for the relevant play
type, divided by the LEAGUE-AVERAGE trailing rate across all 32 teams that
same week (not a fixed historical constant -- the league scoring
environment drifts week to week and season to season, so a rating of 1.10
means "10% worse than the *current* league average," not an arbitrary
historical baseline). Above 1.0 = softer than average defense for that
specific play type.

STATUS: informational only right now, not yet a hard qualifier in
blueprint_qualification.py -- see that module's docstring. This gives real
numbers to look at before deciding on a validated threshold, rather than
gating plays on an untested formula.

Public API:
    build_matchup_ratings(seasons) -> long-format (season, week, defteam)
        table with asof_* trailing counts, rates, and league-normalized
        matchup_rating_rush / matchup_rating_pass, safe to merge onto
        player rows by opponent.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from nfl_data import load_pbp, load_schedules
from rolling_features import _asof_trailing

ROLL_WINDOWS = (4, 8)
DEFENSE_COUNT_COLS = [
    "rush_opp_faced", "rush_td_allowed",
    "pass_opp_faced", "pass_td_allowed",
    "rz_plays_faced", "rz_td_allowed",
]


def _raw_defense_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, defteam) with real counts from that
    week's actual plays -- only exists for weeks that have been played."""
    df = pbp.dropna(subset=["defteam", "play_type"]).copy()

    rush = df[df["play_type"] == "run"]
    pas = df[df["play_type"] == "pass"]
    rz = df[df["yardline_100"] <= 20]

    keys = ["season", "week", "defteam"]
    rush_agg = rush.groupby(keys).agg(
        rush_opp_faced=("play_type", "size"),
        rush_td_allowed=("rush_touchdown", "sum"),
    )
    pass_agg = pas.groupby(keys).agg(
        pass_opp_faced=("play_type", "size"),
        pass_td_allowed=("pass_touchdown", "sum"),
    )
    rz_agg = rz.groupby(keys).agg(
        rz_plays_faced=("play_type", "size"),
        rz_td_allowed=("touchdown", "sum"),
    )

    out = rush_agg.join(pass_agg, how="outer").join(rz_agg, how="outer").fillna(0).reset_index()
    return out


def _team_week_long(sched: pd.DataFrame) -> pd.DataFrame:
    """Every team gets a row for every week it has a scheduled game
    (played or not) -- this is what lets a future week's trailing defense
    still resolve, the same mechanism as game_features.py's pace fix."""
    keep = ["season", "week", "home_team", "away_team"]
    s = sched[keep].copy()
    home = s.rename(columns={"home_team": "defteam"}).drop(columns=["away_team"])
    away = s.rename(columns={"away_team": "defteam"}).drop(columns=["home_team"])
    return pd.concat([home, away], ignore_index=True)


def build_matchup_ratings(seasons: list[int]) -> pd.DataFrame:
    sched = load_schedules(seasons)
    played_seasons = [
        s for s in seasons
        if sched.loc[sched["season"] == s, "home_score"].notna().any()
    ]
    pbp = load_pbp(played_seasons) if played_seasons else pd.DataFrame()

    long = _team_week_long(sched)
    if len(pbp):
        raw = _raw_defense_counts(pbp)
        long = long.merge(raw, on=["season", "week", "defteam"], how="left")
    else:
        for c in DEFENSE_COUNT_COLS:
            long[c] = np.nan

    trailing = _asof_trailing(long, "defteam", DEFENSE_COUNT_COLS, ROLL_WINDOWS)
    long = long.merge(trailing, on=["defteam", "season", "week"], how="left")

    for w in list(ROLL_WINDOWS) + ["cum"]:
        tag = f"roll{w}" if w != "cum" else "cum"
        rush_opp = long[f"asof_{tag}_rush_opp_faced"]
        pass_opp = long[f"asof_{tag}_pass_opp_faced"]
        rz_plays = long[f"asof_{tag}_rz_plays_faced"]

        long[f"asof_{tag}_rush_td_rate_allowed"] = np.where(
            rush_opp > 0, long[f"asof_{tag}_rush_td_allowed"] / rush_opp, np.nan
        )
        long[f"asof_{tag}_pass_td_rate_allowed"] = np.where(
            pass_opp > 0, long[f"asof_{tag}_pass_td_allowed"] / pass_opp, np.nan
        )
        long[f"asof_{tag}_rz_td_rate_allowed"] = np.where(
            rz_plays > 0, long[f"asof_{tag}_rz_td_allowed"] / rz_plays, np.nan
        )

        # League-normalized: this week's rate vs the mean across all 32
        # teams' trailing rates THAT SAME WEEK -- not a fixed historical
        # constant, since league scoring environment drifts over time.
        for stat in ["rush_td_rate_allowed", "pass_td_rate_allowed"]:
            col = f"asof_{tag}_{stat}"
            league_avg = long.groupby(["season", "week"])[col].transform("mean")
            long[f"matchup_rating_{tag}_{stat.split('_')[0]}"] = np.where(
                league_avg > 0, long[col] / league_avg, np.nan
            )

    return long


if __name__ == "__main__":
    print("Run test_defense_features.py to validate this module.")

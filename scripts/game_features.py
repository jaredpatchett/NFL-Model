"""
game_features.py — Team-game modeling table for the Track B (spread / total /
moneyline) model.

Combines:
  - power_ratings.py: leakage-safe, opponent-adjusted SRS-style ratings
  - Leakage-safe trailing points-scored / points-allowed (pace/scoring level,
    separate from the opponent-adjusted rating -- helps the total-points model
    pick up teams in shootout-prone or defense-heavy stretches)
  - Schedule context: home/away, rest days, div game
  - Market lines (moneyline, spread, total + their odds) carried through
    UNCHANGED for backtesting/edge comparison -- these are NOT used as model
    inputs, only as a comparison point after prediction.

Public API:
    build_game_model_table(seasons) -> one row per (season, week, game_id,
        home/away) with model features + market lines for comparison.
"""

from __future__ import annotations
import pandas as pd

from nfl_data import load_schedules
from power_ratings import build_power_ratings
from rolling_features import _asof_trailing
from team_features import build_team_week_model_table

ROLL_WINDOWS = (4, 8)


def _team_game_long(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with that team's own score/opp score,
    home/away flag, rest, and the game's market lines (kept in team-perspective
    form: home team's spread/moneyline as-is, away team's flipped)."""
    keep = [
        "season", "week", "game_id", "home_team", "away_team",
        "home_score", "away_score", "home_rest", "away_rest", "div_game",
        "home_moneyline", "away_moneyline", "spread_line", "total_line",
        "roof", "surface", "temp", "wind",
    ]
    s = sched[[c for c in keep if c in sched.columns]].copy()

    home = s.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "team_score", "away_score": "opp_score",
        "home_rest": "rest_days", "away_rest": "opp_rest_days",
        "home_moneyline": "team_moneyline", "away_moneyline": "opp_moneyline",
    })
    home["is_home"] = 1
    home["team_spread_line"] = home["spread_line"]        # home perspective, as-is

    away = s.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "team_score", "home_score": "opp_score",
        "away_rest": "rest_days", "home_rest": "opp_rest_days",
        "away_moneyline": "team_moneyline", "home_moneyline": "opp_moneyline",
    })
    away["is_home"] = 0
    away["team_spread_line"] = -away["spread_line"]       # flip to away perspective

    cols = [
        "season", "week", "game_id", "team", "opponent", "is_home",
        "team_score", "opp_score", "rest_days", "opp_rest_days", "div_game",
        "team_moneyline", "opp_moneyline", "team_spread_line", "total_line",
        "roof", "surface", "temp", "wind",
    ]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def build_game_model_table(seasons: list[int]) -> pd.DataFrame:
    """
    Returns one row per (season, week, team) with model features. Includes
    FUTURE/unplayed games (team_score/opp_score/team_margin/game_total will
    be NaN for those rows) -- this is intentional so the production pipeline
    can attach leakage-safe as-of features to an upcoming week's games and
    generate predictions for them. Callers that train/validate a model must
    filter to played games themselves (see game_lines_model.py's _prep()).
    """
    sched = load_schedules(seasons)

    long = _team_game_long(sched)
    long["team_margin"] = long["team_score"] - long["opp_score"]
    long["game_total"] = long["team_score"] + long["opp_score"]

    # Opponent-adjusted power ratings, as-of each week (already leakage-safe).
    ratings = build_power_ratings(sched)
    ratings = ratings.rename(columns={"team": "team"})
    long = long.merge(
        ratings[["season", "week", "team", "power_rating", "hfa_as_of_week"]],
        on=["season", "week", "team"], how="left",
    )
    opp_ratings = ratings.rename(columns={"team": "opponent", "power_rating": "opp_power_rating"})
    long = long.merge(
        opp_ratings[["season", "week", "opponent", "opp_power_rating"]],
        on=["season", "week", "opponent"], how="left",
    )

    # Trailing points scored/allowed -- pace/scoring-level signal, separate
    # from (but complementary to) the opponent-adjusted power rating.
    # NOTE: _asof_trailing re-sorts its input internally, so this MUST be a
    # keyed merge, not a positional concat (bit us once already in
    # rolling_features.py's team-week builder -- same failure mode).
    trailing = _asof_trailing(long, "team", ["team_score", "opp_score", "game_total"], ROLL_WINDOWS)
    long = long.merge(trailing, on=["team", "season", "week"], how="left")

    # ---- Weather / venue features ----
    # Indoors (dome/closed roof), weather doesn't affect play -- impute a
    # neutral value rather than the outdoor average, and flag it explicitly
    # so the model can learn "indoors" as its own signal rather than being
    # fed a fake temp/wind reading.
    long["is_indoor"] = long["roof"].isin(["dome", "closed"]).astype(float)
    outdoor_temp_median = long.loc[long["is_indoor"] == 0, "temp"].median()
    outdoor_wind_median = long.loc[long["is_indoor"] == 0, "wind"].median()
    long["temp_filled"] = long["temp"]
    long.loc[long["is_indoor"] == 1, "temp_filled"] = 70.0  # neutral indoor temp
    long["temp_filled"] = long["temp_filled"].fillna(outdoor_temp_median)
    long["wind_filled"] = long["wind"]
    long.loc[long["is_indoor"] == 1, "wind_filled"] = 0.0  # no wind indoors
    long["wind_filled"] = long["wind_filled"].fillna(outdoor_wind_median)

    # ---- Trailing pace (plays run) -- reuses the play-count data pulled for
    # the Layer 2 TD model (team_features.py's pbp-based pipeline). Total
    # points is partly just a function of how many plays get run, so this is
    # complementary to the points-scored trailing feature.
    #
    # IMPORTANT: team_week_model_table only has rows for WEEKS THAT HAVE BEEN
    # PLAYED (it's built from play-by-play, which doesn't exist for a future
    # game). A direct merge would leave the upcoming week's pace as NaN --
    # exactly the row a production prediction needs. Fix: attach the raw
    # (unshifted) play counts for played weeks onto the full `long` table
    # (which does have a row for the upcoming week), THEN run it through the
    # same _asof_trailing shift/rolling used for points above. Shift(1) never
    # uses a row's own value, so the upcoming week's own (unknown) play count
    # being NaN doesn't matter -- its trailing feature is still correctly
    # built from the prior (played) weeks, the same way team_score is.
    # Only pull pbp-based pace for seasons that actually have at least one
    # played game -- a season before its first kickoff (e.g. next season's
    # already-published schedule) has no play-by-play data to fetch at all;
    # requesting it 404s. The upcoming week's pace still resolves correctly
    # via the trailing merge below, sourced from whichever seasons DO have
    # played games.
    pbp_seasons = [
        s for s in seasons
        if sched.loc[sched["season"] == s, "home_score"].notna().any()
    ]
    raw_pace = build_team_week_model_table(pbp_seasons) if pbp_seasons else pd.DataFrame(
        columns=["season", "week", "team", "team_plays"]
    )
    raw_pace = raw_pace.rename(columns={"posteam": "team"})
    if "team_rush_opp" in raw_pace.columns:
        raw_pace["team_plays"] = raw_pace["team_rush_opp"] + raw_pace["team_pass_opp"]
    long = long.merge(
        raw_pace[["season", "week", "team", "team_plays"]],
        on=["season", "week", "team"], how="left",
    )
    pace_trailing = _asof_trailing(long, "team", ["team_plays"], ROLL_WINDOWS)
    long = long.merge(pace_trailing, on=["team", "season", "week"], how="left")

    return long


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    df = build_game_model_table([yr])
    print(f"rows: {len(df):,}  cols: {len(df.columns)}")
    print(df[["season", "week", "team", "opponent", "is_home", "power_rating",
              "opp_power_rating", "team_spread_line", "total_line",
              "asof_roll4_team_score", "team_margin"]].head(10))

"""
rolling_features.py — Leakage-safe, "as-of-kickoff" feature builder.

WHY THIS FILE EXISTS (read this before touching Layer 2/3):

`features.py` builds per-player, per-week opportunity shares FROM that week's
own plays. That's correct for describing what already happened, but it is NOT
a valid model input for predicting that same week's outcome — by the time
week 9's carry_share exists, week 9 has already been played. Feeding it to a
model that predicts week 9's TD probability is hindsight leakage: the model
would be predicting the past using a summary of the past.

This module turns the per-week descriptive table from features.py into
"as-of-kickoff" features: for every (player, season, week) row, every number
is computed ONLY from that player's/team's games strictly BEFORE that week.
These are the features Layer 2/3 models should actually train and predict on.

Two flavors, both included:
  - asof_cum_*     : season-to-date cumulative totals, prior weeks only, reset
                      each season
  - asof_roll{N}_* : trailing N-game window, prior weeks only, can span a
                      season boundary (N in ROLL_WINDOWS)

Shares are recomputed as sum(prior numerator) / sum(prior denominator), not as
an average of prior weekly share values — this avoids overweighting weeks with
tiny opportunity counts (e.g. 1 target out of 1 = 100% target share is noise).

Public API:
    build_asof_team_week(player_week_feat)   -> team-week trailing table
    build_asof_player_features(player_week_feat, windows=(4, 8)) -> player-week
        table of asof_* columns, safe to join onto a modeling frame
"""

from __future__ import annotations
import numpy as np
import pandas as pd

ROLL_WINDOWS = (4, 8)

# Raw (summable) count columns carried at the player-week grain.
PLAYER_COUNT_COLS = [
    "rush_opp", "rush_td", "rush_inside5", "rush_inside10", "rush_rz", "rush_field",
    "pass_opp", "pass_td", "pass_inside5", "pass_inside10", "pass_rz", "pass_field",
    "expected_tds", "actual_tds", "scored_td",
]

# Raw (summable) count columns carried at the team-week grain.
TEAM_COUNT_COLS = [
    "team_rush_opp", "team_pass_opp", "team_rush_inside5", "team_rush_inside10",
    "team_rush_rz", "team_pass_rz", "team_pass_inside10",
]


def _asof_trailing(
    df: pd.DataFrame, group_col: str, count_cols: list[str], windows: tuple[int, ...]
) -> pd.DataFrame:
    """
    Core leakage-safe transform. For every row, computes:
      - asof_cum_{col}    : cumsum of {col} over prior weeks THIS season only
      - asof_roll{N}_{col}: rolling sum of {col} over the prior N games (any season)
    via shift(1) BEFORE any aggregation, so the current row's own value can
    never enter its own features.

    Returns a frame indexed identically to the (sorted) input, with
    [group_col, season, week] + the asof_* columns, plus asof_games_played_prior.
    """
    df = df.sort_values([group_col, "season", "week"]).reset_index(drop=True)
    out = df[[group_col, "season", "week"]].copy()

    for col in count_cols:
        shifted = df.groupby(group_col)[col].shift(1)

        out[f"asof_cum_{col}"] = shifted.groupby([df[group_col], df["season"]]).cumsum()

        for w in windows:
            out[f"asof_roll{w}_{col}"] = shifted.groupby(df[group_col]).transform(
                lambda s, w=w: s.rolling(w, min_periods=1).sum()
            )

    out["asof_games_played_prior"] = df.groupby([group_col, "season"]).cumcount()
    return out


def build_asof_team_week(player_week_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the player-week table to one row per (season, week, posteam) with
    trailing team-level opportunity totals — the denominators for share
    features, and the raw material for the Layer 2 team-TD model.
    """
    team = (
        player_week_feat
        .groupby(["season", "week", "posteam"], as_index=False)[TEAM_COUNT_COLS]
        .first()  # team totals are duplicated identically across a team's player rows
    )
    team_tds = (
        player_week_feat
        .groupby(["season", "week", "posteam"], as_index=False)["actual_tds"]
        .sum()
        .rename(columns={"actual_tds": "team_actual_tds"})
    )
    team = team.merge(team_tds, on=["season", "week", "posteam"], how="left")

    asof = _asof_trailing(team, "posteam", TEAM_COUNT_COLS + ["team_actual_tds"], ROLL_WINDOWS)
    team = team.merge(asof, on=["posteam", "season", "week"], how="left")
    return team


def build_asof_player_features(
    player_week_feat: pd.DataFrame, windows: tuple[int, ...] = ROLL_WINDOWS
) -> pd.DataFrame:
    """
    Build the as-of-kickoff player feature table: original identifying columns
    (season, week, posteam, player_id) plus asof_cum_* / asof_roll{N}_* raw
    trailing totals and recomputed asof_*_share ratios — all leakage-safe.
    """
    df = player_week_feat.copy()

    player_asof = _asof_trailing(df, "player_id", PLAYER_COUNT_COLS, windows)
    player_asof = player_asof.rename(
        columns={"asof_games_played_prior": "asof_player_games_prior"}
    )

    team_week = build_asof_team_week(df)
    team_asof_cols = [c for c in team_week.columns if c.startswith("asof_")]
    team_asof = team_week[["season", "week", "posteam"] + team_asof_cols].rename(
        columns={c: f"team_{c}" for c in team_asof_cols}
    )

    base = df[["season", "week", "posteam", "player_id"]].drop_duplicates()
    out = base.merge(player_asof, on=["player_id", "season", "week"], how="left")
    out = out.merge(team_asof, on=["season", "week", "posteam"], how="left")

    # ---- Recompute shares from trailing sums (sum/sum, not mean-of-weekly-shares) ----
    def _safe_share(num_col, den_col):
        num, den = out[num_col], out[den_col]
        return np.where(den > 0, num / den, np.nan)

    for tag in [f"roll{w}" for w in windows] + ["cum"]:
        out[f"asof_{tag}_carry_share"] = _safe_share(
            f"asof_{tag}_rush_opp", f"team_asof_{tag}_team_rush_opp"
        )
        out[f"asof_{tag}_target_share"] = _safe_share(
            f"asof_{tag}_pass_opp", f"team_asof_{tag}_team_pass_opp"
        )
        out[f"asof_{tag}_inside5_carry_share"] = _safe_share(
            f"asof_{tag}_rush_inside5", f"team_asof_{tag}_team_rush_inside5"
        )
        out[f"asof_{tag}_rz_target_share"] = _safe_share(
            f"asof_{tag}_pass_rz", f"team_asof_{tag}_team_pass_rz"
        )

    return out


if __name__ == "__main__":
    print("Run test_rolling_features.py to validate this module against synthetic data.")

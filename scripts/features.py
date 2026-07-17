"""
features.py — Feature engineering for the NFL touchdown model.

Turns raw nflverse play-by-play + snap counts into the opportunity features
the blueprint calls for:

  - snap_share            (from snap counts)
  - rz_carry_share        red-zone (inside 20) carries / team rz carries
  - rz_target_share       red-zone targets / team rz targets
  - inside5_carry_share   inside-5 carries / team inside-5 carries  (goal-line role)
  - inside10_opp_share    inside-10 carries + targets share
  - carry_share           total carries / team carries
  - target_share          total targets / team targets
  - expected_tds          crude xTD from opportunity-weighted scoring rates
  - actual_tds            rushing + receiving TDs

Design notes:
  * All shares are computed WITHIN team-week, so they sum sensibly across
    a team's players.
  * Expected TDs use league-average conversion rates by field-position bucket,
    which is the "opportunity predicts TDs better than TDs themselves"
    principle from the blueprint's blind-spots section.
  * Everything here is pure pandas — no network — so it is unit-testable.

Public API:
    build_player_week_features(pbp, snaps) -> DataFrame keyed by
        (season, week, team, player_id)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# League-average TD conversion rates per opportunity by field-position bucket.
# These are sensible priors; the real pipeline should estimate them from data
# each season. Used only to build an expected-TD feature, not the final prob.
XTD_RUSH_RATES = {"inside5": 0.34, "inside10": 0.18, "rz": 0.09, "field": 0.007}
XTD_PASS_RATES = {"inside5": 0.28, "inside10": 0.17, "rz": 0.11, "field": 0.010}


def _fp_bucket(yardline_100: pd.Series) -> pd.Series:
    """Bucket a play by distance to the end zone (yardline_100 = yards to goal)."""
    return pd.cut(
        yardline_100,
        bins=[-0.1, 5, 10, 20, 100],
        labels=["inside5", "inside10", "rz", "field"],
    )


def build_player_week_features(
    pbp: pd.DataFrame,
    snaps: pd.DataFrame,
    id_crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build per-player, per-week opportunity features.

    Expects nflverse pbp columns:
        season, week, posteam, yardline_100, play_type,
        rusher_player_id, receiver_player_id,
        rush_touchdown, pass_touchdown, touchdown
    Expects nflverse snaps columns:
        season, week, team, player, pfr_player_id, offense_pct

    IMPORTANT: pbp/weekly player IDs are GSIS IDs ("00-0034796"); nflverse snap
    counts are keyed by PFR IDs ("BrowSp00") — a different ID namespace entirely.
    Pass `id_crosswalk` (from nfl_data.load_id_crosswalk(), columns gsis_id/pfr_id)
    to translate snap counts onto GSIS IDs before merging. Without it, snap_share
    will merge on mismatched keys and come back all-null (this bit us once already —
    the synthetic test in test_features.py didn't catch it because the synthetic
    fixture coincidentally used the same fake ID for both systems).
    """
    pbp = pbp.copy()
    pbp["fp"] = _fp_bucket(pbp["yardline_100"])

    # ---- Rushing opportunities ----
    rush = pbp[pbp["play_type"] == "run"].copy()
    rush = rush.dropna(subset=["rusher_player_id"])
    rush = rush.rename(columns={"rusher_player_id": "player_id"})
    rush["is_rush"] = 1
    rush["rush_td"] = rush["rush_touchdown"].fillna(0)

    # ---- Passing opportunities (targets) ----
    pas = pbp[pbp["play_type"] == "pass"].copy()
    pas = pas.dropna(subset=["receiver_player_id"])
    pas = pas.rename(columns={"receiver_player_id": "player_id"})
    pas["is_target"] = 1
    pas["rec_td"] = pas["pass_touchdown"].fillna(0)

    keys = ["season", "week", "posteam", "player_id"]

    def _agg_by_bucket(df, opp_col, td_col, prefix):
        # total opportunities + TDs
        base = df.groupby(keys).agg(
            **{f"{prefix}_opp": (opp_col, "sum"), f"{prefix}_td": (td_col, "sum")}
        )
        # opportunities by field-position bucket
        buckets = (
            df.groupby(keys + ["fp"], observed=True)[opp_col]
            .sum()
            .unstack("fp", fill_value=0)
        )
        buckets.columns = [f"{prefix}_{c}" for c in buckets.columns]
        return base.join(buckets, how="outer")

    rush_agg = _agg_by_bucket(rush, "is_rush", "rush_td", "rush")
    pass_agg = _agg_by_bucket(pas, "is_target", "rec_td", "pass")

    feat = rush_agg.join(pass_agg, how="outer").fillna(0).reset_index()

    # Ensure all bucket columns exist even if absent in a small sample
    for pfx in ("rush", "pass"):
        for b in ("inside5", "inside10", "rz", "field"):
            col = f"{pfx}_{b}"
            if col not in feat.columns:
                feat[col] = 0.0

    # ---- Team totals for share computation ----
    team_keys = ["season", "week", "posteam"]
    team = feat.groupby(team_keys).agg(
        team_rush_opp=("rush_opp", "sum"),
        team_pass_opp=("pass_opp", "sum"),
        team_rush_inside5=("rush_inside5", "sum"),
        team_rush_inside10=("rush_inside10", "sum"),
        team_rush_rz=("rush_rz", "sum"),
        team_pass_rz=("pass_rz", "sum"),
        team_pass_inside10=("pass_inside10", "sum"),
    ).reset_index()

    feat = feat.merge(team, on=team_keys, how="left")

    def _share(num, den):
        return np.where(feat[den] > 0, feat[num] / feat[den], 0.0)

    feat["carry_share"] = _share("rush_opp", "team_rush_opp")
    feat["target_share"] = _share("pass_opp", "team_pass_opp")
    feat["inside5_carry_share"] = _share("rush_inside5", "team_rush_inside5")
    feat["rz_carry_share"] = _share("rush_rz", "team_rush_rz")
    feat["rz_target_share"] = _share("pass_rz", "team_pass_rz")

    feat["inside10_opp_share"] = np.where(
        (feat["team_rush_inside10"] + feat["team_pass_inside10"]) > 0,
        (feat["rush_inside10"] + feat["pass_inside10"])
        / (feat["team_rush_inside10"] + feat["team_pass_inside10"]),
        0.0,
    )

    # ---- Expected TDs from opportunity (not from actual TDs) ----
    feat["expected_tds"] = (
        feat["rush_inside5"] * XTD_RUSH_RATES["inside5"]
        + feat["rush_inside10"] * XTD_RUSH_RATES["inside10"]
        + feat["rush_rz"] * XTD_RUSH_RATES["rz"]
        + feat["rush_field"] * XTD_RUSH_RATES["field"]
        + feat["pass_inside5"] * XTD_PASS_RATES["inside5"]
        + feat["pass_inside10"] * XTD_PASS_RATES["inside10"]
        + feat["pass_rz"] * XTD_PASS_RATES["rz"]
        + feat["pass_field"] * XTD_PASS_RATES["field"]
    )

    feat["actual_tds"] = feat["rush_td"] + feat["pass_td"]
    feat["scored_td"] = (feat["actual_tds"] > 0).astype(int)  # the model target

    # ---- Merge snap share ----
    if snaps is not None and len(snaps):
        s = snaps.rename(columns={"team": "posteam"})
        if id_crosswalk is not None and len(id_crosswalk):
            # Translate PFR id -> GSIS id so this actually joins onto pbp-derived features.
            xwalk = id_crosswalk[["gsis_id", "pfr_id"]].rename(
                columns={"pfr_id": "pfr_player_id"}
            )
            s = s.merge(xwalk, on="pfr_player_id", how="left")
            s = s.rename(columns={"gsis_id": "player_id"})
        else:
            # Fallback (e.g. synthetic tests): assume pfr_player_id IS the join key.
            s = s.rename(columns={"pfr_player_id": "player_id"})
        s = s.dropna(subset=["player_id"])
        s = s[["season", "week", "posteam", "player_id", "offense_pct"]]
        # A player can in principle have duplicate rows post-crosswalk in edge cases
        # (rare id collisions); keep it safe with a groupby instead of a bare merge key.
        s = s.groupby(["season", "week", "posteam", "player_id"], as_index=False)["offense_pct"].mean()
        feat = feat.merge(s, on=["season", "week", "posteam", "player_id"], how="left")
        feat = feat.rename(columns={"offense_pct": "snap_share"})
    else:
        feat["snap_share"] = np.nan

    return feat


if __name__ == "__main__":
    print("Run test_features.py to validate this module against synthetic data.")

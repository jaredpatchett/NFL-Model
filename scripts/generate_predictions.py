"""
generate_predictions.py — Production entrypoint for Track B (spread / total /
moneyline). Meant to be run on a schedule (GitHub Actions cron): pulls the
latest data, trains on every played game available, finds the next unplayed
week, generates predictions for it, and writes data/nfl_lines.json +
data/nfl_lines.js for the static dashboard to read.

Unlike game_lines_model.py (which holds out 2024 to REPORT how good the
model is), this script trains on ALL played history -- there's no reason to
withhold data in production once the model's validity is already established.

Usage:
    python generate_predictions.py
"""

from __future__ import annotations
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from game_features import build_game_model_table
from game_lines_model import (
    FEATURE_COLS, MIN_GAMES_PRIOR, moneyline_to_implied_prob, prob_to_fair_moneyline,
)

# Generous range -- seasons that don't exist in the source data are silently
# filtered out (see nfl_data._fetch_schedules), so this is safe to over-request.
ALL_SEASONS = list(range(2021, 2028))

OUT_JSON = "../data/nfl_lines.json"
OUT_JS = "../data/nfl_lines.js"


def _prep_played(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["div_game"] = df["div_game"].fillna(0).astype(float)
    df["rest_days"] = df["rest_days"].fillna(df["rest_days"].median())
    df["opp_rest_days"] = df["opp_rest_days"].fillna(df["opp_rest_days"].median())
    df = df.dropna(subset=["power_rating", "opp_power_rating"])
    df = df.dropna(subset=["asof_roll4_team_score", "asof_roll4_team_plays"])
    df = df.dropna(subset=["team_score", "opp_score"])  # played games only
    return df.reset_index(drop=True)


def _prep_upcoming(df: pd.DataFrame, median_rest: float, median_opp_rest: float) -> pd.DataFrame:
    df = df.copy()
    df["div_game"] = df["div_game"].fillna(0).astype(float)
    df["rest_days"] = df["rest_days"].fillna(median_rest)
    df["opp_rest_days"] = df["opp_rest_days"].fillna(median_opp_rest)
    return df


def find_upcoming_week(full: pd.DataFrame) -> tuple[int, int] | None:
    """Earliest (season, week) with at least one unplayed game with a posted line."""
    unplayed = full[full["team_score"].isna() & full["total_line"].notna()]
    if unplayed.empty:
        return None
    row = unplayed[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).iloc[0]
    return int(row["season"]), int(row["week"])


def fit_ridge(train: pd.DataFrame, target_col: str):
    X, y = train[FEATURE_COLS], train[target_col]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=5.0)
    model.fit(X_s, y)
    residual_std = float(np.std(y - model.predict(X_s)))
    return model, scaler, residual_std


def main():
    print("Building game model table (all available seasons)...")
    full = build_game_model_table(ALL_SEASONS)

    target = find_upcoming_week(full)
    if target is None:
        print("No upcoming week with posted lines found. Nothing to predict.")
        return
    season, week = target
    print(f"Predicting: season={season} week={week}")

    played = _prep_played(full)
    print(f"Training rows (all played history, cold-start-filtered): {len(played)}")

    margin_model, margin_scaler, margin_resid_std = fit_ridge(played, "team_margin")
    total_model, total_scaler, total_resid_std = fit_ridge(played, "game_total")

    upcoming = full[(full["season"] == season) & (full["week"] == week)].copy()
    upcoming = _prep_upcoming(upcoming, played["rest_days"].median(), played["opp_rest_days"].median())
    # Same cold-start guard as training -- if a team genuinely has no trailing
    # data (shouldn't happen mid-season, but guards a true week-1-of-history edge case).
    missing_features = upcoming[FEATURE_COLS].isna().any(axis=1)
    if missing_features.any():
        print(f"WARNING: {missing_features.sum()} upcoming rows missing features, dropping:")
        print(upcoming.loc[missing_features, ["team", "opponent"]].to_string(index=False))
    upcoming = upcoming[~missing_features].reset_index(drop=True)

    X_up_margin = margin_scaler.transform(upcoming[FEATURE_COLS])
    X_up_total = total_scaler.transform(upcoming[FEATURE_COLS])
    upcoming["pred_margin"] = margin_model.predict(X_up_margin)
    upcoming["pred_total"] = total_model.predict(X_up_total)
    upcoming["model_win_prob"] = norm.cdf(upcoming["pred_margin"] / margin_resid_std)
    upcoming["model_fair_moneyline"] = upcoming["model_win_prob"].apply(prob_to_fair_moneyline)
    upcoming["market_win_prob"] = upcoming["team_moneyline"].apply(
        lambda ml: moneyline_to_implied_prob(ml) if pd.notna(ml) else np.nan
    )
    upcoming["spread_edge"] = upcoming["pred_margin"] - upcoming["team_spread_line"]
    upcoming["total_edge"] = upcoming["pred_total"] - upcoming["total_line"]
    upcoming["moneyline_edge"] = upcoming["model_win_prob"] - upcoming["market_win_prob"]

    # ---- Build output: one entry per GAME (not per team-perspective row) ----
    home_rows = upcoming[upcoming["is_home"] == 1].set_index(["season", "week", "game_id"])
    games = []
    for (s, w, gid), row in home_rows.iterrows():
        games.append({
            "game_id": gid,
            "season": int(s),
            "week": int(w),
            "home_team": row["team"],
            "away_team": row["opponent"],
            "model": {
                "pred_home_margin": round(float(row["pred_margin"]), 2),
                "pred_total": round(float(row["pred_total"]), 2),
                "home_win_prob": round(float(row["model_win_prob"]), 4),
                "home_fair_moneyline": round(float(row["model_fair_moneyline"]), 1),
            },
            "market": {
                "spread_line": row["team_spread_line"],  # home perspective
                "total_line": row["total_line"],
                "home_moneyline": row["team_moneyline"],
                "away_moneyline": row["opp_moneyline"],
                "home_implied_prob": (
                    round(float(row["market_win_prob"]), 4) if pd.notna(row["market_win_prob"]) else None
                ),
            },
            "edges": {
                "spread_edge": round(float(row["spread_edge"]), 2),
                "total_edge": round(float(row["total_edge"]), 2),
                "moneyline_edge": (
                    round(float(row["moneyline_edge"]), 4) if pd.notna(row["moneyline_edge"]) else None
                ),
            },
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "week": week,
        "model_notes": {
            "margin_residual_std": round(margin_resid_std, 2),
            "total_residual_std": round(total_resid_std, 2),
            "training_games": len(played),
            "caveat": (
                "Total-points model is weaker than margin (see README) -- "
                "treat total_edge with more skepticism than spread_edge."
            ),
        },
        "games": games,
    }

    import os
    out_dir = os.path.dirname(os.path.abspath(OUT_JSON))
    os.makedirs(out_dir, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    with open(OUT_JS, "w") as f:
        f.write("// Auto-generated by scripts/generate_predictions.py -- do not edit by hand.\n")
        f.write(f"const NFL_LINES_DATA = {json.dumps(output, indent=2)};\n")

    print(f"Wrote {len(games)} games to {OUT_JSON} and {OUT_JS}")


if __name__ == "__main__":
    main()

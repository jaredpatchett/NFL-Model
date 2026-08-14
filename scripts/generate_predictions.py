"""
generate_predictions.py — Production entrypoint for Track B (spread / total /
moneyline). Meant to be run on a schedule (GitHub Actions cron): pulls the
latest data, trains on every played game available, finds the next unplayed
week, generates predictions for it, and writes:

  - data/nfl_lines.json + data/nfl_lines.js -- current snapshot, OVERWRITTEN
    each run. This is what the dashboard reads; it only ever shows "the
    latest prediction."
  - data/predictions_log.jsonl -- APPEND-ONLY history, one line per
    (game, run). Every cron run adds a fresh timestamped row for every game
    in the upcoming week, even if nothing changed. This is the actual
    backtesting asset: a genuine pre-game record of what the model said and
    what the market showed, before any outcome was known. The project's
    validation so far has only ever checked calibration against CLOSING
    lines (see README) -- that's a real limitation, not a valid
    profitability backtest. This log is how that gets fixed over time: once
    enough weeks accumulate, join this against final scores (already
    available in the schedule data once games are played) to get a genuine
    time-stamped edge/ROI analysis.

Unlike game_lines_model.py (which holds out 2024 to REPORT how good the
model is), this script trains on ALL played history -- there's no reason to
withhold data in production once the model's validity is already established.

Usage:
    python generate_predictions.py
"""

from __future__ import annotations
import json
import os
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
import odds_api

# Generous range -- seasons that don't exist in the source data are silently
# filtered out (see nfl_data._fetch_schedules), so this is safe to over-request.
ALL_SEASONS = list(range(2021, 2028))

OUT_JSON = "../data/nfl_lines.json"
OUT_JS = "../data/nfl_lines.js"
OUT_LOG = "../data/predictions_log.jsonl"


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

    # ---- Build output: one entry per GAME (not per team-perspective row) ----
    print("\nFetching live odds (The Odds API)...")
    live_odds = odds_api.fetch_game_odds()
    if live_odds is not None:
        print(f"  Got live odds for {len(live_odds)} games.")
    else:
        print("  No live odds available -- using schedule-riding lines only (no EV, edges still computed).")

    home_rows = upcoming[upcoming["is_home"] == 1].set_index(["season", "week", "game_id"])
    games = []
    for (s, w, gid), row in home_rows.iterrows():
        live = None
        if live_odds is not None:
            match = live_odds[
                (live_odds["home_team"] == row["team"]) & (live_odds["away_team"] == row["opponent"])
            ]
            if len(match):
                live = match.iloc[0]

        # Prefer live odds when available; fall back to the schedule-riding
        # lines (still real market data, just not necessarily current/best-price).
        if live is not None and pd.notna(live["live_home_spread_point"]):
            spread_line = float(live["live_home_spread_point"])
            spread_price = live["live_home_spread_price"]
            total_line = float(live["live_total_point"])
            home_ml = live["live_home_moneyline"]
            away_ml = live["live_away_moneyline"]
            market_source = "live"
        else:
            spread_line = row["team_spread_line"]
            spread_price = None
            total_line = row["total_line"]
            home_ml = row["team_moneyline"]
            away_ml = row["opp_moneyline"]
            market_source = "schedule_fallback"

        spread_edge = float(row["pred_margin"]) - spread_line if pd.notna(spread_line) else None
        total_edge = float(row["pred_total"]) - total_line if pd.notna(total_line) else None
        home_implied = moneyline_to_implied_prob(home_ml) if pd.notna(home_ml) else None
        moneyline_edge = (
            float(row["model_win_prob"]) - home_implied if home_implied is not None else None
        )
        # EV per $1 stake at the best available home moneyline price.
        home_ev = None
        if pd.notna(home_ml):
            p = float(row["model_win_prob"])
            profit = home_ml / 100 if home_ml > 0 else 100 / -home_ml
            home_ev = round(p * profit - (1 - p), 4)

        games.append({
            "game_id": gid,
            "season": int(s),
            "week": int(w),
            "home_team": row["team"],
            "away_team": row["opponent"],
            "market_source": market_source,  # "live" (The Odds API) or "schedule_fallback"
            "model": {
                "pred_home_margin": round(float(row["pred_margin"]), 2),
                "pred_total": round(float(row["pred_total"]), 2),
                "home_win_prob": round(float(row["model_win_prob"]), 4),
                "home_fair_moneyline": round(float(row["model_fair_moneyline"]), 1),
            },
            "market": {
                "spread_line": spread_line,  # home perspective
                "spread_price": spread_price,
                "total_line": total_line,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "home_implied_prob": round(home_implied, 4) if home_implied is not None else None,
            },
            "edges": {
                "spread_edge": round(spread_edge, 2) if spread_edge is not None else None,
                "total_edge": round(total_edge, 2) if total_edge is not None else None,
                "moneyline_edge": round(moneyline_edge, 4) if moneyline_edge is not None else None,
                "home_moneyline_ev": home_ev,
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

    out_dir = os.path.dirname(os.path.abspath(OUT_JSON))
    os.makedirs(out_dir, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    with open(OUT_JS, "w") as f:
        f.write("// Auto-generated by scripts/generate_predictions.py -- do not edit by hand.\n")
        f.write(f"const NFL_LINES_DATA = {json.dumps(output, indent=2)};\n")

    # ---- Append-only history: one line per (game, this run) ----
    # This is the file that makes a real backtest possible later -- it's the
    # only place a pre-game snapshot survives once the next run overwrites
    # nfl_lines.json. Every run adds new rows; nothing already logged is
    # ever modified or removed here.
    logged_at = output["generated_at"]
    log_lines = []
    for g in games:
        log_lines.append({
            "logged_at": logged_at,
            "season": g["season"],
            "week": g["week"],
            "game_id": g["game_id"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "pred_home_margin": g["model"]["pred_home_margin"],
            "pred_total": g["model"]["pred_total"],
            "home_win_prob": g["model"]["home_win_prob"],
            "home_fair_moneyline": g["model"]["home_fair_moneyline"],
            "market_source": g["market_source"],
            "market_spread_line": g["market"]["spread_line"],
            "market_total_line": g["market"]["total_line"],
            "market_home_moneyline": g["market"]["home_moneyline"],
            "market_away_moneyline": g["market"]["away_moneyline"],
            "market_home_implied_prob": g["market"]["home_implied_prob"],
            "spread_edge": g["edges"]["spread_edge"],
            "total_edge": g["edges"]["total_edge"],
            "moneyline_edge": g["edges"]["moneyline_edge"],
            "home_moneyline_ev": g["edges"]["home_moneyline_ev"],
        })

    with open(OUT_LOG, "a") as f:
        for line in log_lines:
            f.write(json.dumps(line) + "\n")

    print(f"Wrote {len(games)} games to {OUT_JSON} and {OUT_JS}")
    print(f"Appended {len(log_lines)} rows to {OUT_LOG}")


if __name__ == "__main__":
    main()

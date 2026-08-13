"""
generate_player_predictions.py — Production entrypoint for Track A (anytime
TD player probabilities). Same pattern as generate_predictions.py (Track B):
trains on all played history, predicts the upcoming week, writes a current
snapshot (overwritten) + appends to a persistent log (never overwritten).

NO LIVE ODDS: this only outputs model probabilities (MODEL / xTD / usage
shares) -- no BEST PRICE / IMPL / EDGE / EV columns. Those need The Odds
API's player-prop plan (paid tier), which isn't connected. See README.

ROSTER FRESHNESS CAVEAT: see player_td_features.py's module docstring --
the candidate pool is last season's active players, corrected against a
frequently-updated ID crosswalk for offseason trades/signings, but true
rookies with zero prior NFL usage and any retirements the crosswalk hasn't
caught yet won't be reflected. Gets corrected automatically once real
current-season game data exists.

Usage:
    python generate_player_predictions.py
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from player_td_features import build_player_td_table
from player_td_model import FEATURE_COLS, POSITIONS, MIN_GAMES_PRIOR, _prep
from nfl_data import load_pbp, load_snaps, load_id_crosswalk, load_schedules
from features import build_player_week_features

ALL_SEASONS = list(range(2021, 2028))

OUT_JSON = "../data/player_td.json"
OUT_JS = "../data/player_td.js"
OUT_LOG = "../data/player_td_log.jsonl"

# Same probability-bucket tiers used for spread/total's edge tiers, but
# since there's no market to compare against, tiers here are MODEL
# CONFIDENCE tiers (how likely, not how much value) -- labeled distinctly
# in the output so nothing downstream confuses the two.
TIER_BOUNDS = [(0.40, "A"), (0.25, "B"), (0.15, "C"), (0.0, "D")]


def _tier(p: float) -> str:
    for bound, label in TIER_BOUNDS:
        if p >= bound:
            return label
    return "D"


def find_upcoming_week(sched: pd.DataFrame) -> tuple[int, int] | None:
    unplayed = sched[sched["home_score"].isna()]
    if unplayed.empty:
        return None
    row = unplayed[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).iloc[0]
    return int(row["season"]), int(row["week"])


def main():
    sched = load_schedules(ALL_SEASONS)
    target = find_upcoming_week(sched)
    if target is None:
        print("No upcoming week found. Nothing to predict.")
        return
    season, week = target
    print(f"Predicting: season={season} week={week}")

    print("Building player-week table (all available seasons)...")
    full = build_player_td_table(ALL_SEASONS, upcoming_season=season, upcoming_week=week)

    # ---- Training set: real played rows only, with the real target merged in ----
    played_seasons = [
        s for s in ALL_SEASONS
        if sched.loc[sched["season"] == s, "home_score"].notna().any()
    ]
    pbp = load_pbp(played_seasons)
    snaps = load_snaps(played_seasons)
    xwalk = load_id_crosswalk()
    raw = build_player_week_features(pbp, snaps, id_crosswalk=xwalk)
    target_col = raw[["season", "week", "player_id", "scored_td"]]

    train = full.merge(target_col, on=["season", "week", "player_id"], how="inner")
    train = _prep(train)
    print(f"Training rows (all played history, cold-start-filtered): {len(train)}")

    X_train, y_train = train[FEATURE_COLS], train["scored_td"]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_s, y_train)

    # ---- Predict the upcoming week ----
    upcoming = full[(full["season"] == season) & (full["week"] == week)].copy()
    for p in POSITIONS:
        upcoming[f"pos_{p}"] = (upcoming["position"] == p).astype(float)
    upcoming["implied_team_total"] = upcoming["implied_team_total"].fillna(
        upcoming["implied_team_total"].median()
    )
    upcoming["is_home"] = upcoming["is_home"].fillna(0.5)

    missing = upcoming[FEATURE_COLS].isna().any(axis=1)
    if missing.any():
        print(f"Dropping {missing.sum()} candidates with incomplete trailing features "
              f"(true rookies / insufficient prior-season usage):")
        print(upcoming.loc[missing, ["player_name", "position", "posteam"]].to_string(index=False))
    upcoming = upcoming[~missing].reset_index(drop=True)

    X_up = scaler.transform(upcoming[FEATURE_COLS])
    upcoming["model_prob"] = model.predict_proba(X_up)[:, 1]
    upcoming["tier"] = upcoming["model_prob"].apply(_tier)

    # ---- Build output ----
    players = []
    for _, r in upcoming.sort_values("model_prob", ascending=False).iterrows():
        players.append({
            "player_id": r["player_id"],
            "player_name": r["player_name"],
            "position": r["position"],
            "team": r["posteam"],
            "opponent": r["opponent"],
            "matchup": f"{r['posteam']} vs {r['opponent']}" if r["is_home"] == 1 else f"{r['posteam']} @ {r['opponent']}",
            "usage": {
                "snap_share": round(float(r["asof_roll4_snap_share"]), 3) if pd.notna(r["asof_roll4_snap_share"]) else None,
                "rz_target_share": round(float(r["asof_roll4_rz_target_share"]), 3) if pd.notna(r["asof_roll4_rz_target_share"]) else None,
                "inside5_carry_share": round(float(r["asof_roll4_inside5_carry_share"]), 3) if pd.notna(r["asof_roll4_inside5_carry_share"]) else None,
                "xtd_per_game": round(float(r["asof_roll4_expected_tds"]), 3) if pd.notna(r["asof_roll4_expected_tds"]) else None,
            },
            "model": {
                "anytime_td_prob": round(float(r["model_prob"]), 4),
                "tier": r["tier"],
            },
            "market": None,  # no live odds source connected -- see README
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "week": week,
        "model_notes": {
            "training_rows": len(train),
            "caveat": (
                "No live sportsbook odds connected -- model probability and usage shares "
                "only, no price/edge/EV. Candidate pool is last season's active players "
                "corrected for known offseason trades; true rookies and any very recent "
                "retirements the crosswalk hasn't caught are not included yet."
            ),
        },
        "players": players,
    }

    out_dir = os.path.dirname(os.path.abspath(OUT_JSON))
    os.makedirs(out_dir, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    with open(OUT_JS, "w") as f:
        f.write("// Auto-generated by scripts/generate_player_predictions.py -- do not edit by hand.\n")
        f.write(f"const PLAYER_TD_DATA = {json.dumps(output, indent=2)};\n")

    # ---- Append-only history, same pattern as Track B's predictions_log.jsonl ----
    logged_at = output["generated_at"]
    with open(OUT_LOG, "a") as f:
        for p in players:
            f.write(json.dumps({
                "logged_at": logged_at, "season": season, "week": week,
                "player_id": p["player_id"], "player_name": p["player_name"],
                "position": p["position"], "team": p["team"], "opponent": p["opponent"],
                "anytime_td_prob": p["model"]["anytime_td_prob"], "tier": p["model"]["tier"],
            }) + "\n")

    print(f"Wrote {len(players)} players to {OUT_JSON} and {OUT_JS}")
    print(f"Appended {len(players)} rows to {OUT_LOG}")


if __name__ == "__main__":
    main()

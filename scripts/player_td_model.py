"""
player_td_model.py — Layer 3/4 combined: calibrated anytime-TD probability
per player, per game.

Combines Layer 3 (allocate team TDs across players by opportunity share) and
Layer 4 (calibrated player probability) into one classifier rather than two
separate stages, given the time budget ahead of Week 1 -- a single
well-regularized logistic regression on the right leakage-safe features
gets most of the value of the strict two-stage blueprint design, and is far
easier to validate and keep calibrated than a hand-built allocation formula.

Target: scored_td (did this player score >=1 offensive TD this game).

Features: player's leakage-safe trailing opportunity shares (carry/target/
inside5/red-zone share, snap share) at 4-game/8-game/season-to-date windows,
trailing actual/expected TD rate, position, implied_team_total (the Vegas
signal -- legitimately available pre-kickoff, not leakage), home/away.

TRAIN/VALIDATION SPLIT IS TIME-BASED (2021-2023 train, 2024 validate), same
principle as team_td_model.py and game_lines_model.py -- see those files'
docstrings for why a random split would be invalid here.

Usage:
    python player_td_model.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from player_td_features import build_player_td_table

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS = [2024]
MIN_GAMES_PRIOR = 3  # cold-start guard, same reasoning as the other two models

NUMERIC_FEATURE_COLS = [
    "asof_roll4_carry_share", "asof_roll8_carry_share", "asof_cum_carry_share",
    "asof_roll4_target_share", "asof_roll8_target_share", "asof_cum_target_share",
    "asof_roll4_inside5_carry_share", "asof_roll8_inside5_carry_share", "asof_cum_inside5_carry_share",
    "asof_roll4_rz_target_share", "asof_roll8_rz_target_share", "asof_cum_rz_target_share",
    "asof_roll4_snap_share", "asof_roll8_snap_share",
    "asof_roll4_expected_tds", "asof_roll8_expected_tds",
    "asof_roll4_actual_tds", "asof_roll8_actual_tds",
    "asof_player_games_prior",
    "implied_team_total", "is_home",
]
POSITIONS = ["QB", "RB", "WR", "TE", "FB"]
FEATURE_COLS = NUMERIC_FEATURE_COLS + [f"pos_{p}" for p in POSITIONS]


def _prep(df: pd.DataFrame, scored_td_col: pd.Series | None = None) -> pd.DataFrame:
    df = df.copy()
    for p in POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(float)
    df["implied_team_total"] = df["implied_team_total"].fillna(df["implied_team_total"].median())
    df["is_home"] = df["is_home"].fillna(0.5)
    df = df.dropna(subset=["asof_player_games_prior"])
    df = df[df["asof_player_games_prior"] >= MIN_GAMES_PRIOR]
    df = df.dropna(subset=NUMERIC_FEATURE_COLS)
    return df.reset_index(drop=True)


def calibration_table(y_true, y_pred, bins=6):
    df = pd.DataFrame({"y": y_true, "p": y_pred})
    df["bucket"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bucket", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), mean_actual=("y", "mean")
    )


def main():
    print("Building player-week table (2021-2024, includes real scored_td target)...")
    # For training/validation only real historical rows matter -- pass a
    # dummy upcoming_season far in the future so no stub rows get mixed in.
    full = build_player_td_table(TRAIN_SEASONS + VAL_SEASONS, upcoming_season=2099, upcoming_week=1)

    # Bring the real target back in -- build_player_td_table's asof_* frame
    # doesn't carry it directly (it's an as-of INPUT table); pull actual_tds
    # / scored_td from the un-lagged base table for played rows only.
    from features import build_player_week_features
    from nfl_data import load_pbp, load_snaps, load_id_crosswalk
    pbp = load_pbp(TRAIN_SEASONS + VAL_SEASONS)
    snaps = load_snaps(TRAIN_SEASONS + VAL_SEASONS)
    xwalk = load_id_crosswalk()
    raw = build_player_week_features(pbp, snaps, id_crosswalk=xwalk)
    target = raw[["season", "week", "player_id", "scored_td"]]

    full = full.merge(target, on=["season", "week", "player_id"], how="inner")  # inner: only real played rows
    full = _prep(full)
    print(f"Rows after cold-start filter: {len(full)}  positive rate: {full['scored_td'].mean():.3f}")

    train = full[full["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    val = full[full["season"].isin(VAL_SEASONS)].reset_index(drop=True)
    print(f"Train rows: {len(train)}   Val rows: {len(val)}")

    X_train, y_train = train[FEATURE_COLS], train["scored_td"]
    X_val, y_val = val[FEATURE_COLS], val["scored_td"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print("\n--- Validation performance (held out, 2024 season) ---")

    baseline_pred = np.full(len(val), y_train.mean())
    print(f"  Baseline (mean)        LogLoss={log_loss(y_val, baseline_pred):.4f}  "
          f"Brier={brier_score_loss(y_val, baseline_pred):.4f}")

    logit = LogisticRegression(C=1.0, max_iter=1000)
    logit.fit(X_train_s, y_train)
    logit_pred = logit.predict_proba(X_val_s)[:, 1]
    print(f"  Logistic Regression    LogLoss={log_loss(y_val, logit_pred):.4f}  "
          f"Brier={brier_score_loss(y_val, logit_pred):.4f}  AUC={roc_auc_score(y_val, logit_pred):.4f}")

    gbm = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=40, random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict_proba(X_val)[:, 1]
    print(f"  GBM                    LogLoss={log_loss(y_val, gbm_pred):.4f}  "
          f"Brier={brier_score_loss(y_val, gbm_pred):.4f}  AUC={roc_auc_score(y_val, gbm_pred):.4f}")

    coefs = pd.Series(logit.coef_[0], index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print("\n--- Logistic regression coefficients (standardized) ---")
    print(coefs.to_string())

    print("\n--- Calibration check: predicted vs actual TD rate by probability bucket (Logistic) ---")
    print(calibration_table(y_val, logit_pred).to_string())

    print("\n--- Calibration check: GBM ---")
    print(calibration_table(y_val, gbm_pred).to_string())

    return {
        "train": train, "val": val, "logit": logit, "gbm": gbm, "scaler": scaler,
    }


if __name__ == "__main__":
    main()

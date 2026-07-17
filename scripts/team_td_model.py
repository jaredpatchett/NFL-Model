"""
team_td_model.py — Layer 2: expected offensive TDs per team, per game.

Predicts team_actual_tds (offensive rushing + receiving TDs, i.e. excludes
defensive/special-teams TDs which no player-prop model needs) from:
  - Vegas-implied team total (the single best signal — the market already
    prices in QB injuries, weather, matchup, etc. that we haven't modeled yet)
  - implied opponent total (proxy for game script / negative game-flow risk)
  - is_home, rest_days, div_game
  - leakage-safe trailing team pace/opportunity features (asof_roll4_*,
    asof_cum_*) from rolling_features.py / team_features.py

TRAIN/VALIDATION SPLIT IS TIME-BASED, NOT RANDOM:
  Train: 2021-2023 seasons. Validate: 2024 season (fully held out, never seen
  in training). A random split would let the model see e.g. a team's week 10
  from one season while training on that same team's week 3 from the same
  season, which leaks season-specific team quality into "unseen" data. Time-
  based splitting is the only honest way to estimate real forecasting skill,
  per the blueprint's backtesting principle.

COLD START CAVEAT: rows from a team's first ~3 games of a season have thin/
absent trailing features (asof_games_played_prior < 3). This baseline drops
those rows from train/eval. In production you'll want to bridge that gap with
prior-season trailing stats or a league-average prior — not solved here yet,
flagged as a known next step.

Models compared:
  - Baseline: always predict the training-set mean team TDs (sanity floor)
  - Poisson GLM: interpretable, appropriate for count data
  - Gradient-boosted trees (Poisson loss): captures nonlinearity/interactions

Usage:
    python team_td_model.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from team_features import build_team_week_model_table

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS = [2024]
MIN_GAMES_PRIOR = 3  # drop cold-start rows

FEATURE_COLS = [
    "implied_team_total", "implied_opp_total", "is_home", "rest_days",
    "opp_rest_days", "div_game",
    "asof_roll4_team_rush_opp", "asof_roll4_team_pass_opp",
    "asof_roll4_team_rush_inside5", "asof_roll4_team_rush_rz",
    "asof_roll4_team_pass_rz", "asof_roll4_team_pass_inside10",
    "asof_roll4_team_actual_tds", "asof_cum_team_actual_tds",
    "asof_games_played_prior",
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["div_game"] = df["div_game"].fillna(0).astype(float)
    df["rest_days"] = df["rest_days"].fillna(df["rest_days"].median())
    df["opp_rest_days"] = df["opp_rest_days"].fillna(df["opp_rest_days"].median())
    return df[df["asof_games_played_prior"] >= MIN_GAMES_PRIOR].reset_index(drop=True)


def poisson_deviance(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, None)
    y_true = np.asarray(y_true, dtype=float)
    y_true_safe = np.where(y_true > 0, y_true, 1.0)  # avoid log(0); masked out below anyway
    term = np.where(y_true > 0, y_true * np.log(y_true_safe / y_pred), 0.0)
    return float(2 * np.mean(term - (y_true - y_pred)))


def evaluate(name, y_true, y_pred):
    mae = float(np.mean(np.abs(np.asarray(y_true) - y_pred)))
    dev = poisson_deviance(y_true, y_pred)
    bias = float(np.mean(y_pred) - np.mean(y_true))
    print(f"  {name:22s} MAE={mae:.4f}  PoissonDeviance={dev:.4f}  "
          f"MeanPred={np.mean(y_pred):.3f}  MeanActual={np.mean(y_true):.3f}  Bias={bias:+.4f}")
    return {"name": name, "mae": mae, "deviance": dev, "bias": bias}


def main():
    print("Building team-week model table (2021-2024)...")
    full = build_team_week_model_table(TRAIN_SEASONS + VAL_SEASONS)
    full = _prep(full)

    train = full[full["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    val = full[full["season"].isin(VAL_SEASONS)].reset_index(drop=True)
    print(f"Train rows: {len(train)} ({TRAIN_SEASONS})   Val rows: {len(val)} ({VAL_SEASONS})")

    X_train, y_train = train[FEATURE_COLS], train["team_actual_tds"]
    X_val, y_val = val[FEATURE_COLS], val["team_actual_tds"]

    print("\n--- Validation performance (held out, 2024 season, never trained on) ---")

    # Baseline: training-set mean
    baseline_pred = np.full(len(val), y_train.mean())
    results = [evaluate("Baseline (mean)", y_val, baseline_pred)]

    # Poisson GLM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    glm = PoissonRegressor(alpha=1.0, max_iter=500)
    glm.fit(X_train_s, y_train)
    glm_pred = glm.predict(X_val_s)
    results.append(evaluate("Poisson GLM", y_val, glm_pred))

    # Gradient boosted trees, Poisson loss
    gbm = HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=30, random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_val)
    results.append(evaluate("GBM (Poisson loss)", y_val, gbm_pred))

    print("\n--- Feature importance (GBM, permutation-free / split-based proxy via GLM coefs) ---")
    coefs = pd.Series(glm.coef_, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print(coefs.to_string())

    print("\n--- Calibration check: predicted vs actual TDs by implied-total decile (GBM) ---")
    val = val.copy()
    val["pred_gbm"] = gbm_pred
    val["total_bucket"] = pd.qcut(val["implied_team_total"], 5, duplicates="drop")
    calib = val.groupby("total_bucket", observed=True).agg(
        n=("team_actual_tds", "size"),
        mean_actual=("team_actual_tds", "mean"),
        mean_pred=("pred_gbm", "mean"),
    )
    print(calib.to_string())

    return {"train": train, "val": val, "glm": glm, "gbm": gbm, "scaler": scaler, "results": results}


if __name__ == "__main__":
    main()

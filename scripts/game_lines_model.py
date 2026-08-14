"""
game_lines_model.py — Track B: point margin, total points, and moneyline
win probability, validated against real closing lines.

Two regression targets, modeled differently:
  - team_margin (team_score - opp_score) -> feeds spread edge + moneyline.
    Predicted from scratch (no market input) -- power rating differential,
    trailing scoring pace, home field, rest, division-game flag.
  - game_total  (team_score + opp_score) -> feeds total edge. Predicted as
    market_total_line + a modeled RESIDUAL (see train_total_residual()'s
    docstring for why -- from-scratch total prediction barely beat a naive
    mean-guess baseline; anchoring to the market and predicting the
    deviation is a legitimate, common quant pattern and performs much
    better). total_line is genuinely available pre-kickoff -- using it as
    an anchor is not leakage the way a trailing stat would be.

Margin still avoids market inputs entirely, by design -- the point is to
find value against the market, not reproduce it. Total's market-blended
approach is a deliberate, documented exception to that rule, made because
the from-scratch version wasn't working (see git history / README for the
before/after numbers).

TRAIN/VALIDATION SPLIT IS TIME-BASED (2021-2023 train, 2024 validate),
same principle as team_td_model.py -- see that file's docstring for why.

CLOSING-LINE CAVEAT: the schedule data's spread_line/total_line/moneylines
are closing lines (nflverse/nfldata), not the line available at prediction
time. That's fine for validating whether the MODEL is calibrated (is our
margin/total actually close to what happened), but it is NOT a valid
backtest of betting profitability -- a real edge backtest needs opening or
time-stamped lines (e.g. from The Odds API), never closing/hindsight prices,
per the blueprint's core backtesting principle. Treat the "edge vs market"
numbers below as a sanity check on model calibration, not as expected
betting ROI.

Usage:
    python game_lines_model.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from game_features import build_game_model_table

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS = [2024]
MIN_GAMES_PRIOR = 3  # cold-start guard, same reasoning as team_td_model.py

FEATURE_COLS = [
    "power_rating", "opp_power_rating", "is_home", "hfa_as_of_week",
    "rest_days", "opp_rest_days", "div_game",
    "asof_roll4_team_score", "asof_roll4_opp_score",
    "asof_roll8_team_score", "asof_roll8_opp_score",
    "is_indoor", "temp_filled", "wind_filled",
    "asof_roll4_team_plays", "asof_roll8_team_plays",
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["div_game"] = df["div_game"].fillna(0).astype(float)
    df["rest_days"] = df["rest_days"].fillna(df["rest_days"].median())
    df["opp_rest_days"] = df["opp_rest_days"].fillna(df["opp_rest_days"].median())
    df = df.dropna(subset=["power_rating", "opp_power_rating"])
    # games_used from power_ratings isn't carried into game_features directly;
    # use presence of trailing scoring/pace as the cold-start guard instead.
    df = df.dropna(subset=["asof_roll4_team_score", "asof_roll4_team_plays"])
    return df.reset_index(drop=True)


def moneyline_to_implied_prob(ml: float) -> float:
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)


def prob_to_fair_moneyline(p: float) -> float:
    if p >= 0.5:
        return -100 * p / (1 - p)
    return 100 * (1 - p) / p


def evaluate(name, y_true, y_pred):
    mae = float(np.mean(np.abs(np.asarray(y_true) - y_pred)))
    bias = float(np.mean(y_pred) - np.mean(y_true))
    rmse = float(np.sqrt(np.mean((np.asarray(y_true) - y_pred) ** 2)))
    print(f"  {name:22s} MAE={mae:.3f}  RMSE={rmse:.3f}  Bias={bias:+.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "bias": bias}


def train_total_residual(train, val):
    """
    TOTAL POINTS, market-blended: instead of predicting game_total from
    scratch, anchor to the market's own total_line (legitimately available
    pre-kickoff -- not leakage the way trailing stats would be) and predict
    the RESIDUAL (actual_total - total_line) from our features. Final
    prediction = total_line + predicted_residual.

    HONEST FINDING FROM THIS SESSION'S ALPHA SWEEP: the market's own
    total_line, used alone with ZERO model input, beats every regularization
    strength tried (MAE 9.768 vs the best model-adjusted MAE of ~9.79-9.88
    across alpha 5 through 100,000). This isn't a tuning problem -- it
    converges toward market-alone as regularization increases and never
    crosses it. Conclusion: this feature set does not contain genuine
    signal beyond what's already priced into the market total for 2024.

    RESPONSE: use MUCH heavier regularization (alpha=20000 vs margin's
    alpha=5) than a naive tune would suggest, specifically because "the
    best-scoring alpha on one validation season" would overfit to that
    season's noise -- the sweep showed EVERY alpha underperforms market-
    alone, so the responsible choice is the one that stays closest to
    market-alone (least harm) while preserving a small, football-sane
    directional nudge (wind, pace, rest) rather than chasing a local MAE
    minimum that isn't real signal. total_edge should be treated as
    low-confidence -- closer to "probably noise" than margin's edge, which
    DID show genuine skill over baseline. See README for the full sweep
    table and reasoning.
    """
    TOTAL_ALPHA = 20000

    X_train = train[FEATURE_COLS]
    y_train_actual = train["game_total"]
    y_train_resid = train["game_total"] - train["total_line"]
    X_val = val[FEATURE_COLS]
    y_val_actual = val["game_total"]

    print("\n--- TOTAL POINTS: validation performance (2024, held out) ---")

    baseline_mean_pred = np.full(len(val), y_train_actual.mean())
    evaluate("Baseline (mean)", y_val_actual, baseline_mean_pred)

    market_only_pred = val["total_line"].values
    evaluate("Market total alone", y_val_actual, market_only_pred)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    ridge = Ridge(alpha=TOTAL_ALPHA)
    ridge.fit(X_train_s, y_train_resid)
    ridge_resid_pred = ridge.predict(X_val_s)
    ridge_final_pred = val["total_line"].values + ridge_resid_pred
    evaluate(f"Ridge (alpha={TOTAL_ALPHA}, heavily shrunk)", y_val_actual, ridge_final_pred)

    # GBM kept for reference/comparison only -- NOT used in production
    # (generate_predictions.py uses the Ridge residual model), since a
    # heavily-regularized linear model is the more defensible choice here:
    # more interpretable, and easier to confirm it isn't quietly overfitting
    # given we've already established this feature set lacks real signal.
    gbm = HistGradientBoostingRegressor(
        max_iter=150, max_depth=3, learning_rate=0.03, min_samples_leaf=50, random_state=42,
    )
    gbm.fit(X_train, y_train_resid)
    gbm_resid_pred = gbm.predict(X_val)
    gbm_final_pred = val["total_line"].values + gbm_resid_pred
    evaluate("GBM (reference only, not used in production)", y_val_actual, gbm_final_pred)

    coefs = pd.Series(ridge.coef_, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print(f"  Ridge coefficients (predicting DEVIATION from market total, alpha={TOTAL_ALPHA}):")
    print(coefs.to_string())

    resid_std = float(np.std(y_train_resid - ridge.predict(X_train_s)))
    return ridge, scaler, gbm, ridge_final_pred, gbm_final_pred, resid_std


def train_target(train, val, target_col, label):
    X_train, y_train = train[FEATURE_COLS], train[target_col]
    X_val, y_val = val[FEATURE_COLS], val[target_col]

    print(f"\n--- {label}: validation performance (2024, held out) ---")
    baseline_pred = np.full(len(val), y_train.mean())
    evaluate("Baseline (mean)", y_val, baseline_pred)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train_s, y_train)
    ridge_pred = ridge.predict(X_val_s)
    evaluate("Ridge", y_val, ridge_pred)

    gbm = HistGradientBoostingRegressor(
        max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=40, random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_val)
    evaluate("GBM", y_val, gbm_pred)

    coefs = pd.Series(ridge.coef_, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print(f"  Ridge coefficients ({label}):")
    print(coefs.to_string())

    residual_std = float(np.std(y_train - ridge.predict(X_train_s)))
    return ridge, scaler, gbm, ridge_pred, gbm_pred, residual_std


def main():
    print("Building game model table (2021-2024)...")
    full = build_game_model_table(TRAIN_SEASONS + VAL_SEASONS)
    full = _prep(full)
    train = full[full["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    val = full[full["season"].isin(VAL_SEASONS)].reset_index(drop=True)
    print(f"Train rows: {len(train)}   Val rows: {len(val)}")

    margin_ridge, margin_scaler, margin_gbm, margin_pred, margin_gbm_pred, margin_resid_std = \
        train_target(train, val, "team_margin", "MARGIN (spread)")

    total_ridge, total_scaler, total_gbm, total_pred, total_gbm_pred, total_resid_std = \
        train_total_residual(train, val)

    # ---- Moneyline: win prob from projected margin, assuming ~normal residuals ----
    val = val.copy()
    val["pred_margin"] = margin_pred
    val["pred_total"] = total_pred
    val["model_win_prob"] = norm.cdf(val["pred_margin"] / margin_resid_std)
    val["model_fair_moneyline"] = val["model_win_prob"].apply(prob_to_fair_moneyline)
    val["market_win_prob"] = val["team_moneyline"].apply(moneyline_to_implied_prob)

    print(f"\n--- Sanity: margin residual std = {margin_resid_std:.2f} pts "
          f"(this sets how confident win-prob conversions are) ---")

    print("\n--- Spread edge vs market (calibration sanity check, NOT a profitability backtest"
          " -- these are closing lines, see module docstring) ---")
    val["spread_edge"] = val["pred_margin"] - val["team_spread_line"]
    print(val[["team", "opponent", "week", "pred_margin", "team_spread_line", "spread_edge"]]
          .sort_values("spread_edge", ascending=False).head(5).to_string(index=False))
    print("...")
    print(val[["team", "opponent", "week", "pred_margin", "team_spread_line", "spread_edge"]]
          .sort_values("spread_edge").head(5).to_string(index=False))

    print("\n--- Total edge vs market ---")
    val["total_edge"] = val["pred_total"] - val["total_line"]
    print(f"  Mean |total_edge|: {val['total_edge'].abs().mean():.2f}  "
          f"Mean total_edge (bias check): {val['total_edge'].mean():+.2f}")

    print("\n--- Moneyline: model win prob vs market implied prob (sample) ---")
    sample = val[["team", "opponent", "week", "model_win_prob", "market_win_prob", "team_moneyline"]].head(8)
    print(sample.to_string(index=False))

    return {
        "train": train, "val": val,
        "margin_ridge": margin_ridge, "margin_gbm": margin_gbm, "margin_resid_std": margin_resid_std,
        "total_ridge": total_ridge, "total_gbm": total_gbm, "total_resid_std": total_resid_std,
    }


if __name__ == "__main__":
    main()

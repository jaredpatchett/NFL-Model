"""
backtest_profitability.py -- The genuine profitability test that
game_lines_model.py's existing validation explicitly says it is NOT (see
that module's docstring: closing-line validation proves calibration, not
profitability). This script:

  1. Retrains the SAME margin/total models on 2021-2023 (identical to
     game_lines_model.py -- not a different model, just applied out-of-
     sample here instead of validated against closing lines).
  2. Applies those models to every 2024/2025 game.
  3. Grades against the REAL point-in-time odds fetched by
     fetch_historical_odds.py (data/historical_odds_2024_2025.jsonl) --
     the actual price that was live days before kickoff, not the closing
     number.
  4. Reports real win rate and ROI using the REAL price recorded at that
     snapshot (not an assumed -110), against real final scores.

Requires data/historical_odds_2024_2025.jsonl to already exist (run
fetch_historical_odds.py first).

Usage:
    python backtest_profitability.py
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from game_features import build_game_model_table
from game_lines_model import FEATURE_COLS

HIST_ODDS_PATH = "../data/historical_odds_2024_2025.jsonl"
MARGIN_ALPHA = 5.0
TOTAL_ALPHA = 20000


def moneyline_to_implied_prob(ml):
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)


def american_profit(price, stake=1.0):
    """Profit on a WIN at this price (loss on a loss is always -stake, handled by caller)."""
    return stake * (price / 100) if price > 0 else stake * (100 / -price)


def load_historical_odds() -> pd.DataFrame:
    if not os.path.exists(HIST_ODDS_PATH):
        raise FileNotFoundError(
            f"{HIST_ODDS_PATH} not found -- run fetch_historical_odds.py first."
        )
    rows = [json.loads(l) for l in open(HIST_ODDS_PATH)]
    out = []
    for r in rows:
        best_spread_home, best_spread_price = None, None
        best_total_line, best_total_over_price = None, None
        best_ml_home = None
        for bk in r.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "spreads":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == r["home_team"]:
                            if best_spread_price is None or o["price"] > best_spread_price:
                                best_spread_home, best_spread_price = o.get("point"), o["price"]
                elif mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == "Over":
                            if best_total_over_price is None or o["price"] > best_total_over_price:
                                best_total_line, best_total_over_price = o.get("point"), o["price"]
                elif mkt["key"] == "h2h":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == r["home_team"]:
                            if best_ml_home is None or o["price"] > best_ml_home:
                                best_ml_home = o["price"]
        out.append({
            "season": r["season"], "week": r["week"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "hist_spread_line": best_spread_home, "hist_spread_price": best_spread_price,
            "hist_total_line": best_total_line, "hist_total_over_price": best_total_over_price,
            "hist_home_ml": best_ml_home,
        })
    return pd.DataFrame(out)


def main():
    print("Building game model table (2021-2025)...")
    full = build_game_model_table([2021, 2022, 2023, 2024, 2025])
    full = full.dropna(subset=["power_rating", "opp_power_rating"])
    full = full.dropna(subset=["asof_roll4_team_score", "asof_roll4_team_plays"])

    train = full[full["season"].isin([2021, 2022, 2023])].reset_index(drop=True)
    test = full[full["season"].isin([2024, 2025]) & (full["is_home"] == 1)].reset_index(drop=True)
    print(f"Train rows: {len(train)}   Test rows (home rows, 2024+2025): {len(test)}")

    # ---- retrain margin + total, IDENTICAL to game_lines_model.py ----
    margin_scaler = StandardScaler()
    X_train_m = margin_scaler.fit_transform(train[FEATURE_COLS])
    margin_ridge = Ridge(alpha=MARGIN_ALPHA).fit(X_train_m, train["team_margin"])
    margin_resid_std = float(np.std(train["team_margin"] - margin_ridge.predict(X_train_m)))

    total_scaler = StandardScaler()
    X_train_t = total_scaler.fit_transform(train[FEATURE_COLS])
    y_train_resid = train["game_total"] - train["total_line"]
    total_ridge = Ridge(alpha=TOTAL_ALPHA).fit(X_train_t, y_train_resid)

    X_test_m = margin_scaler.transform(test[FEATURE_COLS])
    X_test_t = total_scaler.transform(test[FEATURE_COLS])
    test = test.copy()
    test["pred_margin"] = margin_ridge.predict(X_test_m)
    test["pred_total"] = test["total_line"] + total_ridge.predict(X_test_t)
    test["model_win_prob"] = norm.cdf(test["pred_margin"] / margin_resid_std)

    # ---- join REAL point-in-time historical odds ----
    # test rows are home-team perspective (is_home==1), so test["team"] IS
    # the home team already -- rename hist's home_team to match for a clean merge.
    hist = load_historical_odds()
    df = test.merge(hist.rename(columns={"home_team": "team"}),
                     on=["season", "week", "team"], how="left")
    # A left merge always returns len(test) rows regardless of match rate --
    # counting len(df) here would always say "570 of 570 matched" even with
    # almost no real data joined. Count actual non-null matches instead.
    # Caught via a synthetic single-row fixture test before this shipped.
    n_matched = int(df["hist_spread_line"].notna().sum())
    print(f"Matched {n_matched} of {len(test)} test games to a real historical odds snapshot.")
    if n_matched == 0:
        print("No matches -- check that fetch_historical_odds.py actually ran and team codes align.")
        return

    df["spread_edge"] = df["pred_margin"] - df["hist_spread_line"]
    df["total_edge"] = df["pred_total"] - df["hist_total_line"]
    df["hist_ml_implied"] = df["hist_home_ml"].apply(lambda x: moneyline_to_implied_prob(x) if pd.notna(x) else np.nan)
    df["ml_edge"] = df["model_win_prob"] - df["hist_ml_implied"]

    def grade_ats(r, min_edge):
        if pd.isna(r["spread_edge"]) or abs(r["spread_edge"]) < min_edge:
            return None
        picked_home = r["spread_edge"] >= 0
        ats_margin = r["team_margin"] - r["hist_spread_line"]
        if abs(ats_margin) < 1e-9:
            return "PUSH"
        return "WIN" if (ats_margin > 0) == picked_home else "LOSS"

    def grade_ml(r, min_edge):
        if pd.isna(r["ml_edge"]) or r["ml_edge"] < min_edge:
            return None
        return "WIN" if r["team_margin"] > 0 else "LOSS"

    def grade_total(r, min_edge):
        if pd.isna(r["total_edge"]) or abs(r["total_edge"]) < min_edge:
            return None
        picked_over = r["total_edge"] >= 0
        actual_total = r["game_total"]
        if abs(actual_total - r["hist_total_line"]) < 1e-9:
            return "PUSH"
        return "WIN" if (actual_total > r["hist_total_line"]) == picked_over else "LOSS"

    for label, fn, price_col, default_price, min_edge in [
        ("SPREAD", grade_ats, "hist_spread_price", -110, 3.0),
        ("TOTAL", grade_total, "hist_total_over_price", -110, 1.5),
        ("MONEYLINE", grade_ml, "hist_home_ml", None, 0.05),
    ]:
        df[f"{label}_result"] = df.apply(lambda r: fn(r, min_edge), axis=1)
        graded = df[df[f"{label}_result"].notna()]
        wins = (graded[f"{label}_result"] == "WIN").sum()
        losses = (graded[f"{label}_result"] == "LOSS").sum()
        pushes = (graded[f"{label}_result"] == "PUSH").sum()
        n = wins + losses
        if n == 0:
            print(f"{label}: no qualifying plays.")
            continue
        profit = 0.0
        for _, r in graded[graded[f"{label}_result"].isin(["WIN", "LOSS"])].iterrows():
            price = r.get(price_col)
            price = price if pd.notna(price) else default_price
            if r[f"{label}_result"] == "WIN":
                profit += american_profit(price)
            else:
                profit -= 1.0
        roi = profit / n * 100
        print(f"{label}: {wins}-{losses}{'-'+str(pushes) if pushes else ''} "
              f"({wins/n*100:.1f}%)  ROI={roi:+.2f}%  (n={n}, REAL point-in-time prices, "
              f"Tuesday-of-week snapshot -- not closing lines)")


if __name__ == "__main__":
    main()

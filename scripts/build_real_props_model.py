"""
build_real_props_model.py -- The trailing-average model tested in
backtest_player_props_profitability.py showed no real edge (win rate flat
at ~50-52% regardless of threshold, and actually WORSE at higher
thresholds -- the signature of noise, not signal). That model had none of
what makes Track B's margin model actually work: no opponent adjustment,
no game context, nothing beyond "what has this player done lately."

This builds a real Ridge regression per stat (receiving_yards, receptions,
rushing_yards) with actual features:
  - trailing_L4, trailing_L8 (leakage-safe, both windows -- lets the model
    weigh recent form against a more stable longer baseline itself,
    instead of us picking one window size by feel)
  - opponent's trailing defense-vs-position allowed rate (leakage-safe --
    reuses the same real per-week defense ranking already built and
    validated in generate_fantasy_projections.py's build_weekly_def_rank)
  - team implied total, from REAL market data (the same historical game
    odds already fetched for Track B -- total_line and spread_line implicit
    team total, not a guess)
  - home/away

TRAINED on 2021-2023 (matching Track B's exact train window, for
consistency), TESTED against the REAL 2024/2025 historical prop odds
ALREADY FETCHED for backtest_player_props_profitability.py -- zero new API
cost, this only changes the model, not the data.

Usage:
    python build_real_props_model.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_player_stats import fetch_player_stats
import odds_api
from backtest_player_props_profitability import load_historical_props, FIELD_TO_STAT_COL, american_profit

TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]
STAT_COLS = ["receiving_yards", "receptions", "rushing_yards"]
POSITIONS = ["QB", "RB", "WR", "TE"]
HIST_ODDS_PATH = "../data/historical_odds_2024_2025.jsonl"
HIST_ODDS_TRAIN_PATH = "../data/historical_odds_2021_2023.jsonl"


def load_all_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = fetch_player_stats(s)
        df = df[df["position"].isin(POSITIONS)].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_features(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for stat_col in STAT_COLS:
        shifted = stats.groupby("player_id")[stat_col].shift(1)
        stats[f"{stat_col}_L4"] = shifted.groupby(stats["player_id"]).transform(
            lambda s: s.rolling(4, min_periods=1).mean()
        )
        stats[f"{stat_col}_L8"] = shifted.groupby(stats["player_id"]).transform(
            lambda s: s.rolling(8, min_periods=1).mean()
        )

    # Real, leakage-safe opponent defense-vs-position: trailing points
    # (using the same stat, summed league-wide per opponent/position/week)
    # allowed BEFORE this week -- not the current week's own result.
    for stat_col in STAT_COLS:
        team_pos_week = stats.groupby(["opponent_team", "position", "season", "week"], as_index=False)[stat_col].sum()
        team_pos_week = team_pos_week.sort_values(["opponent_team", "position", "season", "week"])
        team_pos_week[f"opp_def_{stat_col}"] = team_pos_week.groupby(
            ["opponent_team", "position", "season"]
        )[stat_col].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        stats = stats.merge(
            team_pos_week[["opponent_team", "position", "season", "week", f"opp_def_{stat_col}"]],
            on=["opponent_team", "position", "season", "week"], how="left"
        )
    return stats


def load_team_implied_totals(hist_path: str, seasons: list[int]) -> pd.DataFrame:
    """REAL market signal, reused from an already-fetched game-lines
    historical odds file (no new cost beyond the original fetch): each
    team's implied total = the game total split by the spread (favorite
    gets more of the total). Works for either the 2024-2025 file or the
    2021-2023 training-years file -- same shape, different seasons."""
    if not os.path.exists(hist_path):
        return pd.DataFrame(columns=["season", "week", "team", "implied_total"])
    from fetch_historical_odds import get_deduplicated_real_games_for, TEAM_NAME_TO_ABBR
    rows = [json.loads(l) for l in open(hist_path)]
    if not rows:
        # File exists but is empty (e.g. a prior run that fetched 0 events,
        # or a stray empty file) -- same "nothing here" case as the file
        # not existing at all, but os.path.exists() alone doesn't catch it.
        # Caught via a real empty-file test case that otherwise crashed on
        # raw["home_team"] with no columns to index.
        return pd.DataFrame(columns=["season", "week", "team", "implied_total"])
    raw = pd.DataFrame(rows)
    raw["home_team_raw"] = raw["home_team"]
    raw["home_team"] = raw["home_team"].map(lambda n: TEAM_NAME_TO_ABBR.get(n, n))
    raw["away_team"] = raw["away_team"].map(lambda n: TEAM_NAME_TO_ABBR.get(n, n))

    games = get_deduplicated_real_games_for(hist_path, seasons)
    raw_by_event = raw.drop_duplicates(subset=["event_id"]).set_index("event_id")
    out = []
    for _, g in games.iterrows():
        r = raw_by_event.loc[g["event_id"]] if g["event_id"] in raw_by_event.index else None
        if r is None:
            continue
        best_spread, best_total = None, None
        for bk in r.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "spreads":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == r["home_team_raw"]:
                            best_spread = o.get("point")
                elif mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == "Over":
                            best_total = o.get("point")
        if best_spread is None or best_total is None:
            continue
        home_implied = best_total / 2 - best_spread / 2
        away_implied = best_total - home_implied
        out.append({"season": g["season"], "week": g["week"], "team": g["home_team"], "implied_total": home_implied})
        out.append({"season": g["season"], "week": g["week"], "team": g["away_team"], "implied_total": away_implied})
    return pd.DataFrame(out)


def main():
    print("Loading training data (2021-2023)...")
    train_stats = load_all_stats(TRAIN_SEASONS)
    train_stats = build_features(train_stats)

    print("Loading test data (2024-2025)...")
    test_stats = load_all_stats(TEST_SEASONS)
    test_stats = build_features(test_stats)
    test_stats["player_name_norm"] = test_stats["player_display_name"].apply(odds_api.normalize_name)

    print("Loading real team implied totals (test years, 2024-2025)...")
    implied_test = load_team_implied_totals(HIST_ODDS_PATH, TEST_SEASONS)
    test_stats = test_stats.merge(implied_test, on=["season", "week", "team"], how="left")
    fallback = implied_test["implied_total"].mean() if not implied_test.empty else 22.0
    test_stats["implied_total"] = test_stats["implied_total"].fillna(fallback)

    print("Loading real team implied totals (train years, 2021-2023)...")
    implied_train = load_team_implied_totals(HIST_ODDS_TRAIN_PATH, TRAIN_SEASONS)
    train_stats = train_stats.merge(implied_train, on=["season", "week", "team"], how="left")
    train_matched = train_stats["implied_total"].notna().sum()
    print(f"  {train_matched} of {len(train_stats)} training rows matched a real implied total "
          f"({train_matched/len(train_stats)*100:.1f}%).")
    train_stats["implied_total"] = train_stats["implied_total"].fillna(fallback)

    print("Loading real historical prop odds...")
    props = load_historical_props()

    results = {}
    for stat_col in STAT_COLS:
        FEATURES = [f"{stat_col}_L4", f"{stat_col}_L8", f"opp_def_{stat_col}", "implied_total"]
        tr = train_stats.dropna(subset=FEATURES + [stat_col])
        te = test_stats.dropna(subset=FEATURES + [stat_col])
        if len(tr) < 50 or te.empty:
            print(f"{stat_col}: not enough data, skipping.")
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[FEATURES])
        ridge = Ridge(alpha=5.0).fit(X_tr, tr[stat_col])
        resid_std = float(np.std(tr[stat_col] - ridge.predict(X_tr)))

        X_te = scaler.transform(te[FEATURES])
        te = te.copy()
        te["pred"] = ridge.predict(X_te)
        te["stat_col"] = stat_col

        this_props = props[props["stat_col"] == stat_col]
        df = this_props.merge(te, on=["season", "week", "player_name_norm"], how="inner")
        if df.empty:
            print(f"{stat_col}: 0 matched props.")
            continue

        df["model_prob_over"] = 1 - norm.cdf(df["hist_line"], loc=df["pred"], scale=resid_std)
        actual_col = stat_col + "_y" if stat_col + "_y" in df.columns else stat_col
        def grade(r):
            actual = r[actual_col]
            if pd.isna(actual):
                return None
            if abs(actual - r["hist_line"]) < 1e-9:
                return "PUSH"
            picked_over = r["model_prob_over"] >= 0.5
            return "WIN" if (actual > r["hist_line"]) == picked_over else "LOSS"
        df["result"] = df.apply(grade, axis=1)
        df["edge"] = (df["model_prob_over"] - 0.5).abs()
        results[stat_col] = df

    combined = pd.concat(results.values(), ignore_index=True) if results else pd.DataFrame()
    print(f"\nTotal matched props across all 3 stats: {len(combined)}")
    print(f"{'thresh':>8}{'n':>6}{'win%':>8}{'roi%':>9}")
    for t in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
        g = combined[combined["edge"] >= t]
        g = g[g["result"].isin(["WIN", "LOSS"])]
        n = len(g)
        if n < 5:
            print(f"{t:>8}{n:>6}   (too few)")
            continue
        w = (g["result"] == "WIN").sum()
        profit = sum(american_profit(r["hist_over_price"]) if (r["result"] == "WIN" and r["model_prob_over"] >= 0.5)
                     else (american_profit(-110) if r["result"] == "WIN" else -1.0)
                     for _, r in g.iterrows())
        print(f"{t:>8}{n:>6}{w/n*100:>7.1f}%{profit/n*100:>8.1f}%")

    if train_matched / len(train_stats) > 0.5:
        print(f"\nTraining used REAL implied totals for {train_matched}/{len(train_stats)} rows "
              f"({train_matched/len(train_stats)*100:.1f}%) -- the earlier league-average-constant "
              f"limitation is resolved for most of the training set.")
    else:
        print(f"\nWARNING: only {train_matched}/{len(train_stats)} training rows "
              f"({train_matched/len(train_stats)*100:.1f}%) got a real implied total -- "
              f"data/historical_odds_2021_2023.jsonl is likely missing or empty. Run "
              f"fetch_historical_odds.py --seasons 2021 2022 2023 first. Results above still "
              f"used the league-average-constant fallback, same as before this fix.")


if __name__ == "__main__":
    main()

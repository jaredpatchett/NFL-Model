"""
backtest_player_props_profitability.py -- The genuine profitability test
for Track C's player props (rec yards, receptions, rush yards), same
principle as backtest_profitability.py did for Track B: grade the model's
OWN win probability (trailing average + STAT_RESID_STD, already validated
in generate_fantasy_projections.py against real 2023-2025 data) against
REAL point-in-time odds instead of an after-the-fact market price.

Requires data/historical_player_props_2024_2025.jsonl (run
fetch_historical_player_props.py first).

Usage:
    python backtest_player_props_profitability.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_fantasy_projections import STAT_RESID_STD, TRAILING_WINDOW, SHRINKAGE_GAMES
# fetch_player_stats.py lives at the REPO ROOT, not scripts/ -- same real
# import-path bug caught and fixed once already this session in
# generate_fantasy_projections.py (see that file's load_inputs() comments).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_player_stats import fetch_player_stats
import odds_api

HIST_PROPS_PATH = "../data/historical_player_props_2024_2025.jsonl"
SEASONS = [2024, 2025]

FIELD_TO_STAT_COL = {"player_rush_yds": "rushing_yards",
                      "player_reception_yds": "receiving_yards",
                      "player_receptions": "receptions"}


def american_profit(price, stake=1.0):
    return stake * (price / 100) if price > 0 else stake * (100 / -price)


def load_historical_props() -> pd.DataFrame:
    if not os.path.exists(HIST_PROPS_PATH):
        raise FileNotFoundError(f"{HIST_PROPS_PATH} not found -- run fetch_historical_player_props.py first.")
    rows = [json.loads(l) for l in open(HIST_PROPS_PATH)]
    out = []
    for r in rows:
        best = {}  # (player_norm, stat_col) -> {line, over_price}
        for bk in r.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                stat_col = FIELD_TO_STAT_COL.get(mkt["key"])
                if stat_col is None:
                    continue
                for o in mkt.get("outcomes", []):
                    if o["name"] != "Over":
                        continue
                    player_norm = odds_api.normalize_name(o.get("description", ""))
                    key = (player_norm, stat_col)
                    if key not in best or o["price"] > best[key]["over_price"]:
                        best[key] = {"line": o.get("point"), "over_price": o["price"],
                                     "player_name_raw": o.get("description")}
        for (player_norm, stat_col), v in best.items():
            out.append({
                "season": r["season"], "week": r["week"],
                "player_name_norm": player_norm, "player_name_raw": v["player_name_raw"],
                "stat_col": stat_col, "hist_line": v["line"], "hist_over_price": v["over_price"],
            })
    return pd.DataFrame(out)


def build_trailing_stat_avg(stats: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """Same shrinkage-toward-own-trailing-average logic as
    generate_fantasy_projections.py's build_trailing_player_avg, applied
    generically to one stat column. LEAKAGE-SAFE for backtesting purposes:
    shifted by 1 game so a game's own result never enters its own
    prediction -- deliberately different from the LIVE version (which
    doesn't shift, since it's projecting a genuinely future week). Here
    we're grading historical weeks against each other, so the shift matters."""
    stats = stats.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    shifted = stats.groupby("player_id")[stat_col].shift(1)
    stats[f"{stat_col}_trailing"] = shifted.groupby(stats["player_id"]).transform(
        lambda s: s.rolling(TRAILING_WINDOW, min_periods=1).mean()
    )
    return stats


def main():
    print("Loading real player stats (2024+2025)...")
    frames = []
    for s in SEASONS:
        df = fetch_player_stats(s)
        frames.append(df)
    stats = pd.concat(frames, ignore_index=True)
    stats["player_name_norm"] = stats["player_display_name"].apply(odds_api.normalize_name)

    print("Building leakage-safe trailing averages per stat...")
    for stat_col in ["rushing_yards", "receiving_yards", "receptions"]:
        stats = build_trailing_stat_avg(stats, stat_col)

    print("Loading real historical prop odds...")
    props = load_historical_props()
    print(f"Loaded {len(props)} real historical prop lines.")

    df = props.merge(stats, on=["season", "week", "player_name_norm"], how="inner")
    print(f"Matched {len(df)} of {len(props)} prop lines to a real player-week.")
    if df.empty:
        print("No matches -- check name normalization / season+week alignment.")
        return

    def compute_row(r):
        trailing = r.get(f"{r['stat_col']}_trailing")
        actual = r.get(r["stat_col"])
        resid_std = STAT_RESID_STD.get(r["stat_col"], {}).get(r["position"])
        if pd.isna(trailing) or resid_std is None or pd.isna(r["hist_line"]):
            return pd.Series({"model_prob_over": np.nan, "result": None})
        model_prob = 1 - norm.cdf(r["hist_line"], loc=trailing, scale=resid_std)
        if pd.isna(actual):
            return pd.Series({"model_prob_over": model_prob, "result": None})
        if abs(actual - r["hist_line"]) < 1e-9:
            result = "PUSH"
        else:
            picked_over = model_prob >= 0.5
            went_over = actual > r["hist_line"]
            result = "WIN" if went_over == picked_over else "LOSS"
        return pd.Series({"model_prob_over": model_prob, "result": result})

    extra = df.apply(compute_row, axis=1)
    df = pd.concat([df, extra], axis=1)
    df["edge"] = (df["model_prob_over"] - 0.5).abs()  # distance from a coinflip, proxy for confidence
    df["devig_dir_prob"] = np.where(df["model_prob_over"] >= 0.5, df["model_prob_over"], 1 - df["model_prob_over"])

    print(f"\n{'thresh':>8}{'n':>6}{'win%':>8}{'roi%':>9}")
    for t in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
        g = df[df["edge"] >= t]
        g = g[g["result"].isin(["WIN", "LOSS"])]
        n = len(g)
        if n < 5:
            print(f"{t:>8}{n:>6}   (too few)")
            continue
        w = (g["result"] == "WIN").sum()
        profit = 0.0
        for _, r in g.iterrows():
            price = r["hist_over_price"] if r["model_prob_over"] >= 0.5 else -110  # no real Under price captured -- see caveat below
            profit += american_profit(price) if r["result"] == "WIN" else -1.0
        roi = profit / n * 100
        print(f"{t:>8}{n:>6}{w/n*100:>7.1f}%{roi:>8.1f}%")

    print("\nCAVEAT: only the Over price is captured from the real odds fetch -- Under bets in this "
          "sweep use an assumed -110, not a real captured price. ROI for Under-side picks is an "
          "approximation, not fully real, unlike the win/loss grading itself (which IS real).")


if __name__ == "__main__":
    main()

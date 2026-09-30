"""
backtest_td_profitability.py -- The genuine profitability test for Track A
(anytime-TD), same principle as backtest_profitability.py (Track B) and
backtest_player_props_profitability.py (Track C): retrain the EXACT SAME
model (same features, same LogisticRegression, same 2021-2023 train split
already validated at AUC 0.70 on held-out 2024) and grade it against REAL
point-in-time odds instead of a tiny, noisy live sample.

Requires data/historical_td_odds_2024_2025.jsonl (run
fetch_historical_td_odds.py first).

Usage:
    python backtest_td_profitability.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from player_td_features import build_player_td_table
from player_td_model import FEATURE_COLS, _prep, TRAIN_SEASONS
from nfl_data import load_pbp, load_snaps, load_id_crosswalk
from features import build_player_week_features
from fetch_historical_odds import TEAM_NAME_TO_ABBR
import odds_api
from blueprint_qualification import qualify

TEST_SEASONS = [2024, 2025]
HIST_TD_PATH = "../data/historical_td_odds_2024_2025.jsonl"


def american_profit(price, stake=1.0):
    return stake * (price / 100) if price > 0 else stake * (100 / -price)


def load_historical_td_odds() -> pd.DataFrame:
    if not os.path.exists(HIST_TD_PATH):
        raise FileNotFoundError(f"{HIST_TD_PATH} not found -- run fetch_historical_td_odds.py first.")
    rows = [json.loads(l) for l in open(HIST_TD_PATH)]
    if not rows:
        return pd.DataFrame(columns=["season", "week", "player_name_norm", "hist_price"])
    out = []
    for r in rows:
        best = {}  # player_norm -> best (highest) Yes price
        for bk in r.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] != "player_anytime_td":
                    continue
                for o in mkt.get("outcomes", []):
                    # anytime-TD market: outcome name is usually "Yes"/"No"
                    # with the player in "description", OR the player name
                    # is directly the outcome name with no Yes/No split --
                    # handle both real shapes seen in the live odds_api.py fetch.
                    name = o.get("name", "")
                    desc = o.get("description")
                    if desc:
                        if name not in ("Yes",):
                            continue
                        player_raw = desc
                    else:
                        player_raw = name
                    player_norm = odds_api.normalize_name(player_raw)
                    if player_norm not in best or o["price"] > best[player_norm]["price"]:
                        best[player_norm] = {"price": o["price"], "player_name_raw": player_raw}
        for player_norm, v in best.items():
            out.append({"season": r["season"], "week": r["week"],
                        "player_name_norm": player_norm, "player_name_raw": v["player_name_raw"],
                        "hist_price": v["price"]})
    return pd.DataFrame(out)


def american_to_implied_prob(price):
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def main():
    print(f"Building player-TD feature table ({TRAIN_SEASONS + TEST_SEASONS})...")
    full = build_player_td_table(TRAIN_SEASONS + TEST_SEASONS, upcoming_season=2099, upcoming_week=1)

    pbp = load_pbp(TRAIN_SEASONS + TEST_SEASONS)
    snaps = load_snaps(TRAIN_SEASONS + TEST_SEASONS)
    xwalk = load_id_crosswalk()
    raw = build_player_week_features(pbp, snaps, id_crosswalk=xwalk)
    target = raw[["season", "week", "player_id", "scored_td"]]
    full = full.merge(target, on=["season", "week", "player_id"], how="inner")
    full = _prep(full)
    print(f"Rows after cold-start filter: {len(full)}")

    train = full[full["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test = full[full["season"].isin(TEST_SEASONS)].reset_index(drop=True)
    print(f"Train rows: {len(train)}   Test rows (2024-2025): {len(test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train[FEATURE_COLS])
    logit = LogisticRegression(C=1.0, max_iter=1000)
    logit.fit(X_train_s, train["scored_td"])

    X_test_s = scaler.transform(test[FEATURE_COLS])
    test = test.copy()
    test["model_prob"] = logit.predict_proba(X_test_s)[:, 1]

    from sklearn.metrics import roc_auc_score
    print(f"Out-of-sample AUC on 2024-2025 (sanity check vs the validated 0.70 on 2024 alone): "
          f"{roc_auc_score(test['scored_td'], test['model_prob']):.4f}")

    # player_td_features.py's table only has the ABBREVIATED player_name
    # ("J.Conner"), not a full name -- real sportsbook data uses full names
    # ("James Conner"), so normalizing the abbreviated form produces a
    # completely different string and silently matches nothing. Caught
    # before spending real fetch credits on this, by testing the actual
    # column values rather than assuming a name field would work.
    # Pull real full names the same reliable way the props/fantasy
    # pipeline already does -- fetch_player_stats.py's player_id ->
    # player_display_name, not this table's own abbreviated name field.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fetch_player_stats import fetch_player_stats
    name_frames = []
    for s in sorted(set(test["season"])):
        sdf = fetch_player_stats(s)
        name_frames.append(sdf[["player_id", "player_display_name"]].drop_duplicates())
    names = pd.concat(name_frames, ignore_index=True).drop_duplicates(subset=["player_id"])
    test = test.merge(names, on="player_id", how="left")
    test["player_name_norm"] = test["player_display_name"].apply(
        lambda n: odds_api.normalize_name(n) if pd.notna(n) else None
    )
    unmatched = test["player_name_norm"].isna().sum()
    print(f"  {unmatched} of {len(test)} test rows have no full name available -- these can never "
          f"match a real odds line and will be dropped at the merge step below.")

    print("Loading real historical anytime-TD odds...")
    hist = load_historical_td_odds()
    print(f"Loaded {len(hist)} real historical TD price rows.")

    df = test.merge(hist, on=["season", "week", "player_name_norm"], how="inner")
    print(f"Matched {len(df)} of {len(test)} test player-games to a real historical TD price.")
    if df.empty:
        print("No matches -- check name normalization / season+week alignment.")
        return

    df["market_implied_prob"] = df["hist_price"].apply(american_to_implied_prob)
    df["edge"] = df["model_prob"] - df["market_implied_prob"]
    df["result"] = df["scored_td"].map({1: "WIN", 0: "LOSS"})

    # Real, important gap found in a live run: the deployed OFFICIAL/DEGEN
    # logic uses edge% ALONE, with none of the blueprint's own hard usage
    # filters (min 30% model probability, min 70% snap share, min 5%
    # red-zone role) -- those exist specifically to stop thin-usage bench
    # players from qualifying, and a live check showed exactly that failure:
    # backup TEs/WRs at 12-16% model probability, deep-bench snap shares,
    # flagged OFFICIAL purely because long-shot odds (+2000, +3000+) make
    # small absolute probability edges look large in percentage terms.
    # Re-running the SAME edge sweep gated on ALSO passing those hard
    # filters, rather than assuming edge% alone still means what it did in
    # the unfiltered backtest -- the point of testing this instead of just
    # adding the filter and hoping.
    def hard_fail(r):
        q = qualify(r, r["model_prob"], None)
        return q["tier"] == "D"
    df["hard_fail"] = df.apply(hard_fail, axis=1)
    print(f"\n{df['hard_fail'].sum()} of {len(df)} matched rows fail the blueprint's hard usage filters "
          f"(thin snap share, no real red-zone role, or model prob under 30%) -- these are exactly the "
          f"kind of long-shot bench players the live check flagged.")

    print(f"\n=== UNFILTERED (edge% alone, current live logic) ===")
    print(f"{'thresh':>8}{'n':>6}{'win%':>8}{'roi%':>9}")
    for t in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
        g = df[df["edge"] >= t]  # signed, positive-only -- same convention as the live blueprint (only bet when model > market)
        n = len(g)
        if n < 5:
            print(f"{t:>8}{n:>6}   (too few)")
            continue
        w = (g["result"] == "WIN").sum()
        profit = sum(american_profit(r["hist_price"]) if r["result"] == "WIN" else -1.0 for _, r in g.iterrows())
        print(f"{t:>8}{n:>6}{w/n*100:>7.1f}%{profit/n*100:>8.1f}%")

    print(f"\n=== FILTERED (edge% AND passes hard usage/role filters -- the proposed fix) ===")
    df_filtered = df[~df["hard_fail"]]
    print(f"{'thresh':>8}{'n':>6}{'win%':>8}{'roi%':>9}")
    for t in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
        g = df_filtered[df_filtered["edge"] >= t]
        n = len(g)
        if n < 5:
            print(f"{t:>8}{n:>6}   (too few)")
            continue
        w = (g["result"] == "WIN").sum()
        profit = sum(american_profit(r["hist_price"]) if r["result"] == "WIN" else -1.0 for _, r in g.iterrows())
        print(f"{t:>8}{n:>6}{w/n*100:>7.1f}%{profit/n*100:>8.1f}%")


if __name__ == "__main__":
    main()

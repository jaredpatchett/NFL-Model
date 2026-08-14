"""
backtest.py — Joins the persistent prediction logs (data/predictions_log.jsonl,
data/player_td_log.jsonl) against REAL, now-known game outcomes to report
genuine model performance: ATS/O-U/moneyline records and ROI for Track B,
hit rate and calibration for Track A.

WHY THIS IS DIFFERENT FROM game_lines_model.py / player_td_model.py's
VALIDATION: those check calibration against CLOSING lines from a historical
dataset -- useful for confirming the model itself is sound, but explicitly
NOT a valid profitability backtest per the blueprint's own principle (never
backtest against hindsight/closing prices). This script uses the ACTUAL
logged predictions_log.jsonl / player_td_log.jsonl entries -- genuine
pre-game snapshots, timestamped before any outcome was known, accumulated
by the production scripts on every cron run. This is the real thing.

WHICH SNAPSHOT COUNTS AS "THE BET": a game gets logged every cron run
between when it enters the upcoming slate and kickoff (multiple snapshots
as the week progresses). This script uses the LAST snapshot logged before
kickoff for each game/player -- the freshest information actually available,
which is what a bettor would have acted on. Earlier snapshots are still in
the log (untouched) for anyone who wants to study how predictions moved
over the week; this script just doesn't evaluate every one of them as if
each were a separate bet.

PRICE APPROXIMATION FOR ATS/TOTALS: predictions_log.jsonl records the
market LINE (spread_line, total_line) but not the per-side PRICE for those
markets (only moneyline prices are logged). Standard -110 is assumed for
ATS/totals ROI -- a reasonable default (most books' standard vig) but an
approximation; moneyline ROI uses the actual logged price.

EXPECT ~0 RESULTS RIGHT NOW: built ahead of the season specifically so it's
ready the moment games start finishing -- as of this build there are no
completed 2026 games yet. Validated against synthetic fixtures
(test_backtest.py) to confirm the join/ROI logic itself is correct before
there's real data to point it at.

Usage:
    python backtest.py
"""

from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd

from nfl_data import load_schedules

TRACK_B_LOG = "../data/predictions_log.jsonl"
TRACK_A_LOG = "../data/player_td_log.jsonl"
STANDARD_VIG_PRICE = -110  # assumed price for ATS/totals -- see module docstring


def _load_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _latest_snapshot(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """One row per key_cols group: the row with the max logged_at."""
    if df.empty:
        return df
    df = df.copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    idx = df.groupby(key_cols)["logged_at"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def profit_per_unit(price: float) -> float:
    """American odds -> profit on a 1-unit win (not counting the stake back)."""
    return price / 100 if price > 0 else 100 / -price


def _record_and_roi(bet_won: pd.Series, push: pd.Series, prices: pd.Series) -> dict:
    decided = ~push
    n_bets = int(decided.sum())
    if n_bets == 0:
        return {"n_bets": 0, "wins": 0, "losses": 0, "pushes": int(push.sum()),
                "win_pct": None, "roi_pct": None}
    wins = int((bet_won & decided).sum())
    losses = n_bets - wins
    profits = np.where(
        bet_won[decided], prices[decided].apply(profit_per_unit), -1.0
    )
    roi = float(np.sum(profits) / n_bets * 100)
    return {
        "n_bets": n_bets, "wins": wins, "losses": losses, "pushes": int(push.sum()),
        "win_pct": round(100 * wins / n_bets, 1), "roi_pct": round(roi, 2),
    }


def evaluate_track_b(bets_with_results: pd.DataFrame, min_edge: float = 0.0) -> dict:
    """
    Pure evaluation logic, separated from data-fetching so it's testable
    against synthetic fixtures (see test_backtest.py). Expects one row per
    game with both the logged prediction fields AND home_score/away_score
    already joined on.
    """
    df = bets_with_results.copy()
    if df.empty:
        return {"n_available": 0}

    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]

    # ---- ATS ----
    df["our_side_home"] = df["pred_home_margin"] > df["market_spread_line"]
    df["home_covers"] = df["actual_margin"] > df["market_spread_line"]
    df["ats_push"] = df["actual_margin"] == df["market_spread_line"]
    df["ats_won"] = np.where(df["our_side_home"], df["home_covers"], ~df["home_covers"])
    ats_all = _record_and_roi(df["ats_won"], df["ats_push"], pd.Series(STANDARD_VIG_PRICE, index=df.index))
    edge_mask = df["spread_edge"].abs() >= min_edge
    ats_edge = _record_and_roi(
        df.loc[edge_mask, "ats_won"], df.loc[edge_mask, "ats_push"],
        pd.Series(STANDARD_VIG_PRICE, index=df.index)[edge_mask],
    )

    # ---- Totals ----
    df["our_side_over"] = df["pred_total"] > df["market_total_line"]
    df["actual_over"] = df["actual_total"] > df["market_total_line"]
    df["total_push"] = df["actual_total"] == df["market_total_line"]
    df["total_won"] = np.where(df["our_side_over"], df["actual_over"], ~df["actual_over"])
    total_all = _record_and_roi(df["total_won"], df["total_push"], pd.Series(STANDARD_VIG_PRICE, index=df.index))

    # ---- Moneyline ----
    df["our_side_home_ml"] = df["home_win_prob"] > 0.5
    df["home_won"] = df["actual_margin"] > 0
    df["ml_push"] = df["actual_margin"] == 0  # a tie
    df["ml_won"] = np.where(df["our_side_home_ml"], df["home_won"], ~df["home_won"])
    ml_price = np.where(df["our_side_home_ml"], df["market_home_moneyline"], df["market_away_moneyline"])
    ml_all = _record_and_roi(df["ml_won"], df["ml_push"], pd.Series(ml_price, index=df.index))

    return {
        "n_available": len(df), "ats_all": ats_all, "ats_edge_filtered": ats_edge,
        "totals_all": total_all, "moneyline_all": ml_all,
    }


def backtest_track_b(min_edge: float = 0.0) -> dict:
    log = _load_jsonl(TRACK_B_LOG)
    if log.empty:
        print("Track B: no predictions_log.jsonl entries found.")
        return {}

    bets = _latest_snapshot(log, ["game_id"])
    seasons = sorted(bets["season"].unique().tolist())
    sched = load_schedules(seasons)
    results = sched[sched["home_score"].notna()][
        ["game_id", "home_score", "away_score"]
    ]

    df = bets.merge(results, on="game_id", how="inner")
    print(f"Track B: {len(bets)} games logged, {len(df)} have final scores available.")
    if df.empty:
        return {"n_available": 0}

    out = evaluate_track_b(df, min_edge=min_edge)
    print(f"\n  ATS (all logged bets):        {out['ats_all']}")
    print(f"  ATS (|edge| >= {min_edge} pts):   {out['ats_edge_filtered']}")
    print(f"  Totals (all logged bets):     {out['totals_all']}")
    print(f"  Moneyline (all logged bets):  {out['moneyline_all']}")
    return out


def evaluate_track_a(bets_with_results: pd.DataFrame) -> dict:
    """
    Pure evaluation logic for Track A, separated from data-fetching for
    testability. Expects one row per player-game with scored_td already
    joined on.
    """
    df = bets_with_results.copy()
    if df.empty:
        return {"n_available": 0}

    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    out = {
        "n_available": len(df),
        "hit_rate": float(df["scored_td"].mean()),
        "mean_predicted_prob": float(df["anytime_td_prob"].mean()),
    }
    if df["scored_td"].nunique() > 1:
        out["auc"] = float(roc_auc_score(df["scored_td"], df["anytime_td_prob"]))
        out["log_loss"] = float(log_loss(df["scored_td"], df["anytime_td_prob"]))
        out["brier"] = float(brier_score_loss(df["scored_td"], df["anytime_td_prob"]))

    priced = df.dropna(subset=["live_anytime_td_price"])
    if len(priced):
        priced = priced.copy()
        priced["bet_won"] = priced["scored_td"] == 1
        profits = np.where(
            priced["bet_won"], priced["live_anytime_td_price"].apply(profit_per_unit), -1.0
        )
        out["n_priced_bets"] = len(priced)
        out["roi_all_pct"] = round(100 * float(np.mean(profits)), 2)

        positive_edge = priced[priced["edge"] > 0] if "edge" in priced.columns else pd.DataFrame()
        if len(positive_edge):
            pe_profits = np.where(
                positive_edge["scored_td"] == 1,
                positive_edge["live_anytime_td_price"].apply(profit_per_unit), -1.0,
            )
            out["n_positive_edge_bets"] = len(positive_edge)
            out["roi_positive_edge_pct"] = round(100 * float(np.mean(pe_profits)), 2)
    else:
        out["n_priced_bets"] = 0

    return out


def backtest_track_a() -> dict:
    log = _load_jsonl(TRACK_A_LOG)
    if log.empty:
        print("Track A: no player_td_log.jsonl entries found.")
        return {}

    bets = _latest_snapshot(log, ["season", "week", "player_id"])
    seasons = sorted(bets["season"].unique().tolist())

    # Real outcomes come from play-by-play (scored_td), not the schedule.
    sched = load_schedules(seasons)
    played_seasons = [s for s in seasons if sched.loc[sched["season"] == s, "home_score"].notna().any()]
    if not played_seasons:
        print(f"Track A: {len(bets)} player-games logged, 0 have played seasons with real outcomes yet.")
        return {"n_available": 0}

    from nfl_data import load_pbp, load_snaps, load_id_crosswalk
    from features import build_player_week_features
    pbp = load_pbp(played_seasons)
    snaps = load_snaps(played_seasons)
    xwalk = load_id_crosswalk()
    raw = build_player_week_features(pbp, snaps, id_crosswalk=xwalk)
    results = raw[["season", "week", "player_id", "scored_td"]]

    df = bets.merge(results, on=["season", "week", "player_id"], how="inner")
    print(f"Track A: {len(bets)} player-games logged, {len(df)} have real outcomes available.")
    if df.empty:
        return {"n_available": 0}

    out = evaluate_track_a(df)
    print(f"\n  Actual anytime-TD rate among logged predictions: {out['hit_rate']:.3f}")
    print(f"  Mean predicted probability: {out['mean_predicted_prob']:.3f}")
    if "auc" in out:
        print(f"  AUC: {out['auc']:.4f}")
        print(f"  LogLoss: {out['log_loss']:.4f}")
        print(f"  Brier: {out['brier']:.4f}")
    if out.get("n_priced_bets"):
        print(f"\n  Bets with a live price at logging time: {out['n_priced_bets']}")
        print(f"  ROI if betting every logged player: {out['roi_all_pct']:.2f}%")
        if "roi_positive_edge_pct" in out:
            print(f"  ROI if betting only positive-edge picks ({out['n_positive_edge_bets']} bets): "
                  f"{out['roi_positive_edge_pct']:.2f}%")
    else:
        print("\n  No logged predictions had a live price at the time -- ROI not computable "
              "(this is expected for any run before ODDS_API_KEY was connected).")

    return out


def main():
    print("=" * 70)
    print("TRACK B BACKTEST (spreads / totals / moneyline)")
    print("=" * 70)
    backtest_track_b()

    print("\n" + "=" * 70)
    print("TRACK A BACKTEST (anytime TD)")
    print("=" * 70)
    backtest_track_a()


if __name__ == "__main__":
    main()

"""
test_backtest.py — Validates backtest.py's evaluation logic (ATS/totals/
moneyline record + ROI for Track B, hit rate/ROI for Track A) against
synthetic fixtures with hand-computable expected results. No network needed
-- evaluate_track_b()/evaluate_track_a() take already-joined DataFrames, so
this never touches load_schedules() or any real data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from backtest import evaluate_track_b, evaluate_track_a, profit_per_unit, _latest_snapshot


def approx(a, b, tol=0.01):
    if a is None or b is None:
        return a == b
    return abs(a - b) < tol


def test_profit_per_unit():
    return [
        ("profit +150", profit_per_unit(150), 1.5),
        ("profit -150", profit_per_unit(-150), 100 / 150),
    ]


def test_latest_snapshot():
    """3 snapshots for one game at different times -- must pick the latest."""
    df = pd.DataFrame([
        {"game_id": "g1", "logged_at": "2026-09-08T10:00:00+00:00", "pred_home_margin": 3.0},
        {"game_id": "g1", "logged_at": "2026-09-09T10:00:00+00:00", "pred_home_margin": 4.0},
        {"game_id": "g1", "logged_at": "2026-09-07T10:00:00+00:00", "pred_home_margin": 2.0},
    ])
    latest = _latest_snapshot(df, ["game_id"])
    return [("latest snapshot picks most recent logged_at", latest.iloc[0]["pred_home_margin"], 4.0)]


def test_track_b_ats_totals_ml():
    """
    Hand-computable 4-game fixture:
      G1: home favored by our model AND covers -> ATS win for home side
      G2: home favored by market, we predict away side wins ATS, and away covers -> ATS win
      G3: push (actual margin == spread line exactly)
      G4: our side loses ATS
    Standard -110 assumed on all ATS/total bets -> profit_per_unit(-110) = 100/110 = 0.9091
    """
    rows = [
        # G1: market spread_line=3 (home favored by 3), we predict home margin=7 (favor home),
        #     actual: home wins by 10 -> home covers (10>3) -> WIN
        {"game_id": "g1", "pred_home_margin": 7.0, "market_spread_line": 3.0,
         "pred_total": 45.0, "market_total_line": 44.0,
         "home_win_prob": 0.7, "market_home_moneyline": -150, "market_away_moneyline": 130,
         "home_score": 27, "away_score": 17},  # margin=10, total=44
        # G2: market spread_line=3, we predict home margin=1 (favor AWAY side, since 1<3),
        #     actual: home wins by only 2 -> home does NOT cover (2<3) -> away covers -> WIN for us
        {"game_id": "g2", "pred_home_margin": 1.0, "market_spread_line": 3.0,
         "pred_total": 40.0, "market_total_line": 44.0,
         "home_win_prob": 0.55, "market_home_moneyline": -120, "market_away_moneyline": 100,
         "home_score": 20, "away_score": 18},  # margin=2, total=38
        # G3: push -- actual margin exactly equals spread line
        {"game_id": "g3", "pred_home_margin": 5.0, "market_spread_line": 3.0,
         "pred_total": 44.0, "market_total_line": 44.0,
         "home_win_prob": 0.6, "market_home_moneyline": -130, "market_away_moneyline": 110,
         "home_score": 24, "away_score": 21},  # margin=3 == spread_line -> push; total=45!=44
        # G4: we favor home (pred_margin=6 > spread_line=3), actual home wins by only 1 -> LOSS
        {"game_id": "g4", "pred_home_margin": 6.0, "market_spread_line": 3.0,
         "pred_total": 50.0, "market_total_line": 44.0,
         "home_win_prob": 0.65, "market_home_moneyline": -140, "market_away_moneyline": 120,
         "home_score": 22, "away_score": 21},  # margin=1, total=43
    ]
    for r in rows:
        r["spread_edge"] = r["pred_home_margin"] - r["market_spread_line"]
    df = pd.DataFrame(rows)

    out = evaluate_track_b(df, min_edge=0.0)
    checks = []
    checks.append(("n_available", out["n_available"], 4))

    # ATS: G1 win, G2 win, G3 push, G4 loss -> 2 wins, 1 loss, 1 push, n_bets=3 (decided)
    ats = out["ats_all"]
    checks.append(("ATS n_bets (excl push)", ats["n_bets"], 3))
    checks.append(("ATS wins", ats["wins"], 2))
    checks.append(("ATS losses", ats["losses"], 1))
    checks.append(("ATS pushes", ats["pushes"], 1))
    # ROI: 2 wins @ profit_per_unit(-110)=0.9091, 1 loss @ -1, over 3 bets
    expected_ats_roi = ((2 * (100/110)) + (1 * -1)) / 3 * 100
    checks.append(("ATS ROI %", ats["roi_pct"], round(expected_ats_roi, 2)))

    # Totals: G1 total_line=44, actual=44 -> push (44==44). G2: pred=40<44 (under), actual=38<44 -> under wins -> WIN
    # G3: pred=44==market(44) -> our_side_over = (44>44)=False -> under; actual=45>44 -> over hits -> our under LOSES
    # G4: pred=50>44 (over), actual=43<44 -> under hits -> our over LOSES
    totals = out["totals_all"]
    checks.append(("Totals n_bets (excl push)", totals["n_bets"], 3))
    checks.append(("Totals wins", totals["wins"], 1))
    checks.append(("Totals losses", totals["losses"], 2))
    checks.append(("Totals pushes", totals["pushes"], 1))

    # Moneyline: all 4 games favor home (prob>0.5), all 4 home teams won (positive margins) -> 4 wins
    ml = out["moneyline_all"]
    checks.append(("ML n_bets", ml["n_bets"], 4))
    checks.append(("ML wins", ml["wins"], 4))
    checks.append(("ML losses", ml["losses"], 0))

    return checks


def test_track_a_hit_rate_and_roi():
    """
    3 players: 2 scored, 1 didn't. 2 have a live price logged.
    """
    df = pd.DataFrame([
        {"player_id": "p1", "anytime_td_prob": 0.6, "scored_td": 1,
         "live_anytime_td_price": -150, "edge": 0.05},
        {"player_id": "p2", "anytime_td_prob": 0.3, "scored_td": 0,
         "live_anytime_td_price": 120, "edge": -0.02},
        {"player_id": "p3", "anytime_td_prob": 0.4, "scored_td": 1,
         "live_anytime_td_price": None, "edge": None},
    ])
    out = evaluate_track_a(df)
    checks = []
    checks.append(("n_available", out["n_available"], 3))
    checks.append(("hit_rate", round(out["hit_rate"], 4), round(2/3, 4)))
    checks.append(("n_priced_bets", out["n_priced_bets"], 2))
    # p1 won @ -150 -> profit_per_unit(-150)=100/150=0.6667; p2 lost @ 120 -> -1
    expected_roi_all = ((100/150) + (-1)) / 2 * 100
    checks.append(("roi_all_pct", out["roi_all_pct"], round(expected_roi_all, 2)))
    # positive edge only: p1 (edge=0.05, won) -> roi = +0.6667*100
    checks.append(("n_positive_edge_bets", out["n_positive_edge_bets"], 1))
    checks.append(("roi_positive_edge_pct", out["roi_positive_edge_pct"], round((100/150)*100, 2)))
    return checks


def main():
    all_checks = (
        test_profit_per_unit()
        + test_latest_snapshot()
        + test_track_b_ats_totals_ml()
        + test_track_a_hit_rate_and_roi()
    )

    print(f"{'check':40s} {'got':>15s} {'want':>15s}  ok")
    print("-" * 78)
    all_ok = True
    for name, got, want in all_checks:
        ok = approx(got, want) if isinstance(want, float) else (got == want)
        all_ok &= ok
        print(f"{name:40s} {str(got):>15s} {str(want):>15s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

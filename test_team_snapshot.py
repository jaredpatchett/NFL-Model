"""
test_team_snapshot.py — Validates team_snapshot.py against a hand-designed
synthetic league where the correct answer is known in advance:
  - SCORING / DEFENSE / CEILING are checked EXACTLY (simple arithmetic on
    the same games fed in, computed independently of the module under test).
  - NET / FORM are checked DIRECTIONALLY: Team A is deliberately built to
    play badly in the older half of its window and dominate in the recent
    half, so FORM must come out clearly positive; a control team (D) plays
    at a constant level throughout, so FORM must come out near zero for it.
  - Trailing-window "tail" behavior (bye weeks, fewer than N games
    available) is checked against a team with an irregular schedule.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from team_snapshot import build_team_snapshots, _team_game_log, _percentile_rank


def game(season, week, home, away, home_score, away_score):
    return {
        "season": season, "week": week,
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "home_team": home, "away_team": away,
        "home_score": home_score, "away_score": away_score,
    }


def build_schedule():
    rows = []
    # Team A: older half (weeks 1-3) plays badly, recent half (weeks 4-6)
    # dominates. Alternating home/away against B to keep it simple.
    rows += [
        game(2026, 1, "B", "A", 30, 10),   # A away, loses 10-30
        game(2026, 2, "A", "B", 14, 28),   # A home, loses 14-28
        game(2026, 3, "B", "A", 24, 17),   # A away, loses 17-24
        game(2026, 4, "A", "B", 35, 10),   # A home, wins 35-10
        game(2026, 5, "B", "A", 14, 40),   # A away, wins 40-14
        game(2026, 6, "A", "B", 28, 7),    # A home, wins 28-7
    ]
    # Team D: plays at a constant, boring level across the same 6 weeks
    # (a fixed 20-20 result every time) -- control case, FORM should be ~0.
    rows += [
        game(2026, 1, "D", "C", 20, 20),
        game(2026, 2, "C", "D", 20, 20),
        game(2026, 3, "D", "C", 20, 20),
        game(2026, 4, "C", "D", 20, 20),
        game(2026, 5, "D", "C", 20, 20),
        game(2026, 6, "C", "D", 20, 20),
    ]
    # A few extra cross games so the SRS graph is better connected (using
    # teams NOT already in the precisely-designed 6-game A/B scenario above,
    # so they don't corrupt that exact game count).
    rows += [
        game(2026, 3, "D", "B", 17, 17),
        game(2026, 5, "C", "B", 24, 24),
    ]
    # Team E: irregular schedule -- only 4 games total before week 7, to
    # test that a "last 6" window honestly reports games=4, not a padded 6.
    rows += [
        game(2026, 1, "E", "F", 10, 9),
        game(2026, 2, "F", "E", 13, 12),
        game(2026, 4, "E", "F", 20, 17),
        game(2026, 6, "F", "E", 15, 14),
    ]
    return pd.DataFrame(rows)


def main():
    checks = []
    sched = build_schedule()

    snapshots = build_team_snapshots(sched, as_of_season=2026, as_of_week=7, windows=(6, 12))
    win6 = snapshots[6]

    # ---- Exact arithmetic checks: Team A, last 6 games ----
    a = win6["A"]
    checks.append(("A games used = 6", a["games"], 6))
    checks.append(("A scoring = mean(10,14,17,35,40,28) = 24.0", a["scoring"], 24.0))
    checks.append(("A defense = mean(30,28,24,10,14,7) = 18.8", a["defense"], round((30+28+24+10+14+7)/6, 1)))
    checks.append(("A ceiling = max(...) = 40.0", a["ceiling"], 40.0))

    # ---- Directional checks: FORM ----
    # NOTE: this synthetic league is deliberately tiny (6 teams), so each
    # half-window SRS solve only has a handful of games in its pool. Ridge
    # alpha=25 (reused as-is from power_ratings.py, which was calibrated for
    # full-season solves with far more games) shrinks a small-sample solve
    # like this MUCH harder than it will at real 32-team NFL scale, where a
    # half-window pool naturally has many more games. So this checks
    # DIRECTION and separation from a zero-trend control team, not a
    # specific magnitude -- and this is a genuine real-world consideration
    # worth revisiting with real data: alpha=25 may be over-aggressive for
    # the half-window FORM solve specifically, even though it's well-suited
    # to the full-season solve it was originally tuned for.
    checks.append(("A form is positive (dominant recent half vs bad older half)", a["form"] > 0, True))
    d = win6["D"]
    checks.append(("A's form is clearly separated from D's (near-zero, constant performance)",
                    a["form"] - d["form"] > 2, True))

    # ---- Directional checks: NET should reflect A's real recent strength ----
    checks.append(("A's NET rating is positive (net positive across the window overall)", a["net"] > 0, True))

    # ---- Percentile inversion sanity: defense ----
    # Independently compute defense percentile ranks and confirm the team
    # with the LOWEST points-allowed gets the HIGHEST percentile.
    defense_vals = {t: v["defense"] for t, v in win6.items() if v is not None}
    best_defense_team = min(defense_vals, key=defense_vals.get)
    best_defense_pctile = win6[best_defense_team]["defense_pctile"]
    checks.append((f"team with lowest points-allowed ({best_defense_team}) has the TOP defense percentile",
                    best_defense_pctile, max(v["defense_pctile"] for v in win6.values() if v is not None)))

    # ---- Strength summary is on a 1-99 scale ----
    checks.append(("strength is within [1,99] for every team with data",
                    all(1 <= v["strength"] <= 99 for v in win6.values() if v is not None), True))

    # ---- Honest "games" count for an irregular schedule (E has only 4 real games) ----
    e6 = win6["E"]
    checks.append(("E (irregular schedule) reports real games=4, not padded to 6", e6["games"], 4))
    checks.append(("E scoring = mean(10,12,20,14) = 14.0", e6["scoring"], 14.0))

    # ---- Team with zero games before this point ----
    # (not present in this synthetic schedule at all -- team_game_log should
    # just return an empty log, and the caller should report None cleanly)
    ghost_log = _team_game_log(sched, "ZZ")
    checks.append(("a team with literally zero games returns an empty log, not an error", len(ghost_log), 0))

    # ---- 12-game window on a league that only has 6 weeks of history ----
    # every team should report games <= 6 (can't have more games than exist)
    win12 = snapshots[12]
    checks.append(("12-game window still caps at real available history (A: games<=6)", win12["A"]["games"] <= 6, True))

    # ---- _percentile_rank helper: basic correctness on known input ----
    p = _percentile_rank({"x": 10, "y": 20, "z": 30})
    checks.append(("percentile: highest raw value gets 100", p["z"], 100.0))
    checks.append(("percentile: lowest raw value gets 0", p["x"], 0.0))
    p_inv = _percentile_rank({"x": 10, "y": 20, "z": 30}, invert=True)
    checks.append(("percentile inverted: lowest raw value now gets 100", p_inv["x"], 100.0))

    print(f"{'check':78s} {'got':>8s} {'want':>8s}  ok")
    print("-" * 100)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:78s} {str(got):>8s} {str(want):>8s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 100)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

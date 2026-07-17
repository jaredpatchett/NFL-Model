"""
test_features.py — Validate features.py against synthetic play-by-play
with hand-computable answers. No network needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from features import build_player_week_features


def make_synthetic_pbp():
    """
    One team (BUF), one week. Two players:
      RB1 (rusher): a goal-line back
      WR1 (receiver): a red-zone target hog
    We construct plays so the shares are known by hand.
    """
    rows = []

    def play(play_type, yl, rusher=None, receiver=None, rtd=0, ptd=0):
        rows.append({
            "season": 2024, "week": 1, "posteam": "BUF",
            "play_type": play_type, "yardline_100": yl,
            "rusher_player_id": rusher, "receiver_player_id": receiver,
            "rush_touchdown": rtd, "pass_touchdown": ptd,
            "touchdown": max(rtd, ptd),
        })

    # RB1 rushes: 2 inside-5 (1 TD), 1 inside-10, 3 midfield
    play("run", 3, rusher="RB1", rtd=1)
    play("run", 4, rusher="RB1", rtd=0)
    play("run", 8, rusher="RB1", rtd=0)
    play("run", 45, rusher="RB1", rtd=0)
    play("run", 50, rusher="RB1", rtd=0)
    play("run", 60, rusher="RB1", rtd=0)

    # RB2 rushes: 1 inside-5 (no TD), 2 midfield  -> gives RB1 a share < 1.0
    play("run", 2, rusher="RB2", rtd=0)
    play("run", 40, rusher="RB2", rtd=0)
    play("run", 55, rusher="RB2", rtd=0)

    # WR1 targets: 2 red-zone (1 TD), 2 midfield
    play("pass", 15, receiver="WR1", ptd=1)
    play("pass", 18, receiver="WR1", ptd=0)
    play("pass", 35, receiver="WR1", ptd=0)
    play("pass", 42, receiver="WR1", ptd=0)

    # WR2 targets: 1 red-zone, 3 midfield
    play("pass", 12, receiver="WR2", ptd=0)
    play("pass", 30, receiver="WR2", ptd=0)
    play("pass", 33, receiver="WR2", ptd=0)
    play("pass", 48, receiver="WR2", ptd=0)

    return pd.DataFrame(rows)


def make_synthetic_snaps():
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "BUF", "player": "RB1",
         "pfr_player_id": "RB1", "offense_pct": 0.72},
        {"season": 2024, "week": 1, "team": "BUF", "player": "WR1",
         "pfr_player_id": "WR1", "offense_pct": 0.88},
    ])


def make_synthetic_snaps_realistic_ids():
    """
    Snap counts with PFR-style IDs that DELIBERATELY do not match the GSIS-style
    player IDs used in pbp (e.g. rusher_player_id="RB1"). This mirrors real
    nflverse data, where snap counts are keyed by PFR id and pbp is keyed by
    GSIS id — two different namespaces. A naive direct merge (no crosswalk)
    must fail to match here; only the crosswalk path should recover it.
    """
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "BUF", "player": "RB One",
         "pfr_player_id": "OneRB00", "offense_pct": 0.72},
        {"season": 2024, "week": 1, "team": "BUF", "player": "WR One",
         "pfr_player_id": "OneWR00", "offense_pct": 0.88},
    ])


def make_synthetic_id_crosswalk():
    return pd.DataFrame([
        {"gsis_id": "RB1", "pfr_id": "OneRB00", "name": "RB One", "position": "RB"},
        {"gsis_id": "WR1", "pfr_id": "OneWR00", "name": "WR One", "position": "WR"},
    ])


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def main():
    pbp = make_synthetic_pbp()
    snaps = make_synthetic_snaps()
    feat = build_player_week_features(pbp, snaps)

    rb1 = feat[feat["player_id"] == "RB1"].iloc[0]
    wr1 = feat[feat["player_id"] == "WR1"].iloc[0]

    checks = []

    # --- Regression test: gsis/pfr id crosswalk (see HANDOFF notes) ---
    # Without a crosswalk, mismatched-namespace snap IDs must NOT silently match.
    snaps_mismatched = make_synthetic_snaps_realistic_ids()
    feat_no_xwalk = build_player_week_features(pbp, snaps_mismatched)  # no crosswalk passed
    rb1_no_xwalk = feat_no_xwalk[feat_no_xwalk["player_id"] == "RB1"].iloc[0]
    checks.append(("RB1 snap_share is NaN with mismatched ids + no crosswalk",
                   pd.isna(rb1_no_xwalk["snap_share"]), True))

    # With the crosswalk, it should resolve correctly.
    xwalk = make_synthetic_id_crosswalk()
    feat_xwalk = build_player_week_features(pbp, snaps_mismatched, id_crosswalk=xwalk)
    rb1_xwalk = feat_xwalk[feat_xwalk["player_id"] == "RB1"].iloc[0]
    checks.append(("RB1 snap_share resolves via crosswalk", rb1_xwalk["snap_share"], 0.72))

    # Team rush carries = 9 (RB1 6 + RB2 3). RB1 carry share = 6/9.
    checks.append(("RB1 carry_share", rb1["carry_share"], 6/9))

    # Team inside-5 carries = 3 (RB1 2 + RB2 1). RB1 inside5 share = 2/3.
    checks.append(("RB1 inside5_carry_share", rb1["inside5_carry_share"], 2/3))

    # Team pass targets = 8 (WR1 4 + WR2 4). WR1 target share = 4/8.
    checks.append(("WR1 target_share", wr1["target_share"], 0.5))

    # Team RZ (inside-20) targets: WR1 has 2 (yl 15,18), WR2 has 1 (yl12) = 3.
    # WR1 rz_target_share = 2/3.
    checks.append(("WR1 rz_target_share", wr1["rz_target_share"], 2/3))

    # RB1 actual TDs = 1 rush. scored_td = 1.
    checks.append(("RB1 actual_tds", rb1["actual_tds"], 1))
    checks.append(("RB1 scored_td", rb1["scored_td"], 1))

    # RB1 expected TDs: 2*inside5(0.34) + 1*inside10(0.18) + 3*field(0.007)
    exp_rb1 = 2*0.34 + 1*0.18 + 3*0.007
    checks.append(("RB1 expected_tds", rb1["expected_tds"], exp_rb1))

    # Snap share merged
    checks.append(("RB1 snap_share", rb1["snap_share"], 0.72))

    print(f"{'check':30s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 60)
    all_ok = True
    for name, got, want in checks:
        ok = approx(float(got), float(want))
        all_ok &= ok
        print(f"{name:30s} {got:10.5f} {want:10.5f}  {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

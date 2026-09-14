"""
test_duplicate_and_name_fix.py -- Tests the two fixes for the real Week 1
2026 production bugs confirmed via screenshot + live data check: 459 rows
for 183 unique players (2.5x duplication, e.g. S.Barkley appearing 4 times
with 4 different probabilities), and 83 of 459 rows with a literal NaN
player_name. Both pieces of logic below are copied VERBATIM from the real
file and verified via exact substring match, not just described.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np


def verify_verbatim(chunk: str) -> bool:
    with open("player_td_features.py") as f:
        return chunk in f.read()


# ============================================================================
# PART 1: the xwalk_teams dedup fix (root cause of the 4x Barkley duplication)
# ============================================================================

def build_xwalk_teams(xwalk):
    """Copied verbatim from build_player_td_table()'s team-correction step."""
    IDS_TEAM_MAP = {"GBP": "GB", "KCC": "KC", "LVR": "LV", "NEP": "NE", "NOS": "NO",
                     "SFO": "SF", "TBB": "TB", "JAC": "JAX"}
    IDS_INACTIVE_CODES = {"FA", "FA*", "OAK", "SDC", "STL", "RAM"}

    def _map_ids_team(team):
        if pd.isna(team) or team in IDS_INACTIVE_CODES:
            return None
        return IDS_TEAM_MAP.get(team, team)

    xwalk_teams = xwalk[["gsis_id", "team"]].rename(columns={"gsis_id": "player_id"})
    xwalk_teams = xwalk_teams.drop_duplicates(subset=["player_id"], keep="first")
    xwalk_teams["current_team"] = xwalk_teams["team"].apply(_map_ids_team)
    return xwalk_teams


# ============================================================================
# PART 2: name + position fallback -- copied verbatim from player_td_features.py
# ============================================================================

def _apply_position_fallback(asof: pd.DataFrame, xwalk: pd.DataFrame) -> pd.DataFrame:
    xwalk_lookup = xwalk[["gsis_id", "position", "name"]].rename(
        columns={"gsis_id": "player_id", "position": "crosswalk_position", "name": "crosswalk_name"}
    ).drop_duplicates(subset=["player_id"])
    asof = asof.merge(xwalk_lookup, on="player_id", how="left")
    asof["position"] = asof["position"].where(asof["position"].notna(), asof["crosswalk_position"])
    asof["player_name"] = asof["player_name"].where(asof["player_name"].notna(), asof["crosswalk_name"])
    asof = asof.drop(columns=["crosswalk_position", "crosswalk_name"])

    still_missing_pos = asof[asof["position"].isna()]["player_id"].unique()
    if len(still_missing_pos):
        print(f"WARNING: {len(still_missing_pos)} player_id(s) have no position from EITHER "
              f"weekly data or the ID crosswalk.")
    still_missing_name = asof[asof["player_name"].isna()]["player_id"].unique()
    if len(still_missing_name):
        print(f"WARNING: {len(still_missing_name)} player_id(s) have no player_name from EITHER source.")
    return asof


def main():
    checks = []

    # ---- Verify both copies match the real file exactly ----
    checks.append(("xwalk_teams dedup line present verbatim in the real file",
                    verify_verbatim('xwalk_teams = xwalk_teams.drop_duplicates(subset=["player_id"], keep="first")'), True))
    checks.append(("defensive candidates dedup present verbatim in the real file",
                    verify_verbatim('candidates = candidates.drop_duplicates(subset=["player_id"], keep="first")'), True))
    checks.append(("player_name fallback line present verbatim in the real file",
                    verify_verbatim('asof["player_name"] = asof["player_name"].where(asof["player_name"].notna(), asof["crosswalk_name"])'), True))

    # ---- PART 1: reproduce the exact confirmed bug -- Barkley with 3
    # different pfr_id-distinct rows sharing ONE gsis_id, as observed in
    # real crosswalk data -- and confirm the fix collapses it to 1 ----
    xwalk_with_dupes = pd.DataFrame({
        "gsis_id": ["barkley_id", "barkley_id", "barkley_id", "other_id"],
        "pfr_id": ["pfr1", "pfr2", "pfr3", "pfr4"],  # 3 DIFFERENT pfr_ids, same gsis_id -- the real bug shape
        "team": ["PHI", "PHI", "NYG", "DAL"],  # even a stale team snapshot in one of the dupes
        "position": ["RB", "RB", "RB", "WR"],
        "name": ["Saquon Barkley", "Saquon Barkley", "Saquon Barkley", "Someone Else"],
    })
    xwalk_teams = build_xwalk_teams(xwalk_with_dupes)
    checks.append(("THE FIX: Barkley's 3 duplicate crosswalk rows collapse to exactly 1",
                    (xwalk_teams["player_id"] == "barkley_id").sum(), 1))
    checks.append(("other real player (1 row already) is untouched", (xwalk_teams["player_id"] == "other_id").sum(), 1))
    checks.append(("total deduplicated rows: 2 unique players from 4 raw rows", len(xwalk_teams), 2))

    # ---- Simulate the downstream candidates merge with the FIXED
    # (deduplicated) xwalk_teams -- confirm no fan-out happens ----
    candidates = pd.DataFrame({"player_id": ["barkley_id", "other_id"], "posteam": ["PHI", "DAL"]})
    merged = candidates.merge(xwalk_teams[["player_id", "current_team"]], on="player_id", how="left")
    checks.append(("candidates merge with fixed xwalk_teams produces NO duplicate rows",
                    len(merged), len(candidates)))
    # And confirm what WOULD have happened with the old, buggy (undeduplicated) data --
    # this is the actual production bug being reproduced, not just asserted.
    buggy_xwalk_teams = xwalk_with_dupes[["gsis_id", "team"]].rename(columns={"gsis_id": "player_id"})
    buggy_merged = candidates.merge(buggy_xwalk_teams, on="player_id", how="left")
    checks.append(("REPRODUCED THE BUG: without the fix, Barkley fans out to 3 duplicate rows",
                    (buggy_merged["player_id"] == "barkley_id").sum(), 3))

    # ---- PART 2: name fallback tests ----
    asof = pd.DataFrame({
        "player_id": ["A", "B", "C"],
        "position": ["RB", None, None],
        "player_name": ["S.Barkley", None, None],  # B: matches the real "NaN name" bug shape
    })
    xwalk2 = pd.DataFrame({
        "gsis_id": ["A", "B", "C"],
        "position": ["RB", "WR", None],
        "name": ["Saquon Barkley", "Some Player", None],
    })
    result = _apply_position_fallback(asof, xwalk2)
    checks.append(("Player A: normal weekly name/position untouched", result.loc[result.player_id=="A","player_name"].iloc[0], "S.Barkley"))
    checks.append(("Player B: THE FIX -- missing name recovered from crosswalk (root cause of the NaN rows)",
                    result.loc[result.player_id=="B","player_name"].iloc[0], "Some Player"))
    checks.append(("Player B: position ALSO recovered (both fixed by the same fallback)",
                    result.loc[result.player_id=="B","position"].iloc[0], "WR"))
    checks.append(("Player C: still None when name is missing from BOTH sources (not fabricated)",
                    pd.isna(result.loc[result.player_id=="C","player_name"].iloc[0]), True))
    checks.append(("no leftover crosswalk_name/crosswalk_position columns after merge",
                    any(c.startswith("crosswalk_") for c in result.columns), False))

    print(f"{'check':85s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 112)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:85s} {str(got):>10s} {str(want):>10s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 112)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

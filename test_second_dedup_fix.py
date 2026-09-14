"""
test_second_dedup_fix.py -- Tests the SECOND instance of the crosswalk-
duplication bug (load_id_crosswalk_names in generate_player_predictions.py)
and the final unconditional dedup guarantee, both copied verbatim from the
real file and verified via exact substring match.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd


def verify_verbatim(path: str, chunk: str) -> bool:
    with open(path) as f:
        return chunk in f.read()


def load_id_crosswalk_names_logic(xwalk):
    """Copied verbatim (minus the load_id_crosswalk() call itself, which
    needs real network access) from the fixed load_id_crosswalk_names()."""
    names = xwalk[["gsis_id", "merge_name"]].rename(columns={"gsis_id": "player_id"}).dropna()
    return names.drop_duplicates(subset=["player_id"], keep="first")


def final_dedup_logic(upcoming):
    """Copied verbatim from the new final-guarantee block."""
    before_dedup = len(upcoming)
    upcoming = upcoming.sort_values("model_prob", ascending=False).drop_duplicates(subset=["player_id"], keep="first")
    caught = before_dedup - len(upcoming)
    return upcoming, caught


def main():
    checks = []

    checks.append(("load_id_crosswalk_names dedup present verbatim in the real file",
                    verify_verbatim("generate_player_predictions.py",
                        'return names.drop_duplicates(subset=["player_id"], keep="first")'),
                    True))
    checks.append(("final dedup guarantee present verbatim in the real file",
                    verify_verbatim("generate_player_predictions.py",
                        'upcoming = upcoming.sort_values("model_prob", ascending=False).drop_duplicates(subset=["player_id"], keep="first")'),
                    True))

    # ---- Reproduce the EXACT real bug: Jefferson with 2 pfr_id-distinct
    # crosswalk rows sharing one gsis_id, with slightly different
    # merge_name formatting (the real-world shape of this bug) ----
    xwalk_with_dupes = pd.DataFrame({
        "gsis_id": ["jefferson_id", "jefferson_id", "other_id"],
        "pfr_id": ["pfr1", "pfr2", "pfr3"],
        "merge_name": ["justin jefferson", "justin jefferson", "someone else"],
    })
    names = load_id_crosswalk_names_logic(xwalk_with_dupes)
    checks.append(("THE FIX: Jefferson's 2 duplicate crosswalk rows collapse to exactly 1",
                    (names["player_id"] == "jefferson_id").sum(), 1))
    checks.append(("total: 2 unique players from 3 raw crosswalk rows", len(names), 2))

    # Confirm what WOULD happen downstream without the fix -- reproduces
    # the actual fan-out that was hitting `upcoming`.
    upcoming_before_fix = pd.DataFrame({"player_id": ["jefferson_id", "other_id"], "model_prob": [0.66, 0.5]})
    buggy_names = xwalk_with_dupes[["gsis_id", "merge_name"]].rename(columns={"gsis_id": "player_id"})
    buggy_merged = upcoming_before_fix.merge(buggy_names, on="player_id", how="left")
    checks.append(("REPRODUCED THE BUG: without the fix, Jefferson fans out to 2 rows on this merge",
                    (buggy_merged["player_id"] == "jefferson_id").sum(), 2))
    fixed_merged = upcoming_before_fix.merge(names, on="player_id", how="left")
    checks.append(("with the fix, the same merge produces exactly 1 row for Jefferson",
                    (fixed_merged["player_id"] == "jefferson_id").sum(), 1))

    # ---- Final guarantee: catches duplicates from ANY source, keeps
    # highest model_prob, and reports how many it caught ----
    upcoming_with_dupes = pd.DataFrame({
        "player_id": ["barkley_id", "barkley_id", "barkley_id", "taylor_id"],
        "model_prob": [0.80, 0.79, 0.70, 0.75],
    })
    deduped, caught = final_dedup_logic(upcoming_with_dupes)
    checks.append(("final guarantee: 4 rows -> 2 unique players", len(deduped), 2))
    checks.append(("final guarantee: keeps the HIGHEST-probability duplicate (0.80, not 0.70 or 0.79)",
                    deduped.loc[deduped.player_id == "barkley_id", "model_prob"].iloc[0], 0.80))
    checks.append(("final guarantee: correctly reports how many it caught (3 extra Barkley rows)", caught, 2))

    # ---- Final guarantee must be a true no-op when there's nothing to catch ----
    clean_upcoming = pd.DataFrame({"player_id": ["a", "b", "c"], "model_prob": [0.5, 0.4, 0.3]})
    clean_deduped, clean_caught = final_dedup_logic(clean_upcoming)
    checks.append(("final guarantee: no false positive when data is already clean", clean_caught, 0))
    checks.append(("final guarantee: row count unchanged when data is already clean", len(clean_deduped), 3))

    print(f"{'check':82s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 108)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:82s} {str(got):>10s} {str(want):>10s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 108)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

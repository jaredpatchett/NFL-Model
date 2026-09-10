"""
test_load_snaps.py — Validates the fixed load_snaps(): a season nflverse
hasn't published snap-count data for yet must be skipped with a warning,
not crash the whole call; every other season's data must still come back
correctly and get concatenated.
"""
import sys
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import nfl_data


def fake_import_snap_counts(seasons):
    """Simulates nfl_data_py: takes a list (nfl_data.py always calls this
    with a single-season list under the fix), returns real-shaped data for
    seasons 2021-2025, and raises an HTTPError-like exception for 2026 --
    simulating "not published yet"."""
    season = seasons[0]
    if season == 2026:
        raise Exception("HTTP Error 404: Not Found")
    return pd.DataFrame({
        "season": [season, season],
        "player": [f"player_{season}_a", f"player_{season}_b"],
        "offense_snaps": [50, 60],
    })


def main():
    checks = []
    tmpdir = tempfile.mkdtemp()
    try:
        with patch.object(nfl_data, "CACHE_DIR", Path(tmpdir)), \
             patch.object(nfl_data, "nfl", MagicMock(import_snap_counts=fake_import_snap_counts)):

            # ---- One bad season (2026) mixed with good ones: must not crash ----
            result = nfl_data.load_snaps([2023, 2024, 2025, 2026])
            checks.append(("does not crash despite 2026 failing", True, True))
            checks.append(("returns data for all 3 good seasons (2 rows each)", len(result), 6))
            checks.append(("2026 is correctly absent from the result", 2026 in result["season"].values, False))
            checks.append(("2023 data present", 2023 in result["season"].values, True))
            checks.append(("2025 data present", 2025 in result["season"].values, True))

            # ---- Per-season caching: a second call should hit cache for
            # good seasons (no re-fetch needed) and still skip 2026 again ----
            call_count = {"n": 0}
            def counting_fetch(seasons):
                call_count["n"] += 1
                return fake_import_snap_counts(seasons)
            with patch.object(nfl_data, "nfl", MagicMock(import_snap_counts=counting_fetch)):
                nfl_data.load_snaps([2023])  # already cached from above -- should NOT call fetch again
            checks.append(("already-cached good season is NOT re-fetched", call_count["n"], 0))

            # ---- All seasons bad: must return an empty DataFrame, not crash ----
            with patch.object(nfl_data, "nfl", MagicMock(
                    import_snap_counts=lambda s: (_ for _ in ()).throw(Exception("404")))):
                empty_result = nfl_data.load_snaps([2027])
            checks.append(("all-seasons-fail case returns empty DataFrame, not a crash", len(empty_result), 0))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"{'check':60s} {'got':>8s} {'want':>8s}  ok")
    print("-" * 90)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:60s} {str(got):>8s} {str(want):>8s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 90)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

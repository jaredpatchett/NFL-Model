"""
nfl_data.py — Data ingestion layer for the NFL touchdown model.

Pulls from nflverse (via nfl_data_py) and caches locally as parquet so we
don't re-hit the network on every run. This is Phase 1, Step 1 of the
blueprint: collect weekly player participation, routes, targets, carries,
red-zone/goal-line usage.

Requires network access (run locally or in GitHub Actions, not in a sandbox).

Install:
    pip install nfl_data_py pandas numpy pyarrow

Public API:
    load_pbp(seasons)          -> play-by-play DataFrame (cached)
    load_weekly(seasons)       -> weekly player stats (cached)
    load_snaps(seasons)        -> snap counts (cached)
    load_schedules(seasons)    -> game schedules + results (cached)
    load_rosters(seasons)      -> weekly rosters (cached)
"""

from __future__ import annotations
import os
from pathlib import Path

import pandas as pd

try:
    import nfl_data_py as nfl
except ImportError:  # pragma: no cover - only hit in sandbox
    nfl = None

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _require_nfl():
    if nfl is None:
        raise RuntimeError(
            "nfl_data_py is not installed / no network. "
            "Run this in an environment with internet access: "
            "pip install nfl_data_py"
        )


def _cache_path(name: str, seasons: list[int]) -> Path:
    tag = f"{min(seasons)}_{max(seasons)}"
    return CACHE_DIR / f"{name}_{tag}.parquet"


def _load_cached_flat(name: str, fetch_fn) -> pd.DataFrame:
    """Like _load_cached but for data that isn't keyed by season (e.g. the ID crosswalk)."""
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch_fn()
    df.to_parquet(path, index=False)
    return df


def _load_cached(name: str, seasons: list[int], fetch_fn) -> pd.DataFrame:
    """Return cached parquet if present, else fetch, cache, and return."""
    path = _cache_path(name, seasons)
    if path.exists():
        return pd.read_parquet(path)
    _require_nfl()
    df = fetch_fn(seasons)
    df.to_parquet(path, index=False)
    return df


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    """Play-by-play. This is the source of truth for red-zone / goal-line usage."""
    return _load_cached("pbp", seasons, lambda s: nfl.import_pbp_data(s, downcast=True))


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    """Weekly player box-score stats (targets, carries, receiving/rushing yards, TDs)."""
    return _load_cached("weekly", seasons, lambda s: nfl.import_weekly_data(s))


def load_snaps(seasons: list[int]) -> pd.DataFrame:
    """Snap counts by player-game. Gives us snap share."""
    return _load_cached("snaps", seasons, lambda s: nfl.import_snap_counts(s))


GAMES_CSV_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def _fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    # NOTE: nfl_data_py's import_schedules() hits http://www.habitatring.com/games.csv,
    # which returns 403 from sandboxed / restricted-egress environments (and is a single
    # point of failure generally). The same data is mirrored on GitHub by nflverse, which
    # is reachable anywhere GitHub is — use that instead. Columns include spread_line,
    # total_line, moneylines, rest days, roof/surface/weather — this is where the Layer 2
    # team-implied-total signal comes from.
    df = pd.read_csv(GAMES_CSV_URL)
    return df[df["season"].isin(seasons)].reset_index(drop=True)


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    """Schedules + results (spreads, totals, scores, rest days, weather)."""
    return _load_cached("sched", seasons, _fetch_schedules)


def load_rosters(seasons: list[int]) -> pd.DataFrame:
    """Weekly rosters — position, team, status."""
    return _load_cached("rosters", seasons, lambda s: nfl.import_weekly_rosters(s))


def load_id_crosswalk() -> pd.DataFrame:
    """
    Cross-reference table mapping gsis_id (used in pbp/weekly) to pfr_id (used in
    snap counts) and other ID systems. Needed because nflverse snap counts are keyed
    by Pro-Football-Reference ID, not GSIS ID — without this crosswalk, any merge of
    snap counts onto play-by-play-derived features silently matches nothing.
    """
    def _fetch(_seasons=None):
        _require_nfl()
        df = nfl.import_ids()
        df = df[["gsis_id", "pfr_id", "name", "position"]].dropna(
            subset=["gsis_id", "pfr_id"]
        )
        return df.drop_duplicates(subset=["pfr_id"]).reset_index(drop=True)

    return _load_cached_flat("id_crosswalk", _fetch)


if __name__ == "__main__":
    # Smoke test — run this locally to verify network + caching works.
    import sys

    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    print(f"Fetching {yr} play-by-play (first run downloads, then caches)...")
    pbp = load_pbp([yr])
    print(f"  pbp rows: {len(pbp):,}  cols: {len(pbp.columns)}")
    snaps = load_snaps([yr])
    print(f"  snap rows: {len(snaps):,}")
    weekly = load_weekly([yr])
    print(f"  weekly rows: {len(weekly):,}")
    sched = load_schedules([yr])
    print(f"  schedule rows: {len(sched):,} (has spread_line/total_line: "
          f"{'spread_line' in sched.columns and 'total_line' in sched.columns})")
    crosswalk = load_id_crosswalk()
    print(f"  id crosswalk rows: {len(crosswalk):,}")
    print("OK — cached to data/cache/")

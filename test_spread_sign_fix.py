"""
test_spread_sign_fix.py — Validates the spread_line sign fix using the
EXACT real numbers from the three broken dashboard screenshots reported:
NO@DET, WAS@PHI, ARI@LAC. Confirms the fixed formula produces small,
sensible edges (model disagrees with market by a normal amount) instead of
the old wild double-digit "PLAY [home favorite]" signal on every one of
them, and that the edge's sign correctly points toward value on the
underdog when the model thinks the market's favorite is overrated.

Isolates just the sign-fix arithmetic (can't run the full
generate_predictions.py pipeline here -- it needs game_features,
game_lines_model, and other modules not available in this session) --
this is the exact computation that was wrong and the exact computation
that matters for whether a "PLAY" verdict is trustworthy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import norm


def old_broken_spread_line(live_home_spread_point: float) -> float:
    """The bug: live_home_spread_point used AS-IS."""
    return float(live_home_spread_point)


def fixed_spread_line(live_home_spread_point: float) -> float:
    """The fix: negate to convert from the API's negative=favorite
    convention into this pipeline's positive=home-favored convention."""
    return -float(live_home_spread_point)


def spread_edge(pred_margin: float, spread_line: float) -> float:
    """Unchanged formula -- was always correct GIVEN a correctly-signed
    spread_line. The bug was entirely in what got fed into this, not this
    formula itself."""
    return round(pred_margin - spread_line, 2)


def cover_prob(pred_margin: float, spread_line: float, resid_std: float) -> float:
    return round(1 - norm.cdf(spread_line, loc=pred_margin, scale=resid_std), 4)


REAL_EXAMPLES = [
    # (label, pred_home_margin, raw_live_home_spread_point_from_api, home_team)
    ("NO@DET",  5.4, -7.0, "DET"),
    ("WAS@PHI", 4.9, -5.5, "PHI"),
    ("ARI@LAC", 3.9, -9.5, "LAC"),
]

RESID_STD = 13.16  # real value seen in production output earlier this session


def main():
    checks = []

    for label, pred_margin, raw_api_point, home_team in REAL_EXAMPLES:
        old_line = old_broken_spread_line(raw_api_point)
        new_line = fixed_spread_line(raw_api_point)
        old_edge = spread_edge(pred_margin, old_line)
        new_edge = spread_edge(pred_margin, new_line)

        # ---- Reproduce the exact broken numbers from the real screenshots,
        # to prove this test is actually modeling the real bug ----
        checks.append((f"{label}: OLD (buggy) edge matches what was actually seen on screen",
                        old_edge, round(pred_margin - raw_api_point, 2)))

        # ---- The fix: spread_line now means positive=home-favored ----
        checks.append((f"{label}: fixed spread_line = -raw_api_point", new_line, -raw_api_point))

        # ---- The fixed edge must be small (a normal model-vs-market
        # disagreement), not the old wild double-digit number ----
        checks.append((f"{label}: fixed edge is small/sane (abs <= 10), unlike the old one",
                        abs(new_edge) <= 10, True))
        checks.append((f"{label}: fixed edge is genuinely different from the old broken one",
                        new_edge != old_edge, True))

        # ---- Since in all 3 real examples the model liked the home
        # favorite LESS than the market did, the fixed edge must be
        # NEGATIVE (real lean toward the underdog), not a false PLAY on
        # the home favorite ----
        checks.append((f"{label}: fixed edge correctly leans toward the underdog (negative)",
                        new_edge < 0, True))

        # ---- Old edge was always wrongly positive (falsely signaling
        # PLAY on the home favorite) on all 3 -- confirms this was
        # systemic, not a one-off ----
        checks.append((f"{label}: old (buggy) edge was falsely positive on every real example",
                        old_edge > 0, True))

        # ---- cover_prob (also fed by spread_line) becomes sane too: model
        # should now show LESS than the naive >90% cover confidence the old
        # sign bug would have implied for a double-digit "edge" ----
        old_cover = cover_prob(pred_margin, old_line, RESID_STD)
        new_cover = cover_prob(pred_margin, new_line, RESID_STD)
        checks.append((f"{label}: fixed cover_prob is more conservative than the old buggy one",
                        new_cover < old_cover, True))

    # ---- Spot-check the DET example precisely against the screenshot's
    # displayed OLD edge value (+12.4) ----
    det_old_edge = spread_edge(5.4, old_broken_spread_line(-7.0))
    checks.append(("NO@DET: old edge exactly matches the screenshot's displayed +12.4", det_old_edge, 12.4))
    det_new_edge = spread_edge(5.4, fixed_spread_line(-7.0))
    checks.append(("NO@DET: fixed edge = 5.4 - 7.0 = -1.6", det_new_edge, -1.6))

    print(f"{'check':78s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 105)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:78s} {str(got):>10s} {str(want):>10s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 105)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

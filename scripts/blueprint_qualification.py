"""
blueprint_qualification.py — Implements the eligibility screen and A/B/C/D
tier framework from NFL_Touchdown_Betting_Model_Blueprint.pdf (uploaded by
the user partway through this project), Sections 2, 6 (Layer 1), and 7.

WHAT'S IMPLEMENTED (directly from the PDF's own stated thresholds):
  - Snap share >= 70% (Section 2's hard minimum rule)
  - TD projection >= 0.30 (Section 2; mapped to our calibrated model
    probability -- the PDF uses "projection" and "probability"
    interchangeably throughout, e.g. Section 7's "model probability at
    least 30%")
  - Projection exceeds market implied probability at all (Section 2's hard
    disqualifier: "Projection does not exceed the sportsbook implied
    probability"), with 4%/6% preferred edge thresholds feeding the tier
    (Section 7)
  - Meaningful red-zone role (Section 2's "weak or disappearing red-zone
    role" disqualifier) -- approximated as red-zone target share OR
    inside-5 carry share > 5%. The PDF names this disqualifier but doesn't
    give an exact number, so this threshold is a reasonable interpretation,
    not a value taken directly from the document.
  - Recent TD production unsupported by opportunity (Section 2's hard
    disqualifier + Section 11's "chasing recent touchdowns" blind spot) --
    flagged when trailing actual TDs meaningfully exceeds trailing expected
    TDs. Same caveat: PDF names the concept, this module supplies a
    concrete ratio.
  - Tier thresholds from Section 7: A = edge>=6%, EV>=12%; B = edge 4-6%,
    EV 7-12%; C = passes all hard filters but below B's edge/EV; D = fails
    any hard filter.

WHAT'S DELIBERATELY NOT IMPLEMENTED (needs more specification first, not
a build shortcut):
  - Matchup rating >= 1.10 (Section 2) -- the PDF names this threshold but
    never defines the formula. Nothing in the current pipeline computes a
    single opponent-adjusted "matchup rating" number yet.
  - HEAT windows, "2 of 3 at 60+" (Section 2) -- also named without a
    formula in the PDF. The user's own screenshot has HEAT 3/6/9 columns
    that look like a usage-consistency metric, but the exact calculation
    hasn't been confirmed, so nothing here approximates it.
  - Preferred odds range +120 to +300 (Section 1) -- NOT enforced as a
    HARD filter. Reasoning: the EV calculation already penalizes short or
    heavily-juiced prices on its own terms (a -150 favorite with real edge
    can still clear the EV bar; hard-excluding it on price alone would
    throw away a play the PDF's own EV math would accept). Tracked as an
    informational reason code instead of a disqualifier.

Public API:
    qualify(row, model_prob, market) -> dict with qualifies, tier, reason_codes
        row: a player row with the trailing feature columns (dict-like,
             e.g. a pandas Series)
        model_prob: the calibrated anytime-TD probability (float)
        market: the market dict from generate_player_predictions.py
                (edge, ev, anytime_td_price), or None if no live price
"""

from __future__ import annotations
import math

SNAP_SHARE_MIN = 0.70
TD_PROJECTION_MIN = 0.30
RED_ZONE_ROLE_MIN = 0.05
EDGE_TIER_A = 0.06
EDGE_TIER_B = 0.04
EV_TIER_A = 0.12
EV_TIER_B = 0.07
REGRESSION_RATIO = 1.5
REGRESSION_MIN_ACTUAL = 2.0
PREFERRED_ODDS_LOW = 120
PREFERRED_ODDS_HIGH = 300
MATCHUP_RATING_PREFERRED = 1.10


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def qualify(row, model_prob: float, market: dict | None) -> dict:
    reason_codes: list[str] = []
    hard_fail = False

    # ---- Snap share ----
    snap_share = row.get("asof_roll4_snap_share")
    if _is_missing(snap_share) or snap_share < SNAP_SHARE_MIN:
        hard_fail = True
        reason_codes.append(f"Snap share below {SNAP_SHARE_MIN:.0%} threshold")

    # ---- TD projection ----
    if model_prob < TD_PROJECTION_MIN:
        hard_fail = True
        reason_codes.append(f"TD projection below {TD_PROJECTION_MIN:.0%} threshold")

    # ---- Red-zone role ----
    rz_share = row.get("asof_roll4_rz_target_share")
    i5_share = row.get("asof_roll4_inside5_carry_share")
    rz_share = 0.0 if _is_missing(rz_share) else rz_share
    i5_share = 0.0 if _is_missing(i5_share) else i5_share
    if max(rz_share, i5_share) < RED_ZONE_ROLE_MIN:
        hard_fail = True
        reason_codes.append("No meaningful red-zone role")

    # ---- Regression check: recent TDs outpacing opportunity ----
    actual_tds = row.get("asof_roll4_actual_tds")
    expected_tds = row.get("asof_roll4_expected_tds")
    if not _is_missing(actual_tds) and not _is_missing(expected_tds):
        if (actual_tds >= REGRESSION_MIN_ACTUAL and expected_tds > 0
                and actual_tds > expected_tds * REGRESSION_RATIO):
            hard_fail = True
            reason_codes.append("Recent TDs outpacing opportunity (regression risk)")

    # ---- Matchup rating: INFORMATIONAL ONLY, not a hard filter yet. ----
    # New this session (defense_features.py) -- a real, position-specific
    # (rush TDs allowed for RBs, pass TDs allowed for WR/TE/QB) rate versus
    # league average, not a guess. Not gating plays on it yet because it
    # hasn't been checked against a real season of outcomes -- flagged as a
    # note so it's visible without silently failing plays on an unvalidated
    # number. Revisit once there's backtest data to confirm 1.10 (or some
    # other value) is actually the right bar.
    matchup_rating = row.get("matchup_rating")
    if not _is_missing(matchup_rating) and matchup_rating < MATCHUP_RATING_PREFERRED:
        reason_codes.append(
            f"Matchup rating {matchup_rating:.2f} below {MATCHUP_RATING_PREFERRED:.2f} "
            f"preferred (informational only, not yet a disqualifier)"
        )

    # ---- Market-dependent checks: edge, EV, price range, tier ----
    if market is None:
        reason_codes.append("No live price -- edge/EV not evaluated")
        return {"qualifies": False, "tier": "D" if hard_fail else "U", "reason_codes": reason_codes}

    edge = market["edge"]
    ev = market["ev"]
    price = market["anytime_td_price"]

    if edge <= 0:
        hard_fail = True
        reason_codes.append("Model projection does not exceed market implied probability")

    if not (PREFERRED_ODDS_LOW <= price <= PREFERRED_ODDS_HIGH):
        reason_codes.append(f"Outside preferred price range (+{PREFERRED_ODDS_LOW}/+{PREFERRED_ODDS_HIGH})")

    if hard_fail:
        return {"qualifies": False, "tier": "D", "reason_codes": reason_codes}

    if edge >= EDGE_TIER_A and ev >= EV_TIER_A:
        tier = "A"
    elif edge >= EDGE_TIER_B and ev >= EV_TIER_B:
        tier = "B"
    else:
        tier = "C"

    return {"qualifies": tier in ("A", "B"), "tier": tier, "reason_codes": reason_codes}

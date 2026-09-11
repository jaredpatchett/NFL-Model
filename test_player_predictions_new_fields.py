"""
test_player_predictions_new_fields.py — Validates the two fields just added
to generate_player_predictions.py's output: model.fair_moneyline and
team_implied_total.

SCOPE NOTE: the full main() function has dependencies (player_td_features,
player_td_model, blueprint_qualification, odds_api, manual_odds, nfl_data,
features) not available in this session, so this can't run end to end. This
tests the actual new code precisely -- the same construction expressions
added to the real file, copied exactly -- against synthetic rows, which is
what's actually new and what could actually be wrong. It does NOT
re-validate prob_to_fair_moneyline's internals (that function's real source
isn't available here either); a standalone reference odds-conversion
function stands in for it, clearly separated below, just to confirm the
WIRING (right value, right place, right null-handling) is correct.
"""
import pandas as pd


def reference_prob_to_fair_moneyline(p: float) -> float:
    """Standard fair-odds conversion, used ONLY as a stand-in for testing
    integration -- NOT a claim that this exactly matches game_lines_model.py's
    real prob_to_fair_moneyline (that source isn't available in this
    session). Standard formula: favorite (p>=0.5) -> negative odds,
    underdog (p<0.5) -> positive odds."""
    if p >= 0.5:
        return -100 * p / (1 - p)
    return 100 * (1 - p) / p


def build_team_implied_total(r: dict) -> float | None:
    """Copied EXACTLY from the real edit in generate_player_predictions.py --
    this is the actual new code being tested, not a re-description of it."""
    return round(float(r["implied_team_total"]), 1) if pd.notna(r["implied_team_total"]) else None


def build_model_dict(r: dict) -> dict:
    """Copied EXACTLY from the real edit in generate_player_predictions.py."""
    return {
        "anytime_td_prob": round(float(r["model_prob"]), 4),
        "fair_moneyline": round(float(r["model_fair_moneyline"]), 1) if pd.notna(r["model_fair_moneyline"]) else None,
        "tier": r["tier"],
    }


def main():
    checks = []

    # ---- team_implied_total: normal case ----
    row1 = {"implied_team_total": 27.456}
    checks.append(("team_implied_total rounds to 1 decimal", build_team_implied_total(row1), 27.5))

    # ---- team_implied_total: NaN case must give None, not crash or emit "nan" ----
    row2 = {"implied_team_total": float("nan")}
    checks.append(("team_implied_total is None when NaN (not a crash, not the string 'nan')",
                    build_team_implied_total(row2), None))

    # ---- model.fair_moneyline: favorite (high prob) gives negative odds ----
    prob_fav = 0.64
    fair_odds_fav = reference_prob_to_fair_moneyline(prob_fav)
    row3 = {"model_prob": prob_fav, "model_fair_moneyline": fair_odds_fav, "tier": "A"}
    result3 = build_model_dict(row3)
    checks.append(("fair_moneyline is negative for a likely (64%) outcome", result3["fair_moneyline"] < 0, True))
    checks.append(("anytime_td_prob still present and correctly rounded", result3["anytime_td_prob"], 0.64))
    checks.append(("tier passed through unchanged", result3["tier"], "A"))

    # ---- model.fair_moneyline: underdog (low prob) gives positive odds ----
    prob_dog = 0.15
    fair_odds_dog = reference_prob_to_fair_moneyline(prob_dog)
    row4 = {"model_prob": prob_dog, "model_fair_moneyline": fair_odds_dog, "tier": "C"}
    result4 = build_model_dict(row4)
    checks.append(("fair_moneyline is positive for an unlikely (15%) outcome", result4["fair_moneyline"] > 0, True))

    # ---- model.fair_moneyline: NaN case (e.g. a genuinely edge-case
    # probability of exactly 0 or 1) must give None, not crash ----
    row5 = {"model_prob": 0.30, "model_fair_moneyline": float("nan"), "tier": "B"}
    result5 = build_model_dict(row5)
    checks.append(("fair_moneyline is None when NaN, doesn't crash the whole player entry",
                    result5["fair_moneyline"], None))
    checks.append(("rest of the model dict still builds fine even when fair_moneyline is NaN",
                    result5["anytime_td_prob"], 0.3))

    # ---- Sanity: does the reference conversion behave the way MODEL ODDS
    # should look on the dashboard for a case matching the real screenshot
    # (Gibbs, model 64%)? Not a claim this matches the real function's exact
    # output, just a plausibility check on the shape of the number. ----
    checks.append(("64% probability converts to roughly -178 odds (sanity range, not exact match)",
                    -200 < reference_prob_to_fair_moneyline(0.64) < -160, True))

    print(f"{'check':75s} {'got':>10s} {'want':>10s}  ok")
    print("-" * 105)
    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print(f"{name:75s} {str(got):>10s} {str(want):>10s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 105)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

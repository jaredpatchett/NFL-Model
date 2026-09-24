"""
test_blueprint_qualification.py — Validates blueprint_qualification.py
against synthetic fixtures covering every rule from the PDF: each hard
disqualifier individually, each tier boundary, the no-market case, and the
non-disqualifying "outside preferred odds range" flag.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blueprint_qualification import qualify, flag_top_unpriced_pick


def base_row(**overrides):
    row = {
        "asof_roll4_snap_share": 0.80,
        "asof_roll4_rz_target_share": 0.15,
        "asof_roll4_inside5_carry_share": 0.0,
        "asof_roll4_actual_tds": 1.0,
        "asof_roll4_expected_tds": 1.2,
    }
    row.update(overrides)
    return row


def base_market(edge=0.07, ev=0.13, price=180):
    return {"edge": edge, "ev": ev, "anytime_td_price": price}


def main():
    checks = []

    # ---- Hard disqualifiers, one at a time ----
    r = qualify(base_row(asof_roll4_snap_share=0.60), model_prob=0.35, market=base_market())
    checks.append(("low snap share -> D, not qualifying", (r["tier"], r["qualifies"]), ("D", False)))
    checks.append(("low snap share -> reason code present",
                    any("Snap share" in c for c in r["reason_codes"]), True))

    r = qualify(base_row(), model_prob=0.25, market=base_market())
    checks.append(("low TD projection -> D", r["tier"], "D"))

    r = qualify(base_row(asof_roll4_rz_target_share=0.0, asof_roll4_inside5_carry_share=0.0),
                model_prob=0.35, market=base_market())
    checks.append(("no red-zone role -> D", r["tier"], "D"))

    r = qualify(base_row(asof_roll4_actual_tds=4.0, asof_roll4_expected_tds=1.0),
                model_prob=0.35, market=base_market())
    checks.append(("recent TDs way outpacing opportunity -> D (regression risk)", r["tier"], "D"))

    r = qualify(base_row(asof_roll4_actual_tds=1.0, asof_roll4_expected_tds=1.0),
                model_prob=0.35, market=base_market())
    checks.append(("actual==expected TDs does NOT trigger regression flag", r["tier"] != "D", True))

    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=-0.02))
    checks.append(("negative edge (projection below market) -> D", r["tier"], "D"))

    # ---- Tier boundaries (all hard filters pass) ----
    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=0.06, ev=0.12))
    checks.append(("edge=6%,EV=12% -> exactly Tier A boundary", r["tier"], "A"))
    checks.append(("Tier A qualifies", r["qualifies"], True))

    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=0.059, ev=0.12))
    checks.append(("edge just under 6% -> not Tier A", r["tier"] != "A", True))

    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=0.04, ev=0.07))
    checks.append(("edge=4%,EV=7% -> exactly Tier B boundary", r["tier"], "B"))
    checks.append(("Tier B qualifies", r["qualifies"], True))

    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=0.045, ev=0.05))
    checks.append(("edge ok but EV below B's 7% -> Tier C", r["tier"], "C"))
    checks.append(("Tier C does not qualify", r["qualifies"], False))

    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=0.01, ev=0.02))
    checks.append(("small positive edge, low EV -> Tier C (not D -- no hard filter failed)", r["tier"], "C"))

    # ---- No market: unrated, not auto-D unless a hard filter also failed ----
    r = qualify(base_row(), model_prob=0.35, market=None)
    checks.append(("no market, all else fine -> Tier U (unrated, not D)", r["tier"], "U"))
    checks.append(("no market -> never qualifies", r["qualifies"], False))

    r = qualify(base_row(asof_roll4_snap_share=0.5), model_prob=0.35, market=None)
    checks.append(("no market BUT also fails snap share -> still D", r["tier"], "D"))

    # ---- Preferred odds range: informational only, not a disqualifier ----
    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=0.07, ev=0.13, price=-150))
    checks.append(("short price (-150) with real edge -> still Tier A, not disqualified", r["tier"], "A"))
    checks.append(("short price flagged in reason codes",
                    any("preferred price range" in c for c in r["reason_codes"]), True))

    r = qualify(base_row(), model_prob=0.35, market=base_market(edge=0.07, ev=0.13, price=200))
    checks.append(("price within +120/+300 -> no price-range reason code",
                    any("preferred price range" in c for c in r["reason_codes"]), False))

    # ---- Matchup rating: informational only, never disqualifies ----
    r = qualify(base_row(matchup_rating=0.75), model_prob=0.35, market=base_market())
    checks.append(("soft matchup rating flagged but still qualifies", r["qualifies"], True))
    checks.append(("soft matchup rating -> Tier A unaffected (edge/EV still clear it)", r["tier"], "A"))
    checks.append(("soft matchup rating reason code present",
                    any("Matchup rating" in c for c in r["reason_codes"]), True))

    r = qualify(base_row(matchup_rating=1.30), model_prob=0.35, market=base_market())
    checks.append(("strong matchup rating -> no matchup reason code",
                    any("Matchup rating" in c for c in r["reason_codes"]), False))

    # ---- flag_top_unpriced_pick: real scenario from a live Week 2 case ----
    # (K.Walker/J.Cook/D.Henry all had high model probability but failed the
    # real snap-share hard filter; only J.Gibbs cleared every hard filter
    # with no live price -- should be the one promoted to tier "T".)
    def player(name, tier, prob):
        return {"player_name": name, "qualification": {"tier": tier, "reason_codes": []},
                "model": {"anytime_td_prob": prob}}

    players = [
        player("K.Walker", "D", 0.8141),   # failed snap share -> D, never eligible for promotion
        player("J.Gibbs", "U", 0.7611),    # cleared every hard filter, just unpriced
        player("J.Cook", "D", 0.7554),     # failed snap share -> D
        player("D.Henry", "D", 0.7529),    # failed snap share -> D
        player("S.Barkley", "C", 0.6122),  # has a real price already (Tier C/DEGEN) -- not a "U" candidate
    ]
    flag_top_unpriced_pick(players)
    by_name = {p["player_name"]: p for p in players}
    checks.append(("only Gibbs (the sole U-tier candidate) gets promoted", by_name["J.Gibbs"]["qualification"]["tier"], "T"))
    checks.append(("Walker (failed hard filter, tier D) stays D, not promoted", by_name["K.Walker"]["qualification"]["tier"], "D"))
    checks.append(("Barkley (already priced, tier C) stays C, not touched", by_name["S.Barkley"]["qualification"]["tier"], "C"))
    checks.append(("Gibbs gets an explanatory reason code",
                    any("Top Pick" in c for c in by_name["J.Gibbs"]["qualification"]["reason_codes"]), True))

    # Among MULTIPLE unpriced-but-qualified candidates, only the single
    # highest model-probability one should be promoted.
    multi = [
        player("A", "U", 0.55),
        player("B", "U", 0.70),  # highest -- should be the only one promoted
        player("C", "U", 0.60),
    ]
    flag_top_unpriced_pick(multi)
    tiers = {p["player_name"]: p["qualification"]["tier"] for p in multi}
    checks.append(("only the single highest-probability U candidate promoted", tiers, {"A": "U", "B": "T", "C": "U"}))

    # No U-tier candidates at all -> no-op, doesn't crash
    none_eligible = [player("X", "D", 0.90), player("Y", "C", 0.85)]
    flag_top_unpriced_pick(none_eligible)
    checks.append(("no U-tier candidates -> no-op", [p["qualification"]["tier"] for p in none_eligible], ["D", "C"]))

    # Empty list -> no-op, doesn't crash
    flag_top_unpriced_pick([])
    checks.append(("empty player list -> no crash", True, True))

    print(f"{'check':70s} {'got':>15s} {'want':>15s}  ok")
    print("-" * 108)
    all_ok = True
    for name, got, want in checks:
        ok = (got == want)
        all_ok &= ok
        print(f"{name:70s} {str(got):>15s} {str(want):>15s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 108)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

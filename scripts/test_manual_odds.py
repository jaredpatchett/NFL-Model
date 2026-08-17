"""
test_manual_odds.py — Validates manual_odds.py: CSV parsing, price parsing
(with/without +, malformed rows), name normalization matching both full
names and the dashboard's abbreviated form, and graceful handling of a
missing or malformed file.
"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manual_odds import load_manual_odds, _parse_price


def approx_eq(a, b):
    return a == b


def main():
    checks = []

    # ---- Price parsing ----
    checks.append(("parse '-145'", _parse_price("-145"), -145))
    checks.append(("parse '+180'", _parse_price("+180"), 180))
    checks.append(("parse '180' (bare, no sign)", _parse_price("180"), 180))
    checks.append(("parse empty string -> None", _parse_price(""), None))
    checks.append(("parse 'abc' (garbage) -> None", _parse_price("abc"), None))

    # ---- Missing file ----
    checks.append(("missing file returns None", load_manual_odds("/tmp/does_not_exist.csv"), None))

    # ---- Well-formed file, mixed name formats ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("player_name,price\n")
        f.write("Saquon Barkley,-145\n")
        f.write("J.Taylor,+180\n")  # abbreviated dashboard-style name
        f.write("Ja'Marr Chase,-110\n")  # apostrophe, tests normalize_name reuse
        path = f.name

    df = load_manual_odds(path)
    checks.append(("well-formed file returns 3 rows", len(df), 3))
    checks.append(("full name normalized correctly",
                    "saquon barkley" in df["player_name_norm"].values, True))
    checks.append(("abbreviated name normalized (dot stripped)",
                    "jtaylor" in df["player_name_norm"].values, True))
    checks.append(("apostrophe stripped from Ja'Marr Chase",
                    "jamarr chase" in df["player_name_norm"].values, True))
    barkley_row = df[df["player_name_norm"] == "saquon barkley"].iloc[0]
    checks.append(("Barkley price parsed correctly", int(barkley_row["manual_price"]), -145))
    os.unlink(path)

    # ---- Malformed rows mixed with good ones -- bad rows dropped, not crashed ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("player_name,price\n")
        f.write("Good Player,150\n")
        f.write("Bad Player,not_a_number\n")
        f.write(",200\n")  # missing name
        path = f.name

    df2 = load_manual_odds(path)
    checks.append(("malformed rows dropped, only 1 good row survives", len(df2), 1))
    checks.append(("surviving row is the good one", df2.iloc[0]["player_name_norm"], "good player"))
    os.unlink(path)

    # ---- Duplicate player, last one wins (most recently edited) ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("player_name,price\n")
        f.write("Same Player,100\n")
        f.write("Same Player,-200\n")
        path = f.name

    df3 = load_manual_odds(path)
    checks.append(("duplicate name -> 1 row (last wins)", len(df3), 1))
    checks.append(("last-listed price wins", int(df3.iloc[0]["manual_price"]), -200))
    os.unlink(path)

    # ---- Missing required columns ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,odds\n")  # wrong column names
        f.write("Someone,100\n")
        path = f.name
    checks.append(("wrong column names -> None", load_manual_odds(path), None))
    os.unlink(path)

    print(f"{'check':55s} {'got':>20s} {'want':>20s}  ok")
    print("-" * 100)
    all_ok = True
    for name, got, want in checks:
        ok = approx_eq(got, want)
        all_ok &= ok
        print(f"{name:55s} {str(got):>20s} {str(want):>20s}  {'PASS' if ok else 'FAIL'}")
    print("-" * 100)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

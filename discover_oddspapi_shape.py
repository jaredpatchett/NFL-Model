#!/usr/bin/env python3
"""
Rod's Golf Model -- OddsPapi discovery script
==============================================
Path C, step 1: OddsPapi (oddspapi.io) claims coverage of PrizePicks odds
alongside 300+ other bookmakers, including player props. This script does
NOT assume anything about golf's sportId, market IDs, or PrizePicks' exact
bookmaker key in their system -- every one of those needs to be confirmed
live before any of it gets wired into rods_pipeline.py. Same discipline as
discover_round_shape.py: guessing field shapes has broken this app before --
the matchup dedup bug, the EV%/Edge% swap, the wind double-count -- all three
came from trusting an assumed shape instead of a confirmed one.

What this script does, in order:
  1. GET /v4/sports              -> find golf's sportId
  2. GET /v4/bookmakers          -> confirm "prizepicks" is a real bookmaker
                                     key in this system (not guessed)
  3. GET /v4/tournaments         -> find this week's PGA Tour event's
                                     tournamentId
  4. GET /v4/markets             -> see every real golf market OddsPapi has
                                     -- this is where "Total Birdies",
                                     "Round Score O/U", "2-Ball Matchup" etc
                                     either show up under real names/IDs, or
                                     don't exist at all
  5. GET /v4/odds-by-tournaments -> pull real PrizePicks-tagged odds for that
                                     tournament and print the raw shape

Run this (locally or via the matching discover-oddspapi.yml workflow), paste
the full output back, and the real pipeline/engine integration gets built
against what's actually there -- not against a guess.
"""
import os
import sys
import json
import urllib.request
import urllib.parse
import urllib.error

BASE = "https://api.oddspapi.io"


def get_key():
    key = os.environ.get("ODDSPAPI_KEY")
    if not key:
        sys.exit('ERROR: set your key first ->  export ODDSPAPI_KEY="your_oddspapi_key"')
    return key


def fetch(endpoint, key, **params):
    params["apiKey"] = key
    url = f"{BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        raise SystemExit(f"HTTP {e.code} on {endpoint}: {body[:500]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Network error on {endpoint}: {e.reason}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"Non-JSON reply on {endpoint} (first 300 chars): {raw[:300]}")


def as_list(payload):
    """OddsPapi's docs show bare JSON arrays for these endpoints, but handle a
    dict-wrapped response too in case a given endpoint nests it (e.g. under
    'data') -- don't want a shape assumption breaking this probe script
    itself."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        for v in payload.values():
            if isinstance(v, list):
                return v
    return []


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    key = get_key()

    # ---- Step 1: find golf's sportId ----
    section("STEP 1 -- GET /v4/sports (looking for golf)")
    sports = as_list(fetch("/v4/sports", key))
    print(f"Total sports returned: {len(sports)}")
    golf_matches = [s for s in sports if "golf" in json.dumps(s).lower() or "pga" in json.dumps(s).lower()]
    print("Matches containing 'golf' or 'pga':")
    print(json.dumps(golf_matches, indent=1))
    if not golf_matches:
        print("\nNo obvious golf match -- printing ALL sports so it can be found manually:")
        print(json.dumps(sports, indent=1))
        return
    golf_sport_id = golf_matches[0].get("sportId")
    print(f"\nUsing sportId={golf_sport_id} for the rest of this script.")

    # ---- Step 2: confirm "prizepicks" is a real bookmaker key ----
    section("STEP 2 -- GET /v4/bookmakers (confirming PrizePicks' key name)")
    bookmakers = as_list(fetch("/v4/bookmakers", key))
    print(f"Total bookmakers returned: {len(bookmakers)}")
    pp_matches = [b for b in bookmakers if "priz" in json.dumps(b).lower()]
    print("Matches containing 'priz':")
    print(json.dumps(pp_matches, indent=1))
    if not pp_matches:
        print("\nNo PrizePicks match found by name -- this matters, it may mean")
        print("this plan/key doesn't include PrizePicks, or it's keyed differently.")

    # ---- Step 3: find this week's golf tournament(s) ----
    section(f"STEP 3 -- GET /v4/tournaments?sportId={golf_sport_id}")
    tournaments = as_list(fetch("/v4/tournaments", key, sportId=golf_sport_id))
    print(f"Total golf tournaments returned: {len(tournaments)}")
    print(json.dumps(tournaments, indent=1))

    # ---- Step 4: see every real golf market ----
    section(f"STEP 4 -- GET /v4/markets?sportId={golf_sport_id}")
    markets = as_list(fetch("/v4/markets", key, sportId=golf_sport_id))
    print(f"Total golf markets returned: {len(markets)}")
    print(json.dumps(markets, indent=1))
    prop_markets = [m for m in markets if m.get("playerProp")]
    print(f"\nOf those, {len(prop_markets)} are flagged playerProp=true:")
    for m in prop_markets:
        print(f"  marketId={m.get('marketId')}  marketName={m.get('marketName')}  handicap={m.get('handicap')}")

    # ---- Step 5: pull real PrizePicks odds for the first tournament found ----
    if tournaments:
        first_id = tournaments[0].get("tournamentId")
        section(f"STEP 5 -- GET /v4/odds-by-tournaments?bookmaker=prizepicks&tournamentIds={first_id}&oddsFormat=american")
        odds = fetch("/v4/odds-by-tournaments", key, bookmaker="prizepicks",
                     tournamentIds=first_id, oddsFormat="american")
        dumped = json.dumps(odds, indent=1)
        print(dumped[:8000])
        if len(dumped) > 8000:
            print(f"\n(...truncated -- full response was {len(dumped)} chars. Paste what printed above, that's enough to see the shape.)")
    else:
        print("\nSTEP 5 skipped -- no tournaments found in step 3.")


if __name__ == "__main__":
    main()

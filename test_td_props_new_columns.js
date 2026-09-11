// test_td_props_new_columns.js -- Verifies MODEL ODDS, BOOK %, RAW EDGE,
// and IMP TOT actually render correctly in the TD Props table, using data
// shaped exactly like the new generate_player_predictions.py output
// (model.fair_moneyline, market.implied_prob, team_implied_total).

const fs = require('fs');
const path = require('path');
const http = require('http');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');

const MOCK_LINES_JS = `const NFL_LINES_DATA = null;`;

// Two players: one WITH a live price (all new fields populated), one
// WITHOUT (no market at all) -- to check both the happy path and the
// "no price yet" graceful-dash path don't break with the new columns.
const MOCK_PLAYER_TD_JS = `
const PLAYER_TD_DATA = {
  "generated_at": "2026-09-09T15:06:55Z",
  "model_notes": {"training_rows": 12345, "caveat": "test caveat"},
  "players": [
    {
      "player_name": "Jahmyr Gibbs", "position": "RB", "team": "DET", "opponent": "NO",
      "matchup": "DET vs NO",
      "team_implied_total": 27.456,
      "usage": {"snap_share": 0.7, "rz_target_share": null, "inside5_carry_share": 0.6, "xtd_per_game": 1.1, "matchup_rating": 1.05},
      "model": {"anytime_td_prob": 0.64, "fair_moneyline": -177.8, "tier": "A"},
      "market": {"anytime_td_price": -285, "implied_prob": 0.7403, "edge": -0.1003, "ev": -0.08, "source": "live_api"},
      "qualification": {"qualifies": true, "tier": "A", "reason_codes": []}
    },
    {
      "player_name": "Unpriced Guy", "position": "WR", "team": "SEA", "opponent": "ARI",
      "matchup": "SEA @ ARI",
      "team_implied_total": null,
      "usage": {"snap_share": 0.5, "rz_target_share": 0.1, "inside5_carry_share": null, "xtd_per_game": 0.2, "matchup_rating": null},
      "model": {"anytime_td_prob": 0.22, "fair_moneyline": 354.5, "tier": "C"},
      "market": null,
      "qualification": {"qualifies": false, "tier": "U", "reason_codes": ["no_live_price"]}
    }
  ]
};
`;

let server, baseURL;
function startServer() {
  return new Promise((resolve) => {
    server = http.createServer((req, res) => {
      const filePath = path.join(SITE_DIR, decodeURIComponent(req.url.split('?')[0]));
      fs.readFile(filePath, (err, data) => {
        if (err) { res.writeHead(404); res.end(); return; }
        const ext = path.extname(filePath);
        const type = ext === '.js' ? 'application/javascript' : 'text/html';
        res.writeHead(200, {'Content-Type': type});
        res.end(data);
      });
    });
    server.listen(0, () => { baseURL = `http://127.0.0.1:${server.address().port}/`; resolve(); });
  });
}

async function main() {
  await startServer();
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'nfl_lines.js'), MOCK_LINES_JS);
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'player_td.js'), MOCK_PLAYER_TD_JS);

  const dom = await JSDOM.fromURL(baseURL + 'index.html', {runScripts: 'dangerously', resources: 'usable'});
  await new Promise(r => setTimeout(r, 300));
  const doc = dom.window.document;

  // Switch to the TD Props tab (it's not the default active view).
  doc.querySelector('[data-view="props"]').dispatchEvent(new dom.window.MouseEvent('click', {bubbles: true}));

  const checks = [];
  const rows = [...doc.querySelectorAll('#props-body .board-row')];
  checks.push(['both players rendered', rows.length, 2]);

  const gibbsRow = rows.find(r => r.textContent.includes('Jahmyr Gibbs'));
  const rowText = gibbsRow.textContent;

  // MODEL ODDS: fair_moneyline -177.8 -> displayed via fmtOdds (no + prefix for negative)
  checks.push(['MODEL ODDS shows the fair moneyline for a priced player', rowText.includes('-177.8'), true]);
  // BOOK %: implied_prob 0.7403 -> fmtPct rounds to nearest whole % = 74%
  checks.push(['BOOK % shows the market-implied probability as a percent', rowText.includes('74%'), true]);
  // RAW EDGE: edge -0.1003 -> fmtSigned = "-10.0%"
  checks.push(['RAW EDGE shows the real edge value', rowText.includes('10.0%'), true]);
  // IMP TOT: team_implied_total 27.456 -> rounded client-side to 1 decimal = 27.5
  checks.push(['IMP TOT shows the team implied total to 1 decimal', rowText.includes('27.5'), true]);

  // ---- Unpriced player: every new column must show a graceful dash, not
  // "undefined", "null", "NaN", or a thrown error ----
  const unpricedRow = rows.find(r => r.textContent.includes('Unpriced Guy'));
  const unpricedText = unpricedRow.textContent;
  checks.push(['unpriced player: no literal "undefined" anywhere in the row', unpricedText.includes('undefined'), false]);
  checks.push(['unpriced player: no literal "null" anywhere in the row', unpricedText.includes('null'), false]);
  checks.push(['unpriced player: no literal "NaN" anywhere in the row', unpricedText.includes('NaN'), false]);
  // MODEL ODDS should still show for the unpriced player (it's from model, not market)
  checks.push(['unpriced player still gets a real MODEL ODDS (comes from model, not market)',
    unpricedText.includes('354.5'), true]);
  // BOOK % and IMP TOT should be dashes (both genuinely null/no-market for this player)
  const dashCount = (unpricedText.match(/\u2014/g) || []).length;
  checks.push(['unpriced player shows at least 2 real dashes (BOOK % and IMP TOT, both null)', dashCount >= 2, true]);

  console.log(`${'check'.padEnd(75)} got      want   ok`);
  console.log('-'.repeat(100));
  let allPass = true;
  for (const [name, got, want] of checks) {
    const ok = got === want;
    allPass = allPass && ok;
    console.log(`${name.padEnd(75)} ${String(got).padEnd(8)} ${String(want).padEnd(6)} ${ok ? 'PASS' : 'FAIL'}`);
  }
  console.log('-'.repeat(100));
  console.log(allPass ? 'ALL PASS' : 'SOME FAILED');
  server.close();
  dom.window.close();
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); if (server) server.close(); process.exit(1); });

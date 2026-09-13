// test_status_pills.js -- Verifies the new OFFICIAL/DEGEN status system:
// a qualifying player shows OFFICIAL (green), a non-qualifying player with
// real positive edge shows DEGEN (amber), and players with no real edge
// either way (or no market at all) show no pill, using real-shaped data
// for all four cases.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');
const MOCK_LINES_JS = `const NFL_LINES_DATA = null;`;

function player(name, qualifies, tier, reasons, market) {
  return {
    player_name: name, position: 'WR', team: 'MIN', opponent: 'GB', matchup: 'MIN vs GB',
    team_implied_total: 24.0,
    usage: {snap_share: 0.8, rz_target_share: 0.2, inside5_carry_share: 0.0, xtd_per_game: 1.0, matchup_rating: 1.0},
    model: {anytime_td_prob: 0.4, fair_moneyline: 150, tier: 'A'},
    market,
    qualification: {qualifies, tier, reason_codes: reasons},
  };
}

const MOCK_PLAYER_TD_JS = `
const PLAYER_TD_DATA = {
  "generated_at": "2026-09-11T21:09:43Z",
  "model_notes": {"training_rows": 1, "caveat": "test"},
  "players": [
    ${JSON.stringify(player('Official Guy', true, 'B', ['Matchup rating 0.85 below 1.10 preferred (informational only, not yet a disqualifier)'],
      {anytime_td_price: 165, implied_prob: 0.3774, edge: 0.1022, ev: 0.2709, source: 'live_api'}))},
    ${JSON.stringify(player('Degen Guy', false, 'D', ['Snap share below 70% threshold'],
      {anytime_td_price: 450, implied_prob: 0.1818, edge: 0.357, ev: 1.962, source: 'live_api'}))},
    ${JSON.stringify(player('No Real Edge Guy', false, 'D', ['Model projection does not exceed market implied probability'],
      {anytime_td_price: -175, implied_prob: 0.6364, edge: -0.1243, ev: -0.1953, source: 'live_api'}))},
    ${JSON.stringify(player('No Market Guy', false, 'D', ['No live price -- edge/EV not evaluated'], null))}
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
  doc.querySelector('[data-view="props"]').dispatchEvent(new dom.window.MouseEvent('click', {bubbles: true}));

  const checks = [];
  const rows = [...doc.querySelectorAll('#props-body .board-row')];

  function statusOf(name) {
    const row = rows.find(r => r.textContent.includes(name));
    return row ? row.querySelector('.col-c') : null;
  }

  const official = statusOf('Official Guy');
  checks.push(['qualifying player shows OFFICIAL text', official.textContent.trim(), 'OFFICIAL']);
  checks.push(['OFFICIAL pill uses the green status-official class', official.querySelector('.status-official') != null, true]);

  const degen = statusOf('Degen Guy');
  checks.push(['non-qualifying but positive-edge player shows DEGEN text', degen.textContent.trim(), 'DEGEN']);
  checks.push(['DEGEN pill uses the amber status-degen class', degen.querySelector('.status-degen') != null, true]);
  checks.push(['DEGEN tooltip explains the model sees edge despite not qualifying',
    degen.querySelector('.status-degen').getAttribute('title').includes('real edge'), true]);

  const noEdge = statusOf('No Real Edge Guy');
  checks.push(['non-qualifying, negative-edge player shows no pill (just a dash)', noEdge.textContent.trim(), '\u2014']);
  checks.push(['negative-edge player has neither OFFICIAL nor DEGEN class',
    noEdge.querySelector('.status-official') == null && noEdge.querySelector('.status-degen') == null, true]);

  const noMarket = statusOf('No Market Guy');
  checks.push(['player with no market at all shows no pill (not a crash on null market)', noMarket.textContent.trim(), '\u2014']);

  const pageHTML = doc.getElementById('props-body').innerHTML;
  checks.push(['no leftover qual-tier class anywhere in rendered output', pageHTML.includes('qual-tier'), false]);

  console.log(`${'check'.padEnd(75)} got       want      ok`);
  console.log('-'.repeat(105));
  let allPass = true;
  for (const [name, got, want] of checks) {
    const ok = got === want;
    allPass = allPass && ok;
    console.log(`${name.padEnd(75)} ${String(got).padEnd(9)} ${String(want).padEnd(9)} ${ok ? 'PASS' : 'FAIL'}`);
  }
  console.log('-'.repeat(105));
  console.log(allPass ? 'ALL PASS' : 'SOME FAILED');
  server.close();
  dom.window.close();
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); if (server) server.close(); process.exit(1); });

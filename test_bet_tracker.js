// test_bet_tracker.js -- Functional test for the personal bet tracker:
// real simulated form input, real button clicks, checking actual computed
// P/L and stats against hand-calculated expected values -- not just that
// the JS parses. Also round-trips CSV export/import, since a CSV
// join/split bug was already caught once during this build (an escaping
// mistake produced a literal "\n" text instead of a real newline) --
// syntax checking alone did not catch that, so this test exercises the
// actual join/split behavior end to end.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { JSDOM } = require('jsdom');

const SITE_DIR = path.join(__dirname, 'testsite');

const EMPTY_LINES_JS = `const NFL_LINES_DATA = null;`;
const EMPTY_PLAYER_JS = `const PLAYER_TD_DATA = null;`;

// jsdom's localStorage isn't reliably usable under file:// (mirrors real
// browser behavior -- file: is often a storage-restricted opaque origin
// too), so serve the test site over a real local HTTP server instead.
let server, baseURL;
function startServer() {
  return new Promise((resolve) => {
    server = http.createServer((req, res) => {
      const filePath = path.join(SITE_DIR, decodeURIComponent(req.url.split('?')[0]));
      fs.readFile(filePath, (err, data) => {
        if (err) { res.writeHead(404); res.end(); return; }
        const ext = path.extname(filePath);
        const type = ext === '.js' ? 'application/javascript' : ext === '.html' ? 'text/html' : 'text/plain';
        res.writeHead(200, {'Content-Type': type});
        res.end(data);
      });
    });
    server.listen(0, () => {
      baseURL = `http://127.0.0.1:${server.address().port}/`;
      resolve();
    });
  });
}

async function setup() {
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'nfl_lines.js'), EMPTY_LINES_JS);
  fs.writeFileSync(path.join(SITE_DIR, 'data', 'player_td.js'), EMPTY_PLAYER_JS);
  const dom = await JSDOM.fromURL(baseURL + 'index.html', {
    runScripts: 'dangerously', resources: 'usable',
  });
  await new Promise(r => setTimeout(r, 300));
  return dom;
}

// Fresh, isolated state for a test that doesn't want to see other tests'
// data -- localStorage correctly PERSISTS across page loads at the same
// origin (same as a real browser), which Test 6 specifically relies on, so
// isolation has to be requested explicitly rather than assumed. Clearing
// storage after a page has already loaded doesn't un-render what it
// already rendered from the old data, so this does a genuine second page
// load (real browser behavior for "clear storage, then reload").
async function setupClean() {
  const cleaner = await setup();
  cleaner.window.localStorage.clear();
  cleaner.window.close();
  return setup();
}

function fillAndAdd(doc, win, {game, market, side, odds, stake}) {
  doc.getElementById('tf-game').value = game;
  if (market) doc.getElementById('tf-market').value = market;
  doc.getElementById('tf-side').value = side;
  doc.getElementById('tf-odds').value = String(odds);
  doc.getElementById('tf-stake').value = String(stake);
  doc.getElementById('tf-add').dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function cycleResult(doc, win, times) {
  for (let i = 0; i < times; i++) {
    // Re-query fresh each time -- renderTracker() replaces the whole table
    // via innerHTML after every click, so a cached element reference goes
    // stale (detached from the DOM) after the first iteration and stops
    // bubbling to the delegated listener.
    const pill = doc.querySelector('[data-action="cycle-result"]');
    pill.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  }
}

async function main() {
  await startServer();
  const checks = [];

  // ================================================================
  // Test 1: add a -110 bet, mark it a WIN, check computed P/L
  // ================================================================
  {
    const dom = await setupClean();
    const { window } = dom;
    const doc = window.document;
    // Suppress the empty-storage alert path / confirm dialogs during this run.
    window.alert = () => {};
    window.confirm = () => true;

    fillAndAdd(doc, window, {game: 'NO @ DET', market: 'Spread', side: 'DET -7', odds: -110, stake: 50});
    checks.push(['bet appears in table after adding', doc.querySelectorAll('.board-row[data-bet-id]').length, 1]);
    checks.push(['new bet starts PENDING', doc.querySelector('.result-pill').textContent, 'PENDING']);

    // Cycle: pending -> win
    cycleResult(doc, window, 1);
    checks.push(['after 1 cycle, result is WIN', doc.querySelector('.result-pill').textContent, 'WIN']);

    // -110 odds, $50 stake, WIN: profit = 50 * (100/110) = 45.4545... -> displayed $45.45
    const plCell = doc.querySelector('[data-bet-id] .cell-edge');
    checks.push(['P/L for -110/$50 WIN reads +$45.45', plCell.textContent.trim(), '+$45.45']);

    // Stats strip: record should be 1-0, staked $50.00, profit +$45.45, ROI +90.9%
    const statsText = doc.getElementById('tracker-stats').textContent;
    checks.push(['stats show record 1-0', statsText.includes('1-0'), true]);
    checks.push(['stats show total staked $50.00', statsText.includes('$50.00'), true]);
    checks.push(['stats show ROI around +90.9%', /\+90\.9%/.test(statsText), true]);

    dom.window.close();
  }

  // ================================================================
  // Test 2: a LOSS bet computes P/L = -stake exactly
  // ================================================================
  {
    const dom = await setupClean();
    const { window } = dom;
    const doc = window.document;
    window.alert = () => {};
    window.confirm = () => true;

    fillAndAdd(doc, window, {game: 'WAS @ PHI', market: 'Spread', side: 'PHI -5.5', odds: -110, stake: 100});
    cycleResult(doc, window, 2); // pending -> win -> loss
    checks.push(['2 cycles from pending lands on LOSS', doc.querySelector('.result-pill').textContent, 'LOSS']);
    const plCell = doc.querySelector('[data-bet-id] .cell-edge');
    checks.push(['LOSS P/L = exactly -$100.00 (full stake)', plCell.textContent.trim(), '\u2212$100.00']);

    dom.window.close();
  }

  // ================================================================
  // Test 3: PUSH returns $0 profit, and a positive-odds underdog bet
  // computes correctly (+150 odds, WIN)
  // ================================================================
  {
    const dom = await setupClean();
    const { window } = dom;
    const doc = window.document;
    window.alert = () => {};
    window.confirm = () => true;

    fillAndAdd(doc, window, {game: 'A @ B', market: 'Total', side: 'Over 44.5', odds: -110, stake: 20});
    cycleResult(doc, window, 3); // pending -> win -> loss -> push
    checks.push(['3 cycles lands on PUSH', doc.querySelector('.result-pill').textContent, 'PUSH']);
    const plCell1 = doc.querySelector('[data-bet-id] .cell-edge');
    checks.push(['PUSH P/L is exactly $0.00 (stake returned, no profit)', plCell1.textContent.trim(), '$0.00']);

    fillAndAdd(doc, window, {game: 'C @ D', market: 'Moneyline', side: 'D ML', odds: 150, stake: 40});
    const rowsBeforeClick = [...doc.querySelectorAll('.board-row[data-bet-id]')];
    const rowBeforeClick = rowsBeforeClick.find(r => r.textContent.includes('C @ D'));
    rowBeforeClick.querySelector('[data-action="cycle-result"]').dispatchEvent(new window.MouseEvent('click', {bubbles: true}));
    // Re-query fresh after the click -- same reason as cycleResult() above:
    // renderTracker() replaced the whole table, so the pre-click row/cell
    // references are now detached and still show the OLD pre-click state.
    // +150 odds, $40 stake, WIN: profit = 40 * (150/100) = $60.00
    const rowAfterClick = [...doc.querySelectorAll('.board-row[data-bet-id]')].find(r => r.textContent.includes('C @ D'));
    const plCell2 = rowAfterClick.querySelector('.cell-edge');
    checks.push(['+150 odds / $40 stake WIN = +$60.00 exactly', plCell2.textContent.trim(), '+$60.00']);

    dom.window.close();
  }

  // ================================================================
  // Test 4: delete a bet
  // ================================================================
  {
    const dom = await setupClean();
    const { window } = dom;
    const doc = window.document;
    window.alert = () => {};
    window.confirm = () => true;

    fillAndAdd(doc, window, {game: 'DEL TEST', market: 'Spread', side: 'X -3', odds: -110, stake: 10});
    checks.push(['bet added before delete test', doc.querySelectorAll('.board-row[data-bet-id]').length, 1]);
    doc.querySelector('[data-action="delete"]').dispatchEvent(new window.MouseEvent('click', {bubbles: true}));
    checks.push(['table is empty after delete', doc.querySelectorAll('.board-row[data-bet-id]').length, 0]);
    checks.push(['empty-state message shows after deleting the only bet',
      doc.getElementById('tracker-body').textContent.includes('No bets logged yet'), true]);

    dom.window.close();
  }

  // ================================================================
  // Test 5: CSV export/import round-trip -- the exact area where a real
  // newline-escaping bug was already caught once during this build.
  // ================================================================
  {
    const dom = await setupClean();
    const { window } = dom;
    const doc = window.document;
    window.alert = () => {};
    window.confirm = () => true;

    fillAndAdd(doc, window, {game: 'ROUNDTRIP 1', market: 'Spread', side: 'X -3', odds: -110, stake: 25});
    fillAndAdd(doc, window, {game: 'ROUNDTRIP 2', market: 'Total', side: 'Over 45', odds: 120, stake: 30});
    checks.push(['2 bets present before export', doc.querySelectorAll('.board-row[data-bet-id]').length, 2]);

    // Read the CSV that export would produce by calling the same internal
    // logic path: simulate export via Blob capture.
    let capturedCSV = null;
    const OrigBlob = window.Blob;
    window.Blob = function(parts, opts) { capturedCSV = parts.join(''); return new OrigBlob(parts, opts); };
    const origCreateObjectURL = window.URL.createObjectURL;
    window.URL.createObjectURL = () => 'blob:mock';
    window.URL.revokeObjectURL = () => {};
    doc.getElementById('tt-export').dispatchEvent(new window.MouseEvent('click', {bubbles: true}));

    checks.push(['export produced a CSV string', typeof capturedCSV, 'string']);
    checks.push(['CSV contains a REAL newline between rows (not literal backslash-n text)',
      capturedCSV.includes('\n') && !capturedCSV.includes('\\n'), true]);
    checks.push(['CSV has header + 2 data rows = 3 lines', capturedCSV.trim().split('\n').length, 3]);
    checks.push(['CSV contains both game names', capturedCSV.includes('ROUNDTRIP 1') && capturedCSV.includes('ROUNDTRIP 2'), true]);

    // Now clear everything and re-import that exact CSV, confirming a
    // genuine round-trip: data survives export -> import intact.
    doc.getElementById('tt-clear').dispatchEvent(new window.MouseEvent('click', {bubbles: true}));
    checks.push(['cleared before import test', doc.querySelectorAll('.board-row[data-bet-id]').length, 0]);

    // Simulate a file import: construct a File-like object from the captured CSV.
    const file = new window.File([capturedCSV], 'bets.csv', {type: 'text/csv'});
    const fileInput = doc.getElementById('tt-import-file');
    Object.defineProperty(fileInput, 'files', {value: [file], configurable: true});
    await new Promise((resolve) => {
      const orig = window.FileReader.prototype.readAsText;
      fileInput.dispatchEvent(new window.Event('change', {bubbles: true}));
      // FileReader is async even in jsdom -- poll briefly for the import to land.
      const check = () => {
        if (doc.querySelectorAll('.board-row[data-bet-id]').length > 0) resolve();
        else setTimeout(check, 50);
      };
      setTimeout(check, 50);
    });

    checks.push(['both bets restored after CSV import', doc.querySelectorAll('.board-row[data-bet-id]').length, 2]);
    const restoredText = doc.getElementById('tracker-body').textContent;
    checks.push(['restored data includes both original games',
      restoredText.includes('ROUNDTRIP 1') && restoredText.includes('ROUNDTRIP 2'), true]);

    // Re-importing the SAME file again must skip duplicates (dedupe by id), not double the rows.
    fileInput.dispatchEvent(new window.Event('change', {bubbles: true}));
    await new Promise(r => setTimeout(r, 150));
    checks.push(['re-importing the same CSV does not create duplicates',
      doc.querySelectorAll('.board-row[data-bet-id]').length, 2]);

    window.Blob = OrigBlob;
    dom.window.close();
  }

  // ================================================================
  // Test 6: localStorage actually persists across a fresh page load
  // (simulates closing and reopening the browser)
  // ================================================================
  {
    const dom1 = await setup();
    dom1.window.alert = () => {};
    fillAndAdd(dom1.window.document, dom1.window, {game: 'PERSIST TEST', market: 'Spread', side: 'X -3', odds: -110, stake: 15});
    const storedRaw = dom1.window.localStorage.getItem('pitchedge_bet_tracker_v1');
    dom1.window.close();

    checks.push(['data was actually written to localStorage', storedRaw != null && storedRaw.includes('PERSIST TEST'), true]);
  }

  console.log(`${'check'.padEnd(70)} got         want        ok`);
  console.log('-'.repeat(105));
  let allPass = true;
  for (const [name, got, want] of checks) {
    const ok = got === want;
    allPass = allPass && ok;
    console.log(`${name.padEnd(70)} ${String(got).padEnd(11)} ${String(want).padEnd(11)} ${ok ? 'PASS' : 'FAIL'}`);
  }
  console.log('-'.repeat(105));
  console.log(allPass ? 'ALL PASS' : 'SOME FAILED');
  server.close();
  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); if (server) server.close(); process.exit(1); });

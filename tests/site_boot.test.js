/* The static build boots and answers, and it was built from the fits in play.
 *
 * `site/solver-worker.js` runs unmodified through `vm`, with `fetch` reading
 * `site/`. It must reach ready, report the answer record through the date the
 * repo's record ends, and find a baked opening for a fresh board in every page
 * setting. `site/fits.json` must match `estimator_cache/`, since a refit
 * invalidates `site/` and its openings with it. Skips with no build.
 *
 *     node tests/site_boot.test.js
 */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.join(__dirname, "..");
const SITE = process.env.SITE || path.join(ROOT, "site");
const READY_SECONDS = 120;

let failures = 0;

function check(what, test) {
  try {
    test();
    console.log(`ok    ${what}`);
  } catch (error) {
    failures += 1;
    console.log(`FAIL  ${what}\n      ${error.message}`);
  }
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

/* A `fetch` reply for a file under site/, enough for what the worker reads. */
function response(file) {
  const bytes = fs.readFileSync(path.join(SITE, file));
  const reply = {
    ok: true, status: 200, statusText: "OK",
    clone: () => reply,
    arrayBuffer: async () =>
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length),
    json: async () => JSON.parse(bytes.toString("utf8")),
    text: async () => bytes.toString("utf8"),
  };
  return reply;
}

/* The worker in a context of its own; `ask` resolves with the reply's JSON. */
function boot() {
  const pending = new Map();
  let failed = null;
  const context = {
    TextEncoder, TextDecoder, WebAssembly, performance, console,
    fetch: async (file) => response(file),
    postMessage: (message) => {
      if (message.kind === "failed") failed = message.error;
      else if (pending.has(message.id)) pending.get(message.id)(message.text);
    },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(SITE, "solver-worker.js"), "utf8"),
                  context);
  let next = 0;
  const ask = (route, body) => new Promise((resolve) => {
    const id = ++next;
    pending.set(id, (text) => { pending.delete(id); resolve(JSON.parse(text)); });
    context.onmessage({ data: { id, path: route, body: body ? JSON.stringify(body) : "" } });
  });
  return { ask, failure: () => failed };
}

/* The slug `priors.PriorConfig` gives the page's two switches at the shipped shapes. */
function slug(repeat, curation) {
  return `repeat${+repeat}_curation${+curation}` +
    (repeat ? "_r-binned" : "") + (curation ? "_c-wordlebot" : "");
}

async function main() {
  if (!fs.existsSync(path.join(SITE, "solver-worker.js")) ||
      !fs.existsSync(path.join(SITE, "wordle.wasm"))) {
    console.log(`skipped: no build in ${SITE}; run ./build_site.sh`);
    return 0;
  }

  const fits = readJson(path.join(SITE, "fits.json"));
  check("site/fits.json holds the fits estimator_cache/ has in play", () => {
    for (const [name, fit] of Object.entries(fits)) {
      const cached = readJson(path.join(ROOT, "estimator_cache", `params_${name}.json`));
      assert.strictEqual(fit.form, cached.form, `${name}: form`);
      assert.deepStrictEqual(fit.params, cached.params, `${name}: params`);
    }
  });

  const { ask, failure } = boot();
  const started = Date.now();
  let status = null;
  while (!(status && status.ready) && !failure()) {
    if (Date.now() - started > READY_SECONDS * 1000) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
    status = await ask("/api/status");
  }
  check("the module reaches ready", () => {
    assert.strictEqual(failure(), null, `boot failed: ${failure()}`);
    assert.ok(status && status.ready, `not ready after ${READY_SECONDS} s: ` +
                                      JSON.stringify(status));
  });
  if (!(status && status.ready)) return 1;

  const lastDate = (file) => readJson(file).slice(-1)[0].print_date;
  check("it reports the answer record through the repo's last answer", () => {
    assert.strictEqual(status.record_through,
                       lastDate(path.join(SITE, "data", "solutions.json")));
    assert.strictEqual(status.record_through,
                       lastDate(path.join(ROOT, "data", "solutions.json")));
  });

  const openings = readJson(path.join(SITE, "openings.json"));
  let boards = 0;
  for (const repeat of [false, true]) {
    for (const curation of [false, true]) {
      for (const mode of status.modes) {
        for (const hard of [false, true]) {
          const reply = await ask("/api/new", { repeat, curation, mode, hard,
                                                form: null, top: 5 });
          boards += 1;
          const label = `${slug(repeat, curation)}, ${mode}, ` +
                        `${hard ? "hard" : "free"}`;
          check(`a fresh board serves its baked opening: ${label}`, () => {
            assert.ok(!reply.error, reply.error);
            // form null is *Whatever is cached*, baked from the fit in play
            const key = `${slug(repeat, curation)}|${mode}|${hard ? "hard" : "free"}|cached`;
            assert.ok(openings[key], `no opening under ${key}`);
            const words = [reply.recommendation, ...reply.alternatives]
              .map((guess) => guess.word);
            assert.deepStrictEqual(words, openings[key].map((guess) => guess.word));
          });
        }
      }
    }
  }
  console.log(`\n${boards} boards over the page's switches`);

  // every word a board lists scores the same through /api/score, along games
  // against drawn answers that leave the baked openings after turn one
  let scored = 0;
  let exact = 0;
  for (const mode of status.modes) {
    for (const hard of [false, true]) {
      const settings = { repeat: true, curation: true, mode, hard, form: null, top: 5 };
      const misses = [];
      for (const seed of [0.13, 0.47, 0.81]) {
        const { word: answer } = await ask("/api/draw", { ...settings, seed });
        const rows = [];
        for (let turn = 0; turn < 6; turn += 1) {
          const board = await ask("/api/board", { ...settings, answer, rows });
          if (board.error || board.over) break;
          const listed = [board.recommendation, ...board.alternatives];
          for (const guess of listed) {
            const reply = await ask("/api/score", { ...settings, answer, rows,
                                                    word: guess.word });
            scored += 1;
            if (reply.exact) exact += 1;
            if (reply.error || Math.abs(reply.score - guess.score) > 1e-9) {
              misses.push(`${rows.map((r) => r.word).join(",")}>${guess.word}: ` +
                          `${guess.score} listed, ${reply.error || reply.score} scored`);
            }
          }
          // a runner-up on the first turn, so the rest is played live
          const next = turn === 0 ? listed[listed.length - 1] : listed[0];
          rows.push({ word: next.word, pattern: "NNNNN" });
        }
      }
      check(`a listed word scores as listed: ${mode}, ${hard ? "hard" : "free"}`, () => {
        assert.deepStrictEqual(misses, []);
      });
    }
  }
  console.log(`${scored} listed words scored again, ${exact} of them exactly`);
  console.log(failures ? `${failures} failures` : "all passed");
  return failures ? 1 : 0;
}

main().then((code) => process.exit(code), (error) => {
  console.log(`FAIL  ${error.stack}`);
  process.exit(1);
});

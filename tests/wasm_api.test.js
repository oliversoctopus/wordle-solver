/* The static build's fetch shim keeps the contract the page was written to.
 *
 * `api()` throws on a reply that is not ok and renders a board on one that
 * is, so a refusal arriving as 200 would be applied as a board.  The status
 * is the shim's own invention, which no Python test can see, so
 * `web/wasm-api.js` is loaded unmodified through `vm` against a fake Worker.
 *
 *     node tests/wasm_api.test.js
 */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SHIM = process.env.SHIM || path.join(__dirname, "..", "web", "wasm-api.js");

/* Loads the shim with a worker that replies with `answer(path, body)`. */
function load(answer) {
  const sent = [];
  const context = {
    Response, JSON, Promise, Map, Math, URL, queueMicrotask, console,
    Worker: class {
      constructor(url) {
        this.url = url;
        this.onmessage = null;
      }
      postMessage(request) {
        sent.push(request);
        const text = answer(request.path, request.body);
        queueMicrotask(() =>
          this.onmessage({ data: { kind: "reply", id: request.id, text } }));
      }
    },
    window: { fetch: async () => "not the solver's business" },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(SHIM, "utf8"), context);
  return { fetch: context.window.fetch, sent, context };
}

const post = (fetch, path, body) =>
  fetch(path, { method: "POST", body: JSON.stringify(body || {}) });

/* What `app.js`'s `api()` does with a reply, in the two lines that matter. */
async function asAppJsWouldRead(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

const cases = [];
const test = (name, fn) => cases.push([name, fn]);

test("a board comes back as a board", async () => {
  const board = { session: "wasm", rows: [{ word: "salet", pattern: "NNNNN" }] };
  const { fetch } = load(() => JSON.stringify(board));
  const response = await post(fetch, "/api/board", { rows: [] });

  assert.strictEqual(response.ok, true);
  assert.deepStrictEqual(await asAppJsWouldRead(response), board);
});

test("a refusal comes back as a refusal, not as a board", async () => {
  // the real one: three gray guesses that no word can satisfy at once
  const message = "Row 3 contradicts the rows above it -- no word matches all of them";
  const { fetch } = load(() => JSON.stringify({ error: message }));
  const response = await post(fetch, "/api/board", { rows: [] });

  assert.strictEqual(response.ok, false, "a refusal must not arrive as 200");
  assert.strictEqual(response.status, 400);
  await assert.rejects(() => asAppJsWouldRead(response), { message });
});

test("a reply that is not JSON at all is a refusal too", async () => {
  const { fetch } = load(() => "<html>a proxy got in the way</html>");
  const response = await post(fetch, "/api/board", { rows: [] });

  assert.strictEqual(response.ok, false);
});

test("anything outside /api/ is left alone", async () => {
  const { fetch, sent } = load(() => JSON.stringify({}));

  assert.strictEqual(await fetch("/style.css"), "not the solver's business");
  assert.strictEqual(sent.length, 0, "the solver was asked for a stylesheet");
});

test("a new game is dealt a seed, since the module has no entropy", async () => {
  const { fetch, sent } = load(() => JSON.stringify({ session: "g1" }));
  await post(fetch, "/api/game/new", {});
  await post(fetch, "/api/board", { rows: [] });

  const dealt = JSON.parse(sent[0].body);
  assert.strictEqual(typeof dealt.seed, "number", "no seed on /api/game/new");
  assert.ok(!("seed" in JSON.parse(sent[1].body)), "a seed on a board request");
});

test("a full URL is routed on its path", async () => {
  const { fetch, sent } = load(() => JSON.stringify({ ok: true }));
  await fetch("http://127.0.0.1:8010/api/status", {});

  assert.strictEqual(sent.length, 1);
  assert.strictEqual(sent[0].path, "/api/status");
});

async function main() {
  let failed = 0;
  for (const [name, fn] of cases) {
    try {
      await fn();
      console.log(`      ok    ${name}`);
    } catch (error) {
      failed++;
      console.log(`      FAIL  ${name}`);
      console.log(`            ${error.message}`);
    }
  }
  console.log(failed ? `\n${failed} of ${cases.length} failed`
                     : `      ${cases.length} checks, the shim keeps its contract`);
  process.exit(failed ? 1 : 0);
}

main();

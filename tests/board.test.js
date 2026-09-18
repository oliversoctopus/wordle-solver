/* The solve board's editing rules, over a fake DOM and a fake server.
 *
 * Two halves.  The anchors below name particular sequences and say what the
 * board should look like afterwards; the properties are things that have to be
 * true of every board, checked after every operation of a random walk.  The
 * anchors say what was meant, and the walk finds the sequences nobody thought
 * to write down.
 *
 *     node tests/board.test.js [seeds] [steps]
 *
 * ``web/app.js`` is loaded unmodified through ``vm``, with a getter object
 * appended to reach the module's own variables.
 */

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const APP = process.env.APP || path.join(__dirname, "..", "web", "app.js");
const MAX_GUESSES = 6;

/* ------------------------------------------------------------- a fake DOM */

function el(tag) {
  const node = {
    tagName: tag, className: "", title: "", hidden: false, disabled: false,
    checked: false, value: "", open: false, innerHTML: "", tabIndex: 0,
    type: "", dataset: {}, children: [], options: [], style: {}, handlers: {},
    classList: null,
    appendChild(child) { node.children.push(child); return child; },
    addEventListener(kind, fn) { node.handlers[kind] = fn; },
    setAttribute() {}, blur() {}, focus() {},
    matches() { return false; },
    querySelectorAll() { return []; },
  };
  const classes = new Set();
  node.classList = {
    add: (...names) => names.forEach((name) => classes.add(name)),
    remove: (...names) => names.forEach((name) => classes.delete(name)),
    contains: (name) => classes.has(name),
  };
  let text = "";
  Object.defineProperty(node, "textContent", {
    get: () => text,
    set: (value) => { text = value; if (value === "") node.children.length = 0; },
  });
  return node;
}

const nodes = new Map();
const document = {
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, el(id));
    return nodes.get(id);
  },
  createElement: (tag) => el(tag),
  addEventListener() {},
  querySelectorAll: () => [],
  body: el("body"),
};

/* ---------------------------------------------------------- a fake server */

/* one pick per turn, so a stale suggestion is telling: the word says which
   board the solver was asked about */
const PICKS = ["salet", "crony", "dumpy", "might", "haste", "fixed"];
const WIN = "GGGGG";

let sent = [];      // the rows of every /api/board request, in order

function patternString(pattern) {
  return typeof pattern === "string"
    ? pattern : pattern.map((tile) => "NYG"[tile]).join("");
}

/* Wordle's coloring rule, for the fake server to color a board against a known
   answer with; the real one is pinned in tests/anchors.py */
function score(guess, answer) {
  const tiles = [0, 0, 0, 0, 0];
  const left = answer.split("");
  for (let i = 0; i < 5; i++) {
    if (guess[i] === answer[i]) { tiles[i] = 2; left[i] = null; }
  }
  for (let i = 0; i < 5; i++) {
    if (tiles[i]) continue;
    const at = left.indexOf(guess[i]);
    if (at >= 0) { tiles[i] = 1; left[at] = null; }
  }
  return tiles.map((tile) => "NYG"[tile]).join("");
}

/* what the draw hands out, in turn */
const ANSWERS = ["crony", "haste", "fixed", "dumpy"];
let drawnSoFar = 0;

/* the rules the real server applies that the board can see: a win ends it, and
   so does the sixth row -- and against a known answer the colors are its own */
function boardState(rows, answer = "") {
  if (answer) rows = rows.map((row) => ({ word: row.word, pattern: score(row.word, answer) }));
  const n = rows.length;
  const solved = n > 0 && rows[n - 1].pattern === WIN;
  const over = solved || n >= MAX_GUESSES;
  return {
    session: "test", turn: n + 1, guesses_left: Math.max(0, MAX_GUESSES - n),
    max_guesses: MAX_GUESSES, rows, solved, over, answer: answer || null,
    mode: "expected", hard: false, score_label: "expected guesses to finish",
    prior: {}, candidates_left: 100 - n, effective_size: 100 - n,
    possible_answers: [], alternatives: [], estimator: null, elapsed_ms: 1,
    recommendation: over ? null : {
      word: PICKS[n], score: 1, exact: false, possible: true, win_probability: 0.1,
    },
  };
}

async function fetchStub(url, init) {
  const body = init && init.body ? JSON.parse(init.body) : {};
  let payload;
  if (url === "/api/status") payload = { ready: true, progress: "", forms: ["sqrt"] };
  else if (url === "/api/new") payload = boardState([], body.answer);
  else if (url === "/api/board") {
    sent.push(body.rows);
    payload = boardState(body.rows.map((row) => ({
      word: row.word, pattern: patternString(row.pattern),
    })), body.answer);
  } else if (url === "/api/draw") {
    payload = { word: ANSWERS[drawnSoFar++ % ANSWERS.length] };
  } else throw new Error("unexpected url " + url);
  return { ok: true, json: async () => payload };
}

/* ------------------------------------------------------------------- load */

const source = fs.readFileSync(APP, "utf8") + `
globalThis.probe = {
  get rows() { return rows; },
  get state() { return state; },
  get cursor() { return cursor; },
  get busy() { return busy; },
  get board() { return board; },
  // the old version has neither, and the board it draws is the comparison
  openRow: typeof openRow === "function" ? openRow : () => rows.length - 1,
  lastFilledRow: typeof lastFilledRow === "function" ? lastFilledRow : () => -1,
  dirtyRows, suggestion, rowFilled, moveTo, typeLetter, backspace, cycle,
  deleteRow, submit, acceptSuggestion, applyState,
  answerBox: document.getElementById("solve-answer"),
  drawAnswer: typeof drawAnswer === "function" ? drawAnswer : () => {},
  playOut: typeof playOut === "function" ? playOut : () => {},
};
`;

const context = vm.createContext({
  document, fetch: fetchStub, console, setTimeout, clearTimeout,
  navigator: { clipboard: { writeText: async () => {} } },
});
vm.runInContext(source, context);

// the panel wants a fully described prior, which is not what is under test
context.renderPanel = () => {};
context.renderTechnical = () => {};

const probe = context.probe;

/* The fake server answers immediately, so a request settles through a chain of
   microtasks and never needs the clock.  Draining them is what makes tens of
   thousands of edits take seconds rather than minutes. */
async function flush() {
  for (let i = 0; i < 64; i++) await Promise.resolve();
}

/* ---------------------------------------------------------------- reading */

/* every row as it is drawn: a word, a (shadowed) word, or nothing */
function drawn() {
  return probe.board.children.map((row) => {
    const word = row.children.map((tile) => tile.textContent || " ").join("");
    if (word.trim() === "") return "";
    return row.children.every((tile) => tile.classList.contains("ghost"))
      ? `(${word})` : word;
  });
}

const shadowRows = () => probe.board.children
  .map((row, r) => (row.children.some((tile) => tile.classList.contains("ghost")) ? r : -1))
  .filter((r) => r >= 0);

/* everything about the board that can be seen or acted on, so that two ways of
   arriving at the same words can be compared whole */
const snapshot = () => ({
  drawn: drawn(),
  rows: probe.rows.map((row) => ({ letters: row.letters, pattern: row.pattern })),
  openRow: probe.openRow(),
  cursor: probe.cursor,
  serverRows: probe.state.rows,
  recommendation: probe.state.recommendation,
  turn: probe.state.turn,
  guessesLeft: probe.state.guesses_left,
  over: probe.state.over,
  candidates: probe.state.candidates_left,
});

let failures = 0;

function check(what, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"}  ${what}`);
  if (!ok) {
    console.log(`        got  ${JSON.stringify(got)}`);
    console.log(`        want ${JSON.stringify(want)}`);
  }
}

/* ------------------------------------------------------------- properties */

/* Each returns a complaint, or nothing.  These are the things that hold of
   every board there is, whatever was done to arrive at it. */
const PROPERTIES = {
  "at most one free row after the last word"() {
    const free = probe.rows.length - 1 - probe.lastFilledRow();
    if (free > 1) {
      return `${free} free rows: nothing may be selectable below the shadow`;
    }
  },

  "the cursor is on the board"() {
    const { row, col } = probe.cursor;
    if (!(row >= 0 && row < probe.rows.length)) return `row ${row} of ${probe.rows.length}`;
    if (!(col >= 0 && col <= 4)) return `column ${col}`;
  },

  "the board holds every row the solver was told about"() {
    if (probe.rows.length < probe.state.rows.length) {
      return `${probe.rows.length} rows against the solver's ${probe.state.rows.length}`;
    }
    if (probe.rows.length > MAX_GUESSES) return `${probe.rows.length} rows`;
  },

  "the shadow is in the free row, and nowhere else"() {
    const shadows = shadowRows();
    if (shadows.length > 1) return `shadows in rows ${shadows.join(", ")}`;
    if (shadows.length === 0) return;
    const at = shadows[0];
    if (at !== probe.openRow()) return `shadow in row ${at}, free row is ${probe.openRow()}`;
    if (probe.rowFilled(probe.rows[at])) return `shadow in row ${at}, which holds a word`;
    if (!probe.suggestion()) return `shadow in row ${at} with nothing suggested`;
  },

  "a board with nothing pending is the solver's own"() {
    if (probe.busy || probe.dirtyRows().size) return;
    const words = probe.state.rows.map((row) => row.word);
    const want = words.slice();
    const shadow = probe.state.over ? "" : probe.suggestion();
    if (shadow && want.length < MAX_GUESSES) want.push(`(${shadow})`);
    else if (!probe.state.over && want.length < MAX_GUESSES) want.push("");
    while (want.length < MAX_GUESSES) want.push("");
    const got = drawn();
    if (JSON.stringify(got) !== JSON.stringify(want)) {
      return `drawn ${JSON.stringify(got)}, solver holds ${JSON.stringify(want)}`;
    }
  },

  "against a known answer the colors are the server's"() {
    const answer = probe.answerBox.value;
    if (!answer || probe.busy) return;
    for (const [r, row] of probe.state.rows.entries()) {
      const mine = probe.rows[r];
      if (!mine || mine.letters.join("") !== row.word) continue;   // being retyped
      const pattern = mine.pattern.map((tile) => "NYG"[tile]).join("");
      if (pattern !== row.pattern) return `row ${r} shows ${pattern}, the answer gives ${row.pattern}`;
      if (row.pattern !== score(row.word, answer)) return `row ${r}: server colored ${row.pattern}`;
    }
  },

  "every tile holds a letter and a color it is allowed"() {
    for (const [r, row] of probe.rows.entries()) {
      for (let c = 0; c < 5; c++) {
        const letter = row.letters[c];
        if (letter !== "" && !/^[a-z]$/.test(letter)) return `row ${r} tile ${c}: "${letter}"`;
        if (![0, 1, 2].includes(row.pattern[c])) return `row ${r} tile ${c}: ${row.pattern[c]}`;
        if (letter === "" && row.pattern[c] !== 0) return `row ${r} tile ${c} colored but empty`;
      }
    }
  },
};

/* --------------------------------------------------------- the random walk */

/* mulberry32: a seed prints, and the sequence it names can be run again */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const LETTERS = "abcdefghijklmnopqrstuvwxyz";

/* What a person can do, rather than what a keyboard can send.  A walk over bare
   keystrokes never gets anywhere interesting: clearing a row by hand takes six
   particular presses in a row, so uniform noise effectively never does it, and
   the states behind it go unvisited.  These are whole intentions, and each one
   is spent one keystroke at a time so that the properties are judged in
   between. */
const MOVES = [
  ["a letter", 4, (r) => [["type", LETTERS[Math.floor(r() * 26)]]]],
  ["a word", 6, (r) => word(r).map((letter) => ["type", letter])],
  ["a backspace", 2, () => [["backspace"]]],
  ["clearing the row by hand", 4, () =>
    [["move", probe.cursor.row, 4], ...Array(5).fill(["backspace"])]],
  ["a color", 3, () => [["color"]]],
  ["the cursor moved", 2, (r) =>
    [["move", Math.floor(r() * MAX_GUESSES), Math.floor(r() * 5)]]],
  ["delete row", 4, () => [["delete"]]],
  ["enter", 3, () => [["submit"]]],
  ["taking the shadow", 2, () => [["take"]]],
  ["a word played", 5, (r) => [...word(r).map((letter) => ["type", letter]), ["submit"]]],
  ["an answer set", 2, (r) => [["answer", ANSWERS[Math.floor(r() * ANSWERS.length)]]]],
  ["the answer cleared", 1, () => [["answer", ""]]],
  ["an answer drawn", 1, () => [["draw"]]],
  ["played out", 1, () => [["playout"]]],
];

const word = (random) =>
  Array.from({ length: 5 }, () => LETTERS[Math.floor(random() * 26)]);

function apply([kind, a, b]) {
  switch (kind) {
    case "type": probe.typeLetter(a); return `type ${a}`;
    case "backspace": probe.backspace(); return "backspace";
    case "color": probe.cycle(); return "space";
    case "move": probe.moveTo(a, b); return `move to ${a},${b}`;
    case "delete": probe.deleteRow(); return "delete row";
    case "submit": probe.submit(); return "enter";
    case "take": probe.acceptSuggestion(); return "take the shadow";
    case "answer":
      probe.answerBox.value = a;
      probe.answerBox.handlers.change();
      return a ? `answer ${a}` : "answer cleared";
    case "draw": probe.drawAnswer(); return "draw an answer";
    case "playout": probe.playOut(); return "play it out";
  }
}

async function walk(seed, steps) {
  const random = rng(seed);
  const pick = () => {
    const total = MOVES.reduce((sum, [, weight]) => sum + weight, 0);
    let n = random() * total;
    for (const move of MOVES) {
      n -= move[1];
      if (n <= 0) return move;
    }
    return MOVES[0];
  };

  sent = [];
  probe.answerBox.value = "";
  await context.probe.applyState(boardState([]));
  const log = [];
  let queued = [];

  for (let step = 0; step < steps; step++) {
    if (!queued.length) {
      const [name, , expand] = pick();
      queued = expand(random);
      log.push(`[${name}]`);
    }
    log.push(apply(queued.shift()));
    await flush();   // anything the edit sent has landed before anything is judged

    for (const [name, holds] of Object.entries(PROPERTIES)) {
      const complaint = holds();
      if (complaint) {
        failures += 1;
        console.log(`FAIL  ${name}`);
        console.log(`        ${complaint}`);
        console.log(`        seed ${seed}, step ${step + 1}: ${log.join(" · ")}`);
        console.log(`        board ${JSON.stringify(drawn())}`);
        return false;
      }
    }
  }
  return true;
}

/* ----------------------------------------------------------------- anchors */

const play = (words) => probe.applyState(boardState(
  words.map((word) => ({ word, pattern: "NNNNN" }))));

async function anchors() {
  await flush();   // start() polls the status, then deals a board

  check("a fresh board shadows the opening word in row 1",
        drawn(), ["(salet)", "", "", "", "", ""]);

  play(["crane", "moist", "bluey"]);
  check("three played rows shadow the fourth",
        drawn(), ["crane", "moist", "bluey", "(might)", "", ""]);
  check("the free row is the fourth", probe.openRow(), 3);

  sent = [];
  probe.deleteRow();   // the cursor sits on the shadow, so this takes row 3
  await flush();
  check("deleting the last word asks the solver about what is left",
        sent, [[{ word: "crane", pattern: [0, 0, 0, 0, 0] },
                { word: "moist", pattern: [0, 0, 0, 0, 0] }]]);
  check("and the shadow that comes up is the pick made for that row",
        drawn(), ["crane", "moist", "(dumpy)", "", "", ""]);
  check("the free row is the third", probe.openRow(), 2);

  /* the point of the whole thing: two words left by a deletion are two words */
  const deleted = snapshot();
  play(["crane", "moist"]);
  check("a board left by deleting the third row is the two-row board itself",
        deleted, snapshot());

  play(["crane", "moist", "bluey"]);
  probe.deleteRow();
  await flush();
  probe.deleteRow();
  await flush();
  check("deleting again reduces the board to one row",
        drawn(), ["crane", "(crony)", "", "", "", ""]);

  /* a shadow is not a row of its own: deleting works down through the real
     words and stops at an empty board, which the opening shadow sits on */
  play(["crane", "moist"]);
  for (let i = 0; i < 3; i++) { probe.deleteRow(); await flush(); }
  check("deleting past the first row leaves the opening board",
        drawn(), ["(salet)", "", "", "", "", ""]);

  /* taking the shadow makes a word, and a word is a row like any other -- but
     one the solver was never told about, so putting it back asks for nothing */
  play(["crane", "moist", "bluey"]);
  probe.acceptSuggestion();
  check("taking the shadow writes it into the row it was shadowing",
        drawn(), ["crane", "moist", "bluey", "might", "", ""]);
  sent = [];
  probe.deleteRow();
  await flush();
  check("deleting a word that was never submitted shadows it again",
        drawn(), ["crane", "moist", "bluey", "(might)", "", ""]);
  check("and asks the solver nothing, the board being what it already answered",
        sent, []);

  /* a gap in the middle is left where it is; the shadow belongs after the last
     real word, not in the hole */
  play(["crane", "moist", "bluey"]);
  probe.moveTo(1, 0);
  sent = [];
  probe.deleteRow();
  await flush();
  check("a gap in the middle keeps the shadow after the last word",
        drawn(), ["crane", "", "bluey", "(might)", "", ""]);
  check("and is not sent: a board with a hole in it is not one to answer",
        sent, []);

  /* a word typed over the gap's row, then taken off by hand: the board that
     leaves is one word, and has to read as one word */
  play(["crane", "moist"]);
  const type = (word) => [...word].forEach((letter) => probe.typeLetter(letter));
  type("bluey");
  check("a word typed into the third row hides the shadow",
        drawn(), ["crane", "moist", "bluey", "", "", ""]);

  probe.moveTo(1, 0);
  probe.deleteRow();
  await flush();
  check("deleting the second row under it leaves the hole",
        drawn(), ["crane", "", "bluey", "", "", ""]);

  sent = [];
  probe.moveTo(2, 4);
  for (let i = 0; i < 5; i++) probe.backspace();
  await flush();
  check("backspacing that word out asks about the one word left",
        sent, [[{ word: "crane", pattern: [0, 0, 0, 0, 0] }]]);
  check("and the board is that one word and its shadow, with no row below",
        drawn(), ["crane", "(crony)", "", "", "", ""]);
  check("the cursor is on the shadow, which is where typing goes",
        probe.cursor, { row: 1, col: 0, atEnd: false });

  const backspaced = snapshot();
  play(["crane"]);
  check("all of which is the one-word board itself", backspaced, snapshot());
}

/* -------------------------------------------------------------------- run */

async function main() {
  const seeds = Number(process.argv[2] || 200);
  const steps = Number(process.argv[3] || 60);

  await anchors();

  console.log(`\n${seeds} walks of ${steps} edits, against ` +
              `${Object.keys(PROPERTIES).length} properties`);
  for (let seed = 1; seed <= seeds; seed++) {
    if (!(await walk(seed, steps))) break;
  }
  if (!failures) console.log(`ok    every property held`);

  console.log(failures ? `\n${failures} failed` : "\nall passed");
  process.exitCode = failures ? 1 : 0;
}

main();

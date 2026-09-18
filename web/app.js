/* The page: a board you fill in, and whatever the solver makes of it.
 *
 * All the thinking happens on the server.  The page keeps an editable copy of
 * the board -- every row, not just the one being typed, since an earlier row
 * can be corrected in place -- and sends the whole thing back whenever it is
 * submitted.  What comes back is the board as the solver accepted it, and the
 * editable copy is rebuilt from that.  The two cannot drift apart.
 */

const MAX_GUESSES = 6;
const TILES = "NYG";

const $ = (id) => document.getElementById(id);
const board = $("board");

let view = "solve";                // "solve" (the assistant), "play" (a game) or "finder"
let state = null;                  // the last board the server accepted
let rows = [];                     // the editable copy: [{letters, pattern}]
let busy = false;
let started = false;               // the first board has landed and the veil is down
let pending = null;                // one request held back while another runs

/* Set when a solver setting changes while the board is not showing; it is
   re-run over the rows it already has when it is next looked at. */
let staleBoard = false;
/* The answer box is the one setting that cannot be applied to a game in
   progress, so it is the only one that deals again. */
let needNewGame = false;

/* play mode: the answer stays on the server, so all the page has is the rows
   it has been told about and whatever it has asked the solver */
let game = null;
let draft = { letters: ["", "", "", "", ""], col: 0, atEnd: false };
let hint = null;

/* the finder, which runs the other way round: an answer that is known, a
   pattern to match, and the words that could have produced it */
let finder = { letters: ["", "", "", "", ""], pattern: [0, 0, 0, 0, 0],
               row: "answer", col: 0 };
let found = null;

/* Where the keys land.  ``atEnd`` marks the cursor as sitting past the last
   letter rather than on it, which is the difference between typing to correct
   the fifth tile and typing to start the word again. */
let cursor = { row: 0, col: 0, atEnd: false };

const emptyRow = () => ({ letters: ["", "", "", "", ""], pattern: [0, 0, 0, 0, 0] });

/* ------------------------------------------------------------------- api */

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const payload = await response.json().catch(() => ({ error: "Bad reply from the server" }));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

/** One request at a time; a second made meanwhile is held, not dropped.
    ``describe`` runs once the page is idle, since it may itself ask for
    something.  Returns whether the request went through. */
async function request(path, body, apply, describe) {
  if (busy) {
    pending = { path, body, apply, describe };
    return false;
  }
  busy = true;
  renderLocal();
  let ok = false;
  try {
    apply(await api(path, body));
    ok = true;
  } catch (err) {
    say(err.message, "bad");
  } finally {
    busy = false;
    renderLocal();
  }
  if (ok) {
    if (describe) describe();
    else clearMessage();
  }
  if (pending) {
    const next = pending;
    pending = null;
    await request(next.path, next.body, next.apply, next.describe);
  }
  return ok;
}

const send = (path, body, describe) => request(path, body, applyState, describe);

/* --------------------------------------------------------------- the board */

function applyState(next) {
  state = next;
  rows = state.rows.map((row) => ({
    letters: row.word.split(""),
    pattern: row.pattern.split("").map((tile) => TILES.indexOf(tile)),
  }));
  if (!state.over && rows.length < MAX_GUESSES) rows.push(emptyRow());
  cursor = { row: Math.max(0, rows.length - 1), col: 0, atEnd: false };
  markFittedForm();
  renderPanel();
  renderKeyboard();   // depends only on the rows the solver has accepted
  renderLocal();
}

const rowFilled = (row) => row.letters.filter(Boolean).length;
const rowWord = (row) => row.letters.join("");

/** The last row with anything written in it, or -1 on an empty board. */
const lastFilledRow = () =>
  rows.reduce((last, row, r) => (rowFilled(row) ? r : last), -1);

/** Where the suggestion goes: the row after the last real word, or -1 when
    there is no room for one.  It is a shadow rather than a row of its own, so
    deleting the word above it brings it up instead of stranding it below. */
const openRow = () => {
  const next = lastFilledRow() + 1;
  return next < rows.length ? next : -1;
};

/** Rows the server has not seen in their present state. */
function dirtyRows() {
  const dirty = new Set();
  for (let r = 0; r < rows.length; r++) {
    const sent = state && state.rows[r];
    if (!sent) {
      if (rowFilled(rows[r])) dirty.add(r);
      continue;
    }
    const pattern = rows[r].pattern.map((p) => TILES[p]).join("");
    if (rowWord(rows[r]) !== sent.word || pattern !== sent.pattern) dirty.add(r);
  }
  return dirty;
}

function suggestion() {
  return state && state.recommendation ? state.recommendation.word : "";
}

/* ---------------------------------------------------------------- editing */

/* A color belongs to the letter under it, so clearing one clears both. */
function clearTile(row, col) {
  row.letters[col] = "";
  row.pattern[col] = 0;
}

function moveTo(row, col) {
  if (!rows.length) return;
  cursor = {
    row: Math.min(Math.max(row, 0), rows.length - 1),
    col: Math.min(Math.max(col, 0), 4),
    atEnd: false,   // moving there means editing that tile, not appending
  };
  renderLocal();
}

function typeLetter(letter) {
  if (busy || !rows.length) return;
  const row = rows[cursor.row];

  // at the end of a full word this replaces the last letter, as any full box
  // does; a changed letter loses the color that described the old one
  if (row.letters[cursor.col] !== letter) row.pattern[cursor.col] = 0;
  row.letters[cursor.col] = letter;
  if (cursor.col < 4) cursor.col += 1;
  else cursor.atEnd = true;
  clearMessage();
  renderLocal();
}

function backspace() {
  if (busy || !rows.length) return;
  const row = rows[cursor.row];
  if (cursor.atEnd) clearTile(row, 4);
  else if (row.letters[cursor.col]) clearTile(row, cursor.col);
  else if (cursor.col > 0) {
    cursor.col -= 1;
    clearTile(row, cursor.col);
  }
  // leftwards only, so the first letters can go without the last ones
  cursor.atEnd = false;
  clearMessage();
  trimRows();
  renderLocal();
  refreshAfterRemoval();   // the last letter of a word is a word coming off
}

/** gray -> yellow -> green -> gray, on whichever tile is selected. */
function cycle(row = cursor.row, col = cursor.col) {
  if (busy || !rows.length) return;
  cursor = { row, col, atEnd: false };
  // an empty tile has no color to change
  if (!rows[row].letters[col]) return renderLocal();
  rows[row].pattern[col] = (rows[row].pattern[col] + 1) % 3;
  clearMessage();
  renderLocal();
  const tile = board.children[row].children[col];
  tile.classList.add("pop");
  setTimeout(() => tile.classList.remove("pop"), 90);
}

function fill(word) {
  if (busy || !rows.length) return;
  const row = rows[cursor.row];
  row.letters = word.toLowerCase().split("");
  row.pattern = [0, 0, 0, 0, 0];   // the colors described the word being replaced
  cursor.col = 0;
  cursor.atEnd = false;
  clearMessage();
  renderLocal();
}

/** The suggestion always goes in the row being played, wherever the cursor
    happens to have wandered off to. */
function acceptSuggestion() {
  const open = openRow();
  if (!suggestion() || open < 0) return;
  cursor = { row: open, col: 0, atEnd: false };
  fill(suggestion());
}

/** Send every row that has been written in.  A row is all or nothing. */
function submit() {
  if (busy || !state) return;
  const dirty = dirtyRows();
  const last = rows[rows.length - 1];
  if (dirty.size === 0) {
    if (state.over) return;   // nothing left to send, and nothing to suggest
    if (last && rowFilled(last) === 0) return acceptSuggestion();
  }

  const payload = [];
  for (let r = 0; r < rows.length; r++) {
    const filled = rowFilled(rows[r]);
    // a gap with words under it means a word was played there, so it has to be
    // filled in; an empty row with nothing under it is just not a guess
    if (filled === 0) {
      if (rows.some((row, below) => below > r && rowFilled(row))) {
        return say(`Row ${r + 1} is empty`, "bad");
      }
      continue;
    }
    if (filled < 5) return say(`Row ${r + 1} needs all five letters`, "bad");
    payload.push({ word: rowWord(rows[r]), pattern: rows[r].pattern });
  }
  // an emptied board is still worth sending while the solver holds rows
  if (payload.length === 0 && state.rows.length === 0) return;

  send("/api/board", { ...settings(), session: state.session, rows: payload }, () => {
    if (state.solved) say(`Solved in ${state.rows.length}.`, "good");
    else if (state.over) say("Out of guesses.", "bad");
    else clearMessage();
  });
}

/** Re-run the solver over the board it has already accepted, under whatever the
    switches now say.  The rows sent are the server's own, not the local copy,
    so a word half typed into the open row is neither submitted nor lost. */
function resend() {
  staleBoard = false;
  if (!state) return newBoard();
  const payload = state.rows.map((row) => ({ word: row.word, pattern: row.pattern }));
  const typing = rows[state.rows.length];
  const at = cursor;
  return send("/api/board", { ...settings(), session: state.session, rows: payload },
              () => {
    if (typing && rows.length > state.rows.length) {
      rows[state.rows.length] = typing;
      cursor = at;
    }
    clearMessage();
    renderLocal();
  });
}

/** One free row after the last word, and never fewer rows than the server holds.
    A row emptied by hand would otherwise leave the free row stranded below it,
    selectable and past the end of the board the shadow now describes. */
function trimRows() {
  const keep = Math.max(state ? state.rows.length : 0, lastFilledRow() + 2);
  if (keep >= rows.length) return;
  rows.length = keep;
  cursor = { row: Math.min(cursor.row, rows.length - 1), col: cursor.col,
             atEnd: false };
}

/** Taking a word off the board is a retraction rather than a proposal, so it
    goes back to the solver at once instead of waiting for Enter: what is left
    has to be the board those words alone would have made, shadow and panel
    included.  A hole still open in the middle waits, that being a board the
    solver refuses anyway. */
function refreshAfterRemoval() {
  if (busy || !state) return;
  const words = rows.filter((row) => rowFilled(row)).length;
  if (words >= state.rows.length) return;   // nothing the solver knows of is gone
  const played = lastFilledRow();
  if (rows.some((row, r) => r < played && rowFilled(row) === 0)) return;
  submit();
}

/** Empty the row the cursor is on, leaving its place free.  On the row being
    typed it means the last filled one. */
function deleteRow() {
  if (busy || !rows.length) return;
  const played = lastFilledRow();
  if (played < 0) return;   // only the shadow is showing, and it is not a row
  const target = Math.min(cursor.row, played);
  if (rowFilled(rows[target]) === 0) return;   // already a gap

  rows[target] = emptyRow();
  cursor = { row: target, col: 0, atEnd: false };
  clearMessage();
  trimRows();
  renderLocal();
  refreshAfterRemoval();
}

/* -------------------------------------------------------------- messages */

function say(text, kind) {
  const node = $("message");
  node.textContent = text;
  node.className = "message" + (kind ? " " + kind : "");
  node.hidden = false;
}

function clearMessage() {
  $("message").hidden = true;
}

/* --------------------------------------------------------------- drawing */

function makeTile() {
  const tile = document.createElement("button");
  tile.type = "button";
  tile.className = "tile";
  tile.tabIndex = -1;
  return tile;
}

function renderBoard() {
  if (view === "play") return renderPlayBoard();
  if (view === "finder") return renderFinderBoard();
  return renderSolveBoard();
}

function renderSolveBoard() {
  board.textContent = "";
  const dirty = dirtyRows();
  const open = openRow();
  const ghost = open >= 0 ? suggestion() : "";
  const ghostRow = ghost ? open : -1;

  for (let r = 0; r < MAX_GUESSES; r++) {
    const row = document.createElement("div");
    row.className = "row";
    const live = r < rows.length;
    if (live) row.classList.add("editable");
    if (dirty.has(r)) row.classList.add("dirty");

    for (let c = 0; c < 5; c++) {
      const tile = makeTile();
      if (live) {
        const letter = rows[r].letters[c];
        if (letter) {
          tile.textContent = letter;
          tile.classList.add("filled", "t-" + TILES[rows[r].pattern[c]]);
          tile.title = "Click to change the color";
          tile.addEventListener("click", () => cycle(r, c));
        } else if (r === ghostRow) {
          tile.textContent = ghost[c];
          tile.classList.add("ghost");
          tile.title = "The solver's suggestion; press Enter to take it";
          tile.addEventListener("click", acceptSuggestion);
        } else {
          tile.addEventListener("click", () => moveTo(r, c));
        }
        if (r === cursor.row && c === cursor.col) tile.classList.add("cursor");
      }
      row.appendChild(tile);
    }
    board.appendChild(row);
  }
}

/* In play mode the colors are the server's to give, so a row is either one it
   has scored or the one being typed -- there is nothing to click. */
function renderPlayBoard() {
  board.textContent = "";
  const played = game ? game.rows.length : 0;
  const typing = game && !game.over ? played : -1;
  const ghost = hint && draft.letters.every((letter) => !letter) ? hint.word : "";

  for (let r = 0; r < MAX_GUESSES; r++) {
    const row = document.createElement("div");
    row.className = "row";

    for (let c = 0; c < 5; c++) {
      const tile = makeTile();
      if (r < played) {
        tile.textContent = game.rows[r].word[c];
        tile.classList.add("t-" + game.rows[r].pattern[c]);
      } else if (r === typing) {
        const letter = draft.letters[c];
        const place = () => { draft.col = c; draft.atEnd = false; renderLocal(); };
        if (letter) {
          tile.textContent = letter;
          tile.classList.add("filled");
          tile.addEventListener("click", place);
        } else if (ghost) {
          tile.textContent = ghost[c];
          tile.classList.add("ghost");
          tile.title = "The solver's pick; press Enter to take it";
          tile.addEventListener("click", () => takeHint(c));
        } else {
          tile.addEventListener("click", place);
        }
        if (c === draft.col) tile.classList.add("cursor");
      }
      row.appendChild(tile);
    }
    board.appendChild(row);
  }
}

/* Two rows that do not describe each other: the letters are the answer, and the
   colors are a pattern some other word produced against it.  There is no ghost
   and nothing is grayed out, since a gray tile is a pattern rather than a gap. */
function renderFinderBoard() {
  const answer = $("finder-answer");
  const pattern = $("finder-pattern");
  answer.textContent = "";
  pattern.textContent = "";

  for (let c = 0; c < 5; c++) {
    const letter = makeTile();
    if (finder.letters[c]) {
      letter.textContent = finder.letters[c];
      letter.classList.add("filled");
    }
    letter.title = "Click to type here";
    letter.addEventListener("click", () => finderMoveTo("answer", c));
    if (finder.row === "answer" && finder.col === c) letter.classList.add("cursor");
    answer.appendChild(letter);

    const tile = makeTile();
    tile.classList.add("t-" + TILES[finder.pattern[c]]);
    tile.title = "Click to change the color";
    tile.addEventListener("click", () => finderCycle(c));
    if (finder.row === "pattern" && finder.col === c) tile.classList.add("cursor");
    pattern.appendChild(tile);
  }
}

function finderMoveTo(row, col) {
  finder.row = row;
  finder.col = Math.min(Math.max(col, 0), 4);
  renderLocal();
}

/** A letter is always a letter of the answer, wherever the cursor had got to. */
function finderType(letter) {
  if (busy) return;
  finder.row = "answer";
  finder.letters[finder.col] = letter;
  if (finder.col < 4) finder.col += 1;
  clearMessage();
  renderLocal();
}

function finderBackspace() {
  if (busy) return;
  finder.row = "answer";
  if (finder.letters[finder.col]) finder.letters[finder.col] = "";
  else if (finder.col > 0) {
    finder.col -= 1;
    finder.letters[finder.col] = "";
  }
  clearMessage();
  renderLocal();
}

/** gray -> yellow -> green -> gray, on whichever pattern tile is selected. */
function finderCycle(col = finder.col) {
  if (busy) return;
  finder.row = "pattern";
  finder.col = col;
  finder.pattern[col] = (finder.pattern[col] + 1) % 3;
  clearMessage();
  renderLocal();
  const tile = $("finder-pattern").children[col];
  tile.classList.add("pop");
  setTimeout(() => tile.classList.remove("pop"), 90);
}

function clearPattern() {
  if (busy) return;
  finder.pattern = [0, 0, 0, 0, 0];
  clearMessage();
  renderLocal();
}

function findWords() {
  if (busy) return;
  if (!finder.letters.every(Boolean)) return say("The answer needs all five letters", "bad");
  request("/api/find", {
    answer: finder.letters.join(""),
    pattern: finder.pattern,
    sort: $("opt-sort").value,
    // only the priors reach this far: they shape the two columns that are
    // averages over every answer, not the pattern match itself
    repeat: $("opt-repeat").checked,
    curation: $("opt-curation").checked,
  }, (payload) => {
    found = payload;
    renderFinderPanel();
    renderLocal();
  }, () => {
    if (found.total === 0) say("No word gives that pattern against that answer.", "bad");
    else clearMessage();
  });
}

const FIND_VALUE = {
  left: (word) => `${word.left.toLocaleString()} left`,
  bits: (word) => `${word.bits.toFixed(2)} bits`,
  weight: (word) => `${(word.weight * 100).toFixed(3)}%`,
};

function renderFinderPanel() {
  const list = $("found");
  list.textContent = "";
  if (!found) {
    $("find-count").textContent = "—";
    $("find-sub").textContent = "Type the answer, color the pattern the guess came back as, and press Find words.";
    return;
  }

  $("find-count").textContent = found.total.toLocaleString();
  const answer = found.answer.toUpperCase();
  if (found.total === 0) {
    $("find-sub").textContent = `No word gives ${found.pattern} against ${answer}.`;
    return;
  }
  $("find-sub").textContent = found.total > found.shown
    ? `${found.total.toLocaleString()} words give ${found.pattern} against ${answer}; the best ${found.shown} by this order are listed.`
    : `${found.total.toLocaleString()} word${found.total === 1 ? "" : "s"} give ${found.pattern} against ${answer}.`;

  const value = FIND_VALUE[found.sort] || FIND_VALUE.left;
  for (const entry of found.words) {
    const item = document.createElement("li");
    const word = document.createElement("span");
    word.className = "word";
    word.textContent = entry.word;
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = value(entry);
    item.append(word, score);
    item.title = `${entry.left.toLocaleString()} words left · ${entry.bits.toFixed(2)} bits · `
      + `${(entry.weight * 100).toFixed(3)}% chance of being the answer`;
    list.appendChild(item);
  }
}

function playType(letter) {
  if (busy || !game || game.over) return;
  // as in the solver: at the end of a finished word, the last letter is the
  // one that gets replaced, rather than the whole word being thrown away
  draft.letters[draft.col] = letter;
  if (draft.col < 4) draft.col += 1;
  else draft.atEnd = true;
  clearMessage();
  renderLocal();
}

function playBackspace() {
  if (busy || !game || game.over) return;
  if (draft.atEnd) draft.letters[4] = "";
  else if (draft.letters[draft.col]) draft.letters[draft.col] = "";
  else if (draft.col > 0) {
    draft.col -= 1;
    draft.letters[draft.col] = "";
  }
  draft.atEnd = false;
  clearMessage();
  renderLocal();
}

function clearDraft() {
  draft = { letters: ["", "", "", "", ""], col: 0, atEnd: false };
}

/** Put a word into the row being typed, ready to be guessed or edited. */
function playFill(word, col = 0) {
  draft = { letters: word.split(""), col, atEnd: false };
  clearMessage();
  renderLocal();
}

/** Take the solver's pick; a clicked ghost tile says where to carry on from. */
function takeHint(col = 0) {
  if (!hint) return;
  playFill(hint.word, col);
}

function applyGame(next) {
  game = next;
  clearDraft();
  hint = null;
  renderPlayPanel();
  renderKeyboard();
  renderLocal();
}

function playGuess() {
  if (busy || !game || game.over) return;

  // enter on an empty row plays the solver's pick outright, as answering the
  // terminal game's prompt with a bare enter does
  if (draft.letters.every((letter) => !letter)) {
    if (!hint) return;
    takeHint();
  }
  if (!draft.letters.every(Boolean)) return say("A guess is five letters", "bad");

  request("/api/game/guess", { session: game.session, word: draft.letters.join("") },
          applyGame, () => {
    if (game.solved) say(`Solved in ${game.rows.length}.`, "good");
    else if (game.over) say(`Out of guesses. The answer was ${game.answer.toUpperCase()}.`, "bad");
    else {
      clearMessage();
      if ($("opt-assist").checked) askHint();
    }
  });
}

function askHint() {
  if (busy || !game || game.over) return;
  // every setting goes with the request rather than being fixed when the game
  // was dealt, priors included, so changing one just re-asks.  The answer was
  // drawn under whatever was on at the time and stays what it is.
  request("/api/game/hint", { ...settings(), session: game.session }, (payload) => {
    hint = payload;
    renderPlayPanel();
    renderLocal();
  });
}

/** The hint button is a toggle: press it again and the answer goes away. */
function toggleHint() {
  // pressing Hint means you want to see the pick, so open the card whether or
  // not assist would have
  if (!hint) {
    $("hint-card").open = true;
    return askHint();
  }
  hint = null;
  renderPlayPanel();
  renderLocal();
}

function giveUp() {
  if (busy || !game || game.over) return;
  request("/api/game/reveal", { session: game.session }, applyGame,
          () => say(`The answer was ${game.answer.toUpperCase()}.`, "bad"));
}

function newGame() {
  needNewGame = false;
  return request("/api/game/new",
                 { ...settings(), answer: $("opt-answer").value.trim() },
                 applyGame,
                 // assist covers the first guess too, not just the ones after
                 () => { if ($("opt-assist").checked) askHint(); });
}

function renderPlayPanel() {
  if (!game) return;
  $("game-turn").textContent = game.over
    ? `${game.rows.length} / ${game.max_guesses}`
    : `${game.turn} / ${game.max_guesses}`;

  const headline = $("game-headline");
  if (game.solved) headline.textContent = `Solved in ${game.rows.length}`;
  else if (game.over) headline.textContent = game.answer;
  else headline.textContent = `${game.guesses_left} left`;

  $("game-source").textContent = game.over
    ? `The answer was ${game.answer.toUpperCase()} — ${game.source}.`
    : `The answer is ${game.source}.`;

  const pick = $("hint-card");
  pick.hidden = !hint;
  if (hint) {
    $("hint-word").textContent = hint.word;
    $("hint-score").textContent =
      `${hint.score.toFixed(4)} ${hint.score_label} · ` +
      `${hint.candidates_left.toLocaleString()} candidate${hint.candidates_left === 1 ? "" : "s"} left`;
    const tag = $("hint-tag");
    tag.hidden = false;
    tag.className = "tag" + (hint.exact ? " exact" : "");
    tag.textContent = hint.exact ? "Exact" : (hint.mode === "entropy" ? "Greedy" : "Estimated");
    if ($("opt-assist").checked) pick.open = true;
  }
  fillSameSplit($("hint-same-split"), hint, playFill);

  const alts = (hint && hint.alternatives) || [];
  const card = $("game-alt-card");
  card.hidden = alts.length === 0;
  // assist opens both, and nothing ever closes them: a card opened by hand with
  // assist off is still open after the next guess
  if (alts.length && $("opt-assist").checked) card.open = true;
  fillAlternatives($("game-alternatives"), alts, playFill);

  renderLetterOdds();

  $("share-card").hidden = !game.share;
  if (game.share) $("share-grid").textContent = game.share;
}

const VOWELS = "aeiou";

/** The letter odds, with the vowels in a row of their own. The server drops
    the letters no remaining candidate holds, and an empty list hides the
    card. */
function renderLetterOdds() {
  const letters = (game && game.letters) || [];
  $("letters-card").hidden = letters.length === 0;
  if (!letters.length) return;

  const vowels = letters.filter((entry) => VOWELS.includes(entry.letter));
  const rest = letters.filter((entry) => !VOWELS.includes(entry.letter));
  fillLetters("vowel", vowels);
  fillLetters("consonant", rest);
}

function fillLetters(kind, entries) {
  $(`${kind}s-head`).hidden = entries.length === 0;
  const list = $(`${kind}-letters`);
  list.hidden = entries.length === 0;
  list.textContent = "";
  for (const entry of entries) {
    const item = document.createElement("li");
    const tile = document.createElement("span");
    tile.className = "letter-tile";
    tile.textContent = entry.letter;
    const chance = document.createElement("span");
    chance.className = "letter-chance";
    // "0.0%" would read as impossible when the letter is merely a long shot
    chance.textContent = entry.chance < 0.001 ? "<0.1%" : RATE(entry.chance);
    item.append(tile, chance);
    list.appendChild(item);
  }
}

/** The ranked list under a recommendation, in either view. */
function fillAlternatives(list, alts, take) {
  list.textContent = "";
  for (const alt of alts) {
    const item = document.createElement("li");
    const word = document.createElement("span");
    word.className = "word";
    word.textContent = alt.word;
    const tail = document.createElement("span");
    tail.className = "row-tail";
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = fmtScore(alt.score);
    tail.append(score);
    item.append(word, tail);
    item.title = "Put this word in the selected row";
    item.addEventListener("click", () => take(alt.word));

    const chips = splitChips(alt, take);
    if (chips) {
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "split-toggle";
      toggle.textContent = `+${alt.same_split_count}`;
      toggle.title = splitLabel(alt.same_split_count, false);
      toggle.addEventListener("click", (event) => {
        event.stopPropagation();      // the row itself puts its word in the board
        chips.hidden = !chips.hidden;
        toggle.title = splitLabel(alt.same_split_count, !chips.hidden);
      });
      tail.prepend(toggle);
      item.append(chips);
    }
    list.appendChild(item);
  }
}

function splitLabel(count, open) {
  const many = count === 1 ? "word that plays" : "words that play";
  return `${open ? "Hide" : "Show"} the ${count} ${many} exactly the same`;
}

/** The words giving the same feedback as this guess against every candidate
    still standing, so the same game from here on. Null when it stands alone,
    which is most of the time: distinct splits are near-universal until the
    candidates are nearly gone. */
function splitChips(entry, take) {
  const words = entry.same_split || [];
  if (!words.length) return null;

  const chips = document.createElement("ul");
  chips.className = "chips split-chips";
  chips.hidden = true;
  for (const word of words) {
    const chip = document.createElement("li");
    chip.textContent = word;
    chip.title = "Put this word in the selected row";
    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      take(word);
    });
    chips.appendChild(chip);
  }
  const hidden = entry.same_split_count - words.length;
  if (hidden > 0) {
    const more = document.createElement("li");
    more.className = "more";
    more.textContent = `+${hidden.toLocaleString()} more`;
    chips.appendChild(more);
  }
  return chips;
}

/** The same list under a pick, where there is room to say what it is. */
function fillSameSplit(box, entry, take) {
  box.textContent = "";
  const chips = entry ? splitChips(entry, take) : null;
  box.hidden = !chips;
  if (!chips) return;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "ghost small";
  toggle.textContent = splitLabel(entry.same_split_count, false);
  toggle.addEventListener("click", () => {
    chips.hidden = !chips.hidden;
    toggle.textContent = splitLabel(entry.same_split_count, !chips.hidden);
    note.hidden = chips.hidden;
  });
  const note = document.createElement("p");
  note.className = "sub";
  note.hidden = true;
  note.textContent = "Identical feedback against every word still possible, so any of them plays the same game from here.";
  box.append(toggle, chips, note);
}

const KEY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];

function letterStates() {
  if (view === "finder") return {};   // there is no board behind the finder
  const rank = { N: 0, Y: 1, G: 2 };
  const best = {};
  const source = view === "play" ? (game ? game.rows : []) : (state ? state.rows : []);
  for (const row of source) {
    for (let i = 0; i < 5; i++) {
      const letter = row.word[i], tile = row.pattern[i];
      if (!(letter in best) || rank[tile] > rank[best[letter]]) best[letter] = tile;
    }
  }
  return best;
}

function renderKeyboard() {
  const node = $("keyboard");
  node.textContent = "";
  const best = letterStates();

  KEY_ROWS.forEach((letters, index) => {
    const row = document.createElement("div");
    row.className = "kb-row";
    if (index === 2) row.appendChild(makeKey("enter", "wide", onEnter));
    for (const letter of letters) {
      row.appendChild(makeKey(letter, best[letter] ? "t-" + best[letter] : "",
                              () => onLetter(letter)));
    }
    if (index === 2) row.appendChild(makeKey("del", "wide", onBackspace));
    node.appendChild(row);
  });
}

function makeKey(label, className, onClick) {
  const key = document.createElement("button");
  key.type = "button";
  key.className = "key " + className;
  key.textContent = label;
  key.tabIndex = -1;
  key.addEventListener("click", (event) => { event.currentTarget.blur(); onClick(); });
  return key;
}

function renderHintLine() {
  const line = $("hint");
  if (busy) return void (line.textContent = "Thinking…");

  if (view === "finder") {
    line.textContent = finder.letters.every(Boolean)
      ? "Space colors the selected tile · Enter finds every word that fits"
      : "Type the answer, then color the pattern the guess came back as";
    return;
  }

  if (view === "play") {
    if (!game) return void (line.textContent = " ");
    if (game.over) {
      line.textContent = game.solved
        ? `Solved in ${game.rows.length}. Start a new game to play again.`
        : `The answer was ${game.answer.toUpperCase()}. Start a new game to play again.`;
    } else if (draft.letters.every((letter) => !letter) && hint) {
      line.textContent = "Enter takes the solver's pick, or type your own";
    } else {
      line.textContent = "Type a word and press Enter";
    }
    return;
  }

  if (!state) return;
  if (state.over) return void (line.textContent = "Start a new board to solve another one.");

  const dirty = dirtyRows();
  const here = rows[cursor.row];
  if (dirty.size) {
    line.textContent = dirty.size === 1 && dirty.has(rows.length - 1)
      ? "Space colors the selected tile · Enter submits"
      : "Edited — Enter re-runs the solver from the first changed row";
  } else if (!here || rowFilled(here) === 0) {
    line.textContent = "Type the word you played";
  } else {
    line.textContent = "Space colors the selected tile, enter submits the word";
  }
}

function renderLocal() {
  renderBoard();
  renderHintLine();

  if (view === "finder") {
    $("find").disabled = busy || !finder.letters.every(Boolean);
    $("finder-clear").disabled = busy || finder.pattern.every((tile) => tile === 0);
    return;
  }

  if (view === "play") {
    const live = game && !game.over && !busy;
    const typed = draft.letters.filter(Boolean).length;
    $("guess").disabled = !live || (typed === 0 && !hint);
    $("guess").textContent = typed === 0 && hint ? "Play suggestion" : "Guess";
    $("hint-btn").disabled = !live;
    $("hint-btn").textContent = hint ? "Hide hint" : "Hint";
    $("giveup").disabled = !live;
    return;
  }

  // a finished board can still be corrected, so the button offers the edit
  const edited = dirtyRows().size > 0;
  const ready = rows.length > 0 && rowFilled(rows[rows.length - 1]) === 5;
  const canSubmit = edited || Boolean(state && !state.over && ready);
  const canTake = Boolean(state && !state.over && !edited && !ready && suggestion());
  $("accept").disabled = busy || !state || !(canSubmit || canTake);
  $("accept").textContent = canSubmit ? "Submit" : "Use suggestion";
  // it clears whatever row has something in it, played or still being typed
  $("undo").disabled = busy || !rows.some((row) => rowFilled(row) > 0);
}

/* ----------------------------------------------------------------- panel */

const fmtScore = (score) => score.toFixed(4);

function renderPanel() {
  if (!state) return;
  const rec = state.recommendation;

  if (state.solved) {
    $("rec-word").textContent = "Solved";
    $("rec-score").textContent =
      `In ${state.rows.length} guess${state.rows.length === 1 ? "" : "es"}`;
    $("rec-note").hidden = true;
    $("rec-tag").hidden = true;
  } else if (state.over) {
    $("rec-word").textContent = "Out of guesses";
    $("rec-score").textContent =
      `${state.candidates_left} word${state.candidates_left === 1 ? "" : "s"} were still possible`;
    $("rec-note").hidden = true;
    $("rec-tag").hidden = true;
  } else if (rec) {
    $("rec-word").textContent = rec.word;
    let line = `${fmtScore(rec.score)} ${state.score_label}`;
    if (state.elapsed_ms >= 200) line += `  ·  ${(state.elapsed_ms / 1000).toFixed(1)} s`;
    $("rec-score").textContent = line;

    const tag = $("rec-tag");
    tag.hidden = false;
    tag.className = "tag" + (rec.exact ? " exact" : "");
    tag.textContent = rec.exact ? "Exact" : (state.mode === "entropy" ? "Greedy" : "Estimated");

    let note = "";
    if (state.candidates_left === 1) note = "The only word left.";
    else if (!rec.possible) note = "Cannot be the answer itself. It is played purely to split what is left.";
    else if (state.guesses_left === 1) note = `Last guess: a ${(rec.win_probability * 100).toFixed(1)}% chance of being right.`;
    else if (rec.win_probability > 0) note = `A ${(rec.win_probability * 100).toFixed(1)}% chance of being the answer outright.`;
    $("rec-note").textContent = note;
    $("rec-note").hidden = !note;
  }

  fillSameSplit($("rec-same-split"), state.over ? null : rec, fill);

  const alts = state.alternatives || [];
  $("alt-card").hidden = alts.length === 0;
  fillAlternatives($("alternatives"), alts, fill);

  $("cand-count").textContent = state.candidates_left.toLocaleString();
  $("cand-sub").textContent = state.effective_size !== undefined
    ? `Effective size ${state.effective_size.toFixed(1)} once the prior is taken into account`
    : "";

  const chips = $("candidates");
  chips.textContent = "";
  for (const word of state.possible_answers) {
    const chip = document.createElement("li");
    chip.textContent = word;
    chip.title = "Put this word in the selected row";
    chip.addEventListener("click", () => fill(word));
    chips.appendChild(chip);
  }
  const hidden = state.candidates_left - state.possible_answers.length;
  if (hidden > 0) {
    const more = document.createElement("li");
    more.className = "more";
    more.textContent = `+${hidden.toLocaleString()} more`;
    chips.appendChild(more);
  }

  renderTechnical();
}

/* --------------------------------------------- technical details, expanded */

const PERCENT = (p) => `${(p * 100).toFixed(0)}%`;
const COUNT = (n) => n.toLocaleString();

/* a measured rate, which is worth a decimal the round prior probabilities are not */
const RATE = (p) => `${(p * 100).toFixed(1)}%`;

/* how many times the regime rate is the whole record's */
const TIMES = (recent, ever) => (ever > 0 ? Math.round(recent / ever) : 1);

function repeatFact(prior) {
  const m = prior.measured;
  if (!prior.repeat) {
    return {
      key: "repeat",
      name: "Repeat prior",
      value: "Off",
      why: `<p>Off: a word that has already been an answer is treated as no less
              likely to come up again than one that never has. Every word competes
              on equal terms, subject to the curation prior when that is on.</p>`,
    };
  }
  const note = prior.repeat_shape === "binned"
    ? `That mass is not spread evenly over them: each is weighted by where it
       first appeared, so the early answers nearly every repeat has come from
       carry more of it than the recent ones.`
    : `Every used word carries the same share of that mass, wherever in the run
       it first appeared.`;
  return {
    key: "repeat",
    name: "Repeat prior",
    value: `On: p = ${prior.repeat_prob.toFixed(2)}, ${prior.repeat_shape}`,
    why: `<p>The editor sometimes reuses a word that has already been an answer.
            With this on, the ${COUNT(m.used)} words already used share
            ${PERCENT(prior.repeat_prob)} of the probability between them, instead
            of each getting the same share as an unused word. ${note}</p>
          <p>The ${PERCENT(prior.repeat_prob)} is set for the current regime:
            repeats run at ${RATE(m.regime.repeat)} over the ${COUNT(m.puzzles)}
            puzzles since Wordle #${m.start_puzzle}, but ${RATE(m.all_time.repeat)}
            across the whole record, because nothing repeated at all before that.
            Judging it against all history would set it about
            ${TIMES(m.regime.repeat, m.all_time.repeat)} times too low.</p>`,
  };
}

function curationFact(prior) {
  const m = prior.measured;
  if (!prior.curation) {
    return {
      key: "curation",
      name: "Curation prior",
      value: "Off",
      why: `<p>Off: no distinction is drawn between the curated 2022 answer list
              and the rest of the valid words. All ${COUNT(m.words)} are equally
              eligible, subject to the repeat prior when that is on.</p>`,
    };
  }
  const note = prior.curation_shape === "uniform"
    ? `Every off-list word carries the same share of that mass.`
    : prior.curation_shape === "wordlebot"
      ? `That mass is not spread evenly over them: each is weighted by WordleBot's
         own published weight for it, with a little held back for the words
         WordleBot does not list at all.`
      : `That mass is not spread evenly over them: each is weighted by how
         answer-like the word looks, so the ordinary ones outrank the obscure.`;
  return {
    key: "curation",
    name: "Curation prior",
    value: `On: p = ${prior.noncurated_prob.toFixed(2)}, ${prior.curation_shape}`,
    why: `<p>Most answers are drawn from the curated 2022 list of
            ${COUNT(m.curated)} words. With this on, words outside that list
            share ${PERCENT(prior.noncurated_prob)} of whatever the repeat prior
            has <b>left</b> — all of it, when that prior is off. ${note}</p>
          <p>So it is a conditional probability — off-list <em>given not a
            repeat</em> — and has to be measured with that denominator. On that
            basis off-list answers run at ${RATE(m.regime.noncurated)} over the
            ${COUNT(m.puzzles)} puzzles since Wordle #${m.start_puzzle}, against
            ${RATE(m.all_time.noncurated)} across the whole record.</p>`,
  };
}

function priorFacts() {
  return [repeatFact(state.prior), curationFact(state.prior)];
}

/** Which form the empty choice resolves to. It is fitted per prior, so it
    moves when the toggles do, and no one form is the default. */
function markFittedForm() {
  const fitted = state.estimator && state.estimator.fitted;
  for (const option of $("opt-form").options) {
    if (!option.value) continue;
    option.textContent = option.value === fitted
      ? `${option.value} (fitted)` : option.value;
  }
}

function modeFact() {
  if (state.mode === "entropy") {
    return {
      key: "mode",
      name: "Mode",
      value: "Most information gained",
      why: `<p>Each guess is scored by the information its feedback is expected to
              reveal, in bits, over the feedback patterns <i>r</i> it can come
              back as. Higher is better.</p>
            <p class="equation"><i>H</i>(<i>g</i>) <span class="op">=</span>
              <span class="op">−</span> Σ<sub>r</sub> <i>p</i><sub>r</sub>
              <span class="op">·</span> log<sub>2</sub> <i>p</i><sub>r</sub></p>
            <p>The most informative guess is the one whose feedback is hardest to
              predict, because that is the one splitting the candidates most
              evenly. It is <b>greedy</b>: it never looks past the split it makes
              right now, which is why it is kept as a clean baseline rather than
              as the strong policy.</p>`,
    };
  }
  return {
    key: "mode",
    name: "Mode",
    value: "Fewest expected guesses",
    why: `<p>Each guess is scored by the total number of guesses it is expected to
            cost, itself included:</p>
          <p class="equation"><i>E</i>(<i>g</i>) <span class="op">=</span> 1
            <span class="op">+</span> Σ<sub>r</sub> <i>p</i><sub>r</sub>
            <span class="op">·</span> <i>f</i>(<i>n</i><sub>r</sub>)</p>
          <p>over every feedback pattern <i>r</i> except all-green.
            <i>p</i><sub>r</sub> is the chance the feedback comes back as <i>r</i>,
            <i>n</i><sub>r</sub> is the effective number of candidates still
            standing if it does, and <i>f</i> estimates how many further guesses
            that position needs. Lower is better.</p>
          <p>Where few enough candidates remain, <i>f</i> is not consulted at all:
            the guess comes from an exact search over the remaining words, and its
            score is a true value rather than an estimate. That is what the
            <b>exact</b> tag on a recommendation means.</p>`,
  };
}

/** 'sqrt(x)' and friends, as something worth reading. */
function prettyTerm(label) {
  const log2n = 'log<sub>2</sub><i>n</i>';
  const hinge = label.match(/^\(x-([\d.]+)\)\+$/);
  if (hinge) return `max(0, ${log2n} − ${hinge[1]})`;
  return {
    "1": "",
    "x": log2n,
    "x^2": `(${log2n})<sup>2</sup>`,
    "x^3": `(${log2n})<sup>3</sup>`,
    "1-1/n": "(1 − 1/<i>n</i>)",
    "log2(1+x)": `log<sub>2</sub>(1 + ${log2n})`,
    "sqrt(x)": `√(${log2n})`,
  }[label] ?? label;
}

function equationHTML(est) {
  const parts = [`<i>f</i>(<i>n</i>) <span class="op">=</span> `];
  if (est.base) parts.push(`1 <span class="op">+</span> `);

  est.terms.forEach((term, index) => {
    const magnitude = Math.abs(term.coefficient).toFixed(4);
    const sign = term.coefficient < 0 ? "−" : "+";
    if (index > 0 || est.base) parts.push(`<span class="op">${sign}</span> `);
    else if (sign === "−") parts.push(`<span class="op">−</span> `);

    const symbol = prettyTerm(term.label);
    parts.push(`<span class="coef">${magnitude}</span>`);
    if (symbol) parts.push(` <span class="op">·</span> <span class="var">${symbol}</span>`);
    parts.push(" ");
  });
  return `<div class="equation">${parts.join("")}</div>`;
}

function estimatorFact() {
  const est = state.estimator;
  if (!est) return null;
  const samples = est.samples;
  const table = `<table class="samples">
      <tr><th><i>n</i></th>${samples.map((s) => `<th>${s.n}</th>`).join("")}</tr>
      <tr><td><i>f</i>(<i>n</i>)</td>${samples.map((s) => `<td>${s.f.toFixed(2)}</td>`).join("")}</tr>
    </table>`;

  return {
    key: "estimator",
    name: "f(n)",
    value: est.form,
    why: `${equationHTML(est)} <p><i>f</i>(<i>n</i>) estimates how many guesses are still needed from a position whose candidate set has <b>effective size</b> <i>n</i> = 2<sup><i>H</i></sup>, where <i>H</i> is the entropy in bits of the candidates under the prior. With a uniform prior that is simply how many candidates are left; as the prior concentrates on a few of them it shrinks toward 1.</p> ${table} <p>The coefficients are fitted by policy iteration. We play many games with the current <i>f</i>, record what each position actually cost, refit, and repeat. <i>f</i> must be part of the <b>${est.form}</b> family (${est.description}), where x is log_2(n). The result is floored at 1, since one more guess is always needed.</p>`,
  };
}

function renderTechnical() {
  const body = $("tech-body");
  const wasOpen = new Set(
    [...body.querySelectorAll("details.fact[open]")].map((node) => node.dataset.key));
  body.textContent = "";

  const facts = [...priorFacts(), modeFact(), estimatorFact()].filter(Boolean);
  for (const fact of facts) {
    const details = document.createElement("details");
    details.className = "fact";
    details.dataset.key = fact.key;
    if (wasOpen.has(fact.key)) details.open = true;

    const summary = document.createElement("summary");
    const name = document.createElement("span");
    name.className = "fact-name";
    name.textContent = fact.name;
    const value = document.createElement("span");
    value.className = "fact-value";
    value.textContent = fact.value;
    const why = document.createElement("span");
    why.className = "why";
    why.textContent = "?";
    why.title = "What this means";
    summary.append(name, value, why);

    const explanation = document.createElement("div");
    explanation.className = "fact-why";
    explanation.innerHTML = fact.why;   // fixed copy, plus numbers from the server

    details.append(summary, explanation);
    body.appendChild(details);
  }
}

/* --------------------------------------------------------------- settings */

function settings() {
  return {
    repeat: $("opt-repeat").checked,
    curation: $("opt-curation").checked,
    mode: $("opt-mode").value,
    form: $("opt-form").value || null,
    hard: $("opt-hard").checked,
    top: 5,
  };
}

function newBoard() {
  staleBoard = false;
  return send("/api/new", settings());
}

/** Whichever of the three the page is showing, start it again. */
function restart() {
  if (view === "play") return newGame();
  if (view === "solve") return newBoard();
}

/* ------------------------------------------------------------------- views */

const VIEW_NOTE = {
  solve: `A setting changed here re-runs the solver over the board you already
          have; only New board clears it.`,
  play: `The answer was drawn under the priors that were on when it was dealt and
         stays what it is. Changing one now only changes what the hint plays.`,
  finder: `The two priors shape the bits and chance columns. Nothing else here
           reaches the finder.`,
};

function setView(next) {
  view = next;
  for (const button of document.querySelectorAll("#views .view-btn")) {
    const on = button.dataset.view === next;
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", String(on));
  }

  $("board").hidden = next === "finder";
  $("finder-board").hidden = next !== "finder";
  $("solve-panel").hidden = next !== "solve";
  $("play-panel").hidden = next !== "play";
  $("finder-panel").hidden = next !== "finder";
  $("solve-controls").hidden = next !== "solve";
  $("play-controls").hidden = next !== "play";
  $("finder-controls").hidden = next !== "finder";
  $("solve-keys").hidden = next !== "solve";
  $("play-keys").hidden = next !== "play";
  $("finder-keys").hidden = next !== "finder";
  $("answer-setting").hidden = next !== "play";
  // the finder never asks the solver for a guess, so nothing that shapes one is
  // shown alongside it
  for (const id of ["mode-setting", "form-setting", "hard-setting"]) {
    $(id).hidden = next === "finder";
  }
  $("reset").hidden = next === "finder";
  $("reset").textContent = next === "play" ? "New game" : "New board";
  $("settings-note").textContent = VIEW_NOTE[next];

  clearMessage();

  // drawn before anything is asked for, so the view is never left empty
  if (started) {
    if (next === "play") {
      if (!game || needNewGame) newGame();
      else renderPlayPanel();
    } else if (next === "finder") {
      renderFinderPanel();
    } else {
      if (staleBoard) resend();
      else renderPanel();
    }
  }
  renderKeyboard();
  renderLocal();
}

/* --------------------------------------------------- keys, whichever view */

const BY_VIEW = (actions) => () => actions[view]();

const onLetter = (letter) =>
  BY_VIEW({ play: () => playType(letter), finder: () => finderType(letter),
            solve: () => typeLetter(letter) })();
const onBackspace = BY_VIEW({ play: playBackspace, finder: finderBackspace,
                              solve: backspace });
const onEnter = BY_VIEW({ play: playGuess, finder: findWords, solve: submit });
const onSpace = BY_VIEW({ play: () => {}, finder: () => finderCycle(),
                          solve: () => cycle() });

function onArrow(dRow, dCol) {
  if (view === "finder") {
    if (dCol) finder.col = Math.min(Math.max(finder.col + dCol, 0), 4);
    if (dRow) finder.row = dRow > 0 ? "pattern" : "answer";
    return renderLocal();
  }
  if (view === "play") {
    if (dCol) {
      draft.col = Math.min(Math.max(draft.col + dCol, 0), 4);
      draft.atEnd = false;
      renderLocal();
    }
    return;   // there is nothing above the row you are on in a game
  }
  moveTo(cursor.row + dRow, cursor.col + dCol);
}

/* ------------------------------------------------------------------ start */

/* space and the arrows would scroll the page or press whatever button was last
   clicked, so they are taken over -- but only while nothing that legitimately
   wants them has focus. */
const KEEPS_KEYS = "input, select, textarea, summary, a, [contenteditable]";

function wire() {
  $("settings-toggle").addEventListener("click", (event) => {
    const panel = $("settings");
    panel.hidden = !panel.hidden;
    event.currentTarget.setAttribute("aria-expanded", String(!panel.hidden));
  });
  const onClick = (id, action) => $(id).addEventListener("click", (event) => {
    event.currentTarget.blur();
    action();
  });

  for (const button of document.querySelectorAll("#views .view-btn")) {
    button.addEventListener("click", (event) => {
      event.currentTarget.blur();
      setView(event.currentTarget.dataset.view);
    });
  }
  onClick("reset", restart);
  onClick("undo", deleteRow);
  // submit falls back to taking the suggestion when there is nothing to send
  onClick("accept", submit);
  onClick("guess", playGuess);
  onClick("hint-btn", toggleHint);
  onClick("giveup", giveUp);
  onClick("copy-share", copyShare);
  onClick("find", findWords);
  onClick("finder-clear", clearPattern);

  // every one of these changes what the solver is asked, not where you are, so
  // nothing is thrown away: the board re-runs over the rows it has, the game
  // keeps its answer, and the finder re-sorts what it found
  for (const id of ["opt-repeat", "opt-curation", "opt-mode", "opt-form", "opt-hard"]) {
    $(id).addEventListener("change", () => {
      staleBoard = true;
      if (view === "solve") return resend();
      if (view === "play" && hint) return askHint();
      if (view === "finder" && found) return findWords();
    });
  }
  // the one setting that cannot be applied to a game already in progress
  $("opt-answer").addEventListener("change", () => {
    needNewGame = true;
    if (view === "play") newGame();
  });
  $("opt-assist").addEventListener("change", (event) => {
    if (view !== "play") return;
    if (event.target.checked && !hint) return askHint();
    renderPlayPanel();   // assist also decides whether the alternatives start open
  });
  $("opt-sort").addEventListener("change", () => {
    if (found) findWords();
  });

  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.matches?.(KEEPS_KEYS)) return;

    switch (event.key) {
      case "Enter":      event.preventDefault(); onEnter(); break;
      case "Backspace":  event.preventDefault(); onBackspace(); break;
      case " ":          event.preventDefault(); onSpace(); break;
      case "ArrowLeft":  event.preventDefault(); onArrow(0, -1); break;
      case "ArrowRight": event.preventDefault(); onArrow(0, 1); break;
      case "ArrowUp":    event.preventDefault(); onArrow(-1, 0); break;
      case "ArrowDown":  event.preventDefault(); onArrow(1, 0); break;
      default:
        if (/^[a-zA-Z]$/.test(event.key)) onLetter(event.key.toLowerCase());
    }
  });
}

async function copyShare() {
  if (!game || !game.share) return;
  try {
    await navigator.clipboard.writeText(game.share);
    say("Copied.", "good");
  } catch (err) {
    say("Error: Could not reach the clipboard.", "bad");
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitForSolver() {
  for (;;) {
    let status;
    try {
      status = await (await fetch("/api/status")).json();
    } catch (err) {
      $("veil-text").textContent = "Lost the server; is it still running?";
      await sleep(2000);
      continue;
    }
    if (status.error) {
      $("veil-text").textContent = status.error;
      return null;
    }
    $("veil-text").textContent = status.progress;
    if (status.ready) return status;
    await sleep(400);
  }
}

async function start() {
  wire();
  setView("solve");
  const status = await waitForSolver();
  if (!status) return;

  // the day the answer record was last brought up to date
  if (status.record_through) {
    $("updated").textContent = `Answers through ${status.record_through}`;
    $("updated").hidden = false;
  }

  const forms = $("opt-form");
  for (const name of status.forms) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    forms.appendChild(option);
  }

  // the veil stays up until there is a board behind it
  while (!(await send("/api/new", settings()))) {
    $("veil-text").textContent = "Could not start a board. Trying again…";
    await sleep(1500);
  }
  started = true;
  $("veil").hidden = true;
  // whichever side was asked for while this was still starting up
  setView(view);
}

start();

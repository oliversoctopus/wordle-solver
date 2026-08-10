/* Rewriting what the page says.
 *
 * The list is built from a scan of the site's own source, so it is never a copy
 * that can fall behind: a string added to the page appears here on the next
 * reload, and one that moves is listed wherever it moved to.  Nothing is keyed
 * by hand.
 */

const $ = (id) => document.getElementById(id);

let entries = [];          // as the server last reported them
let pending = new Map();   // id -> the text in the box, where it differs
let filter = "";
let armed = false;         // "write into the code" asks once before doing it

const FILE_NAMES = {
  "web/index.html": "The page",
  "web/app.js": "What it says as you use it",
  "web.py": "What the server says",
};

// kinds that describe where a string came from rather than how it is shown;
// anything else is text you will not find by looking at the page
const PLAIN_KINDS = new Set(["markup", "script", "server", "attribute"]);

const FILE_BLURBS = {
  "web/index.html": "Headings, labels, buttons and the descriptions under them.",
  "web/app.js": "Recommendations, hints, board messages and the technical panel.",
  "web.py": "Errors and loading text. These are matched by wording rather than by "
    + "position, so the same sentence written twice changes in both places.",
};

/* ------------------------------------------------------------------- server */

async function api(path, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({ error: "Bad reply from the server" }));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function adopt(manifest) {
  entries = manifest.entries;
  pending.clear();
  render();
  const edited = entries.filter((e) => e.edited).length;
  say(`${entries.length} things the page says · ${edited} rewritten`,
      manifest.orphans.length
        ? `${manifest.orphans.length} old edit${manifest.orphans.length === 1 ? "" : "s"} `
          + "no longer match anything and are being ignored."
        : "");
}

/* -------------------------------------------------------------------- state */

function say(tally, note, kind) {
  $("tally").textContent = tally;
  $("note").textContent = note || "";
  $("bar").className = kind || "";
}

function problemWith(entry, text) {
  const trimmed = text.trim();
  if (!trimmed) return "Cannot be left empty.";
  for (const slot of entry.slots) {
    if (!text.includes(slot)) return `Has to keep ${slot} — that is where a value goes.`;
  }
  return null;
}

function changeCount() {
  return [...pending].filter(([id, text]) => {
    const entry = entries.find((e) => e.id === id);
    return entry && text !== entry.value;
  }).length;
}

function refreshBar() {
  const changes = changeCount();
  const bad = [...pending].some(([id, text]) => {
    const entry = entries.find((e) => e.id === id);
    return entry && problemWith(entry, text);
  });
  $("save").disabled = changes === 0 || bad;
  $("save").textContent = changes ? `Save ${changes} change${changes === 1 ? "" : "s"}` : "Save";

  // writing into the code is only offered for wording that is already saved,
  // so that what lands in the files is what the page has been showing
  const saved = entries.filter((e) => e.edited).length;
  $("apply").disabled = saved === 0 || changes > 0;
  if (armed && (saved === 0 || changes > 0)) disarm();
  else if (!armed) {
    $("apply").textContent = changes > 0 && saved
      ? "Save first, then write into the code"
      : `Write ${saved || ""} into the code`.replace("  ", " ");
  }
  if (bad) say("Something is not usable yet", "A box below is marked in red.", "problem");
  else if (changes) say(`${changes} change${changes === 1 ? "" : "s"} not saved yet`,
                        "Save to put them on the page.");
  else {
    const edited = entries.filter((e) => e.edited).length;
    say(`${entries.length} things the page says · ${edited} rewritten`, "");
  }
}

/* ------------------------------------------------------------------ drawing */

function matches(entry) {
  if (!filter) return true;
  const needle = filter.toLowerCase();
  return entry.default.toLowerCase().includes(needle)
      || entry.value.toLowerCase().includes(needle)
      || entry.context.toLowerCase().includes(needle);
}

function render() {
  const list = $("list");
  list.textContent = "";
  const shown = entries.filter(matches);

  if (!shown.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Nothing on the page contains that.";
    list.appendChild(empty);
    return;
  }

  let file = null;
  for (const entry of shown) {
    if (entry.file !== file) {
      file = entry.file;
      const head = document.createElement("h2");
      head.className = "file-head";
      head.textContent = FILE_NAMES[file] || file;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = FILE_BLURBS[file] || file;
      head.appendChild(count);
      list.appendChild(head);
    }
    list.appendChild(drawItem(entry));
  }
  refreshBar();
}

function drawItem(entry) {
  const item = document.createElement("div");
  item.className = "item" + (entry.edited ? " edited" : "");
  item.dataset.id = entry.id;

  const where = document.createElement("div");
  where.className = "where";

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = `${entry.file.split("/").pop()}:${entry.line}`;
  const spot = document.createElement("span");
  spot.className = "spot";
  spot.textContent = entry.context || entry.kind;
  where.append(badge, spot);

  // text that is never painted on the page needs saying so, or it is hunted for
  if (!PLAIN_KINDS.has(entry.kind)) {
    const unseen = document.createElement("span");
    unseen.className = "badge unseen";
    unseen.textContent = entry.kind;
    where.appendChild(unseen);
  }

  if (entry.edited) {
    const revert = document.createElement("button");
    revert.type = "button";
    revert.className = "ghost";
    revert.textContent = "Put the original back";
    revert.addEventListener("click", () => revert1(entry));
    where.appendChild(revert);
  }

  const box = document.createElement("textarea");
  box.value = entry.value;
  box.rows = Math.min(9, Math.ceil(entry.value.length / 58) + 1);
  box.spellcheck = true;
  box.addEventListener("input", () => {
    pending.set(entry.id, box.value);
    mark(item, entry, box.value);
    refreshBar();
  });
  const held = pending.get(entry.id);
  if (held !== undefined) box.value = held;

  item.append(where, box);

  if (entry.slots.length) {
    const slots = document.createElement("p");
    slots.className = "slots";
    slots.append("Keeps a value in it: ");
    entry.slots.forEach((slot, index) => {
      if (index) slots.append(", ");
      const code = document.createElement("code");
      code.textContent = slot;
      slots.appendChild(code);
    });
    item.appendChild(slots);
  }

  if (entry.edited) {
    const was = document.createElement("p");
    was.className = "was";
    const label = document.createElement("b");
    label.textContent = "Was: ";
    was.append(label, entry.default);
    item.appendChild(was);
  }

  const problem = document.createElement("p");
  problem.className = "problem-line";
  problem.hidden = true;
  item.appendChild(problem);

  mark(item, entry, box.value);
  return item;
}

function mark(item, entry, text) {
  const problem = problemWith(entry, text);
  const line = item.querySelector(".problem-line");
  line.hidden = !problem;
  line.textContent = problem || "";
  item.classList.toggle("invalid", Boolean(problem));
  item.classList.toggle("changed", !problem && text !== entry.value);
}

/* ------------------------------------------------------------------ actions */

async function save() {
  const edits = {};
  for (const [id, text] of pending) {
    const entry = entries.find((e) => e.id === id);
    if (entry && text !== entry.value) edits[id] = text;
  }
  if (!Object.keys(edits).length) return;
  try {
    adopt(await api("/api/wording/save", { edits }));
    reloadPreview();
    say(`Saved. The page says it now.`, "", "saved");
  } catch (err) {
    say("Could not save", err.message, "problem");
  }
}

async function revert1(entry) {
  try {
    pending.delete(entry.id);
    adopt(await api("/api/wording/reset", { keys: [entry.id] }));
    reloadPreview();
  } catch (err) {
    say("Could not undo that", err.message, "problem");
  }
}

/* --- writing the rewrites into the source ------------------------------- */

function disarm() {
  armed = false;
  $("apply").classList.remove("primary");
  refreshBar();
}

async function apply() {
  if ($("apply").disabled) return;
  if (!armed) {
    armed = true;
    $("apply").classList.add("primary");
    $("apply").textContent = "Yes, write them in";
    say("This one edits the files themselves",
        "index.html, app.js and web.py get your wording written into them. There "
        + "is no undo. Press again to go ahead.");
    return;
  }
  armed = false;
  $("apply").classList.remove("primary");
  try {
    const result = await api("/api/wording/apply", {});
    adopt(result);
    reloadPreview();
    const files = result.written.join(", ") || "nothing";
    say(`Written into ${files}.`,
        `${result.phrases} phrase${result.phrases === 1 ? "" : "s"}; the old and new `
        + `wording is in wording_history.json. The server's own messages need a `
        + `restart before they change.`,
        "saved");
  } catch (err) {
    say("Could not write them in", err.message, "problem");
  }
}

async function revertAll() {
  if (!entries.some((e) => e.edited) && !pending.size) return;
  try {
    adopt(await api("/api/wording/reset", {}));
    reloadPreview();
    say("Every word is back to how it was.", "", "saved");
  } catch (err) {
    say("Could not undo", err.message, "problem");
  }
}

let onPreviewLoad = null;   // at most one thing waiting on the next load

function reloadPreview(path, then) {
  const frame = $("preview");
  if (onPreviewLoad) frame.removeEventListener("load", onPreviewLoad);
  onPreviewLoad = null;
  if (then) {
    onPreviewLoad = () => {
      frame.removeEventListener("load", onPreviewLoad);
      onPreviewLoad = null;
      try {
        then(frame.contentDocument);
      } catch (err) { /* the frame went somewhere else; harmless */ }
    };
    frame.addEventListener("load", onPreviewLoad);
  }
  let here = "/";
  try {
    here = frame.contentWindow.location.pathname || "/";
  } catch (err) { /* nothing loaded there yet */ }
  frame.src = (path || here) + "?t=" + Date.now();
}

/* -------------------------------------------------------------------- start */

function wire() {
  $("save").addEventListener("click", save);
  $("apply").addEventListener("click", apply);
  $("revert-all").addEventListener("click", revertAll);
  $("preview-reload").addEventListener("click", () => reloadPreview("/"));
  $("preview-solve").addEventListener("click", () => reloadPreview("/"));
  $("preview-play").addEventListener("click", () => {
    // the game side is one click inside the page, so ask it to start there
    reloadPreview("/", (page) => page.getElementById("view-toggle").click());
  });

  let typing;
  $("filter").addEventListener("input", (event) => {
    clearTimeout(typing);
    typing = setTimeout(() => { filter = event.target.value.trim(); render(); }, 120);
  });

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "s") {
      event.preventDefault();
      save();
    }
  });
}

async function start() {
  wire();
  try {
    adopt(await api("/api/wording"));
  } catch (err) {
    say("Could not read the page", err.message, "problem");
  }
}

start();

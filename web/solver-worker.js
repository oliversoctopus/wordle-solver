// The solver itself, off the page's thread.
//
// A wide position costs long enough that, run on the main thread, the page
// cannot repaint, scroll or accept a keystroke through it.  The work is the
// same length here; what changes is that the page stays alive through it.
//
// Marshalling is by hand because the module has no wasm-bindgen: ask it for a
// buffer, write UTF-8 in, call handle, read the reply out, free both.

let wasm = null;
// requests that arrived before the module was instantiated, kept rather than
// refused -- the page polls /api/status from its first frame
let waiting = [];

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// The data the solver needs, which the server holds in the other build.  The
// browser caches it like any other static file.
const DATA_FILES = {
  ind2word: "data/derived/ind2word.json",
  wordles: "data/wordles.json",
  solutions: "data/solutions.json",
  answer_like: "data/answer_like.json",
  wordlebot: "data/wordlebot_weights.json",
};

function writeString(text) {
  const bytes = encoder.encode(text);
  const ptr = wasm.alloc(bytes.length);
  // alloc hands back 0 for an empty string and for a failure alike, so the
  // length is what tells them apart; writing at 0 would land on the module's
  // own memory rather than raise anything.
  if (!ptr && bytes.length) throw new Error("the solver is out of memory");
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

function call(path, bodyText) {
  const p = writeString(path);
  const b = writeString(bodyText || "");
  wasm.handle(p.ptr, p.len, b.ptr, b.len);
  const replyPtr = wasm.reply_ptr();
  const replyLen = wasm.reply_len();
  // copied out before anything else can allocate and move the heap
  const reply = decoder.decode(
    new Uint8Array(wasm.memory.buffer, replyPtr, replyLen).slice()
  );
  wasm.dealloc(p.ptr, p.len);
  wasm.dealloc(b.ptr, b.len);
  wasm.dealloc(replyPtr, replyLen);
  return reply;
}

// How long the solve took.  The module has no clock, so it leaves the field at
// zero and this fills it in.  Timing it here rather than on the page's side
// keeps the figure the solve alone, with the two postMessage hops left out.
function timed(reply, started) {
  if (!reply.includes('"elapsed_ms":0')) return reply;
  return reply.replace('"elapsed_ms":0',
                       `"elapsed_ms":${Math.round(performance.now() - started)}`);
}

function serve(request) {
  const started = performance.now();
  let text;
  try {
    text = timed(call(request.path, request.body), started);
  } catch (error) {
    text = JSON.stringify({ error: String(error) });
  }
  postMessage({ kind: "reply", id: request.id, text });
}

// Streaming compilation refuses a module the host did not label
// application/wasm, so a host that mislabels it compiles from the bytes
// instead.  The clone is what leaves those bytes still readable afterwards.
async function instantiate() {
  const response = await fetch("wordle.wasm");
  if (!response.ok) {
    throw new Error(`wordle.wasm: ${response.status} ${response.statusText}`);
  }
  try {
    return await WebAssembly.instantiateStreaming(response.clone(), {});
  } catch (error) {
    return WebAssembly.instantiate(await response.arrayBuffer(), {});
  }
}

async function boot() {
  const module = await instantiate();
  wasm = module.instance.exports;

  // From here the module can answer for itself, and /api/status starts
  // reporting its own progress rather than the page's opening guess at it.
  const held = waiting;
  waiting = [];
  held.forEach(serve);

  const [fits, openings, ...contents] = await Promise.all([
    fetch("fits.json").then((r) => r.json()),
    fetch("openings.json").then((r) => r.json()).catch(() => ({})),
    ...Object.values(DATA_FILES).map((path) => fetch(path).then((r) => r.text())),
  ]);

  const payload = { fits, openings, mask_last: 0 };
  Object.keys(DATA_FILES).forEach((key, i) => {
    payload[key] = contents[i];
  });
  const reply = JSON.parse(call("/api/init", JSON.stringify(payload)));
  if (reply.error) throw new Error(reply.error);
}

onmessage = ({ data }) => {
  if (wasm) serve(data);
  else waiting.push(data);
};

boot().catch((error) => {
  postMessage({ kind: "failed", error: String(error) });
});

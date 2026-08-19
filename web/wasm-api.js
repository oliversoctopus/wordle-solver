// Routes the page's /api/* calls into the WASM solver instead of a server.
//
// `app.js` reaches the solver through exactly two fetch calls, and sends the
// whole board on every submit rather than a delta, so that boundary is already
// a pure function of (path, body).  Intercepting fetch here means app.js needs
// no changes at all -- which matters, since tests/board.test.js loads it
// unmodified and would otherwise be testing a different file from the one that
// ships.
//
// Marshalling is by hand because the module has no wasm-bindgen: ask it for a
// buffer, write UTF-8 in, call handle, read the reply out, free both.

const wasmApi = (() => {
  let wasm = null;

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  // The data the solver needs, which the server used to hold.  686 KB in all,
  // and the browser caches it like any other static file.
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

  async function boot() {
    const module = await WebAssembly.instantiateStreaming(
      fetch("wordle.wasm"),
      {}
    );
    wasm = module.instance.exports;

    const [fits, openings, ...contents] = await Promise.all([
      fetch("fits.json").then((r) => r.json()),
      fetch("openings.json").then((r) => r.json()).catch(() => ({})),
      ...Object.values(DATA_FILES).map((path) => fetch(path).then((r) => r.text())),
    ]);

    const payload = { fits, openings, mask_last: 1 };
    Object.keys(DATA_FILES).forEach((key, i) => {
      payload[key] = contents[i];
    });
    const reply = JSON.parse(call("/api/init", JSON.stringify(payload)));
    if (reply.error) throw new Error(reply.error);
  }

  // Stand in for fetch on /api/*, and leave every other request alone.
  function install() {
    const original = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input.url;
      const path = url.startsWith("http") ? new URL(url).pathname : url;
      if (!path.startsWith("/api/")) return original(input, init);

      const body = init && init.body ? init.body : "";
      const text = wasm ? call(path, body) : JSON.stringify({ ready: false });
      return new Response(text, {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };
  }

  return { boot, install };
})();

wasmApi.install();
wasmApi.boot().catch((error) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    '<pre style="padding:1rem;color:#b00">The solver failed to load: ' +
      String(error).replace(/[<&]/g, "") +
      "</pre>"
  );
});

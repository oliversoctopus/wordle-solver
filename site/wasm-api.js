// Routes the page's /api/* calls into the WASM solver instead of a server.
//
// `app.js` reaches the solver through exactly two fetch calls, and sends the
// whole board on every submit rather than a delta, so that boundary is already
// a pure function of (path, body).  Intercepting fetch here means app.js needs
// no changes at all -- which matters, since tests/board.test.js loads it
// unmodified and would otherwise be testing a different file from the one that
// ships.
//
// The solver runs in `solver-worker.js`, so a request is a message rather than
// a call.  That costs a round trip of a fraction of a millisecond and buys a
// page that keeps repainting through a turn that takes 150.  `fetch` is
// asynchronous anyway, which is what makes the swap invisible to app.js.

const wasmApi = (() => {
  const worker = new Worker("solver-worker.js");

  const pending = new Map();
  let nextId = 1;
  let failure = null;

  worker.onmessage = ({ data }) => {
    if (data.kind === "reply") {
      const settle = pending.get(data.id);
      pending.delete(data.id);
      if (settle) settle(data.text);
      return;
    }
    if (data.kind === "failed") {
      // Answered rather than left hanging: the page waits on /api/status
      // behind its veil, and reports whatever error comes back there.
      failure = JSON.stringify({ error: `The solver failed to load: ${data.error}` });
      pending.forEach((settle) => settle(failure));
      pending.clear();
    }
  };

  function ask(path, body) {
    if (failure) return Promise.resolve(failure);
    return new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      worker.postMessage({ id, path, body });
    });
  }

  // Dealing an answer needs a random draw, and the module has no entropy
  // source of its own.  So the randomness comes from here, while the draw
  // itself stays in Rust, where the prior weights are.
  function seeded(path, body) {
    if (path !== "/api/game/new") return body;
    const parsed = body ? JSON.parse(body) : {};
    return JSON.stringify({ ...parsed, seed: Math.random() });
  }

  // Whether a reply is a refusal, which has to reach the page as a status and
  // not just a field: `app.js` throws on a reply that is not ok and renders the
  // board on one that is, so an error arriving as 200 gets applied as though it
  // were a board and takes the page down with it.  `web.py` answers 400 here.
  //
  // Which of 400, 404 and 503 it would have been is not reconstructed. Nothing
  // on the page tells them apart -- it shows the message either way -- and the
  // only way to tell from here is to match on the message text, which the
  // editor is free to rewrite.
  function refused(text) {
    try {
      return JSON.parse(text).error !== undefined;
    } catch (error) {
      return true;
    }
  }

  // Stand in for fetch on /api/*, and leave every other request alone.
  function install() {
    const original = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input.url;
      const path = url.startsWith("http") ? new URL(url).pathname : url;
      if (!path.startsWith("/api/")) return original(input, init);

      const text = await ask(path, seeded(path, init && init.body ? init.body : ""));
      return new Response(text, {
        status: refused(text) ? 400 : 200,
        headers: { "Content-Type": "application/json" },
      });
    };
  }

  return { install };
})();

wasmApi.install();

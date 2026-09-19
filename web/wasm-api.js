// Routes the page's /api/* calls into the WASM solver instead of a server.
//
// `app.js` sends the whole board on every submit, so that boundary is a pure
// function of (path, body); intercepting fetch here means app.js needs no
// changes, and tests/board.test.js keeps loading the file that ships.  The
// solver runs in `solver-worker.js`, so a request is a message and a reply,
// and `fetch` was asynchronous already.

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
    if (path !== "/api/game/new" && path !== "/api/draw") return body;
    const parsed = body ? JSON.parse(body) : {};
    return JSON.stringify({ ...parsed, seed: Math.random() });
  }

  // Whether a reply is a refusal, which has to reach the page as a status:
  // `app.js` throws on a reply that is not ok and renders a board on one that
  // is, so an error arriving as 200 would be applied as a board.  Which of
  // 400, 404 and 503 it would have been is not reconstructed.
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

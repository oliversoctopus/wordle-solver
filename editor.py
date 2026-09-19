"""Rewrite what the site says, previewing the site that gets deployed.

    python editor.py                edit, on http://127.0.0.1:8001/editor
    python editor.py --port 9001    somewhere else
    python editor.py --no-browser   do not open one

It previews ``site/``, so it loads no word lists and no table.  *Save* applies
rewrites to the served bytes; *Write into the code* puts them in the source,
and a phrase written into the Rust needs ``./build_site.sh`` to show.
"""

import argparse
import json
import os
import posixpath
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wording import Wording

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")

# the two files whose text is editable; everything else is served as it is
REWRITTEN = {"index.html": "web/index.html", "app.js": "web/app.js"}

TYPES = {".html": "text/html; charset=utf-8",
         ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8",
         ".json": "application/json; charset=utf-8",
         ".wasm": "application/wasm"}

WORDING = Wording()


class Handler(BaseHTTPRequestHandler):
    server_version = "wordle-editor"
    protocol_version = "HTTP/1.1"

    site_dir = None

    def log_message(self, fmt, *args):
        if not self.server.quiet:
            super().log_message(fmt, *args)

    # ---------------------------------------------------------------- plumbing

    def send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # the module is fetched once and the page is edited over and over
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        self.send_bytes(json.dumps(payload).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def resolve(self, path):
        """Where a request's file lives, or None if it is outside both roots.

        The editor's own three files are in ``web/``; everything the page needs
        is in the built site.  Neither directory is allowed to be escaped.
        """
        name = posixpath.normpath("/" + path).lstrip("/")
        if not name or name.endswith("/"):
            name += "index.html"
        for root in (self.site_dir, WEB_DIR):
            full = os.path.normpath(os.path.join(root, name))
            try:
                inside = not os.path.relpath(full, root).startswith(os.pardir)
            except ValueError:      # a name naming another drive
                continue
            if inside and os.path.isfile(full):
                return name, full
        return None, None

    # ------------------------------------------------------------------ routes

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/editor":
            path = "/editor.html"
        if path == "/api/wording":
            return self.send_json(WORDING.manifest())
        if path.startswith("/api/"):
            return self.send_json({"error": "Not found"}, status=404)

        name, full = self.resolve(path)
        if full is None:
            return self.send_json({"error": "Not found"}, status=404)
        source = REWRITTEN.get(name)
        if source is not None:
            body = WORDING.served(source, full).encode("utf-8")
        else:
            with open(full, "rb") as handle:
                body = handle.read()
        extension = os.path.splitext(full)[1]
        return self.send_bytes(body, TYPES.get(extension, "application/octet-stream"))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self.read_body()
            if path == "/api/wording/save":
                WORDING.save(body.get("edits") or {})
            elif path == "/api/wording/reset":
                keys = body.get("keys")
                if keys is not None and not isinstance(keys, list):
                    raise ValueError("expected a list of keys")
                WORDING.forget(keys)
            elif path == "/api/wording/apply":
                return self.send_json({**WORDING.bake(), **WORDING.manifest()})
            else:
                return self.send_json({"error": "Not found"}, status=404)
        except ValueError as error:
            return self.send_json({"error": str(error)}, status=400)
        return self.send_json(WORDING.manifest())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--site", default="site",
                        help="the built site to preview (default site/)")
    parser.add_argument("--overrides", default=None,
                        help="where the rewrites are kept (default "
                             "wording.json); point it at a scratch file to try "
                             "the editor out without keeping anything")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    site = os.path.abspath(os.path.join(HERE, args.site))
    if not os.path.isfile(os.path.join(site, "index.html")):
        print(f"no built site in {site} -- run ./build_site.sh first")
        return 1
    if args.overrides:
        WORDING.path = os.path.abspath(args.overrides)
        WORDING.load()

    Handler.site_dir = site
    # localhost only: anyone who can reach this can rewrite the site and the
    # source it is built from
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.quiet = args.quiet
    url = f"http://127.0.0.1:{args.port}/editor"
    print(f"previewing {os.path.relpath(site, HERE)}")
    print(f"rewrite what it says on {url}")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

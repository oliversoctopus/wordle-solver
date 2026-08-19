#!/bin/sh
# Assembles the static site into site/, ready to serve or to deploy.
#
#     ./build_site.sh [output directory]     default: site/
#
# Everything in it is a file: no server, no session, no 167 MB table. The page
# is web/'s own, unmodified -- web/wasm-api.js routes its /api/* calls into the
# WASM module instead of over the network.

set -e
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

OUT="${1:-site}"
rm -rf "$OUT"
mkdir -p "$OUT/data/derived"

echo "building the solver for wasm32"
cargo build --release --target wasm32-unknown-unknown -p wordle_wasm
cp target/wasm32-unknown-unknown/release/wordle_wasm.wasm "$OUT/wordle.wasm"

echo "working out the fits and the openings"
cargo build --release --quiet
./target/release/build_site "$(pwd)" "$OUT" >/dev/null

echo "copying the page and the data"
cp web/style.css web/wasm-api.js web/vercel.json "$OUT/"
cp data/wordles.json data/solutions.json data/answer_like.json \
   data/wordlebot_weights.json "$OUT/data/"
cp data/derived/ind2word.json "$OUT/data/derived/"

# The page and the script go through wording.py rather than being copied, so a
# rewrite saved in the editor and not yet written into the code still ships.
# The shim goes in ahead of app.js, so fetch is already routed by its first call.
python - "$OUT" <<'PY'
import io, os, sys

from wording import Wording

out = sys.argv[1]
wording = Wording()
for name in ("web/index.html", "web/app.js"):
    text = wording.served(name)
    if name.endswith("index.html") and "wasm-api.js" not in text:
        text = text.replace('<script src="app.js"',
                            '<script src="wasm-api.js"></script>\n'
                            '    <script src="app.js"', 1)
    path = os.path.join(out, os.path.basename(name))
    io.open(path, "w", encoding="utf-8", newline="").write(text)

applied = len(wording.overrides)
if applied:
    print(f"  {applied} rewrite{'' if applied == 1 else 's'} from wording.json")
PY

# Vercel reads .gitignore when a deployment has no .vercelignore, and site/ is
# gitignored, so without this an upload from here could come out empty.
echo '# nothing to leave out: every file here is meant to deploy'     > "$OUT/.vercelignore"

echo
du -sh "$OUT" | sed 's/^/  total  /'
echo "  serve it with: python -m http.server -d $OUT 8010"

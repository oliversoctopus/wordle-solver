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
cp web/index.html web/app.js web/style.css web/wasm-api.js "$OUT/"
cp data/wordles.json data/solutions.json data/answer_like.json \
   data/wordlebot_weights.json "$OUT/data/"
cp data/derived/ind2word.json "$OUT/data/derived/"

# the page loads the shim before app.js, so fetch is already routed when app.js
# makes its first call
python - "$OUT/index.html" <<'PY'
import io, sys
path = sys.argv[1]
html = io.open(path, encoding="utf-8", newline="").read()
tag = '<script src="wasm-api.js"></script>\n'
if "wasm-api.js" not in html:
    html = html.replace('<script src="app.js"', tag + '    <script src="app.js"', 1)
io.open(path, "w", encoding="utf-8", newline="").write(html)
PY

echo
du -sh "$OUT" | sed 's/^/  total  /'
echo "  serve it with: python -m http.server -d $OUT 8010"

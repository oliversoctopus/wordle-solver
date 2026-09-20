"""What the static build weighs, on disk and over the wire.

Measures `site/` as `./build_site.sh` left it: the module, the data, the
whole, and each of them after brotli.  Counts the openings and turn-two
rankings baked into `openings.json`.  Backs README *Without a server*.
"""

import argparse
import json
import os
import shutil
import subprocess

import common

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--site", default=os.path.join(common.ROOT, "site"),
                    help="the built site to measure")
args = parser.parse_args()

SKIP = {".vercel", ".vercelignore"}


def brotli(payload):
    """Bytes after brotli at its default quality, by the tool on PATH."""
    if shutil.which("brotli"):
        done = subprocess.run(["brotli", "-c"], input=payload,
                              capture_output=True, check=True)
        return len(done.stdout)
    import brotli as module
    return len(module.compress(payload))


files = {}
for folder, dirs, names in os.walk(args.site):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for name in names:
        if name in SKIP:
            continue
        path = os.path.join(folder, name)
        with open(path, "rb") as file:
            files[os.path.relpath(path, args.site).replace("\\", "/")] = file.read()
if not files:
    raise SystemExit(f"nothing in {args.site}; run ./build_site.sh first")

common.header(f"what {os.path.relpath(args.site, common.ROOT)}/ weighs")
kb = lambda n: n / 1000  # noqa: E731
raw = {name: len(body) for name, body in files.items()}
wire = {name: brotli(body) for name, body in files.items()}
print(f"  {'file':28s} {'on disk':>10s} {'brotli':>10s}")
for name in sorted(raw, key=raw.get, reverse=True):
    print(f"  {name:28s} {kb(raw[name]):8.0f} KB {kb(wire[name]):8.0f} KB")
data = [name for name in raw if name.startswith("data/")]
print(f"\n  total {kb(sum(raw.values())) / 1000:.1f} MB on disk, "
      f"{kb(sum(wire.values())):.0f} KB over the wire")
common.record("total_mb", kb(sum(raw.values())) / 1000)
common.record("wire_kb", kb(sum(wire.values())))
common.record("wasm_kb", kb(raw["wordle.wasm"]))
common.record("wasm_wire_kb", kb(wire["wordle.wasm"]))
common.record("data_kb", kb(sum(raw[name] for name in data)))
common.record("solutions_kb", kb(raw["data/solutions.json"]))
common.record("solutions_wire_kb", kb(wire["data/solutions.json"]))

common.header("what openings.json holds")
rankings = json.loads(files["openings.json"])
openings = {key: value for key, value in rankings.items() if key.count("|") == 3}
turn2 = {key: value for key, value in rankings.items() if key.count("|") == 4}
priors = {key.split("|")[0] for key in turn2}
print(f"  {len(openings)} openings, {len(turn2)} turn-two rankings over "
      f"{len(priors)} priors, {len(turn2) / len(priors):.0f} buckets each")
turn2_wire = kb(brotli(json.dumps(turn2, separators=(",", ":")).encode()))
print(f"  the turn-two rankings are {turn2_wire:.0f} KB over the wire")
common.record("openings", len(openings))
common.record("turn2.rankings", len(turn2))
common.record("turn2.priors", len(priors))
common.record("turn2.per_prior", len(turn2) / len(priors))
common.record("turn2.wire_kb", turn2_wire)

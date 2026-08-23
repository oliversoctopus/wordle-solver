"""Rescrape WordleBot's per-word weights, live and from the 2023 Wayback snapshot.

Two files, from the same page at two points in time.  `wordlebot_weights.json`
is the live list and is a solver input -- the `wordlebot` curation shape spreads
the off-list mass by it -- so replacing it changes the shipped prior and every
fit made against it.  `wordlebot_weights_2023-10-12.json` is a frozen copy that
nothing plays with; it exists so the claim that an *old* list predicts *later*
answers can be recomputed rather than remembered.

The page gives a percentage per word.  Weights are stored relative to the
largest, so the heaviest word is 1.0 and the file does not move when the site
rescales.  Only guessable words are kept, the rest of the solver having no
column for the others.

Like the other fetchers here it verifies and writes nothing without `--write`,
because a changed live list invalidates every cached fit.
"""

import argparse
import json
import os
import re

import requests

from priors import (WORDLEBOT_PATH, WORDLEBOT_SNAPSHOT_DATE,
                    WORDLEBOT_SNAPSHOT_PATH)
from wordle_data import DATA_DIR, write_json

LIVE_URL = "https://wordletools.azurewebsites.net/weightedbottles"
# `id_` asks the Wayback Machine for the bytes it archived rather than a copy
# with its own toolbar injected
SNAPSHOT_URL = ("https://web.archive.org/web/{stamp}id_/"
                "https://wordletools.azurewebsites.net/weightedbottles")
AVAILABILITY = "https://archive.org/wayback/available"

ROW = re.compile(r"<td>([A-Za-z]{5})</td>\s*<td>\s*([0-9.]+)%\s*</td>")
HEADERS = {"User-Agent": "Mozilla/5.0 (wordle-solver data build)"}


def guessable():
    with open(os.path.join(DATA_DIR, "wordles.json"), "r") as file:
        words = set(json.load(file))
    with open(os.path.join(DATA_DIR, "nonwordles.json"), "r") as file:
        return words | set(json.load(file))


def parse(html, pool):
    """Word -> weight relative to the heaviest, for the words we can guess."""
    percent = {word.lower(): float(pct) for word, pct in ROW.findall(html)}
    if not percent:
        raise ValueError("no word/weight rows found; the page layout changed")
    top = max(percent.values())
    # six places is what the shipped file carries, and the page quotes no more
    kept = {w: round(v / top, 6) for w, v in percent.items() if w in pool}
    return kept, len(percent)


def fetch(url):
    reply = requests.get(url, timeout=120, headers=HEADERS)
    reply.raise_for_status()
    return reply.text


def snapshot_url():
    """The archived copy nearest the date the validation is stated against."""
    stamp = WORDLEBOT_SNAPSHOT_DATE.replace("-", "")
    reply = requests.get(AVAILABILITY, timeout=60, headers=HEADERS,
                         params={"url": LIVE_URL.split("//", 1)[1],
                                 "timestamp": stamp})
    reply.raise_for_status()
    closest = reply.json().get("archived_snapshots", {}).get("closest")
    if not closest or not closest.get("available"):
        raise RuntimeError(f"no Wayback snapshot near {WORDLEBOT_SNAPSHOT_DATE}")
    return SNAPSHOT_URL.format(stamp=closest["timestamp"]), closest["timestamp"]


def report(label, weights, scraped, path):
    print(f"{label}: {scraped} words scraped, {len(weights)} of them guessable")
    if not os.path.exists(path):
        print("  nothing shipped to compare against")
        return
    with open(path, "r") as file:
        shipped = json.load(file)
    added = sorted(set(weights) - set(shipped))
    dropped = sorted(set(shipped) - set(weights))
    moved = [w for w in weights
             if w in shipped and abs(shipped[w] - weights[w]) > 5e-7]
    print(f"  shipped {len(shipped)}; added {len(added)}, dropped {len(dropped)}, "
          f"reweighted {len(moved)}")
    for name, words in (("added", added), ("dropped", dropped), ("reweighted", moved)):
        if words:
            print(f"    {name}: {', '.join(words[:10])}"
                  f"{' ...' if len(words) > 10 else ''}")
    if not (added or dropped or moved):
        print("  identical to the shipped copy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="overwrite the files; the live one invalidates "
                             "every cached fit")
    parser.add_argument("--snapshot-only", action="store_true",
                        help="leave the live list alone, which is the safe half")
    args = parser.parse_args()

    pool = guessable()
    jobs = []
    if not args.snapshot_only:
        live, scraped = parse(fetch(LIVE_URL), pool)
        report("live", live, scraped, WORDLEBOT_PATH)
        jobs.append((WORDLEBOT_PATH, live))

    url, stamp = snapshot_url()
    frozen, scraped = parse(fetch(url), pool)
    report(f"snapshot {stamp}", frozen, scraped, WORDLEBOT_SNAPSHOT_PATH)
    jobs.append((WORDLEBOT_SNAPSHOT_PATH, frozen))

    if args.write:
        for path, payload in jobs:
            write_json(path, payload, sort_keys=True)
            print(f"wrote {path}")
        if not args.snapshot_only:
            print("the live list is a solver input: refit with "
                  "`python wordle_solver.py train`")
    else:
        print("\nnot written; pass --write to replace them")


if __name__ == "__main__":
    main()

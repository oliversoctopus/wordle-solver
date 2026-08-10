"""Refetch `wordles.json` and `nonwordles.json` from stuartpb/wordles.

`wordles.json` is the curated 2022 answer list and `nonwordles.json` is
everything else that is a legal guess.  Together they are the vocabulary the
whole solver is built on, so changing them invalidates `feedback_table.npy` and
everything derived from it -- rerun `feedback_table.py` afterwards.
"""

import argparse
import json
import os

import requests

from wordle_data import DATA_DIR, write_json

BASE = "https://raw.githubusercontent.com/stuartpb/wordles/main/{name}.json"
LISTS = {"wordles": os.path.join(DATA_DIR, "wordles.json"),
         "nonwordles": os.path.join(DATA_DIR, "nonwordles.json")}


def fetch(name):
    reply = requests.get(BASE.format(name=name), timeout=30)
    reply.raise_for_status()
    words = sorted(w.lower() for w in reply.json())
    if any(len(w) != 5 for w in words):
        raise ValueError(f"{name}: not every entry is five letters")
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="overwrite the lists, which invalidates the feedback table")
    args = parser.parse_args()

    for name, path in LISTS.items():
        words = fetch(name)
        print(f"{name}: {len(words)} words")
        if os.path.exists(path):
            with open(path, "r") as file:
                shipped = json.load(file)
            added = sorted(set(words) - set(shipped))
            dropped = sorted(set(shipped) - set(words))
            print(f"  shipped {len(shipped)}; added {len(added)}, dropped {len(dropped)}")
            if added:
                print(f"    added:   {', '.join(added[:10])}")
            if dropped:
                print(f"    dropped: {', '.join(dropped[:10])}")
        if args.write:
            write_json(path, words, indent=None)
            print(f"  wrote {path}")

    if not args.write:
        print("\nnot written; pass --write to replace the lists, then rerun "
              "feedback_table.py")


if __name__ == "__main__":
    main()

"""Rebuild `word_freq.json`: how common each listed word is in English.

The Zipf frequency `wordfreq` gives, log10 of uses per billion words, for every
word in the list that it has seen. The finder ranks guesses by it. Writes
nothing without `--write`. Needs `wordfreq`:

    pip install wordfreq
"""

import argparse
import json
import os

from wordfreq import zipf_frequency

from wordle_data import IND2WORD_PATH, WORD_FREQ_PATH, write_json

DECIMALS = 2


def build():
    """`word -> zipf` for every listed word with a nonzero frequency."""
    with open(IND2WORD_PATH, "r") as file:
        words = json.load(file)
    table = {w: round(zipf_frequency(w, "en"), DECIMALS) for w in words}
    return {w: z for w, z in table.items() if z > 0}, len(words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="overwrite word_freq.json, which the page reads")
    args = parser.parse_args()

    table, listed = build()
    print(f"{len(table)} of {listed} words have a frequency")

    identical = False
    if os.path.exists(WORD_FREQ_PATH):
        with open(WORD_FREQ_PATH, "r") as file:
            shipped = json.load(file)
        identical = shipped == table
        changed = sum(shipped.get(w) != z for w, z in table.items())
        dropped = len(set(shipped) - set(table))
        print(f"against the shipped word_freq.json: {changed} changed, "
              f"{dropped} dropped")

    if args.write:
        write_json(WORD_FREQ_PATH, table, sort_keys=True)
        print(f"wrote {WORD_FREQ_PATH}")
    elif identical:
        print("not written; it already matches the shipped file exactly")
    else:
        print("not written; pass --write to replace word_freq.json")


if __name__ == "__main__":
    main()

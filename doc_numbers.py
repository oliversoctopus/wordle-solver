"""Bind the numbers in the documents to the scripts that measured them.

Every measured figure in the Markdown here comes from a script in `analysis/`,
and until now the documents were brought up to date by hand, one number at a
time.  Instead, a script records what it prints -- `common.record(key, value)`
-- into `analysis/results/<script>.json`, and the documents mark where each
figure comes from with an HTML comment, invisible when rendered:

    <!--n repeat_shapes.binned_vs_uniform-->+1.13 ± 0.21      one number
    <!--n regime.since.repeat .1%-->10.5%                     with a format
    <!--s replay.winner-->wins 19 of 24<!--/s-->               any string
    <!--table results_tables.priors-->                        a whole table
    | ... |
    <!--/table-->

Run with no arguments to check that every bound figure matches its results
file, which `tests/run.py` does; `--write` rewrites the ones that differ and
lists them, old beside new, which is the reading list for the prose around
them -- a number that moved is a sentence to reread, and the tool cannot
reread it.  `--unbound FILE` lists what looks like a figure and carries no
marker, for the markup pass.

A raw number takes the marker's Python format spec (`.1%`, `+.4f`, `,d`),
defaulting to a thousands-separated integer or `g`, and a minus is rendered as
U+2212 the way the documents write it.  Only the number is replaced: a unit
after it -- `%`, `×`, `σ` -- is the document's own and stays, so a `%` spec
scales the value without printing the sign.  A recorded string is inserted as
it is.  The key is `<results file>.<name>`, and a name may itself carry dots.
"""

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "analysis", "results")
DOCS = ["README.md", "SOLVER_EXPLAINED.md", "ABOUT_SOLVER_EXPLAINED.md",
        "CLAUDE.md", os.path.join("analysis", "README.md"),
        os.path.join("tests", "README.md")]

N_RE = re.compile(r"<!--n ([\w.\-/]+)(?: ([^ >]+))?-->")
# what follows an inline marker: a signed number and its ± if any; the unit
# after it (%, ×, σ, s) belongs to the document and stays put
NUMBER = r"[+\-−]?\d(?:[\d,]*\d)?(?:\.\d+)?(?:e[+\-−]?\d+)?"
TOKEN_RE = re.compile(NUMBER + r"(?:\s*±\s*" + NUMBER + r")?")
S_RE = re.compile(r"<!--s ([\w.\-/]+)-->(.*?)<!--/s-->", re.S)
TABLE_RE = re.compile(r"(<!--table ([\w.\-/]+)-->\n)(.*?)(\n<!--/table-->)", re.S)
# a figure-shaped thing in prose, for --unbound
FIGURE_RE = re.compile(r"(?<![\w.<!\-])[+\-−]?\d[\d,]*(?:\.\d+)?(?:%|×|σ)?(?![\w\-])")


def load_results():
    values = {}
    for path in glob.glob(os.path.join(RESULTS, "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as file:
            for key, value in json.load(file)["values"].items():
                values[f"{name}.{key}"] = value
    return values


def render(value, spec):
    """A recorded value as the document should print it."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if spec is None:
        spec = ",d" if isinstance(value, int) else "g"
    text = format(value, spec)
    if "e" in text:
        # the documents write 5.4e-5: a short exponent, ASCII minus
        mantissa, exponent = text.split("e")
        return f"{mantissa}e{int(exponent)}"
    # a percent spec scales the value; the sign itself is the document's
    return text.replace("-", "−").rstrip("%")


def bind(text, values, path, changes, missing):
    """``text`` with every marker's figure replaced by the recorded one."""

    def note(key, old, new, pos):
        if old != new:
            changes.append((path, text.count("\n", 0, pos) + 1, key, old, new))

    # inline markers: splice left to right, dropping the old token after each
    pieces, last = [], 0
    for match in N_RE.finditer(text):
        key, spec = match.group(1), match.group(2)
        pieces.append(text[last:match.end()])
        last = match.end()
        if key not in values:
            missing.append((path, key))
            continue
        token = TOKEN_RE.match(text, match.end())
        if token is None:
            missing.append((path, f"{key} (no figure follows the marker)"))
            continue
        new = render(values[key], spec)
        note(key, token.group(0), new, match.start())
        pieces.append(new)
        last = token.end()
    pieces.append(text[last:])
    out = "".join(pieces)

    def span(match):
        key = match.group(1)
        if key not in values:
            missing.append((path, key))
            return match.group(0)
        new = render(values[key], None)
        note(key, match.group(2), new, match.start())
        return f"<!--s {key}-->{new}<!--/s-->"

    out = S_RE.sub(span, out)

    def table(match):
        key = match.group(2)
        if key not in values:
            missing.append((path, key))
            return match.group(0)
        new = render(values[key], None).rstrip("\n")
        note(key, match.group(3), new, match.start())
        return match.group(1) + new + match.group(4)

    return TABLE_RE.sub(table, out)


def unbound(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
    for number, line in enumerate(lines, start=1):
        if line.startswith("|") or line.startswith("```"):
            continue
        clean = S_RE.sub("", line)
        for match in reversed(list(N_RE.finditer(clean))):
            token = TOKEN_RE.match(clean, match.end())
            clean = clean[:match.start()] + clean[token.end() if token else match.end():]
        clean = re.sub(r"<!--.*?-->", "", clean)
        hits = [m.group(0) for m in FIGURE_RE.finditer(clean)
                if not re.fullmatch(r"[+\-−]?\d{1,2}", m.group(0))]
        if hits:
            print(f"{path}:{number}: {' '.join(hits)}")


def main():
    # the figures carry − ± × σ, which a cp1252 console would refuse to print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite the documents; the default only checks")
    parser.add_argument("--unbound", metavar="FILE",
                        help="list figures in FILE that carry no marker")
    args = parser.parse_args()

    if args.unbound:
        unbound(args.unbound)
        return 0

    values = load_results()
    changes, missing = [], []
    for name in DOCS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as file:
            text = file.read()
        bound = bind(text, values, name, changes, missing)
        if args.write and bound != text:
            with open(path, "w", encoding="utf-8", newline="") as file:
                file.write(bound)

    for path, key in missing:
        print(f"{path}: no recorded value for {key}")
    for path, line, key, old, new in changes:
        flag = ""
        sign = lambda s: s.lstrip().startswith(("-", "−"))  # noqa: E731
        if "\n" not in old and sign(old) != sign(new):
            flag = "   <- sign"
        old_s = old if "\n" not in old else f"({old.count(chr(10)) + 1} lines)"
        new_s = new if "\n" not in new else f"({new.count(chr(10)) + 1} lines)"
        print(f"{path}:{line}: {key}: {old_s} -> {new_s}{flag}")

    if not changes and not missing:
        print("every bound figure is current")
        return 0
    if args.write:
        print(f"rewrote {len(changes)} figures; reread the sentences above")
        return 1 if missing else 0
    print(f"{len(changes)} stale; pass --write to update the documents")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

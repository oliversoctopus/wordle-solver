"""Bind the numbers in the documents to the scripts that measured them.

A script records what it prints with `common.record(key, value)` into
`analysis/results/<script>.json`; a document marks each figure with a comment
naming the key: `<!--n script.key .4f-->` before a number, `<!--s key-->…
<!--/s-->` around a string, `<!--table key-->`…`<!--/table-->` around a table.
Run bare to check, `--write` to rewrite and list what moved, `--unbound` to
fail on an unmarked figure.  `<!--c-->` marks a definition and `<!--u-->` a
figure analysis/README *What is not reproduced here* lists as measured once.
"""

import argparse
import ast
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
# the modules whose upper-case assignments are the shipped defaults, and the
# check whose limits the test catalogue quotes
CONSTANT_MODULES = ["priors.py", "exact.py", "solver.py", "train.py",
                    "wordle_data.py", os.path.join("tests", "voice.py")]

N_RE = re.compile(r"<!--n ([\w.\-/]+)(?: ([^ >]+))?-->")
# what follows an inline marker: a signed number and its error term; the unit
# after it (%, ×, σ, s) belongs to the document and stays put
NUMBER = r"[+\-−]?\d(?:[\d,]*\d)?(?:\.\d+)?(?:e[+\-−]?\d+)?"
TOKEN_RE = re.compile(NUMBER + r"(?:\s*±\s*" + NUMBER + r")?")
S_RE = re.compile(r"<!--s ([\w.\-/]+)-->(.*?)<!--/s-->", re.S)
TABLE_RE = re.compile(r"(<!--table ([\w.\-/]+)-->\n)(.*?)(\n<!--/table-->)", re.S)
# a figure-shaped thing in prose, for --unbound
FIGURE = r"[+\-−]?\d(?:[\d,]*\d)?(?:\.\d+)?(?:e[+\-−]?\d+)?(?:%|×|σ|x)?"
FIGURE_RE = re.compile(r"(?<![\w.<!\-])" + FIGURE + r"(?![\w\-])")
# the exemption markers: a span, or the one figure that follows
EXEMPT_SPAN_RE = re.compile(r"<!--([cu])-->.*?<!--/\1-->")
EXEMPT_NEXT_RE = re.compile(r"<!--[cu]-->\s*" + FIGURE)
# figures that name something: a section, a puzzle number, a date, a year, a
# function's argument, and an integer short enough to be a count in prose
NAMED_RE = re.compile(r"§\d+(?:\.\d+)?|#\d+(?:[–-]#?\d+)?|\d{4}-\d\d-\d\d|"
                      r"\b20[12]\d\b|\(\d[\d,.]*\)")
SHORT_RE = re.compile(r"[+\-−]?\d{1,2}σ?")


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
        # the documents write a short exponent with an ASCII minus
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


def constants():
    """Figures that follow from the code or the data, exempt from binding."""
    found = {0, 1}
    for name in CONSTANT_MODULES:
        with open(os.path.join(HERE, name), "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.isupper()):
                continue
            for leaf in ast.walk(node.value):
                if (isinstance(leaf, ast.Constant)
                        and isinstance(leaf.value, (int, float))
                        and not isinstance(leaf.value, bool)):
                    found.add(leaf.value)
    sys.path.insert(0, HERE)
    import wordle_data
    with open(wordle_data.CURATED_PATH, "r", encoding="utf-8") as file:
        curated = len(json.load(file))
    with open(os.path.join(wordle_data.DERIVED_DIR, "ind2word.json"), "r",
              encoding="utf-8") as file:
        words = len(json.load(file))
    patterns = len(wordle_data.TILE_VALS) ** 5
    found |= {words, words - 1, words * words, words * words // 10 ** 6,
              curated, words - curated, patterns, patterns - 1,
              wordle_data.MAX_GUESSES + 1}
    # a probability is written as itself or as the percent of its complement
    found |= {1 - p for p in list(found) if 0 < p < 1}
    return found


def figure_value(token):
    """The number a figure token stands for, percent scaled."""
    text = token.replace(",", "").replace("−", "-").rstrip("%×σx")
    value = float(text)
    return value / 100 if token.endswith("%") else value


def unbound(name, exempt):
    """Prints the figures in document ``name`` with no marker; returns how many."""
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
    count, fenced, bound_table = 0, False, False
    for number, line in enumerate(lines, start=1):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if line.startswith("<!--table "):
            bound_table = True
        if line.startswith("<!--/table-->"):
            bound_table = False
        # a row of a bound table is rewritten whole; any other row is prose
        if fenced or line.startswith(("#", "$$")) or (bound_table and line.startswith("|")):
            continue
        clean = re.sub(r"`[^`]*`", "", line)
        clean = S_RE.sub("", clean)
        clean = EXEMPT_SPAN_RE.sub("", clean)
        clean = EXEMPT_NEXT_RE.sub("", clean)
        for match in reversed(list(N_RE.finditer(clean))):
            token = TOKEN_RE.match(clean, match.end())
            clean = clean[:match.start()] + clean[token.end() if token else match.end():]
        clean = re.sub(r"<!--.*?-->", "", clean)
        clean = NAMED_RE.sub(" ", clean)
        hits = [m.group(0) for m in FIGURE_RE.finditer(clean)
                if not SHORT_RE.fullmatch(m.group(0))
                and not any(abs(figure_value(m.group(0)) - c) <= 1e-9 * max(1, abs(c))
                            for c in exempt)]
        if hits:
            print(f"{name}:{number}: {' '.join(hits)}")
            count += len(hits)
    return count


def main():
    # the figures carry non-ASCII signs and units a cp1252 console refuses
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite the documents; the default only checks")
    parser.add_argument("--unbound", metavar="FILE", nargs="?", const="all",
                        help="list figures that carry no marker, in FILE or "
                             "in every document")
    args = parser.parse_args()

    if args.unbound:
        exempt = constants()
        docs = DOCS if args.unbound == "all" else [args.unbound]
        loose = sum(unbound(name, exempt) for name in docs)
        if loose:
            print(f"{loose} figures carry no marker")
            return 1
        print("every figure is bound or marked")
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

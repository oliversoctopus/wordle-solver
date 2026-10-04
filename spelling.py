"""American spelling in the documents, comments and docstrings, checked and fixed.

`data/british_spellings.json` maps a British spelling to the American one, for
words VarCon marks preferred in no American sense. Words match whole, a fix
writes only the listed spelling in the same case, and code is never touched.

    python spelling.py                      report each British spelling
    python spelling.py --write              replace them
    python spelling.py --refresh [--write]  rebuild the list from VarCon
"""

import argparse
import io
import json
import os
import re
import sys
import tokenize
import urllib.request

from wordle_data import BRITISH_SPELLINGS_PATH, write_json

VARCON = "https://raw.githubusercontent.com/en-wl/wordlist/master/varcon/"

# usage American English shares, though VarCon marks each British only
KEEP = {"axe", "backwards", "diss", "whisky"}

# reviewed pairs VarCon lacks, or maps to another non-American spelling
ADD = {
    "annexe": "annex",
    "misspelt": "misspelled",
    "carcase": "carcass", "carcases": "carcasses",
    "reflexion": "reflection", "reflexions": "reflections",
    "deflexion": "deflection", "deflexions": "deflections",
    "oecumenical": "ecumenical",
    "caesarean": "cesarean", "caesareans": "cesareans",
    "draught": "draft", "draughts": "drafts",
    "draughtsman": "draftsman", "draughtsmen": "draftsmen",
    "faery": "fairy", "faeries": "fairies",
    "gipsy": "gypsy", "gipsies": "gypsies",
    "moustachioed": "mustachioed",
    "biassed": "biased", "biassing": "biasing",
    "chimaera": "chimera", "chimaeras": "chimeras", "chimaeric": "chimeric",
    "amoxycillin": "amoxicillin", "doyley": "doily",
    "bougainvillaea": "bougainvillea", "bougainvillaeas": "bougainvilleas",
    "paeony": "peony", "edoema": "edema",
    "featurise": "featurize", "featurised": "featurized",
    "featurises": "featurizes", "featurising": "featurizing",
    "featurisation": "featurization",
}

# a spelling British treats as a variant is listed only when the American one
# is this much more common, in wordfreq's zipf units
VARIANT_MARGIN = 0.4

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".venv", ".git", "node_modules", "target", "site", "__pycache__",
             "data", "estimator_cache", "results"}
EXTENSIONS = {".md", ".py", ".rs", ".js", ".sh", ".toml"}

# a run of word characters; only a run of letters alone is checked
TOKEN = re.compile(r"[A-Za-z0-9_]+")
LETTERS = re.compile(r"[A-Za-z]+")

# what Markdown keeps out of prose: fences, code spans, comments, URLs, links
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.M | re.S)
MD_SKIP = re.compile(r"`[^`\n]*`|<!--.*?-->|https?://\S+|\]\([^)\n]*\)",
                     re.S)


def parse_varcon(text):
    """British spelling -> American, for every word VarCon never marks `A`.

    A word marked plain `B` or `Z` is listed. One marked only as a British
    variant (`Bv`, `B.`, `ZV`, ...) is listed when the American spelling clears
    `VARIANT_MARGIN`. `ADD` comes first and wins.
    """
    from wordfreq import zipf_frequency
    american, mapped, variant = set(), {}, set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        groups = []
        line = line.split(" # ")[0].split(" | ")[0]
        for group in line.split(" / "):
            tags, sep, word = group.partition(": ")
            if not sep:
                break
            groups.append((tags.split(), word.strip()))
        else:
            # `_` is a spelling preferred in every region, so in American too
            preferred = [word for tags, word in groups
                         if "A" in tags or "_" in tags]
            for tags, word in groups:
                if "A" in tags or "_" in tags:
                    american.add(word)
                    continue
                if len(preferred) != 1:
                    continue
                if "B" in tags or "Z" in tags:
                    mapped.setdefault(word, set()).add(preferred[0])
                elif any(t[0] in "BZ" for t in tags) and "Av" not in tags:
                    mapped.setdefault(word, set()).add(preferred[0])
                    variant.add(word)
    pairs = dict(ADD)
    for word, targets in mapped.items():
        if word in pairs or word in american or word in KEEP or \
                len(targets) != 1:
            continue
        target = targets.pop()
        if not (LETTERS.fullmatch(word) and LETTERS.fullmatch(target)):
            continue
        if word in variant and zipf_frequency(target, "en") < \
                zipf_frequency(word, "en") + VARIANT_MARGIN:
            continue
        pairs[word] = target
    # a listed spelling never maps onto another listed one
    return {w: t for w, t in sorted(pairs.items()) if t not in pairs}


def load_pairs():
    with open(BRITISH_SPELLINGS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["pairs"]


def offsets(text):
    """The offset each line starts at, for tokenize's (row, col)."""
    starts, total = [0], 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        starts.append(total)
    return starts


def markdown_spans(text):
    """Prose: everything but fenced code, code spans, comments and URLs."""
    blocked = [m.span() for m in FENCE.finditer(text)]
    blocked += [m.span() for m in MD_SKIP.finditer(text)]
    blocked.sort()
    spans, at = [], 0
    for start, end in blocked:
        if start > at:
            spans.append((at, start))
        at = max(at, end)
    spans.append((at, len(text)))
    return spans


def python_spans(text):
    """Comments, and strings standing alone as a statement (docstrings)."""
    starts, spans = offsets(text), []
    position = lambda rc: starts[rc[0] - 1] + rc[1]  # noqa: E731
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    lead = {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENCODING}
    previous = tokenize.NEWLINE
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.COMMENT:
            spans.append((position(tok.start), position(tok.end)))
        elif tok.type == tokenize.STRING and previous in lead:
            after = next((t.type for t in tokens[i + 1:]
                          if t.type not in (tokenize.NL, tokenize.COMMENT)),
                         tokenize.ENDMARKER)
            if after in (tokenize.NEWLINE, tokenize.ENDMARKER):
                spans.append((position(tok.start), position(tok.end)))
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            previous = tok.type
    return spans


def slashed_spans(text, quotes):
    """`//` and `/* */` comments, skipping string literals in `quotes`."""
    spans, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in quotes:
            i += 1
            while i < n and text[i] != c:
                i += 2 if text[i] == "\\" else 1
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            spans.append((i, end))
            i = end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            spans.append((i, end))
            i = end
        else:
            i += 1
    return spans


def hash_spans(text):
    """`#` comments in shell and TOML, a `#` at a line start or after a space."""
    return [(m.start(1), m.end()) for m in
            re.finditer(r"(?:^|(?<=\s))(#[^\n]*)", text)]


def spans_for(path, text):
    ext = os.path.splitext(path)[1]
    if ext == ".md":
        return markdown_spans(text)
    if ext == ".py":
        return python_spans(text)
    if ext == ".rs":
        return slashed_spans(text, '"')
    if ext == ".js":
        return slashed_spans(text, "\"'`")
    return hash_spans(text)


def american(word, pairs):
    """The American spelling for ``word`` in its own case, or None."""
    if word in pairs:
        return pairs[word]
    lower = word.lower()
    if lower == word or lower not in pairs:
        return None
    if word.isupper() and len(word) > 1:
        return pairs[lower].upper()
    if word == lower.capitalize():
        return pairs[lower].capitalize()
    return None


def hits(text, spans, pairs):
    """(offset, British, American) for each listed word inside ``spans``."""
    found = []
    for start, end in spans:
        for m in TOKEN.finditer(text, start, end):
            word = m.group(0)
            if not LETTERS.fullmatch(word):
                continue
            fix = american(word, pairs)
            if fix is not None:
                found.append((m.start(), word, fix))
    return found


def fixed(text, found):
    """``text`` with each hit replaced, checked to change those words alone."""
    out, at = [], 0
    for offset, word, fix in found:
        out.append(text[at:offset])
        out.append(fix)
        at = offset + len(word)
    out.append(text[at:])
    result = "".join(out)
    before = TOKEN.findall(text)
    after = TOKEN.findall(result)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    if len(before) != len(after) or len(changed) != len(found) or \
            re.sub(TOKEN, "", text) != re.sub(TOKEN, "", result):
        raise SystemExit("a fix changed more than its words; nothing written")
    return result


def files_under(root):
    for here, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            if os.path.splitext(name)[1] in EXTENSIONS:
                yield os.path.join(here, name)


def fetch(name):
    with urllib.request.urlopen(VARCON + name, timeout=60) as reply:
        return reply.read().decode("latin-1")


def refresh(write):
    pairs = parse_varcon(fetch("varcon.txt"))
    # the license asks for its notice in every copy, so it travels with the list
    notice = fetch("README").split("\nCopyright\n=========\n", 1)[1].strip()
    print(f"{len(pairs):,} British spellings from {VARCON}varcon.txt")
    if write:
        write_json(BRITISH_SPELLINGS_PATH,
                   {"source": VARCON + "varcon.txt",
                    "notice": notice.splitlines(), "pairs": pairs})
        print(f"wrote {BRITISH_SPELLINGS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="replace each British spelling found")
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild the list from VarCon")
    parser.add_argument("paths", nargs="*", help="files to check (default: the repo)")
    args = parser.parse_args()
    if args.refresh:
        return refresh(args.write)

    pairs = load_pairs()
    paths = args.paths or list(files_under(ROOT))
    total = 0
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as file:
            text = file.read()
        found = hits(text, spans_for(path, text), pairs)
        if not found:
            continue
        total += len(found)
        starts = offsets(text)
        name = os.path.relpath(path, ROOT).replace(os.sep, "/")
        for offset, word, fix in found:
            line = next(i for i, s in enumerate(starts) if s > offset)
            print(f"  {name}:{line:<6} {word} -> {fix}")
        if args.write:
            with open(path, "w", encoding="utf-8", newline="") as file:
                file.write(fixed(text, found))
    if not total:
        print("no British spellings")
        return 0
    print(f"{total} British spellings" + ("; replaced" if args.write else
                                          "; pass --write to replace them"))
    return 0 if args.write else 1


if __name__ == "__main__":
    sys.exit(main())

"""The voice and comment rules in CLAUDE.md, checked rather than remembered.

Scans every document, comment and docstring for the phrases CLAUDE.md *Voice*
bans, for headers and comment blocks over their line limits, for measurements
and dates in source comments, and for CLAUDE.md's size.  A file listed in
``voice_pending.txt`` is still to be cleaned: it reports a count, and fails
only once it is clean and should come off the list.

    python tests/voice.py                    the repo
    python tests/voice.py PATH...            these files or directories
    python tests/voice.py --write-pending    list every failing file as pending
"""

import argparse
import ast
import io
import os
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PENDING_PATH = os.path.join(HERE, "voice_pending.txt")

SKIP_DIRS = {".venv", ".git", "node_modules", "target", "site", "__pycache__",
             "data", "estimator_cache", "results"}
SKIP_FILES = {"tests/voice.py", "tests/voice_pending.txt"}
EXTENSIONS = {".md", ".py", ".rs", ".js", ".sh"}

HEADER_LINES = 10      # a file header says what the module is
BLOCK_LINES = 4        # a run of plain comment lines in the body
CLAUDE_WORDS = 1500    # CLAUDE.md is a rule list

# (pattern, rule name); applied to prose and to comments alike
VOICE = [
    (r",\s*which is what\b", "trailing justification"),
    (r"\bthat is deliberate\b|\bon purpose\b|\bdeliberately\b", "deliberate"),
    (r"\bthe whole point\b|\bthe point is\b", "the point"),
    (r"\bnot a small one\b", "dramatized"),
    (r"\bworth (?:knowing|stating|saying|noting)\b", "worth knowing"),
    (r"\bgenuinely\b|\bhonest(?:ly)?\b|\bquietly\b|\bmerely\b", "filler"),
    (r"(?<!\bis )(?<!\bare )(?<!\bwas )(?<!\bwere )(?<!\bbe )(?<!\bbeen )"
     r"\bused to\b|\buntil now\b|\bha(?:s|ve) since\b"
     r"|\bwas tried\b|\btried, (?:rejected|reverted|not shipped)\b"
     r"|\brecently\b(?! used)", "history"),
]
VOICE = [(re.compile(pattern, re.I), name) for pattern, name in VOICE]

# a measurement or an edit date, which belong in README and not in source
MEASURE = re.compile(
    r"±|\b\d[\d,]*(?:\.\d+)?\s*(?:ms|secs?|seconds?|minutes?|hours?|MB|KB|GB"
    r"|%|σ|nats|×)(?!\w)|\b\d[\d,]*(?:\.\d+)? s(?!\w)|\b\d+(?:\.\d+)?x(?!\w)"
    r"|\b\d+\.\d+ guesses\b|\b\d+(?:\.\d+)?e-\d+\b")
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")

# text that is quoting a phrase rather than using it
QUOTED = re.compile(r'"[^"\n]*"|`[^`\n]*`')


class Hit:
    __slots__ = ("line", "rule", "text")

    def __init__(self, line, rule, text):
        self.line, self.rule, self.text = line, rule, text.strip()[:70]


def voice_hits(lines, offset=0, source=False):
    """Voice hits in numbered prose lines; ``source`` adds the comment rules."""
    hits = []
    for number, line in lines:
        clean = QUOTED.sub(lambda m: " " * len(m.group(0)), line)
        for pattern, rule in VOICE:
            if pattern.search(clean):
                hits.append(Hit(number + offset, rule, line))
        if source:
            if MEASURE.search(clean):
                hits.append(Hit(number + offset, "measurement in source", line))
            if DATE.search(clean):
                hits.append(Hit(number + offset, "date in source", line))
    return hits


def block_hits(comment_lines, header_end):
    """Runs of plain comment lines in the body longer than BLOCK_LINES."""
    hits, run, start = [], 0, 0
    for number in sorted(comment_lines) + [None]:
        if number is not None and number <= header_end:
            continue
        if number is not None and run and number == start + run:
            run += 1
            continue
        if run > BLOCK_LINES:
            hits.append(Hit(start, f"comment block of {run} lines", ""))
        run, start = (1, number) if number is not None else (0, 0)
    return hits


# ------------------------------------------------------------------ readers

def scan_markdown(text, name):
    lines, in_fence, skipping = [], False, False
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("#"):
            # the section that lists the banned phrases is allowed to
            skipping = stripped.lstrip("#").strip().lower() == "voice"
            continue
        if skipping:
            continue
        line = re.sub(r"<!--.*?-->", "", line)
        lines.append((number, line))
    hits = voice_hits(lines)
    if name == "CLAUDE.md":
        words = len(text.split())
        if words > CLAUDE_WORDS:
            hits.append(Hit(1, f"CLAUDE.md is {words:,} words, limit {CLAUDE_WORDS:,}", ""))
    return hits


def scan_python(text):
    hits, comment_lines = [], set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return [Hit(1, "could not tokenize", "")]
    prose = []
    for token in tokens:
        if token.type == tokenize.COMMENT:
            prose.append((token.start[0], token.string))
            if token.line.strip().startswith("#"):
                comment_lines.add(token.start[0])
    tree = ast.parse(text)
    header_end = 0
    for node in [tree] + [n for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                           ast.ClassDef))]:
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        first = node.body[0]
        start = first.lineno
        if node is tree:
            header_end = first.end_lineno
            length = doc.count("\n") + 1
            if length > HEADER_LINES:
                hits.append(Hit(start, f"header of {length} lines, limit {HEADER_LINES}", ""))
        for k, line in enumerate(doc.split("\n")):
            prose.append((start + k, line))
    hits += voice_hits(prose, source=True)
    hits += block_hits(comment_lines, header_end)
    return hits


TRAILING = re.compile(r"(?<![:\w])//(?!/)(.*)$")


def scan_slashed(text, header_marker, doc_marker=None):
    """Rust and JavaScript: ``//`` comments, ``/* */`` blocks, and a header
    made of ``header_marker`` lines at the top."""
    hits, prose, comment_lines = [], [], set()
    lines = text.split("\n")

    # the header: a /* */ block opening on line 1, else the leading run of
    # header_marker lines
    header_end = 0
    if lines and lines[0].strip().startswith("/*"):
        header_end = next((n for n, line in enumerate(lines, start=1)
                           if "*/" in line), len(lines))
    else:
        while header_end < len(lines) and \
                lines[header_end].strip().startswith(header_marker):
            header_end += 1

    in_block = in_doc = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if in_block or stripped.startswith("/*"):
            prose.append((number, stripped.strip("/* ")))
            if not in_block:
                in_doc = stripped.startswith("/**")   # documents an item
            if number > header_end and not in_doc:
                comment_lines.add(number)
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("//"):
            prose.append((number, stripped.lstrip("/")))
            if number > header_end and not (doc_marker and
                                            stripped.startswith(doc_marker)):
                comment_lines.add(number)   # /// documents an item, as a docstring does
            continue
        found = TRAILING.search(line)
        if found:
            prose.append((number, found.group(1)))
    if header_end > HEADER_LINES:
        hits.append(Hit(1, f"header of {header_end} lines, limit {HEADER_LINES}", ""))
    hits += voice_hits(prose, source=True)
    hits += block_hits(comment_lines, header_end)
    return hits


def scan_shell(text):
    hits, prose, comment_lines = [], [], set()
    header_end, header_lines, in_header = 0, 0, True
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if number == 1 and stripped.startswith("#!"):
            continue
        if stripped.startswith("#"):
            prose.append((number, stripped.lstrip("#")))
            if in_header:
                header_end = number
                header_lines += 1
            else:
                comment_lines.add(number)
            continue
        in_header = False
    if header_lines > HEADER_LINES:
        hits.append(Hit(1, f"header of {header_lines} lines, limit {HEADER_LINES}", ""))
    hits += voice_hits(prose, source=True)
    hits += block_hits(comment_lines, header_end)
    return hits


def scan(path, name):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    extension = os.path.splitext(path)[1]
    if extension == ".md":
        return scan_markdown(text, os.path.basename(name))
    if extension == ".py":
        return scan_python(text)
    if extension == ".rs":
        return scan_slashed(text, "//!", doc_marker="///")
    if extension == ".js":
        return scan_slashed(text, "//")
    return scan_shell(text)


# --------------------------------------------------------------------- main

def files_under(root):
    for path, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            if os.path.splitext(name)[1] in EXTENSIONS:
                yield os.path.join(path, name)


def relative(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def load_pending():
    if not os.path.exists(PENDING_PATH):
        return set()
    with open(PENDING_PATH, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()
                and not line.startswith("#")}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="default: the repo")
    parser.add_argument("--write-pending", action="store_true",
                        help="rewrite voice_pending.txt from what fails")
    parser.add_argument("--verbose", action="store_true",
                        help="print the hits in pending files too")
    args = parser.parse_args()

    if args.paths:
        paths = []
        for given in args.paths:
            paths += list(files_under(given)) if os.path.isdir(given) else [given]
        pending = set()
    else:
        paths = list(files_under(ROOT))
        pending = load_pending()

    failed, dirty, total = [], [], 0
    for path in paths:
        name = relative(path)
        if name in SKIP_FILES:
            continue
        hits = scan(path, name)
        total += len(hits)
        if not hits:
            if name in pending:
                failed.append(name)
                print(f"{name}: clean; remove it from voice_pending.txt")
            continue
        dirty.append(name)
        if name in pending:
            print(f"{name}: pending, {len(hits)} hits")
            if not args.verbose:
                continue
        else:
            failed.append(name)
        for hit in hits:
            where = f"{name}:{hit.line}"
            print(f"  {where:40s} {hit.rule}" + (f"   {hit.text}" if hit.text else ""))

    if args.write_pending:
        with open(PENDING_PATH, "w", encoding="utf-8", newline="") as handle:
            handle.write("# files the voice check has not yet been applied to;"
                         " tests/voice.py fails once one is clean\n")
            handle.write("\n".join(sorted(dirty)) + "\n")
        print(f"wrote {len(dirty)} pending files")
        return 0

    print(f"{total} hits in {len(dirty)} files; {len(pending)} pending")
    if failed:
        print(f"failed: {len(failed)} files")
        return 1
    print("every file outside the pending list is clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

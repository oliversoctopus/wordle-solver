"""The voice and comment rules in CLAUDE.md, checked rather than remembered.

Scans every document, comment and docstring for the phrases CLAUDE.md *Voice*
bans, for headers, docstrings and comment blocks over their line limits, for
measurements and dates in source comments and help text, for CLAUDE.md's
size, for the density of dashes, "rather than" and short closing sentences in
a document, and for sentence lengths too uniform to be a person's.

    python tests/voice.py                    the repo
    python tests/voice.py PATH...            these files or directories
"""

import argparse
import ast
import io
import os
import re
import sys
import textwrap
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SKIP_DIRS = {".venv", ".git", "node_modules", "target", "site", "__pycache__",
             "data", "estimator_cache", "results"}
SKIP_FILES = {"tests/voice.py"}
EXTENSIONS = {".md", ".py", ".rs", ".js", ".sh", ".toml"}
NAMES = {".gitignore", ".gitattributes"}    # commented like a shell script

HEADER_LINES = 10      # a file header says what the module is
DOC_LINES = 6          # prose lines in a docstring or doc comment
BLOCK_LINES = 2        # a comment is one or two lines
CLAUDE_WORDS = 1500    # CLAUDE.md is a rule list

# cadence ceilings per thousand words of prose, with a floor for short documents
DENSITY = [
    (re.compile("—"), "em-dashes", 0.5),
    (re.compile(r"\brather than\b"), "rather than", 0.5),
]
CLOSER_WORDS = 4       # a closing sentence this short is a punchline
CLOSERS_PER_K = 1.0
DENSITY_FLOOR = 2

# sentence length varies at least this much in every writer VOICE.md names
LENGTH_SPREAD_FLOOR = 0.45
LENGTH_SPREAD_MIN_SENTENCES = 30

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
    r"|\b\d+\.\d+ guesses\b|\b\d+(?:\.\d+)?e-\d+\b|\+-\s*\d")
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")

# text that is quoting a phrase rather than using it
QUOTED = re.compile(r'"[^"\n]*"|`[^`\n]*`')

# a section banner, not a comment
BANNER = re.compile(r"^\s*-{4,}")


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
            hits.append(Hit(start, f"comment block of {run} lines, limit {BLOCK_LINES}", ""))
        run, start = (1, number) if number is not None else (0, 0)
    return hits


def doc_hit(start, lines):
    """A doc comment with more than DOC_LINES prose lines; blank lines and
    indented examples do not count."""
    lines = textwrap.dedent("\n".join(lines)).split("\n")
    length = sum(1 for line in lines
                 if line.strip() and not line.startswith(" "))
    if length > DOC_LINES:
        return [Hit(start, f"doc comment of {length} prose lines, "
                           f"limit {DOC_LINES}", "")]
    return []


# ------------------------------------------------------------------ readers

def density_hits(lines, rule_list):
    """Dashes, "rather than" and short closing sentences per thousand words."""
    prose = [line for _, line in lines
             if line.strip() and not line.lstrip().startswith(("|", "$$", ">"))]
    words = sum(len(line.split()) for line in prose)
    hits = []
    for pattern, rule, per_k in DENSITY:
        count = sum(len(pattern.findall(line)) for line in prose)
        limit = max(DENSITY_FLOOR, -(-per_k * words // 1000))
        if count > limit:
            hits.append(Hit(1, f"{count} {rule} in prose, limit {int(limit)}", ""))
    if rule_list:
        return hits
    closers, lengths = 0, []
    for line in prose:
        plain = re.sub(r"`[^`]*`", "x", line).rstrip()
        sentences = re.split(r"(?<=[.!?])\s+", plain)
        if len(sentences) > 1 and len(sentences[-1].split()) <= CLOSER_WORDS:
            closers += 1
        lengths += [len(s.split()) for s in sentences if s.strip()]
    limit = max(DENSITY_FLOOR, -(-CLOSERS_PER_K * words // 1000))
    if closers > limit:
        hits.append(Hit(1, f"{closers} short closing sentences, limit {int(limit)}", ""))
    if len(lengths) >= LENGTH_SPREAD_MIN_SENTENCES:
        mean = sum(lengths) / len(lengths)
        spread = (sum((n - mean) ** 2 for n in lengths) / len(lengths)) ** 0.5 / mean
        if spread < LENGTH_SPREAD_FLOOR:
            hits.append(Hit(1, f"sentence lengths vary by {spread:.2f} of the mean, "
                               f"floor {LENGTH_SPREAD_FLOOR}", ""))
    return hits


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
    hits += density_hits(lines, rule_list=name == "CLAUDE.md")
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
            if token.line.strip().startswith("#") and not BANNER.match(token.string[1:]):
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
        length = doc.count("\n") + 1
        if node is tree:
            header_end = first.end_lineno
            if length > HEADER_LINES:
                hits.append(Hit(start, f"header of {length} lines, limit {HEADER_LINES}", ""))
        else:
            hits += doc_hit(start, ast.get_docstring(node).split("\n"))
        for k, line in enumerate(doc.split("\n")):
            prose.append((start + k, line))
    prose += help_strings(tree)
    hits += voice_hits(prose, source=True)
    hits += block_hits(comment_lines, header_end)
    return hits


def help_strings(tree):
    """The literal text of every ``help=`` and ``description=`` argument,
    which the comment rules cover as they cover a comment."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg not in ("help", "description"):
                continue
            parts = (keyword.value.values
                     if isinstance(keyword.value, ast.JoinedStr)
                     else [keyword.value])
            text = "".join(part.value for part in parts
                           if isinstance(part, ast.Constant)
                           and isinstance(part.value, str))
            if text:
                out.append((keyword.value.lineno, text.replace("%%", "%")))
    return out


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
    block_start = 0
    doc_lines, block_lines = [], []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if in_block or stripped.startswith("/*"):
            prose.append((number, stripped.strip("/* ")))
            if not in_block:
                in_doc = stripped.startswith("/**")   # documents an item
                block_start = number
                block_lines = []
            block_lines.append(stripped.strip("/* "))
            if number > header_end and not in_doc and not BANNER.match(stripped.strip("/* ")):
                comment_lines.add(number)
            in_block = "*/" not in stripped
            if not in_block and in_doc and block_start > header_end:
                hits += doc_hit(block_start, block_lines)
            continue
        is_doc = bool(doc_marker) and stripped.startswith(doc_marker)
        if is_doc:
            doc_lines.append(stripped[len(doc_marker):].removeprefix(" "))
        else:
            if doc_lines:
                hits += doc_hit(number - len(doc_lines), doc_lines)
            doc_lines = []
        if stripped.startswith("//"):
            prose.append((number, stripped.lstrip("/")))
            if number > header_end and not is_doc and \
                    not BANNER.match(stripped.lstrip("/")):
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
            elif not BANNER.match(stripped.lstrip("#")):
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
            if os.path.splitext(name)[1] in EXTENSIONS or name in NAMES:
                yield os.path.join(path, name)


def relative(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="default: the repo")
    args = parser.parse_args()

    paths = []
    for given in args.paths or [ROOT]:
        paths += list(files_under(given)) if os.path.isdir(given) else [given]

    failed, total = [], 0
    for path in paths:
        name = relative(path)
        if name in SKIP_FILES:
            continue
        hits = scan(path, name)
        total += len(hits)
        if hits:
            failed.append(name)
        for hit in hits:
            where = f"{name}:{hit.line}"
            print(f"  {where:40s} {hit.rule}" + (f"   {hit.text}" if hit.text else ""))

    print(f"{total} hits in {len(failed)} files")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

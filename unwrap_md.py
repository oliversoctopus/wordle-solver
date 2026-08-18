"""Rewrite Markdown as one line per paragraph, and check that it stays that way.

Every `.md` file here is unwrapped; CLAUDE.md's *Conventions* say why.
Run with no arguments to report which files still hold a wrapped paragraph, and
with ``--write`` to convert them.  Verifying by default is the same bargain
``word_lists.py`` and ``solutions.py`` strike: this rewrites files by hand-edited
prose, so it says what it would do and stops.

Fenced code, ``$$`` math blocks, tables, headings and rules are passed through
untouched; prose, list items and blockquotes are joined.  Two guards decide
whether the result is written at all.  The first collapses all whitespace and
compares, so no word may be added, dropped or reordered.  The second fingerprints
the file by blank-line-delimited chunk, typed by the chunk's first line, which
wrapping cannot change.  Either one failing aborts the file untouched.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__"}

LIST_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+")
RULE_RE = re.compile(r"-{3,}|\*{3,}|_{3,}")


def unwrap(text):
    """``text`` with every paragraph, list item and blockquote on one line."""
    out, buf, prefix = [], [], ""
    in_fence = in_math = False

    def flush():
        nonlocal buf, prefix
        if buf:
            out.append(prefix + " ".join(buf))
        buf, prefix = [], ""

    for line in text.split("\n"):
        stripped = line.strip()

        if in_fence:
            out.append(line)
            if stripped.startswith("```"):
                in_fence = False
            continue
        if in_math:
            out.append(line)
            if stripped.endswith("$$"):
                in_math = False
            continue

        if stripped.startswith("```"):
            flush(); out.append(line); in_fence = True; continue
        if stripped.startswith("$$"):
            flush(); out.append(line)
            if not (stripped.endswith("$$") and len(stripped) > 2):
                in_math = True
            continue
        if not stripped:
            flush(); out.append(""); continue
        if stripped.startswith("#") or stripped.startswith("|"):
            flush(); out.append(line); continue
        if RULE_RE.fullmatch(stripped):
            flush(); out.append(line); continue

        if stripped.startswith(">"):
            content = stripped[1:].strip()
            if not content:
                flush(); out.append(line); continue
            if prefix == "> ":
                buf.append(content)
            else:
                flush(); prefix = "> "; buf = [content]
            continue

        match = LIST_RE.match(line)
        if match:
            # CommonMark lets an ordered list interrupt a paragraph only at
            # "1.", so a sentence wrapping onto "18. If those still come out"
            # is prose.  Inside an open list any marker starts the next item.
            ordered = match.group(2).endswith(".")
            if ordered and buf and not prefix and match.group(2) != "1.":
                buf.append(stripped)
                continue
            flush()
            prefix = match.group(0)
            buf = [line[match.end():].strip()]
            continue

        if prefix == "> ":
            flush()
        buf.append(stripped)

    flush()
    return "\n".join(out)


def chunk_kinds(text):
    """One entry per blank-line-delimited chunk, typed by its first line."""
    kinds, in_fence, fresh = [], False, True
    for line in text.split("\n"):
        stripped = line.strip()
        if in_fence:
            if stripped.startswith("```"):
                in_fence = False
            continue
        if not stripped:
            fresh = True
            continue
        if fresh:
            kinds.append(_kind(line))
            fresh = False
        if stripped.startswith("```"):
            in_fence = True
    return kinds


def _kind(line):
    stripped = line.strip()
    if stripped.startswith("```"):
        return "fence"
    if stripped.startswith("#"):
        return "h%d" % (len(stripped) - len(stripped.lstrip("#")))
    if stripped.startswith("|"):
        return "tr"
    if stripped.startswith("$$"):
        return "math"
    if RULE_RE.fullmatch(stripped):
        return "rule"
    if stripped.startswith(">"):
        return "quote"
    if LIST_RE.match(line):
        return "li"
    return "p"


def _words(text):
    """The prose, with blockquote markers and all whitespace normalized away."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text)).strip()


def check(old, new):
    """A complaint about converting ``old`` to ``new``, or None."""
    if _words(old) != _words(new):
        return "text content would change"
    before, after = chunk_kinds(old), chunk_kinds(new)
    if before != after:
        at = next(i for i in range(max(len(before), len(after)))
                  if before[i:i + 1] != after[i:i + 1])
        return "block structure would change at chunk %d (%s -> %s)" % (
            at, before[at:at + 3], after[at:at + 3])
    return None


def _name(path):
    """``path`` relative to the repo, or as given if it lies outside it."""
    relative = os.path.relpath(path, HERE)
    return path if relative.startswith("..") else relative


def markdown_files(root):
    for path, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(names):
            if name.endswith(".md"):
                yield os.path.join(path, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="default: every .md in the repo")
    parser.add_argument("--write", action="store_true",
                        help="rewrite the files instead of reporting")
    args = parser.parse_args()

    paths = args.paths or list(markdown_files(HERE))
    wrapped, refused = [], []

    for path in paths:
        with open(path, encoding="utf-8", newline="") as handle:
            old = handle.read()
        if "\r" in old:
            refused.append((path, "CRLF line endings"))
            continue
        if any(line.endswith("  ") for line in old.split("\n")):
            refused.append((path, "trailing double-space hard break"))
            continue

        new = unwrap(old)
        if new == old:
            continue
        complaint = check(old, new)
        if complaint:
            refused.append((path, complaint))
            continue

        wrapped.append(_name(path))
        if args.write:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(new)
            print("%s: %d -> %d lines"
                  % (_name(path), old.count("\n"), new.count("\n")))

    for path, why in refused:
        print("%s: %s, left alone" % (_name(path), why))
    if wrapped and not args.write:
        print("wrapped paragraphs in: %s" % ", ".join(wrapped))
        print("pass --write to unwrap them")
    elif not wrapped:
        print("every file checked is already one line per paragraph")
    # non-zero when there is something to do, so a plain run is a check
    return 1 if refused or (wrapped and not args.write) else 0


if __name__ == "__main__":
    sys.exit(main())

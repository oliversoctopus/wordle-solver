"""Every pointer into the documents and into the code resolves to something.

The documents cross-reference each other, `analysis/README.md` pairs each
script with the claim it backs, and the scripts say the same in their
docstrings.  None of that is checked by anything else, and all of it drifts:
a heading gets reworded and the pointers to it quietly stop pointing.

So collect every reference and resolve it against the headings themselves.
The oracle is the documents and the code, so nothing here is a figure written
down by hand.

Two forms point at a section.  A *named* one quotes the heading in italics --
``README *Results*``, ``SOLVER_EXPLAINED *A better map, and next to no*`` -- and
matches on a prefix, since a reference often names a heading's opening words
rather than all of it.  A *numbered* one is SOLVER_EXPLAINED's ``§12`` or
``section 12`` or ``6.4``, which resolves against that document's numbering.

A third points at code.  CLAUDE.md's convention is to name a symbol rather
than a line -- ``priors.build_weights``, ``WordleSolver.guess_pool``,
``source::Columns`` -- on the grounds that a name outlives an edit above it.
It outlives a *rename* only if something looks, which is what this does: the
owner has to be a module or a class here, and the name has to be something it
defines.  The Python is read with ``ast``, so a method or an attribute set in
``__init__`` resolves as readily as a top-level ``def``; the Rust is matched
against its item declarations.  An owner that is neither is reported rather
than assumed to be an import, and ``EXTERNAL`` is the list of ones that are.

Whitespace is collapsed before matching.  Python is hard-wrapped, so a title
inside a docstring is routinely split across a line break, and a check that
read one line at a time would miss exactly the references hardest to grep for.
"""

import ast
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DOCS = {"README": "README.md", "SOLVER_EXPLAINED": "SOLVER_EXPLAINED.md"}

# where references are written: the prose documents and every analysis script
SOURCES = ["README.md", "SOLVER_EXPLAINED.md", "ABOUT_SOLVER_EXPLAINED.md",
           "CLAUDE.md", os.path.join("analysis", "README.md"),
           os.path.join("tests", "README.md")]

# where the Python and the Rust a pointer can name are kept
PACKAGES = [".", "analysis", "tests"]
CRATE = os.path.join("core", "src")
CRATE_NAME = "wordle_core"

NAMED = re.compile(
    r"(README|SOLVER_EXPLAINED)(?:\.md)?\s*(?:§\s*)?\*([^*]{4,90})\*")
NUMBERED = re.compile(
    r"SOLVER_EXPLAINED(?:\.md)?\s*(?:§|sections?)?\s*(\d+(?:\.\d+)?)")
# "SOLVER_EXPLAINED sections 9 and 10" -- the second number rides on the first
ALSO = re.compile(r"\A\s*(?:and|,)\s*§?\s*(\d+(?:\.\d+)?)")

# a code pointer fills its whole span, so `editor.wordWrap: on` is not one
CODE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*)(\.|::)([A-Za-z_][A-Za-z0-9_]*)`")
# `app.js` and `feedback_table.npy` are files, and match the same shape
EXTENSIONS = {"py", "js", "json", "md", "sh", "rs", "npy", "html", "css",
              "toml", "txt", "lock", "ipynb", "ps1", "yml", "svg", "bat"}
# modules a pointer may name that are nobody's file here
EXTERNAL = {"np", "http", "os", "re", "sys", "io", "json", "math", "time",
            "numpy", "numba", "std", "core"}

RUST_ITEM = "fn|struct|enum|trait|type|const|static|union|mod|macro_rules"


def collapse(text):
    return re.sub(r"\s+", " ", text)


def read(path):
    with io.open(os.path.join(ROOT, path), encoding="utf-8", newline="") as f:
        return f.read()


def headings(markdown):
    """Every heading, and every section number it declares."""
    names, numbers = [], set()
    for line in markdown.split("\n"):
        match = re.match(r"#{1,6}\s+(.+?)\s*$", line)
        if not match:
            continue
        title = match.group(1)
        number = re.match(r"(\d+(?:\.\d+)?)\.?\s+(.*)", title)
        if number:
            numbers.add(number.group(1))
            title = number.group(2)
        # strip the emphasis a heading may carry, so *priors* matches priors
        names.append(collapse(re.sub(r"[*`]", "", title)))
    return names, numbers


def declared(nodes):
    """The names these statements bind, ignoring where they nest."""
    names = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Attribute):
                    names.add(target.attr)   # self.thing, in a method
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def python_names(text):
    """What a module defines, and what each of its classes defines."""
    tree = ast.parse(text)
    classes = {node.name: declared(ast.walk(node))
               for node in tree.body if isinstance(node, ast.ClassDef)}
    names = declared(tree.body)
    for members in classes.values():
        names |= members                     # exact.solve is a method
    return names, classes


def code():
    """Every module and class a pointer may name, and what each defines."""
    python, rust, classes = {}, {}, {}
    for package in PACKAGES:
        for name in sorted(os.listdir(os.path.join(ROOT, package))):
            if not name.endswith(".py") or name[:-3] in python:
                continue
            names, found = python_names(read(os.path.join(package, name)))
            python[name[:-3]] = names
            classes.update(found)
    for name in sorted(os.listdir(os.path.join(ROOT, CRATE))):
        if name.endswith(".rs"):
            rust[name[:-3]] = read(os.path.join(CRATE, name))
    return python, rust, classes


def rust_declares(text, item):
    return re.search(r"\b(?:%s)\s+%s\b"
                     % (RUST_ITEM, re.escape(item)), text) is not None


def code_problem(owner, separator, symbol, known):
    """Why this pointer resolves to nothing, or None if it does."""
    python, rust, classes = known
    if separator == "::":
        if owner == CRATE_NAME:
            return None if symbol in rust else "no such module in the crate"
        if owner not in rust:
            return None if owner in EXTERNAL else "no such module in the crate"
        return None if rust_declares(rust[owner], symbol) else \
            "no such item in %s.rs" % owner
    if symbol.lower() in EXTENSIONS:
        return None                          # a filename, not a pointer
    if owner in PACKAGES:
        return None if symbol in python else "no such module in %s" % owner
    if owner in python:
        return None if symbol in python[owner] else \
            "no such name in %s.py" % owner
    if owner in classes:
        return None if symbol in classes[owner] else \
            "no such member of %s" % owner
    return None if owner in EXTERNAL else "no module or class of that name"


def sources():
    yield from SOURCES
    analysis = os.path.join(ROOT, "analysis")
    for name in sorted(os.listdir(analysis)):
        if name.endswith(".py"):
            yield os.path.join("analysis", name)


def check():
    docs = {key: headings(read(path)) for key, path in DOCS.items()}
    known = code()
    problems = []

    for source in sources():
        text = collapse(read(source))
        for match in NAMED.finditer(text):
            doc, quoted = match.group(1), collapse(match.group(2)).strip(" :")
            names = docs[doc][0]
            wanted = re.sub(r"[*`]", "", quoted).lower()
            if not any(n.lower().startswith(wanted) for n in names):
                problems.append((source, "%s *%s*" % (doc, quoted),
                                 "no heading starts with that"))
        for match in NUMBERED.finditer(text):
            found = [match.group(1)]
            tail = text[match.end():]
            while True:
                more = ALSO.match(tail)
                if not more:
                    break
                found.append(more.group(1))
                tail = tail[more.end():]
            for number in found:
                if number not in docs["SOLVER_EXPLAINED"][1]:
                    problems.append((source, "SOLVER_EXPLAINED %s" % number,
                                     "no such section"))
        for match in CODE.finditer(text):
            owner, separator, symbol = match.groups()
            why = code_problem(owner, separator, symbol, known)
            if why:
                problems.append((source, owner + separator + symbol, why))
    return problems


def main():
    problems = check()
    for source, reference, why in problems:
        print("  %-28s %-52s %s" % (source, reference, why))
    if problems:
        print("\n%d pointer%s point nowhere" % (
            len(problems), "" if len(problems) == 1 else "s"))
        return 1
    print("every pointer into the documents and the code resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())

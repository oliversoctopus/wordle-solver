"""Every reference to a document section resolves to a section that exists.

The documents cross-reference each other, `analysis/README.md` pairs each
script with the claim it backs, and the scripts say the same in their
docstrings.  None of that is checked by anything else, and all of it drifts:
a heading gets reworded and the pointers to it quietly stop pointing.

So collect every reference and resolve it against the headings themselves.
The oracle is the documents, so nothing here is a figure written down by hand.

Two forms are recognized.  A *named* one quotes the heading in italics --
``README *Results*``, ``SOLVER_EXPLAINED *The result worth framing*`` -- and
matches on a prefix, since a reference often names a heading's opening words
rather than all of it.  A *numbered* one is SOLVER_EXPLAINED's ``§12`` or
``section 12`` or ``6.4``, which resolves against that document's numbering.

Whitespace is collapsed before matching.  Python is hard-wrapped, so a title
inside a docstring is routinely split across a line break, and a check that
read one line at a time would miss exactly the references hardest to grep for.
"""

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DOCS = {"README": "README.md", "SOLVER_EXPLAINED": "SOLVER_EXPLAINED.md"}

# where references are written: the prose documents and every analysis script
SOURCES = ["README.md", "SOLVER_EXPLAINED.md", "ABOUT_SOLVER_EXPLAINED.md",
           "CLAUDE.md", os.path.join("analysis", "README.md")]

NAMED = re.compile(
    r"(README|SOLVER_EXPLAINED)(?:\.md)?\s*(?:§\s*)?\*([^*]{4,90})\*")
NUMBERED = re.compile(
    r"SOLVER_EXPLAINED(?:\.md)?\s*(?:§|sections?)?\s*(\d+(?:\.\d+)?)")
# "SOLVER_EXPLAINED sections 9 and 10" -- the second number rides on the first
ALSO = re.compile(r"\A\s*(?:and|,)\s*§?\s*(\d+(?:\.\d+)?)")


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


def sources():
    yield from SOURCES
    analysis = os.path.join(ROOT, "analysis")
    for name in sorted(os.listdir(analysis)):
        if name.endswith(".py"):
            yield os.path.join("analysis", name)


def check():
    docs = {key: headings(read(path)) for key, path in DOCS.items()}
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
    return problems


def main():
    problems = check()
    for source, reference, why in problems:
        print("  %-28s %-52s %s" % (source, reference, why))
    if problems:
        print("\n%d reference%s point nowhere" % (
            len(problems), "" if len(problems) == 1 else "s"))
        return 1
    print("every document reference resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())

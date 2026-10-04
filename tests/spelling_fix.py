"""`spelling.py` fixes prose and nothing else, over words drawn from its list.

Each drawn British spelling is planted in prose, in an identifier, in a string
literal and in a code span, in Markdown, Python, Rust and JS. The fix must turn
the prose copies into the listed American spelling in the same case, leave
every other copy alone, and leave nothing for a second pass. A capitalized
entry is a proper noun and must match in its own case.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spelling

DRAWS = 200
SEED = 20261004

failures = 0


def report(ok, what):
    global failures
    if not ok:
        failures += 1
        print(f"FAIL  {what}")


def cases(word):
    """(extension, text, the prose copies, the copies that must stay)."""
    upper, title = word.upper(), word.capitalize()
    return [
        (".md", f"# {title}\n\nThe {word} and {upper}, not `{word}` or "
                f"{word}_x or [a](https://x.org/{word}).\n\n```\n{word}\n```\n",
         [title, word, upper], [f"`{word}`", f"{word}_x", f"/{word})",
                                f"```\n{word}\n```"]),
        (".py", f'"""The {word}."""\n\n{word}_x = "{word}"  # a {title}\n',
         [word, title], [f'{word}_x = "{word}"']),
        (".rs", f'/// The {word}.\nconst X: &str = "{word} // {word}"; '
                f'// {upper}\n',
         [word, upper], [f'"{word} // {word}"']),
        (".js", f"// the {word}\nconst {word}_x = '{word}'; /* {title} */\n",
         [word, title], [f"const {word}_x = '{word}'"]),
    ]


def main():
    pairs = spelling.load_pairs()
    report(not any(t in pairs for t in pairs.values()),
           "no American spelling is itself listed as British")
    rng = random.Random(SEED)
    # a capitalized entry is a proper noun, and matches in its own case only
    proper = sorted(w for w in pairs if not w.islower())
    for word in rng.sample(proper, min(DRAWS, len(proper))):
        text = f"The {word} and the {word.lower()}.\n"
        found = spelling.hits(text, spelling.spans_for("x.md", text), pairs)
        want = [word] + ([word.lower()] if word.lower() in pairs else [])
        report([w for _, w, _ in found] == want, f"{word}: proper noun")
    for word in rng.sample(sorted(w for w in pairs if w.islower()), DRAWS):
        for ext, text, prose, kept in cases(word):
            path = "x" + ext
            found = spelling.hits(text, spelling.spans_for(path, text), pairs)
            out = spelling.fixed(text, found)
            want = [spelling.american(w, pairs) for w in prose]
            report(sorted(w for _, w, _ in found) == sorted(prose),
                   f"{word}{ext}: found {[w for _, w, _ in found]}")
            report(all(w in out for w in want),
                   f"{word}{ext}: the fix lacks one of {want}")
            report(all(k in out for k in kept),
                   f"{word}{ext}: a copy outside prose changed")
            report(not spelling.hits(out, spelling.spans_for(path, out), pairs),
                   f"{word}{ext}: a second pass still finds something")
    print(f"{DRAWS} spellings drawn with seed {SEED}, each planted in four "
          f"languages")
    print(f"\n{failures} failures" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

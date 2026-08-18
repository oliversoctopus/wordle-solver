"""The C++ core agrees with the Python it was ported from.

Not a test of either implementation on its own: each check runs the two against
the same inputs and diffs them, so the Python is the oracle for the port in the
same way the other files here use one -- the second implementation, the answer
record, or a property that holds of every board.  Nothing below asserts a figure
written down by hand.

Skips rather than fails when the C++ has not been built, so a checkout without
a compiler still runs the rest of the suite.  Build it with ``./build.sh``.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(ROOT, "build")

# (label, binary, what its agreement establishes)
CHECKS = [
    ("the coloring rule", "parity_feedback",
     "every one of the 12,947^2 pairs matches feedback_table.npy"),
]


def run(binary):
    """``(ok, output)`` for one parity binary, or None if it is not built."""
    path = os.path.join(BUILD, binary + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(path):
        return None, path
    done = subprocess.run([path, ROOT], capture_output=True, text=True)
    return done.returncode == 0, done.stdout + done.stderr


def main():
    results = []
    for label, binary, establishes in CHECKS:
        ok, output = run(binary)
        if ok is None:
            print(f"skip  {label}: {output} not built")
            results.append(None)
            continue
        print(f"{'ok  ' if ok else 'FAIL'}  {label} -- {establishes}")
        for line in output.rstrip().splitlines():
            print(f"      {line.strip()}")
        results.append(ok)

    failed = results.count(False)
    skipped = results.count(None)
    if failed:
        print(f"\n{failed} of {len(results)} disagree with the Python")
        return 1
    if skipped == len(results):
        print("\nnothing built; run ./build.sh")
        return 0
    print(f"\nall {len(results) - skipped} agree with the Python")
    return 0


if __name__ == "__main__":
    sys.exit(main())

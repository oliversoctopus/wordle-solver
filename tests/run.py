"""Everything here that is quick, in one go.

    python tests/run.py           the fast ones, a few seconds
    python tests/run.py --full    also the ones that load the table

Each file below also runs on its own, and says more when it does.
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FAST = [
    ("the board's editing rules", ["node", os.path.join(HERE, "board.test.js")]),
    ("the two front ends agree", [sys.executable, os.path.join(HERE, "front_ends_agree.py")]),
    ("the fitted f is current", [sys.executable, os.path.join(HERE, "fit_provenance.py")]),
]

SLOW = [
    ("the two front ends recommend alike",
     [sys.executable, os.path.join(HERE, "front_ends_agree.py"), "--full"]),
    ("one word per split, and the words it stands for",
     [sys.executable, os.path.join(HERE, "splits.py")]),
    ("the scoring rules are what they claim",
     [sys.executable, os.path.join(HERE, "scoring.py")]),
]


def run(label, command):
    if command[0] == "node" and not shutil.which("node"):
        print(f"---- {label}: skipped, node is not on the path")
        return None
    # the children write straight to the terminal, so anything of ours still
    # sitting in a buffer would surface after their output rather than above it
    print(f"---- {label}", flush=True)
    result = subprocess.run(command, cwd=ROOT)
    print(flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true",
                        help="also run what needs the feedback table")
    args = parser.parse_args()

    results = [(label, run(label, command))
               for label, command in FAST + (SLOW if args.full else [])]

    failed = [label for label, ok in results if ok is False]
    skipped = [label for label, ok in results if ok is None]
    for label in skipped:
        print(f"skipped  {label}")
    for label in failed:
        print(f"failed   {label}")
    if not failed:
        print(f"passed   {len(results) - len(skipped)} of {len(results)}"
              + (", and nothing here needs the table" if not args.full else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

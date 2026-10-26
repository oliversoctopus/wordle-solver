"""A run that drops arms rebuilds each arm table its last run recorded.

``common.arm_table`` carries over the rows of the arms ``--repeat-only``
skipped.  Every arm table in the results files is cut to each proper subset of
its rows and merged back against the whole, which has to come out unchanged.
"""

import contextlib
import io
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import common  # noqa: E402  -- needs the path above

# the tables recorded through `common.arm_table`, by script
ARM_TABLES = {"results_tables": ["priors_table", "split_table", "entropy_table",
                                 "bonus_table"],
              "hard_mode": ["table", "entropy.table"]}

failures = checked = 0
common._DROPPED.append("an arm")
for script, keys in ARM_TABLES.items():
    for key in keys:
        whole = common.result(script, key)
        if whole is None:
            print(f"FAIL  {script} records no {key}")
            failures += 1
            continue
        lines = whole.split("\n")
        columns = [c.strip() for c in lines[0].strip("|").split("|")]
        rows = [[c.strip() for c in line.strip("|").split("|")]
                for line in lines[2:]]
        if common.md_table(columns, rows) != whole:
            print(f"FAIL  {script} {key} does not parse back to itself")
            failures += 1
            continue
        common.last = lambda k, default=None, whole=whole: whole
        for n in range(1, len(rows)):
            for kept in itertools.combinations(rows, n):
                checked += 1
                # the merge prints each row it carries
                with contextlib.redirect_stdout(io.StringIO()):
                    merged = common.arm_table(columns, list(kept), key)
                if merged != whole:
                    print(f"FAIL  {script} {key} keeping "
                          f"{[row[0] for row in kept]}")
                    failures += 1

print(f"{checked} cuts of {sum(map(len, ARM_TABLES.values()))} tables, "
      f"{failures} failed")
sys.exit(1 if failures else 0)

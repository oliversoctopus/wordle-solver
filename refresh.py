"""The routine refresh: new answers, a refit, the live scripts, the documents.

Runs each step in order and stops at the first failure; `--from` resumes
there.  analysis/README.md *Rerunning them* says which scripts are live.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from priors import ALL_CONFIGS, depends_on_answer_record

HERE = os.path.dirname(os.path.abspath(__file__))

# the live scripts in order: (script, its flags, whether it takes --repeat-only)
LIVE = [
    ("regime", [], False),
    ("repeat_shapes", [], False),
    ("drift", [], False),
    ("curation_shapes", [], False),
    ("id_eras", [], False),
    ("replay", [], False),
    ("replay", ["--all-priors"], False),
    ("replay", ["--shapes"], False),
    ("results_tables", [], True),
    ("head_to_head", [], True),
    ("head_to_head", ["--baseline"], True),
    ("hard_mode", ["--all-priors"], True),
    ("hard_mode", ["--all-priors", "--mode", "entropy"], True),
    ("hard_mode_rules", [], True),
    ("hard_mode_rules", ["--no-endgame"], True),
    ("hard_mode_rules", ["--mode", "entropy"], True),
    ("turn2_regret", [], False),
    ("prior_basin", [], False),
]


def prior_flags(config):
    """The `wordle_solver.py` flags that name this prior."""
    flags = []
    if not config.repeat:
        flags.append("--no-repeat-prior")
    if not config.curation:
        flags.append("--no-curation-prior")
    flags += ["--repeat-shape", config.repeat_shape,
              "--curation-shape", config.curation_shape]
    return flags


def steps(repeat_only):
    """Every step as (name, argv), in the order they run."""
    py, sh = sys.executable, shutil.which("bash") or "bash"
    only = ["--repeat-only"] if repeat_only else []
    out = [("build", [sh, "build.sh"]),
           ("solutions", [py, "solutions.py", "--write"])]
    for config in ALL_CONFIGS:
        if repeat_only and not depends_on_answer_record(config):
            continue
        out.append(("compare " + config.slug,
                    [py, "wordle_solver.py", "compare"] + prior_flags(config)))
    out.append(("promote", [py, "analysis/form_default.py", "--promote"] + only))
    for script, flags, filters in LIVE:
        out.append((" ".join([script] + flags),
                    [py, f"analysis/{script}.py"] + flags
                    + (only if filters else [])))
    out += [("build_site", [sh, "build_site.sh"]),
            ("site_size", [py, "analysis/site_size.py"]),
            ("doc_numbers", [py, "doc_numbers.py", "--write"]),
            ("tests", [py, "tests/run.py"])]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-only", action="store_true",
                        help="skip the priors an answer-record update cannot move")
    parser.add_argument("--list", action="store_true",
                        help="print the steps and run nothing")
    parser.add_argument("--from", dest="start", metavar="STEP",
                        help="start at this step, by its listed name")
    args = parser.parse_args()

    todo = steps(args.repeat_only)
    if args.start is not None:
        names = [name for name, _ in todo]
        if args.start not in names:
            parser.error(f"no step named {args.start!r}; --list shows them")
        todo = todo[names.index(args.start):]
    if args.list:
        for name, argv in todo:
            print(f"{name:<40} {' '.join(argv[1:])}")
        return 0

    # build_site.sh runs $PYTHON, which a shell profile cannot reorder
    env = dict(os.environ, PYTHON=sys.executable)
    began = time.perf_counter()
    for n, (name, argv) in enumerate(todo, start=1):
        print(f"\n### [{n}/{len(todo)}] {name}", flush=True)
        started = time.perf_counter()
        code = subprocess.run(argv, cwd=HERE, env=env).returncode
        took = time.perf_counter() - started
        print(f"### {name}: exit {code} after {took:.0f}s", flush=True)
        if code != 0:
            print(f"### stopped; resume with --from {name!r}", flush=True)
            return code
    print(f"\n### refresh done in {(time.perf_counter() - began) / 60:.0f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

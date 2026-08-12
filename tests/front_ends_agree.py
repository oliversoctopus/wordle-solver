"""The page and the terminal have to ask the solver the same question.

Two front ends drive one solver by different routes: `wordle_solver.py play`
builds a `PriorConfig` from command line flags, `web.py` builds one from a JSON
body, and each has its own parser with its own defaults.  Nothing makes them
agree, and a divergence is invisible from either side -- the page would simply
recommend under a prior nobody selected, with a panel describing it correctly.

So the second implementation is the oracle.  Everything here compares the two
paths rather than checking either against a number written down by hand.

    python tests/front_ends_agree.py            the settings, in a second
    python tests/front_ends_agree.py --full     also the recommendations

``--full`` builds real solvers, so it loads the 167 MB table and compiles the
kernels.  Everything else needs neither.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web
import wordle_solver

failures = 0


def check(what, got, want):
    global failures
    ok = got == want
    if not ok:
        failures += 1
    print(f"{'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        print(f"        page     {got!r}")
        print(f"        terminal {want!r}")


def page_args(argv=()):
    return web.build_parser().parse_args(list(argv))


def terminal_args(argv=()):
    return wordle_solver.build_parser().parse_args(["play", *argv])


# --------------------------------------------------------------- the settings

# --exact-max is a sentinel on the terminal, resolved per subcommand by
# solver_kwargs, so the raw defaults differ by design; check_solver_options
# compares what it resolves to.  mode, form and top are process-level flags
# there and per-request fields here, which is why they are not shared at all.
SENTINELS = {"exact_max"}


def check_defaults():
    """Every flag both parsers offer has to start in the same place."""
    page = vars(page_args())
    terminal = vars(terminal_args())
    shared = sorted(set(page) & set(terminal) - SENTINELS)
    print(f"comparing {len(shared)} flags both front ends take: {', '.join(shared)}")
    for name in shared:
        check(f"--{name.replace('_', '-')} starts the same on both",
              page[name], terminal[name])


def check_prior():
    """The page has two switches; the terminal has two flags.  Every one of the
    four combinations has to name the same prior."""
    for repeat in (True, False):
        for curation in (True, False):
            body = {"repeat": repeat, "curation": curation}
            flags = ([] if repeat else ["--no-repeat-prior"]) + \
                    ([] if curation else ["--no-curation-prior"])
            page = web.parse_config(body)
            terminal = wordle_solver.config_from_args(terminal_args(flags))
            check(f"repeat={repeat}, curation={curation} is one prior",
                  page.slug, terminal.slug)
            check(f"    and the same prior in every field",
                  (page.repeat, page.curation, page.repeat_prob,
                   page.noncurated_prob, page.repeat_shape, page.curation_shape),
                  (terminal.repeat, terminal.curation, terminal.repeat_prob,
                   terminal.noncurated_prob, terminal.repeat_shape,
                   terminal.curation_shape))


def check_solver_options():
    """And the solver each front end builds has to be built the same way."""
    page = web.solver_options(page_args())
    terminal = wordle_solver.solver_kwargs(terminal_args())
    terminal.pop("hard")   # the page carries hard mode per request, not per process
    check("the solver options are the same set", sorted(page), sorted(terminal))
    for name in sorted(set(page) & set(terminal)):
        check(f"    {name} is built the same", page[name], terminal[name])


def check_request_defaults():
    """A request with nothing in it means what the terminal means with no flags,
    since that is what the page sends before anything is touched."""
    settings = web.parse_settings({})
    terminal = terminal_args()
    check("an empty request means the default mode", settings.mode, terminal.mode)
    check("an empty request means the default form", settings.form, terminal.form)
    check("an empty request is not hard mode", settings.hard, terminal.hard)
    check("an empty request means the default prior",
          settings.config.slug,
          wordle_solver.config_from_args(terminal).slug)


# -------------------------------------------------- the recommendation itself

# an answer and the words played at it, never a hand-written pattern: made-up
# feedback contradicts itself and is refused, and then nothing is compared
BOARDS = [
    ("crimp", []),
    ("crimp", ["salet"]),
    ("crimp", ["salet", "round"]),
    ("jaunt", ["crane", "moist"]),
]


def check_recommendations():
    """The expensive half: both front ends, same board, same word out.

    The settings checks above compare how the question is put; this compares the
    answer, which is the only thing that catches a divergence further down --
    a different candidate filter, a different guesses_left, a solver cached
    under a key that does not describe it.
    """
    from wordle_data import load_data, score_guess

    args = terminal_args()
    data = load_data(mask_last=args.mask_last)
    config = wordle_solver.config_from_args(args)
    terminal = wordle_solver.WordleSolver(
        data=data, config=config, mode=args.mode, form=args.form,
        **wordle_solver.solver_kwargs(args))

    pool = web.SolverPool(mask_last=page_args().mask_last,
                          options=web.solver_options(page_args()))
    settings = web.parse_settings({})

    for answer, guesses in BOARDS:
        rows = [(data.word2ind[word], score_guess(word, answer)) for word in guesses]

        # the page's route: whatever /api/board would have put in the panel
        state = web.build_state(pool, "test", settings, rows)
        page_word = state["recommendation"]["word"]

        # the terminal's route: what `wordle_solver.py play` prints
        candidates = terminal.all_candidates()
        for guess, pattern in rows:
            candidates = terminal.filter(guess, pattern, candidates)
        guess, _ = terminal.best_guess(candidates, wordle_solver.MAX_GUESSES - len(rows))
        terminal_word = data.all_words[guess]

        played = " ".join(guesses) or "an empty board"
        check(f"after {played}, one recommendation", page_word, terminal_word)
        check(f"    and the same candidates left after {played}",
              state["candidates_left"], len(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true",
                        help="also compare recommendations, which loads the table")
    args = parser.parse_args()

    check_defaults()
    check_prior()
    check_solver_options()
    check_request_defaults()
    if args.full:
        print("\nloading the table and compiling the kernels...")
        check_recommendations()
    else:
        print("\n(--full also compares the recommendations themselves)")

    print(f"\n{failures} failed" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

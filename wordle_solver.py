"""The terminal front end: fitting, measuring and inspecting the solver.

    python wordle_solver.py train      fit + cache f for every prior combination
    python wordle_solver.py benchmark  self-play, entropy vs. expected guesses
    python wordle_solver.py compare    policy iteration once per form of f
    python wordle_solver.py info       show the cached fits
    python wordle_solver.py play       solve a puzzle here; the page's oracle

The prior toggles, --mode and --form go after the subcommand.
"""

import argparse
import sys

import estimator
import train as training
from exact import DEFAULT_MAX_BRANCH, DEFAULT_ROOT_BRANCH
from priors import (ALL_CONFIGS, CURATION_SHAPES, REPEAT_SHAPES, PriorConfig,
                    depends_on_answer_record)
from solver import (DEFAULT_EXACT_MAX, DEFAULT_LOOKAHEAD_BRANCH,
                    DEFAULT_LOSS_PENALTY, ENTROPY, EXPECTED, MODES,
                    WordleSolver, solver_options)
from wordle_data import MAX_GUESSES, WIN_PATTERN, feedback_to_num, load_data


def config_from_args(args):
    return PriorConfig(
        repeat=not args.no_repeat_prior,
        curation=not args.no_curation_prior,
        repeat_prob=args.repeat_prob,
        noncurated_prob=args.noncurated_prob,
        repeat_shape=args.repeat_shape,
        curation_shape=args.curation_shape,
    )


def score_label(mode, score):
    if mode == ENTROPY:
        return f"{score:.4f} bits of expected information"
    return f"{score:.4f} expected guesses to finish"


def cmd_play(args):
    data = load_data(mask_last=args.mask_last)
    config = config_from_args(args)
    solver = WordleSolver(data=data, config=config, mode=args.mode, form=args.form,
                          hard=args.hard, **solver_options(args))

    print(f"prior: {config.describe()}")
    print(f"mode:  {args.mode}" + (", hard" if args.hard else ""))
    if args.mode == EXPECTED:
        print(estimator.describe(solver.params, solver.form))
    print()

    candidates = solver.all_candidates()
    for turn in range(1, MAX_GUESSES + 1):
        guesses_left = MAX_GUESSES - turn + 1
        guess, score = solver.best_guess(candidates, guesses_left)
        word = data.all_words[guess]
        print(f"--- guess {turn} --- {len(candidates)} candidate(s) left "
              f"(effective {solver.effective_size(candidates):.1f})")
        print(f"recommended: {word}   [{score_label(args.mode, score)}]")
        if len(candidates) == 1:
            print("this is the only possible answer.")
        elif guesses_left == 1:
            print("last guess: restricted to words that could be the answer "
                  f"(this one has a {solver.win_probability(candidates, guess) * 100:.1f}% "
                  "chance).")
        elif args.top > 1:
            for alt, alt_score in solver.top_guesses(candidates, k=args.top,
                                                     guesses_left=guesses_left)[1:]:
                print(f"    alt: {alt}   [{score_label(args.mode, alt_score)}]")

        played = prompt_word(data, default=word)
        if played is None:
            return 1
        pattern = prompt_feedback()
        if pattern is None:
            return 1
        if pattern == WIN_PATTERN:
            print(f"\nsolved in {turn} guesses.")
            return 0

        candidates = solver.filter(played, pattern, candidates)
        if len(candidates) == 0:
            print("\nno words match that feedback -- a tile color or the word list is off.")
            return 1

    print(f"\nout of guesses. remaining candidates: "
          f"{', '.join(data.all_words[i] for i in candidates[:10])}")
    return 1


def prompt_word(data, default):
    while True:
        try:
            raw = input(f"word you guessed [{default}]: ").strip().lower()
        except EOFError:
            return None
        if not raw:
            raw = default
        if raw in data.word2ind:
            return data.word2ind[raw]
        print(f"  '{raw}' is not in the word list; try again.")


def prompt_feedback():
    while True:
        try:
            raw = input("feedback (G green, Y yellow, N gray), e.g. NYGNN: ").strip()
        except EOFError:
            return None
        try:
            return feedback_to_num(raw)
        except ValueError:
            print("  need exactly 5 characters, each G, Y or N.")


def cmd_train(args):
    data = load_data(mask_last=args.mask_last)
    if args.this_config_only:
        configs = [config_from_args(args)]
    else:
        configs = [
            PriorConfig(repeat=c.repeat, curation=c.curation,
                        repeat_prob=args.repeat_prob,
                        noncurated_prob=args.noncurated_prob,
                        repeat_shape=c.repeat_shape,
                        curation_shape=c.curation_shape)
            for c in ALL_CONFIGS
        ]
        if args.repeat_only:
            configs = [c for c in configs if depends_on_answer_record(c)]
    training.train_all(
        configs=configs,
        data=data,
        form=args.form,
        n_games=args.games,
        max_iters=args.max_iters,
        tol=args.tol,
        seed=args.seed,
        use_rust=args.rust,
        solver_options=solver_options(args),
    )
    return 0


def cmd_benchmark(args):
    data = load_data(mask_last=args.mask_last)
    config = config_from_args(args)
    modes = MODES if args.mode == "both" else (args.mode,)
    for mode in modes:
        for deadline in ((True, False) if args.no_deadline else (True,)):
            print(f"\n=== {mode}, last guess restricted: {deadline} ===")
            solver = WordleSolver(data=data, config=config, mode=mode, form=args.form,
                                  hard=args.hard, **solver_options(args))
            training.evaluate(solver, n_games=args.games, seed=args.seed, deadline=deadline)
    return 0


def cmd_compare(args):
    data = load_data(mask_last=args.mask_last)
    config = config_from_args(args)
    forms = sorted(estimator.FORMS) if "all" in args.forms else args.forms
    training.compare_forms(
        config,
        forms=forms,
        data=data,
        n_games=args.games,
        holdout_games=args.holdout_games,
        seed=args.seed,
        max_iters=args.max_iters,
        solver_options=solver_options(args),
    )
    return 0


def cmd_info(args):
    for config in ALL_CONFIGS:
        record = estimator.load_record(config)
        print(f"\n=== {config.slug}: {config.describe()} ===")
        if record is None:
            print("  no cached fit (run: python wordle_solver.py train)")
            continue
        form = estimator.get_form(record.get("form"))
        print(f"  fitted {record.get('fitted_at')} over {record.get('n_games')} games, "
              f"{record.get('iterations')} rounds, converged={record.get('converged')}")
        print(f"  opening word: {record.get('opening_word')} "
              f"(E = {record.get('opening_expected_guesses', float('nan')):.4f})")
        print(f"  mean guesses: {record.get('best_train_mean_guesses', float('nan')):.4f} "
              f"(entropy baseline "
              f"{record.get('entropy_baseline_mean_guesses', float('nan')):.4f})")
        holdout = record.get("holdout")
        if holdout:
            print(f"  holdout: solved {holdout['solve_rate'] * 100:.1f}% of "
                  f"{holdout['games']} games, mean "
                  f"{holdout.get('mean_guesses_penalized', float('nan')):.4f}")
        print("  " + estimator.describe(record["params"], form).replace("\n", "\n  "))

    if args.forms:
        print("\n=== available forms ===")
        for name in sorted(estimator.FORMS):
            marker = " (default)" if name == estimator.DEFAULT_FORM else ""
            print(f"  {name:<12} {estimator.FORMS[name].description}{marker}")
    return 0


COMMANDS = ("play", "train", "benchmark", "compare", "info")


def build_parser():
    # shared by every subcommand, so the toggles go after the command name
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-repeat-prior", action="store_true",
                        help="do not down-weight words that have already been answers")
    common.add_argument("--no-curation-prior", action="store_true",
                        help="do not down-weight words off the curated answer list")
    common.add_argument("--repeat-prob", type=float, default=PriorConfig.repeat_prob,
                        help="probability the editor reuses a past answer")
    common.add_argument("--noncurated-prob", type=float, default=PriorConfig.noncurated_prob,
                        help="probability an unused answer is off the curated list")
    common.add_argument("--repeat-shape", choices=REPEAT_SHAPES,
                        default=PriorConfig.repeat_shape,
                        help="how the repeat mass is spread over past answers "
                             "(default %(default)s); 'binned' favors the start "
                             "of the run, where the reuse actually comes from")
    common.add_argument("--curation-shape", choices=CURATION_SHAPES,
                        default=PriorConfig.curation_shape,
                        help="how the off-list mass is spread (default "
                             "%(default)s); 'frequency' by how answer-like the "
                             "word is, 'wordlebot' by WordleBot's own weights")
    common.add_argument("--mask-last", type=int, default=0,
                        help="ignore the last N entries of solutions.json "
                             "(default %(default)s)")
    common.add_argument("--no-exact", action="store_true",
                        help="never use the exact endgame search, only the estimator")
    common.add_argument("--exact-max", type=int, default=None,
                        help=f"largest candidate set the exact search takes on when "
                             f"three or more turns remain (default {DEFAULT_EXACT_MAX}, "
                             f"or 0 while training); with two or fewer turns it is "
                             f"always used")
    common.add_argument("--exact-branch", type=int, default=DEFAULT_MAX_BRANCH,
                        help="how many splits the exact search tries per position, "
                             "best first (default %(default)s); 0 tries every one, "
                             "which is provably optimal but can take tens of seconds "
                             "on a hard position")
    common.add_argument("--exact-root-branch", type=int, default=DEFAULT_ROOT_BRANCH,
                        help="how many it tries at the position being ranked, which "
                             "wants more than the pick does (default %(default)s)")
    common.add_argument("--lookahead", type=int, default=0,
                        help="plies of branch-capped lookahead above the exact "
                             "search (default %(default)s, off).  1 measures "
                             "-0.0034 +- 0.0041 guesses, which is nothing, for "
                             "~10x the wall clock; 2 is not affordable")
    common.add_argument("--lookahead-branch", type=int,
                        default=DEFAULT_LOOKAHEAD_BRANCH,
                        help="guesses expanded per position when --lookahead is on "
                             "(default %(default)s; a cap of 25 measured no "
                             "better).  It is a fan-out at each node, as "
                             "--exact-branch is, so a second ply multiplies the "
                             "work by it again")
    common.add_argument("--loss-penalty", type=float, default=DEFAULT_LOSS_PENALTY,
                        help="cost of running out of turns, on top of the six spent, so "
                             "1.0 scores a lost game as 7 and is risk neutral "
                             "(default %(default)s); the default removes ~40%% of "
                             "losses for no measurable change in mean score")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    form_choices = sorted(estimator.FORMS)

    play = sub.add_parser("play", parents=[common], help="interactive solving")
    play.add_argument("--mode", choices=MODES, default=EXPECTED)
    play.add_argument("--form", choices=form_choices, default=None,
                      help="form of f to use (default: whichever the cache holds)")
    play.add_argument("--top", type=int, default=1, help="also show N-1 alternatives")
    play.add_argument("--hard", action="store_true",
                      help="only guess words that could still be the answer")
    play.set_defaults(func=cmd_play)

    tr = sub.add_parser("train", parents=[common],
                        help="fit and cache the estimator by policy iteration")
    tr.add_argument("--form", choices=form_choices, default=estimator.DEFAULT_FORM)
    tr.add_argument("--games", type=int, default=training.DEFAULT_GAMES)
    tr.add_argument("--max-iters", type=int, default=training.DEFAULT_MAX_ITERS)
    tr.add_argument("--tol", type=float, default=training.DEFAULT_TOL)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--rust", action=argparse.BooleanOptionalAction, default=True,
                    help="fit with the native trainer where it can reproduce "
                         "the settings (default); --no-rust fits here instead, "
                         "which is a different valid fit rather than a better "
                         "one (README *Fitting in Rust*)")
    tr.add_argument("--this-config-only", action="store_true",
                    help="fit only the prior combination given by the prior flags, "
                         "instead of all five")
    tr.add_argument("--repeat-only", action="store_true",
                    help="skip the repeat-off priors, whose fits an extended "
                         "answer record cannot have moved")
    # f is only ever queried where the exact search does not reach, so training
    # fits only that free part by default; README *What it is worth* has why
    tr.set_defaults(func=cmd_train, default_exact_max=0)

    bench = sub.add_parser("benchmark", parents=[common],
                           help="self-play under the 6 guess limit")
    bench.add_argument("--mode", choices=(*MODES, "both"), default="both")
    bench.add_argument("--form", choices=form_choices, default=None)
    bench.add_argument("--games", type=int, default=200)
    bench.add_argument("--seed", type=int, default=12345)
    bench.add_argument("--no-deadline", action="store_true",
                       help="also report play without the last-guess restriction")
    bench.add_argument("--hard", action="store_true",
                       help="only guess words that could still be the answer")
    bench.set_defaults(func=cmd_benchmark)

    comp = sub.add_parser("compare", parents=[common],
                          help="policy iteration once per form of f, then tabulate")
    comp.add_argument("--forms", nargs="+", choices=form_choices + ["all"],
                      default=[estimator.DEFAULT_FORM],
                      help="forms to compare (default: %(default)s; 'all' for "
                           "every one)")
    comp.add_argument("--games", type=int, default=training.DEFAULT_GAMES)
    # a sanity check only: the ranking that picks a winner is form_default's
    # exact sweep, not this sample
    comp.add_argument("--holdout-games", type=int, default=200)
    comp.add_argument("--max-iters", type=int, default=training.DEFAULT_MAX_ITERS)
    comp.add_argument("--seed", type=int, default=0)
    # same default as train; README *What it is worth* has the measurement
    comp.set_defaults(func=cmd_compare, default_exact_max=0)

    info = sub.add_parser("info", parents=[common], help="show cached estimator fits")
    info.add_argument("--forms", action="store_true", help="also list the available forms")
    info.set_defaults(func=cmd_info)

    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg in COMMANDS for arg in argv) and not {"-h", "--help"} & set(argv):
        argv.insert(0, "play")  # bare invocation, or plain flags, means play
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except estimator.NotFitted as exc:
        # the message carries the command that fixes it; a traceback would bury it
        print(f"\n{exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

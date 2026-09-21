"""Is each cached f still a fit of the answer record we have?

The estimator cache is keyed on the `PriorConfig` alone, so `solutions.py
--write` leaves every fit describing a shorter record.  This compares each
cache's `fitted_at` stamp, or its `ranked_at` stamp where `form_default.py
--promote` kept it as the best on a longer record, with the newest answer the
solver uses.  A repeat-off prior is exempt, since `priors.build_weights` reads
the record only under `config.repeat`.  Refit what it names with
`python wordle_solver.py train --this-config-only`.
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import estimator
import web
from priors import depends_on_answer_record
from wordle_data import DATA_DIR

failures = 0


def report(ok, what, detail=""):
    global failures
    if not ok:
        failures += 1
    print(f"{'ok  ' if ok else 'FAIL'}  {what}")
    if detail:
        print(f"        {detail}")


def answer_record(mask_last):
    """The newest answer the solver is allowed to see, and how many there are."""
    with open(os.path.join(DATA_DIR, "solutions.json"), encoding="utf-8") as fh:
        solutions = json.load(fh)
    used = solutions[:-mask_last] if mask_last else solutions
    newest = used[-1]
    return len(used), newest["solution"], date_of(newest["print_date"])


def date_of(text):
    return datetime.date.fromisoformat(text[:10])


def page_configs():
    """Every prior the page's two switches can name, which is every prior a fit
    has to exist for before expected-guesses play works at all."""
    return [(f"repeat={repeat}, curation={curation}",
             web.parse_config({"repeat": repeat, "curation": curation}))
            for repeat in (True, False) for curation in (True, False)]


def main():
    mask_last = web.build_parser().parse_args([]).mask_last
    count, newest_word, newest_date = answer_record(mask_last)
    print(f"the answer record holds {count} answers the solver may use, "
          f"the newest {newest_word} on {newest_date}")
    if mask_last:
        print(f"(the last {mask_last} are masked, so today's answer is not in it)")
    print()

    for label, config in page_configs():
        path = estimator.cache_path(config)
        if not os.path.exists(path):
            report(False, f"{label} has a fit",
                   f"nothing at {os.path.relpath(path)}; expected-guesses play "
                   f"raises NotFitted for this prior")
            continue

        with open(path, encoding="utf-8") as fh:
            fit = json.load(fh)
        fitted = max(date_of(fit["fitted_at"]),
                     date_of(fit.get("ranked_at", fit["fitted_at"])))
        if not depends_on_answer_record(config):
            report(True, f"{label} does not read the answer record",
                   f"{config.slug}: fitted {fitted} in the '{fit['form']}' form; "
                   f"build_weights consults the used words only under the repeat "
                   f"prior, so a longer record cannot stale this fit")
            continue
        stale = fitted < newest_date
        report(not stale, f"{label} is fitted against the record we have",
               f"{config.slug}: fitted {fitted} in the '{fit['form']}' form, "
               f"{(newest_date - fitted).days} days behind the newest answer; "
               f"refit before trusting a replay"
               if stale else
               f"{config.slug}: fitted {fitted} in the '{fit['form']}' form")

    print(f"\n{failures} stale or missing" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

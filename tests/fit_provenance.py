"""Is each cached f still a fit of the answer record we have?

The estimator cache is keyed on the `PriorConfig` alone, so `solutions.py
--write` leaves every fit describing a shorter record.  This compares each
cache's `fitted_at` stamp, or its `ranked_at` stamp where `form_default.py
--promote` kept it as the best on a longer record, with the newest answer the
solver uses.  A repeat-off prior is exempt, since `priors.build_weights` reads
the record only under `config.repeat`, and every fit must be in
`estimator.DEFAULT_FORM`.  Refit what it names with `wordle_solver.py train`.
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import estimator
from priors import PriorConfig, depends_on_answer_record
from wordle_data import DATA_DIR

failures = 0


def report(ok, what, detail=""):
    global failures
    if not ok:
        failures += 1
    print(f"{'ok  ' if ok else 'FAIL'}  {what}")
    if detail:
        print(f"        {detail}")


def answer_record():
    """The newest answer in the record, and how many there are."""
    with open(os.path.join(DATA_DIR, "solutions.json"), encoding="utf-8") as fh:
        solutions = json.load(fh)
    newest = solutions[-1]
    return len(solutions), newest["solution"], date_of(newest["print_date"])


def date_of(text):
    return datetime.date.fromisoformat(text[:10])


def page_configs():
    """Every prior the page's two switches can name, which is every prior a fit
    has to exist for before expected-guesses play works at all."""
    return [(f"repeat={repeat}, curation={curation}",
             PriorConfig(repeat=repeat, curation=curation))
            for repeat in (True, False) for curation in (True, False)]


def main():
    count, newest_word, newest_date = answer_record()
    print(f"the answer record holds {count} answers, "
          f"the newest {newest_word} on {newest_date}")
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
        report(fit["form"] == estimator.DEFAULT_FORM,
               f"{label} is fitted in the {estimator.DEFAULT_FORM} form",
               "" if fit["form"] == estimator.DEFAULT_FORM else
               f"{config.slug} plays '{fit['form']}'; README *Which form is "
               f"the default*")
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

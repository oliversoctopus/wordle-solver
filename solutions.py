"""Bring `solutions.json`, the answer history, up to date from the NYT's endpoint.

`solutions.json` is its own cache: the days already in it are the days not
fetched, so a rerun costs one request per day since the last one.  There is no
second copy of the scrape -- the file the solver reads is the record.

`id` is the NYT's own field and is the index into their internal answer list,
not the puzzle number -- it is what puts the off-list answers in a block above
2309.
"""

import argparse
import concurrent.futures
import datetime
import json
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wordle_data import SOLUTIONS_PATH, write_json

NYT_URL = "https://www.nytimes.com/svc/wordle/v2/{date}.json"
LAUNCH_DATE = datetime.date(2021, 6, 19)
MAX_WORKERS = 8

# the pre-NYT era has no editor and no day count; those keys are left out
# rather than stored null
FIELDS = ("id", "solution", "print_date", "days_since_launch", "editor")


def make_session():
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "Mozilla/5.0 (wordle-solver data build)"
    return session


def load_existing():
    if not os.path.exists(SOLUTIONS_PATH):
        return []
    with open(SOLUTIONS_PATH, "r") as file:
        return json.load(file)


def as_row(body):
    """One NYT reply as `solutions.json` stores it."""
    row = {"id": body.get("id"), "solution": body["solution"].lower(),
           "print_date": body.get("print_date"),
           "days_since_launch": body.get("days_since_launch"),
           "editor": body.get("editor")}
    return {key: row[key] for key in FIELDS if row[key] is not None}


def fetch_day(session, date):
    reply = session.get(NYT_URL.format(date=date.isoformat()), timeout=10)
    if reply.status_code != 200:
        return None
    body = reply.json()
    return as_row(body) if body.get("solution") else None


def collect(offline=False, until=None):
    """Everything already known, plus whatever days are missing before `until`."""
    rows = {r["print_date"]: r for r in load_existing()}
    until = until or datetime.date.today()
    wanted = [LAUNCH_DATE + datetime.timedelta(days=i)
              for i in range((until - LAUNCH_DATE).days + 1)]
    missing = [d for d in wanted if d.isoformat() not in rows]

    if missing and not offline:
        session = make_session()
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for row in pool.map(lambda d: fetch_day(session, d), missing):
                if row:
                    rows[row["print_date"]] = row

    ordered = [rows[key] for key in sorted(rows)]
    return ordered, len(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="fetch nothing; just report what is already there")
    parser.add_argument("--write", action="store_true",
                        help="write solutions.json, which the solver reads")
    args = parser.parse_args()

    before = load_existing()
    rows, missing = collect(offline=args.offline)
    print(f"{len(rows)} days, {rows[0]['print_date']} to {rows[-1]['print_date']}")
    print(f"  already had  {len(before)}")
    print(f"  days missing {missing}" + (" (not fetched)" if args.offline else ""))
    print(f"  added        {len(rows) - len(before)}")

    known = {r["print_date"]: r for r in before}
    changed = [r for r in rows
               if r["print_date"] in known and known[r["print_date"]] != r]
    if changed:
        print(f"  CHANGED      {len(changed)} existing days, e.g. {changed[0]}")

    if args.write:
        # indented, one field per line: a day added should read as a few added
        # lines, not as a rewrite of the whole file
        write_json(SOLUTIONS_PATH, rows)
        print(f"wrote {SOLUTIONS_PATH}")
    elif len(rows) == len(before) and not changed:
        print("not written; it is already up to date")
    else:
        print("not written; pass --write to update solutions.json")


if __name__ == "__main__":
    main()

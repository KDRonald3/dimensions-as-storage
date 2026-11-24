"""
v3/choose_rates.py - choose the ONE learning rate of the whole third experiment from the practice
runs (v3/PLAN.md, Part B.2), and write it to v3/settings.json.

    .venv/bin/python -m v3.choose_rates            # reads runs_v3/practice, writes v3/settings.json

The rule, applied by the program and not by judgement:
  1. For each rate, the average of facts known at the end over ALL 80 practice settings (every
     columns, kinds, body width and number of creatures).
  2. The rate with the highest average wins. Rates within half a point of the best (half a
     point below it included) are tied.
  3. A tie goes to the rate with the higher average of questions about one creature, and if
     that is equal too, to the lower rate.

(An earlier draft chose a rate for each combination of columns and kinds. The owner decided on
one rate for all, so every counted run trains with the same rate.)
"""

import argparse
import json
import time
from pathlib import Path

from v3.grid import (COLUMNS, KINDS, PRACTICE_BODIES, PRACTICE_CREATURES, PRACTICE_RATES,
                     SETTINGS_FILE, practice)

ROOT = Path(__file__).resolve().parent.parent
TIE = 0.5  # rates within this many points of the best facts known are tied
ROUNDING = 1e-9  # so that "exactly half a point" is not lost to the last digit of an average
SETTINGS_PER_RATE = len(COLUMNS) * len(KINDS) * len(PRACTICE_BODIES) * len(PRACTICE_CREATURES)


def read_practice(practice_dir):
    """The practice summaries: {name: summary}, for every practice run that has one."""
    found = {}
    for config in practice():
        path = Path(practice_dir) / config["name"] / "summary.json"
        if path.exists():
            found[config["name"]] = json.loads(path.read_text())
    return found


def average(values):
    return sum(values) / len(values)


def choose(summaries):
    """The chosen rate, with the reasons. Returns {"rate", "why", "averages"}, where averages
    has, per rate, the average facts known and questions about one creature over all 80
    practice settings. Raises ValueError when a practice run is missing."""
    by_rate = {rate: [] for rate in PRACTICE_RATES}
    for config in practice():
        by_rate[config["lr"]].append(config["name"])
    missing = [name for names in by_rate.values() for name in names if name not in summaries]
    if missing:
        raise ValueError(f"{len(missing)} practice runs have no summary yet, for example "
                         f"{', '.join(missing[:3])}")
    table = {}
    for rate, names in by_rate.items():
        assert len(names) == SETTINGS_PER_RATE  # 80 settings for every rate
        finals = [summaries[name]["final"] for name in names]
        table[rate] = {"facts": average([f["facts"] for f in finals]),
                       "questions_one": average([f["questions_one"] for f in finals])}
    best = max(item["facts"] for item in table.values())
    tied = [rate for rate in PRACTICE_RATES if table[rate]["facts"] >= best - TIE - ROUNDING]
    # Highest questions about one creature first; then the lower rate.
    winner = sorted(tied, key=lambda rate: (-table[rate]["questions_one"], rate))[0]
    if len(tied) == 1:
        why = "highest facts known"
    else:
        others = [rate for rate in tied if rate != winner]
        same_questions = [rate for rate in others
                          if table[rate]["questions_one"] == table[winner]["questions_one"]]
        why = (f"tied on facts known with {', '.join(map(str, others))}; "
               + ("equal on questions about one creature, so the lower rate"
                  if same_questions else "higher questions about one creature"))
    return {"rate": winner, "why": why, "averages": {str(rate): table[rate]
                                                     for rate in PRACTICE_RATES}}


def write_settings(chosen, settings_path):
    settings = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "rule": ("one rate for all counted runs: the highest facts known at the end of practice, "
                 "averaged over all 80 practice settings; rates within 0.5 point of the best are "
                 "tied; a tie goes to the higher questions about one creature, then to the "
                 "lower rate"),
        "rate": chosen["rate"],
        "details": chosen,
    }
    Path(settings_path).write_text(json.dumps(settings, indent=1) + "\n")
    return settings


def print_table(chosen):
    print("rate     facts known   questions about one creature   (averages over 80 settings)")
    for rate in PRACTICE_RATES:
        item = chosen["averages"][str(rate)]
        mark = "  <- chosen" if rate == chosen["rate"] else ""
        print(f"{rate:<7}  {item['facts']:11.2f}   {item['questions_one']:28.2f}{mark}")
    print(f"chosen: {chosen['rate']} ({chosen['why']})")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Choose the learning rate from practice.")
    parser.add_argument("--practice-dir", default=str(ROOT / "runs_v3" / "practice"))
    parser.add_argument("--settings", default=str(SETTINGS_FILE))
    args = parser.parse_args(argv)
    chosen = choose(read_practice(args.practice_dir))
    print_table(chosen)
    write_settings(chosen, args.settings)
    print(f"written to {args.settings}")
    return chosen


if __name__ == "__main__":
    main()

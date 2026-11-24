"""
compare.py - the experiment in one command: the same model on a small and on a large vocabulary.

What this file is for
---------------------
The hypothesis under test says that a model which is big enough for a few words should do worse
when it has to hold many. This file trains the SAME model twice, first on a small vocabulary and
then on a large one, each until its own run ends (train.py explains the three endings: solved,
plateau, budget). Then it prints the two results side by side, and reads them twice, because
"the same amount of training" can mean two different things:

  (1) at the same number of sets   Both models have read the same amount of text (with zipf
                                   usage: to within a few sentences). But the large vocabulary
                                   spreads that text over many more facts, so each fact was
                                   practiced far less often.
  (2) at the end of each run       Each model trained until it was solved, flat or out of budget.

The file prints numbers and points out what to look at. It gives NO verdict on the hypothesis:
one pair of runs cannot settle it. The rounds in sweep.py and the rule in report.py are for that.

How to run it
-------------
    python compare.py                                     # 16 animals against 1,024, width 32
    python compare.py --small 16 --large 256 --width 8
    python compare.py --usage zipf --max-minutes 5
    python compare.py --help                              # every flag of train.py works here too

    python compare.py --again                             # train both runs again

The two runs are saved like any other run: with the default settings in runs/a16_w32_equal_s0
and runs/a1024_w32_equal_s0, so ask.py and probe.py can open them. Those two are also runs of
the main round of sweep.py. So a run that is already saved is never replaced without being asked:
  - if it was trained with the settings asked for now, it is used as it is and not trained again;
  - if it was trained with other settings (say another --learning-rate, which is not part of the
    name), this file stops and says what differs. Give --name or another --out-dir to keep both
    runs, or --again to train the run again and replace the saved one.

Terms used in this file
-----------------------
vocabulary      Here: the number of animal words in use. Everything else about the language
                stays the same.
parameter       One of the numbers inside the model that training adjusts. The "table"
                parameters are the word table (one row of numbers per word); the "body"
                parameters are everything else.
set             A fixed number of training sentences. A model is graded only at the end of a set.
times seen      How many training sentences there have been about each fact, on average.
loss            The model's average surprise at the next word, in bits per word. Lower is better.
best possible   The loss of a predictor that knows every fact. It is not zero, because some
                words cannot be predicted by anyone (which animal comes next?).
gap             loss - best possible. 0 means the model predicts as well as anyone could.
                The gap measures prediction, not only memory: a model can also predict a fact
                that was stated earlier in the same row by copying it. With few animals a row
                repeats facts often, with many animals hardly ever, so between two vocabulary
                sizes the gap can be compared only at 0.
facts known     0 = knows no animal, 100 = knows every fact of every animal in use. It is asked
                without any earlier text, so nothing can be copied. It is the ONLY number in
                these tables that can be compared between vocabulary sizes.
"""

import argparse
import json
import textwrap
from dataclasses import replace
from pathlib import Path

from sweep import settings_that_differ  # tells whether a saved run is the run that is asked for
from train import add_run_flags, build, config_from_args, default_name, run

SMALL = 16  # animal words in the small vocabulary
LARGE = 1024  # animal words in the large vocabulary
AGAIN = False  # True: train both runs again, also if they are already saved

LINE = "=" * 96

# What to read in place of the loss. The two percentages were measured once for this language
# with equal usage, by giving a reader that knows NO fact the right to copy from earlier in the
# row: it closed that share of the distance between knowing nothing and the best possible loss.
GAP_NOTE = (
    "Only facts known can be compared between the two rows. The gap can be compared only at 0, "
    "which means perfect at every vocabulary size. Above 0 it cannot: with few animals a row "
    "often repeats a fact, so part of the gap can be closed by copying from earlier in the row "
    "without knowing anything (measured with equal usage: about 11% of the gap at 16 animals, "
    "against 0.1% at 1,024)."
)


def one_of_two(config, animals):
    """The settings of one of the two runs: `config` with this many animals in use.

    With --name, the two runs are saved as NAME_a16 and NAME_a1024 (for 16 and 1,024).
    """
    name = None if config.name is None else f"{config.name}_a{animals}"
    return replace(config, animals=animals, name=name)


def saved_run(config):
    """The saved summary of a run that is already there and was trained with the settings asked
    for now. None if there is no such run yet, so that it has to be trained.

    If a run of this name is there but was trained with OTHER settings, this raises a ValueError
    that says what differs. Such a run is not replaced without being asked: it may be part of a
    round of sweep.py that took hours to train.
    """
    run_dir = Path(config.out_dir) / (config.name or default_name(config))
    try:
        with open(run_dir / "summary.json") as file:
            summary = json.load(file)
    except (FileNotFoundError, ValueError):  # no such run, or cut off while it was being saved
        return None
    different = settings_that_differ(summary, config, asked_by="asked now")
    if different:
        raise ValueError(
            f"{run_dir} already holds a run, and it was trained with other settings: "
            f"{', '.join(different)}. It is not replaced without being asked. Give --name or "
            f"another --out-dir to keep both runs, or --again to train this run again and "
            f"replace the saved one.")
    return summary


def compare(config, small=SMALL, large=LARGE, again=AGAIN):
    """Train the same model on both vocabularies, the small one first. Returns both summaries.

    `config` holds every setting the two runs share. Only the number of animals in use differs,
    so both models have the same word table, the same number of parameters, the same starting
    values and the same training recipe. A run that is already saved with these settings is not
    trained again, unless `again` is True (see saved_run).
    """
    summaries = []
    for animals, which in [(small, "small"), (large, "large")]:
        of_run = one_of_two(config, animals)
        print(LINE)
        print(f"THE {which.upper()} VOCABULARY: {animals:,} animal words")
        print(LINE)
        saved = None if again else saved_run(of_run)
        if saved is None:
            summaries.append(run(of_run))
        else:
            print_wrapped(f"Already trained with these settings: the run saved in "
                          f"{Path(of_run.out_dir) / saved['name']} is used as it is. (--again "
                          f"trains it again.) It ended by {saved['stop_explained']}", "", "")
            summaries.append(saved)
        print()
    return summaries[0], summaries[1]


# ------------------------------------------------------------------------------------------------
# Reading the two runs
# ------------------------------------------------------------------------------------------------


def grading_at(summary, sets):
    """A run as it stood after `sets` sets: that entry of its learning curve, or None."""
    for point in summary["curve"]:
        if point["sets"] == sets:
            return point
    return None


def scheduled_sets(summary):
    """The set counts of a run's scheduled gradings.

    A run that was ended off the schedule (by its budget, or by fixed_sets) is graded once more
    where it stopped. train.py marks that extra grading "scheduled": False. It is left out
    here: the two runs are only ever lined up at gradings that every run gets.
    """
    return {point["sets"] for point in summary["curve"] if point["scheduled"]}


def last_shared_sets(small, large):
    """The largest set count at which BOTH runs had a scheduled grading.

    Every run is graded at the same set counts (1, 2, 3, 4, 6, 8, 11, 16, ...). Normally the
    small vocabulary ends first, so this is the set count at which the small-vocabulary run
    ended, and the large-vocabulary run was graded there too, on its way to its own ending.
    """
    return max(scheduled_sets(small) & scheduled_sets(large))


def table_line(summary, point):
    """One line of a table: a run as it stood at the grading `point` (an entry of its curve)."""
    if point["sets"] < summary["sets"]:
        ended = "still training"
    elif summary["stop_reason"] == "budget":
        ended = "budget: NOT FINISHED"
    else:
        ended = summary["stop_reason"]
    return (f"  {summary['language']['animals']:6,d} {summary['parameters']['body']:10,d} "
            f"{summary['parameters']['table']:10,d} {point['sets']:7,d} "
            f"{point['sightings']:12,.1f} {point['loss']:7.3f} {point['floor']:8.3f} "
            f"{point['gap']:7.3f} {point['facts_known']:6.1f}  {ended}")


def print_table(lines):
    print(f"  {'animal':>6s} {'body':>10s} {'table':>10s} {'':>7s} {'times each':>12s} "
          f"{'':>7s} {'best':>8s} {'':>7s} {'facts':>6s}")
    print(f"  {'words':>6s} {'parameters':>10s} {'parameters':>10s} {'sets':>7s} "
          f"{'fact seen':>12s} {'loss':>7s} {'possible':>8s} {'gap':>7s} {'known':>6s}  "
          f"how the run ended")
    for line in lines:
        print(line)


def sets_in_words(sets):
    """A number of sets with its noun: "1 set", "6 sets"."""
    return f"{sets:,} set" + ("" if sets == 1 else "s")


def print_wrapped(text, first="    ", rest="    "):
    # break_on_hyphens=False keeps a flag such as --max-minutes in one piece on one line.
    print(textwrap.fill(text, width=96, initial_indent=first, subsequent_indent=rest,
                        break_on_hyphens=False))


def print_comparison(small, large):
    """The two runs side by side, read twice, and what to notice. No verdict."""
    n_small, n_large = small["language"]["animals"], large["language"]["animals"]
    same_sets = last_shared_sets(small, large)

    print(LINE)
    print(f"THE SAME MODEL ON {n_small:,} AND ON {n_large:,} ANIMAL WORDS")
    print(LINE)
    print("  loss, best possible and gap are in bits per word; gap = loss - best possible.")
    print("  facts known: 0 = knows no animal, 100 = knows every fact of every animal in use.")
    print()
    print(f"(1) AT THE SAME NUMBER OF SETS: both models after {sets_in_words(same_sets)}, so "
          f"after the same amount of text.")
    if same_sets == small["sets"]:
        print_wrapped(f"This is where the {n_small:,}-animal run ended.")
    else:
        print_wrapped(f"The {n_small:,}-animal run ended after {sets_in_words(small['sets'])}, "
                      f"but that is not a set count at which both runs had a scheduled grading. "
                      f"{same_sets:,} is the last set count at which both had one.")
    print_table([table_line(small, grading_at(small, same_sets)),
                 table_line(large, grading_at(large, same_sets))])
    print()
    print("(2) AT THE END OF EACH RUN: each model trained until its own run ended.")
    if small["sets"] == large["sets"]:
        print_wrapped("Both runs ended after the same number of sets, so this table repeats "
                      "the first one.")
    print_table([table_line(small, small["curve"][-1]), table_line(large, large["curve"][-1])])
    print()
    print_what_to_notice(small, large, same_sets)


def print_what_to_notice(small, large, same_sets):
    """A few plain sentences for a first reading of the tables. Facts about the numbers only."""
    notes = [note_on_parameters(small, large), note_on_loss(small, large),
             note_on_practice(small, large, same_sets)]
    if small["language"]["usage"] == "zipf":
        notes.append('With zipf usage "times each fact seen" is an average: the facts of '
                     'common animals were seen far more often, those of rare animals far less.')
    unfinished = [f"{summary['language']['animals']:,}-animal run"
                  for summary in [small, large] if summary["stop_reason"] == "budget"]
    if unfinished:
        notes.append(
            f"NOT FINISHED: the {' and the '.join(unfinished)} ended by budget. Table (2) "
            f"shows where such a run stood when the time or the sets ran out, not what the "
            f"model can do. --max-minutes gives a run more time.")
    print("What to notice")
    for note in notes:
        print_wrapped(note, first="  * ", rest="    ")


def note_on_parameters(small, large):
    """The point of the design: the two models are the same size."""
    if small["parameters"] != large["parameters"]:
        return ("The two models do NOT have the same number of parameters (see the columns). "
                "Something other than the vocabulary differs between the runs.")
    counts = small["parameters"]
    return (f"Both models have exactly the same number of parameters: {counts['body']:,} in the "
            f"body + {counts['table']:,} in the word table = {counts['total']:,}. Both word "
            f"tables have a row for each of the {small['language']['table_animals']:,} animals "
            f"of the table, whether the model ever read about the animal or not. Only the "
            f"training text differs.")


def note_on_loss(small, large):
    """Why the loss column cannot be compared between vocabularies, and what to read instead."""
    n_small, n_large = small["language"]["animals"], large["language"]["animals"]
    floor_small, floor_large = small["language"]["floor"], large["language"]["floor"]
    note = (f"Do not compare the loss column between the two rows. The best possible loss is "
            f"{floor_small:.3f} bits per word for {n_small:,} animals and {floor_large:.3f} for "
            f"{n_large:,}: nobody can predict which animal the next sentence is about, and with "
            f"more animals there is more to guess. ")
    if floor_large > floor_small:
        note += "So even a perfect model has a higher loss on the large vocabulary. "
    return note + GAP_NOTE


def note_on_practice(small, large, same_sets):
    """Why table (1) alone is not a fair comparison: equal text is unequal practice per fact."""
    n_small, n_large = small["language"]["animals"], large["language"]["animals"]
    seen_small = grading_at(small, same_sets)["sightings"]
    seen_large = grading_at(large, same_sets)["sightings"]
    note = (f"The same number of sets is not the same amount of practice. After "
            f"{sets_in_words(same_sets)} each fact of the {n_small:,}-animal language had been "
            f"seen {seen_small:,.0f} times and each fact of the {n_large:,}-animal language "
            f"{seen_large:,.0f} times")
    if seen_small > seen_large > 0:
        note += f", which is {seen_small / seen_large:,.0f} times less practice per fact"
    return note + (". A lower score in table (1) can come from that alone. Table (2) lets each "
                   "run practice until it ended by itself.")


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------


def main(argv=None):
    # conflict_handler="resolve" allows the last two flags below to replace train.py's --animals
    # and --name, which mean something else when there are two runs.
    parser = argparse.ArgumentParser(
        description="Train the same model on a small and on a large vocabulary and compare.",
        conflict_handler="resolve",
    )
    parser.add_argument("--small", type=int, default=SMALL,
                        help="animal words in the small vocabulary")
    parser.add_argument("--large", type=int, default=LARGE,
                        help="animal words in the large vocabulary")
    add_run_flags(parser)  # every setting of train.py: --width, --usage, --max-minutes, ...
    parser.add_argument("--animals", type=int, default=None,
                        help="not used here: give --small and --large instead")
    parser.add_argument("--name", default=None,
                        help="start of both run names (NAME_a16 and NAME_a1024); by default the "
                             "names are built from the settings")
    parser.add_argument("--again", action="store_true", default=AGAIN,
                        help="train both runs again and replace the saved ones; without it a "
                             "run that is already saved with these settings is used as it is")
    args = parser.parse_args(argv)

    if args.animals is not None:
        parser.error("compare.py trains two vocabularies: give --small and --large instead of "
                     "--animals.")
    if not 1 <= args.small < args.large:
        parser.error(f"--small ({args.small}) must be at least 1 and less than --large "
                     f"({args.large}).")
    if args.large > args.table_animals:
        parser.error(f"--large ({args.large}) is more animals than the word table holds "
                     f"(--table-animals {args.table_animals}).")

    config = config_from_args(args)
    try:
        # Build both languages and models once before any training. Settings that do not suit
        # one of the two vocabularies (a set size that is no multiple of the larger number of
        # animals, say) are then refused at once, not after the first run has been trained.
        # A saved run with other settings is refused here too, before anything is trained.
        for animals in (args.small, args.large):
            build(replace(config, animals=animals))
            if not args.again:
                saved_run(one_of_two(config, animals))
        small, large = compare(config, args.small, args.large, args.again)
    except ValueError as problem:  # settings that make no sense: print the plain message
        parser.error(str(problem))
    except RuntimeError as problem:  # training broke down (see train.check_not_broken)
        raise SystemExit(str(problem))
    print_comparison(small, large)


if __name__ == "__main__":
    main()

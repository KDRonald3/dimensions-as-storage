"""
sweep.py - the named rounds of the experiment: many runs, one after another.

What this file is for
---------------------
One run (train.py) is one model on one vocabulary. The experiment needs many: every vocabulary
size against every model size, and each of those several times with a different seed, so that
one lucky or unlucky run is not mistaken for a result. A "round" is such a list of runs with a
name. The rounds are written out as plain data at the top of this file (ROUNDS). This file
trains the runs of one round, by default one at a time, and leaves each in runs/<name>/, where
report.py finds them.

How to run it
-------------
    python sweep.py                            # list the rounds
    python sweep.py quick --dry-run            # list the runs of a round and stop
    python sweep.py quick                      # 4 runs, a first look
    python sweep.py round1-equal               # the main round, 48 runs
    python sweep.py round1-equal --max-minutes 5
    python sweep.py round1-equal --jobs 4      # four runs at a time, 2 threads each
    python sweep.py round1-equal --widths 8 16 32   # part of a round now ...
    python sweep.py round1-equal --widths 4         # ... and the rest of it later

A run that is already finished (runs/<name>/summary.json exists) is skipped. So a round that
was interrupted is continued by typing the same command again, and a run that two rounds share
(the four runs of "quick" are also part of "round1-equal") is trained only once.

--widths, --seeds and --animals pick out part of a round. Give every part of a round the same
--max-minutes: a note is printed if runs of the round were saved with a larger budget.

Two things to keep in mind
--------------------------
1. A finished run is recognized by its NAME, and the name holds only the animals, the width, the
   usage, the seasons, the table size and the seed. If a run of that name was saved with other
   settings (another learning rate, or a shorter budget that ended it early), it is still
   skipped, but a note says which settings differ. Delete its directory to train it again.
2. The budget of a run (--max-minutes) is time on the clock. With --jobs above 1 every run gets
   fewer threads and is slower, so more runs may end by budget, which means NOT FINISHED.

Terms used in this file
-----------------------
round    A named list of runs.
seed     The number that decides everything random in a run: the model's starting values, the
         invented animals' facts and the shuffling of the training sets. Runs with different
         seeds are repetitions of the same experiment.
job      One run being trained. --jobs 4 trains four runs at the same time.
thread   One of the processor's lanes of work. A single run uses several at once.
"""

import argparse
import contextlib
import json
import math
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from pathlib import Path

from language import automatic_set_sentences
from train import RunConfig, default_name, run

JOBS = 1  # runs trained at the same time
THREADS_ONE_JOB = 8  # threads per run when one run trains at a time (fastest on this machine)
THREADS_MANY_JOBS = 2  # threads per run when several train at the same time
MAX_MINUTES = RunConfig.max_minutes  # the budget of every run, in minutes
OUT_DIR = "runs"  # every run is saved in OUT_DIR/<name of the run>/

# THE ROUNDS. In each round the four lists are the dials that vary: every combination of their
# values is one run. "settings" holds what is the same for every run of the round; any setting
# of RunConfig (see train.py) may be written there.
ROUNDS = {
    "quick": {
        "about": "a first look: smallest and largest vocabulary, smallest and largest model",
        "animals": [16, 1024],
        "widths": [4, 32],
        "table_animals": [1024],
        "seeds": [0],
        "settings": {"usage": "equal", "seasons": 1},
    },
    "round1-equal": {
        "about": "the main round: every vocabulary size on every model size, equal usage",
        "animals": [16, 64, 256, 1024],
        "widths": [4, 8, 16, 32],
        "table_animals": [1024],
        "seeds": [0, 1, 2],
        "settings": {"usage": "equal", "seasons": 1},
    },
    "round1-zipf": {
        "about": "the main round again with zipf usage: a few common animals, many rare ones",
        "animals": [16, 64, 256, 1024],
        "widths": [4, 8, 16, 32],
        "table_animals": [1024],
        "seeds": [0, 1, 2],
        "settings": {"usage": "zipf", "seasons": 1},
    },
    # The set size is written out here, for a reason. The automatic size follows the size of the
    # table (960 sentences for a table of 16 animals, 61,440 for one of 1,024), so without it the
    # two halves of this round would read sets of different sizes: one training step per set
    # against 53, and gradings after very different amounts of text. With 61,440 for both, the
    # two halves read exactly the same text in the same batches and are graded at the same
    # points, and the word table is the only difference. (61,440 is what the 1,024-animal table
    # gets automatically, so that half is the same run as in round1-equal.)
    "table-check": {
        "about": "does an oversized word table change anything? (table of 16 or of 1,024)",
        "animals": [16],
        "widths": [4, 32],
        "table_animals": [16, 1024],
        "seeds": [0, 1, 2],
        "settings": {"usage": "equal", "seasons": 1, "set_sentences": 61440},
    },
    "scaleup": {
        "about": "four seasons: 32 bits of facts per animal instead of 8, widths up to 64",
        "animals": [16, 64, 256, 1024],
        "widths": [8, 16, 32, 64],
        "table_animals": [1024],
        "seeds": [0, 1, 2],
        "settings": {"usage": "equal", "seasons": 4},
    },
}


def planned_runs(round_name, max_minutes=MAX_MINUTES, threads=THREADS_ONE_JOB, out_dir=OUT_DIR):
    """Every run of a round, as a list of RunConfig, in the order in which they are trained."""
    plan = ROUNDS[round_name]
    configs = []
    # Small vocabularies come first, so the first results of a round arrive early. Within a
    # vocabulary all widths of one seed come before the next seed.
    for animals in sorted(plan["animals"]):
        for table_animals in plan["table_animals"]:
            for seed in plan["seeds"]:
                for width in plan["widths"]:
                    settings = dict(plan["settings"])
                    # name=None: every run gets a name of its own, built from its settings.
                    settings.update(animals=animals, table_animals=table_animals, seed=seed,
                                    width=width, max_minutes=max_minutes, threads=threads,
                                    out_dir=out_dir, name=None)
                    configs.append(RunConfig(**settings))
    return configs


# ------------------------------------------------------------------------------------------------
# Runs that are already finished
# ------------------------------------------------------------------------------------------------


def load_finished(config):
    """The saved summary of a run that is already finished, or None if it has to be trained.

    train.py writes summary.json last, so if the file is there the run is complete.
    """
    path = Path(config.out_dir) / default_name(config) / "summary.json"
    if not path.exists():
        return None
    try:
        with open(path) as file:
            return json.load(file)
    except json.JSONDecodeError:  # cut off while it was being written: train the run again
        return None


def set_size(settings):
    """The sentences per set that a run's settings come to: 0 stands for the automatic size."""
    return settings["set_sentences"] or automatic_set_sentences(settings["table_animals"],
                                                                settings["seasons"])


def settings_that_differ(summary, config, asked_by="this round"):
    """Settings a saved run was trained with that are not the ones this round asks for.

    Returns a list of plain phrases, empty if the saved run is the run that was asked for.
    asked_by names who is asking, for the phrases (compare.py uses this function too).
    """
    ignore = ["name", "out_dir", "threads"]  # where it was saved and how fast it ran
    asked = asdict(config)
    different = []
    # Every setting this round asks for is looked up in the saved run, not the other way round:
    # a setting that did not exist yet when the run was saved must not go unnoticed.
    for key in asked:
        if key in ignore:
            continue
        if key not in summary["config"]:
            # The run was saved by older code, before this setting existed.
            different.append(f"{key} not recorded (saved by older code; {asked_by}: "
                             f"{asked[key]})")
            continue
        saved = summary["config"][key]
        if asked[key] == saved:
            continue
        if key == "set_sentences" and set_size(summary["config"]) == set_size(asked):
            # The same set size, once written out and once left to the automatic rule: the
            # 1,024-table runs of "table-check" are the 16-animal runs of "round1-equal".
            continue
        if key in ("max_minutes", "max_sets"):
            # A budget matters only if it ended the saved run and this round would allow more.
            if summary["stop_reason"] != "budget" or saved >= asked[key]:
                continue
        different.append(f"{key} {saved} ({asked_by}: {asked[key]})")
    return different


# ------------------------------------------------------------------------------------------------
# Printing
# ------------------------------------------------------------------------------------------------


def result_line(summary):
    """A finished run in one line."""
    ended = summary["stop_reason"]
    if ended == "budget":
        ended = "budget: NOT FINISHED"
    return (f"{summary['name']:24s} facts known {summary['final']['facts_known']:5.1f}   "
            f"gap {summary['final']['gap']:6.3f}   {summary['sets']:6,d} sets   "
            f"{summary['minutes']:5.1f} min   {ended}")


def count(number, thing):
    """A number with its noun: "1 run", "3 runs"."""
    return f"{number:,} {thing}" + ("" if number == 1 else "s")


def minutes_text(minutes):
    """A length of time in plain words: "0.4 min", "12 min" or "3.5 hours"."""
    if minutes < 10:
        return f"{minutes:.1f} min"
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} hours"


def time_left_text(runs_left, runs_trained, minutes_trained, jobs, max_minutes,
                   larger_to_come=False):
    """One sentence on how long the rest of the round may take.

    runs_trained runs took minutes_trained minutes between them. The sentence gives two numbers:
    an estimate that assumes the runs still to come take as long as those, and a ceiling from
    the budget. Small vocabularies are trained first and larger ones may well take longer, so
    early in a round the estimate is likely to be too low. larger_to_come says that runs on a
    larger vocabulary than any trained so far are still to come; the sentence then says so.
    """
    if runs_left == 0:
        return "no runs left"
    # `jobs` runs train side by side, so the runs left take as long as this many runs in a row.
    in_a_row = math.ceil(runs_left / jobs)
    at_most = in_a_row * max_minutes
    text = f"{count(runs_left, 'run')} left: "
    if runs_trained > 0:
        estimate = min(in_a_row * minutes_trained / runs_trained, at_most)
        text += f"about {minutes_text(estimate)} if they take as long as the runs so far"
        if larger_to_come:
            text += (" (likely too low: the runs so far were on the small vocabularies, which "
                     "come first)")
        text += ", "
    return text + (f"at most about {minutes_text(at_most)} "
                   f"(the budget is {max_minutes:g} min for each run)")


def print_plan(round_name, configs, jobs):
    """The heading of a round and one line for every run. Returns (finished, to_train).

    finished: the saved summaries of the runs that need no training, by run name.
    to_train: the configs of the other runs, in order.
    """
    first = configs[0]
    print(f"ROUND {round_name}: {ROUNDS[round_name]['about']}")
    print(f"  {count(len(configs), 'run')}, {jobs} at a time, {count(first.threads, 'thread')} "
          f"each, a budget of {first.max_minutes:g} min each, saved in {first.out_dir}/")
    if jobs * first.threads > os.cpu_count():
        print(f"  NOTE: {jobs} jobs x {first.threads} threads is more than the {os.cpu_count()} "
              f"threads of this machine. The runs will slow each other down.")
    finished, to_train = {}, []
    for number, config in enumerate(configs, start=1):
        name = default_name(config)
        summary = load_finished(config)
        if summary is None:
            to_train.append(config)
            print(f"  {number:3d}  to train      {name:24s} {config.animals:5,d} animals "
                  f"(table {config.table_animals:,}), width {config.width}, seed {config.seed}")
            continue
        finished[name] = summary
        print(f"  {number:3d}  already done  {result_line(summary)}")
        different = settings_that_differ(summary, config)
        if different:
            print(f"       NOTE: this saved run was trained with {', '.join(different)}. It "
                  f"is skipped all the same. Delete {Path(config.out_dir) / name} to train "
                  f"it again.")
    return finished, to_train


def longer_budget_note(round_name, max_minutes, out_dir):
    """A note, or None: were runs of this round saved with a larger budget than this command
    gives? Every run of the round is looked at, also those that --widths, --seeds or --animals
    leave out this time: a round trained in parts should get the same budget in every part.
    """
    largest, example = max_minutes, None
    for config in planned_runs(round_name, max_minutes, out_dir=out_dir):
        summary = load_finished(config)
        if summary is not None and summary.get("config", {}).get("max_minutes", 0) > largest:
            largest, example = summary["config"]["max_minutes"], default_name(config)
    if example is None:
        return None
    return (f"NOTE: runs of this round were saved with a budget of {largest:g} min each (for "
            f"example {example}); this command gives {max_minutes:g}. With less time a run can "
            f"end as NOT FINISHED where the others had time to finish. To give every part of the "
            f"round the same budget, add --max-minutes {largest:g}.")


def print_rounds():
    print("The rounds. Run one with: .venv/bin/python sweep.py NAME   (add --dry-run to only "
          "list its runs)")
    for name, plan in ROUNDS.items():
        print(f"  {name:14s} {len(planned_runs(name)):3d} runs   {plan['about']}")


# ------------------------------------------------------------------------------------------------
# Training the runs
# ------------------------------------------------------------------------------------------------


def train_one(config):
    """Train one run. Returns (config, summary, None), or (config, None, what went wrong) if
    the run failed.

    Whatever goes wrong in one run, the round goes on with the next: training that broke down
    (train.py raises a RuntimeError when the loss is no longer a number), a full disk while
    saving, settings that make no sense. The failed run is reported and nothing is saved for it.
    """
    try:
        return config, run(config), None
    except Exception as problem:  # any error; Ctrl-C is not an Exception and still gets through
        return config, None, f"{type(problem).__name__}: {problem}"


def train_quietly(config):
    """train_one without the run's own table on the screen.

    Used when several runs train at the same time: their tables would be mixed up line by line.
    Nothing is lost, because every line of the table is also saved in summary.json (the curve).
    """
    with open(os.devnull, "w") as nowhere, contextlib.redirect_stdout(nowhere):
        return train_one(config)


def leave_ctrl_c_to_the_main_program():
    """Runs once at the start of every worker process (see train_all).

    Ctrl-C reaches every process started from the terminal. The workers ignore it; the main
    program notices it and stops the workers itself, so an interrupted round ends at once.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def stop_like_ctrl_c(signal_number, frame):
    """What the main program does when it is told to end by a plain `kill` (the signal SIGTERM):
    the same as for Ctrl-C. Without this the main program would die at once and leave the worker
    processes of --jobs training on their own, each until its run is over."""
    raise KeyboardInterrupt


def stop_workers():
    """End every worker process at once, whatever it is doing. A run that was being trained is
    lost; nothing is saved for it, so the next start of the round trains it again."""
    for worker in multiprocessing.active_children():
        worker.terminate()


WORKER_DIED = ("the worker process that trained this run ended without a word, also at a "
               "second try (the system ends a process when memory runs out, for example). "
               "Nothing was saved for this run")


def train_side_by_side(to_train, jobs, work):
    """Train the runs in one pool of `jobs` worker processes, each of which trains one run after
    another. Hands back every result as it comes in. At the very end it hands back one more
    thing: the list of the runs that were NOT trained because a worker process died.

    "spawn" starts every worker as a fresh Python, which is the safe way for a program that uses
    PyTorch's threads.
    """
    fresh = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(min(jobs, len(to_train)), mp_context=fresh,
                               initializer=leave_ctrl_c_to_the_main_program)
    given_up = []
    try:
        # The runs are handed out in order; each result comes back as soon as its run has ended.
        waiting = {pool.submit(work, config): config for config in to_train}
        for ended in as_completed(waiting):
            try:
                yield ended.result()
            except BrokenProcessPool:
                # A worker process died. A pool cannot go on after that, and it says so for
                # every run it had not finished. (Without this a round would wait for ever for
                # the run that the dead worker was training.)
                given_up.append(waiting[ended])
    finally:
        # Reached when all runs are done, and also on Ctrl-C or `kill`: no worker is left behind.
        pool.shutdown(wait=False, cancel_futures=True)
        stop_workers()
    yield [config for config in to_train if config in given_up]  # in the order of the round


def train_all(to_train, jobs, work=train_quietly):
    """Train the runs, `jobs` at a time. Hands back (config, summary, problem) whenever a run
    ends. `yield` is what lets the caller print a line after each run while the rest goes on.

    work is the function a worker process calls for one run (the tests put in another one).
    """
    if jobs == 1:
        for config in to_train:
            print()
            yield train_one(config)
        return
    died_at_once = 0  # how many pools in a row lost a worker before a single run had ended
    while to_train:
        came_back = 0
        for result in train_side_by_side(to_train, jobs, work):
            if isinstance(result, list):  # the last thing handed back: the runs given up
                given_up = result
            else:
                came_back += 1
                yield result
        if not given_up:
            return
        # The runs a broken pool gave up are handed to a fresh pool. That is tried again and
        # again as long as runs get through. Two pools in a row that die before any run has
        # ended are taken to mean that a third would die the same way.
        died_at_once = died_at_once + 1 if came_back == 0 else 0
        if died_at_once == 2:
            for config in given_up:
                yield config, None, WORKER_DIED
            return
        print(f"NOTE: a worker process died. {count(len(given_up), 'run')} that it left "
              f"untrained will be started again in fresh worker processes.")
        to_train = given_up


def only(configs, widths=None, seeds=None, animals=None):
    """The runs of a round picked out by --widths, --seeds and --animals; None means "all"."""
    return [config for config in configs
            if (widths is None or config.width in widths)
            and (seeds is None or config.seed in seeds)
            and (animals is None or config.animals in animals)]


def nothing_chosen(round_name):
    """What to say when --widths, --seeds and --animals together leave no run of a round."""
    plan = ROUNDS[round_name]

    def values(numbers):
        return " ".join(str(number) for number in numbers)

    return (f"no run of the round {round_name} has these widths, seeds and animals. The round "
            f"has --widths {values(plan['widths'])}, --seeds {values(plan['seeds'])} and "
            f"--animals {values(sorted(plan['animals']))}.")


def run_round(round_name, max_minutes=MAX_MINUTES, threads=THREADS_ONE_JOB, out_dir=OUT_DIR,
              jobs=JOBS, dry_run=False, widths=None, seeds=None, animals=None):
    """Train every run of a round that is not finished yet.

    widths, seeds and animals, if given, limit the round to those model widths, seeds and
    vocabulary sizes, so that part of a round can be trained now and the rest later with the
    same command. Returns the summaries of all finished runs among those, old and new, by name.
    """
    configs = only(planned_runs(round_name, max_minutes, threads, out_dir), widths, seeds,
                   animals)
    if not configs:
        raise ValueError(nothing_chosen(round_name))
    finished, to_train = print_plan(round_name, configs, jobs)
    note = longer_budget_note(round_name, max_minutes, out_dir)
    if note:
        print("  " + note)
    print(f"  {len(finished)} already done. "
          + time_left_text(len(to_train), 0, 0.0, jobs, max_minutes) + ".")
    if dry_run:
        print("Dry run: nothing was trained.")
        return finished
    if jobs > 1 and to_train:
        print(f"  With {jobs} runs at a time their own tables are not shown, only one line "
              f"for each run when it ends.")

    trained, minutes_trained, failed = 0, 0.0, {}
    to_come = [default_name(config) for config in to_train]  # the runs that have not ended yet
    animals_of = {default_name(config): config.animals for config in to_train}
    largest_so_far = 0  # the largest vocabulary among the runs trained in this round so far
    for config, summary, problem in train_all(to_train, jobs):
        name = default_name(config)
        to_come.remove(name)
        if summary is None:
            failed[name] = problem
            print(f"FAILED  {name}: {problem}")
        else:
            trained += 1
            minutes_trained += summary["minutes"]
            largest_so_far = max(largest_so_far, config.animals)
            finished[name] = summary
            print(f"RESULT  {result_line(summary)}")
        larger_to_come = any(animals_of[other] > largest_so_far for other in to_come)
        print("        " + time_left_text(len(to_come), trained, minutes_trained, jobs,
                                          max_minutes, larger_to_come))

    print()
    print(f"ROUND {round_name}: {len(finished)} of {len(configs)} runs are finished.")
    for config in configs:
        if default_name(config) in finished:
            print(f"  {result_line(finished[default_name(config)])}")
    for name, problem in failed.items():
        print(f"  {name:24s} FAILED, nothing was saved: {problem}")
    print("Next: .venv/bin/python report.py"
          + ("" if out_dir == OUT_DIR else f" --runs-dir {out_dir}"))
    return finished


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the runs of one named round, one after another."
    )
    parser.add_argument("round", nargs="?", choices=list(ROUNDS),
                        help="the round to run; leave it out to see the list of rounds")
    parser.add_argument("--jobs", type=int, default=JOBS, help="runs trained at the same time")
    parser.add_argument("--threads", type=int, default=None,
                        help=f"processor threads for each run; by default {THREADS_ONE_JOB} "
                             f"when --jobs is 1, otherwise {THREADS_MANY_JOBS}")
    parser.add_argument("--max-minutes", type=float, default=MAX_MINUTES,
                        help="the budget of every run, in minutes")
    parser.add_argument("--widths", type=int, nargs="+", default=None,
                        help="train only these model widths of the round, for example "
                             "--widths 8 16 32; the others can be trained later")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="train only these seeds of the round, for example --seeds 0")
    parser.add_argument("--animals", type=int, nargs="+", default=None,
                        help="train only these vocabulary sizes of the round, for example "
                             "--animals 16 1024")
    parser.add_argument("--dry-run", action="store_true", help="list the runs and stop")
    parser.add_argument("--out-dir", default=OUT_DIR, help="every run is saved in OUT_DIR/NAME/")
    args = parser.parse_args(argv)

    if args.round is None:
        print_rounds()
        return
    if args.jobs < 1:
        parser.error("--jobs must be at least 1.")
    threads = args.threads
    if threads is None:
        threads = THREADS_ONE_JOB if args.jobs == 1 else THREADS_MANY_JOBS
    if not only(planned_runs(args.round), args.widths, args.seeds, args.animals):
        parser.error(nothing_chosen(args.round))
    # A plain `kill` is handled like Ctrl-C from here on (see stop_like_ctrl_c).
    handler_before = signal.signal(signal.SIGTERM, stop_like_ctrl_c)
    try:
        finished = run_round(args.round, args.max_minutes, threads, args.out_dir, args.jobs,
                             args.dry_run, args.widths, args.seeds, args.animals)
    except KeyboardInterrupt:
        stop_workers()  # with --jobs above 1: no worker may go on training behind our back
        print("\nInterrupted. Type the same command again to go on: finished runs are skipped.")
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGTERM, handler_before)
    # Tell a calling script that something is missing (a failed run), by a non-zero exit code.
    if not args.dry_run and len(finished) < len(only(planned_runs(args.round), args.widths,
                                                       args.seeds, args.animals)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
report.py - what the finished runs say: a table, four charts and the agreed decision rule.

What this file is for
---------------------
Every finished training run leaves a file runs/<name>/summary.json. This file reads all of them
and does three things:

  1. It prints a results table and saves it as results/summary.csv (one row per run), together
     with results/curves.csv (one row per grading of every run). Every dot and every thin line
     of a chart can be read as a number in one of the two files; the thick lines are means of
     those numbers. A short second table gives the means over seeds.
  2. It draws up to four charts into results/. A chart is drawn only if there are runs for it.
  3. It applies THE AGREED DECISION RULE (just below) and says in plain sentences, with the
     numbers, for which model widths the rule is met and why it is not met for the others.
     Before that it checks what the rule takes for granted: that the runs it compares were
     trained the same way, and under the agreed stopping rule. Where that is not so, it says
     which runs differ and gives no verdict.

It trains nothing and changes nothing under runs/.

How to run it
-------------
    python report.py
    python report.py --runs-dir runs --results-dir results

Terms used in this file
-----------------------
run           One model trained on one vocabulary until it ended (train.py).
width         The model-size dial: how many numbers the model keeps per word.
seed          The number that decides a run's random choices. Runs that differ only in the seed
              are repeats of the same experiment; runs with the same seed read the same text.
facts known   The score, 0 to 100: 0 = knows no animal, 100 = knows every fact (score.py).
solved        The run ended because the model knew practically every fact.
plateau       The run ended because more training had stopped helping: a limit was reached.
budget        The run ran out of time. It is NOT FINISHED and says nothing about any limit.
fixed         The run was given a fixed number of sets (--fixed-sets in train.py) and the
              stopping rule was never asked. It can never count as a plateau.
recipe        Everything about how a run was trained apart from the dials of the experiment:
              the learning rate, the batch, the set size, the stopping rule, ...
log scale     An axis on which equal distances mean equal ratios: from 16 to 64 is as far as
              from 64 to 256. It keeps small and large numbers readable in one picture.
"""

import argparse
import csv
import dataclasses
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight into picture files; no window is opened

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from matplotlib.transforms import offset_copy

from language import build_lexicon
from score import chance_accuracy
from train import RunConfig, default_name

# ------------------------------------------------------------------------------------------------
# THE AGREED DECISION RULE
#
# These numbers were agreed before any experiment was run. They are deliberately NOT command-line
# flags: a rule that is adjusted after the results are in proves nothing. If you do change one,
# change it here, say so, and rerun the report on ALL runs.
#
# For one width, at one usage and one number of seasons, the hypothesis "clearly holds" when, in
# at least SEEDS_NEEDED seeds:
#   (a) the width scores HIGH or more on the 16-animal language;
#   (b) on some larger vocabulary its run ended by plateau and scored at least DROP points below
#       its own 16-animal score;
#   (c) on that vocabulary the next width up, with the same seed and so the same training sets,
#       scores at least RESCUE points higher.
# A run that ended by budget is not finished and can never satisfy (b). Nor can a run that was
# trained for a fixed number of sets (stop reason "fixed"): the stopping rule was never asked, so
# nobody has checked that it was flat.
# ------------------------------------------------------------------------------------------------
HIGH = 95  # (a) "can learn the small language": facts known on 16 animals is at least this
DROP = 10  # (b) "does worse on more words": at least this many points below its 16-animal score
RESCUE = 10  # (c) "a bigger model fixes it": the next width up scores at least this much higher
SEEDS_NEEDED = 2  # (a), (b) and (c) must all hold in at least this many seeds
SMALL = 16  # the small vocabulary every width is measured against: the 16 hand-written animals
# "The next width up" in (c). The agreed widths are 4, 8, 16, 32 (and 8 to 64 in the 32-bit
# round), each twice the last, so the next width up of a width is twice that width. If there is no
# run of that width, (c) cannot be tested: a still larger width does not stand in for it.
WIDTH_STEP = 2  # the next width up has this many times the width

# ------------------------------------------------------------------------------------------------
# WHAT THE RULE TAKES FOR GRANTED
#
# The rule compares runs that differ in the vocabulary, the width and the seed, and in nothing
# else. And (b) rests on the word "plateau": what a plateau is, is decided by the stopping rule
# of train.py. Both are checked here, before the rule is applied to a group of runs:
#   - every run of the group must have been trained with the same recipe;
#   - every run must have been trained under the agreed stopping rule, which is the stopping rule
#     with train.py's default settings.
# A group that fails either check gets no verdict. A run's settings are the fields of RunConfig
# in train.py. The four lists below name the settings that are NOT part of the recipe; every
# other setting is, so a setting that is added to train.py later is checked without a change here.
# ------------------------------------------------------------------------------------------------
DIALS = ["animals", "width", "seed"]  # what the experiment varies on purpose
GROUP_KEYS = ["usage", "seasons", "table_animals"]  # each combination is judged on its own
# How long a run was allowed to train. This may differ between runs: what came of it is told by
# the run's ending (budget = not finished, fixed = the stopping rule was never asked).
LENGTH = ["max_minutes", "max_sets", "fixed_sets"]
RUNNING = ["threads", "out_dir", "name"]  # where a run was saved and how fast it ran
# The settings of the stopping rule. They are part of the recipe, and they must also have the
# values train.py gives them by default, or "plateau" would not mean what was agreed.
STOPPING_RULE = ["solved_at", "flat_points", "flat_gap", "flat_gap_bits", "min_sightings",
                 "min_steps", "started_at"]

# ------------------------------------------------------------------------------------------------
# How the charts look
#
# The colors are the reference palette of the "dataviz" chart guidelines, in its documented
# order, which was chosen so that neighboring colors stay distinct for color-blind readers.
# A color belongs to a WIDTH, not to a place in a list: width 8 is orange in every chart.
# Some of these colors are pale on the light background, and orange and yellow are not far
# apart. So color is never the only clue: every line has its name written at its end, and every
# number can be read in the tables.
# ------------------------------------------------------------------------------------------------
WIDTH_COLORS = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a", 32: "#eda100", 64: "#e87ba4",
                128: "#008300"}
OTHER_WIDTH_COLOR = "#898781"  # any width that is not listed above is drawn gray
# Chart 3 colors its dots by the number of seasons (1 season = 8 bits per animal, 4 = 32 bits).
# Each also has its own shape, so that the two can be told apart without seeing color.
SEASON_STYLES = {1: ("#2a78d6", "o"), 4: ("#eb6834", "s"), 2: ("#1baf7a", "^"), 3: ("#898781", "D")}

SURFACE = "#fcfcfb"  # the background of every chart
INK = "#0b0b0b"  # titles
INK_SOFT = "#52514e"  # labels and legends
INK_MUTED = "#898781"  # the numbers along the axes
GRID = "#e1e0d9"  # grid lines
AXIS = "#c3c2b7"  # the zero line and the thin lines that join a label to its line

PX = 0.72  # one pixel of the picture in matplotlib's unit (the point), at 100 dots per inch
LINE = 2 * PX  # lines are 2 pixels thick
HAIR = 1 * PX  # grid lines are 1 pixel thick
DOT = 11 * PX  # a filled dot, with its ring in the background color: 9 pixels of color
HOLLOW_DOT = 8 * PX  # a hollow dot (a run that was not finished), plus its 2-pixel outline
HALF_DOT = 10.5 * PX  # a half-filled dot (a run of a fixed length), plus its 1.5-pixel outline
DOT_STEP = 9  # dots at the same place are set this many pixels apart; fewer where room is short
WIDTH_GAP = 0.6  # extra room between the dots of two widths, as a share of the step between dots
LABEL_GAP = 7.5  # line-end labels are kept at least this far apart, in units of the 0-100 axis
SAVE_DPI = 200  # dots per inch of the saved pictures (twice the drawing size, to stay sharp)

PANELS_PER_ROW = 3
PANEL_WIDTH, WIDE_PANEL, PANEL_HEIGHT = 3.7, 6.2, 3.0  # inches; WIDE_PANEL when there is only one
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_BOTTOM = 0.8, 1.05, 0.7  # inches around the panels
GAP_X, GAP_Y = 1.55, 1.15  # inches between panels

CHART_FILES = {  # chart number -> file name in the results directory
    1: "chart1_more_words.png",
    2: "chart2_learning.png",
    3: "chart3_knowledge_per_parameter.png",
    4: "chart4_common_and_rare.png",
}
CHART_TITLES = {
    1: "More words, same model",
    2: "Learning against practice",
    3: "Knowledge per parameter",
    4: "Common and rare words",
}


# ------------------------------------------------------------------------------------------------
# Reading the runs
# ------------------------------------------------------------------------------------------------


def load_runs(runs_dir):
    """Read every runs_dir/*/summary.json. Returns a list of summary dictionaries (see train.py),
    sorted the way the table shows them."""
    runs = []
    for path in sorted(Path(runs_dir).glob("*/summary.json")):
        try:
            with open(path) as file:
                run = json.load(file)
        except ValueError:  # the file is cut off: the run was interrupted while it was saving
            print(f"Skipped {path}: the file is not complete. Delete its folder and train the "
                  f"run again.")
            continue
        missing = what_is_missing(run)
        if missing:
            print(f"Skipped {path}: it is not a complete summary of a run ({missing}). Delete "
                  f"its folder and train the run again.")
            continue
        # Two runs can carry the same name (a copy of a folder); the folder tells them apart.
        run["folder"] = str(path.parent)
        runs.append(run)

    def table_order(run):
        c = run["config"]
        return (c["usage"], c["seasons"], c["table_animals"], c["width"], c["animals"], c["seed"],
                run["name"], run["folder"])

    return sorted(runs, key=table_order)


def what_is_missing(run):
    """Why the content of a summary.json cannot be used, in a few words; None if it can.

    A file can be complete as a file and still not be the summary of a run: empty braces, a
    summary cut short by hand, a file of some other program. The check is made by reading
    everything the tables and the charts read. One such file must not stop the whole report.
    """
    if not isinstance(run, dict):
        return "the file holds something else"
    try:
        table_row(run)  # every number of the table
        how_ended(run)
        for point in run["curve"]:  # the learning curve, for chart 2 and curves.csv
            point["sets"], point["sightings"], point["facts_known"]
        run["curve"][-1]
    except (KeyError, TypeError, IndexError) as problem:
        return f"missing or not readable: {problem}"
    return None


def facts(run):
    """The score of a run: facts known, 0 to 100."""
    return run["final"]["facts_known"]


def how_ended(run):
    """A run's ending as one of three words. Tables and charts give each a mark of its own.

      "finished"  the stopping rule ended the run: solved or plateau
      "budget"    the run ran out of time: it is NOT FINISHED
      "fixed"     the run had a fixed number of sets: the stopping rule was never asked, so
                  nobody knows whether the run was flat or still improving
    """
    if run["stop_reason"] in ("budget", "fixed"):
        return run["stop_reason"]
    return "finished"


def folder(run):
    """Where a run lies. Runs that did not come from load_runs have only their name."""
    return run.get("folder", run["name"])


def bits_per_parameter(run):
    """The load: bits of facts in the language for every parameter of the model."""
    return run["final"]["bits_total"] / run["parameters"]["total"]


def share_held(run):
    """How much of the language's knowledge the model holds, 0 to 100. A lower bound: see
    bits_known in score.py."""
    if run["final"]["bits_total"] <= 0:
        return 0.0
    return 100 * run["final"]["bits_known"] / run["final"]["bits_total"]


def group_label(usage, seasons):
    return f"{usage} usage, {seasons} season" + ("" if seasons == 1 else "s")


def rank_groups(animals):
    """The animals in use, grouped by their number: 1-16, 17-64, 65-256, 257-1024, ...

    Animal number 1 is the first of the table. Under zipf usage it is the most used and every
    group is used less than the one before, so the number is the animal's RANK. Each group
    reaches four times as far as the last. Returns a list of (label, first, last).
    """
    groups = []
    first, last = 1, SMALL
    while first <= animals:
        groups.append((f"{first}-{min(last, animals)}", first, min(last, animals)))
        first, last = last + 1, last * 4
    return groups


def right_by_rank_group(run):
    """For each rank group, the share (0 to 100) of its facts that the model has right.

    This is plain accuracy, NOT corrected for guessing: a model that knows no animal still gets
    some facts right by always naming the most common value (see guessing_levels).
    """
    right = run["final"]["facts_right_by_animal"]  # for every animal in use: its facts right
    facts_per_animal = run["config"]["seasons"] * 4
    shares = {}
    for label, first, last in rank_groups(len(right)):
        n_facts = (last - first + 1) * facts_per_animal
        shares[label] = 100 * sum(right[first - 1 : last]) / n_facts
    return shares


def guessing_levels(config):
    """For each rank group, the share of facts (0 to 100) a model gets right that knows no animal
    and always names the group's most common value. 25 when every value is equally common.

    It is worked out from the animals' actual facts, which are rebuilt here from the run's
    settings (the facts depend only on the seed and on the base table in language.py).
    """
    lexicon = build_lexicon(config["table_animals"], config["seasons"], config["seed"])
    levels = {}
    for label, first, last in rank_groups(config["animals"]):
        of_group = lexicon[first - 1 : last]
        every_animal_alike = np.full(len(of_group), 1 / len(of_group))
        levels[label] = 100 * chance_accuracy(of_group, every_animal_alike)
    return levels


# ------------------------------------------------------------------------------------------------
# The table
# ------------------------------------------------------------------------------------------------


def table_row(run):
    """The numbers of one run, as one row of results/summary.csv."""
    config, final, parameters = run["config"], run["final"], run["parameters"]
    row = {
        "name": run["name"],
        "usage": config["usage"],
        "seasons": config["seasons"],
        "table_animals": config["table_animals"],
        "width": config["width"],
        "animals": config["animals"],
        "seed": config["seed"],
        "parameters_body": parameters["body"],
        "parameters_table": parameters["table"],
        "parameters_total": parameters["total"],
        "bits_total": final["bits_total"],
        "bits_per_animal": run["language"]["bits_per_animal"],
        "bits_per_parameter": bits_per_parameter(run),
        "sets": run["sets"],
        "steps": run["steps"],
        "minutes": run["minutes"],
        "sightings": run["sightings"],
        "loss": final["loss"],
        "floor": final["floor"],
        "gap": final["gap"],
        "facts_known": final["facts_known"],
        "facts_known_by_use": final["facts_known_by_use"],
        "questions_right": final["questions_right"],
        "unseen_known": final["unseen_known"],
        "words_known": final["words_known"],
        "n_words": final["n_words"],
        "bits_known": final["bits_known"],
        "share_held": share_held(run),
        "stop_reason": run["stop_reason"],
        # yes = solved or plateau; no = ended by budget; fixed = the stopping rule was not asked
        "finished": {"finished": "yes", "budget": "no", "fixed": "fixed"}[how_ended(run)],
    }
    for label, share in right_by_rank_group(run).items():
        row["pct_right_animals_" + label.replace("-", "_")] = share
    return row


def curve_rows(runs):
    """One row per grading of every run, for results/curves.csv: the learning curves as numbers."""
    rows = []
    for run in runs:
        config = run["config"]
        for point in run["curve"]:
            row = {"name": run["name"]}
            for key in ["usage", "seasons", "table_animals", "width", "animals", "seed"]:
                row[key] = config[key]
            row.update(point)
            rows.append(row)
    return rows


def write_csv(path, rows):
    """Write a list of dictionaries as a CSV file. The columns are the keys, in order of first
    appearance; a row without a column gets an empty cell there. Numbers are written with six
    significant digits, which is more than any chart can show."""
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: float(f"{value:.6g}") if isinstance(value, float) else value
                             for key, value in row.items()})


def print_table(runs):
    """The results table: one block per usage, number of seasons and size of the word table."""
    blocks = {}
    for run in runs:
        c = run["config"]
        blocks.setdefault((c["usage"], c["seasons"], c["table_animals"]), []).append(run)
    print(f"RESULTS OF {len(runs)} RUNS")
    for (usage, seasons, table_animals), block in blocks.items():
        by_use = f" {'by use':>6s}" if usage == "zipf" else ""
        print()
        print(f"{group_label(usage, seasons)}, word table of {table_animals:,} animals")
        print(f"  {'width':>5s} {'animals':>7s} {'seed':>4s} {'parameters':>10s} {'sets':>7s} "
              f"{'seen':>10s} {'gap':>6s} {'facts':>6s}{by_use} {'questions':>9s}  "
              f"{'ended':<12s} name")
        for run in block:
            c, final = run["config"], run["final"]
            by_use = f" {final['facts_known_by_use']:6.1f}" if usage == "zipf" else ""
            ended = "NOT FINISHED" if how_ended(run) == "budget" else run["stop_reason"]
            print(f"  {c['width']:5d} {c['animals']:7,d} {c['seed']:4d} "
                  f"{run['parameters']['total']:10,d} {run['sets']:7,d} {run['sightings']:10,.0f} "
                  f"{final['gap']:6.3f} {facts(run):6.1f}{by_use} "
                  f"{final['questions_right']:9.1f}  {ended:<12s} {run['name']}")
    print()
    print(textwrap.fill(
        "parameters = every number in the model (body + word table). seen = times each fact was "
        "seen in training, on average. gap = loss - best possible loss, in bits per word (0 = "
        "perfect). facts, by use and questions: 0 = knows no animal, 100 = knows all. NOT "
        "FINISHED = the run ended by budget; its numbers are not a limit of the model. fixed = "
        "the run was given a fixed number of sets and the stopping rule was not asked. More "
        "columns (loss, bits of knowledge held, ...) are in summary.csv.",
        width=100, initial_indent="  ", subsequent_indent="  "))


def runs_behind_the_means(group):
    """One sentence on how many runs each mean of a group's table is taken over. A mean of three
    seeds and a single run look the same in the table, so the places that differ are named."""
    counts = {}  # (width, animals) -> how many runs
    for width, animals, _ in group:
        counts[(width, animals)] = counts.get((width, animals), 0) + 1
    numbers = list(counts.values())
    usual = max(set(numbers), key=numbers.count)  # the number of runs most cells have
    others = [f"width {width} on {animals:,} animals ({number})"
              for (width, animals), number in sorted(counts.items()) if number != usual]
    sentence = f"Each mean is taken over {usual} run" + ("" if usual == 1 else "s")
    if others:
        sentence += ", except: " + ", ".join(others)
    return sentence + "."


def print_means(groups, cautions=()):
    """The short table: facts known as the mean over seeds, one row per width and one column per
    vocabulary. These are the numbers the lines of chart 1 go through.

    cautions: the keys of the groups whose runs were not all trained the same way (see
    recipe_problems). Their means mix what should not be mixed, and the table says so.
    """
    print("FACTS KNOWN, THE MEAN OVER SEEDS (0 to 100)")
    for key in sorted(groups):
        group = groups[key]
        vocabularies = sorted({animals for _, animals, _ in group})
        print()
        print(group_label(*key))
        print(f"  {'width':>5s}" + "".join(f"{animals:9,d}  " for animals in vocabularies)
              + " animal words")
        for width in sorted({width for width, _, _ in group}):
            cells = ""
            for animals in vocabularies:
                runs = [run for (w, a, _), run in group.items() if w == width and a == animals]
                if not runs:
                    cells += f"{'-':>9s}  "
                    continue
                endings = {how_ended(run) for run in runs}
                mark = ("*" if "budget" in endings else "") + ("#" if "fixed" in endings else "")
                cells += f"{np.mean([facts(run) for run in runs]):9.1f}{mark:<2s}"
            print(f"  {width:5d}{cells}".rstrip())
        print(textwrap.fill(runs_behind_the_means(group), width=100, initial_indent="  ",
                            subsequent_indent="  "))
        if key in cautions:
            print(textwrap.fill(
                "CAREFUL: the runs of this group were not all trained the same way, or not under "
                "the agreed stopping rule. These means mix them; the decision rule below names "
                "the runs.", width=100, initial_indent="  ", subsequent_indent="  "))
    print()
    print("  * = at least one of these runs was NOT FINISHED (ended by budget).")
    print("  # = at least one of these runs had a fixed number of sets: the stopping rule was "
          "not asked.")
    print("  - = no run.")


# ------------------------------------------------------------------------------------------------
# The agreed decision rule
# ------------------------------------------------------------------------------------------------


def at_least(value, threshold):
    """True if value >= threshold, give or take the last digits of computer arithmetic.

    A computer keeps decimal numbers only approximately: 16.4 - 6.4 comes out as
    9.999999999999998, and that must still count as "at least 10".
    """
    return value >= threshold - 1e-9


def listed(items):
    """["a"] -> "a"; ["a", "b"] -> "a and b"; ["a", "b", "c"] -> "a, b and c"."""
    items = [str(item) for item in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def default_name_of(run):
    """The name train.py gives by default to a run with these settings.

    Only the settings this RunConfig knows are used. A summary saved by other code may hold a
    setting that has since been renamed or removed, and that must not stop the report.
    """
    known = {field.name for field in dataclasses.fields(RunConfig)}
    settings = {key: value for key, value in run["config"].items() if key in known}
    return default_name(RunConfig(**settings))


def rank_at_its_place(run):
    """Which of several runs with the same vocabulary, width and seed counts: the lowest rank.

    An ordinary run comes before a run of a fixed length, because only an ordinary run asked the
    stopping rule. After that, a run that lies in the folder train.py gives it by default comes
    before a run in any other folder: that folder is where sweep.py looks for the run. How the
    names fall in the alphabet plays no part. Two runs of the same rank cannot be told apart.
    """
    in_its_own_folder = Path(folder(run)).name == default_name_of(run)
    return (how_ended(run) == "fixed", not in_its_own_folder)


def in_short(run):
    """A run in a few words, for the notes: its folder, its score and its ending."""
    return f"{folder(run)} (facts known {facts(run):.1f}, ended as {run['stop_reason']})"


def one_run_per_place(candidates, place):
    """Of the runs that share a vocabulary, a width and a seed, the one that counts.

    Returns (the run, notes about the runs left out). If nothing says which run counts, the run
    is None: no run is used, because a verdict must not hang on which folder happened to be read
    first. `place` names the vocabulary, width and seed in words.
    """
    ranked = sorted(candidates, key=rank_at_its_place)
    first = ranked[0]
    tied = [run for run in ranked if rank_at_its_place(run) == rank_at_its_place(first)]
    if len(tied) > 1:
        return None, [f"There are {len(tied)} runs for {place} and nothing says which one "
                      f"counts: {listed([in_short(run) for run in tied])}. NONE of them is used: "
                      f"for the rule and the charts this place is empty. Remove or move all but "
                      f"one of these folders."]
    notes = []
    for other in ranked[1:]:
        if how_ended(other) == "fixed" and how_ended(first) != "fixed":
            why = "a run of a fixed length never asked the stopping rule"
        else:
            why = "it lies in the folder train.py gives a run of these settings by default"
        note = (f"{in_short(other)} is left out. {in_short(first)} has the same vocabulary, "
                f"width and seed and is the run that counts, because {why}.")
        apart = abs(facts(other) - facts(first))
        if at_least(apart, min(DROP, RESCUE)):
            note += (f" CAREFUL: the two scores are {apart:.1f} points apart, so it matters "
                     f"which of them counts.")
        notes.append(note)
    return first, notes


def comparison_groups(runs):
    """Sort the runs into the groups inside which vocabularies are compared.

    Returns (groups, notes). groups maps (usage, seasons) to {(width, animals, seed): run}.
    notes are plain sentences about runs that were left out, for two reasons:
      - a run whose word table is not the size most runs of its group have (the table-check
        round, or a single try with another table): its model has another number of parameters,
        so it is not "the same model";
      - a second run with the same vocabulary, width and seed: one place, one run. Which one is
        kept is decided by rank_at_its_place, and the note gives the score and the ending of both.
    The runs left out still appear in the table, in the CSV files and in chart 3.
    """
    table_counts = {}  # (usage, seasons) -> {size of the word table: how many runs have it}
    for run in runs:
        c = run["config"]
        counts = table_counts.setdefault((c["usage"], c["seasons"]), {})
        counts[c["table_animals"]] = counts.get(c["table_animals"], 0) + 1
    # The usual table of a group: the size most of its runs have (the larger one in a tie).
    usual_table = {key: max(counts, key=lambda size: (counts[size], size))
                   for key, counts in table_counts.items()}

    found, notes = {}, []  # found: (usage, seasons) -> {(width, animals, seed): [runs there]}
    for run in runs:
        c = run["config"]
        key = (c["usage"], c["seasons"])
        if c["table_animals"] != usual_table[key]:
            notes.append(f"{folder(run)} has a word table of {c['table_animals']:,} animals, not "
                         f"the {usual_table[key]:,} of its group.")
        else:
            at_place = found.setdefault(key, {})
            at_place.setdefault((c["width"], c["animals"], c["seed"]), []).append(run)

    groups = {}
    for key, at_place in found.items():
        group = {}
        for (width, animals, seed), candidates in at_place.items():
            place = f"width {width} on {animals:,} animals with seed {seed}"
            run, more_notes = one_run_per_place(candidates, place)
            notes += more_notes
            if run is not None:
                group[(width, animals, seed)] = run
        if group:
            groups[key] = group
    return groups, sorted(notes)


def recipe_settings():
    """The settings that must be the same in every run of a group: every setting of a run
    (every field of RunConfig in train.py) that is not named in one of the four lists at the
    top of this file."""
    apart = DIALS + GROUP_KEYS + LENGTH + RUNNING
    return [field.name for field in dataclasses.fields(RunConfig) if field.name not in apart]


def setting_of(run, setting):
    """The value of a setting a run was trained with, or "not recorded" if the run was saved by
    older code, before the setting existed."""
    if setting == "set_sentences":
        # The set size is read as the language resolved it: 0 in the settings means "automatic".
        return run["language"]["set_sentences"]
    return run["config"].get(setting, "not recorded")


def plain(value):
    """A setting's value as it is written in a sentence: 0.01, 2000, not recorded."""
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def some_of(runs, limit=3):
    """How many runs, and the folders of the first few: "5 runs: A, B, C and 2 more"."""
    names = [folder(run) for run in runs[:limit]]
    if len(runs) > limit:
        names.append(f"{len(runs) - limit} more")
    return f"{len(runs)} run{'' if len(runs) == 1 else 's'}: {listed(names)}"


def recipe_problems(group):
    """Why the rule cannot be applied to a group, as a list of sentences; empty if it can.

    Two things are checked (see "WHAT THE RULE TAKES FOR GRANTED" at the top): every setting of
    the recipe has one and the same value in all runs, and every setting of the stopping rule has
    the value train.py gives it by default. A setting that a saved run does not have at all
    reads "not recorded" and counts as different.
    """
    runs = list(group.values())
    agreed = RunConfig()
    problems = []
    for setting in recipe_settings():
        with_value = {}  # value -> the runs trained with it
        for run in runs:
            with_value.setdefault(setting_of(run, setting), []).append(run)
        if setting in STOPPING_RULE:
            wanted = getattr(agreed, setting)
            for value, those in with_value.items():
                if value != wanted:
                    problems.append(
                        f"CAREFUL: the stopping rule was not the agreed one. {setting} is "
                        f"{plain(value)} in {some_of(those)}. The agreed value is {plain(wanted)}.")
        elif len(with_value) > 1:
            usual = max(with_value, key=lambda value: len(with_value[value]))
            for value, those in with_value.items():
                if value != usual:
                    problems.append(
                        f"CAREFUL: the runs were not all trained the same way. {setting} is "
                        f"{plain(value)} in {some_of(those)}. In the other runs of this group it "
                        f"is {plain(usual)}.")
    return problems


NOT_APPLIED = (
    "The rule is NOT applied to this group, and no verdict is given for it. The rule takes "
    "everything except the vocabulary, the width and the seed to be equal, and a plateau to be a "
    "plateau under the agreed stopping rule; here that is not so. Train the runs named above "
    "again with the settings of the others (delete their folders, then run the round again), or "
    "move them out of this folder."
)


def group_notes(group):
    """Things a reader should know about a group that do not stop the rule from being applied."""
    notes = []
    fixed = [run for run in group.values() if how_ended(run) == "fixed"]
    if fixed:
        notes.append(f"NOTE: this group holds runs of a fixed length ({some_of(fixed)}). The "
                     f"stopping rule was never asked for them: such a run can never count as a "
                     f"plateau, and where it does not rescue, nobody knows whether it would.")
    saved_by = {}  # code mark -> the runs saved by that version of the code (see train.py)
    for run in group.values():
        saved_by.setdefault(run.get("code", "not recorded"), []).append(run)
    if len(saved_by) > 1:
        marks = "; ".join(f"mark {mark} in {some_of(those, limit=1)}"
                          for mark, those in saved_by.items())
        notes.append(f"NOTE: the runs of this group were not all saved by the same version of "
                     f"the code ({marks}). The mark changes with every edit of language.py, "
                     f"model.py, score.py or train.py, also with one that changes no result. If "
                     f"the code was changed between these runs in a way that matters, they are "
                     f"not comparable.")
    return notes


def most_telling(vocabularies):
    """Of one seed's larger vocabularies, the one that says the most about the rule: the first on
    which (b) and (c) both hold; failing that, the first on which (b) holds; failing that, the
    one with the largest drop. The seed's sentence is about this one; the others with a drop are
    named after it (see describe_others)."""
    both = [v for v in vocabularies if v["b"] and v["c"]]
    if both:
        return both[0]
    plateau_drops = [v for v in vocabularies if v["b"]]
    if plateau_drops:
        return plateau_drops[0]
    return max(vocabularies, key=lambda v: v["drop"])


def outcome_on(vocabulary):
    """What one larger vocabulary says for a seed, as one word. CONCLUSIONS and IN_FEW_WORDS
    turn it into words. (Whether (a) holds is a matter of the seed: see judge_seed.)"""
    v = vocabulary
    if not v["dropped"]:
        return "no_drop"
    if not v["b"]:  # a drop, but the run did not end by plateau
        if v["stop_reason"] != "budget":
            return "not_plateau"
        return "not_begun" if v["below_start"] else "not_finished"
    if v["c"]:
        return "met"
    if v["next_score"] is None:
        return "no_next_width"
    # A drop after a plateau that the next width up did not fix. That is "not a size effect" only
    # if the next width up was itself trained to an ending of the stopping rule.
    return {"finished": "not_rescued", "budget": "next_not_finished",
            "fixed": "next_fixed"}[v["next_ended"]]


def judge_seed(group, width, seed):
    """Test (a), (b) and (c) for one width and one seed. Returns a dictionary with the answers
    and the numbers behind them. group is {(width, animals, seed): run}.

    "vocabularies" holds one dictionary for every larger vocabulary the width was trained on
    with this seed. The one that says most (see most_telling) is also copied into the result
    itself, so result["animals"], result["score"], result["b"], ... are about that vocabulary.
    """
    result = {"seed": seed, "a": False, "b": False, "c": False, "met": False, "dropped": False,
              "outcome": "no_small_run", "vocabularies": []}
    small = group.get((width, SMALL, seed))
    if small is None:
        return result
    result["small_score"] = facts(small)
    result["small_ended"] = how_ended(small)
    result["a"] = at_least(facts(small), HIGH)  # (a)

    vocabularies = []
    for animals in sorted({animals for w, animals, s in group if w == width and s == seed}):
        if animals <= SMALL:
            continue
        run = group[(width, animals, seed)]
        drop = facts(small) - facts(run)
        dropped = at_least(drop, DROP)
        # The rescuer: the next width up on the same vocabulary with the same seed. The same seed
        # means the same training sets, so only the model's size differs between the two runs.
        rescuer = group.get((width * WIDTH_STEP, animals, seed))
        rescue = None if rescuer is None else facts(rescuer) - facts(run)
        vocabulary = {
            "animals": animals, "score": facts(run), "stop_reason": run["stop_reason"],
            "drop": drop, "dropped": dropped,
            # (b): a drop counts only if the run ended by plateau. A run that ended by budget is
            # not finished: nobody knows how far it would have got. A run of a fixed length
            # ("fixed") never asked the stopping rule, so it cannot count either.
            "b": dropped and run["stop_reason"] == "plateau",
            # Below started_at (see train.py) the stopping rule never calls a plateau. A run
            # that stays there can only end by budget, however long it is given.
            "below_start": facts(run) < RunConfig.started_at,
            # The best score the run had at any grading. If the run ended far below it, the
            # score fell during training, which no limit of size explains.
            "best": max(point["facts_known"] for point in run["curve"]),
            "next_score": None if rescuer is None else facts(rescuer),
            "next_ended": None if rescuer is None else how_ended(rescuer),
            "rescue": rescue,
            "c": rescue is not None and at_least(rescue, RESCUE),  # (c)
        }
        vocabulary["outcome"] = outcome_on(vocabulary)
        vocabularies.append(vocabulary)
    if not vocabularies:
        result["outcome"] = "no_larger_run"
        return result
    result["vocabularies"] = vocabularies
    result.update(most_telling(vocabularies))
    # All three must hold together, (b) and (c) on the same vocabulary.
    result["met"] = result["a"] and result["b"] and result["c"]
    if not result["a"]:  # without (a) nothing else can make the seed count
        result["outcome"] = "not_high"
    return result


CONCLUSIONS = {
    "met": "All three hold.",
    "not_high": f"Without (a) this seed cannot count: the width does not reach {HIGH} even on "
                f"{SMALL} animals.",
    "not_rescued": "A drop that the next width up does not fix is NOT a size effect.",
    "next_not_finished": "The next width up was not finished, so nobody knows whether it would "
                         "fix the drop: train it longer before calling this either way.",
    "next_fixed": "The next width up had a fixed number of sets and never asked the stopping "
                  "rule, so nobody knows whether it would fix the drop: train it as an ordinary "
                  "run before calling this either way.",
    "no_next_width": "Without a run of the next width up the drop cannot be called a size effect.",
    "not_finished": "A run that is not finished shows no limit of the model: train it longer "
                    "before reading anything into this drop.",
    "not_begun": "The stopping rule never calls a plateau before a model has begun to learn "
                 "facts. More training helps only if this model then begins; if it never does, "
                 "the run ends by budget however long it trains, and this drop can never count.",
    "not_plateau": "Only a drop in a run that ended by plateau counts.",
    "no_drop": "No drop in this seed.",
}

# The same outcomes in a few words, for the vocabularies a seed's main sentence is not about.
IN_FEW_WORDS = {
    "met": "(b) and (c) hold there too",
    "not_rescued": "not fixed by the next width up, so not a size effect",
    "next_not_finished": "the next width up is not finished, so this cannot be judged yet",
    "next_fixed": "the next width up never asked the stopping rule, so this cannot be judged yet",
    "no_next_width": "there is no run of the next width up, so (c) cannot be tested",
    "not_finished": "that run cannot count",
    "not_begun": "that run had not begun to learn facts and cannot count",
    "not_plateau": "only a plateau counts",
}

# What is said about a run in passing, by how it ended (see how_ended).
REMARK = {"finished": "", "budget": " (that run was NOT FINISHED)",
          "fixed": " (that run had a fixed number of sets)"}
ENDED_AS = {"plateau": "after a plateau", "solved": "solved",
            "budget": "ended by budget: NOT FINISHED", "fixed": "after a fixed number of sets"}


def shown(value, threshold=None):
    """A score or a difference as the tables show it, with one decimal.

    If the value is below a threshold it has to reach, it is given as many decimals as it takes
    to see that: 9.993 beside "a drop needs 10" is printed as 9.99, not as 10.0.
    """
    for decimals in range(1, 7):
        text = f"{value:.{decimals}f}"
        if threshold is None or at_least(value, threshold) or float(text) < threshold:
            break
    return text


def change(points, more, less, threshold=None):
    """A difference in words: change(3.2, "lower", "higher") -> "3.2 points lower"."""
    return f"{shown(abs(points), threshold)} points {more if points >= 0 else less}"


def describe_b(v):
    """The answer to (b) for one larger vocabulary of a seed, as a sentence."""
    where = f"{v['score']:.1f} on {v['animals']:,} animals"
    lower = change(v["drop"], "lower", "higher", DROP)
    if v["b"]:
        b = f"(b) yes: {where} after a plateau, {lower}."
        if at_least(v["best"] - v["score"], DROP):
            b += (f" CAREFUL: this run had reached {v['best']:.1f} earlier and fell back. A "
                  f"score that falls during training points to unstable training, not to a "
                  f"limit of the model's size: look at this run in chart 2.")
        return b
    if v["dropped"] and v["stop_reason"] == "budget" and v["below_start"]:
        return (f"(b) no: {where}, {lower}, but that run ended by budget before it had begun to "
                f"learn facts (fewer than {RunConfig.started_at:g} of 100 known): it is NOT "
                f"FINISHED and cannot count.")
    if v["dropped"] and v["stop_reason"] == "budget":
        return (f"(b) no: {where}, {lower}, but that run ended by budget: it is NOT FINISHED and "
                f"cannot count.")
    if v["dropped"] and v["stop_reason"] == "fixed":
        return (f"(b) no: {where}, {lower}, but that run had a fixed number of sets: the stopping "
                f"rule was never asked, so it cannot count as a plateau.")
    if v["dropped"]:
        return f"(b) no: {where}, {lower}, but that run ended as {v['stop_reason']}."
    return (f"(b) no: its lowest score on a larger vocabulary is {where}, {lower}; a drop "
            f"needs {DROP}.")


def describe_c(v, width, widths):
    """The answer to (c) for one larger vocabulary of a seed, as a sentence. `widths` are the
    widths that have runs in the group."""
    next_width = width * WIDTH_STEP
    if v["next_score"] is not None:
        higher = change(v["rescue"], "higher", "lower", RESCUE)
        needs = "" if v["c"] else f"; a rescue needs {RESCUE}"
        return (f"(c) {'yes' if v['c'] else 'no'}: width {next_width} scores "
                f"{v['next_score']:.1f} there{REMARK[v['next_ended']]}, {higher}{needs}.")
    if next_width in widths:
        return (f"(c) cannot be tested: no run of width {next_width} on {v['animals']:,} animals "
                f"with this seed.")
    c = (f"(c) cannot be tested: there are no runs of width {next_width}, the next width up, "
         f"among these runs.")
    larger = [w for w in widths if w > width]
    if larger:
        c += f" Width {larger[0]} is not the next width up and cannot stand in for it."
    return c


def describe_others(result, width):
    """Sentences about the larger vocabularies that a seed's main sentence is not about.

    Every other vocabulary on which the score dropped gets one sentence with its own answer, so
    that a drop the main sentence does not mention cannot go unseen. And where the main sentence
    is about a drop, the larger vocabularies WITHOUT a drop are listed: a reader should know
    when a drop does not go on as the vocabulary grows.
    """
    others = [v for v in result["vocabularies"] if v["animals"] != result["animals"]]
    sentences = []
    for v in others:
        if not v["dropped"]:
            continue
        ended = ENDED_AS.get(v["stop_reason"], v["stop_reason"])
        beside = ""
        if v["next_score"] is not None:
            beside = (f"; width {width * WIDTH_STEP} scores {v['next_score']:.1f} "
                      f"there{REMARK[v['next_ended']]}")
        sentences.append(f"Also lower on {v['animals']:,} animals: {v['score']:.1f} ({ended}), "
                         f"{change(v['drop'], 'lower', 'higher')}{beside}. In short: "
                         f"{IN_FEW_WORDS[v['outcome']]}.")
    no_drop = [f"{v['score']:.1f} on {v['animals']:,} animals" for v in others
               if not v["dropped"] and v["animals"] > result["animals"]]
    if result["dropped"] and no_drop:
        sentences.append(f"No such drop on the larger vocabularies: {listed(no_drop)}.")
    return sentences


def describe_seed(result, width, widths):
    """One seed's answers to (a), (b) and (c) as plain sentences with the numbers. `widths` are
    the widths that have runs in the group."""
    start = f"seed {result['seed']}: "
    if result["outcome"] == "no_small_run":
        return start + f"no run on {SMALL} animals, so nothing can be compared."
    a = (f"(a) {'yes' if result['a'] else 'no'}: {shown(result['small_score'], HIGH)} on {SMALL} "
         f"animals{REMARK[result['small_ended']]}.")
    if result["outcome"] == "no_larger_run":
        return start + a + " No run on a larger vocabulary, so (b) and (c) cannot be tested."
    sentences = [start + a, describe_b(result), describe_c(result, width, widths),
                 CONCLUSIONS[result["outcome"]]]
    return " ".join(sentences + describe_others(result, width))


def judge_width(group, width):
    """Apply the rule to one width of one group. Returns the verdict as a dictionary:
    width, next_width, met (True or False), seeds_met, held (in how many seeds each of a, b, c
    held), seeds (one judge_seed dictionary per seed) and sentence (the verdict in words)."""
    seeds = sorted({seed for w, _, seed in group if w == width})
    results = [judge_seed(group, width, seed) for seed in seeds]
    seeds_met = [result["seed"] for result in results if result["met"]]
    held = {letter: sum(result[letter] for result in results) for letter in "abc"}
    met = len(seeds_met) >= SEEDS_NEEDED
    sentence = (
        f"Width {width}: the rule is {'MET' if met else 'NOT met'}. (a), (b) and (c) all hold in "
        f"{len(seeds_met)} of {len(seeds)} seeds ({SEEDS_NEEDED} needed). Taken one at a time, "
        f"(a) holds in {held['a']} of {len(seeds)} seeds, (b) in {held['b']} and (c) in "
        f"{held['c']}."
    )
    return {"width": width, "next_width": width * WIDTH_STEP, "met": met, "seeds_met": seeds_met,
            "held": held, "seeds": results, "sentence": sentence}


def widths_in_words(widths):
    """[4] -> "width 4"; [4, 8, 16] -> "widths 4, 8 and 16"."""
    return ("width " if len(widths) == 1 else "widths ") + listed(widths)


def where_found(verdicts, outcome):
    """The widths, with their vocabularies, on which some seed shows `outcome`, in words:
    "width 4 on 64 animals and width 8 on 64 and 256 animals". Empty if there are none.

    Every larger vocabulary of every seed is looked at, not only the one a seed's main sentence
    is about. Widths for which the rule is met are left out, and so are seeds without (a).
    """
    found = {}  # width -> the vocabularies
    for verdict in verdicts:
        for seed in verdict["seeds"]:
            for v in seed["vocabularies"]:
                if v["outcome"] == outcome and seed["a"] and not verdict["met"]:
                    found.setdefault(verdict["width"], set()).add(v["animals"])
    return listed(f"width {width} on {listed(f'{a:,}' for a in sorted(animals))} animals"
                  for width, animals in sorted(found.items())) if found else ""


def notes_on_a_met_width(verdict):
    """Two things the agreed rule allows and a reader should still be told about.

    The rule asks for SEEDS_NEEDED seeds, not for most of the seeds trained. And it asks for
    "some larger vocabulary" in each seed, not for the same vocabulary in all of them.
    """
    notes = []
    width, seeds = verdict["width"], verdict["seeds"]
    counted = [seed for seed in seeds if seed["met"]]
    if 2 * len(counted) < len(seeds):
        notes.append(f"Note on width {width}: the three conditions hold in {len(counted)} of "
                     f"{len(seeds)} seeds. The agreed rule asks for {SEEDS_NEEDED} seeds, however "
                     f"many were trained; it does not ask for most of them.")
    on_vocabulary = {}  # vocabulary -> in how many of the counted seeds (b) and (c) hold there
    for seed in counted:
        for v in seed["vocabularies"]:
            if v["b"] and v["c"]:
                on_vocabulary[v["animals"]] = on_vocabulary.get(v["animals"], 0) + 1
    if max(on_vocabulary.values()) < SEEDS_NEEDED:
        places = listed(f"{animals:,}" for animals in sorted(on_vocabulary))
        notes.append(f"Note on width {width}: no single vocabulary shows the drop and the rescue "
                     f"in {SEEDS_NEEDED} seeds; the seeds that count do so on different ones "
                     f"({places} animals). The agreed rule asks for some larger vocabulary in "
                     f"each seed, not for the same one.")
    return notes


def summarize_group(label, verdicts):
    """The summary line of one group: the widths for which the rule is met. It is followed by a
    plain sentence for every way in which the other widths fell short."""
    met = [verdict["width"] for verdict in verdicts if verdict["met"]]
    if met:
        lines = [f"SUMMARY for {label}: the rule is met for {widths_in_words(met)}. By the "
                 f"agreed rule the hypothesis clearly holds there."]
    else:
        lines = [f"SUMMARY for {label}: the rule is met for no width."]
    for verdict in verdicts:
        if verdict["met"]:
            lines += notes_on_a_met_width(verdict)

    seeds = [seed for verdict in verdicts for seed in verdict["seeds"]]
    # The larger vocabularies that some width could be compared on, with its 16-animal run.
    compared = sorted({v["animals"] for seed in seeds for v in seed["vocabularies"]})
    any_drop = any(seed["dropped"] for seed in seeds)
    any_high = any(verdict["held"]["a"] > 0 for verdict in verdicts)
    if not compared:
        lines.append(f"Nothing could be compared: no width has, with the same seed, a run on "
                     f"{SMALL} animals and a run on a larger vocabulary. This says nothing about "
                     f"the hypothesis.")
    elif not any_drop and any_high:
        on = listed(f"{animals:,}" for animals in compared)
        lines.append(f"No width scored {DROP} or more points lower on a larger vocabulary than on "
                     f"{SMALL} animals (larger vocabularies compared: {on} animals). The models "
                     f"were not stretched: this is neither support for the hypothesis nor a "
                     f"refutation of it.")
    elif not any_drop and not any_high:
        lines.append(f"No width reached {HIGH} on {SMALL} animals, so there was no good score to "
                     f"drop from. Nothing was tested: this is neither support for the hypothesis "
                     f"nor a refutation of it.")

    where = {outcome: where_found(verdicts, outcome) for outcome in IN_FEW_WORDS}
    if where["not_rescued"]:
        lines.append(f"A drop for {where['not_rescued']} was not fixed by the next width up: "
                     f"that drop is not a size effect.")
    if where["next_not_finished"]:
        lines.append(f"A drop for {where['next_not_finished']} could not be judged: the next "
                     f"width up ended by budget and did not fix it so far. Train that run longer "
                     f"(--max-minutes).")
    if where["next_fixed"]:
        lines.append(f"A drop for {where['next_fixed']} could not be judged: the next width up "
                     f"had a fixed number of sets, so the stopping rule was never asked, and it "
                     f"did not fix the drop. Train that run as an ordinary run (without "
                     f"--fixed-sets).")
    if where["not_finished"]:
        lines.append(f"A drop for {where['not_finished']} comes from a run that ended by budget. "
                     f"That run is not finished and cannot count; train it longer (--max-minutes) "
                     f"before reading anything into the drop.")
    if where["not_begun"]:
        lines.append(f"A drop for {where['not_begun']} comes from a run that ended by budget "
                     f"before it had begun to learn facts (fewer than {RunConfig.started_at:g} "
                     f"facts out of 100 known). The stopping rule never calls a plateau there, so "
                     f"such a run is NOT FINISHED and cannot count for (b). More minutes help "
                     f"only if the model then begins to learn. If it never begins, the run ends "
                     f"by budget however long it trains, and the agreed rule can reach no verdict "
                     f"from it.")
    if where["not_plateau"]:
        lines.append(f"A drop for {where['not_plateau']} comes from a run that did not end by "
                     f"plateau (a run with a fixed number of sets never asks the stopping rule). "
                     f"Such a drop cannot count.")
    if where["no_next_width"]:
        lines.append(f"A drop for {where['no_next_width']} has no run of the next width up "
                     f"beside it, so (c) cannot be tested.")
    one_seed = [verdict["width"] for verdict in verdicts
                if 0 < len(verdict["seeds_met"]) < SEEDS_NEEDED]
    if one_seed:
        lines.append(f"For {widths_in_words(one_seed)} all three conditions hold in fewer than "
                     f"{SEEDS_NEEDED} seeds: one seed alone could be luck.")
    not_high = [verdict["width"] for verdict in verdicts if not verdict["met"]
                and any(seed["outcome"] == "not_high" for seed in verdict["seeds"])]
    if not_high:
        lines.append(f"A seed of {widths_in_words(not_high)} did not reach {HIGH} "
                     f"on {SMALL} animals: too small, or not trained enough, even for the small "
                     f"vocabulary, so it cannot show the effect of a larger one.")
    return lines


def rule_in_words():
    agreed = RunConfig()
    stopping_rule = ", ".join(f"{setting} {plain(getattr(agreed, setting))}"
                              for setting in STOPPING_RULE)
    return [
        "THE AGREED DECISION RULE (fixed before any experiment was run)",
        f'For one model width, at one usage and one number of seasons, the hypothesis "clearly '
        f'holds" when, in at least {SEEDS_NEEDED} seeds, all of these are true:',
        f"  (a) the width scores {HIGH} or more on the {SMALL}-animal language (the score is "
        f"facts known, 0 to 100);",
        f"  (b) on some larger vocabulary its run ended by plateau and scored at least {DROP} "
        f"points below its own {SMALL}-animal score;",
        f"  (c) on that vocabulary the next width up, with the same seed and so the same training "
        f"sets, scores at least {RESCUE} points higher.",
        "A run that ended by budget is NOT FINISHED and can never satisfy (b).",
        "Nor can a run that was given a fixed number of sets (--fixed-sets): the stopping rule "
        "was never asked.",
        f"The next width up of a width is {WIDTH_STEP} times that width (8 for width 4). No "
        f"other width can stand in for it.",
        f"A plateau is a plateau under the stopping rule of train.py with its agreed settings "
        f"({stopping_rule}). It is never called while a run knows fewer than "
        f"{agreed.started_at:g} facts out of 100, so a run that never begins to learn facts ends "
        f"by budget and cannot satisfy (b).",
        "The rule is applied only to a group of runs that were all trained the same way and "
        "under that stopping rule.",
    ]


def rule_met(runs):
    """Apply THE AGREED DECISION RULE to a list of runs (from load_runs).

    Returns a dictionary:
      met       {(usage, seasons): [the widths for which the rule is met]}, one entry for every
                group the rule could be applied to
      refused   {(usage, seasons): [why]} for every group the rule was NOT applied to, because
                its runs were not trained the same way or not under the agreed stopping rule
      verdicts  one dictionary per group and width (see judge_width), with usage and seasons
      lines     the whole verdict as lines of plain text; print_rule prints them
    """
    groups, notes = comparison_groups(runs)
    lines = rule_in_words()
    met, refused, verdicts = {}, {}, []
    for (usage, seasons), group in sorted(groups.items()):
        label = group_label(usage, seasons)
        lines += ["", f"--- {label} ---"]
        problems = recipe_problems(group)
        if problems:
            refused[(usage, seasons)] = problems
            lines += problems + [NOT_APPLIED]
            continue
        lines += group_notes(group)
        vocabularies = sorted({animals for _, animals, _ in group})
        if SMALL not in vocabularies or vocabularies[-1] <= SMALL:
            found = ", ".join(f"{animals:,}" for animals in vocabularies)
            lines.append(f"The rule cannot be applied here: it needs runs on {SMALL} animals and "
                         f"on at least one larger vocabulary (found: {found}).")
            continue
        widths = sorted({width for width, _, _ in group})
        of_group = []
        for width in widths:
            verdict = judge_width(group, width)
            verdict.update(usage=usage, seasons=seasons)
            of_group.append(verdict)
            lines.append(verdict["sentence"])
            lines += ["    " + describe_seed(seed, width, widths) for seed in verdict["seeds"]]
        met[(usage, seasons)] = [verdict["width"] for verdict in of_group if verdict["met"]]
        lines += summarize_group(label, of_group)
        verdicts += of_group
    if notes:
        lines += ["", "Left out of the rule and of charts 1, 2 and 4 (still in the table):"]
        lines += ["  " + note for note in notes]
    return {"met": met, "refused": refused, "verdicts": verdicts, "lines": lines}


def print_rule(result):
    """Print the lines of rule_met, folded to the width of a terminal."""
    for line in result["lines"]:
        indent = " " * (len(line) - len(line.lstrip()))
        # break_on_hyphens=False keeps a flag such as --max-minutes in one piece on one line.
        print(textwrap.fill(line, width=100, subsequent_indent=indent + "    ",
                            break_on_hyphens=False) if line else "")


# ------------------------------------------------------------------------------------------------
# Chart building blocks
# ------------------------------------------------------------------------------------------------


def width_color(width):
    return WIDTH_COLORS.get(width, OTHER_WIDTH_COLOR)


def style_panel(panel):
    """The quiet look shared by every panel: thin gray grid lines, no box, a 0-100 axis."""
    panel.set_facecolor(SURFACE)
    for side in ["top", "right", "left", "bottom"]:
        panel.spines[side].set_visible(False)
    panel.grid(axis="y", color=GRID, linewidth=HAIR, linestyle="-")
    panel.axhline(0, color=AXIS, linewidth=HAIR, zorder=1)  # the zero line, a little darker
    panel.set_axisbelow(True)
    panel.tick_params(axis="both", which="both", colors=INK_MUTED, labelsize=8.5, length=0)
    panel.set_ylim(-5, 105)  # a little room, so that dots at 0 and at 100 are not cut in half
    panel.set_yticks([0, 25, 50, 75, 100])


def new_figure(n_panels, title, subtitle, legend_rows, panel_titles=True):
    """An empty picture: title, subtitle, legend, and n_panels panels (three to a row).

    legend_rows is a list of rows, each a list of legend entries (see key_line and key_dot).
    panel_titles says whether room is kept above each panel for a title of its own.
    The layout is worked out in inches from the top down. Returns (figure, list of panels).
    """
    columns = min(n_panels, PANELS_PER_ROW)
    rows = math.ceil(n_panels / PANELS_PER_ROW)
    panel_width = WIDE_PANEL if n_panels == 1 else PANEL_WIDTH
    width = MARGIN_LEFT + columns * panel_width + (columns - 1) * GAP_X + MARGIN_RIGHT
    subtitle_lines = textwrap.wrap(subtitle, width=int(width * 13.5))
    legend_top = 0.62 + 0.2 * len(subtitle_lines) + 0.08
    header = legend_top + 0.27 * len(legend_rows) + (0.5 if panel_titles else 0.25)
    height = header + rows * PANEL_HEIGHT + (rows - 1) * GAP_Y + MARGIN_BOTTOM
    figure = plt.figure(figsize=(width, height), dpi=100, facecolor=SURFACE)

    text_left = 0.3 / width
    figure.text(text_left, 1 - 0.22 / height, title, fontsize=14, fontweight="bold", color=INK,
                ha="left", va="top")
    figure.text(text_left, 1 - 0.6 / height, "\n".join(subtitle_lines), fontsize=9.5,
                color=INK_SOFT, ha="left", va="top", linespacing=1.45)
    for number, entries in enumerate(legend_rows):
        figure.legend(handles=entries, loc="upper left", ncol=len(entries), frameon=False,
                      bbox_to_anchor=(text_left, 1 - (legend_top + 0.27 * number) / height),
                      fontsize=9, labelcolor=INK_SOFT, borderaxespad=0, borderpad=0,
                      handlelength=1.7, handletextpad=0.5, columnspacing=1.6)

    panels = []
    for number in range(n_panels):
        row, column = divmod(number, PANELS_PER_ROW)
        left = MARGIN_LEFT + column * (panel_width + GAP_X)
        bottom = height - header - (row + 1) * PANEL_HEIGHT - row * GAP_Y
        panel = figure.add_axes([left / width, bottom / height, panel_width / width,
                                 PANEL_HEIGHT / height])
        style_panel(panel)
        panels.append(panel)
    return figure, panels


def save_figure(figure, path):
    figure.savefig(path, dpi=SAVE_DPI, facecolor=SURFACE)
    plt.close(figure)


def key_line(color, label, thickness=LINE):
    """A legend entry for a line."""
    return Line2D([], [], color=color, linewidth=thickness, label=label)


def key_dot(color, ended, label, shape="o"):
    """A legend entry for a dot, drawn like the dots of draw_dot. `ended` is a word of
    how_ended: filled for "finished", hollow for "budget", half-filled for "fixed"."""
    if ended == "finished":
        return Line2D([], [], linestyle="none", marker=shape, markersize=DOT - 2 * PX,
                      markerfacecolor=color, markeredgecolor=color, label=label)
    if ended == "fixed":
        return Line2D([], [], linestyle="none", marker=shape, markersize=HALF_DOT,
                      fillstyle="left", markerfacecolor=color, markerfacecoloralt=SURFACE,
                      markeredgecolor=color, markeredgewidth=0.75 * LINE, label=label)
    return Line2D([], [], linestyle="none", marker=shape, markersize=HOLLOW_DOT,
                  markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=LINE,
                  label=label)


def run_keys():
    """The two legend entries that say what a filled and a hollow dot mean."""
    return [key_dot(INK_MUTED, "finished", "one run"),
            key_dot(INK_MUTED, "budget", "one run that was NOT FINISHED (ended by budget)")]


def fixed_key(runs):
    """One more legend row, only if a run of a fixed length is among the runs drawn: what a
    half-filled dot means. Returns a list of legend rows, empty if there is no such run."""
    if not any(how_ended(run) == "fixed" for run in runs):
        return []
    return [[key_dot(INK_MUTED, "fixed", "one run of a fixed length: the stopping rule was not "
                                         "asked, so this is no limit either")]]


def draw_dot(panel, x, y, color, ended, shift=0.0, shape="o"):
    """One run as a dot. `ended` is a word of how_ended: the dot is filled if the stopping rule
    ended the run, hollow if it ended by budget, half-filled if it had a fixed number of sets.

    shift moves the dot sideways by that many pixels, whatever the scale of the axis. A filled
    dot has a thin ring in the background color, which keeps overlapping dots apart. Hollow and
    half-filled dots are drawn above the filled ones, so that a filled dot never hides them.
    """
    shifted = offset_copy(panel.transData, fig=panel.figure, x=shift * PX, units="points")
    if ended == "finished":
        look = dict(markersize=DOT, markerfacecolor=color, markeredgecolor=SURFACE,
                    markeredgewidth=LINE, zorder=3)
    elif ended == "fixed":
        look = dict(markersize=HALF_DOT, fillstyle="left", markerfacecolor=color,
                    markerfacecoloralt=SURFACE, markeredgecolor=color,
                    markeredgewidth=0.75 * LINE, zorder=3.5)
    else:
        look = dict(markersize=HOLLOW_DOT, markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=LINE, zorder=3.5)
    panel.plot([x], [y], linestyle="none", marker=shape, transform=shifted, clip_on=False, **look)


def dot_step(panel, xs, n_widths, n_seeds):
    """How many pixels apart neighboring dots are set: DOT_STEP, or fewer if the dots of one
    place on the x-axis would then run into those of the next place.

    xs are the places on the x-axis. Call this after the limits of the x-axis are set: the
    room is measured on the axis as it will be drawn.
    """
    if len(xs) < 2:
        return DOT_STEP
    slots = n_widths * (n_seeds + WIDTH_GAP) - WIDTH_GAP  # dots side by side at one place
    pixels = panel.transData.transform([(x, 0) for x in xs])[:, 0]
    room = 0.75 * np.diff(np.sort(pixels)).min()  # keep some air between two places
    return min(DOT_STEP, room / slots)


def sideways(width_place, seed_place, n_widths, n_seeds, step):
    """How far, in pixels, the dot of one run is set from its place on the x-axis.

    Runs at the same place on the x-axis differ in their width and in their seed. If they were
    all drawn at that place, runs with equal scores would hide each other: on 16 animals every
    width is expected to score near 100. So every run gets a slot of its own: the widths stand
    side by side as blocks (width_place 0, 1, 2, ... from the left), and inside a block the
    seeds stand side by side (seed_place 0, 1, 2).
    """
    block = n_seeds + WIDTH_GAP  # the slots one width takes, with the gap after it
    slot = width_place * block + seed_place
    middle = (n_widths * block - WIDTH_GAP - 1) / 2
    return (slot - middle) * step


def draw_dots_and_means(panel, points, color, width_place, n_widths, n_seeds, step):
    """Draw one width in one panel: a dot for every run and a line through the means.

    points is a list of (x, y, ended, seed_place); ended is a word of how_ended. Every dot is
    set a little to the side of its x (see sideways), and the line runs through the middle of
    its own dots. Returns where the line ends, as (x, y, pixels to the side).
    """
    xs = sorted({x for x, _, _, _ in points})
    means = [np.mean([y for x, y, _, _ in points if x == place]) for place in xs]
    middle = sideways(width_place, (n_seeds - 1) / 2, n_widths, n_seeds, step)
    shifted = offset_copy(panel.transData, fig=panel.figure, x=middle * PX, units="points")
    panel.plot(xs, means, color=color, linewidth=LINE, solid_capstyle="round",
               solid_joinstyle="round", zorder=2, transform=shifted, clip_on=False)
    for x, y, ended, seed_place in points:
        draw_dot(panel, x, y, color, ended,
                 sideways(width_place, seed_place, n_widths, n_seeds, step))
    return xs[-1], means[-1], middle


def push_up(heights, gap):
    """Sorted numbers, each raised just enough to be `gap` above the one before it."""
    raised = list(heights)
    for i in range(1, len(raised)):
        raised[i] = max(raised[i], raised[i - 1] + gap)
    return np.array(raised)


def push_down(heights, gap):
    """Sorted numbers, each lowered just enough to be `gap` below the one after it."""
    lowered = list(heights)
    for i in range(len(lowered) - 2, -1, -1):
        lowered[i] = min(lowered[i], lowered[i + 1] - gap)
    return np.array(lowered)


def spread_apart(heights, gap, low, high):
    """Move numbers apart until neighbors are at least `gap` apart, keeping their order and
    staying inside low..high. Used so that labels do not print on top of each other."""
    order = np.argsort(heights, kind="stable")
    wanted = np.clip(np.array(heights, dtype=float)[order], low, high)
    # Pushing only upward would carry a crowd of labels far above their lines, and pushing only
    # downward far below. Halfway between the two keeps the crowd centered where it belongs.
    placed = (push_up(wanted, gap) + push_down(wanted, gap)) / 2
    if placed[0] < low:  # the crowd sticks out at the bottom: stack it up from the bottom
        placed = push_up(np.maximum(placed, low), gap)
    if placed[-1] > high:  # it sticks out at the top: stack it down from the top
        placed = push_down(np.minimum(placed, high), gap)
    result = np.empty(len(placed))
    result[order] = placed
    return result


def label_line_ends(panel, ends):
    """Write each line's name at the right edge of the panel, level with where the line ends.

    ends is a list of (x, y, pixels to the side, text): where a line ends and what it is called.
    Names that would print on top of each other are moved apart; a thin gray line joins each
    name to its line. The names are in ink, not in the line's color: colored text is hard to read.
    """
    if not ends:
        return
    heights = spread_apart([y for _, y, _, _ in ends], LABEL_GAP, 0, 100)
    for (x, y, shift, text), height in zip(ends, heights):
        shifted = offset_copy(panel.transData, fig=panel.figure, x=shift * PX, units="points")
        panel.annotate(
            text, xy=(x, y), xycoords=shifted,
            xytext=(1.035, height), textcoords=panel.get_yaxis_transform(),  # x: share of panel
            ha="left", va="center", fontsize=9, color=INK_SOFT, annotation_clip=False, zorder=1.5,
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=HAIR, shrinkA=2, shrinkB=8),
        )


def plain_number(value, _=None):
    """Axis numbers written out: 1,000 instead of 10^3, and 0.01 instead of 10^-2."""
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def log_axis(panel, low, high):
    """Give a panel a log-scale x-axis from low to high (see "log scale" at the top)."""
    panel.set_xscale("log")
    if high / low < 3.3:  # a very short range: widen it, or no marked number might fall inside
        middle = math.sqrt(low * high)
        low, high = middle / 1.82, middle * 1.82
    panel.set_xlim(low / 1.35, high * 1.35)
    # Over a short range also mark 2 and 5 (100, 200, 500, 1,000); otherwise only 10, 100, ...
    steps = [1, 2, 5] if high / low < 1000 else [1]
    panel.xaxis.set_major_locator(LogLocator(base=10, subs=steps))
    panel.xaxis.set_major_formatter(FuncFormatter(plain_number))
    panel.xaxis.set_minor_locator(NullLocator())


def label_axes(panel, number, x_label, y_label):
    """Name the axes. The y-axis is named only on the first panel of each row."""
    panel.set_xlabel(x_label, fontsize=9, color=INK_SOFT, labelpad=7)
    if number % PANELS_PER_ROW == 0:
        panel.set_ylabel(y_label, fontsize=9, color=INK_SOFT, labelpad=7)


def panel_title(panel, text):
    panel.set_title(text, loc="left", fontsize=10, color=INK, pad=12)


def bits_note(runs):
    """ "8 bits per animal", from the runs' own numbers."""
    bits = np.mean([run["language"]["bits_per_animal"] for run in runs])
    return f"{bits:.0f} bits per animal"


# ------------------------------------------------------------------------------------------------
# The four charts. Each returns True if it was drawn and False if there were no runs for it.
# ------------------------------------------------------------------------------------------------


def chart_more_words(groups, path):
    """Chart 1, "More words, same model": does the score fall as the vocabulary grows?

    x = animal words in use (log scale), y = facts known, one line per width, one panel per
    usage and number of seasons. A panel needs runs on at least two vocabulary sizes.
    """
    keys = [key for key in sorted(groups) if len({animals for _, animals, _ in groups[key]}) >= 2]
    if not keys:
        return False
    widths = sorted({width for key in keys for width, _, _ in groups[key]})
    drawn = [run for key in keys for run in groups[key].values()]
    legend = [[key_line(width_color(width), f"width {width}") for width in widths],
              run_keys() + [key_line(INK_MUTED, "line: the mean over seeds")]] + fixed_key(drawn)
    figure, panels = new_figure(
        len(keys), CHART_TITLES[1],
        "Every model of one width has exactly the same number of parameters, whatever the "
        "vocabulary. A line that falls to the right means: more animal words, fewer facts known. "
        "The runs on one vocabulary are set side by side: the widths from left to right, and "
        "for each width one dot per seed.", legend)
    for number, (panel, key) in enumerate(zip(panels, keys)):
        group = groups[key]
        seeds = sorted({seed for _, _, seed in group})
        of_panel = sorted({width for width, _, _ in group})
        vocabularies = sorted({animals for _, animals, _ in group})
        panel.set_xscale("log", base=2)
        # The limits come before the dots: dot_step measures the room on the finished axis.
        panel.set_xlim(vocabularies[0] / 2, vocabularies[-1] * 2)
        step = dot_step(panel, vocabularies, len(of_panel), len(seeds))
        ends = []
        for place, width in enumerate(of_panel):
            points = [(animals, facts(run), how_ended(run), seeds.index(seed))
                      for (w, animals, seed), run in group.items() if w == width]
            end = draw_dots_and_means(panel, points, width_color(width), place, len(of_panel),
                                      len(seeds), step)
            ends.append((*end, f"width {width}"))
        label_line_ends(panel, ends)
        panel.set_xticks(vocabularies)
        panel.set_xticklabels([f"{animals:,}" for animals in vocabularies])
        panel.xaxis.set_minor_locator(NullLocator())
        panel_title(panel, f"{group_label(*key)} ({bits_note(group.values())})")
        label_axes(panel, number, "animal words in use (log scale)", "facts known (0 to 100)")
    save_figure(figure, path)
    return True


def mean_curve(runs):
    """The learning curves of several seeds as one curve: the mean at every set count that ALL
    of them reached. Returns (times each fact was seen, facts known).

    Where one seed has already ended, a mean over the others would jump for no reason, so the
    mean stops where the first seed stops.
    """
    shared = set.intersection(*[{point["sets"] for point in run["curve"]} for run in runs])
    xs, ys = [], []
    for sets in sorted(shared):
        points = [point for run in runs for point in run["curve"] if point["sets"] == sets]
        xs.append(np.mean([point["sightings"] for point in points]))
        ys.append(np.mean([point["facts_known"] for point in points]))
    return xs, ys


def chart_learning(groups, path):
    """Chart 2, "Learning against practice": the score as training goes on, for the largest
    vocabulary of each group. x = times each fact was seen (log scale), y = facts known."""
    keys = sorted(groups)
    if not keys:
        return False
    largest = {key: max(animals for _, animals, _ in groups[key]) for key in keys}
    widths = sorted({w for key in keys for w, animals, _ in groups[key] if animals == largest[key]})
    legend = [[key_line(width_color(width), f"width {width}") for width in widths],
              [key_line(INK_MUTED, "thick: the mean over seeds, while all are training"),
               key_line(INK_MUTED, "thin: one seed", HAIR)],
              [key_dot(INK_MUTED, "finished", "where a run ended"),
               key_dot(INK_MUTED, "budget", "where a run stopped NOT FINISHED (budget)")]]
    legend += fixed_key([run for key in keys for (_, animals, _), run in groups[key].items()
                         if animals == largest[key]])
    figure, panels = new_figure(
        len(keys), CHART_TITLES[2],
        "The largest vocabulary, as training goes on. A line that has gone flat below 100 has "
        "stopped gaining from practice; a line that is still rising where it ends has not shown "
        "its limit.", legend)
    for number, (panel, key) in enumerate(zip(panels, keys)):
        ends, seen = [], []
        for width in widths:
            runs = [run for (w, animals, _), run in sorted(groups[key].items())
                    if w == width and animals == largest[key]]
            if not runs:
                continue
            color = width_color(width)
            for run in runs:
                xs = [point["sightings"] for point in run["curve"]]
                panel.plot(xs, [point["facts_known"] for point in run["curve"]], color=color,
                           linewidth=HAIR, alpha=0.45, zorder=2)
                draw_dot(panel, xs[-1], facts(run), color, how_ended(run))
                seen += xs
            panel.plot(*mean_curve(runs), color=color, linewidth=LINE, solid_capstyle="round",
                       solid_joinstyle="round", zorder=2.5)
            longest = max(runs, key=lambda run: run["sightings"])
            ends.append((longest["sightings"], facts(longest), 0, f"width {width}"))
        log_axis(panel, min(seen), max(seen))
        label_line_ends(panel, ends)
        panel_title(panel, f"{group_label(*key)}: {largest[key]:,} animal words")
        label_axes(panel, number, "times each fact was seen in training, on average (log scale)",
                   "facts known (0 to 100)")
    save_figure(figure, path)
    return True


def chart_knowledge(runs, path):
    """Chart 3, "Knowledge per parameter": one dot per run. x = bits of facts in the language
    for every parameter of the model (log scale), y = share of that knowledge the model holds."""
    # A language with nothing to know (one animal in use: 0 bits) has no place on a log scale.
    runs = [run for run in runs if bits_per_parameter(run) > 0]
    if not runs:
        return False
    by_seasons = {}
    for run in runs:
        by_seasons.setdefault(run["config"]["seasons"], []).append(run)
    styles = {seasons: SEASON_STYLES.get(seasons, (OTHER_WIDTH_COLOR, "D"))
              for seasons in by_seasons}
    legend = [[key_dot(styles[seasons][0], "finished", bits_note(of_season), styles[seasons][1])
               for seasons, of_season in sorted(by_seasons.items())],
              run_keys()] + fixed_key(runs)
    figure, (panel,) = new_figure(
        1, CHART_TITLES[3],
        "Each dot is one run. To the left the model has many parameters for the facts it should "
        "learn; to the right it has few. If only this load mattered, all the dots would lie "
        "along one curve, whatever their color.", legend, panel_titles=False)
    for run in runs:
        color, shape = styles[run["config"]["seasons"]]
        draw_dot(panel, bits_per_parameter(run), share_held(run), color, how_ended(run),
                 shape=shape)
    loads = [bits_per_parameter(run) for run in runs]
    log_axis(panel, min(loads), max(loads))
    panel.grid(axis="x", color=GRID, linewidth=HAIR, linestyle="-")
    label_axes(panel, 0, "bits of facts in the language per parameter of the model (log scale)",
               "share of the knowledge held (%), at least")
    save_figure(figure, path)
    return True


def chart_common_and_rare(groups, path):
    """Chart 4, "Common and rare words", for zipf usage only: the animals of the largest
    vocabulary grouped by rank of use, and the share of facts right in each group."""
    keys = [key for key in sorted(groups) if key[0] == "zipf"
            and max(animals for _, animals, _ in groups[key]) > SMALL]
    if not keys:
        return False
    largest = {key: max(animals for _, animals, _ in groups[key]) for key in keys}
    widths = sorted({w for key in keys for w, animals, _ in groups[key] if animals == largest[key]})
    legend = [[key_line(width_color(width), f"width {width}") for width in widths],
              run_keys() + [key_line(INK_MUTED, "line: the mean over seeds")],
              [key_line(INK_MUTED, "guessing: no animal known, always the most common value",
                        HAIR)]]
    legend += fixed_key([run for key in keys for (_, animals, _), run in groups[key].items()
                         if animals == largest[key]])
    figure, panels = new_figure(
        len(keys), CHART_TITLES[4],
        "Zipf usage: the animals with the lowest numbers are read about most often. A model can "
        "be poor on rare animals simply because it has hardly practiced them; that is not a "
        "limit of its size.", legend)
    for number, (panel, key) in enumerate(zip(panels, keys)):
        runs = {place: run for place, run in groups[key].items() if place[1] == largest[key]}
        seeds = sorted({seed for _, _, seed in runs})
        of_rank = rank_groups(largest[key])
        labels = [label for label, _, _ in of_rank]
        of_panel = sorted({width for width, _, _ in runs})
        # The limits come before the dots: dot_step measures the room on the finished axis.
        panel.set_xlim(-0.5, len(labels) - 0.5)
        step = dot_step(panel, range(len(labels)), len(of_panel), len(seeds))
        ends = []
        for width_place, width in enumerate(of_panel):
            points = []
            for (w, _, seed), run in runs.items():
                if w == width:
                    shares = right_by_rank_group(run)
                    points += [(place, shares[label], how_ended(run), seeds.index(seed))
                               for place, label in enumerate(labels)]
            end = draw_dots_and_means(panel, points, width_color(width), width_place,
                                      len(of_panel), len(seeds), step)
            ends.append((*end, f"width {width}"))
        # The guessing level: the mean over the seeds, because each seed has its own facts.
        levels = [guessing_levels(run["config"]) for run in runs.values()]
        guessing = [np.mean([level[label] for level in levels]) for label in labels]
        panel.plot(range(len(labels)), guessing, color=INK_MUTED, linewidth=HAIR, zorder=1.8)
        ends.append((len(labels) - 1, guessing[-1], 0, "guessing"))
        label_line_ends(panel, ends)
        panel.set_xticks(range(len(labels)))
        panel.set_xticklabels([f"{first:,}-{last:,}" for _, first, last in of_rank])
        panel_title(panel, f"{group_label(*key)}: {largest[key]:,} animal words")
        label_axes(panel, number, "animals by rank of use (1 = the most used)",
                   "facts right (%), not corrected for guessing")
    save_figure(figure, path)
    return True


def draw_charts(runs, groups, results_dir):
    """Draw every chart there are runs for. Returns the list of files written.

    A chart that cannot be drawn this time is removed if an older copy exists: a picture from
    other runs must not sit next to this table.
    """
    drawn = {
        1: chart_more_words(groups, results_dir / CHART_FILES[1]),
        2: chart_learning(groups, results_dir / CHART_FILES[2]),
        3: chart_knowledge(runs, results_dir / CHART_FILES[3]),
        4: chart_common_and_rare(groups, results_dir / CHART_FILES[4]),
    }
    for number, was_drawn in drawn.items():
        if not was_drawn:
            (results_dir / CHART_FILES[number]).unlink(missing_ok=True)
    return [results_dir / CHART_FILES[number] for number, was_drawn in drawn.items() if was_drawn]


# ------------------------------------------------------------------------------------------------
# Everything together
# ------------------------------------------------------------------------------------------------

NOT_DRAWN = {
    1: "chart 1 needs runs on at least two vocabulary sizes",
    2: "chart 2 needs at least one run",
    3: "chart 3 needs at least one run",
    4: "chart 4 needs zipf runs on more than 16 animals",
}


def report(runs_dir="runs", results_dir="results"):
    """Read the runs, print the table, write the CSV files and the charts, apply the rule.
    Returns the dictionary of rule_met, or None if there are no finished runs."""
    runs = load_runs(runs_dir)
    results_dir = Path(results_dir)
    if not runs:
        print(f"No finished runs in {runs_dir}/ (looked for {runs_dir}/*/summary.json).")
        print("Train one first, for example: .venv/bin/python train.py")
        # Nothing is written, and nothing is removed either: a mistyped --runs-dir must not
        # cost the tables and charts of the real runs. But they must not be taken for these.
        old = [name for name in ["summary.csv", "curves.csv", *CHART_FILES.values()]
               if (results_dir / name).exists()]
        if old:
            print(f"NOTE: {results_dir}/ still holds the files of an earlier report "
                  f"({', '.join(old)}). They were made from other runs, not from {runs_dir}/.")
        return None
    results_dir.mkdir(parents=True, exist_ok=True)

    groups, _ = comparison_groups(runs)
    print_table(runs)
    print()
    print_means(groups, cautions=[key for key in groups if recipe_problems(groups[key])])
    write_csv(results_dir / "summary.csv", [table_row(run) for run in runs])
    write_csv(results_dir / "curves.csv", curve_rows(runs))
    charts = draw_charts(runs, groups, results_dir)

    print()
    print("FILES WRITTEN")
    print(f"  {results_dir / 'summary.csv'}   the first table with more columns, one row per run")
    print(f"  {results_dir / 'curves.csv'}    every grading of every run (the learning curves)")
    for number, name in CHART_FILES.items():
        if results_dir / name in charts:
            print(f"  {results_dir / name}   {CHART_TITLES[number]}")
        else:
            print(f"  not drawn: {NOT_DRAWN[number]}")

    print()
    result = rule_met(runs)
    print_rule(result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Table, charts and the agreed decision rule, from the finished runs.")
    parser.add_argument("--runs-dir", default="runs",
                        help="where the runs are: one folder per run, each with a summary.json")
    parser.add_argument("--results-dir", default="results",
                        help="where the table (CSV) and the charts (PNG) are written")
    args = parser.parse_args()
    report(args.runs_dir, args.results_dir)


if __name__ == "__main__":
    main()

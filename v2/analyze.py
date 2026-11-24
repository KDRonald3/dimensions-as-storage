"""Reads the finished runs of the second experiment and builds everything the report needs.

    .venv/bin/python v2/analyze.py --runs-root runs_v2

It reads every stage folder that exists in RUNS_ROOT (main8, bits16, bits32, bits64 are the
counted stages; pilot and pilot2 are practice runs that are never counted). A stage that is not
complete yet is read as far as it goes, and what is missing is written down. It writes, into
v2/report/:

    facts.json       every score of every setting, the outcome of P1 to P4 and the verdict
    numbers.tex      one LaTeX macro per number the report text uses (the text never types a number)
    below95.tex      a table of the settings that ended below 95 facts known
    figures/*.png    the pictures (200 dots per inch)

The predictions and the verdict are computed exactly as Part A of v2/PLAN.md defines them.
Nothing here reads made-up data: v2/make_fake_runs.py is not used by this program.
"""

import argparse
import datetime
import json
import re
import statistics
import sys
from math import comb, factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------------------------------------
# The experiment's fixed numbers (from v2/PLAN.md)
# ------------------------------------------------------------------------------------------------
CREATURES = [16, 64, 256, 1024]
BODIES = [8, 16, 32, 64, 128]
USAGES = ["equal", "zipf"]
USAGE_TITLE = {"equal": "Every creature used equally often",
               "zipf": "Some creatures used far more often than others"}
USAGE_SHORT = {"equal": "Equal use", "zipf": "Unequal use"}
HELD = 95  # "holds" means this score or more
NEEDED = 4  # ... in at least this many of 5 repeats (P1 to P4 are defined on the 5 repeats of main8)
REPEATS = 5
GAP = 10  # points, in P2, P3 and the verdict
FACTS_GOAL = 97  # "facts known" counts as learned at this score (used for the speed picture)
QUESTIONS_GOAL = 90  # the same for questions about one creature
RISE = 1.0  # points gained since half-way for a score to count as "still rising"
COUNTED = ["main8", "bits16", "bits32", "bits64"]
PRACTICE = ["pilot", "pilot2"]
STAGES = COUNTED + PRACTICE
STAGE_BITS = {"main8": 8, "bits16": 16, "bits32": 32, "bits64": 64}
BITS_OF_KINDS = {4: 8, 8: 16, 16: 32, 32: 64}  # kinds of fact -> bits per creature (2 bits per fact)
BITS_LEVELS = [8, 16, 32, 64]
COLUMNS = 32  # numbers in every word's row

# The scores kept for every run: name in summary.json["final"], short name used in numbers.tex.
METRICS = [("facts", "facts"), ("questions_one", "q1"), ("questions_new", "qnew"),
           ("questions_seen", "qseen"), ("not_known", "notk")]

# ------------------------------------------------------------------------------------------------
# Look of the pictures: quiet, off-white, colour-blind-safe (Okabe and Ito palette)
# ------------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#6f6e69"
GRID = "#e1e0d9"
WIDTH_COLORS = {8: "#d55e00", 16: "#e69f00", 32: "#009e73", 64: "#0072b2", 128: "#cc79a7"}
CREATURE_COLORS = {16: "#56b4e9", 64: "#009e73", 256: "#e69f00", 1024: "#d55e00"}
LR_COLORS = {0.003: "#0072b2", 0.01: "#009e73", 0.03: "#d55e00"}
DPI = 200

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "legend.fontsize": 10.5, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "text.color": INK, "axes.labelcolor": INK_SOFT, "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


# ------------------------------------------------------------------------------------------------
# Reading the runs
# ------------------------------------------------------------------------------------------------

def load_stage(runs_root, stage):
    """Every readable summary.json of a stage: a list of dictionaries (empty if no such stage)."""
    folder = Path(runs_root) / stage
    summaries = []
    if folder.is_dir():
        for path in sorted(folder.glob("*/summary.json")):
            try:
                summaries.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                print(f"warning: could not read {path}")
    return summaries


def median(values):
    return statistics.median(values) if values else None


def first_step(curve, field, goal):
    """The first grading at which `field` reached `goal`, or None if it never did."""
    for point in curve:
        if point[field] >= goal:
            return point["step"]
    return None


def gain_since_halfway(curve, field="facts"):
    """Score at the last grading minus the score at the grading nearest to half of the last step.
    This is how "still rising" is told: a score that gained at least RISE points in the second
    half of training is still rising."""
    last = curve[-1]
    half = min(curve, key=lambda point: abs(point["step"] - last["step"] / 2))
    return last[field] - half[field]


def group_settings(stage, summaries):
    """Group the runs of a stage by setting (creatures, body, usage, kinds). One dictionary per
    setting with the final score of each repeat, and the median, lowest and highest."""
    groups = {}
    for summary in summaries:
        c = summary["config"]
        key = (c["creatures"], c["body"], c["usage"], c["kinds"], c["lr"])
        groups.setdefault(key, []).append(summary)
    settings = []
    for (creatures, body, usage, kinds, lr), runs in sorted(groups.items()):
        runs.sort(key=lambda run: run["config"]["seed"])
        setting = {"stage": stage, "creatures": creatures, "body": body, "usage": usage,
                   "kinds": kinds, "lr": lr, "bits_per_creature": BITS_OF_KINDS.get(kinds),
                   "repeats": len(runs), "seeds": [run["config"]["seed"] for run in runs],
                   "steps": runs[0]["steps"], "sentences": runs[0]["sentences"],
                   "parameters": runs[0]["parameters"],
                   "minutes": [run["minutes"] for run in runs]}
        for field, _ in METRICS:
            scores = [run["final"][field] for run in runs]
            setting[field] = {"scores": scores, "median": median(scores), "lowest": min(scores),
                              "highest": max(scores)}
        setting["questions_new_count"] = runs[0]["final"]["questions_new_detail"]["count"]
        setting["reached_95"] = sum(score >= HELD for score in setting["facts"]["scores"])
        setting["held"] = setting["reached_95"] >= NEEDED
        steps = [first_step(run["curve"], "facts", FACTS_GOAL) for run in runs]
        setting["steps_facts"] = steps
        setting["steps_questions"] = [first_step(run["curve"], "questions_one", QUESTIONS_GOAL)
                                      for run in runs]
        gains = [gain_since_halfway(run["curve"]) for run in runs]
        setting["gain_since_halfway"] = {"each": gains, "median": median(gains)}
        setting["still_rising"] = median(gains) >= RISE
        setting["curves"] = [{"seed": run["config"]["seed"], "points": run["curve"]} for run in runs]
        settings.append(setting)
    return settings


# ------------------------------------------------------------------------------------------------
# The predictions and the verdict, exactly as Part A of PLAN.md defines them
# ------------------------------------------------------------------------------------------------
# Every test below is answered with True (holds), False (does not hold) or None (cannot be told
# yet, because repeats are missing). A test holds when it is passed in at least 4 of the 5 repeats.

def decide(passed, found):
    if passed >= NEEDED:
        return True
    if passed + (REPEATS - found) < NEEDED:
        return False
    return None


def all_of(answers):
    if any(answer is False for answer in answers):
        return False
    if any(answer is None for answer in answers):
        return None
    return True


def result_text(answer):
    return {True: "holds", False: "does not hold", None: "is not decided yet"}[answer]


class Main8:
    """The scores of stage main8 looked up by (creatures, body, usage): {seed: facts known}."""

    def __init__(self, settings):
        self.facts = {}
        for s in settings:
            if s["stage"] == "main8":
                self.facts[(s["creatures"], s["body"], s["usage"])] = dict(
                    zip(s["seeds"], s["facts"]["scores"]))

    def scores(self, creatures, body, usage):
        return self.facts.get((creatures, body, usage), {})

    def reaches(self, creatures, body, usage):
        scores = self.scores(creatures, body, usage)
        passed = sum(score >= HELD for score in scores.values())
        return {"passed": passed, "found": len(scores), "answer": decide(passed, len(scores))}

    def paired(self, first, second, usage, test):
        """Compare two settings repeat by repeat (same seed). first and second are
        (creatures, body). test(a, b) is True when the repeat passes. Also returns the gaps."""
        a = self.scores(first[0], first[1], usage)
        b = self.scores(second[0], second[1], usage)
        seeds = sorted(set(a) & set(b))
        passed = sum(test(a[s], b[s]) for s in seeds)
        gaps = [a[s] - b[s] for s in seeds]
        return {"passed": passed, "found": len(seeds), "answer": decide(passed, len(seeds)),
                "gap_median": median(gaps)}


def predictions(main8, usage):
    """P1 to P4 and the verdict for one kind of word use (stage main8, trained word table)."""
    out = {}
    # P1: with 16 creatures every body width reaches 95 or more.
    p1 = {b: main8.reaches(16, b, usage) for b in BODIES}
    out["P1"] = {"answer": all_of([t["answer"] for t in p1.values()]),
                 "by_body": {str(b): t for b, t in p1.items()}}
    # P2: at width 8, 1,024 creatures score at least 10 below 16 creatures.
    p2 = main8.paired((16, 8), (1024, 8), usage, lambda small, large: small - large >= GAP)
    out["P2"] = {"answer": p2["answer"], **p2}
    # P3: on 1,024 creatures, width 128 scores at least 10 above width 8.
    p3 = main8.paired((1024, 128), (1024, 8), usage, lambda big, small: big - small >= GAP)
    out["P3"] = {"answer": p3["answer"], **p3}
    # Supported: P1, P2 and P3 hold. Contradicted: at every width, 1,024 creatures score within
    # 10 points of 16 creatures (in at least 4 repeats).
    supported = all_of([out["P1"]["answer"], out["P2"]["answer"], out["P3"]["answer"]])
    within = {b: main8.paired((1024, b), (16, b), usage, lambda large, small: abs(large - small) < GAP)
              for b in BODIES}
    contradicted = all_of([t["answer"] for t in within.values()])
    out["within_by_body"] = {str(b): t for b, t in within.items()}
    # P4 is the same test as "contradicted" (the plan words it as the opposite of P2).
    out["P4"] = {"answer": contradicted, "by_body": out["within_by_body"]}
    out["supported"], out["contradicted"] = supported, contradicted
    if supported is True:
        out["verdict"] = "supported"
    elif contradicted is True:
        out["verdict"] = "contradicted"
    elif supported is False and contradicted is False:
        out["verdict"] = "mixed"
    else:
        out["verdict"] = "not decided yet"
    return out


def overall_verdict(verdicts):
    values = list(verdicts.values())
    if all(v == "supported" for v in values):
        return "supported"
    if all(v == "contradicted" for v in values):
        return "contradicted"
    if any(v == "not decided yet" for v in values):
        return "not decided yet"
    return "mixed"


# ------------------------------------------------------------------------------------------------
# Facts about the language, counted from its definition (v2/PLAN.md B.1 to B.3)
# ------------------------------------------------------------------------------------------------

def valid_sentences(creatures, kinds):
    """How many different true sentences the language has, counting a list in a different order
    as a different sentence, and not yet taking away the sentences with a held-back pair in them.

    Per kind of fact and list length k (creatures with each value: N/4):
      statement  4 * C(N/4, k)          "not" statement  4 * C(3N/4, k)
      yes question  4 * C(N/4, k)       no question  4 * (C(N, k) - C(N/4, k))"""
    total = 0
    for k in range(1, 5):
        per_kind = (4 * comb(creatures // 4, k) + 4 * comb(3 * creatures // 4, k)
                    + 4 * comb(creatures // 4, k)
                    + 4 * (comb(creatures, k) - comb(creatures // 4, k)))
        total += factorial(k) * kinds * per_kind
    return total


def words_big(value):
    """A big count in words: 970 thousand, 23 trillion."""
    for size, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if value >= size:
            number = value / size
            return f"{number:.0f} {name}" if number >= 10 else f"{number:.1f} {name}"
    return f"{value:,}"


def held_pairs(creatures):
    """One tenth of all pairs, rounded down, at least 1 (B.3)."""
    return max(1, (creatures * (creatures - 1) // 2) // 10)


# ------------------------------------------------------------------------------------------------
# numbers.tex: one macro per number. In the report text write \N{key}; a missing key prints ??.
# ------------------------------------------------------------------------------------------------

class Numbers:
    def __init__(self):
        self.values = {}

    def put(self, key, value):
        self.values[key] = value

    def score(self, key, value):
        if value is None:
            self.put(key, "--")
            return
        text = f"{value:.1f}"
        text = text[:-2] if text.endswith(".0") else text
        if text in ("-0", "-0.0"):
            text = "0"
        self.put(key, text.replace("-", "\\textminus{}"))

    def whole(self, key, value):
        self.put(key, "--" if value is None else f"{value:,}")

    def write(self, path):
        lines = ["% Written by v2/analyze.py. Do not edit by hand.",
                 "\\newcommand{\\N}[1]{\\ifcsname num@#1\\endcsname\\csname num@#1\\endcsname"
                 "\\else\\textbf{??}\\fi}"]
        for key, value in sorted(self.values.items()):
            lines.append(f"\\expandafter\\gdef\\csname num@{key}\\endcsname{{{value}}}")
        Path(path).write_text("\n".join(lines) + "\n")


def tag_of(s):
    return f"{s['stage']}.c{s['creatures']}.b{s['body']}.{s['usage']}"


def fill_numbers(numbers, settings, expected, found, missing, preds, verdict, extras):
    counted = [s for s in settings if s["stage"] in COUNTED]
    for s in settings:
        if s["stage"] in PRACTICE:  # practice runs: one run per setting
            tag = f"{s['stage']}.k{s['kinds']}.c{s['creatures']}.b{s['body']}.lr{s['lr']:g}"
            numbers.score(f"{tag}.facts", s["facts"]["median"])
            numbers.score(f"{tag}.q1", s["questions_one"]["median"])
            numbers.score(f"{tag}.qnew", s["questions_new"]["median"])
            numbers.whole(f"{tag}.steps97", s["steps_facts"][0])
            numbers.whole(f"{tag}.steps90q", s["steps_questions"][0])
            numbers.whole(f"{tag}.steps", s["steps"])
            numbers.score(f"{tag}.gain", s["gain_since_halfway"]["median"])
            continue
        tag = tag_of(s)
        for field, short in METRICS:
            numbers.score(f"{tag}.{short}.med", s[field]["median"])
            numbers.score(f"{tag}.{short}.lo", s[field]["lowest"])
            numbers.score(f"{tag}.{short}.hi", s[field]["highest"])
        numbers.whole(f"{tag}.n95", s["reached_95"])
        numbers.whole(f"{tag}.repeats", s["repeats"])
        done = [x for x in s["steps_facts"] if x is not None]
        numbers.whole(f"{tag}.steps97.med", int(median(done)) if done else None)
        numbers.whole(f"{tag}.steps97.never", s["repeats"] - len(done))
        done = [x for x in s["steps_questions"] if x is not None]
        numbers.whole(f"{tag}.steps90q.med", int(median(done)) if done else None)
        numbers.whole(f"{tag}.steps90q.never", s["repeats"] - len(done))
        numbers.score(f"{tag}.gain", s["gain_since_halfway"]["median"])
        p = s["parameters"]
        numbers.whole(f"{tag}.params.trained", p["trained"])
        numbers.whole(f"{tag}.params.rest", p["trained_without_table"])
        numbers.whole(f"{tag}.params.table", p["table_numbers"])
        numbers.whole(f"{tag}.params.rows", p["table_rows_in_use"])
        numbers.whole(f"{tag}.params.middle", p["body_middle"])
    for stage in STAGES:
        numbers.whole(f"runs.{stage}", found[stage])
        numbers.whole(f"expected.{stage}", expected[stage])
        numbers.whole(f"missing.{stage}", len(missing[stage]))
        mine = [s for s in settings if s["stage"] == stage]
        if mine:
            numbers.whole(f"steps.{stage}", mine[0]["steps"])
            numbers.whole(f"sentences.{stage}", mine[0]["sentences"])
            numbers.put(f"lr.{stage}", f"{mine[0]['lr']:g}")
            numbers.whole(f"hours.{stage}", round(sum(sum(s["minutes"]) for s in mine) / 60))
    for stage in COUNTED:
        mine = [s for s in counted if s["stage"] == stage]
        if not mine:
            continue
        numbers.whole(f"settings.{stage}", len(mine))
        below = [s for s in mine if s["facts"]["median"] < HELD]
        numbers.whole(f"below95.{stage}", len(below))
        numbers.whole(f"below95.rising.{stage}", sum(s["still_rising"] for s in below))
        numbers.whole(f"below95.flat.{stage}", sum(not s["still_rising"] for s in below))
        low = min(mine, key=lambda s: s["facts"]["median"])
        high = max(mine, key=lambda s: s["facts"]["median"])
        numbers.score(f"{stage}.facts.lowest", low["facts"]["median"])
        numbers.score(f"{stage}.facts.highest", high["facts"]["median"])
        numbers.put(f"{stage}.facts.lowest.where",
                    f"{low['creatures']:,} creatures, body width {low['body']}, "
                    f"{USAGE_SHORT[low['usage']].lower()}")
        for usage in USAGES:
            part = [s for s in mine if s["usage"] == usage]
            numbers.score(f"{stage}.{usage}.facts.lowest", min(s["facts"]["median"] for s in part))
            numbers.score(f"{stage}.{usage}.facts.highest", max(s["facts"]["median"] for s in part))
            numbers.whole(f"below95.{stage}.{usage}",
                          sum(s["facts"]["median"] < HELD for s in part))
            numbers.whole(f"settings.{stage}.{usage}", len(part))
        for field, short in METRICS[1:]:
            numbers.score(f"{stage}.{short}.lowest", min(s[field]["median"] for s in mine))
            numbers.score(f"{stage}.{short}.highest", max(s[field]["median"] for s in mine))
        # runs of this stage that ended below 95 facts known, and the share still rising
        runs_below = sum(sum(x < HELD for x in s["facts"]["scores"]) for s in mine)
        numbers.whole(f"runs.below95.{stage}", runs_below)
    # who ran where
    for key, value in extras.items():
        numbers.put(key, value)
    # the predictions
    for usage, p in preds.items():
        for name in ("P1", "P2", "P3", "P4"):
            numbers.put(f"{name.lower()}.{usage}.result", result_text(p[name]["answer"]))
        numbers.put(f"verdict.{usage}", p["verdict"])
        numbers.whole(f"p2.{usage}.passed", p["P2"]["passed"])
        numbers.whole(f"p3.{usage}.passed", p["P3"]["passed"])
        numbers.score(f"p2.{usage}.gap", p["P2"]["gap_median"])
        numbers.score(f"p3.{usage}.gap", p["P3"]["gap_median"])
        for b, t in p["P1"]["by_body"].items():
            numbers.whole(f"p1.{usage}.b{b}.passed", t["passed"])
        for b, t in p["within_by_body"].items():
            numbers.whole(f"within.{usage}.b{b}.passed", t["passed"])
    numbers.put("verdict.overall", verdict)
    # the rules and thresholds, so that the text never types them
    for key, value in (("rule.held", HELD), ("rule.needed", NEEDED), ("rule.repeats", REPEATS),
                       ("rule.gap", GAP), ("rule.facts_goal", FACTS_GOAL),
                       ("rule.questions_goal", QUESTIONS_GOAL), ("rule.columns", COLUMNS)):
        numbers.whole(key, value)
    numbers.score("rule.rise", RISE)
    # the language
    for kinds in (4, 8, 16, 32):
        for creatures in CREATURES:
            numbers.put(f"lang.sentences.k{kinds}.c{creatures}",
                        words_big(valid_sentences(creatures, kinds)))
    for creatures in CREATURES:
        numbers.whole(f"lang.held.c{creatures}", held_pairs(creatures))
        numbers.whole(f"lang.pairs.c{creatures}", creatures * (creatures - 1) // 2)
    harmonic = sum(1 / i for i in range(1, 1025))  # unequal use: the rarest of 1,024 against the average
    numbers.put("zipf.rarest.c1024", f"{harmonic:.1f}")
    # how many exam questions about new combinations each size of vocabulary had
    for s in counted:
        if s["stage"] == "main8" and s["body"] == 8 and s["usage"] == "equal":
            numbers.whole(f"qnew.count.c{s['creatures']}", s["questions_new_count"])


def gap_numbers(numbers, settings):
    """The size of each effect, for the test of P4's last sentence (the agent's expectation)."""
    table = {(s["stage"], s["creatures"], s["body"], s["usage"]): s for s in settings
             if s["stage"] in COUNTED}

    def med(stage, creatures, body, usage):
        s = table.get((stage, creatures, body, usage))
        return s["facts"]["median"] if s else None

    for stage in COUNTED:
        for usage in USAGES:
            drops = []
            for body in BODIES:
                a, b = med(stage, 16, body, usage), med(stage, 1024, body, usage)
                if a is not None and b is not None:
                    numbers.score(f"gap.creatures.{stage}.{usage}.b{body}", a - b)
                    drops.append(a - b)
                a, b = med(stage, 1024, 128, usage), med(stage, 1024, body, usage)
            numbers.whole(f"gap.creatures.big.{stage}.{usage}", sum(d >= GAP for d in drops))
            numbers.whole(f"gap.creatures.n.{stage}.{usage}", len(drops))
            for creatures in CREATURES:
                a, b = med(stage, creatures, 128, usage), med(stage, creatures, 8, usage)
                if a is not None and b is not None:
                    numbers.score(f"gap.body.{stage}.{usage}.c{creatures}", a - b)
            # at the most creatures: body 128 against body 8
    # new combinations against combinations that were read (S3 against S4), over all counted settings
    diffs = [s["questions_new"]["median"] - s["questions_seen"]["median"] for s in settings
             if s["stage"] in COUNTED]
    # A folder may lack some runs (for example no 64-bit runs): a number with nothing to work
    # from is then left out instead of crashing.
    if diffs:
        numbers.score("qdiff.median", median(diffs))
        numbers.score("qdiff.lowest", min(diffs))
        numbers.score("qdiff.highest", max(diffs))
    small = [s["questions_new"]["median"] - s["questions_seen"]["median"] for s in settings
             if s["stage"] == "bits64" and s["creatures"] == 16]
    if small:
        numbers.score("qdiff.bits64.c16.lowest", min(small))
    others = [d for s, d in zip([s for s in settings if s["stage"] in COUNTED], diffs)
              if not (s["stage"] == "bits64" and s["creatures"] == 16)]
    if others:
        numbers.score("qdiff.others.lowest", min(others))
    not_known_c16 = [s["not_known"]["median"] for s in settings
                     if s["stage"] in COUNTED and s["creatures"] == 16]
    if not_known_c16:
        numbers.score("notk.c16.lowest", min(not_known_c16))
    for usage in USAGES:
        for creatures in (16, 256, 1024):
            for body in (8, 32, 128):
                a, b = med("main8", creatures, body, usage), med("bits64", creatures, body, usage)
                if a is not None and b is not None:
                    numbers.score(f"gap.bits.{usage}.c{creatures}.b{body}", a - b)


# ------------------------------------------------------------------------------------------------
# Table of the settings that ended below 95
# ------------------------------------------------------------------------------------------------

def below95_table(settings):
    """LaTeX for the settings whose median facts known ended below 95, one table per stage."""
    out, rows_total = [], {}
    for stage in COUNTED:
        mine = [s for s in settings if s["stage"] == stage and s["facts"]["median"] < HELD]
        rows_total[stage] = len(mine)
        if not mine:
            continue
        mine.sort(key=lambda s: (s["usage"], s["creatures"], s["body"]))
        out.append(f"\\paragraph{{{STAGE_BITS[stage]} bits per creature.}} "
                   f"{len(mine)} settings ended with a median below {HELD}.\n")
        out.append("\\begin{center}\\small\n\\begin{tabular}{@{}llrrrrl@{}}\n\\toprule\n"
                   "Word use & Creatures & Body width & Median & Lowest to highest & "
                   "Gain in the second half & Still rising?\\\\\n\\midrule")
        for s in mine:
            rise = "yes" if s["still_rising"] else "no"
            out.append(f"{USAGE_SHORT[s['usage']]} & {s['creatures']:,} & {s['body']} & "
                       f"{s['facts']['median']:.1f} & {s['facts']['lowest']:.1f} to "
                       f"{s['facts']['highest']:.1f} & {s['gain_since_halfway']['median']:+.1f} & "
                       f"{rise}\\\\")
        out.append("\\bottomrule\n\\end{tabular}\n\\end{center}\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------
# Pictures
# ------------------------------------------------------------------------------------------------

def thousands(value, _=None):
    return f"{value:,.0f}"


def sentences_label(value, _=None):
    if value >= 1e6:
        return f"{value / 1e6:g} million"
    if value >= 1e3:
        return f"{value / 1e3:g} thousand"
    return f"{value:g}"


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def quiet(ax, y_range=(-3, 103), ticks=(0, 25, 50, 75, 100)):
    """The look of every panel: no box, thin grid lines."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if y_range:
        ax.set_ylim(*y_range)
        ax.set_yticks(list(ticks))


def spread_labels(wanted, gap, low=2.0, high=100.0):
    """Keep the heights of line-end labels at least `gap` apart and between low and high."""
    order = sorted(range(len(wanted)), key=lambda i: wanted[i])
    placed = [wanted[i] for i in order]
    for _ in range(200):
        moved = False
        for k in range(1, len(placed)):
            if placed[k] - placed[k - 1] < gap - 1e-9:
                push = (gap - (placed[k] - placed[k - 1])) / 2
                placed[k - 1] -= push
                placed[k] += push
                moved = True
        if not moved:
            break
    placed[0] = max(placed[0], low)
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    if placed[-1] > high:
        placed[-1] = high
        for k in range(len(placed) - 2, -1, -1):
            placed[k] = min(placed[k], placed[k + 1] - gap)
    result = [0.0] * len(wanted)
    for slot, i in enumerate(order):
        result[i] = placed[slot]
    return result


def by_setting(settings, stage="main8"):
    return {(s["creatures"], s["body"], s["usage"]): s for s in settings if s["stage"] == stage}


def body_legend_handles(extra=True):
    handles = [Line2D([], [], color=WIDTH_COLORS[b], linewidth=2.4, label=str(b)) for b in BODIES]
    if extra:
        handles += [Line2D([], [], linestyle="none", marker="o", markersize=4, color=INK_SOFT,
                           label="one repeat"),
                    Line2D([], [], color=INK_SOFT, linewidth=1.0, alpha=0.6,
                           label="lowest to highest repeat")]
    return handles


def legend_on_top(figure, handles, title, ncol=None):
    figure.legend(handles=handles, loc="outside upper center", ncol=ncol or len(handles),
                  frameon=False, title=title, title_fontsize=10.5, columnspacing=1.3,
                  handlelength=1.6)


def draw_body_lines(ax, table, usage, field, label_gap=7.5, label_size=10):
    """Score against number of creatures (evenly spaced), one line per body width through the
    medians, one dot per repeat, and a thin bar from the lowest to the highest repeat."""
    ends = []
    for index, body in enumerate(BODIES):
        shift = (index - 2) * 0.05
        color = WIDTH_COLORS[body]
        xs, meds = [], []
        for place, creatures in enumerate(CREATURES):
            s = table.get((creatures, body, usage))
            if not s:
                continue
            x = place + shift
            data = s[field]
            ax.plot([x, x], [data["lowest"], data["highest"]], color=color, linewidth=1.0,
                    alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.plot([x] * len(data["scores"]), data["scores"], linestyle="none", marker="o",
                    markersize=3.6, color=color, alpha=0.85, markeredgewidth=0, zorder=3)
            xs.append(x)
            meds.append(data["median"])
        if xs:
            ax.plot(xs, meds, color=color, linewidth=2.0, zorder=2.5, solid_capstyle="round")
            ends.append((body, xs[-1], meds[-1]))
    if ends and label_size:
        heights = spread_labels([m for _, _, m in ends], label_gap)
        for (body, x, _), height in zip(ends, heights):
            ax.text(x + 0.12, height, str(body), color=WIDTH_COLORS[body], fontsize=label_size,
                    fontweight="bold", va="center", ha="left")
    ax.set_xlim(-0.35, len(CREATURES) - 1 + 0.6)
    ax.set_xticks(range(len(CREATURES)))
    ax.set_xticklabels([f"{c:,}" for c in CREATURES])


def figure_main(settings, path):
    """Figure 1: facts known against number of creatures at 8 bits, both kinds of word use."""
    table = by_setting(settings, "main8")
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 4.4), layout="constrained", sharey=True)
    for ax, usage in zip(axes, USAGES):
        draw_body_lines(ax, table, usage, "facts", label_size=0)
        quiet(ax)
        ax.axhline(HELD, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1.5)
        ax.axhline(25, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 3)), zorder=1.5)
        ax.text(-0.3, HELD - 1.5, "95: counted as held", fontsize=9.5, color=INK_SOFT, va="top")
        ax.text(-0.3, 25 + 1.2, "25: pure guessing", fontsize=9.5, color=INK_SOFT, va="bottom")
        ax.set_title(USAGE_TITLE[usage], loc="left", wrap=True)
        ax.set_xlabel("number of creatures")
    axes[0].set_ylabel("facts known (0 to 100)")
    legend_on_top(figure, body_legend_handles(),
                  "Neurons in the middle part of each layer (body width)")
    save(figure, path)


def figure_speed(settings, path):
    """Figure 2: training steps until the facts were known (97), and until the questions about
    one creature were answered (90), by number of creatures. One line per body width."""
    table = by_setting(settings, "main8")
    rows = [("steps_facts", f"Steps until facts known first reached {FACTS_GOAL}"),
            ("steps_questions", f"Steps until questions about one creature first reached {QUESTIONS_GOAL}")]
    figure, axes = plt.subplots(2, 2, figsize=(7.8, 7.0), layout="constrained", sharey="row")
    for row, (field, title) in enumerate(rows):
        for column, usage in enumerate(USAGES):
            ax = axes[row][column]
            ends = []
            for index, body in enumerate(BODIES):
                shift = (index - 2) * 0.05
                xs, meds = [], []
                for place, creatures in enumerate(CREATURES):
                    s = table.get((creatures, body, usage))
                    if not s:
                        continue
                    values = [v for v in s[field] if v is not None]
                    if not values:
                        continue
                    ax.plot([place + shift] * len(values), values, linestyle="none", marker="o",
                            markersize=3.4, color=WIDTH_COLORS[body], alpha=0.8, markeredgewidth=0)
                    xs.append(place + shift)
                    meds.append(median(values))
                if xs:
                    ax.plot(xs, meds, color=WIDTH_COLORS[body], linewidth=2.0)
                    ends.append((body, xs[-1], meds[-1]))
            if ends:
                from math import log10
                heights = spread_labels([log10(v) for _, _, v in ends], 0.075, low=-9, high=9)
                for (body, x, _), height in zip(ends, heights):
                    ax.text(x + 0.12, 10 ** height, str(body), color=WIDTH_COLORS[body],
                            fontsize=9.5, fontweight="bold", va="center")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(thousands))
            ax.yaxis.set_minor_locator(NullLocator())
            quiet(ax, y_range=None)
            ax.set_xlim(-0.35, len(CREATURES) - 1 + 0.6)
            ax.set_xticks(range(len(CREATURES)))
            ax.set_xticklabels([f"{c:,}" for c in CREATURES])
            ax.set_title(f"{USAGE_SHORT[usage]}", loc="left")
            if row == 1:
                ax.set_xlabel("number of creatures")
        axes[row][0].set_ylabel(title.replace("until", "until\n", 1) + "\n(log scale)", fontsize=9.5)
    handles = body_legend_handles(extra=False) + [
        Line2D([], [], linestyle="none", marker="o", markersize=4, color=INK_SOFT, label="one repeat")]
    legend_on_top(figure, handles, "Body width (neurons in the middle part of each layer); line = median")
    save(figure, path)


def figure_questions(settings, path):
    """Figure 3: the three other scores at 8 bits; rows are the kinds of word use."""
    table = by_setting(settings, "main8")
    columns = [("questions_one", "Questions: one creature"),
               ("questions_new", "Questions: new combinations"),
               ("not_known", "\u201cnot\u201d known")]
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 6.4), layout="constrained", sharey=True)
    for row, usage in enumerate(USAGES):
        for column, (field, title) in enumerate(columns):
            ax = axes[row][column]
            draw_body_lines(ax, table, usage, field, label_size=0)
            quiet(ax)
            chance = 25 if field == "not_known" else 0
            ax.axhline(chance, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 3)), zorder=1.5)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=10.5)
            if row == 1:
                ax.set_xlabel("number of creatures")
            if column == 0:
                ax.set_ylabel(f"{USAGE_SHORT[usage]}\nscore (0 to 100)")
    legend_on_top(figure, body_legend_handles(), "Body width (neurons in the middle part of each layer)")
    save(figure, path)


def figure_parameters(settings, path):
    """Figure 4: trained numbers by number of creatures, split into the word table and the rest."""
    table = by_setting(settings, "main8")
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), layout="constrained", sharey=True)
    for ax, body in zip(axes, (8, 128)):
        for place, creatures in enumerate(CREATURES):
            s = table.get((creatures, body, "equal"))
            if not s:
                continue
            p = s["parameters"]
            ax.bar(place, p["trained_without_table"], color="#0072b2", width=0.62)
            ax.bar(place, p["table_numbers"], bottom=p["trained_without_table"], color="#e69f00",
                   width=0.62)
            ax.text(place, p["trained"] + 1200, f"{p['trained']:,}", ha="center", va="bottom",
                    fontsize=9.5, color=INK)
        ax.set_xticks(range(len(CREATURES)))
        ax.set_xticklabels([f"{c:,}" for c in CREATURES])
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        quiet(ax, y_range=None)
        ax.set_ylim(0, 74000)
        ax.set_xlabel("number of creatures")
        ax.set_title(f"Body width {body}", loc="left")
    axes[0].set_ylabel("trained numbers in the model")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#e69f00", label="word table (a row of 32 numbers for every word in use)"),
               plt.Rectangle((0, 0), 1, 1, color="#0072b2", label="everything else (the layers and the output)")]
    figure.legend(handles=handles, loc="outside upper center", ncol=1, frameon=False)
    save(figure, path)


def bits_axis(ax, label=True):
    ax.set_xscale("log", base=2)
    ax.set_xlim(7, 90)
    ax.xaxis.set_major_locator(FixedLocator(BITS_LEVELS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\n({v / COLUMNS:g})"))
    ax.xaxis.set_minor_locator(NullLocator())


def bits_figure(settings, path, fields, rows_by_usage, title_fields, height, y_label):
    """Figures 5 and 6: score against bits per creature. One line per number of creatures
    (median of the repeats), one dot per repeat. Columns are the body widths 8, 32 and 128."""
    bodies = (8, 32, 128)
    rows = [(usage, field) for field in fields for usage in USAGES]
    figure, axes = plt.subplots(len(rows), len(bodies), figsize=(8.0, height),
                                layout="constrained", sharex=True, sharey=True, squeeze=False)
    figure.get_layout_engine().set(h_pad=0.12, w_pad=0.1)
    # stage -> bits; main8 is the 8-bit stage
    for r, (usage, field) in enumerate(rows):
        for c, body in enumerate(bodies):
            ax = axes[r][c]
            ends = []
            for creatures in CREATURES:
                xs, ys = [], []
                for s in settings:
                    if (s["stage"] in COUNTED and s["creatures"] == creatures and s["body"] == body
                            and s["usage"] == usage):
                        bits = STAGE_BITS[s["stage"]]
                        scores = s[field]["scores"]
                        ax.plot([bits] * len(scores), scores, linestyle="none", marker="o",
                                markersize=3.2, color=CREATURE_COLORS[creatures], alpha=0.55,
                                markeredgewidth=0, zorder=3)
                        xs.append(bits)
                        ys.append(s[field]["median"])
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                xs, ys = [xs[i] for i in order], [ys[i] for i in order]
                if xs:
                    ax.plot(xs, ys, color=CREATURE_COLORS[creatures], linewidth=2.0, zorder=2.5)
                    ends.append((creatures, xs[-1], ys[-1]))
            bits_axis(ax)
            if field == "not_known":
                quiet(ax, y_range=(15, 103), ticks=(25, 50, 75, 100))
                ax.axhline(25, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 3)), zorder=1.5)
            else:
                quiet(ax, y_range=(48, 103), ticks=(50, 75, 100))
            if field == "facts":
                ax.axhline(HELD, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1.5)
            if len(fields) == 1:
                if r == 0:
                    ax.set_title(f"Body width {body}", loc="left")
                if c == 0:
                    ax.set_ylabel(f"{USAGE_SHORT[usage]}\n{title_fields[field]}", fontsize=10)
            else:
                ax.set_title(f"{title_fields[field]}\n{USAGE_SHORT[usage].lower()}, body width {body}",
                             loc="left", fontsize=10)
                if c == 0:
                    ax.set_ylabel("score (0 to 100)")
            if r == len(rows) - 1:
                ax.set_xlabel("bits per creature\n(in brackets: bits per number)", fontsize=9)
    handles = [Line2D([], [], color=CREATURE_COLORS[c], linewidth=2.4, label=f"{c:,}")
               for c in CREATURES]
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=4, color=INK_SOFT,
                          alpha=0.6, label="one repeat"))
    legend_on_top(figure, handles, "Number of creatures (line = median of the repeats)")
    save(figure, path)


def figure_bits_facts(settings, path):
    bits_figure(settings, path, ["facts"], None, {"facts": "facts known"}, 4.9, "")


def figure_bits_notknown(settings, path):
    bits_figure(settings, path, ["not_known"], None, {"not_known": "\u201cnot\u201d known"}, 4.9, "")


def figure_bits_questions(settings, path):
    bits_figure(settings, path, ["questions_one", "questions_new"], None,
                {"questions_one": "Questions about one creature", "questions_new": "Questions about new combinations"},
                7.6, "")


def figure_curves64(settings, path):
    """Figure 7: 64 bits, facts known against sentences read, every run its own line."""
    table = {(s["creatures"], s["body"], s["usage"]): s for s in settings if s["stage"] == "bits64"}
    creatures_shown = (16, 256, 1024)
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 6.2), layout="constrained", sharey=True,
                                sharex=True)
    longest = max([p["sentences"] for s in settings if s["stage"] == "bits64"
                   for curve in s["curves"] for p in curve["points"]] or [1])
    for row, usage in enumerate(USAGES):
        for column, creatures in enumerate(creatures_shown):
            ax = axes[row][column]
            ends = []
            for body in (8, 32, 128):
                s = table.get((creatures, body, usage))
                if not s:
                    continue
                finals = []
                for curve in s["curves"]:
                    xs = [p["sentences"] for p in curve["points"]]
                    ys = [p["facts"] for p in curve["points"]]
                    ax.plot(xs, ys, color=WIDTH_COLORS[body], linewidth=1.7, alpha=0.9)
                    finals.append(ys[-1])
                ends.append((body, xs[-1], sum(finals) / len(finals)))
            if ends:
                heights = spread_labels([y for _, _, y in ends], 7)
                for (body, x, _), height in zip(ends, heights):
                    ax.text(x * 1.12, height, str(body), color=WIDTH_COLORS[body], fontsize=9,
                            fontweight="bold", va="center")
            ax.set_xscale("log")
            ax.set_xlim(4e4, longest * 2.6)
            ax.xaxis.set_major_locator(FixedLocator([1e6, 1e8]))
            ax.xaxis.set_major_formatter(FuncFormatter(sentences_label))
            ax.xaxis.set_minor_locator(NullLocator())
            quiet(ax)
            ax.axhline(HELD, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1.5)
            ax.set_title(f"{creatures:,} creatures, {USAGE_SHORT[usage].lower()}", loc="left",
                         fontsize=10.5)
            if row == 1:
                ax.set_xlabel("lines of text read (log scale)")
            if column == 0:
                ax.set_ylabel("facts known (0 to 100)")
    legend_on_top(figure, body_legend_handles(extra=False)[:0] + [
        Line2D([], [], color=WIDTH_COLORS[b], linewidth=2.4, label=str(b)) for b in (8, 32, 128)],
        "Body width (one line for each of the 2 repeats)")
    save(figure, path)


def figure_practice(settings, path):
    """Figure 8: the practice runs. Final facts known and questions about one creature for each
    learning rate, one panel for each number of bits per creature."""
    practice = [s for s in settings if s["stage"] in PRACTICE]
    kinds_levels = [4, 8, 16, 32]
    figure, axes = plt.subplots(2, 4, figsize=(9.2, 5.6), layout="constrained", sharey=True)
    for column, kinds in enumerate(kinds_levels):
        mine = [s for s in practice if s["kinds"] == kinds]
        places = sorted({(s["creatures"], s["body"]) for s in mine})
        for row, (field, name) in enumerate((("facts", "facts known"),
                                              ("questions_one", "questions, one creature"))):
            ax = axes[row][column]
            for s in mine:
                x = places.index((s["creatures"], s["body"]))
                lr = s["lr"]
                shift = {0.003: -0.18, 0.01: 0.0, 0.03: 0.18}[lr]
                ax.plot([x + shift], [s[field]["median"]], marker="o", markersize=7,
                        color=LR_COLORS[lr], linestyle="none", markeredgewidth=0)
            ax.set_xticks(range(len(places)))
            ax.set_xticklabels([f"{c:,}\n{b}" for c, b in places], fontsize=8.5)
            ax.set_xlim(-0.6, len(places) - 0.4)
            quiet(ax)
            if row == 0:
                ax.set_title(f"{BITS_OF_KINDS[kinds]} bits per creature", loc="left", fontsize=10.5)
            if column == 0:
                ax.set_ylabel(f"{name}\n(0 to 100)")
            if row == 1:
                ax.set_xlabel("creatures (top)\nbody width (bottom)", fontsize=9)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7, color=LR_COLORS[lr],
                      label=f"{lr:g}") for lr in sorted(LR_COLORS)]
    legend_on_top(figure, handles, "Learning rate (how big each step of learning is)")
    save(figure, path)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def meter_reading(root):
    """Modal's own cost meter: the last reading in v2/report/fetch_output.txt, else cost.log."""
    reading = {}
    fetch = HERE / "report" / "fetch_output.txt"
    if fetch.exists():
        match = re.findall(r"Metered Cost[^$0-9]*\$?\s*([0-9]+\.[0-9]+)", fetch.read_text())
        if match:
            reading["cost.meter"] = match[-1]
            reading["cost.meter.source"] = "billing summary after the last download"
    log = ROOT / "logs" / "v2" / "cost.log"
    if log.exists():
        lines = [l for l in log.read_text().splitlines() if "metered cost" in l]
        if lines:
            first = re.search(r"metered cost: ([0-9.]+)", lines[0])
            last = re.search(r"metered cost: ([0-9.]+)", lines[-1])
            reading["cost.log.last"] = last.group(1)
            reading["cost.log.first"] = first.group(1)
            reading["cost.log.last.time"] = lines[-1][:16]
            reading["cost.log.first.time"] = lines[0][:16]
            reading.setdefault("cost.meter", last.group(1))
            reading.setdefault("cost.meter.source", "the cost log")
    return reading


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", required=True, help="folder holding main8/, bits16/, ...")
    parser.add_argument("--report-dir", default=str(HERE / "report"),
                        help="where facts.json, numbers.tex and figures/ are written")
    args = parser.parse_args()
    root = Path(args.runs_root)
    report = Path(args.report_dir)
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    from v2.grid import grid  # the names of the runs that every stage should have

    settings, found, expected, missing = [], {}, {}, {}
    for stage in STAGES:
        summaries = load_stage(root, stage)
        found[stage] = len(summaries)
        names = {c["name"] for c in grid(stage)}
        expected[stage] = len(names)
        done = {s["name"] for s in summaries}
        missing[stage] = sorted(names - done)
        settings += group_settings(stage, summaries)
    if not found["main8"]:
        raise SystemExit(f"no runs found in {root / 'main8'}")

    main8 = Main8(settings)
    preds = {usage: predictions(main8, usage) for usage in USAGES}
    verdict = overall_verdict({usage: p["verdict"] for usage, p in preds.items()})

    extras = meter_reading(root)
    local = ROOT / "logs" / "v2" / "local"
    local_names = sorted(p.stem for p in local.glob("*.log")) if local.is_dir() else []
    for stage in ("bits16", "bits32"):
        extras[f"local.{stage}"] = str(sum(n.startswith(stage + "_") for n in local_names))
    for stage in COUNTED:
        log = ROOT / "logs" / "v2" / f"{stage}.log"
        resumed = set()
        if log.exists():
            clean = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
            resumed = set(re.findall(r"^(\S+): going on from the checkpoint", clean, re.M))
        extras[f"resumed.{stage}"] = str(len(resumed))
    extras["local.pilot2"] = str(sum(n.startswith("pilot2") for n in local_names))
    extras["local.total"] = str(sum(extras[f"local.{stage}"] != "" and int(extras[f"local.{stage}"])
                                    for stage in ("bits16", "bits32")))
    extras["local.total"] = str(int(extras["local.bits16"]) + int(extras["local.bits32"]))

    facts = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
             "runs_root": str(root), "runs_found": found, "runs_expected": expected,
             "runs_missing": missing,
             "rules": {"held_score": HELD, "repeats_needed": NEEDED, "repeats": REPEATS,
                       "gap": GAP, "facts_goal": FACTS_GOAL, "questions_goal": QUESTIONS_GOAL,
                       "rise_points": RISE},
             "settings": [{k: v for k, v in s.items() if k not in ("curves",)} for s in settings],
             "predictions": preds, "verdict": verdict}
    (report / "facts.json").write_text(json.dumps(facts, indent=1))

    numbers = Numbers()
    fill_numbers(numbers, settings, expected, found, missing, preds, verdict, extras)
    gap_numbers(numbers, settings)
    numbers.write(report / "numbers.tex")
    (report / "below95.tex").write_text(below95_table(settings))

    figure_main(settings, figures / "fig1_main.png")
    figure_speed(settings, figures / "fig2_speed.png")
    figure_questions(settings, figures / "fig3_questions.png")
    figure_parameters(settings, figures / "fig4_parameters.png")
    if any(s["stage"] in ("bits16", "bits32", "bits64") for s in settings):
        figure_bits_facts(settings, figures / "fig5_bits_facts.png")
        figure_bits_questions(settings, figures / "fig6_bits_questions.png")
        figure_bits_notknown(settings, figures / "fig9_bits_notknown.png")
    if any(s["stage"] == "bits64" for s in settings):
        figure_curves64(settings, figures / "fig7_curves64.png")
    figure_practice(settings, figures / "fig8_practice.png")
    print(f"read {found} (expected {expected}); verdict: {verdict}")
    print(f"missing: { {k: len(v) for k, v in missing.items() if v} }")
    print(f"wrote {report / 'facts.json'}, {report / 'numbers.tex'} and the pictures in {figures}")


if __name__ == "__main__":
    main()

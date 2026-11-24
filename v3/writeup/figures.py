"""Figures and numbers for the report on the third experiment (23 November 2026).

    .venv/bin/python v3/writeup/figures.py
    cd v3/writeup && ~/.local/share/tectonic-0.17.0/tectonic report.tex

Reads every counted run in runs_v3/counted and every run of the rate check in
runs_v3/rate_check_third (and, for single checks, the practice runs, the second experiment's
runs and the launcher's log) and writes

    v3/writeup/figures/*.pdf      the eighteen figures of the report, in its order
    v3/writeup/numbers.tex        every number the text uses, as LaTeX macros

so that no result in the report is typed by hand and text and figures cannot disagree. The
sentences of the report that state a fact without giving a number are tested where their numbers
are made and in check_the_claims(): if the data stopped supporting one of them, this script
would stop with an error.

This script only reads the rest of the repository. It writes inside v3/writeup/ and nowhere else.

The rate check (the best repeat of every setting again at a third of the learning rate) is part
of the report: Figure 14 and finding 8. The further stage that the plan allowed, at a tenth of
the rate, was not run.


THE VISUAL SYSTEM (one for the whole report; decided once, here)
----------------------------------------------------------------
Table columns         one teal, light to dark: 8, 16, 32, 64. More columns = darker. This is the
                      new dial of this experiment and teal has no other job. In a key the four
                      teals are plain short bars; a line is also named where it ends. Where the
                      number of columns is itself the horizontal position (Figure 7), lines are ink.
Number of creatures   never a colour. Always a panel or a block.
Word use              equal use = solid line; unequal use = dashed line (as in the second
                      report). A use has a row, a panel or a half of its own, named in words;
                      only in Figure 7 do both share a panel, and its key shows the two lines.
Trained past the      orange, and nothing else is orange: a run that was carried on after the rule
rule's stop           had ended it. (In the second report orange was the "longer runs": the same idea.)
Facts missed          grey tiles, pale to dark: a tile at 100 is pale, a tile at 25 (only guessing)
                      is dark. The number stands on every tile, dark on a light tile and white on a
                      dark one (white below 77, where the two are equally easy to read).
Body width            never a colour. Always a position along the horizontal, or a row.
Bits per creature     never a colour. Always a position along the horizontal, or a panel.
Repeats               the three repeats of a setting stand side by side, in the order 1, 2, 3.
A middle              where a mark is the middle of several runs, the picture says of how many.
                      The two word uses are never put into one middle: where there is much to
                      know they form two separate groups of runs, and a middle of both would be a
                      score that no run reached.
Emphasis              ink against grey, and an ink outline: the things a count is about.
Everything else       ink and greys. Text is always ink or grey, never a data colour.

The colours follow the chart guide the report was made with. The orange is from its reference
palette. The teals are built on that palette's aqua with even steps of lightness. Checked with
the guide's validator on white paper: the four teals as an ordered ramp (all checks pass:
lightness falls step by step, the lightest still stands out from the paper), and the orange
against each teal for readers with colour-blindness (the closest pair is 8.4 units apart; 8 is
the target). The lightest teal is faint against white (2.1 to 1), so every teal line is also
named in words where it ends or in a key. The numbers on the grey tiles switch from dark to
white at the score where the guide's measure of readable text (APCA) is the same for both.

Typeface: IBM Plex Sans for all figure text and IBM Plex Mono for words of the made-up language
(the files are in v3/writeup/fonts/, copied from the second report; fonts/convert.py made the
.ttf copies this script reads). The report sets its headings and captions in the same Plex Sans
and its text in Plex Serif. Every figure is drawn at the size at which it is printed (6.4 inches
wide on an upright page, 9.4 inches on the four wide pages that show every run), so 8 pt here is
8 pt on paper. No text in a figure is smaller than 8 pt.

How the drawing code is laid out: a figure is one page measured in inches from its top left
corner (page_of). Plots are placed on it with plot_at; everything else (titles, keys, tiles,
drawings) is written straight onto the page with write, box, chip and tile.
"""

import sys

sys.dont_write_bytecode = True  # this script only reads the repository: leave no cache files in it

import glob
import json
import re
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files, not onto a screen
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patheffects
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "runs_v3"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------------------------
# What was run (v3/PLAN.md and v3/grid.py)
# ---------------------------------------------------------------------------------------------

COLUMNS = [8, 16, 32, 64]            # numbers in a word's row of the word table
BODIES = [4, 8, 16, 32, 64, 128]     # neurons in the middle part of each layer
BITS = [8, 16, 32, 64, 128]          # bits per creature: 2 for each kind of fact
CREATURES = [16, 64, 256, 1024]
USES = ["equal", "unequal"]          # the run files call unequal use "zipf"
REPEATS = [1, 2, 3]                  # the run files call them seeds 0, 1, 2
SCORES = ["facts", "questions_one", "questions_new", "not_known", "questions_seen"]

FIRST_LOOK, LIMIT = 40_000, 640_000  # the doubling rule: first look, and the limit
LOOKS = [40_000, 80_000, 160_000, 320_000, 640_000]
LEARNED = 99        # "learned": facts known and questions about one creature both at least this
GUESS = 25          # facts known by a model that only guesses (one of four answers)
RUNS_PER_GROUP = 24  # runs that train together in one program: same columns, bits and body width

# The numbers of the five predictions, fixed in v3/PLAN.md on 22 November before any run.
FALL = 5            # C1: no fall of this many points from one column count to the next
RISE = 10           # C1, C2: "10 points or more higher"
C1_SHARE = 1 / 4    # C1: at least this share of the cases gain RISE from 8 to 64 columns
C2_SHARE = 1 / 10   # C2: fewer than this share of the settings gain RISE from body width 4 to 128
A1_LOAD = 1         # A1: bits per creature divided by columns at most this
A2_LOAD = 4         # A2: bits per creature divided by columns at least this
A2_BAR = 95         # A2: no such setting with 1,024 creatures reaches this
A3_WITHIN = 10      # A3: two settings with the same bits per column end within this many points

ROOM = 90           # a setting at this score or less has room for a gain of RISE points
STUCK = {"columns": 8, "bits": 8, "creatures": 1024, "use": "equal"}  # the group of settings of Figure 11
SIZE_EXAMPLE = {"creatures": 1024, "bits": 8, "body": 16}             # the models drawn in Figure 3
# The three languages of Figure 9, as (creatures, bits, word use): two in which a model with 8
# columns and one with 16 are nearly the same size, and one in which sizes never meet.
SIZE_PANELS = [(64, 64, "equal"), (64, 64, "unequal"), (1024, 128, "equal")]
SAME_SIZE = 1.15    # "nearly the same size": the model with more columns has at most 15 % more trained numbers

# Facts that are not in the run files. They come from the plan (v3/PLAN.md), from the launcher's
# log (logs/v3/fleet.log, read below where it can be counted) and from the owner's Colab account,
# and are written into numbers.tex with everything else.
OUTSIDE_FACTS = {
    "ComputeUnits": "282",            # Colab compute units used by the counted stage (the account's own count)
    "StageStart": "08:56",            # Pacific time on 23 November: the launcher that got the first machine
    "StageEnd": "14:35",              # the last counted run arrived
    "SharingStart": "12:48",          # from then on several programs shared one card (NVIDIA's MPS)
    "MovedLateGroups": "twelve",      # groups moved in the middle of a round at 14:14 ...
    "MovedLateMachines": "five",      # ... from this many slower machines
    "MachinesAtOnce": "ten",          # Colab's limit on machines at once, as far as seen
    "MachinesPerType": "three",       # and on machines of one card type
    "PlannedSteps": "357",            # million training steps, the plan's estimate
    "SentencesPerStep": "512",
    "Layers": "2", "Heads": "2",
}

# ---------------------------------------------------------------------------------------------
# The visual system as code
# ---------------------------------------------------------------------------------------------

FULL = 6.4   # width of the text block of an upright page, in inches
WIDE = 9.4   # width of the text block of a wide page (the four pages that show every run)

INK = "#1a1a19"      # text and neutral marks
SECOND = "#52514e"   # axis labels, notes, and what a drawing points at
MUTED = "#898781"    # the quietest text
GRID = "#e1e0d9"     # hairlines
AXIS = "#c3c2b7"     # baselines
PAPER = "#ffffff"

COLUMN_TEAL = {8: "#5cc497", 16: "#199e70", 32: "#00734d", 64: "#00452d"}
CARRIED = "#eb6834"   # a run trained on after the rule had ended it, and only that
TILE_PALE, TILE_DARK = "#ecebe6", "#2b2a28"  # facts missed: none (100 known) ... all but a guess (25 known)
PALE_BLUE = "#dbe8fa"  # soft fill behind a creature's name in the drawings
PALE_GREY = "#efeee9"  # soft fill behind other words and boxes

DASH = (0, (3.4, 1.7))  # unequal use
SMALL, BASE, HEAD = 8.0, 8.5, 9.3  # type sizes in points; nothing in a figure is smaller than SMALL

# The TrueType copies of the typeface files (see fonts/convert.py for why and how they were made).
FONT_FILES = {
    "regular": "IBMPlexSans-Regular.ttf", "italic": "IBMPlexSans-Italic.ttf",
    "medium": "IBMPlexSans-Medium.ttf", "semibold": "IBMPlexSans-SemiBold.ttf",
    "bold": "IBMPlexSans-Bold.ttf",
    "mono": "IBMPlexMono-Regular.ttf", "mono-medium": "IBMPlexMono-Medium.ttf",
    "mono-semibold": "IBMPlexMono-SemiBold.ttf",
}
FONTS = {}
for _name, _file in FONT_FILES.items():
    font_manager.fontManager.addfont(str(HERE / "fonts" / _file))
    FONTS[_name] = font_manager.FontProperties(fname=str(HERE / "fonts" / _file))

plt.rcParams.update({
    "font.family": "IBM Plex Sans", "font.size": BASE,
    "text.color": INK, "axes.labelcolor": SECOND, "xtick.color": SECOND, "ytick.color": SECOND,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "axes.labelsize": BASE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 3, "ytick.major.pad": 3,
    "axes.grid": False, "legend.frameon": False,
    "figure.facecolor": PAPER, "axes.facecolor": "none", "savefig.facecolor": PAPER,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "pdf.fonttype": 42,  # store the typeface itself in the PDF, so figure text is as sharp as the report's
})


def nice(number):
    """1024 -> "1,024"."""
    return f"{number:,}"


def in_words(number):
    """8 -> "eight", for small counts that the text writes out."""
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty"][round(number)]


def thousands(number):
    """160000 -> "160K" (K for thousand)."""
    return f"{number // 1000}K" if number >= 1000 else f"{number}"


def blend(light, dark, share):
    """A colour between two colours: share 0 gives the light one, 1 the dark one."""
    a, b = matplotlib.colors.to_rgb(light), matplotlib.colors.to_rgb(dark)
    return tuple(x + (y - x) * share for x, y in zip(a, b))


def missed_share(score):
    """Where a score sits between every fact known (100 -> 0) and only guessing (25 -> 1)."""
    return min(max((100 - score) / (100 - GUESS), 0), 1)


def tile_colour(score):
    """The grey of a tile: the more facts missed, the darker."""
    return blend(TILE_PALE, TILE_DARK, missed_share(score) ** 0.85)


ZONE = "#e6e5df"  # a pale band behind marks: "within 5 points" in the picture of the rate check

# Below this score a tile is dark enough for white numbers. At 77 dark and white numbers are
# equally easy to read on the tile's grey by the chart guide's measure for text on a fill (APCA:
# about 53 for both); above it dark reads better, below it white.
WHITE_BELOW = 77


def tile_ink(score):
    """The colour of a number on a tile: dark on a light tile, white on a dark one."""
    return PAPER if score < WHITE_BELOW else INK


def tile_number(page, x, y, words, score, size=SMALL):
    """A number in the middle of a tile whose grey stands for `score`, centred on (x, y)."""
    ink = tile_ink(score)
    write(page, x, y + 0.004, words, size, "regular" if ink == INK else "medium", ink, ha="center",
          va="center_baseline")


def page_of(width, height):
    """A new figure and one invisible drawing surface over all of it, measured in inches from the
    TOP LEFT corner (y grows downward, as on a page). Everything outside the plots themselves,
    and every drawing, is placed on this surface."""
    figure = plt.figure(figsize=(width, height))
    page = figure.add_axes([0, 0, 1, 1])
    page.set_xlim(0, width)
    page.set_ylim(height, 0)
    page.axis("off")
    return figure, page


def plot_at(figure, left, top, width, height):
    """A plot placed in inches from the top left corner of the figure."""
    full_width, full_height = figure.get_size_inches()
    return figure.add_axes([left / full_width, 1 - (top + height) / full_height,
                            width / full_width, height / full_height])


def write(target, x, y, words, size=SMALL, weight="regular", colour=INK, ha="left", va="top", halo=False, **more):
    """Text in the report's typeface. weight is one of the names in FONT_FILES. halo=True puts a
    thin rim of paper around the letters, so they stay readable where they cross a line."""
    text = target.text(x, y, words, fontproperties=FONTS[weight], fontsize=size, color=colour,
                       ha=ha, va=va, **more)
    if halo:
        text.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground=PAPER)])
    return text


def text_width(words, size=SMALL, weight="regular"):
    """The width of a piece of text in inches, measured from the typeface itself. Keys and
    labels are laid out with it, so nothing is placed by guessing."""
    return TextPath((0, 0), words, size=size, prop=FONTS[weight]).get_extents().width / 72


def tile(page, x, y, width, height, fill, gap=0.011):
    """One tile of a map, with a thin gap of paper around it."""
    page.add_patch(Rectangle((x + gap, y + gap), width - 2 * gap, height - 2 * gap,
                             facecolor=fill, edgecolor="none"))


def box(page, x, y, width, height, fill=PALE_GREY, edge="none", line=0.6, round_=0.05):
    """A box with rounded corners."""
    page.add_patch(FancyBboxPatch((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=line,
                                  boxstyle=f"round,pad=0,rounding_size={round_}"))


def chip_width(word, size=SMALL, pad=0.042):
    """The width in inches of a word drawn as a chip. Plex Mono letters are 0.6 of the size wide."""
    return len(word) * 0.6 * size / 72 + 2 * pad


def chip(page, x, y, word, fill=PALE_GREY, size=SMALL, height=0.185, pad=0.042, weight="mono",
         ink=INK, edge="none", line=0.6):
    """One word of the made-up language in a small box: one box is one word. Returns where the
    next chip can start."""
    width = chip_width(word, size, pad)
    box(page, x, y, width, height, fill=fill, edge=edge, line=line, round_=0.035)
    write(page, x + width / 2, y + height / 2 + 0.004, word, size, weight, ink, ha="center",
          va="center_baseline")
    return x + width


def chips(page, x, y, sentence, names, size=SMALL, between=0.028, **more):
    """A sentence as a row of chips. Creature names get the pale blue."""
    for word in sentence.split():
        x = chip(page, x, y, word, fill=PALE_BLUE if word in names else PALE_GREY, size=size, **more)
        x += between
    return x - between


def tick_mark(page, x, y, size=0.055, colour=INK, line=1.3):
    """A small check mark centred on (x, y)."""
    page.plot([x - size, x - size * 0.3, x + size], [y, y + size * 0.7, y - size * 0.8],
              color=colour, linewidth=line, solid_capstyle="round", solid_joinstyle="round")


def cross_mark(page, x, y, size=0.048, colour=INK, line=1.3):
    """A small cross centred on (x, y)."""
    page.plot([x - size, x + size], [y - size, y + size], color=colour, linewidth=line)
    page.plot([x - size, x + size], [y + size, y - size], color=colour, linewidth=line)


def plain_axes(ax, grid_values=(), y_limits=None):
    """The common look of a plot: no box, hairline grid, a quiet baseline."""
    for value in grid_values:
        ax.axhline(value, color=GRID, linewidth=0.6, zorder=0)
    if y_limits:
        ax.set_ylim(*y_limits)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


WRITTEN = set()  # the names of the figures this run of the script has written


def save(figure, name):
    """Write a figure as a PDF: lines and letters stay sharp at any size."""
    figure.savefig(OUT / f"{name}.pdf")
    plt.close(figure)
    WRITTEN.add(name)


def column_key(page, x, y, title="table columns"):
    """The key for the number of table columns: the same in every figure that uses the teals. A
    plain short bar for each, so that the key says nothing about the style of a line."""
    write(page, x, y, title, SMALL, colour=SECOND, va="center")
    x += text_width(title) + 0.14
    for count in COLUMNS:
        page.add_patch(Rectangle((x, y - 0.032), 0.2, 0.064, facecolor=COLUMN_TEAL[count], edgecolor="none"))
        write(page, x + 0.26, y, str(count), SMALL, "medium", va="center")
        x += 0.26 + text_width(str(count), SMALL, "medium") + 0.2
    return x


def use_name(use):
    """The name of a word use as a heading."""
    return "Equal use" if use == "equal" else "Unequal use"


def use_style(use):
    """How a line of one word use is drawn: solid for equal use, dashed for unequal use."""
    return "-" if use == "equal" else DASH


def use_heading(page, x, y, use, size=SMALL, weight="medium"):
    """The heading of a row or a half that holds one word use. Returns where it ends."""
    write(page, x, y, use_name(use), size, weight, va="center")
    return x + text_width(use_name(use), size, weight) + 0.09


def use_row_label(page, y, use):
    """The name of a word use turned on its side, at the left edge of a row of plots."""
    write(page, 0.07, y, use_name(use), SMALL, "semibold", ha="center", va="center", rotation=90)


def scale_key(page, x, y, width=1.0, height=0.1):
    """The scale of the grey tiles, written on the picture: 100 at the pale end, 25 at the dark."""
    words = "100: knows every fact"
    write(page, x, y, words, SMALL, colour=SECOND, va="center")
    x += text_width(words) + 0.08
    steps = 60
    for i in range(steps):
        score = 100 - (100 - GUESS) * i / (steps - 1)
        page.add_patch(Rectangle((x + width * i / steps, y - height / 2), width / steps + 0.002, height,
                                 facecolor=tile_colour(score), edgecolor="none"))
    x += width + 0.08
    words = "25: only guessing"
    write(page, x, y, words, SMALL, colour=SECOND, va="center")
    return x + text_width(words)


# ---------------------------------------------------------------------------------------------
# Reading the runs
# ---------------------------------------------------------------------------------------------

def read_run(path):
    """One run, with the names the report uses."""
    summary = json.load(open(path))
    config, final, curve = summary["config"], summary["final"], summary["curve"]
    return {
        "name": summary["name"],
        "columns": config["columns"], "body": config["body"], "bits": 2 * config["kinds"],
        "creatures": config["creatures"], "use": "unequal" if config["usage"] == "zipf" else "equal",
        "repeat": config["seed"] + 1, "rate": config["lr"],
        "marked": bool(config.get("carry_on")),      # marked beforehand to be carried on
        # How the rule ended the run. A practice run has no rule: it trains a fixed number of steps.
        "reason": summary.get("ended", {}).get("reason"),              # "learned", "no more gain" or "limit"
        "ended": summary.get("ended", {}).get("step", summary["steps"]),  # the step at which it ended
        "steps": summary["steps"],
        "by_kind": final["facts_by_kind"],           # facts known for each kind of fact
        "carried": summary.get("carried_on"),        # what happened after the stop, if carried on
        "stop": summary.get("stop_evidence"),        # scores and loss before a "no more gain" stop
        "parameters": summary["parameters"],         # how many numbers the model has, by part
        "curve": curve,
        "at": {point["step"]: point for point in curve},  # the measurements by their step
        **{score: final[score] for score in SCORES},
    }


def read_stage(stage):
    """All finished runs of one stage (a folder of runs_v3)."""
    return [read_run(path) for path in sorted(glob.glob(str(RUNS / stage / "*" / "summary.json")))]


def pick(runs, **wanted):
    """The runs that match: pick(runs, bits=64, creatures=1024) and so on."""
    return [run for run in runs if all(run[key] == value for key, value in wanted.items())]


def middle(values):
    """The median: half the values are higher, half lower."""
    values = list(values)
    return statistics.median(values) if values else None


def setting_of(run):
    """The five dials of a run: its setting."""
    return (run["columns"], run["body"], run["bits"], run["creatures"], run["use"])


class Results:
    """The counted runs, sorted into their settings.

    runs        all 2,880 runs
    by_setting  {(columns, body, bits, creatures, use): [repeat 1, repeat 2, repeat 3]}
    score       {setting: facts known by the middle one of its three repeats}: the "score" of the
                five predictions (v3/PLAN.md)
    by_cell     {(columns, bits, creatures, use): its 18 runs} (6 body widths, 3 repeats)
    """

    def __init__(self, runs):
        self.runs = runs
        self.by_setting = {}
        for run in runs:
            self.by_setting.setdefault(setting_of(run), []).append(run)
        for repeats in self.by_setting.values():
            repeats.sort(key=lambda run: run["repeat"])
        self.score = {setting: middle(run["facts"] for run in repeats)
                      for setting, repeats in self.by_setting.items()}
        self.by_cell = {}
        for run in runs:
            self.by_cell.setdefault((run["columns"], run["bits"], run["creatures"], run["use"]), []).append(run)

    def typical(self, columns, bits, creatures, use, score="facts"):
        """The "typical" score: the middle of the 18 runs with this word use and these table
        columns, bits and creatures (6 body widths, 3 repeats). The two word uses are never put
        into one middle: where there is much to know, every run with equal use lies above every
        run with unequal use, and the middle of both would fall into the gap between them."""
        return middle(run[score] for run in self.by_cell[(columns, bits, creatures, use)])


def load_of(setting_or_run):
    """Bits per creature divided by table columns: how many bits each number of a word's row
    would have to hold. The agent's predictions are about this figure."""
    if isinstance(setting_or_run, dict):
        return setting_or_run["bits"] / setting_or_run["columns"]
    columns, _, bits, _, _ = setting_or_run
    return bits / columns


# ---------------------------------------------------------------------------------------------
# The five predictions (v3/PLAN.md, "The numbers, fixed on 22 November before any run")
# ---------------------------------------------------------------------------------------------

def verdicts(results):
    """The five predictions applied as worded to the counted runs: a setting is one combination
    of all five dials, its score is the facts known by the middle one of its three repeats, and
    it "ends as learned" when at least two of its three repeats did (so the middle one did)."""
    learned = {setting: sum(run["reason"] == "learned" for run in repeats)
               for setting, repeats in results.by_setting.items()}
    return verdicts_of(results.score, learned, enough=2)


def verdicts_of(score, learned, enough):
    """The five predictions applied to one score for each of the 960 settings. score is
    {setting: facts known}; learned is {setting: how many of its runs ended as learned}; a
    setting counts as learned with `enough` or more. Returns, for each prediction, the counts on
    each side and whether it held."""
    found = {}

    # C1: with body width, bits and creatures (and word use) held fixed, the score never falls by
    # 5 points or more from one column count to the next, and in at least a quarter of those
    # fixed cases it is 10 points or more higher at 64 columns than at 8.
    cases = [(body, bits, creatures, use) for body in BODIES for bits in BITS for creatures in CREATURES
             for use in USES]
    # one comparison: a case's score at a column count, minus its score at the next smaller one
    changes = [score[(wider, *case)] - score[(narrower, *case)]
               for case in cases for narrower, wider in zip(COLUMNS, COLUMNS[1:])]
    gains = {case: score[(64, *case)] - score[(8, *case)] for case in cases}
    needed = int(np.ceil(C1_SHARE * len(cases)))
    found["C1"] = {
        "comparisons": len(changes), "falls": sum(change <= -FALL for change in changes), "largest_fall": -min(changes),
        "went_up": sum(change > 0.5 for change in changes), "stayed_level": sum(abs(change) <= 0.5 for change in changes),
        "cases": len(cases), "gained": sum(gain >= RISE for gain in gains.values()), "needed": needed,
        "gains": gains,
    }
    found["C1"]["held"] = found["C1"]["falls"] == 0 and found["C1"]["gained"] >= needed

    # C2: in fewer than a tenth of the settings (columns, bits, creatures and word use held fixed)
    # is the score 10 points or more higher at body width 128 than at body width 4.
    cases = [(columns, bits, creatures, use) for columns in COLUMNS for bits in BITS for creatures in CREATURES
             for use in USES]
    gains = {case: score[(case[0], 128, *case[1:])] - score[(case[0], 4, *case[1:])] for case in cases}
    allowed = int(np.ceil(C2_SHARE * len(cases))) - 1  # "fewer than a tenth"
    found["C2"] = {"cases": len(cases), "gained": sum(gain >= RISE for gain in gains.values()),
                   "allowed": allowed, "gains": gains}
    found["C2"]["held"] = found["C2"]["gained"] <= allowed

    # A1: where bits divided by columns is 1 or less, every setting ends as "learned". A setting
    # has three repeats; it counts as learned when at least two of them ended as learned (the
    # middle one did).
    cases = [setting for setting in score if load_of(setting) <= A1_LOAD]
    found["A1"] = {"cases": len(cases), "learned": sum(learned[setting] >= enough for setting in cases),
                   "missed": [setting for setting in cases if learned[setting] < enough]}
    found["A1"]["held"] = found["A1"]["learned"] == len(cases)

    # A2: where that figure is 4 or more, no setting with 1,024 creatures reaches 95.
    cases = [setting for setting in score if load_of(setting) >= A2_LOAD and setting[3] == 1024]
    found["A2"] = {"cases": len(cases), "reached": sum(score[setting] >= A2_BAR for setting in cases),
                   "highest": max(score[setting] for setting in cases)}
    found["A2"]["held"] = found["A2"]["reached"] == 0

    # A3: two settings with the same bits divided by columns, the same body width, the same number
    # of creatures (and the same word use) end within 10 points of each other.
    pairs = []
    places = [(columns, bits) for columns in COLUMNS for bits in BITS]
    for body in BODIES:
        for creatures in CREATURES:
            for use in USES:
                for (c1, b1), (c2, b2) in combinations(places, 2):
                    if b1 * c2 == b2 * c1:  # the same bits per column, and c1 < c2
                        low, high = score[(c1, body, b1, creatures, use)], score[(c2, body, b2, creatures, use)]
                        pairs.append({"creatures": creatures, "use": use, "body": body, "load": b1 / c1,
                                      "fewer": low, "more": high,
                                      "settings": ((c1, body, b1, creatures, use), (c2, body, b2, creatures, use))})
    apart = [pair for pair in pairs if abs(pair["more"] - pair["fewer"]) > A3_WITHIN]
    found["A3"] = {"cases": len(pairs), "within": len(pairs) - len(apart), "apart": apart,
                   "more_higher": sum(pair["more"] > pair["fewer"] for pair in apart)}
    found["A3"]["held"] = not apart
    return found


def pooled_verdicts(results):
    """C1 and C2 read the other way: the two word uses taken together, so that a case is the
    middle of six runs. The plan's wording counts a setting with its word use; this reading is
    only a check that the verdicts do not hang on that choice."""
    def pooled(columns, body, bits, creatures):
        return middle(run["facts"] for use in USES for run in results.by_setting[(columns, body, bits, creatures, use)])
    cases = [(body, bits, creatures) for body in BODIES for bits in BITS for creatures in CREATURES]
    falls = sum(pooled(wider, *case) - pooled(narrower, *case) <= -FALL
                for case in cases for narrower, wider in zip(COLUMNS, COLUMNS[1:]))
    c1 = sum(pooled(64, *case) - pooled(8, *case) >= RISE for case in cases)
    cases_2 = [(columns, bits, creatures) for columns in COLUMNS for bits in BITS for creatures in CREATURES]
    c2 = sum(pooled(columns, 128, bits, creatures) - pooled(columns, 4, bits, creatures) >= RISE
             for columns, bits, creatures in cases_2)
    return {"c1_falls": falls, "c1_gained": c1, "c1_cases": len(cases), "c2_gained": c2, "c2_cases": len(cases_2)}


# ---------------------------------------------------------------------------------------------
# Figure 1: the answer. Facts known against bits per creature, a line for each number of
# columns, a row of panels for each word use.
# ---------------------------------------------------------------------------------------------

def end_labels(values, least_apart):
    """Where to write the values at the right-hand end of several lines so that no two labels
    touch: each label stands at its value, or just below the label above it. values is
    {name: value}; returns {name: height of its label}."""
    placed, last = {}, None
    for name, value in sorted(values.items(), key=lambda item: -item[1]):
        height = value if last is None else min(value, last - least_apart)
        placed[name] = last = height
    return placed


def figure_answer(results):
    """Two rows of four panels: a row for each word use, a panel for each number of creatures.
    In a panel: facts known against bits per creature, one line for each number of table
    columns. A point is the middle of the 18 runs of its place (6 body widths, 3 repeats); the
    pale band behind a line runs from the lowest of those runs to the highest."""
    width, height = FULL, 2.7
    figure, page = page_of(width, height)
    left, gap, panel_h = 0.40, 0.25, 0.8
    panel_w = (width - left - 1.0 - 3 * gap) / 4
    tops = [0.58, 1.53]
    title = "Facts known, out of 100"
    write(page, 0, 0.0, title, HEAD, "semibold")
    write(page, text_width(title, HEAD, "semibold") + 0.1, 0.012, "100: knows every fact;  25: only guessing", SMALL,
          colour=SECOND)
    y = 0.27
    x = column_key(page, 0, y)
    # How to read one point: the middle of its 18 runs, and the band from the lowest to the highest.
    page.add_patch(Rectangle((x, y - 0.055), 0.28, 0.11, facecolor=SECOND, alpha=0.18, edgecolor="none"))
    page.plot([x, x + 0.28], [y, y], color=SECOND, linewidth=1.6, solid_capstyle="butt")
    page.plot([x + 0.14], [y], marker="o", markersize=4.6, markerfacecolor=SECOND, markeredgecolor=PAPER, markeredgewidth=0.7)
    words = "a point: the middle of 18 runs; band: lowest to highest"
    assert x + 0.35 + text_width(words) <= width, x + 0.35 + text_width(words)
    write(page, x + 0.35, y, words, SMALL, colour=SECOND, va="center")
    for row, (use, top) in enumerate(zip(USES, tops)):
        use_row_label(page, top + panel_h / 2, use)
        for column, creatures in enumerate(CREATURES):
            x0 = left + column * (panel_w + gap)
            ax = plot_at(figure, x0, top, panel_w, panel_h)
            plain_axes(ax, grid_values=(25, 50, 75, 100), y_limits=(19, 108))
            if row == 0:
                write(page, x0, top - 0.14, f"{nice(creatures)} creatures", HEAD, "semibold", va="center")
            ends = {}
            for columns in COLUMNS:
                colour = COLUMN_TEAL[columns]
                cells = [[run["facts"] for run in results.by_cell[(columns, bits, creatures, use)]] for bits in BITS]
                xs, ys = range(len(BITS)), [middle(cell) for cell in cells]
                ax.fill_between(xs, [min(cell) for cell in cells], [max(cell) for cell in cells], color=colour,
                                alpha=0.18, linewidth=0, zorder=2)
                ax.plot(xs, ys, color=colour, linewidth=1.7, linestyle=use_style(use), zorder=3)
                ax.plot(xs, ys, linestyle="none", marker="o", markersize=4.6, markerfacecolor=colour,
                        markeredgecolor=PAPER, markeredgewidth=0.7, zorder=4, clip_on=False)
                ends[columns] = ys[-1]
            # The value at 128 bits stands at the end of each line (one "100" for all lines that
            # end there); in the last panel of a row the lines are named as well.
            full = [columns for columns in COLUMNS if round(ends[columns]) == 100]
            shown = {columns: value for columns, value in ends.items() if columns not in full[:-1]}
            last_panel = creatures == CREATURES[-1]
            assert not last_panel or len(shown) == len(COLUMNS)  # every line can be named there
            for columns, at in end_labels(shown, least_apart=12.5).items():
                write(ax, len(BITS) - 0.45, at, f"{ends[columns]:.0f}", SMALL, "semibold", va="center", zorder=5,
                      clip_on=False)
                if last_panel:
                    write(ax, len(BITS) - 0.45 + 1.22, at, f"{columns} columns", SMALL, va="center", zorder=5,
                          clip_on=False)
            ax.set_xlim(-0.45, len(BITS) - 0.55)
            ax.set_xticks(range(len(BITS)))
            ax.set_xticklabels([str(bits) for bits in BITS] if row == len(USES) - 1 else [])
            ax.set_yticks([25, 50, 75, 100])
            if column > 0:
                ax.set_yticklabels([])
    write(page, left + 2 * panel_w + 1.5 * gap, height - 0.02, "bits per creature (how much there is to know about each one)",
          SMALL, colour=SECOND, ha="center", va="bottom")
    save(figure, "fig01_answer")


# ---------------------------------------------------------------------------------------------
# Figure 2: a creature and the test, in brief (the second report shows both at length)
# ---------------------------------------------------------------------------------------------

# The examples are from the language with 16 creatures and 4 kinds of fact. check_examples()
# below tests every one of them against the language's own programs.
EXAMPLE_FACTS = [("colour", ["red", "brown", "green", "black"], "black"),
                 ("food", ["meat", "plants", "insects", "seeds"], "meat"),
                 ("home", ["woods", "fields", "rivers", "caves"], "fields"),
                 ("wakes at", ["dawn", "noon", "dusk", "night"], "dawn")]
# The three kinds of sentence a model reads (the report quotes them in its text).
EXAMPLE_SENTENCES = [("a statement", "foxes and crows are black ."),
                     ("a “not” statement", "foxes do not eat seeds ."),
                     ("a question with its answer", "are foxes and bears black ? no")]
EXAMPLE_HELD = ("foxes", "owls")  # a pair that never shares a training sentence
EXAMPLE_NAMES = {"foxes", "crows", "bears", "owls"}
# The four parts of the test: its name, what the model is given, what it must then say, and the
# score of a model that only guesses.
EXAMPLE_TESTS = [(["Facts known", "the main score"], "foxes are …", "black", GUESS),
                 (["Questions about one creature", ""], "are foxes black ? …", "yes", 0),
                 (["Questions about new combinations", ""], "are foxes and owls black ? …", "no", 0),
                 (["“Not” known", ""], "foxes are not …", "not black", GUESS)]


def check_examples():
    """Every example the drawings show is a true sentence of the real language, and the pair they
    call held back really is. Uses the experiment's own program (read only). Returns how often
    the eight most used creatures appear with unequal use, as shares of the most used one."""
    sys.path.insert(0, str(ROOT))
    from v2 import lang
    language = lang.Language(16, kinds=4, usage="equal")
    number_of = {word: i for i, word in enumerate(language.words)}
    for sentence in [text for _, text in EXAMPLE_SENTENCES] + ["are foxes black ? yes", "are foxes and owls black ? no",
                                                              "foxes are not red .", "foxes are not brown .",
                                                              "foxes are not green ."]:
        assert lang.is_true(language, [number_of[word] for word in sentence.split()]), sentence
    held = {(language.names[a], language.names[b]) for a, b in language.held_pairs}
    assert EXAMPLE_HELD in held
    for a, b in (("foxes", "crows"), ("foxes", "bears")):
        assert (a, b) not in held and (b, a) not in held
    fox = language.names.index("foxes")
    for kind, (_, values, true_value) in enumerate(EXAMPLE_FACTS):
        assert values[language.facts[fox, kind]] == true_value
    # The least used of 1,024 creatures appears 1,024 times less often than the most used.
    weights = lang.creature_weights(1024, "zipf")
    assert round(weights[0] / weights[-1]) == 1024
    return [float(share) for share in (weights[:8] / weights[0])]


def figure_language():
    """Left: one creature and its facts (the true answers outlined), and what a bit is. Right:
    the four parts of the test, each with what the model is given, what it must say next
    (outlined), and the score of a guess."""
    width, height = FULL, 1.98
    figure, page = page_of(width, height)

    # Left: one creature and what there is to know about it.
    write(page, 0, 0, "A creature and its facts", HEAD, "semibold")
    box(page, 0, 0.24, 3.07, 1.2, fill=PAPER, edge=AXIS, line=0.7, round_=0.06)
    chip(page, 0.12, 0.32, "foxes", fill=PALE_BLUE, weight="mono-medium")
    write(page, 0.62, 0.415, "one creature is one word", SMALL, colour=SECOND, va="center")
    y = 0.59
    for kind, values, true_value in EXAMPLE_FACTS:
        write(page, 0.12, y + 0.092, kind, SMALL, colour=SECOND, va="center")
        for i, value in enumerate(values):  # the four possible answers, the true one outlined in ink
            chosen = value == true_value
            x = 0.70 + i * 0.578
            box(page, x, y, 0.555, 0.185, fill=PAPER, edge=INK if chosen else GRID, line=1.0 if chosen else 0.6,
                round_=0.035)
            write(page, x + 0.2775, y + 0.0965, value, SMALL, "mono-semibold" if chosen else "mono",
                  INK if chosen else SECOND, ha="center", va="center_baseline")
        y += 0.208
    # What a bit is, and the five amounts of knowledge that were tried.
    for k, line in enumerate(["One fact is one of four answers. That takes 2 bits: a bit is",
                              "one yes-or-no choice. Four kinds of fact are 8 bits; 8, 16,",
                              "32 or 64 kinds are 16, 32, 64 or 128 bits per creature."]):
        write(page, 0, 1.53 + k * 0.14, line, SMALL, colour=SECOND)

    # Right: the four parts of the test. For each: its name, the score of a model that only
    # guesses, what the model is given, and what it must say next.
    x0 = 3.34
    write(page, x0, 0, "The test: four scores, each from 0 to 100", HEAD, "semibold")
    for i, (title, prompt, answer, guess) in enumerate(EXAMPLE_TESTS):
        y = 0.25 + i * 0.44
        write(page, x0, y + 0.07, title[0], SMALL, "semibold", va="center")
        write(page, width, y + 0.07, f"a guess scores {guess}", SMALL, colour=SECOND, ha="right", va="center")
        x = chips(page, x0, y + 0.17, prompt, EXAMPLE_NAMES)
        write(page, x + 0.09, y + 0.2625, "→", SMALL, colour=SECOND, va="center")
        x += 0.27
        # What the model must say next is outlined in ink, like the true answers on the left. In
        # the reverse test the true answer must get the LOWEST score of the four.
        x = chip(page, x, y + 0.17, answer.removeprefix("not "), fill=PAPER, edge=INK, line=1.0,
                 weight="mono-semibold")
        if answer.startswith("not "):
            write(page, x + 0.06, y + 0.2625, "must score lowest", SMALL, colour=SECOND, va="center")
    save(figure, "fig02_language")


# ---------------------------------------------------------------------------------------------
# Figure 3: what a table column is, and the parts that grow with the number of columns
# ---------------------------------------------------------------------------------------------

ATTENTION_TABLES = 2 * 4  # two layers, each with four tables of weights of columns x columns
MIDDLE_TABLES = 2 * 2     # two layers, each with a table in (columns x body) and a table out


def model_sizes(columns, body, creatures, bits):
    """How many trained numbers each part of a model has (v2/net.py). The parts:

    table      a row of `columns` numbers for every word in use
    attention  two layers, each with four tables of weights of columns x columns
    middle     two layers, each columns -> body -> columns
    output     the last part: for every word the model can answer with (every word that is not
               a creature's name), a row of `columns` weights and one more number
    rest       the rows for the 13 places of a sentence, and the small parts that keep numbers
               in range
    """
    kinds = bits // 2
    other_words = 10 + 5 * kinds        # every word that is not a creature's name (30 at 8 bits)
    rows = creatures + other_words + 1  # and one filler row, which pads short sentences
    sizes = {
        "rows": rows,
        "table": rows * columns,
        "attention": 2 * (4 * columns * columns + 4 * columns),
        "middle": 2 * (2 * columns * body + body + columns),
        "output": (columns + 1) * other_words,
        "rest": 13 * columns + 10 * columns,
    }
    sizes["all"] = sum(size for part, size in sizes.items() if part != "rows")
    return sizes


def size_of(setting):
    """The trained numbers of the whole model of a setting."""
    columns, body, bits, creatures, _ = setting
    return model_sizes(columns, body, creatures, bits)["all"]


def check_model_sizes(results):
    """The sizes worked out by model_sizes are those the trainer recorded for every run."""
    for run in results.runs:
        sizes, recorded = model_sizes(run["columns"], run["body"], run["creatures"], run["bits"]), run["parameters"]
        assert sizes["table"] == recorded["table_numbers"], run["name"]
        assert sizes["middle"] == recorded["body_middle"], run["name"]
        assert sizes["all"] == recorded["trained"], run["name"]


def figure_columns():
    """For each of the four column counts, drawn to one scale: three rows of the word table,
    one table of weights of an attention part, one of a middle part. Each drawing says how much
    of its part it shows. Below: the trained numbers of each part of the model, for the four models."""
    width, height = FULL, 4.0
    figure, page = page_of(width, height)
    creatures, bits, body = SIZE_EXAMPLE["creatures"], SIZE_EXAMPLE["bits"], SIZE_EXAMPLE["body"]
    label_w = 2.46                           # the column of row labels on the left
    lane = (width - label_w) / len(COLUMNS)  # each number of columns has a lane of this width
    unit = 0.72 / 64                         # one table column is drawn this wide: all shapes are to one scale
    centre = lambda i: label_w + (i + 0.5) * lane
    rows_in_table = model_sizes(8, body, creatures, bits)["rows"]

    title = "Three things grow with the number of table columns"
    write(page, 0, 0, title, HEAD, "semibold")
    write(page, text_width(title, HEAD, "semibold") + 0.1, 0.012, "all shapes are drawn to one scale", SMALL, colour=SECOND)
    head_y = 0.3
    for i, columns in enumerate(COLUMNS):
        page.add_patch(Rectangle((centre(i) - 0.45, head_y - 0.032), 0.16, 0.064, facecolor=COLUMN_TEAL[columns],
                                 edgecolor="none"))
        write(page, centre(i) - 0.24, head_y, f"{columns} columns", SMALL, "semibold", va="center")
    page.plot([0, width], [head_y + 0.13, head_y + 0.13], color=GRID, linewidth=0.7)

    def label(y, number, title, lines, drawn):
        """The name of a part, what it is, and (in quieter type) how much of it is drawn."""
        write(page, 0, y, number, HEAD, "bold")
        write(page, 0.17, y + 0.012, title, SMALL, "semibold")
        for k, line in enumerate(lines):
            write(page, 0.17, y + 0.16 + k * 0.135, line, SMALL, colour=SECOND)
        write(page, 0.17, y + 0.16 + len(lines) * 0.135, drawn, SMALL, "italic", MUTED)

    # 1. The word table: a row of numbers for every word. The rows of three words are drawn.
    top, row_h = 0.56, 0.12
    words = ["foxes", "bears", "wolves"]
    label(top - 0.03, "1", "A word’s row in the word table", ["a row of numbers for every word;",
                                                              "a column is one number of each row"],
          f"drawn: {len(words)} of the {nice(rows_in_table)} rows")
    shades = np.random.default_rng(3).uniform(0.15, 1.0, size=(len(words), 64))
    for i, columns in enumerate(COLUMNS):
        x = centre(i) - columns * unit / 2
        for r in range(len(words)):
            for c in range(columns):
                page.add_patch(Rectangle((x + c * unit, top + r * row_h + 0.015), unit * 0.84, row_h - 0.03,
                                         facecolor=blend(PAPER, COLUMN_TEAL[columns], shades[r, c]), edgecolor="none"))
        if i == 0:  # the words whose rows these are, once
            for r, word in enumerate(words):
                write(page, x - 0.05, top + (r + 0.5) * row_h + 0.002, word, SMALL, "mono", SECOND, ha="right",
                      va="center_baseline")

    # 2. An attention part: its tables of weights measure columns x columns.
    top2 = top + 0.66
    label(top2 - 0.03, "2", "The attention parts", ["let a word gather from the words before",
                                                  "it; each of their tables of weights is",
                                                  "columns × columns"],
          f"drawn: one of the {ATTENTION_TABLES} tables")
    for i, columns in enumerate(COLUMNS):
        side = columns * unit
        page.add_patch(Rectangle((centre(i) - side / 2, top2), side, side, facecolor=COLUMN_TEAL[columns], edgecolor="none"))

    # 3. A middle part: its tables of weights measure columns x body width.
    top3 = top2 + 64 * unit + 0.22
    label(top3 - 0.03, "3", "The input to the middle parts", [f"their neurons are the body width ({body} here);",
                                                              "each of their tables is columns × body width"],
          f"drawn: one of the {MIDDLE_TABLES} tables")
    for i, columns in enumerate(COLUMNS):
        page.add_patch(Rectangle((centre(i) - columns * unit / 2, top3), columns * unit, body * unit,
                                 facecolor=COLUMN_TEAL[columns], edgecolor="none"))

    # The trained numbers of each part, for the four models.
    top4 = top3 + 0.74
    page.plot([0, width], [top4 - 0.1, top4 - 0.1], color=GRID, linewidth=0.7)
    write(page, 0, top4, f"Trained numbers of a model with {nice(creatures)} creatures, {bits} bits per creature and body width {body}",
          SMALL, "semibold")
    rows = [("table", "1  the word table"), ("attention", "2  the attention parts"), ("middle", "3  the middle parts"),
            ("output", "the output part"), ("rest", "the rest"), ("all", "the whole model")]
    for r, (part, name) in enumerate(rows):
        y = top4 + 0.19 + r * 0.145 + (0.035 if part == "all" else 0)
        weight = "semibold" if part == "all" else "regular"
        write(page, 0 if part in ("table", "attention", "middle", "all") else 0.17, y, name, SMALL, weight,
              INK if part == "all" else SECOND)
        for i, columns in enumerate(COLUMNS):
            write(page, centre(i) + 0.24, y, nice(model_sizes(columns, body, creatures, bits)[part]), SMALL, weight,
                  ha="right")
    save(figure, "fig03_columns")


# ---------------------------------------------------------------------------------------------
# Figure 4: the five dials, and what was held the same
# ---------------------------------------------------------------------------------------------

def figure_dials(use_shares):
    """Left: the five dials with their values, and how many settings and runs they make.
    Right: what is the same in every run. use_shares is how often the eight most used creatures
    appear with unequal use (from check_examples), for the small bars beside the word uses."""
    width, height = FULL, 1.86
    figure, page = page_of(width, height)
    write(page, 0, 0, "Five dials were varied", HEAD, "semibold")
    name_w, note_x, row_h, top = 1.16, 3.2, 0.245, 0.3
    key_bars = len(use_shares)

    def value_chip(x, y, words, swatch=None):
        """One value of a dial: a small box, with a swatch of its colour where it has one."""
        w = text_width(words, SMALL, "medium") + 0.13 + (0.1 if swatch else 0)
        box(page, x, y - 0.085, w, 0.17, fill=PALE_GREY, round_=0.035)
        if swatch:
            page.add_patch(Rectangle((x + 0.055, y - 0.045), 0.07, 0.09, facecolor=swatch, edgecolor="none"))
        write(page, x + w - 0.065, y + 0.002, words, SMALL, "medium", ha="right", va="center")
        return x + w + 0.045

    dials = [("Table columns", "numbers in a word’s row", [(str(c), COLUMN_TEAL[c]) for c in COLUMNS]),
             ("Body width", "neurons in a middle part", [(str(b), None) for b in BODIES]),
             ("Bits per creature", "2 for each kind of fact", [(str(b), None) for b in BITS]),
             ("Creatures", "the size of the vocabulary", [(nice(c), None) for c in CREATURES])]
    for r, (name, note, values) in enumerate(dials):
        y = top + r * row_h
        write(page, 0, y, name, SMALL, "semibold", va="center")
        x = name_w
        for words, swatch in values:
            x = value_chip(x, y, words, swatch)
        write(page, note_x, y, note, SMALL, colour=SECOND, va="center")
    # Word use: how often the eight most used creatures appear, with equal and with unequal use.
    y = top + 4 * row_h
    write(page, 0, y, "Word use", SMALL, "semibold", va="center")
    x = name_w
    for use in USES:
        words = "equal" if use == "equal" else "unequal"
        w = text_width(words, SMALL, "medium") + 0.13 + key_bars * 0.034 + 0.06
        box(page, x, y - 0.085, w, 0.17, fill=PALE_GREY, round_=0.035)
        write(page, x + 0.065, y + 0.002, words, SMALL, "medium", va="center")
        bx = x + 0.065 + text_width(words, SMALL, "medium") + 0.07
        for k in range(key_bars):
            share = 1.0 if use == "equal" else use_shares[k]
            page.add_patch(Rectangle((bx + k * 0.034, y + 0.055 - 0.11 * share), 0.024, 0.11 * share,
                                     facecolor=SECOND, edgecolor="none"))
        x += w + 0.045
    write(page, note_x, y, "how often a creature appears", SMALL, colour=SECOND, va="center")
    # The sum.
    y = top + 5 * row_h + 0.05
    page.plot([0, 4.72], [y - 0.14, y - 0.14], color=GRID, linewidth=0.7)
    settings = len(COLUMNS) * len(BODIES) * len(BITS) * len(CREATURES) * len(USES)
    words = f"{len(COLUMNS)} × {len(BODIES)} × {len(BITS)} × {len(CREATURES)} × {len(USES)} = {settings} settings"
    write(page, 0, y, words, SMALL, "semibold", va="center")
    write(page, text_width(words, SMALL, "semibold") + 0.08, y,
          f"each trained {len(REPEATS)} times (its repeats): {nice(settings * len(REPEATS))} runs", SMALL,
          colour=SECOND, va="center")

    # Right: what every run has in common.
    x0 = 4.98
    page.plot([x0 - 0.13, x0 - 0.13], [0.02, height - 0.02], color=GRID, linewidth=0.7)
    write(page, x0, 0, "The same in every run", HEAD, "semibold")
    same = [["one learning rate"], ["one text for all models", "of a language"],
            ["one starting word table,", "cut to the columns"], ["2 layers, 2 heads"],
            ["512 sentences a step"], ["one rule that ends a run"]]
    y = 0.3
    for lines in same:
        tick_mark(page, x0 + 0.05, y, size=0.04, colour=SECOND, line=1.1)
        for k, line in enumerate(lines):
            write(page, x0 + 0.17, y + k * 0.135, line, SMALL, va="center")
        y += 0.135 * len(lines) + 0.082
    save(figure, "fig04_dials")


# ---------------------------------------------------------------------------------------------
# Figure 5: when a run ends, and how the 2,880 runs ended
# ---------------------------------------------------------------------------------------------

# Four real runs as examples, all with 1,024 creatures, body width 16, equal use, repeat 1:
# (table columns, bits), how the rule ended it, its two lines of label, and where the label
# stands (left or right end of the label at this step, and at this height).
RULE_EXAMPLES = [((16, 8), "learned", ["learned at the first look", "16 columns, 8 bits"], "right", 40_000, 123),
                 ((32, 32), "learned", ["learned at the fourth look", "32 columns, 32 bits"], "left", 54_000, 123),
                 ((16, 64), "limit", ["the limit: still gaining", "16 columns, 64 bits"], "right", 640_000, 87),
                 ((8, 128), "no more gain", ["no more gain, below 99", "8 columns, 128 bits"], "right", 320_000, 30.5)]


def facts_known(run):
    """Does the run know the facts: 99 or more of 100, the bar of "learned"?"""
    return run["facts"] >= LEARNED


# How a run ended, in four columns: the rule's three reasons, with "no more gain" split by
# whether the run knew the facts when it ended. (name of the column, its test)
ENDINGS = [("learned", lambda run: run["reason"] == "learned"),
           ("known", lambda run: run["reason"] == "no more gain" and facts_known(run)),
           ("below", lambda run: run["reason"] == "no more gain" and not facts_known(run)),
           ("limit", lambda run: run["reason"] == "limit")]


def endings(results):
    """How many runs ended at each look, in each column of ENDINGS: {(column, step): runs}."""
    counts = Counter()
    for run in results.runs:
        column = next(name for name, test in ENDINGS if test(run))
        counts[(column, run["ended"])] += 1
    return counts


def figure_rule(results):
    """Left: four real runs, each drawn up to the look at which the rule ended it. Right: how
    many of the 2,880 runs ended at each look, for each of the three reasons; "no more gain" is
    split into the runs that knew the facts and those that did not."""
    width, height = FULL, 2.94
    figure, page = page_of(width, height)
    write(page, 0, 0, "One rule ends every run", HEAD, "semibold")
    write(page, 0, 0.17, "four runs as examples: facts known during training", SMALL, colour=SECOND)
    left, top, plot_w, plot_h = 0.4, 0.42, 2.82, 1.94
    ax = plot_at(figure, left, top, plot_w, plot_h)
    plain_axes(ax, grid_values=(25, 50, 75, 100), y_limits=(13, 131))
    ax.set_xscale("log")
    ax.set_xlim(1000, 900_000)
    for look in LOOKS:  # the five looks
        ax.plot([look, look], [13, 106], color=AXIS, linewidth=0.8, zorder=1)
    for (columns, bits), reason, words, side, at_step, at_height in RULE_EXAMPLES:
        run = pick(results.runs, columns=columns, bits=bits, creatures=1024, body=16, use="equal", repeat=1)[0]
        assert run["reason"] == reason and ("below 99" not in words[0] or not facts_known(run)), run["name"]
        points = [(p["step"], p["facts"]) for p in run["curve"] if 1000 <= p["step"] <= run["ended"]]
        colour = COLUMN_TEAL[columns]
        ax.plot(*zip(*points), color=colour, linewidth=1.6, zorder=3)
        ax.plot([run["ended"]], [run["facts"]], marker="o", markersize=5.4, markerfacecolor=colour,
                markeredgecolor=PAPER, markeredgewidth=0.8, zorder=4, clip_on=False)
        write(ax, at_step, at_height, words[0], SMALL, "semibold", ha=side, va="center", zorder=5, clip_on=False,
              halo=True)
        write(ax, at_step, at_height - 9.5, words[1], SMALL, colour=SECOND, ha=side, va="center", zorder=5,
              clip_on=False, halo=True)
    ax.set_xticks([1000, 10_000] + LOOKS)  # every look has its grey line; every other one its label
    ax.set_xticklabels(["1K", "10K"] + [thousands(look) if k % 2 == 0 else "" for k, look in enumerate(LOOKS)])
    ax.minorticks_off()
    ax.set_yticks([25, 50, 75, 100])
    write(page, 0.07, top + (131 - 62.5) / 118 * plot_h, "facts known", SMALL, colour=SECOND, ha="center", va="center",
          rotation=90)
    for k, line in enumerate(["training steps so far (K = thousand)",
                              "each grey line is a look, at twice the steps of the last"]):
        write(page, left + plot_w / 2, top + plot_h + 0.2 + k * 0.14, line, SMALL, colour=SECOND, ha="center")
    write(ax, 1080, 23.5, "25: only guessing", SMALL, colour=SECOND, va="top")

    # Right: the count of runs that ended at each look, in each of the four columns.
    x0 = 3.5
    counts = endings(results)
    write(page, x0, 0, f"How the {nice(len(results.runs))} runs ended", HEAD, "semibold")
    write(page, x0, 0.17, "runs that ended at each look, by the reason", SMALL, colour=SECOND)
    label_w, row_h, ty, bar = 0.34, 0.26, 0.9, 0.26
    column_x = dict(zip([name for name, _ in ENDINGS], np.cumsum([x0 + label_w, 0.66, 0.74, 0.62])))
    # The heads of the columns: "no more gain" stands over its two halves.
    write(page, column_x["learned"], ty - 0.13, "learned", SMALL, "semibold", va="center")
    write(page, column_x["limit"], ty - 0.13, "the limit", SMALL, "semibold", va="center")
    write(page, column_x["known"], ty - 0.38, "no more gain", SMALL, "semibold", va="center")
    page.plot([column_x["known"], column_x["limit"] - 0.12], [ty - 0.27, ty - 0.27], color=AXIS, linewidth=0.7)
    write(page, column_x["known"], ty - 0.13, "facts known", SMALL, colour=SECOND, va="center")
    write(page, column_x["below"], ty - 0.13, "below 99", SMALL, colour=SECOND, va="center")
    most = max(counts.values())
    for name, _ in ENDINGS:
        x = column_x[name]
        for r, look in enumerate(LOOKS):
            y = ty + (r + 0.5) * row_h
            count = counts.get((name, look), 0)
            if count:
                page.add_patch(Rectangle((x, y - 0.045), bar * count / most, 0.09, facecolor=SECOND, edgecolor="none"))
                write(page, x + bar * count / most + 0.04, y, nice(count), SMALL, va="center")
            else:
                write(page, x, y, "–", SMALL, colour=MUTED, va="center")
        total = sum(count for (column, _), count in counts.items() if column == name)
        write(page, x, ty + 5 * row_h + 0.14, nice(total), SMALL, "semibold", va="center")
    for r, look in enumerate(LOOKS):
        write(page, x0 + label_w - 0.08, ty + (r + 0.5) * row_h, thousands(look), SMALL, colour=SECOND, ha="right",
              va="center")
    page.plot([x0, width], [ty + 5 * row_h + 0.03, ty + 5 * row_h + 0.03], color=GRID, linewidth=0.7)
    write(page, x0 + label_w - 0.08, ty + 5 * row_h + 0.14, "all", SMALL, colour=SECOND, ha="right", va="center")
    # How many runs knew the facts, whatever ended them.
    known = [run for run in results.runs if facts_known(run)]
    at_limit = sum(run["reason"] == "limit" for run in known)
    for k, line in enumerate([f"{nice(len(known))} of the {nice(len(results.runs))} runs knew the facts (99 or more):",
                              f"the first two columns, and {at_limit} of those at the limit"]):
        write(page, x0, ty + 5 * row_h + 0.36 + k * 0.14, line, SMALL, "medium" if k == 0 else "regular",
              INK if k == 0 else SECOND, va="center")
    save(figure, "fig05_rule")


# ---------------------------------------------------------------------------------------------
# Figure 6: the five predictions, word for word, each with its counts and its verdict
# ---------------------------------------------------------------------------------------------

def wrap(words, width, size=SMALL, weight="regular"):
    """Break a sentence into lines no wider than `width` inches, measured in the typeface."""
    lines, line = [], ""
    for word in words.split():
        longer = f"{line} {word}".strip()
        if line and text_width(longer, size, weight) > width:
            lines.append(line)
            line = word
        else:
            line = longer
    return lines + [line]


def scorecard_rows(found):
    """The five predictions in the words of v3/PLAN.md, in two groups, each prediction with the
    bars that show its counts. A bar is (what was counted, how many, of how many, the lowest and
    the highest count the prediction allows, the words for that). The 720 of C1 are comparisons
    of a setting with the same setting at the next larger number of columns."""
    c1, c2, a1, a2, a3 = (found[name] for name in ("C1", "C2", "A1", "A2", "A3"))
    return [
        ("From the owner’s claim", [
            ("C1", c1["held"], [
                ("For a body width, bits and number of creatures held fixed, the score never falls by 5 points "
                 "or more from one column count to the next larger one,",
                 ("comparisons fell by 5 or more", c1["falls"], c1["comparisons"],
                  (0, 0), "predicted: none")),
                ("and in at least a quarter of those fixed settings it is 10 points or more higher at 64 "
                 "columns than at 8.",
                 ("were 10 or more higher", c1["gained"], c1["cases"],
                  (c1["needed"], c1["cases"]), f"predicted: {c1['needed']} or more")),
            ]),
            ("C2", c2["held"], [
                ("Body width does not do this: in fewer than a tenth of the settings (columns, bits and "
                 "creatures held fixed) is the score 10 points or more higher at body width 128 than at body "
                 "width 4.",
                 ("were 10 or more higher at body width 128", c2["gained"], c2["cases"],
                  (0, c2["allowed"]), f"predicted: {c2['allowed']} or fewer")),
            ]),
        ]),
        ("The agent’s, from the second experiment", [
            ("A1", a1["held"], [
                ("Where bits per creature divided by columns is 1 or less, every setting ends as “learned”, "
                 "at every body width and every number of creatures.",
                 ("such settings ended as learned", a1["learned"], a1["cases"],
                  (a1["cases"], a1["cases"]), f"predicted: all {a1['cases']}")),
            ]),
            ("A2", a2["held"], [
                ("Where that figure is 4 or more, no setting with 1,024 creatures reaches 95.",
                 ("such settings reached 95", a2["reached"], a2["cases"], (0, 0), "predicted: none")),
            ]),
            ("A3", a3["held"], [
                ("Two settings with the same bits divided by columns (for example 16 bits in 8 columns and 64 "
                 "bits in 32 columns), the same body width and the same number of creatures end within 10 "
                 "points of each other.",
                 ("such pairs ended within 10 points", a3["within"], a3["cases"],
                  (a3["cases"], a3["cases"]), f"predicted: all {a3['cases']}")),
            ]),
        ]),
    ]


def figure_scorecard(found):
    """The five predictions in the words of the plan, each beside a bar of its count (dark), a
    line under the bar that marks where the prediction put that count, and the verdict."""
    groups = scorecard_rows(found)
    words_x, words_w = 0.36, 2.9             # the wording of a prediction
    bar_x, bar_w = 3.5, 1.68                 # the bar that shows its counts
    verdict_x = 5.5
    line_h, part_h, part_gap, row_gap, group_h = 0.137, 0.45, 0.12, 0.12, 0.22

    def height_of(lines):
        return max(len(lines) * line_h, part_h)

    # Measure first: how tall is the whole card?
    y = 0.48
    for _, predictions in groups:
        y += group_h
        for _, _, parts in predictions:
            y += sum(height_of(wrap(words, words_w)) + part_gap for words, _ in parts) - part_gap + row_gap
    width, height = FULL, y - row_gap + 0.05
    figure, page = page_of(width, height)

    def predicted_range(x_from, x_to, bar_top, ends):
        """Where a prediction put the count: a line just under the bar from x_from to x_to, with
        an upright mark through the bar at each end in `ends` (the ends that are a limit)."""
        under = bar_top + 0.085 + 0.04
        if x_to > x_from:
            page.plot([x_from, x_to], [under, under], color=INK, linewidth=1.0, solid_capstyle="butt")
        for x in ends:  # ink above and below the bar, and a notch of paper where the mark crosses a dark bar
            page.plot([x, x], [bar_top - 0.045, under], color=INK, linewidth=1.0, solid_capstyle="butt", zorder=5)
            page.plot([x, x], [bar_top, bar_top + 0.085], color=PAPER, linewidth=1.0, solid_capstyle="butt", zorder=6)

    write(page, 0, 0, "The five predictions, in the words of the plan", HEAD, "semibold")
    write(page, bar_x, 0, "What the runs gave", HEAD, "semibold")
    # The key of a bar, drawn once under the heading of the bars.
    page.add_patch(Rectangle((bar_x, 0.215), 0.3, 0.085, facecolor=PALE_GREY, edgecolor="none"))
    page.add_patch(Rectangle((bar_x, 0.215), 0.13, 0.085, facecolor=INK, edgecolor="none"))
    write(page, bar_x + 0.36, 0.26, "the count", SMALL, colour=SECOND, va="center")
    kx = bar_x + 0.36 + text_width("the count") + 0.2
    page.add_patch(Rectangle((kx, 0.215), 0.34, 0.085, facecolor=PALE_GREY, edgecolor="none"))
    predicted_range(kx + 0.12, kx + 0.34, 0.215, [kx + 0.12])
    write(page, kx + 0.4, 0.26, "where the prediction put it", SMALL, colour=SECOND, va="center")

    y = 0.48
    for group_name, predictions in groups:
        page.plot([0, width], [y - 0.02, y - 0.02], color=AXIS, linewidth=0.7)
        write(page, 0, y + 0.045, group_name, SMALL, "semibold", SECOND)
        y += group_h
        for name, held, parts in predictions:
            write(page, 0, y - 0.012, name, HEAD, "bold")
            bars = []  # the height of the middle of each bar of this prediction
            for words, (counted, count, total, (low, high), limit_words) in parts:
                lines = wrap(words, words_w)
                for k, line in enumerate(lines):
                    write(page, words_x, y + k * line_h, line, SMALL)
                # the count in words, the bar, and where the prediction put the count
                count_words = f"{nice(count)} of {nice(total)}"
                write(page, bar_x, y, count_words, SMALL, "semibold")
                write(page, bar_x + text_width(count_words, SMALL, "semibold") + 0.05, y, counted, SMALL)
                by = y + 0.2
                at = lambda number: bar_x + bar_w * number / total
                page.add_patch(Rectangle((bar_x, by), bar_w, 0.085, facecolor=PALE_GREY, edgecolor="none"))
                if count:
                    page.add_patch(Rectangle((bar_x, by), at(count) - bar_x, 0.085, facecolor=INK, edgecolor="none"))
                # A limit is an end of the predicted range that is not simply the end of the scale
                # (or the whole range, where the prediction names one count: "none", "all").
                limits = [at(low)] if low == high else [at(n) for n in (low, high) if 0 < n < total]
                predicted_range(at(low), at(high), by, limits)
                write(page, bar_x, by + 0.165, limit_words, SMALL, colour=SECOND)
                bars.append(by + 0.0425)
                y += height_of(lines) + part_gap
            y += row_gap - part_gap
            # The verdict, level with the bar; where a prediction has two parts, a bracket joins
            # their bars, and the verdict on both stands at its middle.
            vy = (bars[0] + bars[-1]) / 2
            if len(bars) > 1:
                bx = verdict_x - 0.13
                page.plot([bx - 0.05, bx, bx, bx - 0.05], [bars[0], bars[0], bars[-1], bars[-1]], color=SECOND,
                          linewidth=0.8, solid_capstyle="butt", solid_joinstyle="miter")
            if held:
                tick_mark(page, verdict_x + 0.06, vy)
            else:
                cross_mark(page, verdict_x + 0.06, vy)
            write(page, verdict_x + 0.18, vy, "held" if held else "did not hold", SMALL, "semibold", va="center")
    save(figure, "fig06_predictions")


# ---------------------------------------------------------------------------------------------
# Figure 7: every setting against the number of table columns. No line goes down.
# ---------------------------------------------------------------------------------------------

def figure_by_columns(results, found):
    """A panel for every number of creatures (down) and every number of bits (across). In a
    panel, one line for each of the 12 cases of C1 that it holds: 6 body widths, each with equal
    use (solid) and unequal use (dashed). A line runs through the middle one of the three
    repeats at each number of table columns; the small dots are the three repeats themselves.
    The number in a panel counts its lines that end 10 points or more higher at 64 columns
    than at 8: the second half of C1 (the caption says so)."""
    width, height = FULL, 3.98
    figure, page = page_of(width, height)
    left, gap, top, row_gap = 0.78, 0.11, 0.78, 0.14
    panel_w = (width - left - (len(BITS) - 1) * gap) / len(BITS)
    panel_h = 0.615
    gains = found["C1"]["gains"]  # {(body, bits, creatures, use): score at 64 columns minus score at 8}
    write(page, 0, 0, "Facts known against the number of table columns", HEAD, "semibold")
    # the key: what a line is, the two word uses, the repeats, and the number in a panel
    x, y = 0, 0.27
    for use in USES:
        words = "equal use" if use == "equal" else "unequal use"
        page.plot([x, x + 0.3], [y, y], color=INK, linewidth=0.8, linestyle=use_style(use))
        write(page, x + 0.36, y, words, SMALL, colour=SECOND, va="center")
        x += 0.36 + text_width(words) + 0.22
    for shift in (-0.05, 0, 0.05):
        page.plot([x + 0.06 + shift], [y], marker="o", markersize=2.2, color=SECOND, alpha=0.7, markeredgewidth=0)
    write(page, x + 0.2, y, "the three repeats of a setting; its line runs through the middle one", SMALL,
          colour=SECOND, va="center")
    write(page, 0, y + 0.16, "A panel has 12 lines: 6 body widths, each with both word uses.   "
          "100: knows every fact; 25: only guessing.", SMALL, colour=SECOND, va="center")
    for row, creatures in enumerate(CREATURES):
        y0 = top + row * (panel_h + row_gap)
        write(page, 0, y0 + panel_h / 2 - 0.07, nice(creatures), SMALL, "semibold", va="center")
        write(page, 0, y0 + panel_h / 2 + 0.07, "creatures", SMALL, colour=SECOND, va="center")
        for column, bits in enumerate(BITS):
            ax = plot_at(figure, left + column * (panel_w + gap), y0, panel_w, panel_h)
            plain_axes(ax, grid_values=(25, 50, 75, 100), y_limits=(17, 107))
            for body in BODIES:
                for use in USES:
                    repeats = [results.by_setting[(columns, body, bits, creatures, use)] for columns in COLUMNS]
                    for place, runs in enumerate(repeats):  # the three repeats, side by side
                        ax.plot([place - 0.12, place, place + 0.12], [run["facts"] for run in runs], linestyle="none",
                                marker="o", markersize=1.9, color=SECOND, alpha=0.45, markeredgewidth=0, zorder=2)
                    ax.plot(range(len(COLUMNS)), [middle(run["facts"] for run in runs) for runs in repeats],
                            color=INK, linewidth=0.75, alpha=0.8, linestyle=use_style(use), zorder=3)
            rose = sum(gains[(body, bits, creatures, use)] >= RISE for body in BODIES for use in USES)
            write(ax, len(COLUMNS) - 0.72, 21, f"{rose} of {len(BODIES) * len(USES)}", SMALL, "medium", ha="right",
                  va="bottom", zorder=5, halo=True)
            ax.set_xlim(-0.35, len(COLUMNS) - 0.65)
            ax.set_xticks(range(len(COLUMNS)))
            ax.set_xticklabels([str(columns) for columns in COLUMNS] if row == len(CREATURES) - 1 else [])
            ax.set_yticks([25, 50, 75, 100])
            if column > 0:
                ax.set_yticklabels([])
            if row == 0:
                write(page, left + column * (panel_w + gap), top - 0.12, f"{bits} bits", HEAD, "semibold", va="center")
    write(page, left + (width - left) / 2, height - 0.02, "table columns", SMALL, colour=SECOND, ha="center", va="bottom")
    save(figure, "fig07_by_columns")


# ---------------------------------------------------------------------------------------------
# The frame of the two maps on the plain dials (Figures 8 and 10): down, the table columns;
# across, the bits per creature; a panel for each number of creatures; a row of panels for each
# word use; and the stepped line that marks "as many bits as columns"
# ---------------------------------------------------------------------------------------------

MAP_LEFT, MAP_GAP, MAP_TILE_H = 0.92, 0.14, 0.2


def stepped_line_key(page, x, y):
    """The key of the stepped line of a map. Returns where it ends."""
    page.plot([x, x + 0.12, x + 0.12, x + 0.24], [y - 0.05, y - 0.05, y + 0.05, y + 0.05], color=INK, linewidth=1.3,
              solid_capstyle="butt", solid_joinstyle="miter")
    words = "left of the stepped line: no more bits than columns"
    write(page, x + 0.3, y, words, SMALL, colour=SECOND, va="center")
    return x + 0.3 + text_width(words)


def dial_map(page, width, tops, draw_tile):
    """Draws the frame of a map and calls draw_tile(x, y, tile width, tile height, columns,
    bits, creatures, use) for each of its 160 tiles. tops are the heights of the two rows of
    panels (equal use, unequal use). Returns the height just under the axis labels."""
    panel_w = (width - MAP_LEFT - 3 * MAP_GAP) / 4
    tile_w = panel_w / len(BITS)
    for row, (use, top) in enumerate(zip(USES, tops)):
        use_heading(page, 0, top - 0.13, use)
        for r, columns in enumerate(COLUMNS):  # the rows of every panel: the table columns
            y = top + (r + 0.5) * MAP_TILE_H
            page.add_patch(Rectangle((0, y - 0.032), 0.16, 0.064, facecolor=COLUMN_TEAL[columns], edgecolor="none"))
            write(page, MAP_LEFT - 0.08, y, f"{columns} columns", SMALL, colour=SECOND, ha="right", va="center")
        for column, creatures in enumerate(CREATURES):
            x0 = MAP_LEFT + column * (panel_w + MAP_GAP)
            if row == 0:
                write(page, x0 + panel_w / 2, top - 0.36, f"{nice(creatures)} creatures", HEAD, "semibold",
                      ha="center", va="center")
            for r, columns in enumerate(COLUMNS):
                for c, bits in enumerate(BITS):
                    draw_tile(x0 + c * tile_w, top + r * MAP_TILE_H, tile_w, MAP_TILE_H, columns, bits, creatures, use)
            # The stepped line: in the row of 8 columns it stands after 8 bits, in the row of 16
            # columns after 16 bits, and so on.
            xs, ys = [], []
            for r, columns in enumerate(COLUMNS):
                edge = x0 + (BITS.index(columns) + 1) * tile_w
                xs += [edge, edge]
                ys += [top + r * MAP_TILE_H, top + (r + 1) * MAP_TILE_H]
            page.plot(xs, ys, color=INK, linewidth=1.3, solid_capstyle="butt", solid_joinstyle="miter", zorder=5)
            for c, bits in enumerate(BITS):
                write(page, x0 + (c + 0.5) * tile_w, top + 4 * MAP_TILE_H + 0.04, str(bits), SMALL, colour=SECOND,
                      ha="center")
    bottom = tops[-1] + 4 * MAP_TILE_H + 0.2
    write(page, MAP_LEFT + 2 * panel_w + 1.5 * MAP_GAP, bottom, "bits per creature", SMALL, colour=SECOND, ha="center")
    return bottom + 0.14


# ---------------------------------------------------------------------------------------------
# Figure 8: what a wider body gained, for every combination of the other four dials (C2)
# ---------------------------------------------------------------------------------------------

def figure_body_gain(results, found):
    """A map of the 160 settings of C2. A tile carries the points of facts known that body width
    128 scored above body width 4 (each the middle of three repeats). The tiles that count
    against C2 (10 points or more) are outlined in ink; where the narrowest body already scored
    over 90, and so left no room for such a gain, the number is grey."""
    gains = found["C2"]["gains"]  # {(columns, bits, creatures, use): score at body 128 minus score at body 4}
    width = FULL
    tops = [1.04, 2.22]
    figure, page = page_of(width, 3.42)
    title = "Points of facts known gained from body width 4 to body width 128"
    write(page, 0, 0, title, HEAD, "semibold")
    note = "each score: the middle of three repeats"
    assert text_width(title, HEAD, "semibold") + 0.1 + text_width(note) <= width
    write(page, text_width(title, HEAD, "semibold") + 0.1, 0.012, note, SMALL, colour=SECOND)
    # The key: an outlined tile, a grey number, the stepped line.
    y = 0.3
    page.add_patch(Rectangle((0.02, y - 0.075), 0.24, 0.15, facecolor=PALE_GREY, edgecolor=INK, linewidth=1.3))
    write(page, 0.14, y + 0.004, "+12", SMALL, "semibold", ha="center", va="center_baseline")
    words = f"{RISE} points or more: counts against C2 ({found['C2']['gained']} of {found['C2']['cases']})"
    write(page, 0.34, y, words, SMALL, colour=SECOND, va="center")
    stepped_line_key(page, 0.34 + text_width(words) + 0.26, y)
    y = 0.47
    page.add_patch(Rectangle((0.02, y - 0.075), 0.24, 0.15, facecolor=PALE_GREY, edgecolor="none"))
    write(page, 0.14, y + 0.004, "+2", SMALL, colour=MUTED, ha="center", va="center_baseline")
    write(page, 0.34, y, f"a grey number: the narrowest body already scored over {ROOM}, so there was no room for a gain of {RISE}",
          SMALL, colour=SECOND, va="center")

    def draw_tile(x, y, w, h, columns, bits, creatures, use):
        gain = gains[(columns, bits, creatures, use)]
        counts = gain >= RISE
        room = results.score[(columns, 4, bits, creatures, use)] <= ROOM
        tile(page, x, y, w, h, PALE_GREY)
        if counts:
            page.add_patch(Rectangle((x + 0.024, y + 0.024), w - 0.048, h - 0.048, facecolor="none", edgecolor=INK,
                                     linewidth=1.3, zorder=4))
        words = f"{gain:+.0f}".replace("-", "−")
        words = "0" if words in ("+0", "−0") else words
        write(page, x + w / 2, y + h / 2 + 0.004, words, SMALL, "semibold" if counts else "regular",
              INK if room else MUTED, ha="center", va="center_baseline")

    bottom = dial_map(page, width, tops, draw_tile)
    assert bottom <= 3.42
    save(figure, "fig08_body_gain")


# ---------------------------------------------------------------------------------------------
# Figure 9: facts known against the size of the whole model. Where a model with 8 columns and
# one with 16 are nearly the same size, the one with 16 columns knows more.
# ---------------------------------------------------------------------------------------------

def same_size_pairs(results):
    """Every pair of settings in the same language (bits, creatures, word use) with different
    numbers of table columns in which the model with MORE columns has at most 15 % more trained
    numbers than the other (it may have fewer), and at least one of the two scores below 99, so
    that there is something to compare. Returns [(setting with fewer columns, setting with more)]."""
    pairs = []
    for bits in BITS:
        for creatures in CREATURES:
            for use in USES:
                language = [(columns, body, bits, creatures, use) for columns in COLUMNS for body in BODIES]
                for one, other in combinations(language, 2):
                    if one[0] == other[0]:
                        continue
                    fewer, more = sorted((one, other))  # by table columns
                    if size_of(more) <= SAME_SIZE * size_of(fewer) and min(results.score[fewer], results.score[more]) < LEARNED:
                        pairs.append((fewer, more))
    return pairs


def figure_size(results):
    """Three languages, a panel each. In a panel: facts known against the trained numbers of
    the whole model. A line is one number of table columns; its six points are the six body
    widths, from 4 at the left to 128 at the right. A point is the middle of three repeats; the
    small dots are the repeats. The two ringed points are the widest body with 8 columns and the
    narrowest body with 16: the note beside them gives their sizes and scores."""
    width, height = FULL, 2.92
    figure, page = page_of(width, height)
    left, gap, top, panel_h = 0.40, 0.2, 0.92, 1.5
    panel_w = (width - left - 0.06 - (len(SIZE_PANELS) - 1) * gap) / len(SIZE_PANELS)
    low, high = 3_600, 230_000
    write(page, 0, 0, "Facts known against the size of the whole model", HEAD, "semibold")
    x = column_key(page, 0, 0.3)
    write(page, x + 0.12, 0.3, "the six points of a line: body width 4 (left) to 128 (right)", SMALL,
          colour=SECOND, va="center")
    x = 0
    for shift in (-0.05, 0, 0.05):
        page.plot([x + 0.06 + shift], [0.46], marker="o", markersize=2.3, color=SECOND, markeredgewidth=0)
    words = "the three repeats; the line runs through the middle one"
    write(page, x + 0.2, 0.46, words, SMALL, colour=SECOND, va="center")
    x += 0.2 + text_width(words) + 0.28
    page.plot([x + 0.05], [0.46], marker="o", markersize=7.5, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.9)
    write(page, x + 0.17, 0.46, "the two models of the note", SMALL, colour=SECOND, va="center")
    for column, (creatures, bits, use) in enumerate(SIZE_PANELS):
        x0 = left + column * (panel_w + gap)
        ax = plot_at(figure, x0, top, panel_w, panel_h)
        plain_axes(ax, grid_values=(25, 50, 75, 100), y_limits=(15, 108))
        ax.set_xscale("log")
        ax.set_xlim(low, high)
        write(page, x0, top - 0.27, f"{nice(creatures)} creatures, {bits} bits", HEAD, "semibold", va="center")
        write(page, x0, top - 0.11, f"{use} use", SMALL, colour=SECOND, va="center")
        # The two neighbours in size: 8 columns with the widest body, 16 columns with the narrowest.
        wide, narrow = (8, BODIES[-1], bits, creatures, use), (16, BODIES[0], bits, creatures, use)
        meet = size_of(narrow) <= SAME_SIZE * size_of(wide)   # are the two nearly the same size?
        for columns in COLUMNS:
            colour = COLUMN_TEAL[columns]
            settings = [(columns, body, bits, creatures, use) for body in BODIES]
            sizes = [size_of(setting) for setting in settings]
            for size, setting in zip(sizes, settings):  # the three repeats, side by side
                ax.plot([size * 0.97, size, size * 1.03], [run["facts"] for run in results.by_setting[setting]],
                        linestyle="none", marker="o", markersize=2.3, color=colour, markeredgewidth=0, zorder=4)
            middles = [results.score[setting] for setting in settings]
            ax.plot(sizes, middles, color=colour, linewidth=1.5, linestyle=use_style(use), zorder=3)
            # The line is named under its right-hand end; where the sizes do not meet the lines
            # climb steeply, and a name stands to the left of the line's start instead.
            if meet:
                write(ax, sizes[-1], middles[-1] - 5.5, str(columns), SMALL, "semibold", ha="center", va="top", zorder=5)
            else:
                write(ax, sizes[0] * 0.82, middles[0], str(columns), SMALL, "semibold", ha="right", va="center", zorder=5)
        for setting in (wide, narrow):
            ax.plot([size_of(setting)], [results.score[setting]], marker="o", markersize=7.5, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=0.9, zorder=6)
        lines = [(f"8 columns, body {BODIES[-1]}:", "medium", INK),
                 (f"{nice(size_of(wide))} numbers, {results.score[wide]:.0f} facts", "regular", SECOND),
                 (f"16 columns, body {BODIES[0]}:", "medium", INK),
                 (f"{nice(size_of(narrow))} numbers, {results.score[narrow]:.0f} facts", "regular", SECOND)]
        if meet:                                          # sizes meet: the note stands at the lower right
            note_x, heights = 11_500, (62, 53, 41, 32)
        else:                                             # sizes do not meet: the note stands at the upper left
            note_x, heights = low * 1.08, (103, 94)
            lines = [("the sizes do not meet:", "medium", INK),
                     (f"{nice(size_of(wide))} against {nice(size_of(narrow))}", "regular", SECOND)]
        for (words, weight, colour), at in zip(lines, heights):
            write(ax, note_x, at, words, SMALL, weight, colour, va="center", zorder=5)
        ax.set_xticks([5_000, 10_000, 20_000, 40_000, 80_000, 160_000])
        ax.set_xticklabels(["5K", "10K", "20K", "40K", "80K", "160K"])
        ax.minorticks_off()
        ax.set_yticks([25, 50, 75, 100])
        if column > 0:
            ax.set_yticklabels([])
    write(page, 0.07, top + panel_h / 2, "facts known", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left + (width - left) / 2, height - 0.02,
          "trained numbers in the whole model (K = thousand); each mark is twice the one before", SMALL,
          colour=SECOND, ha="center", va="bottom")
    save(figure, "fig09_size")


# ---------------------------------------------------------------------------------------------
# Figure 10: the typical score for every word use and number of columns, bits and creatures,
# with the stepped line that marks one bit for each table column
# ---------------------------------------------------------------------------------------------

def figure_load_map(results):
    """Tiles: down, the table columns; across, the bits per creature; a panel for each number
    of creatures and a row of panels for each word use. A tile is the middle of the 18 runs of
    its place (6 body widths, 3 repeats). The stepped line divides each panel: to its left a
    word's row has at least as many numbers as the creature has bits."""
    width = FULL
    tops = [0.82, 2.04]
    figure, page = page_of(width, 3.24)
    title = "Facts known by the typical model, for each word use"
    write(page, 0, 0, title, HEAD, "semibold")
    write(page, text_width(title, HEAD, "semibold") + 0.1, 0.012, "a tile: the middle of 18 runs (6 body widths × 3 repeats)",
          SMALL, colour=SECOND)
    x = scale_key(page, 0, 0.3)
    stepped_line_key(page, x + 0.3, 0.3)

    def draw_tile(x, y, w, h, columns, bits, creatures, use):
        score = results.typical(columns, bits, creatures, use)
        tile(page, x, y, w, h, tile_colour(score))
        tile_number(page, x + w / 2, y + h / 2, f"{score:.0f}", score)

    bottom = dial_map(page, width, tops, draw_tile)
    assert bottom <= 3.24
    save(figure, "fig10_load_map")


# ---------------------------------------------------------------------------------------------
# Figure 11: runs that stopped one whole kind of fact short
# ---------------------------------------------------------------------------------------------

KIND_NAMES = ["colour", "food", "home", "waking"]  # the four kinds of fact at 8 bits, in the language's order
GUESS_LEVEL, KNOWN_LEVEL = 30, 95                  # a kind "at a guess" and a kind "known"


def kinds_short(run):
    """How many whole kinds of fact a run is short: the number of kinds it only guesses (30 or
    less of 100) when it knows every other kind (95 or more). 0 for every other run."""
    guessed = sum(score <= GUESS_LEVEL for score in run["by_kind"])
    known = sum(score >= KNOWN_LEVEL for score in run["by_kind"])
    return guessed if guessed and guessed + known == len(run["by_kind"]) else 0


def figure_plateau(results):
    """The 18 runs of one group of settings (8 columns, 8 bits, 1,024 creatures, equal use): for
    each run the facts it knew of each of the four kinds, and its score, which is their average."""
    group = pick(results.runs, **STUCK)
    width, height = FULL, 1.78
    figure, page = page_of(width, height)
    write(page, 0, 0, f"{STUCK['columns']} columns, {STUCK['bits']} bits, {nice(STUCK['creatures'])} creatures, "
          f"{STUCK['use']} use: all {len(group)} runs, kind by kind", HEAD, "semibold")
    write(page, 0, 0.19, "The four tiles of a run: the facts it knew of each kind (" + ", ".join(KIND_NAMES)
          + "), out of 100; 25 is a guess.", SMALL, colour=SECOND)
    label_w, tile_w, row_h, top = 0.86, 0.3, 0.2, 0.56
    group_w = (width - label_w) / len(REPEATS)
    score_x = 4 * tile_w + 0.46  # where a run's score ends, from the left edge of its tiles
    for place, repeat in enumerate(REPEATS):
        x0 = label_w + place * group_w
        write(page, x0, top - 0.14, f"repeat {repeat}", SMALL, "semibold", va="center")
        write(page, x0 + score_x, top - 0.14, "score", SMALL, colour=SECOND, ha="right", va="center")
    write(page, 0, top - 0.14, "body width", SMALL, colour=SECOND, va="center")
    for line, body in enumerate(BODIES):
        y = top + line * row_h
        write(page, 0.56, y + row_h / 2 + 0.004, str(body), SMALL, colour=SECOND, ha="right", va="center_baseline")
        for place, repeat in enumerate(REPEATS):
            run = pick(group, body=body, repeat=repeat)[0]
            x0 = label_w + place * group_w
            for k, known in enumerate(run["by_kind"]):
                tile(page, x0 + k * tile_w, y, tile_w, row_h, tile_colour(known))
                tile_number(page, x0 + (k + 0.5) * tile_w, y + row_h / 2, f"{known:.0f}", known)
            write(page, x0 + score_x, y + row_h / 2 + 0.004, f"{run['facts']:.1f}", SMALL, "semibold", ha="right",
                  va="center_baseline")
    save(figure, "fig11_plateau")


# ---------------------------------------------------------------------------------------------
# What a run was doing when the rule ended it as "no more gain" (finding 6, in words)
# ---------------------------------------------------------------------------------------------

def stood_below(results):
    """The runs that the rule ended as "no more gain" below 99 facts known. (The others it
    ended so knew the facts: their facts could not rise while their questions were still short.)"""
    return [run for run in results.runs if run["reason"] == "no more gain" and not facts_known(run)]


def last_gain(run):
    """The points of facts known a "no more gain" run gained over its last doubling: from half
    its length to its end (the trainer recorded both in the run's summary)."""
    return run["stop"]["facts_whole"] - run["stop"]["facts_half"]


# ---------------------------------------------------------------------------------------------
# Figure 12: the runs that were trained on after the rule had ended them
# ---------------------------------------------------------------------------------------------

TOO_EARLY = 1.0  # a carried-on run that gained this many points of facts known was ended too early


def carried_runs(results):
    """The marked runs that the rule ended as "no more gain" and that were then trained on."""
    return [run for run in results.runs if run["carried"]]


def carried_groups(results):
    """The carried-on runs in three groups: those that already knew the facts when the rule
    ended them (no room to gain a point), those below 99 that then gained less than a point, and
    those below 99 that gained a point or more. Each group in the order of its gain."""
    carried = sorted(carried_runs(results), key=lambda run: run["carried"]["gain"])
    knew = [run for run in carried if facts_known(run)]
    assert all(run["carried"]["gain"] < TOO_EARLY for run in knew)
    level = [run for run in carried if not facts_known(run) and run["carried"]["gain"] < TOO_EARLY]
    gained = [run for run in carried if not facts_known(run) and run["carried"]["gain"] >= TOO_EARLY]
    return knew, level, gained


def figure_carried(results):
    """Left: one square for every run that was trained on after the rule had ended it, in three
    groups; a dot marks a run that was one kind of fact short. Right: those three runs, a small
    plot each: facts known during training, in ink up to the rule's stop and in orange after it."""
    knew, level, gained = carried_groups(results)
    carried = knew + level + gained
    short = sorted((run for run in carried if kinds_short(run)), key=lambda run: run["body"])
    width, height = FULL, 2.36
    figure, page = page_of(width, height)

    # Left: one square for each carried-on run, group by group.
    write(page, 0, 0, f"The {len(carried)} runs trained on after “no more gain”", HEAD, "semibold")
    write(page, 0, 0.19, "one square is one run", SMALL, colour=SECOND)
    square, per_row, label_x = 0.14, 11, 1.68
    others = [run["carried"]["gain"] for run in gained[:-1]]
    groups = [(knew, GRID, [f"{len(knew)} already knew the", "facts (99 or more):", "no room to gain a point"]),
              (level, MUTED, [f"{len(level)} stood below 99 and", "gained less than 1 point"]),
              (gained, CARRIED, [f"{len(gained)} stood below 99 and", "gained 1 point or more:",
                                 f"{in_words(len(others))} by {min(others):.1f} to {max(others):.1f}, one by "
                                 f"{gained[-1]['carried']['gain']:.1f}"])]
    y = 0.46
    for runs, colour, lines in groups:
        for i, run in enumerate(runs):
            sx, sy = (i % per_row) * square, y + (i // per_row) * square
            tile(page, sx, sy, square, square, colour, gap=0.012)
            if kinds_short(run):
                page.add_patch(Circle((sx + square / 2, sy + square / 2), 0.026, facecolor=INK, edgecolor="none"))
        for k, line in enumerate(lines):
            write(page, label_x, y + 0.07 + k * 0.135, line, SMALL, "medium" if k == 0 else "regular",
                  INK if k == 0 else SECOND, va="center")
        rows = -(-len(runs) // per_row)
        y += max(rows * square, len(lines) * 0.135) + 0.11
    tile(page, 0, y - 0.02, square, square, GRID, gap=0.012)
    page.add_patch(Circle((square / 2, y - 0.02 + square / 2), 0.026, facecolor=INK, edgecolor="none"))
    write(page, 0.2, y + 0.05, "a run one kind of fact short (finding 5)", SMALL, colour=SECOND, va="center")
    assert y + 0.12 <= height

    # Right: the three carried-on runs that were one kind short, each in a small plot.
    x0 = 3.28
    assert len(short) == 3 and len({(run["columns"], run["bits"], run["creatures"], run["use"]) for run in short}) == 1
    one = short[0]
    write(page, x0, 0, "The three of them that were one kind short", HEAD, "semibold")
    write(page, x0, 0.19, f"{one['columns']} columns, {one['bits']} bits, {nice(one['creatures'])} creatures, {one['use']} use",
          SMALL, colour=SECOND)
    left, top, panel_h, gap = x0 + 0.42, 0.72, 1.06, 0.12
    panel_w = (width - left - 0.04 - 2 * gap) / 3
    for place, run in enumerate(short):
        px = left + place * (panel_w + gap)
        ax = plot_at(figure, px, top, panel_w, panel_h)
        plain_axes(ax, grid_values=(25, 50, 75, 100), y_limits=(15, 119))
        ax.set_xscale("log")
        ax.set_xlim(900, 420_000)
        write(page, px, top - 0.13, f"body width {run['body']}", SMALL, "semibold", va="center")
        before = [(p["step"], p["facts"]) for p in run["curve"] if 1000 <= p["step"] <= run["ended"]]
        after = [(p["step"], p["facts"]) for p in run["curve"] if p["step"] >= run["ended"]]
        ax.plot(*zip(*before), color=INK, linewidth=1.6, zorder=3)
        ax.plot(*zip(*after), color=CARRIED, linewidth=1.9, zorder=4)
        ax.plot([run["ended"]], [run["facts"]], marker="o", markersize=5, markerfacecolor=INK, markeredgecolor=PAPER,
                markeredgewidth=0.8, zorder=5)
        ax.plot([after[-1][0]], [after[-1][1]], marker="o", markersize=5, markerfacecolor=CARRIED,
                markeredgecolor=PAPER, markeredgewidth=0.8, zorder=5, clip_on=False)
        # the facts known at the rule's stop (under the line) and after the further training (at its end)
        write(ax, run["ended"], run["facts"] - 8, f"{run['facts']:.1f}", SMALL, "medium", ha="center", va="top")
        end = run["carried"]["facts"]
        write(ax, after[-1][0] * 1.25, end + 6, f"{end:.1f}" if end < 99.95 else "100", SMALL, "medium", ha="right",
              va="bottom")
        ax.set_xticks([1000, 10_000, 100_000])
        ax.set_xticklabels(["1K", "10K", "100K"])
        ax.minorticks_off()
        ax.set_yticks([25, 50, 75, 100])
        if place > 0:
            ax.set_yticklabels([])
    write(page, x0 + 0.06, top + panel_h / 2, "facts known", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left + (width - left) / 2, top + panel_h + 0.2, "training steps so far (K = thousand)", SMALL,
          colour=SECOND, ha="center")
    y = top + panel_h + 0.47
    page.plot([left + 0.04], [y], marker="o", markersize=5, markerfacecolor=INK, markeredgecolor=PAPER, markeredgewidth=0.8)
    write(page, left + 0.13, y, "the rule ended the run", SMALL, colour=SECOND, va="center")
    kx = left + 0.13 + text_width("the rule ended the run") + 0.2
    page.plot([kx, kx + 0.26], [y, y], color=CARRIED, linewidth=1.9)
    write(page, kx + 0.32, y, "trained on", SMALL, colour=SECOND, va="center")
    save(figure, "fig12_carried_on")


# ---------------------------------------------------------------------------------------------
# Figure 13: the four parts of the test, for each number of table columns
# ---------------------------------------------------------------------------------------------

SCORE_PANELS = [("facts", ["Facts known", ""], GUESS), ("questions_one", ["Questions about", "one creature"], 0),
                ("questions_new", ["Questions about", "new combinations"], 0), ("not_known", ["“Not” known", ""], GUESS)]
SCORES_CREATURES = 1024  # the vocabulary the picture is drawn for
SCORES_USE = "unequal"   # ... and the word use: the one in which no run stood one kind of fact short


def figure_scores(results):
    """A panel for each part of the test: the middle of the 18 runs with 1,024 creatures and
    unequal use, for each number of table columns and bits. The lines are named where they end
    in the first panel."""
    creatures, use = SCORES_CREATURES, SCORES_USE
    width, height = FULL, 1.9
    figure, page = page_of(width, height)
    left, gap, top, panel_h, names_w = 0.40, 0.17, 0.8, 0.78, 0.66   # names_w: room for the names of the lines
    panel_w = (width - left - names_w - 0.05 - 2 * gap) / 4
    panel_x = [left, left + panel_w + names_w] + [left + panel_w + names_w + k * (panel_w + gap) for k in (1, 2)]
    write(page, 0, 0, f"The four parts of the test, with {nice(creatures)} creatures and {use} use", HEAD, "semibold")
    x = column_key(page, 0, 0.3)
    write(page, x + 0.1, 0.3, "a point: the middle of 18 runs (6 body widths × 3 repeats)", SMALL,
          colour=SECOND, va="center")
    for column, (score, title, guess) in enumerate(SCORE_PANELS):
        x0 = panel_x[column]
        ax = plot_at(figure, x0, top, panel_w, panel_h)
        plain_axes(ax, grid_values=(0, 25, 50, 75, 100), y_limits=(-9, 107))
        for k, line in enumerate(title):
            write(page, x0, top - 0.3 + k * 0.14 + (0.14 if not title[1] else 0), line, SMALL, "semibold", va="center")
        ends = {}
        for columns in COLUMNS:
            values = [results.typical(columns, bits, creatures, use, score=score) for bits in BITS]
            ax.plot(range(len(BITS)), values, color=COLUMN_TEAL[columns], linewidth=1.6, linestyle=use_style(use),
                    zorder=3)
            ax.plot(range(len(BITS)), values, linestyle="none", marker="o", markersize=4.2,
                    markerfacecolor=COLUMN_TEAL[columns], markeredgecolor=PAPER, markeredgewidth=0.7, zorder=4,
                    clip_on=False)
            ends[columns] = values[-1]
        if column == 0:  # the lines are named where they end
            for columns, at in end_labels(ends, least_apart=17).items():
                write(ax, len(BITS) - 0.45, at, f"{columns} columns", SMALL, va="center", clip_on=False)
        # the score of a guess: named above its line where the guess is 0, below it where it is 25
        ax.axhline(guess, color=SECOND, linewidth=0.8, zorder=1)
        write(ax, -0.4, guess + (3 if guess == 0 else -3), f"a guess: {guess}", SMALL, colour=SECOND,
              va="bottom" if guess == 0 else "top", zorder=2)
        ax.set_xlim(-0.45, len(BITS) - 0.55)
        ax.set_xticks(range(len(BITS)))
        ax.set_xticklabels([str(bits) for bits in BITS])
        ax.set_yticks([0, 25, 50, 75, 100])
        if column > 0:
            ax.set_yticklabels([])
    write(page, 0.07, top + panel_h / 2, "score", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left + (width - left) / 2, height - 0.02, "bits per creature", SMALL, colour=SECOND, ha="center",
          va="bottom")
    save(figure, "fig13_scores")


# ---------------------------------------------------------------------------------------------
# Figure 14: the rate check. The best repeat of every setting again at a third of the rate.
# ---------------------------------------------------------------------------------------------

RATE_STAGE = "rate_check_third"  # the folder of the stage in runs_v3 (v3/grid.py)
RATE_BAND = 5    # a setting moved "clearly" when its facts known changed by this many points or more
RATE_MARGIN = 1  # ... and "a little" when they changed by this many
AT_THE_TOP = 99.5  # a score that is printed as 100
PREDICTION_NAMES = ["C1", "C2", "A1", "A2", "A3"]


def read_rate_check(results):
    """The 960 runs of the rate check, each beside the counted run it repeats: the same setting
    and the same repeat, at the chosen rate. Returns [(counted run, run at a third of the rate)]."""
    pairs = []
    for run in read_stage(RATE_STAGE):
        match = [counted for counted in results.by_setting.get(setting_of(run), []) if counted["repeat"] == run["repeat"]]
        if match:
            pairs.append((match[0], run))
    assert len(pairs) == len(results.by_setting), "runs_v3/rate_check_third is not complete"
    # The counted run of every pair is the best repeat of its setting, by the plan's rule: most
    # facts known, then the better questions about one creature, then the lower repeat.
    for counted, _ in pairs:
        best = min(results.by_setting[setting_of(counted)],
                   key=lambda run: (-run["facts"], -run["questions_one"], run["repeat"]))
        assert best is counted
    return pairs


def single_run_verdicts(runs):
    """The five predictions applied to one run for each setting (not to the middle of three)."""
    return verdicts_of({setting_of(run): run["facts"] for run in runs},
                       {setting_of(run): int(run["reason"] == "learned") for run in runs}, enough=1)


def against_the_setting(results, pairs, score="facts"):
    """For every setting: the score of its run at a third of the rate, minus the setting's score
    at the chosen rate (the middle of its three counted repeats). This is the fair yardstick: the
    run that was repeated is the BEST of the three, so against that run a fall is to be expected
    at any rate. Returns {setting: difference}."""
    return {setting_of(counted): lower[score] - middle(run[score] for run in results.by_setting[setting_of(counted)])
            for counted, lower in pairs}


def against_the_run(pairs, score="facts"):
    """For every setting: the score of its run at a third of the rate, minus the score of the
    very run it repeats (the best of the setting's three). Returns {setting: difference}."""
    return {setting_of(counted): lower[score] - counted[score] for counted, lower in pairs}


def by_columns(change):
    """How many settings ended a little higher, and a little lower, for each number of table
    columns: {columns: (higher, lower)}, from the differences of one of the two yardsticks."""
    return {columns: (sum(c >= RATE_MARGIN for setting, c in change.items() if setting[0] == columns),
                      sum(c <= -RATE_MARGIN for setting, c in change.items() if setting[0] == columns))
            for columns in COLUMNS}


def body_count_on_single_runs(results):
    """C2's count taken on single runs at the chosen rate, in every way of picking one repeat for
    body width 4 and one for body width 128: {(repeat at 4, repeat at 128): settings in which
    body width 128 scored 10 points or more higher}. Nine counts; three use the same repeat twice."""
    cases = [(columns, bits, creatures, use) for columns in COLUMNS for bits in BITS for creatures in CREATURES
             for use in USES]
    return {(a, b): sum(results.by_setting[(case[0], 128, *case[1:])][b - 1]["facts"]
                        - results.by_setting[(case[0], 4, *case[1:])][a - 1]["facts"] >= RISE for case in cases)
            for a in REPEATS for b in REPEATS}


def rate_check_rows(columns):
    """The rows of the table in the figure: for each prediction what is counted, what the
    prediction put the count at, and the count in every column. columns is a list of the
    verdicts of single_run_verdicts: at the chosen rate first, then at a third of it."""
    first = columns[0]
    return [
        ("C1", "falls of 5 or more", "predicted: none",
         [found["C1"]["falls"] for found in columns]),
        ("", "10 or more higher at 64 columns", f"predicted: {first['C1']['needed']} or more of {first['C1']['cases']}",
         [found["C1"]["gained"] for found in columns]),
        ("C2", "10 or more higher at body width 128", f"predicted: {first['C2']['allowed']} or fewer of {first['C2']['cases']}",
         [found["C2"]["gained"] for found in columns]),
        ("A1", "ended as learned", f"predicted: all {first['A1']['cases']}",
         [found["A1"]["learned"] for found in columns]),
        ("A2", "reached 95", f"predicted: none of {first['A2']['cases']}",
         [found["A2"]["reached"] for found in columns]),
        ("A3", "pairs within 10 points", f"predicted: all {first['A3']['cases']}",
         [found["A3"]["within"] for found in columns]),
    ]


def figure_rate_check(results, pairs, at_chosen, at_third):
    """Left: every setting as a dot, its score at the chosen rate (the middle of its three
    repeats) across and its run at a third of the rate up; on the line the rate made no
    difference, and the dots outside the grey band (5 points or more away) are drawn in ink.
    Right: the five predictions applied to single runs, the best repeat of every setting at the
    chosen rate and the same repeat at a third of it, each with a tick or a cross."""
    change = against_the_setting(results, pairs)
    panel, top = 1.74, 0.7
    width, height = FULL, top + panel + 0.52
    figure, page = page_of(width, height)
    write(page, 0, 0, "Every setting again at a third of the learning rate", HEAD, "semibold")
    write(page, 0, 0.19, "the best repeat of every setting, trained again at a third of the rate", SMALL, colour=SECOND)

    # The scatter.
    x0, low = 0.42, GUESS - 5
    write(page, x0, top - 0.14, "Facts known: one dot is one setting", SMALL, "semibold", va="center")
    ax = plot_at(figure, x0, top, panel, panel)
    ax.set_xlim(low, 103)
    ax.set_ylim(low, 103)
    ax.fill_between([low, 103], [low - RATE_BAND, 103 - RATE_BAND], [low + RATE_BAND, 103 + RATE_BAND],
                    color=ZONE, linewidth=0, zorder=1)
    ax.plot([low, 103], [low, 103], color=AXIS, linewidth=0.8, zorder=2)
    for inside in (True, False):  # the dots outside the band last, in ink
        chosen = [(results.score[setting_of(counted)], lower["facts"]) for counted, lower in pairs
                  if (abs(change[setting_of(counted)]) < RATE_BAND) == inside]
        ax.plot(*zip(*chosen), linestyle="none", marker="o", markersize=2.6 if inside else 3.6,
                color=MUTED if inside else INK, alpha=0.5 if inside else 1, markeredgewidth=0, zorder=3 if inside else 4)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_yticks([25, 50, 75, 100])
    ax.spines["bottom"].set_color(AXIS)
    higher = sum(c >= RATE_BAND for c in change.values())
    lower = sum(c <= -RATE_BAND for c in change.values())
    for k, line in enumerate([f"{higher} ended {RATE_BAND} points", "or more higher"]):
        write(ax, low + 3, 99 - k * 7, line, SMALL, "medium" if k == 0 else "regular", va="center")
    for k, line in enumerate([f"{lower} ended {RATE_BAND} points", "or more lower"]):
        write(ax, 101, 33 - k * 7, line, SMALL, "medium" if k == 0 else "regular", ha="right", va="center")
    for k, line in enumerate(["the setting’s score at the chosen rate", "(the middle of its three repeats)"]):
        write(page, x0 + panel / 2, top + panel + 0.2 + k * 0.135, line, SMALL, colour=SECOND, ha="center")
    write(page, x0 - 0.35, top + panel / 2, "its run at a third of the rate", SMALL, colour=SECOND, ha="center",
          va="center", rotation=90)
    # Beside the scatter: how many dots lie on each other in the corner, and the band.
    at_top = sum(results.score[setting_of(counted)] >= AT_THE_TOP and lower_run["facts"] >= AT_THE_TOP
                 for counted, lower_run in pairs)
    nx = x0 + panel + 0.1
    page.plot([x0 + panel - 0.02, nx - 0.02], [top + 0.07, top + 0.07], color=SECOND, linewidth=0.6)
    for k, line in enumerate([f"{at_top} dots lie", "here, at 100", "and 100"]):
        write(page, nx, top + 0.07 + k * 0.135, line, SMALL, "medium" if k == 0 else "regular",
              INK if k == 0 else SECOND, va="center")
    for k, line in enumerate([f"{len(pairs) - higher - lower} within", f"{RATE_BAND} points:", "the grey band"]):
        write(page, nx, top + 0.66 + k * 0.135, line, SMALL, "medium" if k == 0 else "regular",
              INK if k == 0 else SECOND, va="center")
    for k, line in enumerate(["on the line:", "no difference"]):
        write(page, nx, top + 1.25 + k * 0.135, line, SMALL, colour=SECOND, va="center")

    # The five predictions on single runs: what is counted, what the prediction put the count
    # at, and the count at the chosen rate and at a third of it, with a tick or a cross.
    columns = [at_chosen, at_third]
    tx, row_h = 3.12, 0.3
    count_x = [width - 0.62, width - 0.07]   # where the marks of the two columns stand
    write(page, tx, top - 0.14, "The five predictions on single runs", SMALL, "semibold", va="center")
    for x, lines in zip(count_x, (["chosen", "rate"], ["a third", "of it"])):
        for k, line in enumerate(lines):
            write(page, x + 0.05, top - 0.02 + k * 0.135, line, SMALL, colour=SECOND, ha="right", va="center")
    y = top + 0.3
    for name, counted, predicted, counts in rate_check_rows(columns):
        page.plot([tx, width], [y - 0.055, y - 0.055], color=GRID, linewidth=0.6)
        write(page, tx, y, name, SMALL, "bold")
        write(page, tx + 0.27, y, counted, SMALL)
        write(page, tx + 0.27, y + 0.125, predicted, SMALL, colour=MUTED)
        for x, count, found in zip(count_x, counts, columns):
            write(page, x - 0.14, y, nice(count), SMALL, "semibold", ha="right")
            if name:  # the verdict on the whole prediction stands in its first row
                (tick_mark if found[name]["held"] else cross_mark)(page, x - 0.02, y + 0.065, 0.04)
        y += row_h
    page.plot([tx, width], [y - 0.055, y - 0.055], color=GRID, linewidth=0.6)
    assert y - 0.055 <= height
    save(figure, "fig14_rate_check")


def rate_check_numbers(results):
    """What the report says about the rate check: draws its figure and returns its numbers."""
    pairs = read_rate_check(results)
    # Like is compared with like for the verdicts: the best repeat of every setting at the
    # chosen rate, and the same repeat at a third of it.
    at_chosen = single_run_verdicts([counted for counted, _ in pairs])
    at_third = single_run_verdicts([lower for _, lower in pairs])
    figure_rate_check(results, pairs, at_chosen, at_third)

    rates = {lower["rate"] for _, lower in pairs}
    assert len(rates) == 1
    numbers = {"RcRate": f"{rates.pop():g}", "RcRuns": nice(len(pairs))}
    # The plan's rule for "the best repeat" has two tie-breaks; how often did they decide?
    ties = 0
    for repeats in results.by_setting.values():
        best, second = sorted((run["facts"] for run in repeats), reverse=True)[:2]
        ties += best == second
    numbers["RcTies"] = ties

    # Against the setting's score at the chosen rate (the middle of its three repeats).
    change = against_the_setting(results, pairs)
    numbers["RcHigher"] = sum(c >= RATE_BAND for c in change.values())
    numbers["RcLower"] = sum(c <= -RATE_BAND for c in change.values())
    numbers["RcWithin"] = len(pairs) - numbers["RcHigher"] - numbers["RcLower"]
    assert numbers["RcHigher"] < numbers["RcLower"]  # "no better results"
    counts = by_columns(change)
    for columns in COLUMNS:
        numbers[f"RcUp{COLUMN_NAMES[columns]}"], numbers[f"RcDown{COLUMN_NAMES[columns]}"] = counts[columns]
    # "A little worse with few columns": more settings ended lower than higher with 8 and with 16
    # columns, and more ended higher with 64. This holds by the other yardstick too: against
    # the very run that was repeated (the best of three), where a fall is to be expected.
    counts_run = by_columns(against_the_run(pairs))
    assert all(both[columns][0] < both[columns][1] for columns in (8, 16) for both in (counts, counts_run))
    assert all(both[64][0] > both[64][1] for both in (counts, counts_run))
    numbers["RcDownSixtyFourWord"] = "none" if counts[64][1] == 0 else str(counts[64][1])
    lost = [c for c in change.values() if c <= -RATE_MARGIN]
    numbers["RcLostMiddle"] = one(-middle(lost))
    # The typical run with 1,024 creatures at 128 bits at a third of the rate, for each word use
    # (the middle of 6 runs: one for each body width).
    for use in USES:
        heavy = [middle(lower["facts"] for counted, lower in pairs
                        if (counted["columns"], counted["bits"], counted["creatures"], counted["use"]) == (columns, 128, 1024, use))
                 for columns in COLUMNS]
        numbers[f"RcTypical{USE_NAMES[use]}"] = in_a_row(heavy, whole)
        # "the gaps between the column counts did not close": still rising with the columns, and
        # as far apart from 8 to 64 columns as in the counted runs, give or take 2 points
        assert heavy == sorted(heavy)
        assert heavy[-1] - heavy[0] >= results.typical(64, 128, 1024, use) - results.typical(8, 128, 1024, use) - 2
    # The questions about one creature, against the setting's middle score on them.
    questions = against_the_setting(results, pairs, "questions_one")
    numbers["RcQuestionsHigher"] = sum(c >= RATE_BAND for c in questions.values())
    numbers["RcQuestionsLower"] = sum(c <= -RATE_BAND for c in questions.values())
    # A lower rate learns more slowly: how many runs the limit ended at each rate, and what
    # those runs still gained over their last doubling.
    numbers["RcLimitChosen"] = sum(counted["reason"] == "limit" for counted, _ in pairs)
    numbers["RcLimitThird"] = sum(lower["reason"] == "limit" for _, lower in pairs)
    assert numbers["RcLimitThird"] > numbers["RcLimitChosen"]
    still = lambda runs: middle(run["at"][LIMIT]["facts"] - run["at"][LIMIT // 2]["facts"] for run in runs if run["reason"] == "limit")
    numbers["RcLimitGainChosen"] = one(still([counted for counted, _ in pairs]))
    numbers["RcLimitGainThird"] = one(still([lower for _, lower in pairs]))

    # The five verdicts on single runs. The text says: C1, A1, A2 and A3 come out the same at
    # both rates, and C2's count crosses its line.
    held = {name: (at_chosen[name]["held"], at_third[name]["held"]) for name in PREDICTION_NAMES}
    assert held == {"C1": (True, True), "C2": (False, True), "A1": (False, False), "A2": (True, True), "A3": (False, False)}
    assert at_chosen["C1"]["falls"] == at_third["C1"]["falls"] == 0
    numbers["RcCTwoChosen"], numbers["RcCTwoThird"] = at_chosen["C2"]["gained"], at_third["C2"]["gained"]
    # Most settings with 64 columns were at 100 already and could not move.
    numbers["RcTopSixtyFour"] = sum(results.score[setting] >= AT_THE_TOP for setting in change if setting[0] == 64)
    numbers["RcPerColumns"] = len(pairs) // len(COLUMNS)
    # C2's count on other single runs at the chosen rate: the yardstick for 17 against 13.
    single = body_count_on_single_runs(results)
    same_repeat = [single[(repeat, repeat)] for repeat in REPEATS]
    numbers["CTwoSingleLow"], numbers["CTwoSingleHigh"] = min(single.values()), max(single.values())
    numbers["CTwoSameRepeatLow"], numbers["CTwoSameRepeatHigh"] = min(same_repeat), max(same_repeat)
    assert numbers["CTwoSingleLow"] <= at_chosen["C2"]["allowed"] < numbers["CTwoSingleHigh"]  # both sides of the line
    # At a third of the rate: how many of the settings that count are made by a run one kind short.
    third_run = {setting_of(lower): lower for _, lower in pairs}
    third_cases = [case for case, gain in at_third["C2"]["gains"].items() if gain >= RISE]
    numbers["RcCTwoByChance"] = in_words(sum(kinds_short(third_run[(case[0], 4, *case[1:])]) > 0 for case in third_cases))

    # When and where the stage ran, and that no further stage followed, from the launcher's log.
    lines = open(ROOT / "logs" / "v3" / "fleet.log").read().split("\n")
    start = next(i for i, line in enumerate(lines) if "the counted stage is complete" in line)
    end = next(i for i, line in enumerate(lines) if f"stage {RATE_STAGE}: 960 of 960 runs here" in line)
    stage_lines = lines[start:end + 1]
    cards = Counter(dict(re.findall(r"machine (\S+) \((\w+)\): group", "\n".join(stage_lines))).values())
    numbers["RcStart"], numbers["RcEnd"] = lines[start][11:16], lines[end][11:16]
    numbers["RcMachines"] = in_words(sum(cards.values()))
    numbers["RcMachineList"] = in_a_row([f"{in_words(count)} {card}" for card, count in sorted(cards.items())])
    failed = next(line for line in stage_lines if "(by hand)" in line and "failed once each" in line)
    numbers["RcGroupFailures"] = in_words(int(re.search(r"(\d+) groups failed once each", failed).group(1)))
    # The stage at a tenth of the rate, which the plan allowed: not run when the report was
    # written (the launcher's log says why). Should its folder appear later, the report says so
    # in one sentence and shows nothing of it; this script would have to be extended for it.
    assert any("no further stage is started" in line for line in lines[start:])
    numbers["RcTenth"] = ("It was not run: the machines were given back to save compute units."
                          if not (RUNS / "rate_check_tenth").exists() else
                          "It was started after this report was written and is not part of it.")
    return numbers


# ---------------------------------------------------------------------------------------------
# The four wide pages: every run, as a tile with its score
# ---------------------------------------------------------------------------------------------

def corner_mark(page, x, y, width, colour, gap=0.011, size=0.042):
    """A small triangle in the upper right corner of a tile: the run was still training when the
    limit of 640,000 steps ended it. It is small enough to stay clear of the number."""
    page.add_patch(Polygon([(x + width - gap - size, y + gap), (x + width - gap, y + gap),
                            (x + width - gap, y + gap + size)], closed=True, facecolor=colour, edgecolor="none"))


def figure_every_run(results, columns, name):
    """All 720 runs with one number of table columns. Left half equal use, right half unequal
    use; across, the bits per creature; down, the creatures (four blocks) and inside each the
    body widths. Each place holds three tiles: the three repeats of that setting."""
    width, height = WIDE, 6.58
    figure, page = page_of(width, height)
    between = 0.2                       # paper between the two halves
    half_w = (width - between) / 2
    gutter, group_gap, block_gap = 0.46, 0.1, 0.11
    tile_w = (half_w - gutter - (len(BITS) - 1) * group_gap) / (len(BITS) * len(REPEATS))
    row_h, tiles_top = 0.215, 1.03

    # The title line, with the teal of this number of columns.
    page.add_patch(Rectangle((0, 0.035), 0.2, 0.115, facecolor=COLUMN_TEAL[columns], edgecolor="none"))
    title = f"{columns} table columns"
    write(page, 0.28, 0.0, title, 11.5, "bold")
    write(page, 0.28 + text_width(title, 11.5, "bold") + 0.12, 0.035,
          "facts known, out of 100, at the end of each of the 720 runs whose word table has this many columns",
          BASE, colour=SECOND)
    # The key: the scale, what three tiles are, and the corner mark.
    y = 0.38
    x = scale_key(page, 0, y) + 0.26
    for place in range(3):
        tile(page, x + place * 0.16, y - 0.075, 0.16, 0.15, tile_colour(100 - 14 * place))
    x += 3 * 0.16 + 0.07
    words = "the three repeats of one setting, in the order 1, 2, 3"
    write(page, x, y, words, SMALL, colour=SECOND, va="center")
    x += text_width(words) + 0.26
    tile(page, x, y - 0.075, 0.16, 0.15, tile_colour(60))
    corner_mark(page, x, y - 0.075, 0.16, PAPER)
    write(page, x + 0.23, y, "the limit of 640,000 steps ended the run", SMALL, colour=SECOND, va="center")

    for half, use in enumerate(USES):
        x0 = half * (half_w + between)
        tail = "every creature appears equally often" if use == "equal" else "a few creatures appear far more often"
        x = use_heading(page, x0 + gutter, 0.72, use, BASE, "semibold")
        write(page, x, 0.72, tail, SMALL, colour=SECOND, va="center")
        for group, bits in enumerate(BITS):
            gx = x0 + gutter + group * (len(REPEATS) * tile_w + group_gap)
            write(page, gx + 1.5 * tile_w, 0.935, f"{bits} bits", SMALL, "medium", ha="center", va="center")
        write(page, x0 + gutter - 0.07, 0.935, "body", SMALL, colour=SECOND, ha="right", va="center")
        for block, creatures in enumerate(CREATURES):
            block_top = tiles_top + block * (len(BODIES) * row_h + block_gap)
            write(page, x0 + 0.075, block_top + len(BODIES) * row_h / 2, f"{nice(creatures)} creatures", SMALL,
                  "medium", ha="center", va="center", rotation=90)
            for line, body in enumerate(BODIES):
                y = block_top + line * row_h
                # the body width of the row, on the same baseline as the numbers in its tiles
                write(page, x0 + gutter - 0.07, y + row_h / 2 + 0.004, str(body), SMALL, colour=SECOND, ha="right",
                      va="center_baseline")
                for group, bits in enumerate(BITS):
                    gx = x0 + gutter + group * (len(REPEATS) * tile_w + group_gap)
                    for place, run in enumerate(results.by_setting[(columns, body, bits, creatures, use)]):
                        tx = gx + place * tile_w
                        tile(page, tx, y, tile_w, row_h, tile_colour(run["facts"]))
                        tile_number(page, tx + tile_w / 2, y + row_h / 2, f"{run['facts']:.0f}", run["facts"])
                        if run["reason"] == "limit":
                            corner_mark(page, tx, y, tile_w, tile_ink(run["facts"]))
    assert tiles_top + len(CREATURES) * len(BODIES) * row_h + (len(CREATURES) - 1) * block_gap <= height
    save(figure, name)


# ---------------------------------------------------------------------------------------------
# The numbers the text uses
# ---------------------------------------------------------------------------------------------

COLUMN_NAMES = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour"}  # for macro names
USE_NAMES = {"equal": "Equal", "unequal": "Unequal"}


def one(value):
    """A score with one decimal: 97.04 -> "97.0"."""
    return f"{value:.1f}"


def whole(value):
    """A score as a whole number: 78.4 -> "78"."""
    return f"{value:.0f}"


def in_a_row(values, form=str):
    """[1, 2, 3] -> "1, 2 and 3"."""
    values = [form(value) for value in values]
    return ", ".join(values[:-1]) + " and " + values[-1]


def capital(words):
    """five -> Five, for a count that starts a sentence."""
    return words[0].upper() + words[1:]


def all_languages():
    """Every combination of bits, creatures and body width, for comparisons of size."""
    return [(body, bits, creatures) for body in BODIES for bits in BITS for creatures in CREATURES]


def numbers_of_the_design(results):
    """What was run, how the runs ended, and how large the models are."""
    runs = results.runs
    numbers = {"RunsAll": nice(len(runs)), "Settings": len(results.by_setting),
               "RunsPerUsePoint": len(results.by_cell[(8, 8, 16, "equal")]),
               "SettingsPerColumns": len(pick(runs, columns=8)) // 3,
               "FirstLook": nice(FIRST_LOOK), "Limit": nice(LIMIT), "LearnedBar": LEARNED,
               "RunsPerGroup": RUNS_PER_GROUP, "Groups": len(runs) // RUNS_PER_GROUP}
    rates = {run["rate"] for run in runs}
    assert len(rates) == 1  # one learning rate for all
    numbers["Rate"] = f"{rates.pop():g}"
    # How the runs ended (Figure 5).
    counts = endings(results)
    total = lambda column: sum(count for (name, _), count in counts.items() if name == column)
    # "Three in four of the runs ended as no more gain knew the facts."
    assert 0.72 < total("known") / (total("known") + total("below")) < 0.78
    numbers["StepsMillions"] = whole(sum(run["ended"] for run in runs) / 1e6)
    # The models of Figure 3, and how much larger more columns and a wider body make a model.
    example = dict(SIZE_EXAMPLE)
    small, large = (model_sizes(columns, example["body"], example["creatures"], example["bits"]) for columns in (8, 64))
    numbers["AttentionTimes"] = whole(large["attention"] / small["attention"])
    # The output part is largest at the most bits.
    numbers["OutputManySmall"] = nice(model_sizes(8, 16, 1024, BITS[-1])["output"])
    numbers["OutputManyLarge"] = nice(model_sizes(64, 16, 1024, BITS[-1])["output"])
    by_columns_ = [model_sizes(64, body, creatures, bits)["all"] / model_sizes(8, body, creatures, bits)["all"]
                   for body, bits, creatures in all_languages()]
    numbers["SizeGridLow"], numbers["SizeGridHigh"] = whole(min(by_columns_)), whole(max(by_columns_))
    by_body = {creatures: [model_sizes(columns, 128, creatures, bits)["all"] / model_sizes(columns, 4, creatures, bits)["all"]
                           for columns in COLUMNS for bits in BITS] for creatures in CREATURES}
    everywhere = [ratio for ratios in by_body.values() for ratio in ratios]
    numbers["BodyGridLow"], numbers["BodyGridHigh"] = one(min(everywhere)), one(max(everywhere))
    numbers["BodyFewLow"], numbers["BodyFewHigh"] = one(min(by_body[16])), one(max(by_body[16]))
    numbers["BodyManyLow"], numbers["BodyManyHigh"] = one(min(by_body[1024])), one(max(by_body[1024]))
    return numbers


def numbers_of_the_predictions(results, found):
    """The five verdicts with the counts on each side, and what the findings say about them."""
    score = results.score
    c1, c2, a1, a2, a3 = (found[name] for name in ("C1", "C2", "A1", "A2", "A3"))
    numbers = {"HeldCount": in_words(sum(found[name]["held"] for name in found)),
               "NotHeldCount": in_words(sum(not found[name]["held"] for name in found))}
    assert [found[name]["held"] for name in PREDICTION_NAMES] == [True, False, False, True, False]
    # A setting whose score is made by runs one kind of fact short (finding 5): at least two of
    # its three repeats are, so its middle is.
    by_chance = lambda setting: sum(kinds_short(run) > 0 for run in results.by_setting[setting]) >= 2

    # C1
    numbers.update({"COneComparisons": c1["comparisons"], "COneFalls": c1["falls"], "COneLargestFall": one(c1["largest_fall"]),
                    "COneWentUp": c1["went_up"], "COneStayedLevel": c1["stayed_level"],
                    "COneCases": c1["cases"], "COneGained": c1["gained"], "COneNeeded": c1["needed"],
                    "COneNotGained": c1["cases"] - c1["gained"], "COneLargestGain": whole(max(c1["gains"].values()))})
    room = [case for case in c1["gains"] if score[(8, *case)] <= ROOM]
    numbers["COneRoom"], numbers["RoomBar"] = len(room), ROOM
    assert all(c1["gains"][case] >= RISE for case in room) and len(room) == c1["gained"]  # all with room gained
    numbers["COneNoRoom"] = sum(score[(8, *case)] >= A2_BAR for case in c1["gains"] if case not in room)
    chance = [case for case, gain in c1["gains"].items() if gain >= RISE and by_chance((8, *case))]
    numbers["COneByChance"] = in_words(len(chance))
    numbers["COneWithoutChance"] = c1["gained"] - len(chance)
    assert c1["gained"] - len(chance) >= c1["needed"]
    # C1 run by run: the largest amount by which any run with fewer columns stood above any run
    # of the same case with the next larger number of columns (all 9 pairings of repeats).
    run_fall = max(max(run["facts"] for run in results.by_setting[(narrower, *case)])
                   - min(run["facts"] for run in results.by_setting[(wider, *case)])
                   for case in c1["gains"] for narrower, wider in zip(COLUMNS, COLUMNS[1:]))
    numbers["COneRunFall"] = one(run_fall)
    assert run_fall < 0.5
    best = lambda setting: max(run["facts"] for run in results.by_setting[setting])
    worst = lambda setting: min(run["facts"] for run in results.by_setting[setting])
    numbers["COneRunLow"] = sum(worst((64, *case)) - best((8, *case)) >= RISE for case in c1["gains"])
    numbers["COneRunHigh"] = sum(best((64, *case)) - worst((8, *case)) >= RISE for case in c1["gains"])
    assert numbers["COneRunLow"] >= c1["needed"]
    # The typical model with 1,024 creatures at 128 bits, for each word use, and how far apart
    # neighbouring column counts lie there.
    gaps = []
    for use in USES:
        heavy = [results.typical(columns, 128, 1024, use) for columns in COLUMNS]
        numbers[f"Typical{USE_NAMES[use]}"] = in_a_row(heavy, whole)
        gaps += [high - low for low, high in zip(heavy, heavy[1:])]
    numbers["TypicalGapLow"], numbers["TypicalGapHigh"] = whole(min(gaps)), whole(max(gaps))
    pooled = pooled_verdicts(results)
    numbers.update({"PooledCOneGained": pooled["c1_gained"], "PooledCOneCases": pooled["c1_cases"],
                    "PooledCTwoGained": pooled["c2_gained"], "PooledCTwoCases": pooled["c2_cases"]})
    assert pooled["c1_falls"] == 0 and pooled["c1_gained"] >= C1_SHARE * pooled["c1_cases"]  # C1 holds this way too
    assert pooled["c2_gained"] >= C2_SHARE * pooled["c2_cases"]                               # and C2 does not

    # C2
    gained = {case: gain for case, gain in c2["gains"].items() if gain >= RISE}
    narrow, wide = (lambda case: (case[0], 4, *case[1:])), (lambda case: (case[0], 128, *case[1:]))
    numbers.update({"CTwoCases": c2["cases"], "CTwoGained": c2["gained"], "CTwoAllowed": c2["allowed"],
                    "CTwoPercent": whole(100 * c2["gained"] / c2["cases"]), "CTwoOver": in_words(c2["gained"] - c2["allowed"])})
    for columns in COLUMNS:
        numbers[f"CTwoAt{COLUMN_NAMES[columns]}"] = in_words(sum(case[0] == columns for case in gained))
    largest = max(gained, key=gained.get)
    numbers["CTwoLargest"] = one(gained[largest])
    numbers["CTwoLargestWhere"] = f"{largest[0]} columns, {largest[1]} bits, {nice(largest[2])} creatures, {largest[3]} use"
    numbers["CTwoLargestFrom"], numbers["CTwoLargestTo"] = whole(score[narrow(largest)]), whole(score[wide(largest)])
    # One of the cases is made by runs that were one kind of fact short (finding 5). Every other
    # case has more bits than columns: it lies right of the stepped line of Figure 8.
    chance = [case for case in gained if by_chance(narrow(case))]
    numbers["CTwoByChance"] = in_words(len(chance))
    numbers["CTwoWithoutChance"] = c2["gained"] - len(chance)
    assert c2["gained"] - len(chance) > c2["allowed"]
    assert all(case[1] > case[0] for case in gained if case not in chance)
    assert all(case[1] <= case[0] for case in chance)
    # How many settings had room for a gain of 10 points, and how many of those gained it.
    room = [case for case in c2["gains"] if score[narrow(case)] <= ROOM]
    assert all(case in room for case in gained)
    numbers["CTwoRoom"], numbers["CTwoNoRoom"] = len(room), c2["cases"] - len(room)
    for name, chosen in (("Few", (16, 64)), ("Mid", (256,)), ("Many", (1024,))):
        numbers[f"CTwoRoom{name}"] = sum(case[2] in chosen for case in room)
        numbers[f"CTwoRoom{name}Gained"] = sum(case[2] in chosen for case in gained)
    assert numbers["CTwoRoomFew"] == numbers["CTwoRoomFewGained"]  # "with 16 and 64 creatures all of them"
    # C2 run by run: the worst run at body width 128 against the best at 4, and the other way.
    numbers["CTwoRunLow"] = sum(worst(wide(case)) - best(narrow(case)) >= RISE for case in c2["gains"])
    numbers["CTwoRunHigh"] = sum(best(wide(case)) - worst(narrow(case)) >= RISE for case in c2["gains"])
    # The widest body (128) against the narrowest body (4) with twice the columns, everything
    # else the same: the model with twice the columns never scored clearly lower.
    doubled = [(score[(wider, BODIES[0], bits, creatures, use)] - score[(narrower, BODIES[-1], bits, creatures, use)],
                score[(narrower, BODIES[-1], bits, creatures, use)])
               for narrower, wider in zip(COLUMNS, COLUMNS[1:]) for bits in BITS for creatures in CREATURES for use in USES]
    numbers["DoubledCases"] = len(doubled)
    numbers["DoubledMore"] = sum(margin > 0.5 for margin, _ in doubled)
    numbers["DoubledTied"] = sum(abs(margin) <= 0.5 for margin, _ in doubled)
    assert numbers["DoubledMore"] + numbers["DoubledTied"] == len(doubled)  # never clearly lower
    # "clearly more in all comparisons where the model with fewer columns was not already at 99"
    assert all((margin > 0.5) == (other < LEARNED) for margin, other in doubled)
    numbers["BodyTimes"] = BODIES[-1] // BODIES[0]

    # Models of nearly the same size with different numbers of columns (finding 3).
    pairs = same_size_pairs(results)
    assert all((fewer[0], fewer[1], more[0]) == (8, BODIES[-1], 16) for fewer, more in pairs)
    margins = [score[more] - score[fewer] for fewer, more in pairs]
    numbers.update({
        "SameSizePairs": len(pairs), "SameSizeLanguages": in_words(len({fewer[2:] for fewer, _ in pairs})),
        "SameSizePercent": round(100 * (SAME_SIZE - 1)),
        "SameSizeHigher": sum(margin > 0 for margin in margins),
        "SameSizeByTwo": sum(margin >= 2 for margin in margins), "SameSizeByFive": in_words(sum(margin >= 5 for margin in margins)),
        "SameSizeLargest": whole(max(margins)),
        "SameSizeSmaller": in_words(sum(size_of(more) < size_of(fewer) for fewer, more in pairs)),
        "SameSizeAllRuns": sum(worst(more) > best(fewer) for fewer, more in pairs),
        "SameSizeMostCreatures": nice(max(fewer[3] for fewer, _ in pairs)),
    })
    assert numbers["SameSizeHigher"] == len(pairs)
    assert all(score[more] > score[fewer] for fewer, more in pairs if size_of(more) < size_of(fewer))
    # The three panels of Figure 9: the two neighbours in size, and their scores.
    for name, (creatures, bits, use) in zip(("First", "Second", "Third"), SIZE_PANELS):
        wide_body, narrow_body = (8, BODIES[-1], bits, creatures, use), (16, BODIES[0], bits, creatures, use)
        numbers[f"Size{name}Wide"], numbers[f"Size{name}Narrow"] = nice(size_of(wide_body)), nice(size_of(narrow_body))
        numbers[f"Size{name}WideScore"], numbers[f"Size{name}NarrowScore"] = whole(score[wide_body]), whole(score[narrow_body])
        numbers[f"Size{name}Gap"] = one(score[narrow_body] - score[wide_body])
        numbers[f"Size{name}Percent"] = whole(100 * (size_of(narrow_body) / size_of(wide_body) - 1))
    for creatures, bits, use in SIZE_PANELS[:2]:  # in the first two panels the two ringed models are such a pair
        assert ((8, BODIES[-1], bits, creatures, use), (16, BODIES[0], bits, creatures, use)) in pairs
    # With 1,024 creatures no two models with different columns are within a factor of 1.5 in size.
    closest = min(size_of((wider, 4, bits, 1024, "equal")) / size_of((narrower, 128, bits, 1024, "equal"))
                  for narrower, wider in zip(COLUMNS, COLUMNS[1:]) for bits in BITS)
    numbers["SizeManyClosest"] = one(np.floor(closest * 10) / 10)  # rounded down: "at least ... times as large"
    assert closest > 1.4

    # A1
    missed = a1["missed"]
    assert all(load_of(setting) == A1_LOAD for setting in missed)  # every miss is at exactly 1 bit per column
    light = [setting for setting in results.by_setting if load_of(setting) <= A1_LOAD]
    at_one = [setting for setting in light if load_of(setting) == A1_LOAD]
    facts_short = [setting for setting in light if score[setting] < LEARNED]
    assert all(setting in missed for setting in facts_short)
    # In how many of the misses were the questions the only shortfall: every repeat that did not
    # end as learned knew the facts and was short on the questions about one creature.
    questions_only = [setting for setting in missed
                      if all(facts_known(run) and run["questions_one"] < LEARNED
                             for run in results.by_setting[setting] if run["reason"] != "learned")]
    light_runs = [run for run in results.runs if load_of(run) <= A1_LOAD]
    numbers.update({"AOneCases": a1["cases"], "AOneLearned": a1["learned"], "AOneMissed": len(missed),
                    "AOneAllThree": sum(all(run["reason"] == "learned" for run in results.by_setting[setting]) for setting in light),
                    "AOneBelow": a1["cases"] - len(at_one), "AOneAtOne": len(at_one),
                    "AOneFactsKnown": len(light) - len(facts_short),
                    "AOneQuestionsShort": len(missed) - len(facts_short), "AOneFactsShort": in_words(len(facts_short)),
                    "AOneQuestionsOnly": len(questions_only),
                    "AOneRuns": nice(len(light_runs)),
                    "AOneRunsLearned": nice(sum(run["reason"] == "learned" for run in light_runs))})
    # The settings up to the line whose score is under 99: all 8 columns, 8 bits, 1,024 creatures.
    assert all(setting[0] == 8 and setting[2] == 8 and setting[3] == 1024 for setting in facts_short)
    kind_short = [setting for setting in facts_short if by_chance(setting)]
    missed_by = sorted((LEARNED - score[setting] for setting in facts_short if setting not in kind_short), reverse=True)
    numbers["AOneShortByKind"] = in_words(len(kind_short))
    numbers["AOneShortOthers"] = in_words(len(missed_by))
    numbers["AOneShortOthersBy"] = in_a_row(missed_by, lambda gap: f"{gap:.1f}" if gap >= 0.1 else f"{gap:.2f}")
    numbers["AOneMissedEight"] = sum(setting[0] == 8 for setting in missed)

    # A2
    heavy = [setting for setting in score if load_of(setting) >= A2_LOAD and setting[3] == 1024]
    heavy_runs = [run for setting in heavy for run in results.by_setting[setting]]
    top = max(heavy, key=score.get)
    top_gains = [run["at"][LIMIT]["facts"] - run["at"][LIMIT // 2]["facts"] for run in results.by_setting[top]]
    assert all(run["reason"] == "limit" for run in results.by_setting[top])
    numbers.update({"ATwoCases": a2["cases"], "ATwoReached": a2["reached"], "ATwoHighest": one(a2["highest"]),
                    "ATwoBar": A2_BAR, "ATwoHighestRun": one(max(run["facts"] for run in heavy_runs)),
                    "ATwoRuns": len(heavy_runs), "ATwoRunsLimit": sum(run["reason"] == "limit" for run in heavy_runs),
                    "ATwoTopGainLow": one(min(top_gains)), "ATwoTopGainHigh": one(max(top_gains))})
    assert max(run["facts"] for run in heavy_runs) < A2_BAR  # no single run reached 95 either

    # A3
    fewer_higher = [pair for pair in a3["apart"] if pair["fewer"] > pair["more"]]
    numbers.update({"AThreeCases": a3["cases"], "AThreeWithin": a3["within"], "AThreeApart": len(a3["apart"]),
                    "AThreeMoreHigher": a3["more_higher"], "AThreeFewerHigher": in_words(len(fewer_higher))})
    assert all(pair["creatures"] == 1024 and pair["use"] == "unequal" for pair in fewer_higher)
    # The pairs that are apart only because one of the two settings is made by runs one kind
    # short: its score is 81, the other's 100.
    chance = [pair for pair in a3["apart"] if any(by_chance(setting) for setting in pair["settings"])]
    numbers["AThreeByChance"] = in_words(len(chance))
    numbers["AThreeWithoutChance"] = len(a3["apart"]) - len(chance)
    assert all(pair["more"] > pair["fewer"] for pair in chance)
    return numbers


def numbers_of_the_line(results):
    """What Figure 10 shows: up to as many bits as columns the facts are known, and what lies
    one doubling of the bits past that line."""
    typical = results.typical
    numbers = {}
    # "each doubling of the columns bought one doubling of the bits": with 1,024 creatures the
    # typical score is 99 or more up to bits = columns (one tile excepted: the runs one kind
    # short). One doubling on it is below 90 with unequal use; with equal use see below.
    for use in USES:
        for columns in COLUMNS:
            for bits in BITS:
                stuck = (columns, bits, use) == (STUCK["columns"], STUCK["bits"], STUCK["use"])
                if bits <= columns and not stuck:
                    assert typical(columns, bits, 1024, use) >= LEARNED
                if bits == 2 * columns and use == "unequal":
                    assert typical(columns, bits, 1024, use) < 90
    numbers["StuckTile"] = whole(typical(STUCK["columns"], STUCK["bits"], STUCK["creatures"], STUCK["use"]))
    # One doubling of the bits past the line, with 1,024 creatures and with 256.
    behind = []
    for use in USES:
        values = [typical(columns, 2 * columns, 1024, use) for columns in COLUMNS]
        numbers[f"SameLoad{USE_NAMES[use]}"] = in_a_row(values, whole)
        assert values == sorted(values, reverse=(use == "unequal"))  # rises with equal use, falls with unequal use
        behind += [run for columns in COLUMNS for run in results.by_cell[(columns, 2 * columns, 1024, use)]]
        fewer = [typical(columns, 2 * columns, 256, use) for columns in COLUMNS]
        numbers[f"SameLoadMid{USE_NAMES[use]}Low"], numbers[f"SameLoadMid{USE_NAMES[use]}High"] = whole(min(fewer)), whole(max(fewer))
        for creatures in CREATURES[:-1]:  # "with fewer creatures the fall past the line is gentler"
            assert all(typical(columns, 2 * columns, creatures, use) >= many - 1 for columns, many in zip(COLUMNS, values))
    numbers["SameLoadRuns"] = len(behind)
    numbers["SameLoadRunsLimit"] = sum(run["reason"] == "limit" for run in behind)
    # More creatures lower the score only past the line.
    example = [typical(8, 32, creatures, "equal") for creatures in CREATURES]
    numbers["CreaturesExample"] = in_a_row(example, whole)
    assert example == sorted(example, reverse=True)
    for use in USES:
        for columns in COLUMNS:
            for bits in BITS:
                stuck = (columns, bits, use) == (STUCK["columns"], STUCK["bits"], STUCK["use"])
                if bits <= columns and not stuck:  # up to the line: 1,024 creatures known as completely as 16
                    assert abs(typical(columns, bits, 1024, use) - typical(columns, bits, 16, use)) < 1
    # With 16 creatures only the 8-column row falls.
    assert all(typical(columns, bits, 16, use) >= AT_THE_TOP for columns in COLUMNS[1:] for bits in BITS for use in USES)
    numbers["FewCreaturesLowest"] = whole(min(typical(8, bits, 16, use) for bits in BITS for use in USES))
    assert all(typical(8, bits, 16, "equal") >= 98 for bits in BITS)  # and only with unequal use
    return numbers


def numbers_of_the_plateaus(results):
    """The runs that ended one whole kind of fact short (finding 5)."""
    short = [run for run in results.runs if kinds_short(run)]
    assert all(kinds_short(run) == 1 for run in short)
    assert all(run["columns"] == 8 and run["use"] == "equal" for run in short)
    group = pick(results.runs, **STUCK)
    here = [run for run in group if kinds_short(run)]
    numbers = {"PlateauRuns": len(short), "PlateauRunsWord": in_words(len(short)), "PlateauHere": in_words(len(here)),
               "PlateauGroup": len(group), "PlateauScore": f"{(3 * 100 + GUESS) / 4:.2f}",
               "PlateauScoreEight": one((7 * 100 + GUESS) / 8)}
    numbers["PlateauFewer"] = in_words(len([run for run in short if run["bits"] == 8 and run["creatures"] == 256]))
    numbers["PlateauMoreBits"] = in_words(len([run for run in short if run["bits"] == 16]))
    assert len(short) == len(here) + len([run for run in short if run["bits"] == 8 and run["creatures"] == 256]) \
        + len([run for run in short if run["bits"] == 16])
    assert all(run["reason"] == "no more gain" for run in short)
    kinds = Counter(next(k for k, known in enumerate(run["by_kind"]) if known <= GUESS_LEVEL) for run in short
                    if run["bits"] == 8)
    assert len(kinds) > 1  # "the missed kind is not always the same one"
    example = results.by_setting[(8, 4, 8, 1024, "equal")]
    numbers["PlateauExample"] = in_a_row([run["facts"] for run in example], one)
    # At how many body widths of that corner a run one kind short stands beside a repeat of the
    # same setting that knows every fact.
    beside = [body for body in BODIES if any(kinds_short(run) for run in pick(group, body=body))
              and any(facts_known(run) for run in pick(group, body=body))]
    numbers["PlateauBeside"] = in_words(len(beside))
    numbers["PlateauBodies"] = in_words(len(BODIES))
    # Three of them were among the runs carried on: what the further training did.
    carried = [run for run in short if run["carried"]]
    learned = [run for run in carried if run["carried"]["facts"] >= LEARNED]
    numbers["PlateauCarried"] = in_words(len(carried))
    numbers["PlateauCarriedLearned"] = in_words(len(learned))
    numbers["PlateauCarriedNot"] = in_words(len(carried) - len(learned))
    assert all(run["carried"]["gain"] < TOO_EARLY for run in carried if run not in learned)
    numbers["PlateauOthers"] = in_words(len(short) - len(learned))
    # No model with 16 columns or more ended one kind short.
    numbers["PlateauWiderRuns"] = nice(sum(run["columns"] > 8 for run in results.runs))
    spread = [max(run["facts"] for run in repeats) - min(run["facts"] for run in repeats)
              for repeats in results.by_setting.values()]
    numbers["SpreadMiddle"] = f"{middle(spread):.2f}"
    numbers["SpreadOverFive"] = sum(s > 5 for s in spread)
    return numbers


def numbers_of_the_rule(results):
    """The two proofs that the rule did not end runs too early (finding 6): what every "no more
    gain" run was doing when it ended, and what the carried-on runs did afterwards. And the runs
    that the limit ended."""
    # What the runs ended as "no more gain" were doing.
    ended = [run for run in results.runs if run["reason"] == "no more gain"]
    below = stood_below(results)
    knew = [run for run in ended if facts_known(run)]
    assert all(run["questions_one"] < LEARNED for run in knew)  # the questions were short, or it had "learned"
    gains = [last_gain(run) for run in below]
    numbers = {"NoGainRuns": len(ended), "NoGainKnew": len(knew), "NoGainBelow": len(below),
               "NoGainMiddle": one(middle(gains)), "NoGainOverHalf": sum(gain > 0.5 for gain in gains),
               "NoGainEight": sum(run["columns"] == 8 for run in below),
               "NoGainLossFall": f"{middle(run['stop']['loss_fall'] for run in below):.3f}",
               "NoGainLoss": one(middle(run["stop"]["loss_whole"] for run in below))}
    assert max(gains) < 1  # the rule
    assert numbers["NoGainOverHalf"] > len(below) / 2  # "more than half of them more than half a point"

    # The marked runs that were trained on.
    marked = [run for run in results.runs if run["marked"]]
    no_gain = [run for run in marked if run["reason"] == "no more gain"]
    carried_knew, level, gained = carried_groups(results)
    carried = carried_knew + level + gained
    assert all(run["ended"] == LIMIT for run in no_gain if not run["carried"])  # no room left to train on
    jump, others = gained[-1], gained[:-1]
    numbers.update({
        "Marked": len(marked), "MarkedNoGain": len(no_gain), "MarkedAtLastLook": in_words(len(no_gain) - len(carried)),
        "Carried": len(carried), "CarriedKnew": len(carried_knew), "CarriedRoom": len(level) + len(gained),
        "CarriedLevel": in_words(len(level)),
        "CarriedGained": in_words(len(gained)), "CarriedGainedDigits": len(gained),
        "CarriedPercent": whole(100 * len(gained) / len(carried)),
        "CarriedOthers": in_words(len(others)),
        "CarriedOthersLow": one(others[0]["carried"]["gain"]), "CarriedOthersHigh": one(others[-1]["carried"]["gain"]),
        "CarriedOneDoubling": sum(run["carried"]["step"] == 2 * run["ended"] for run in carried),
        "JumpFrom": one(jump["facts"]), "JumpTo": whole(jump["carried"]["facts"]),
        "JumpGain": one(jump["carried"]["gain"]), "JumpQuestions": whole(jump["questions_one"]),
    })
    assert all(run["carried"]["step"] in (2 * run["ended"], 4 * run["ended"]) for run in carried)
    assert kinds_short(jump) == 1 and setting_of(jump)[2:] == (STUCK["bits"], STUCK["creatures"], STUCK["use"])
    learned_at = next(p["step"] for p in jump["curve"] if p["step"] > jump["ended"] and p["facts"] >= LEARNED)
    numbers["JumpStepsAfter"] = nice(round(learned_at - jump["ended"], -3))
    # The rule watches the facts only: what the questions about one creature did meanwhile.
    question_gains = [run["carried"]["questions_one"] - run["questions_one"] for run in carried]
    numbers["CarriedQuestionsGained"] = sum(gain >= 1 for gain in question_gains)
    light = [run for run in carried if load_of(run) <= A1_LOAD]
    numbers["CarriedLight"] = len(light)
    numbers["CarriedLightLearned"] = sum(run["carried"]["facts"] >= LEARNED and run["carried"]["questions_one"] >= LEARNED
                                         for run in light)
    # The runs that the limit ended, and what they still gained over their last doubling.
    at_limit = [run for run in results.runs if run["reason"] == "limit"]
    last = {run["name"]: run["at"][LIMIT]["facts"] - run["at"][LIMIT // 2]["facts"] for run in at_limit}
    numbers["LimitRuns"] = len(at_limit)
    numbers["LimitStillGaining"] = sum(gain >= 1 for gain in last.values())
    typical = [middle(last[run["name"]] for run in at_limit if run["columns"] == columns) for columns in COLUMNS]
    numbers["LimitGainLow"], numbers["LimitGainHigh"] = one(min(typical)), one(max(typical))
    return numbers


def numbers_of_the_scores(results):
    """The four parts of the test (finding 7)."""
    numbers = {}
    typical = results.typical
    names = {"facts": "Facts", "questions_one": "Questions", "questions_new": "New", "not_known": "Not"}
    # Where a creature has one bit for each table column, with 1,024 creatures and unequal use
    # (with equal use one of these places is the tile of the runs one kind short).
    for score, name in names.items():
        values = [typical(columns, columns, SCORES_CREATURES, "unequal", score=score) for columns in COLUMNS]
        numbers[f"AtLine{name}Low"], numbers[f"AtLine{name}High"] = whole(min(values)), whole(max(values))
    # The most there is to know: 64 columns at 128 bits, for each word use.
    for use in USES:
        for score, name in names.items():
            numbers[f"Heavy{name}{USE_NAMES[use]}"] = whole(typical(64, 128, SCORES_CREATURES, use, score=score))
    return numbers


def numbers_of_the_second_experiment(results):
    """At 32 columns the grid repeats the second experiment: do the two agree?"""
    stages = {"main8": 8, "bits16": 16, "bits32": 32, "bits64": 64}
    earlier = {}
    for folder, bits in stages.items():
        for path in sorted(glob.glob(str(ROOT / "runs_v2" / folder / "*" / "summary.json"))):
            summary = json.load(open(path))
            config = summary["config"]
            assert config.get("columns", 32) == 32
            setting = (32, config["body"], bits, config["creatures"], "unequal" if config["usage"] == "zipf" else "equal")
            earlier.setdefault(setting, []).append(summary["final"]["facts"])
    apart = {setting: results.score[setting] - middle(scores) for setting, scores in earlier.items()}
    lighter = [gap for setting, gap in apart.items() if setting[2] < 64]
    heavy = [gap for setting, gap in apart.items() if setting[2] == 64]
    assert min(heavy) > -0.5
    return {"EarlierSettings": len(lighter), "EarlierApart": one(max(abs(gap) for gap in lighter)),
            "EarlierHeavy": len(heavy), "EarlierHeavyLow": whole(abs(min(heavy))), "EarlierHeavyHigh": one(max(heavy))}


def numbers_of_the_practice():
    """The practice runs, and the choice of the one learning rate (v3/settings.json)."""
    practice = read_stage("practice")
    settings = json.load(open(ROOT / "v3" / "settings.json"))
    rates = sorted({run["rate"] for run in practice})
    numbers = {"PracticeRuns": len(practice), "PracticeSettings": len(practice) // len(rates),
               "PracticeRates": in_a_row(rates, lambda rate: f"{rate:g}"), "PracticeSteps": nice(practice[0]["steps"])}
    names = {0.003: "Low", 0.01: "Mid", 0.03: "High"}
    rows, by_columns_ = [], {}
    for rate in rates:
        runs = pick(practice, rate=rate)
        facts, questions = (sum(run[score] for run in runs) / len(runs) for score in ("facts", "questions_one"))
        assert abs(facts - settings["details"]["averages"][f"{rate:g}"]["facts"]) < 0.01  # as the chooser recorded it
        by_columns_[rate] = [sum(run["facts"] for run in pick(runs, columns=columns)) / len(pick(runs, columns=columns))
                             for columns in COLUMNS]
        for columns, value in zip(COLUMNS, by_columns_[rate]):
            numbers[f"Practice{COLUMN_NAMES[columns]}{names[rate]}"] = one(value)
        rows.append(f"{rate:g} & {one(facts)} & {one(questions)} & " + " & ".join(one(value) for value in by_columns_[rate]) + r" \\")
    assert settings["rate"] == 0.003
    # "in the practice runs 8 columns did best at 0.01 and 64 columns at 0.003"
    assert by_columns_[0.01][0] == max(by_columns_[rate][0] for rate in rates)
    assert by_columns_[0.003][-1] == max(by_columns_[rate][-1] for rate in rates)
    numbers["PracticeRows"] = " ".join(rows)
    return numbers


def numbers_of_the_machines():
    """What the launcher's log says about the counted stage: which machines, how many moves and
    restarts. Read up to the line that reports the stage complete."""
    lines = []
    for line in open(ROOT / "logs" / "v3" / "fleet.log"):
        lines.append(line)
        if "the counted stage is complete" in line:
            break
    machines = dict(re.findall(r"machine (\S+) \((\w+)\) added", "".join(lines)))
    # Where each group trained: the log gives a speed reading for a group on a machine, with the
    # group's shape (table columns x kinds of fact x body width).
    trained_on, cards_of_columns = {}, {}
    for machine, card, group, columns in re.findall(r"machine (\S+) \((\w+)\): group (\d+) \(shape (\d+)x", "".join(lines)):
        trained_on.setdefault(group, set()).add((machine, card))
        cards_of_columns.setdefault(int(columns), set()).add(card)
    types = Counter(machines.values())
    order = ["T4", "L4", "A100", "G4", "H100"]
    assert set(types) == set(order)
    # "Every number of table columns was trained on all five types of card."
    assert all(cards_of_columns[columns] == set(order) for columns in COLUMNS)
    numbers = {"Machines": len(machines), "MachineTypes": in_words(len(types)),
               "MachineList": ", ".join(f"{in_words(types[card])} {card}" for card in order[:-1])
                              + f" and {in_words(types[order[-1]])} {order[-1]}",
               "MachinesPaused": in_words(sum("(by hand)" in line and "training paused" in line for line in lines)),
               "LauncherRestarts": in_words(sum("(by hand) restarting the launcher" in line
                                                or "(by hand) switching to the launcher" in line for line in lines)),
               "GroupFailures": in_words(sum(bool(re.search(r"group \d+ failed", line)) for line in lines)),
               "GroupsOnSeveralMachines": sum(len(where) > 1 for where in trained_on.values()),
               "GroupsOnSeveralTypes": sum(len({card for _, card in where}) > 1 for where in trained_on.values())}
    return numbers


def numbers_of_what_comes_next():
    """For "what comes next": how wide a body a model needs to have as many trained numbers as the
    model with twice its columns and the narrowest body, with 1,024 creatures."""
    def body_needed(columns, bits):
        target = model_sizes(2 * columns, BODIES[0], 1024, bits)["all"]
        without = model_sizes(columns, 0, 1024, bits)["all"]
        for_a_neuron = model_sizes(columns, 1, 1024, bits)["all"] - without
        return (target - without) / for_a_neuron
    needed = [body_needed(columns, bits) for columns in (8, 16) for bits in BITS]
    return {"NextBodyLow": int(round(min(needed), -1)), "NextBodyHigh": int(round(max(needed), -1))}


def write_numbers(results, found):
    """Every number the text uses, written to numbers.tex as LaTeX macros (\\RunsAll and so on).
    The macros are grouped here in the order in which the report needs them."""
    numbers = dict(OUTSIDE_FACTS)
    numbers.update(numbers_of_the_design(results))
    numbers.update(numbers_of_the_predictions(results, found))
    numbers.update(numbers_of_the_line(results))
    numbers.update(numbers_of_the_plateaus(results))
    numbers.update(numbers_of_the_rule(results))
    numbers.update(numbers_of_the_scores(results))
    numbers.update(numbers_of_the_second_experiment(results))
    numbers.update(numbers_of_the_practice())
    numbers.update(numbers_of_the_machines())
    numbers.update(numbers_of_what_comes_next())
    numbers.update(rate_check_numbers(results))  # also draws the figure of the rate check
    # A count written as a word also gets a form for the start of a sentence: \CarriedGained is
    # "six" and \CarriedGainedCap is "Six".
    for name, value in list(numbers.items()):
        if isinstance(value, str) and value.isalpha() and value.islower():
            numbers[name + "Cap"] = capital(value)
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in numbers.items()]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    return numbers


# ---------------------------------------------------------------------------------------------
# The sentences of the report that state a fact without a number
# ---------------------------------------------------------------------------------------------

def check_the_claims(results, found):
    """Stops with an error if the data no longer supports a sentence of the report."""
    score, typical = results.score, results.typical
    runs = results.runs
    stuck = (STUCK["columns"], STUCK["bits"], STUCK["creatures"], STUCK["use"])  # the tile of the runs one kind short
    cells = [(columns, bits, creatures, use) for columns in COLUMNS for bits in BITS for creatures in CREATURES
             for use in USES]

    # Figure 1: "The more table columns, the more facts a model knows; the lines part as the
    # creatures or the bits grow, and lie lower with unequal use."
    def parted(bits, creatures, use):
        values = [typical(columns, bits, creatures, use) for columns in COLUMNS]
        return max(values) - min(values)
    for use in USES:
        assert all(parted(8, creatures, use) < 1 for creatures in CREATURES if (8, 8, creatures, use) != stuck)
        assert all(parted(128, creatures, use) > 30 for creatures in CREATURES[1:])
        assert all(parted(bits, 1024, use) > 8 for bits in BITS[1:])
        # ... "the more table columns, the more facts a model knows": no typical score falls.
        assert all(typical(wider, bits, creatures, use) > typical(narrower, bits, creatures, use) - 0.5
                   for bits in BITS for creatures in CREATURES for narrower, wider in zip(COLUMNS, COLUMNS[1:]))
    assert all(typical(*cell[:3], "unequal") <= typical(*cell[:3], "equal") + 1 for cell in cells if cell[:3] != stuck[:3])

    # Figure 5: "nearly half of all runs had learned at the first look."
    first = sum(run["reason"] == "learned" and run["ended"] == FIRST_LOOK for run in runs)
    assert 0.45 < first / len(runs) < 0.5

    # Finding 1: the comparisons that "stayed within half a point" were "all of them at 99 or more".
    for case in found["C1"]["gains"]:
        for narrower, wider in zip(COLUMNS, COLUMNS[1:]):
            low, high = score[(narrower, *case)], score[(wider, *case)]
            assert abs(high - low) > 0.5 or min(low, high) >= LEARNED
    # "C1 held, with about twice the cases it asked for."
    assert 1.8 < found["C1"]["gained"] / found["C1"]["needed"] < 2.3
    # "Where there is most to know, the rise is steady": 10 points or more with every doubling.
    for use in USES:
        heavy = [typical(columns, 128, 1024, use) for columns in COLUMNS]
        assert all(high - low >= 10 for low, high in zip(heavy, heavy[1:]))

    # Finding 2: "A wider body helped far less than more columns."
    assert found["C2"]["gained"] / found["C2"]["cases"] < 0.3 * found["C1"]["gained"] / found["C1"]["cases"]
    # Figure 8: the settings that count against C2 are "nearly all with 8 or 16 columns".
    counting = [case for case, gain in found["C2"]["gains"].items() if gain >= RISE]
    assert sum(case[0] <= 16 for case in counting) >= 0.85 * len(counting)
    # "The gain shrinks as the creatures grow": of the settings with room, the share that gained
    # 10 points or more falls from each number of creatures to the next.
    shares = []
    for creatures in CREATURES:
        room = [case for case in found["C2"]["gains"] if case[2] == creatures and score[(case[0], 4, *case[1:])] <= ROOM]
        shares.append(sum(found["C2"]["gains"][case] >= RISE for case in room) / len(room))
    assert shares == sorted(shares, reverse=True) and shares[-1] < 0.25

    # Figure 10: "Left of the stepped line the facts are known" (one tile excepted, which finding 5
    # explains); "right of it they fall away with many creatures, more so with unequal use."
    for use in USES:
        short = [(columns, bits, creatures, use) for columns in COLUMNS for bits in BITS for creatures in CREATURES
                 if bits <= columns and typical(columns, bits, creatures, use) < LEARNED]
        assert short == ([stuck] if use == STUCK["use"] else [])
    right = [(columns, bits) for columns in COLUMNS for bits in BITS if bits > columns]
    for columns, bits in right:
        for use in USES:  # more creatures, fewer facts known
            by_creatures = [typical(columns, bits, creatures, use) for creatures in CREATURES]
            assert all(more <= fewer + 1 for fewer, more in zip(by_creatures, by_creatures[1:]))

    # Finding 5: "Which repeat it strikes looks like chance, at any body width": in the corner of
    # Figure 11 a run one kind short stands beside a repeat of the same setting that knows every
    # fact, at three of the six body widths, the narrowest among them.
    group = pick(runs, **STUCK)
    beside = [body for body in BODIES if any(kinds_short(run) for run in pick(group, body=body))
              and any(run["facts"] >= LEARNED for run in pick(group, body=body))]
    assert len(beside) >= 3 and BODIES[0] in beside

    # Finding 7: "the three kinds of question give way before the plain facts", with 1,024
    # creatures at every number of columns and bits and with both word uses; and where there is
    # most to know, "not" is the lowest of the four.
    for use in USES:
        for columns in COLUMNS:
            for bits in BITS:
                at = {name: typical(columns, bits, SCORES_CREATURES, use, score=name) for name in SCORES}
                assert max(at["not_known"], at["questions_new"], at["questions_one"]) <= at["facts"] + 0.5
        at = {name: typical(64, 128, SCORES_CREATURES, use, score=name) for name in SCORES}
        assert at["not_known"] < min(at["questions_new"], at["questions_one"]) < at["facts"]
        # Figure 13: "Every part of the test holds for more bits with more columns" (both word uses).
        for name, _, _ in SCORE_PANELS:
            for bits in BITS:
                values = [typical(columns, bits, SCORES_CREATURES, use, score=name) for columns in COLUMNS]
                assert all(high > low - 2.5 for low, high in zip(values, values[1:])), (name, bits, use, values)

    # The four pages of every run. "With 8 columns the facts are known at 8 bits and fall away
    # from 16 bits on, sooner with more creatures and with unequal use."
    assert all(typical(8, 8, creatures, use) > 90 for creatures in CREATURES for use in USES)
    assert all(typical(8, bits, 1024, use) < 90 for bits in BITS[1:] for use in USES)
    # "With 16 columns the facts are known up to 16 bits, and with 16 creatures at every number of
    # bits." "With 32 columns ... up to 32 bits, and with 16 or 64 creatures at every number of bits."
    assert all(typical(16, bits, creatures, use) >= LEARNED for bits in (8, 16) for creatures in CREATURES for use in USES)
    assert all(run["facts"] >= LEARNED for bits in BITS for use in USES for run in results.by_cell[(16, bits, 16, use)])
    assert all(typical(32, bits, creatures, use) >= LEARNED for bits in (8, 16, 32) for creatures in CREATURES for use in USES)
    assert all(typical(32, bits, creatures, use) >= LEARNED for bits in BITS for creatures in (16, 64) for use in USES)
    # "With 64 columns nearly everything is known; only 1,024 creatures at 128 bits with unequal
    # use fall clearly short."
    low = {(bits, creatures, use) for bits in BITS for creatures in CREATURES for use in USES
           if typical(64, bits, creatures, use) < A2_BAR}
    assert low == {(128, 1024, "unequal")}


if __name__ == "__main__":
    counted = Results(read_stage("counted"))
    assert len(counted.runs) == 2880 and all(len(repeats) == 3 for repeats in counted.by_setting.values())
    found = verdicts(counted)
    use_shares = check_examples()
    check_model_sizes(counted)
    check_the_claims(counted, found)
    figure_answer(counted)
    figure_language()
    figure_columns()
    figure_dials(use_shares)
    figure_rule(counted)
    figure_scorecard(found)
    figure_by_columns(counted, found)
    figure_body_gain(counted, found)
    figure_size(counted)
    figure_load_map(counted)
    figure_plateau(counted)
    figure_carried(counted)
    figure_scores(counted)
    for number, columns in enumerate(COLUMNS):
        figure_every_run(counted, columns, f"fig{15 + number}_every_run_{columns}")
    written = write_numbers(counted, found)  # also draws the figure of the rate check (Figure 14)
    # Figures left from an earlier version of the report, under other names, are removed.
    for old in OUT.glob("*.pdf"):
        if old.stem not in WRITTEN:
            old.unlink()
    for name, value in written.items():
        if not name.endswith("Rows"):
            print(f"{name:28} {value}")

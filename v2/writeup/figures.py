"""Figures and numbers for the report on the second experiment (23 November 2026).

    .venv/bin/python v2/writeup/figures.py

Reads every counted run and every longer run in runs_v2/ and writes

    v2/writeup/figures/fig01 ... fig11 (PDF)   the eleven figures of the report, in its order
    v2/writeup/numbers.tex                     every number the text uses, as LaTeX macros

so that no result in the report is typed by hand and text and figures cannot disagree. The
sentences of the report that state a fact without giving a number are tested in check_the_claims():
if the data stopped supporting one of them, this script would stop with an error.

This script only reads the rest of the repository. It writes inside v2/writeup/ and nowhere else.


THE VISUAL SYSTEM (one for the whole report; decided once, here)
----------------------------------------------------------------
Number of creatures   one blue, light to dark: 16, 64, 256, 1,024. More creatures = darker.
                      In a key the four blues are plain short bars.
A creature's name     a pale blue box behind the word, in the drawings of the language and the model.
                      Blue has no other job: what a drawing points at (the true answer, the row that
                      is looked up, the guess) is dark grey.
Longer runs           orange, and nothing else is orange. A counted run is never orange.
Word use              equal use = solid line, filled marker;
                      unequal use = dashed line, hollow marker. Where the two have a row or a panel
                      each, its heading ("Equal use", "Unequal use") stands above its left end and
                      carries a sample of the line, so the heading is also the key (use_heading).
                      Where the two share a row (Figure 6), a key of the two markers stands below.
Counted repeats       where two dots of one setting would cover each other, they stand side by side
                      (Figure 10), so that no run is hidden behind another.
Body width            never a colour. Always a position along the horizontal, labelled "body width".
Bits per creature     never a colour. Always a panel, or a position, labelled with the bits.
Facts missed          grey tiles, pale to mid grey (Figure 1 only): the darker, the more facts missed.
                      The grey stays light enough for dark numbers on every tile.
Everything else       ink and greys. Text is always ink or grey, never a data colour.
Numbers               a score of a 64-bit setting or run is printed with one decimal everywhere.

The colours are the reference palette of the chart guide the report follows. They were checked
with its validator on white paper: the four blues as an ordered ramp (all checks pass), and the
orange against each blue for readers with colour-blindness (the closest pair is 23 units apart;
8 is the target).

Typeface: IBM Plex Sans for all figure text and IBM Plex Mono for words of the made-up language
(the files are in v2/writeup/fonts/; fonts/convert.py made the .ttf copies this script reads).
The report sets its headings and captions in the same Plex Sans and its text in Plex Serif. Every
figure is drawn at the size at which it is printed, 6.4 inches wide, so 8 pt here is 8 pt on
paper. No text in a figure is smaller than 8 pt.

How the drawing code is laid out: a figure is one page measured in inches from its top left
corner (page_of). Plots are placed on it with plot_at; everything else (titles, keys, the
drawings of the language and the model) is written straight onto the page with write, box, chip
and tile.
"""

import sys

sys.dont_write_bytecode = True  # this script only reads the repository: leave no cache files in it

import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files, not onto a screen
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "runs_v2"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------------------------
# What was run
# ---------------------------------------------------------------------------------------------

STAGES = {"main8": 8, "bits16": 16, "bits32": 32, "bits64": 64}  # folder -> bits per creature
BITS = list(STAGES.values())
CREATURES = [16, 64, 256, 1024]
BODIES = [8, 16, 32, 64, 128]
USES = ["equal", "unequal"]  # the run files call unequal use "zipf"
BODIES_64 = [8, 32, 128]  # the smaller grid that was run at 64 bits
CREATURES_64 = [16, 256, 1024]
SCORES = ["facts", "questions_one", "questions_new", "not_known", "questions_seen"]
KNOWN = 95  # the report's bar: a score of 95 or more counts as "known"
NEVER = float("inf")  # "had not learned the facts when training ended", as a number of steps
AXIS_START_SETTINGS = 70  # where the facts-known axis of Figure 10 starts
AXIS_START_CURVES = 70    # where the facts-known axis of Figure 11 starts (the same, so the two can be compared)
KEY_BARS = 8              # creatures shown in the small picture of equal and unequal use (Figure 5)
QUESTION_EXAMPLE = 83     # the worked example of a question score: this many of 100 right in each kind

# Facts that are not in the run files. They come from the plan (v2/PLAN.md, v3/PLAN.md), from the
# logs of the runs (logs/v2/) and from the cloud service's own meter, and are written into
# numbers.tex with everything else.
OUTSIDE_FACTS = {
    "CloudCost": "24.39",            # dollars, by Modal's meter (logs/v2/cost.log)
    "CloudMachines": "100",          # small machines at once, at most
    "RunsOnOwnComputer": "48",       # counted runs of the 16-bit and 32-bit stages run locally
    "PracticeEight": "15", "PracticeSixteen": "4", "PracticeThirtyTwo": "4", "PracticeSixtyFour": "6",
    "TimingRuns": "Four",            # runs_v2/speed; one more run in runs_v2/smoke tested the programs
    "PlannedSixtyFour": "120",       # the 64-bit stage as first planned
    "SentencesPerStep": "512",
    "ThirdRuns": f"{4 * 6 * 5 * 4 * 2 * 3:,}",  # columns x body widths x bits x creatures x uses x repeats
    # When the longer runs were started (logs/v2/local_queue_long.log: 18:56 on 22 November), 34 of
    # the 36 runs of the 64-bit stage were in. The two still running were the two repeats of 256
    # creatures, equal use, body width 128 (their files arrived at 19:22).
    "LongerStartRunsIn": "34",
}
PRACTICE_RUNS = sum(int(OUTSIDE_FACTS[f"Practice{name}"]) for name in ("Eight", "Sixteen", "ThirtyTwo", "SixtyFour"))

# ---------------------------------------------------------------------------------------------
# The visual system as code
# ---------------------------------------------------------------------------------------------

FULL = 6.4  # width of the text block in inches; every figure is drawn exactly this wide

INK = "#1a1a19"      # text and neutral marks
SECOND = "#52514e"   # axis labels, notes, and what a drawing points at
MUTED = "#898781"    # the quietest text
GRID = "#e1e0d9"     # hairlines
AXIS = "#c3c2b7"     # baselines
PAPER = "#ffffff"

CREATURE_BLUE = {16: "#86b6ef", 64: "#3987e5", 256: "#1c5cab", 1024: "#0d366b"}
LONGER = "#eb6834"   # the eight longer runs, and only they
TILE_PALE, TILE_DARK = "#ecebe6", "#878682"  # facts missed: none ... 25 or more of 100
PALE_BLUE = "#dbe8fa"  # soft fill behind a creature's name in the drawings
PALE_GREY = "#efeee9"  # soft fill behind other words and boxes
ZONE = "#ebeae4"       # "what was predicted" in the scorecard (a grey: blue means creatures)

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
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen"][round(number)]


def thousands(number):
    """160000 -> "160K" (K for thousand), for axes that reach ten thousand."""
    return f"{number // 1000}K" if number >= 1000 else f"{number}"


def exact(score):
    """A score as it is printed for a 64-bit setting or run: one decimal, and "100" for a full score."""
    return "100" if round(score, 1) == 100 else f"{score:.1f}"


def blend(light, dark, share):
    """A colour between two colours: share 0 gives the light one, 1 the dark one."""
    a, b = matplotlib.colors.to_rgb(light), matplotlib.colors.to_rgb(dark)
    return tuple(x + (y - x) * share for x, y in zip(a, b))


def missed_share(score):
    """Where a score sits between 100 facts known (0) and 75 or fewer (1)."""
    return min(max((100 - score) / 25, 0), 1)


def tile_colour(score):
    """The grey of a tile in Figure 1: the more facts missed, the darker."""
    return blend(TILE_PALE, TILE_DARK, missed_share(score))


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


def tile(page, x, y, width, height, fill, edge="none", line=0.0, gap=0.011):
    """One tile of a map, with a thin gap of paper around it."""
    page.add_patch(Rectangle((x + gap, y + gap), width - 2 * gap, height - 2 * gap,
                             facecolor=fill, edgecolor=edge, linewidth=line))


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


def creature_key(page, x, y, creatures=CREATURES, title="creatures in the language"):
    """The key for the number of creatures: the same in every figure that uses the blues. A
    plain short bar for each, so that the key says nothing about the style of a line."""
    write(page, x, y, title, SMALL, colour=SECOND, va="center")
    x += text_width(title) + 0.14
    for count in creatures:
        page.add_patch(Rectangle((x, y - 0.032), 0.2, 0.064, facecolor=CREATURE_BLUE[count], edgecolor="none"))
        write(page, x + 0.26, y, nice(count), SMALL, "medium", va="center")
        x += 0.26 + text_width(nice(count), SMALL, "medium") + 0.2
    return x


def use_heading(page, x, y, use, size=SMALL, weight="medium"):
    """The heading of a row or a panel that holds one word use, with a sample of how that use is
    drawn: a solid line and a filled dot for equal use, a dashed line and a hollow dot for
    unequal use. It stands above the left end of its row in every figure with two rows, so the
    reader finds it in the same place each time and needs no separate key for the line styles."""
    words = "Equal use" if use == "equal" else "Unequal use"
    write(page, x, y, words, size, weight, va="center")
    x += text_width(words, size, weight) + 0.09
    page.plot([x, x + 0.3], [y, y], color=SECOND, linewidth=1.5, linestyle="-" if use == "equal" else DASH)
    page.plot([x + 0.3], [y], marker="o", markersize=4.4, markerfacecolor=SECOND if use == "equal" else PAPER,
              markeredgecolor=SECOND, markeredgewidth=1.1)
    return x + 0.36


def column_header(page, x, y, bold, rest=""):
    """The heading of a column of panels on one line: "8 bits" in bold, then "40,000 steps"."""
    write(page, x, y, bold, HEAD, "semibold", va="center")
    if rest:
        write(page, x + text_width(bold, HEAD, "semibold") + 0.09, y + 0.004, rest, SMALL, colour=SECOND, va="center")


# ---------------------------------------------------------------------------------------------
# Reading the runs
# ---------------------------------------------------------------------------------------------

def read_run(path, bits):
    """One run, with the names the report uses."""
    summary = json.load(open(path))
    config, final, curve = summary["config"], summary["final"], summary["curve"]
    return {
        "bits": bits, "creatures": config["creatures"], "body": config["body"],
        "use": "unequal" if config["usage"] == "zipf" else "equal",
        "repeat": config["seed"], "steps": summary["steps"], "rate": config["lr"],
        "threads": config["threads"],  # processor threads the run was made with
        "hours": summary["minutes"] / 60, "parameters": summary["parameters"], "curve": curve,
        "at": {point["step"]: point for point in curve},  # the measurements by their step
        **{score: final[score] for score in SCORES},
    }


def read_counted():
    """The 476 counted runs."""
    return [read_run(path, bits) for stage, bits in STAGES.items()
            for path in sorted(glob.glob(str(RUNS / stage / "*" / "summary.json")))]


def read_longer():
    """The eight longer runs, by their setting: (word use, creatures, body width)."""
    found = [read_run(path, 64) for path in sorted(glob.glob(str(RUNS / "long" / "*" / "summary.json")))]
    return {(run["use"], run["creatures"], run["body"]): run for run in found}


def pick(runs, **wanted):
    """The runs that match: pick(runs, bits=64, creatures=1024) and so on."""
    return [run for run in runs if all(run[key] == value for key, value in wanted.items())]


def middle(values):
    """The median: half the values are higher, half lower. With two values, halfway between."""
    values = list(values)
    return statistics.median(values) if values else None


def middle_curve(group, score="facts"):
    """The middle of a group of runs at every step they were measured."""
    steps = [point["step"] for point in group[0]["curve"]]
    return steps, [middle(run["curve"][i][score] for run in group) for i in range(len(steps))]


def setting_score(counted, bits, use, creatures, body):
    """The facts known for one setting: the middle of its repeats."""
    return middle(run["facts"] for run in pick(counted, bits=bits, use=use, creatures=creatures, body=body))


def learning_time(runs):
    """How long learning took for a group of runs: the first measured step at which the middle
    of the group knew the facts, that is, at which the middle of their scores was 95 or more.
    NEVER if the middle had not got there when training ended. The answer is always a step at
    which scores were measured."""
    steps, scores = middle_curve(runs)
    return next((step for step, score in zip(steps, scores) if score >= KNOWN), NEVER)


def rows_in_use_never_changed():
    """The words whose row of the word table is in use and yet was never changed by training.
    Found by comparing the saved model of every counted run and every longer run (model.pt, read
    only) with the starting table. They are the same four in every run: the filler row, and the
    three words that only ever end a sentence (the full stop, "yes" and "no"). Nothing is
    guessed after them, so nothing pulls on their rows. Returns those words.

    The same pass checks what Figure 4 says: the rows of creatures that are not in a model's
    language are exactly as they started."""
    import torch  # only needed here, to open the saved models
    sys.path.insert(0, str(ROOT))
    from v2 import lang, net
    languages, found = {}, set()
    for stage in list(STAGES) + ["long"]:
        for path in sorted(glob.glob(str(RUNS / stage / "*" / "model.pt"))):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = saved["config"]
            key = (config["creatures"], config["kinds"], config["usage"])
            if key not in languages:
                language = lang.Language(config["creatures"], kinds=config["kinds"], usage=config["usage"])
                languages[key] = (language, net.word_table(language))
            language, start = languages[key]
            changed = (saved["model"]["word_table"].numpy() != start).any(axis=1)
            in_use = np.arange(len(changed)) < language.rows_in_use
            assert not changed[~in_use].any(), "a row that is not in use was changed"
            found.add(tuple(language.words[row] for row in np.flatnonzero(in_use & ~changed)))
    assert len(found) == 1, "the rows that never change are the same in every run"
    return list(found.pop())


def last_shared_step(counted, longer):
    """The last step before the end of the 64-bit counted runs at which a longer run was also
    measured. A longer run has no measurement at the counted end itself (160,000)."""
    counted_end = pick(counted, bits=64)[0]["steps"]
    return max(step for step in next(iter(longer.values()))["at"] if step <= counted_end)


# ---------------------------------------------------------------------------------------------
# Figure 1 (page one): facts known for every setting, and for the eight that were trained longer
# ---------------------------------------------------------------------------------------------

def figure_result_map(counted, longer, shared_step):
    width, height = FULL, 3.0
    figure, page = page_of(width, height)
    left, gap, tile_h = 0.70, 0.11, 0.18
    small, big, wide = 0.66, 1.11, 2.10  # width of a panel: 8 to 32 bits, 64 bits counted, 64 bits longer
    lefts = [left + i * (small + gap) for i in range(3)]
    lefts += [lefts[2] + small + gap, lefts[2] + small + gap + big + 0.16]
    long_steps = next(iter(longer.values()))["steps"]

    # Which panels hold counted runs and which the longer runs, and what a tile in each stands for.
    words = f"The {len(counted)} counted runs"
    write(page, lefts[0], 0.0, words, SMALL, "semibold")
    write(page, lefts[0] + text_width(words, SMALL, "semibold") + 0.1, 0.0,
          "a tile is the middle of its setting’s repeats", SMALL, colour=SECOND)
    page.plot([lefts[0], lefts[3] + big], [0.155, 0.155], color=AXIS, linewidth=0.8)
    words = f"The {len(longer)} longer runs"
    write(page, lefts[4], 0.0, words, SMALL, "semibold")
    write(page, lefts[4] + text_width(words, SMALL, "semibold") + 0.1, 0.0, "each a new run", SMALL, colour=SECOND)
    page.plot([lefts[4], lefts[4] + wide], [0.155, 0.155], color=LONGER, linewidth=1.6)

    headers = [(f"{bits} bits", f"{nice(pick(counted, bits=bits)[0]['steps'])} steps") for bits in BITS]
    headers.append(("64 bits", f"step {nice(shared_step)} → step {nice(long_steps)}"))
    for x, (first, second) in zip(lefts, headers):
        write(page, x, 0.205, first, HEAD, "semibold")
        write(page, x, 0.365, second, SMALL, colour=SECOND)

    row_tops = [0.72, 1.67]
    row_names = ["Equal use: every creature appears equally often",
                 "Unequal use: a few creatures appear far more often than the rest"]
    for use, top, name in zip(USES, row_tops, row_names):
        write(page, lefts[0], top - 0.17, name, SMALL, "medium")
        write(page, 0.07, top + 2 * tile_h, "creatures", SMALL, colour=SECOND, ha="center", va="center",
              rotation=90)
        for row, creatures in enumerate(CREATURES):
            write(page, left - 0.07, top + (row + 0.5) * tile_h, nice(creatures), SMALL, colour=SECOND,
                  ha="right", va="center")

        # 8, 16 and 32 bits: every body width, small tiles without numbers (the line "facts known"
        # under the panels gives their scores).
        for x, bits in zip(lefts, BITS[:3]):
            for row, creatures in enumerate(CREATURES):
                for column, body in enumerate(BODIES):
                    score = setting_score(counted, bits, use, creatures, body)
                    tile(page, x + column * small / 5, top + row * tile_h, small / 5, tile_h, tile_colour(score))

        # 64 bits: the smaller grid, larger tiles with the score in them.
        for row, creatures in enumerate(CREATURES):
            y = top + row * tile_h
            if creatures not in CREATURES_64:
                write(page, lefts[3] + big / 2, y + tile_h / 2, "not run at 64 bits", SMALL, colour=MUTED,
                      ha="center", va="center")
                write(page, lefts[4] + wide / 2, y + tile_h / 2, "not run", SMALL, colour=MUTED,
                      ha="center", va="center")
                continue
            for column, body in enumerate(BODIES_64):
                x = lefts[3] + column * big / 3
                score = setting_score(counted, 64, use, creatures, body)
                tile(page, x, y, big / 3, tile_h, tile_colour(score))
                write(page, x + big / 6, y + tile_h / 2, exact(score), SMALL, "medium", ha="center", va="center")
                # The same setting in the panel of the longer runs: where that new run stood near
                # the counted length, and where it ended.
                x = lefts[4] + column * wide / 3
                centre = x + wide / 6
                run = longer.get((use, creatures, body))
                if run is None:
                    tile(page, x, y, wide / 3, tile_h, PAPER, edge=GRID, line=0.7, gap=0.018)
                    write(page, centre, y + tile_h / 2, "–", SMALL, colour=MUTED, ha="center", va="center")
                else:
                    tile(page, x, y, wide / 3, tile_h, tile_colour(run["facts"]), edge=LONGER, line=1.5, gap=0.02)
                    write(page, centre - 0.058, y + tile_h / 2, exact(run["at"][shared_step]["facts"]), SMALL,
                          ha="right", va="center")
                    write(page, centre, y + tile_h / 2, "→", SMALL, ha="center", va="center")
                    write(page, centre + 0.058, y + tile_h / 2, exact(run["facts"]), SMALL, "semibold",
                          ha="left", va="center")

    # Under the panels: the body widths, and the range of the scores in each panel.
    y = row_tops[1] + 4 * tile_h + 0.04
    write(page, left - 0.07, y, "body width", SMALL, colour=SECOND, ha="right")
    for x in lefts[:3]:
        for column, body in enumerate(BODIES):
            if body in BODIES_64:  # every second label: the tiles are narrow
                write(page, x + (column + 0.5) * small / 5, y, str(body), SMALL, colour=SECOND, ha="center")
    for x, panel in ((lefts[3], big), (lefts[4], wide)):
        for column, body in enumerate(BODIES_64):
            write(page, x + (column + 0.5) * panel / 3, y, str(body), SMALL, colour=SECOND, ha="center")
    y += 0.15
    write(page, left - 0.07, y, "facts known", SMALL, colour=SECOND, ha="right")
    for x, words in zip(lefts, result_map_ranges(counted, longer)):
        write(page, x, y, words, SMALL, "medium")

    # The key: what the greys, the orange frame and the dash mean.
    y, half = height - tile_h, tile_h / 2
    write(page, 0, y + half, "Facts known, of 100", SMALL, colour=SECOND, va="center")
    x = text_width("Facts known, of 100") + 0.1
    for score in (100, 95, 90, 85, 80, 75):
        tile(page, x, y, 0.3, tile_h, tile_colour(score))
        write(page, x + 0.15, y + half, f"{score}", SMALL, "medium", ha="center", va="center")
        x += 0.3
    x += 0.3
    tile(page, x, y, 0.32, tile_h, PAPER, edge=LONGER, line=1.5, gap=0.02)
    write(page, x + 0.38, y + half, "a longer run", SMALL, colour=SECOND, va="center")
    x += 0.38 + text_width("a longer run") + 0.3
    tile(page, x, y, 0.32, tile_h, PAPER, edge=GRID, line=0.7, gap=0.018)
    write(page, x + 0.16, y + half, "–", SMALL, colour=MUTED, ha="center", va="center")
    write(page, x + 0.38, y + half, "not trained longer", SMALL, colour=SECOND, va="center")
    save(figure, "fig01_result_map")


def result_map_ranges(counted, longer):
    """The line "facts known" under the five panels of Figure 1: the lowest and the highest
    setting of each stage, and where the longer runs ended."""
    def spread(scores):
        low, high = exact(min(scores)), exact(max(scores))
        return f"all {low}" if low == high else f"{low} to {high}"
    words = []
    for bits in BITS:
        stage = pick(counted, bits=bits)
        settings = {(run["use"], run["creatures"], run["body"]) for run in stage}
        words.append(spread([setting_score(counted, bits, *setting) for setting in settings]))
    words.append("ended at " + spread([run["facts"] for run in longer.values()]))
    return words


# ---------------------------------------------------------------------------------------------
# Figure 2: the language. A creature, its facts, and the sentences made from them.
# ---------------------------------------------------------------------------------------------

# The examples are from the language with 16 creatures and 4 kinds of fact. check_examples()
# below tests every one of them against the language's own programs.
EXAMPLE_FACTS = [("colour", "are", ["red", "brown", "green", "black"], "black"),
                 ("food", "eat", ["meat", "plants", "insects", "seeds"], "meat"),
                 ("home", "live in", ["woods", "fields", "rivers", "caves"], "fields"),
                 ("wakes at", "wake at", ["dawn", "noon", "dusk", "night"], "dawn")]
EXAMPLE_SENTENCES = [("a statement", "foxes , crows and hens are black ."),
                     ("a “not” statement", "foxes and bats do not eat seeds ."),
                     ("a question with its answer", "are foxes and bears black ? no")]
EXAMPLE_HELD = ("foxes", "owls")  # a pair that never shares a training sentence
EXAMPLE_TEST = "do foxes and owls live in fields ? yes"
EXAMPLE_NAMES = {"foxes", "crows", "hens", "bats", "bears", "owls"}


def check_examples():
    """Every example the drawings show is a true sentence of the real language, and the pair they
    call held back really is. Uses the experiment's own program (read only)."""
    sys.path.insert(0, str(ROOT))
    from v2 import lang
    language = lang.Language(16, kinds=4, usage="equal")
    number_of = {word: i for i, word in enumerate(language.words)}
    for sentence in [text for _, text in EXAMPLE_SENTENCES] + [EXAMPLE_TEST, "are foxes and owls black ? no",
                                                              "are foxes black ? yes"]:
        assert lang.is_true(language, [number_of[word] for word in sentence.split()]), sentence
    held = {(language.names[a], language.names[b]) for a, b in language.held_pairs}
    assert EXAMPLE_HELD in held
    for a, b in (("foxes", "crows"), ("foxes", "hens"), ("crows", "hens"), ("foxes", "bats"), ("foxes", "bears")):
        assert (a, b) not in held and (b, a) not in held
    fox = language.names.index("foxes")
    for kind, (_, _, values, true_value) in enumerate(EXAMPLE_FACTS):
        assert values[language.facts[fox, kind]] == true_value
    # The worked example of a question score in the caption of Figure 3: so many of 100 "yes"
    # questions right and as many of 100 "no" questions right.
    right = np.array([True] * QUESTION_EXAMPLE + [False] * (100 - QUESTION_EXAMPLE))
    exam = {"right": np.zeros(200, dtype=int), "answer": np.array([1] * 100 + [0] * 100), "wrong": np.ones(200, dtype=int)}
    score, _ = lang.score_of_picks(np.where(np.concatenate([right, right]), 0, 1), exam)
    assert round(score) == 2 * QUESTION_EXAMPLE - 100
    # The least used of 1,024 creatures appears 1,024 times less often than the most used.
    weights = lang.creature_weights(1024, "zipf")
    assert round(weights[0] / weights[-1]) == 1024
    return len(language.held_pairs), 16 * 15 // 2


def figure_language():
    width, height = FULL, 2.5
    figure, page = page_of(width, height)

    # Left: one creature and what there is to know about it.
    write(page, 0, 0, "A creature and its facts", HEAD, "semibold")
    box(page, 0, 0.24, 3.07, 1.3, fill=PAPER, edge=AXIS, line=0.7, round_=0.06)
    chip(page, 0.12, 0.33, "foxes", fill=PALE_BLUE, weight="mono-medium")
    write(page, 0.62, 0.425, "one creature is one word", SMALL, colour=SECOND, va="center")
    y = 0.63
    for kind, _, values, true_value in EXAMPLE_FACTS:
        write(page, 0.12, y + 0.092, kind, SMALL, colour=SECOND, va="center")
        for i, value in enumerate(values):  # the four possible answers, the true one dark
            chosen = value == true_value
            x = 0.70 + i * 0.578
            box(page, x, y, 0.555, 0.185, fill=SECOND if chosen else PAPER,
                edge="none" if chosen else GRID, round_=0.035)
            write(page, x + 0.2775, y + 0.0965, value, SMALL, "mono-medium" if chosen else "mono",
                  PAPER if chosen else SECOND, ha="center", va="center_baseline")
        y += 0.225
    # What a bit is, and the four amounts of knowledge that were tried.
    write(page, 0, 1.63, "One fact is one of four answers. That takes 2 bits:", SMALL, colour=SECOND)
    write(page, 0, 1.77, "a bit is one yes-or-no choice, and two choices pick one of four.", SMALL, colour=SECOND)
    y = 1.99
    for bits in BITS:
        kinds = bits // 2
        write(page, 0.0, y + 0.045, f"{bits} bits", SMALL, "medium", va="center")
        for i in range(kinds):  # one small square for each kind of fact
            page.add_patch(Rectangle((0.52 + i * 0.05, y), 0.039, 0.09, facecolor=SECOND, edgecolor="none"))
        write(page, 0.52 + kinds * 0.05 + 0.05, y + 0.045, f"{kinds} kinds of fact", SMALL, colour=SECOND,
              va="center")
        y += 0.14

    # Right: the sentences. The model reads these and nothing else.
    x0 = 3.34
    write(page, x0, 0, "What a model reads: sentences only", HEAD, "semibold")
    write(page, x0, 0.19, "One box is one word. A comma counts as a word too.", SMALL, colour=SECOND)
    y = 0.42
    for name, sentence in EXAMPLE_SENTENCES:
        write(page, x0, y, name, SMALL, "medium")
        chips(page, x0, y + 0.16, sentence, EXAMPLE_NAMES)
        y += 0.425
    page.plot([x0, width], [y + 0.02, y + 0.02], color=GRID, linewidth=0.7)
    y += 0.09
    write(page, x0, y, "kept for the test", SMALL, "medium")
    write(page, x0, y + 0.15, "A tenth of all pairs of creatures never share a training", SMALL, colour=SECOND)
    write(page, x0, y + 0.29, "sentence. Foxes and owls are such a pair. The test asks:", SMALL, colour=SECOND)
    chips(page, x0, y + 0.475, EXAMPLE_TEST, EXAMPLE_NAMES)
    save(figure, "fig02_language")


# ---------------------------------------------------------------------------------------------
# Figure 3: how a model is scored. Four small tests.
# ---------------------------------------------------------------------------------------------

def figure_scoring():
    width, height = FULL, 1.9
    figure, page = page_of(width, height)
    # The two cards with four choices are a little wider than the two with "yes" and "no".
    card_widths = [1.585, 1.515, 1.515, 1.585]
    gap, inset = (width - sum(card_widths)) / 3, 0.075
    cards = [
        (["Facts known"], "the main score", ["foxes are …"], ["red", "brown", "green", "black"], "black", True,
         ["Right if black gets the", "highest score of the four."], 25),
        (["Questions about", "one creature"], "", ["are foxes black ? …"], ["yes", "no"], "yes", True,
         ["Right if the true answer", "gets the higher score."], 0),
        (["Questions about", "new combinations"], "", ["are foxes and owls", "black ? …"], ["yes", "no"], "no", True,
         ["The same, about creatures", "never read together."], 0),
        (["“Not” known"], "", ["foxes are not …"], ["red", "brown", "green", "black"], "black", False,
         ["Right if black gets the", "lowest score of the four."], 25),
    ]
    x0 = 0.0
    for card_w, (title, tag, prompt, options, marked, highest, rule, guess) in zip(card_widths, cards):
        box(page, x0, 0, card_w, height, fill=PAPER, edge=AXIS, line=0.7, round_=0.06)
        for k, line in enumerate(title):
            write(page, x0 + inset, 0.08 + k * 0.14, line, SMALL, "semibold")
        if tag:
            write(page, x0 + inset, 0.22, tag, SMALL, colour=SECOND)
        y = 0.44
        for line in prompt:
            chips(page, x0 + inset, y, line, EXAMPLE_NAMES, pad=0.026, between=0.022, height=0.18)
            y += 0.21
        x = x0 + inset
        for option in options:
            # What the model chooses between. The choice that must score highest is dark; the one
            # that must score lowest ("not") is its opposite, an outline.
            chosen = option == marked
            fill = SECOND if chosen and highest else PAPER
            edge = "none" if chosen and highest else (INK if chosen else GRID)
            ink = PAPER if chosen and highest else (INK if chosen else SECOND)
            x = chip(page, x, 0.9, option, fill=fill, ink=ink, edge=edge, line=0.9 if chosen else 0.6,
                     pad=0.022, height=0.18, weight="mono-medium" if chosen else "mono") + 0.018
        for k, line in enumerate(rule):
            write(page, x0 + inset, 1.15 + k * 0.135, line, SMALL, colour=SECOND)
        # Where blind guessing lands on the scale from 0 to 100: the label stands on its tick.
        y, x1, x2 = 1.64, x0 + inset, x0 + card_w - inset
        page.plot([x1, x2], [y, y], color=AXIS, linewidth=2.2, solid_capstyle="butt")
        gx = x1 + (x2 - x1) * guess / 100
        page.plot([gx, gx], [y - 0.06, y + 0.06], color=INK, linewidth=1.4)
        write(page, gx - 0.01, y - 0.075, f"a guess scores {guess}", SMALL, colour=SECOND, va="bottom")
        write(page, x1, y + 0.075, "0", SMALL, colour=SECOND)
        write(page, x2, y + 0.075, "100", SMALL, colour=SECOND, ha="right")
        x0 += card_w + gap
    save(figure, "fig03_scoring")


# ---------------------------------------------------------------------------------------------
# Figure 4: the model. Word table, body, and what "body width" is.
# ---------------------------------------------------------------------------------------------

def table_rows(counted, bits, creatures):
    """The rows of the word table that a language uses: its creatures, every other word, and the
    one filler row that pads short sentences. Training changes only these rows, and among them
    all but four (rows_in_use_never_changed)."""
    return pick(counted, bits=bits, creatures=creatures)[0]["parameters"]["table_rows_in_use"]


def figure_model(counted):
    width, height = FULL, 3.08
    figure, page = page_of(width, height)
    small_table, whole_table = table_rows(counted, 8, 16), table_rows(counted, 8, 1024)
    largest_table = table_rows(counted, 64, 1024)

    def arrow(x0, y0, x1, y1, colour=SECOND):
        page.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                       color=colour, linewidth=0.9, shrinkA=0, shrinkB=0))

    flow_y = 1.01  # the height at which the arrows run from step to step

    # 1. The words read so far.
    write(page, 0, 0, "1  Words so far", HEAD, "semibold")
    write(page, 0, 0.17, "one at a time", SMALL, colour=SECOND)
    chip(page, 0.02, 0.62, "foxes", fill=PALE_BLUE, weight="mono-medium")
    chip(page, 0.02, 0.9, "are", fill=PALE_GREY, weight="mono-medium")
    box(page, 0.02, 1.18, 0.3, 0.185, fill=PAPER, edge=AXIS, round_=0.035)
    write(page, 0.17, 1.2765, "…", SMALL, "mono", SECOND, ha="center", va="center_baseline")
    write(page, 0.02, 1.43, "the next word,", SMALL, colour=SECOND)
    write(page, 0.02, 1.57, "to be guessed", SMALL, colour=SECOND)

    # 2. The word table: a row of 32 numbers for every word.
    table_x, cell, row_h, top = 1.78, 0.039, 0.118, 0.42
    write(page, 1.08, 0, "2  The word table", HEAD, "semibold")
    write(page, 1.08, 0.17, "a row of 32 numbers for every word", SMALL, colour=SECOND)
    rows = ["and", "not", "are", "black", "…", "foxes", "bears", "wolves", "…", "dubos"]
    shades = np.random.default_rng(3).uniform(0.12, 0.85, size=(len(rows), 32))
    for r, word in enumerate(rows):
        y = top + r * row_h
        if word == "…":
            for k in (-1, 0, 1):  # three dots: rows that are left out of the drawing
                page.plot([table_x + 16 * cell], [y + row_h / 2 + k * 0.03], marker="o", markersize=1.3, color=MUTED)
            continue
        looked_up = word in ("foxes", "are")  # the rows of the two words read so far: drawn dark
        write(page, table_x - 0.06, y + row_h / 2 + 0.003, word, SMALL, "mono-semibold" if looked_up else "mono",
              INK if looked_up else SECOND, ha="right", va="center_baseline")
        for c in range(32):
            shade = shades[r, c]
            fill = blend("#cfcec8", INK, shade) if looked_up else blend(PAPER, "#c9c8c1", shade)
            page.add_patch(Rectangle((table_x + c * cell, y + 0.014), cell - 0.006, row_h - 0.028,
                                     facecolor=fill, edgecolor="none"))
    table_right = table_x + 32 * cell
    arrow(0.62, flow_y, 1.2, flow_y)

    # 3. The body: two layers.
    body_x, body_w, body_top, body_h = 3.42, 1.62, 0.42, 1.18
    write(page, body_x, 0, "3  The body", HEAD, "semibold")
    write(page, body_x, 0.17, "two layers work on those rows", SMALL, colour=SECOND)
    box(page, body_x, body_top, body_w, body_h, fill=PALE_GREY, round_=0.07)
    for i in range(2):
        x = body_x + 0.09 + i * 0.755
        write(page, x, body_top + 0.07, f"layer {i + 1}", SMALL, "medium", SECOND)
        box(page, x, body_top + 0.27, 0.685, 0.37, fill=PAPER, round_=0.04)
        write(page, x + 0.3425, body_top + 0.385, "attention", SMALL, ha="center", va="center")
        write(page, x + 0.3425, body_top + 0.525, "looks back", SMALL, colour=SECOND, ha="center", va="center")
        box(page, x, body_top + 0.72, 0.685, 0.37, fill=PAPER, round_=0.04)
        write(page, x + 0.3425, body_top + 0.835, "middle part", SMALL, "medium", ha="center", va="center")
        write(page, x + 0.3425, body_top + 0.975, "its neurons", SMALL, colour=SECOND, ha="center", va="center")
    arrow(table_right + 0.08, flow_y, body_x - 0.07, flow_y)

    # 4. The guess.
    out_x = 5.4
    write(page, out_x, 0, "4  Its guess", HEAD, "semibold")
    write(page, out_x, 0.17, "a score for each", SMALL, colour=SECOND)
    write(page, out_x, 0.31, "possible next word", SMALL, colour=SECOND)
    arrow(body_x + body_w + 0.07, flow_y, out_x - 0.05, flow_y)
    for i, (word, score) in enumerate((("black", 1.0), ("red", 0.16), ("brown", 0.22), ("green", 0.1))):
        y = 0.66 + i * 0.2
        write(page, out_x + 0.02, y + 0.07, word, SMALL, "mono-semibold" if i == 0 else "mono",
              INK if i == 0 else SECOND, va="center_baseline")
        page.add_patch(Rectangle((out_x + 0.47, y), 0.48 * score, 0.13,
                                 facecolor=SECOND if i == 0 else AXIS, edgecolor="none"))
    write(page, out_x, 1.5, "the highest score", SMALL, colour=SECOND)
    write(page, out_x, 1.64, "is its guess", SMALL, colour=SECOND)

    # Below, left: the word table is the same everywhere, and a language changes only its own rows.
    # "In use" are the rows a model can meet: its creatures, the other words, and one filler row
    # that pads short sentences. (Training changes all of them but four; Appendix A.)
    y = 1.9
    write(page, 0, y, "One word table for every model", SMALL, "medium")
    lines = ["A row for each of the 1,024 creature names and for every",
             f"other word ({nice(whole_table)} rows at 8 bits, {nice(largest_table)} at 64 bits), with the",
             "same starting numbers in every model. Training changes",
             "only the rows that a model’s language uses. At 8 bits:"]
    for k, line in enumerate(lines):
        write(page, 0, y + 0.15 + k * 0.14, line, SMALL, colour=SECOND)
    bar_w, bar_h = 1.5, 0.085
    for i, (creatures, rows_used) in enumerate(((16, small_table), (1024, whole_table))):
        y_bar = y + 0.82 + i * 0.19
        page.add_patch(Rectangle((0, y_bar), bar_w, bar_h, facecolor=PALE_GREY, edgecolor="none"))
        page.add_patch(Rectangle((0, y_bar), bar_w * rows_used / whole_table, bar_h,
                                 facecolor=CREATURE_BLUE[creatures], edgecolor="none"))
        words = (f"{nice(creatures)} creatures: {rows_used} rows in use" if creatures == 16
                 else f"{nice(creatures)} creatures: all {nice(rows_used)} in use")
        write(page, bar_w + 0.08, y_bar + bar_h / 2, words, SMALL, va="center")

    # Below, right: what "body width" is. A word's 32 numbers run straight on along the top
    # line. A middle part is a side path under that line: the numbers go in, are worked on by
    # its neurons (the waist of the shape: 8 or 128 of them), come out as 32 numbers again, and
    # are ADDED to the line at the plus. So a narrow waist limits what a middle part can work
    # out. It does not limit what gets past it (v2/net.py, Layer.forward: x = x + middle(x)).
    page.plot([body_x + 0.4325, body_x + 0.4325], [body_top + body_h + 0.03, y - 0.05], color=AXIS, linewidth=0.8)
    write(page, body_x, y, "Body width: the number of neurons in a middle part", SMALL, "medium")
    write(page, body_x, y + 0.15, "A word’s 32 numbers run on along the top line. A middle", SMALL, colour=SECOND)
    write(page, body_x, y + 0.29, "part works on a copy and adds its result to them (+).", SMALL, colour=SECOND)
    unit = 0.085                 # the drawn height of 32 numbers
    line_y, side_y = y + 0.545, y + 0.815  # the height of the top line, and of the middle part under it
    for label, neurons, x in (("body width 8", 8, body_x), ("body width 128", 128, body_x + 1.56)):
        line_from, line_to = x + 0.17, x + 1.23      # the top line: 32 numbers in, 32 numbers out
        down, shape_from, shape_to, up = x + 0.26, x + 0.37, x + 0.99, x + 1.10
        waist = (shape_from + shape_to) / 2
        write(page, x, line_y, "32", SMALL, colour=SECOND, va="center")
        arrow(line_from, line_y, line_to, line_y)
        write(page, line_to + 0.04, line_y, "32", SMALL, colour=SECOND, va="center")
        # The side path: down from the line, through the middle part, and up to the plus.
        page.plot([down, down, shape_from - 0.02], [line_y, side_y, side_y], color=SECOND, linewidth=0.9,
                  solid_capstyle="butt", solid_joinstyle="miter")
        page.plot([shape_to + 0.02, up], [side_y, side_y], color=SECOND, linewidth=0.9, solid_capstyle="butt")
        arrow(up, side_y, up, line_y + 0.055)
        # The middle part: as tall as 32 numbers at both ends, as tall as its neurons at the waist.
        tall = unit * neurons / 32
        shape = [(shape_from, side_y - unit / 2), (waist, side_y - tall / 2), (shape_to, side_y - unit / 2),
                 (shape_to, side_y + unit / 2), (waist, side_y + tall / 2), (shape_from, side_y + unit / 2)]
        page.add_patch(Polygon(shape, closed=True, facecolor=PALE_GREY, edgecolor="none"))
        for edge in (shape_from, shape_to):  # the 32 numbers going in and coming out
            page.plot([edge, edge], [side_y - unit / 2, side_y + unit / 2], color=SECOND, linewidth=1.6,
                      solid_capstyle="butt")
        page.plot([waist, waist], [side_y - tall / 2, side_y + tall / 2], color=INK, linewidth=2.4,
                  solid_capstyle="butt")  # the neurons
        write(page, waist + 0.06, side_y, str(neurons), SMALL, "semibold", va="center")
        # The plus: where the result of the middle part is added to the numbers on the line.
        page.plot([up], [line_y], marker="o", markersize=7.6, markerfacecolor=PAPER, markeredgecolor=SECOND,
                  markeredgewidth=0.9, zorder=5)
        page.plot([up - 0.028, up + 0.028], [line_y, line_y], color=INK, linewidth=0.9, zorder=6, solid_capstyle="butt")
        page.plot([up, up], [line_y - 0.028, line_y + 0.028], color=INK, linewidth=0.9, zorder=6, solid_capstyle="butt")
        write(page, waist, height - 0.01, label, SMALL, colour=SECOND, ha="center", va="bottom")
    save(figure, "fig04_model")


# ---------------------------------------------------------------------------------------------
# Figure 5: the runs behind the results. One dot is one run.
# ---------------------------------------------------------------------------------------------

def figure_runs_map(counted, longer):
    width, height = FULL, 2.64
    figure, page = page_of(width, height)
    left, column_w, row_h, gap, top = 0.72, 0.25, 0.3, 0.16, 0.26
    panel_w = 5 * column_w
    pitch = 0.053  # distance between two dots, in inches
    long_steps = next(iter(longer.values()))["steps"]
    for i, bits in enumerate(BITS):
        x0 = left + i * (panel_w + gap)
        stage = pick(counted, bits=bits)
        repeats = len({run["repeat"] for run in stage})
        write(page, x0, 0.0, f"{bits} bits per creature", HEAD, "semibold")
        for row, creatures in enumerate(CREATURES):
            for column, body in enumerate(BODIES):
                cell = pick(stage, creatures=creatures, body=body)
                x, y = x0 + column * column_w, top + row * row_h
                if not cell:
                    continue  # this setting was cut from the stage: leave the paper empty
                # A setting is an outlined box. (A grey fill would look like the tiles of Figure 1,
                # where grey means a score.)
                tile(page, x, y, column_w, row_h, PAPER, edge=AXIS, line=0.6, gap=0.016)
                centre_x, centre_y = x + column_w / 2, y + row_h / 2
                for use, side in (("equal", -1), ("unequal", 1)):
                    count = len(pick(cell, use=use))
                    for k in range(count):  # one dot for each repeat
                        dot_y = centre_y + (k - (count - 1) / 2) * pitch
                        page.plot([centre_x + side * 0.036], [dot_y], marker="o", markersize=3.1,
                                  markerfacecolor=INK if use == "equal" else PAPER, markeredgecolor=INK,
                                  markeredgewidth=0.7)
                    if bits == 64 and (use, creatures, body) in longer:
                        # The longer run of this setting: a diamond under the dots of its word use.
                        page.plot([centre_x + side * 0.036], [centre_y + 0.088], marker="D", markersize=3.3,
                                  markerfacecolor=LONGER if use == "equal" else PAPER,
                                  markeredgecolor=LONGER, markeredgewidth=0.9)
        for column, body in enumerate(BODIES):
            write(page, x0 + (column + 0.5) * column_w, top + 4 * row_h + 0.04, str(body), SMALL, colour=SECOND,
                  ha="center")
        y = top + 4 * row_h + 0.25
        write(page, x0, y, f"{len(stage)} runs" + (f" + {len(longer)} longer" if bits == 64 else ""), SMALL,
              "semibold")
        write(page, x0, y + 0.145, f"{repeats} repeats, {nice(stage[0]['steps'])} steps", SMALL, colour=SECOND)
        if bits == 64:
            write(page, x0, y + 0.285, f"longer: {nice(long_steps)} steps", SMALL, colour=SECOND)
    for row, creatures in enumerate(CREATURES):
        write(page, left - 0.07, top + (row + 0.5) * row_h, nice(creatures), SMALL, colour=SECOND, ha="right",
              va="center")
    write(page, 0.07, top + 2 * row_h, "creatures", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left - 0.07, top + 4 * row_h + 0.04, "body width", SMALL, colour=SECOND, ha="right")

    # The key, first line: the four kinds of mark.
    y = height - 0.31
    x = 0.0
    marks = [("o", INK, True, "counted run, equal use"), ("o", INK, False, "counted run, unequal use"),
             ("D", LONGER, True, "longer run, equal use"),
             ("D", LONGER, False, "longer run, unequal use")]
    for marker, colour, filled, words in marks:
        page.plot([x + 0.035], [y], marker=marker, markersize=3.6, markerfacecolor=colour if filled else PAPER,
                  markeredgecolor=colour, markeredgewidth=0.8)
        write(page, x + 0.12, y, words, SMALL, colour=SECOND, va="center")
        x += 0.12 + text_width(words) + 0.2
    # Second line: what equal and unequal use mean, as a small picture of how often the most used
    # creatures appear.
    y = height - 0.08
    words = f"How often the {in_words(KEY_BARS)} most used creatures appear:"
    write(page, 0, y, words, SMALL, colour=SECOND, va="center")
    x = text_width(words) + 0.14
    for use, words in (("equal", "with equal use"), ("unequal", "with unequal use")):
        for rank in range(KEY_BARS):
            share = 1.0 if use == "equal" else 1 / (rank + 1)
            page.add_patch(Rectangle((x + rank * 0.04, y + 0.06 - 0.14 * share), 0.028, 0.14 * share,
                                     facecolor=SECOND, edgecolor="none"))
        x += KEY_BARS * 0.04 + 0.05
        write(page, x, y, words, SMALL, colour=SECOND, va="center")
        x += text_width(words) + 0.3
    save(figure, "fig05_runs_map")


# ---------------------------------------------------------------------------------------------
# Figure 6: the four predictions, each next to what was measured, at every stage
# ---------------------------------------------------------------------------------------------

def prediction_measures(counted, bits, use):
    """The four quantities the predictions are about, for one stage and one word use."""
    bodies = BODIES if bits != 64 else BODIES_64
    score = lambda creatures, body: setting_score(counted, bits, use, creatures, body)
    return {
        "P1": min(score(16, body) for body in bodies),             # lowest score with 16 creatures
        "P2": score(16, 8) - score(1024, 8),                       # how far 1,024 fall below 16, body 8
        "P3": score(1024, 128) - score(1024, 8),                   # what the widest body gains
        "P4": max(abs(score(16, body) - score(1024, body)) for body in bodies),  # the largest gap
    }


def predictions_at_eight_bits(counted, use):
    """The rule exactly as fixed beforehand: a prediction holds for a setting when it holds in at
    least 4 of its 5 repeats. Repeat i of one setting is compared with repeat i of the other."""
    facts = {(run["creatures"], run["body"], run["repeat"]): run["facts"] for run in pick(counted, bits=8, use=use)}
    repeats = range(5)
    most = lambda checks: sum(checks) >= 4
    return {
        "P1": all(most(facts[16, body, r] >= 95 for r in repeats) for body in BODIES),
        "P2": most(facts[16, 8, r] - facts[1024, 8, r] >= 10 for r in repeats),
        "P3": most(facts[1024, 128, r] - facts[1024, 8, r] >= 10 for r in repeats),
        "P4": all(most(abs(facts[16, body, r] - facts[1024, body, r]) <= 10 for r in repeats) for body in BODIES),
    }


def verdict_of(held):
    """The verdict rule of the plan, word for word. held[use][prediction] is True or False."""
    if all(held[use][p] for use in USES for p in ("P1", "P2", "P3")):
        return "supported"
    if all(held[use]["P4"] for use in USES):
        return "contradicted"
    return "mixed"


def verdict_at_eight_bits(counted):
    """The verdict at 8 bits, the stage the rule was written for (5 repeats, 4 of them must agree)."""
    return verdict_of({use: predictions_at_eight_bits(counted, use) for use in USES})


def verdict_from_middles(counted, bits):
    """What the same rule would say of a later stage, read on the middle of its repeats (those
    stages have fewer than five repeats, so "4 of 5" cannot be applied to them)."""
    held = {}
    for use in USES:
        m = prediction_measures(counted, bits, use)
        held[use] = {"P1": m["P1"] >= 95, "P2": m["P2"] >= 10, "P3": m["P3"] >= 10, "P4": m["P4"] <= 10}
    return verdict_of(held)


PREDICTIONS = [
    ("P1", "from the claim", ["With 16 creatures, every", "body width reaches 95", "or more."],
     "lowest score", (80, 100), (95, 100)),
    ("P2", "from the claim", ["At body width 8, 1,024", "creatures score at least", "10 points below 16."],
     "points below", (0, 25), (10, 25)),
    ("P3", "from the claim", ["On 1,024 creatures, body", "width 128 scores at least", "10 points above width 8."],
     "points above", (0, 25), (10, 25)),
    ("P4", "against the claim", ["At every body width, 1,024", "creatures score within", "10 points of 16."],
     "largest gap, in points", (0, 25), (0, 10)),
]


def figure_predictions(counted):
    width, height = FULL, 2.3
    figure, page = page_of(width, height)
    left, panel_w, gap, top, plot_h = 0.62, 1.3, 0.16, 0.8, 0.72
    for i, (name, source, words, axis_name, limits, zone) in enumerate(PREDICTIONS):
        x0 = left + i * (panel_w + gap)
        write(page, x0, 0.0, name, HEAD, "bold")
        write(page, x0 + 0.24, 0.012, source, SMALL, colour=SECOND)
        for k, line in enumerate(words):
            write(page, x0, 0.2 + k * 0.14, line, SMALL)
        ax = plot_at(figure, x0, top, panel_w, plot_h)
        ax.axvspan(*zone, color=ZONE, linewidth=0)  # where the prediction says the marks would be
        edge = zone[0] if zone[0] > limits[0] else zone[1]  # the side of the band that decides
        ax.axvline(edge, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_xlim(limits[0] - 0.04 * (limits[1] - limits[0]), limits[1] + 0.04 * (limits[1] - limits[0]))
        ax.set_ylim(len(BITS) - 0.5, -0.5)
        for row, bits in enumerate(BITS):
            ax.axhline(row, color=GRID, linewidth=0.6, zorder=0)
            for use, shift in (("equal", -0.17), ("unequal", 0.17)):
                value = prediction_measures(counted, bits, use)[name]
                ax.plot([value], [row + shift], marker="o", markersize=4.6, clip_on=False, zorder=3,
                        markerfacecolor=INK if use == "equal" else PAPER, markeredgecolor=INK, markeredgewidth=1.0)
                # At 64 bits the marks of P2, P3 and P4 leave the place they held in the rows above,
                # and the text quotes them, so these marks carry their value. It stands to the
                # right of the mark, or to its left where the right is taken: near the end of the
                # axis, and for the hollow mark of P3, which has the filled one to its right.
                if bits == 64 and name != "P1":
                    on_left = value > limits[1] - 6 or (name == "P3" and use == "unequal")
                    write(ax, value + (-1.1 if on_left else 1.1), row + shift, f"{value:.1f}", SMALL, "medium",
                          ha="right" if on_left else "left", va="center", zorder=4)
        ax.set_xticks([80, 90, 100] if name == "P1" else [0, 10, 20])
        ax.set_yticks([])
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlabel(axis_name, fontsize=SMALL, labelpad=2)
        ax.text((zone[0] + zone[1]) / 2, -0.62, "predicted", fontproperties=FONTS["regular"], fontsize=SMALL,
                color=SECOND, ha="center", va="bottom")
        # Whether the prediction held by the rule, which was fixed for the 8-bit stage.
        held = [predictions_at_eight_bits(counted, use)[name] for use in USES]
        assert held[0] == held[1]  # the two word uses agree at 8 bits, so one mark is shown
        y = height - 0.34
        if held[0]:
            tick_mark(page, x0 + 0.06, y)
        else:
            cross_mark(page, x0 + 0.06, y)
        write(page, x0 + 0.17, y, "at 8 bits: " + ("held" if held[0] else "did not hold"), SMALL, "semibold", va="center")
    for row, bits in enumerate(BITS):
        y = top + (row + 0.5) * plot_h / len(BITS)
        write(page, left - 0.08, y, f"{bits} bits", SMALL, "semibold" if bits == 8 else "regular",
              INK if bits == 8 else SECOND, ha="right", va="center")
    # The verdict the four marks add up to.
    verdict = verdict_at_eight_bits(counted)
    reason = {"contradicted": "P4 held", "supported": "P1, P2 and P3 held",
              "mixed": "neither P1 to P3 nor P4 held throughout"}[verdict]
    y = height - 0.1
    words = "The rule’s verdict at 8 bits:"
    write(page, left, y, words, SMALL, colour=SECOND, va="center")
    x = left + text_width(words) + 0.07
    write(page, x, y, f"the claim is {verdict}", SMALL, "semibold", va="center")
    write(page, x + text_width(f"the claim is {verdict}", SMALL, "semibold") + 0.07, y, f"({reason})", SMALL,
          colour=SECOND, va="center")
    # The key for the two marks of a row, at the right end of the same line.
    x = width - 0.03
    for words, filled in (("unequal use", False), ("equal use", True)):  # placed from the right edge inward
        x -= text_width(words)
        write(page, x, y, words, SMALL, colour=SECOND, va="center")
        x -= 0.09
        page.plot([x], [y], marker="o", markersize=4.6, markerfacecolor=INK if filled else PAPER,
                  markeredgecolor=INK, markeredgewidth=1.0)
        x -= 0.24
    save(figure, "fig06_predictions")


# ---------------------------------------------------------------------------------------------
# Figure 7: the learning curves of all 476 counted runs
# ---------------------------------------------------------------------------------------------

# Figures 7, 8 and 9 share one grid: four columns of panels, a row for each word use.
GRID_LEFT, GRID_PANEL, GRID_GAP = 0.56, 1.355, 0.106


def figure_learning_curves(counted):
    width, height = FULL, 2.62
    figure, page = page_of(width, height)
    left, panel_w, gap, panel_h = GRID_LEFT, GRID_PANEL, GRID_GAP, 0.68
    tops = [0.62, 1.59]
    x = creature_key(page, left, 0.07)
    page.plot([x + 0.2, x + 0.45], [0.07, 0.07], color=CREATURE_BLUE[256], linewidth=0.5, alpha=0.5)
    write(page, x + 0.51, 0.07, "thin: every single run", SMALL, colour=SECOND, va="center")
    for row, (use, top) in enumerate(zip(USES, tops)):
        use_heading(page, 0, top - 0.12, use)
        for column, bits in enumerate(BITS):
            ax = plot_at(figure, left + column * (panel_w + gap), top, panel_w, panel_h)
            plain_axes(ax, grid_values=(25, 50, 75, 100), y_limits=(17, 104))
            stage = pick(counted, bits=bits, use=use)
            for creatures in CREATURES:
                group = pick(stage, creatures=creatures)
                if not group:
                    continue
                colour = CREATURE_BLUE[creatures]
                for run in group:  # every single run, faintly
                    ax.plot([p["step"] for p in run["curve"]], [p["facts"] for p in run["curve"]],
                            color=colour, linewidth=0.45, alpha=0.22, zorder=2)
                steps, scores = middle_curve(group)  # the middle of the runs, boldly
                ax.plot(steps, scores, color=colour, linewidth=1.7, zorder=3,
                        linestyle="-" if use == "equal" else DASH)
                ax.plot(steps[-1:], scores[-1:], marker="o", markersize=4.4, zorder=4, clip_on=False,
                        markerfacecolor=colour if use == "equal" else PAPER, markeredgecolor=colour,
                        markeredgewidth=1.1)
            ax.set_xscale("log")
            ax.set_xlim(85, 260000)
            ax.set_xticks([100, 1000, 10000, 100000])
            ax.set_xticklabels(["100", "1K", "10K", "100K"] if row == 1 else [])
            ax.minorticks_off()
            ax.set_yticks([25, 50, 75, 100])
            if column > 0:
                ax.set_yticklabels([])
            if row == 0:
                column_header(page, left + column * (panel_w + gap), top - 0.32, f"{bits} bits",
                              f"{nice(stage[0]['steps'])} steps")
            if column == 0:  # every run starts at the score of a guess
                write(ax, 230000, 27, "a guess scores 25", SMALL, colour=SECOND, ha="right", va="bottom", halo=True)
        write(page, 0.07, top + panel_h / 2, "facts known", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left + 2 * panel_w + 1.5 * gap, height - 0.02,
          "training steps so far (each mark is ten times the one before; K = thousand)",
          SMALL, colour=SECOND, ha="center", va="bottom")
    save(figure, "fig07_learning_curves")


# ---------------------------------------------------------------------------------------------
# Figure 8: how long learning took, against body width
# ---------------------------------------------------------------------------------------------

def figure_learning_time(counted):
    width, height = FULL, 2.56
    figure, page = page_of(width, height)
    left, panel_w, gap, panel_h = GRID_LEFT, GRID_PANEL, GRID_GAP, 0.65
    tops = [0.62, 1.56]
    lane, lane_from, top_edge = 950000, 420000, 2200000  # the lane "not yet" above the axis of steps
    creature_key(page, left, 0.07)
    for row, (use, top) in enumerate(zip(USES, tops)):
        use_heading(page, 0, top - 0.12, use)
        for column, bits in enumerate(BITS):
            ax = plot_at(figure, left + column * (panel_w + gap), top, panel_w, panel_h)
            plain_axes(ax, grid_values=(100, 1000, 10000, 100000))
            ax.axhspan(lane_from, top_edge, color=PALE_GREY, linewidth=0, zorder=0)
            stage = pick(counted, bits=bits, use=use)
            not_yet = {}  # body width -> the vocabularies that had not learned the facts by the end
            for creatures in CREATURES:
                colour = CREATURE_BLUE[creatures]
                points = []  # (place along the body-width axis, steps); None breaks the line
                for place, body in enumerate(BODIES):
                    group = pick(stage, creatures=creatures, body=body)
                    if not group:
                        continue  # not run at 64 bits
                    steps = learning_time(group)
                    if steps == NEVER:
                        not_yet.setdefault(place, []).append(creatures)
                        points.append(None)
                    else:
                        points.append((place, steps))
                stretch = []  # a line joins neighbouring body widths that both learned the facts
                for point in points + [None]:
                    if point is not None:
                        stretch.append(point)
                        continue
                    if stretch:
                        ax.plot([x for x, _ in stretch], [y for _, y in stretch], color=colour, linewidth=1.5,
                                linestyle="-" if use == "equal" else DASH, zorder=3)
                        ax.plot([x for x, _ in stretch], [y for _, y in stretch], linestyle="none", marker="o",
                                markersize=4.0, markerfacecolor=colour if use == "equal" else PAPER,
                                markeredgecolor=colour if use == "unequal" else PAPER,
                                markeredgewidth=1.0 if use == "unequal" else 0.5, zorder=4, clip_on=False)
                    stretch = []
            for place, waiting in not_yet.items():  # the marks in the lane, side by side
                for k, creatures in enumerate(waiting):
                    colour = CREATURE_BLUE[creatures]
                    ax.plot([place + (k - (len(waiting) - 1) / 2) * 0.3], [lane], marker="o", markersize=4.0,
                            markerfacecolor=colour if use == "equal" else PAPER,
                            markeredgecolor=colour if use == "unequal" else PAPER,
                            markeredgewidth=1.0 if use == "unequal" else 0.5, zorder=4)
            ax.set_yscale("log")
            ax.set_ylim(75, top_edge)
            ax.set_xlim(-0.5, 4.5)
            ax.set_xticks(range(5))
            ax.set_xticklabels([str(body) for body in BODIES] if row == 1 else [])
            ax.minorticks_off()
            ax.set_yticks([100, 1000, 10000, 100000, lane])
            ax.set_yticklabels(["100", "1K", "10K", "100K", "not yet"] if column == 0 else [])
            if row == 0:
                column_header(page, left + column * (panel_w + gap), top - 0.32, f"{bits} bits",
                              f"{nice(stage[0]['steps'])} steps")
    for top in tops:
        write(page, 0.07, top + panel_h / 2, "steps to learn", SMALL, colour=SECOND, ha="center", va="center",
              rotation=90)
    write(page, left + 2 * panel_w + 1.5 * gap, height - 0.02, "body width", SMALL, colour=SECOND,
          ha="center", va="bottom")
    save(figure, "fig08_learning_time")


# ---------------------------------------------------------------------------------------------
# Figure 9: the four scores against bits per creature
# ---------------------------------------------------------------------------------------------

# The four scores: its name in the run files, its title in two lines, and what a guess scores.
SCORE_PANELS = [("facts", ["Facts known", ""], 25), ("questions_one", ["Questions about", "one creature"], 0),
                ("questions_new", ["Questions about", "new combinations"], 0), ("not_known", ["“Not” known", ""], 25)]


def figure_scores_by_bits(counted):
    width, height = FULL, 2.66
    figure, page = page_of(width, height)
    left, panel_w, gap, panel_h = GRID_LEFT, GRID_PANEL, GRID_GAP, 0.65
    tops = [0.74, 1.68]
    creature_key(page, left, 0.07)
    for row, (use, top) in enumerate(zip(USES, tops)):
        use_heading(page, 0, top - 0.12, use)
        for column, (score, title, guess) in enumerate(SCORE_PANELS):
            ax = plot_at(figure, left + column * (panel_w + gap), top, panel_w, panel_h)
            plain_axes(ax, grid_values=(0, 25, 50, 75, 100), y_limits=(-4, 106))
            ax.spines["bottom"].set_visible(False)  # the line at 0 is the foot of the plot
            ax.axhline(0, color=AXIS, linewidth=0.7, zorder=1)
            if guess:  # a guess that scores more than 0 gets a line of its own
                ax.axhline(guess, color=AXIS, linewidth=1.3, zorder=1)
            write(ax, -0.3, guess + 3, f"a guess scores {guess}", SMALL, colour=SECOND, va="bottom")
            for creatures in CREATURES:
                points = [(x, middle(run[score] for run in pick(counted, bits=bits, use=use, creatures=creatures)))
                          for x, bits in enumerate(BITS)]
                points = [(x, y) for x, y in points if y is not None]  # 64 creatures: not run at 64 bits
                colour = CREATURE_BLUE[creatures]
                ax.plot([x for x, _ in points], [y for _, y in points], color=colour, linewidth=1.7,
                        linestyle="-" if use == "equal" else DASH, zorder=3)
                ax.plot([x for x, _ in points], [y for _, y in points], linestyle="none", marker="o",
                        markersize=4.6, markerfacecolor=colour if use == "equal" else PAPER,
                        markeredgecolor=colour if use == "unequal" else PAPER,
                        markeredgewidth=1.1 if use == "unequal" else 0.6, zorder=4, clip_on=False)
            ax.set_xlim(-0.35, 3.45)
            ax.set_xticks(range(4))
            ax.set_xticklabels([str(bits) for bits in BITS] if row == 1 else [])
            ax.set_yticks([0, 25, 50, 75, 100])
            if column > 0:
                ax.set_yticklabels([])
            if row == 0:  # the name of the score
                x = left + column * (panel_w + gap)
                for k, line in enumerate(title):
                    write(page, x, 0.24 + k * 0.14, line, SMALL, "semibold")
        write(page, 0.07, top + panel_h / 2, "score", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left + 2 * panel_w + 1.5 * gap, height - 0.02, "bits per creature", SMALL, colour=SECOND,
          ha="center", va="bottom")
    save(figure, "fig09_scores_by_bits")


# ---------------------------------------------------------------------------------------------
# Figure 10: 64 bits. All 18 settings: the two counted repeats, and the longer run where there is one
# ---------------------------------------------------------------------------------------------

def clear_height(wanted, half, blocked):
    """The height nearest to `wanted` at which a label that reaches `half` above and below its
    middle touches none of the blocked stretches (low, high) of the axis."""
    for step in range(40):
        for y in (wanted + step * 0.1, wanted - step * 0.1):
            if all(y + half <= low or y - half >= high for low, high in blocked):
                return y
    return wanted


def figure_sixty_four(counted, longer, shared_step):
    width, height = FULL, 3.02
    figure, page = page_of(width, height)
    left, panel_w, gap, top, panel_h = 0.40, 2.96, 0.08, 0.6, 1.8
    stage = pick(counted, bits=64)
    long_steps = next(iter(longer.values()))["steps"]
    # Where the three body widths of each vocabulary stand, in inches from the left of a panel.
    # The 16 creatures need little room (every run at 100); the others need room for an arrow
    # and its number beside each pair of dots.
    places = {16: (0.10, 0.33, 0.56), 256: (0.80, 1.15, 1.50), 1024: (1.82, 2.17, 2.52)}
    per_inch = (102 - AXIS_START_SETTINGS) / panel_h  # points of score in one inch of height
    dot_half = 0.0375   # half the width of a dot, in inches
    arrow_at = 0.115    # how far right of its setting's place an arrow stands, in inches

    # The key.
    y = 0.08
    page.plot([left + 0.025, left + 0.1], [y, y], linestyle="none", marker="o", markersize=5.4,
              color=CREATURE_BLUE[256], markeredgecolor=PAPER, markeredgewidth=0.6)
    write(page, left + 0.2, y, f"the two counted runs of a setting ({nice(stage[0]['steps'])} steps)", SMALL,
          colour=SECOND, va="center")
    x = left + 3.04
    page.add_patch(FancyArrowPatch((x, y), (x + 0.36, y), arrowstyle="-|>", mutation_scale=8, color=LONGER,
                                   linewidth=1.6, shrinkA=0, shrinkB=0))
    page.plot([x], [y], marker="o", markersize=3.4, color=LONGER)
    write(page, x + 0.46, y, f"its longer run, from step {nice(shared_step)} to step {nice(long_steps)}", SMALL,
          colour=SECOND, va="center")

    for column, use in enumerate(USES):
        x0 = left + column * (panel_w + gap)
        ax = plot_at(figure, x0, top, panel_w, panel_h)
        plain_axes(ax, grid_values=(70, 75, 80, 85, 90, 100), y_limits=(AXIS_START_SETTINGS, 102))
        ax.axhline(KNOWN, color=SECOND, linewidth=0.8, zorder=1)
        ax.set_xlim(0, panel_w)  # the horizontal axis is in inches, so marks and labels can be placed exactly
        filled = use == "equal"
        for creatures in CREATURES_64:
            colour = CREATURE_BLUE[creatures]
            middles = []
            for place, body in zip(places[creatures], BODIES_64):
                finals = sorted(run["facts"] for run in pick(stage, use=use, creatures=creatures, body=body))
                middles.append((place, middle(finals)))
                ax.plot([place, place], finals, color=colour, linewidth=1.0, zorder=3)
                # The two counted repeats. Where one dot would cover the other (both at 100, for
                # one), they are set side by side, so that every setting shows its two dots.
                covered = finals[1] - finals[0] < 2 * dot_half * per_inch
                for value, side in zip(finals, (-dot_half, dot_half) if covered else (0, 0)):
                    ax.plot([place + side], [value], marker="o", markersize=5.4, zorder=4, clip_on=False,
                            markerfacecolor=colour if filled else PAPER, markeredgecolor=PAPER if filled else colour,
                            markeredgewidth=0.6 if filled else 1.3)
                run = longer.get((use, creatures, body))
                if run:  # the longer run: from where it stood near the counted end to where it ended
                    start, end = run["at"][shared_step]["facts"], run["facts"]
                    ax.add_patch(FancyArrowPatch((place + arrow_at, start), (place + arrow_at, end), arrowstyle="-|>",
                                                 mutation_scale=8, color=LONGER, linewidth=1.6, shrinkA=0,
                                                 shrinkB=0, zorder=5))
                    ax.plot([place + arrow_at], [start], marker="o", markersize=3.4, color=LONGER, zorder=5)
                    # Its final score, always just right of the arrow head and at its height. It
                    # moves up or down by the least amount only where it would sit on the line at
                    # 95 or touch a dot of the next body width.
                    label = exact(end)
                    label_left = place + arrow_at + 0.05
                    label_right = label_left + text_width(label, SMALL, "medium")
                    blocked = [(KNOWN - 0.25, KNOWN + 0.25)]
                    wider = [p for p in places[creatures] if p > place][:1]
                    if wider and label_right > wider[0] - 2 * dot_half - 0.03:
                        next_body = BODIES_64[places[creatures].index(wider[0])]
                        blocked += [(value - (dot_half + 0.02) * per_inch, value + (dot_half + 0.02) * per_inch)
                                    for value in (r["facts"] for r in pick(stage, use=use, creatures=creatures,
                                                                           body=next_body))]
                    label_y = clear_height(end, 0.058 * per_inch, blocked)
                    write(ax, label_left, label_y, label, SMALL, "medium", ha="left", va="center", halo=True, zorder=6)
            # A thin line through the middles shows what a wider body does.
            ax.plot([x for x, _ in middles], [y for _, y in middles], color=colour, linewidth=0.8,
                    linestyle="-" if filled else DASH, zorder=2, alpha=0.75)
            write(page, x0 + places[creatures][1], top + panel_h + 0.2, f"{nice(creatures)} creatures", SMALL,
                  "medium", ha="center")
        ax.set_xticks([place for creatures in CREATURES_64 for place in places[creatures]])
        ax.set_xticklabels([str(body) for _ in CREATURES_64 for body in BODIES_64])
        ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
        if column == 1:
            ax.set_yticklabels([])
        use_heading(page, x0, top - 0.17, use, HEAD, "semibold")
        if column == 0:  # the eight settings under this line are the ones that were trained longer
            write(ax, 0.03, KNOWN - 0.4, f"below {KNOWN}: trained longer", SMALL, colour=SECOND, ha="left", va="top")
    write(page, 0.07, top + panel_h / 2, "facts known", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left + panel_w + gap / 2, height - 0.02, "body width (8, 32, 128), for each number of creatures",
          SMALL, colour=SECOND, ha="center", va="bottom")
    save(figure, "fig10_sixty_four")


# ---------------------------------------------------------------------------------------------
# Figure 11: 64 bits over time. Every counted run, and the eight longer runs, on one axis
# ---------------------------------------------------------------------------------------------

def figure_longer_curves(counted, longer):
    width, height = FULL, 2.92
    figure, page = page_of(width, height)
    left, panel_w, gap, panel_h = 0.5, 1.9, 0.05, 0.82
    tops = [0.62, 1.73]
    stage = pick(counted, bits=64)
    counted_end = stage[0]["steps"]
    long_steps = next(iter(longer.values()))["steps"]
    tag_right, tag_step = 625000, 289631  # where the name of an orange line's vocabulary ends, and the
    # measurement near its left end: the line rises, so the name must clear the line there

    x = creature_key(page, left, 0.07, CREATURES_64, "counted runs, by creatures")
    page.plot([x + 0.2, x + 0.5], [0.07, 0.07], color=LONGER, linewidth=1.9)
    page.plot([x + 0.35], [0.07], marker="o", markersize=3.6, color=LONGER)
    write(page, x + 0.56, 0.07, "longer run", SMALL, "medium", va="center")

    for row, (use, top) in enumerate(zip(USES, tops)):
        style = "-" if use == "equal" else DASH
        use_heading(page, 0, top - 0.12, use)
        for column, body in enumerate(BODIES_64):
            ax = plot_at(figure, left + column * (panel_w + gap), top, panel_w, panel_h)
            plain_axes(ax, y_limits=(AXIS_START_CURVES, 102))
            # The grid and the line at 95 end where the training ends, so that the final scores
            # written to the right of it stand on clear paper.
            for value in (70, 80, 90, 100):
                ax.plot([0, long_steps], [value, value], color=GRID, linewidth=0.6, zorder=0)
            ax.plot([0, long_steps], [KNOWN, KNOWN], color=SECOND, linewidth=0.8, zorder=1)
            ax.axvline(counted_end, color=AXIS, linewidth=0.9, zorder=1)
            for creatures in CREATURES_64:
                colour = CREATURE_BLUE[creatures]
                for run in pick(stage, use=use, body=body, creatures=creatures):  # the counted runs
                    ax.plot([p["step"] for p in run["curve"]], [p["facts"] for p in run["curve"]], color=colour,
                            linewidth=0.9, linestyle=style, zorder=3)
                    ax.plot([run["steps"]], [run["facts"]], marker="o", markersize=3.8, zorder=4,
                            markerfacecolor=colour if use == "equal" else PAPER, markeredgecolor=colour,
                            markeredgewidth=0.9)
                run = longer.get((use, creatures, body))
                if run:  # the longer run of this setting, with a mark at every measurement after the counted end
                    steps = [p["step"] for p in run["curve"]]
                    facts = [p["facts"] for p in run["curve"]]
                    ax.plot(steps, facts, color=LONGER, linewidth=1.9, linestyle=style, zorder=5)
                    late = [(s, f) for s, f in zip(steps, facts) if s > counted_end]
                    ax.plot([s for s, _ in late], [f for _, f in late], linestyle="none", marker="o", markersize=3.6,
                            markerfacecolor=LONGER if use == "equal" else PAPER, markeredgecolor=LONGER,
                            markeredgewidth=0.9, zorder=6, clip_on=False)
                    # Its final score at the end of the line, and under the line the vocabulary it
                    # belongs to. The name stays below the line at 95.
                    write(ax, long_steps * 1.035, facts[-1], exact(facts[-1]), SMALL, "medium", va="center")
                    tag_top = min(run["at"][tag_step]["facts"] - 1.4, KNOWN - 0.9)
                    write(ax, tag_right, tag_top, f"{nice(creatures)} creatures", SMALL, colour=SECOND, ha="right",
                          va="top", zorder=7)
            if not any((use, creatures, body) in longer for creatures in CREATURES_64):
                # A panel without an orange line: say why.
                assert all(run["facts"] >= KNOWN for run in pick(stage, use=use, body=body))
                write(ax, 400000, 84, f"no longer run here:\nevery counted run\nended at {KNOWN} or more", SMALL,
                      colour=SECOND, ha="center", va="center", linespacing=1.25)
            ax.set_xlim(-14000, long_steps * 1.22)
            ax.set_xticks([0, counted_end, 2 * counted_end, 3 * counted_end, long_steps])
            ax.set_xticklabels([thousands(int(v)) for v in ax.get_xticks()] if row == 1 else [])
            ax.set_yticks([70, 80, 90, 100])
            if column > 0:
                ax.set_yticklabels([])
            if row == 0:
                column_header(page, left + column * (panel_w + gap), top - 0.32, f"Body width {body}")
            if column == 0:
                write(ax, 300000, KNOWN + 0.5, f"{KNOWN} = known", SMALL, colour=SECOND, ha="left", va="bottom")
            if row == 0 and column == 0:
                write(ax, counted_end + 14000, AXIS_START_CURVES + 0.8, "the counted\nruns end here", SMALL,
                      colour=SECOND, va="bottom", linespacing=1.15)
        write(page, 0.07, top + panel_h / 2, "facts known", SMALL, colour=SECOND, ha="center", va="center", rotation=90)
    write(page, left + 1.5 * panel_w + gap, height - 0.02, "training steps so far, evenly spaced (K = thousand)",
          SMALL, colour=SECOND, ha="center", va="bottom")
    save(figure, "fig11_longer_curves")


# ---------------------------------------------------------------------------------------------
# The numbers the text uses
# ---------------------------------------------------------------------------------------------

def write_numbers(counted, longer, shared_step):
    """Every number the text uses, written to numbers.tex as LaTeX macros (\\RunsCounted and so on).
    The macros are grouped here in the order in which the report needs them."""
    numbers = dict(OUTSIDE_FACTS)
    numbers.update(numbers_of_the_design(counted, longer, shared_step))
    numbers.update(numbers_of_the_predictions(counted))
    numbers.update(numbers_of_learning_time(counted))
    numbers.update(numbers_of_the_scores(counted))
    numbers.update(numbers_at_sixty_four_bits(counted, longer))
    numbers.update(numbers_of_the_longer_runs(counted, longer, shared_step))
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in numbers.items()]
    # The rows of the table in Appendix C as one macro: LaTeX does not accept a file read in just
    # before a table rule.
    lines.append("\\newcommand{\\SixtyFourRows}{" + " ".join(table_of_sixty_four_bits(counted, longer, shared_step)) + "}")
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    return numbers


STAGE_NAMES = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour"}  # for macro names
USE_NAMES = {"equal": "Equal", "unequal": "Unequal"}


def one(value):
    """A score with one decimal: 97.04 -> "97.0"."""
    return f"{value:.1f}"


def whole(value):
    """A score as a whole number: 78.4 -> "78"."""
    return f"{value:.0f}"


def in_a_row(values):
    """[1, 2, 3] -> "1, 2 and 3"."""
    values = [nice(value) for value in values]
    return ", ".join(values[:-1]) + " and " + values[-1]


def numbers_of_the_design(counted, longer, shared_step):
    """What was run: how many runs, how long, at what rate, and how large the models are."""
    numbers = {"KnownBar": KNOWN, "AxisStartSettings": AXIS_START_SETTINGS, "AxisStartCurves": AXIS_START_CURVES,
               "KeyBarsWord": in_words(KEY_BARS), "PracticeRuns": PRACTICE_RUNS,
               "QuestionExampleRight": QUESTION_EXAMPLE, "QuestionExampleScore": 2 * QUESTION_EXAMPLE - 100}
    numbers["RunsCounted"] = len(counted)
    numbers["RunsLonger"] = len(longer)
    numbers["RunsLongerWord"] = in_words(len(longer))
    numbers["RunsAll"] = len(counted) + len(longer)
    numbers["StagesWord"] = in_words(len(STAGES))
    numbers["CreatureTimes"] = CREATURES[-1] // CREATURES[0]  # 1,024 creatures against 16
    for bits, name in STAGE_NAMES.items():
        stage = pick(counted, bits=bits)
        numbers[f"Runs{name}"] = len(stage)
        numbers[f"Repeats{name}"] = len({run["repeat"] for run in stage})
        numbers[f"Repeats{name}Word"] = in_words(len({run["repeat"] for run in stage}))
        numbers[f"Steps{name}"] = nice(stage[0]["steps"])
        numbers[f"Rate{name}"] = f"{stage[0]['rate']:g}"
        numbers[f"Sentences{name}"] = f"{stage[0]['steps'] * int(OUTSIDE_FACTS['SentencesPerStep']) / 1e6:.1f}"
    numbers["RunsEightSixteen"] = len(pick(counted, bits=8)) + len(pick(counted, bits=16))
    repeats = [numbers[f"Repeats{name}"] for name in STAGE_NAMES.values()]
    numbers["RepeatsFewest"], numbers["RepeatsMost"] = min(repeats), max(repeats)  # "trained 2 to 5 times"
    any_longer = next(iter(longer.values()))
    counted_64 = pick(counted, bits=64)[0]
    numbers["StepsLonger"] = nice(any_longer["steps"])
    numbers["SentencesLonger"] = f"{any_longer['steps'] * int(OUTSIDE_FACTS['SentencesPerStep']) / 1e6:.1f}"
    numbers["LongerTimesWord"] = in_words(any_longer["steps"] / counted_64["steps"])
    numbers["LongerTimesWordCap"] = numbers["LongerTimesWord"].capitalize()
    numbers["LongerHoursLow"] = one(min(run["hours"] for run in longer.values()))
    numbers["LongerHoursHigh"] = one(max(run["hours"] for run in longer.values()))
    # The steps at which scores were measured.
    steps_counted = [point["step"] for point in counted_64["curve"]]
    steps_longer = [point["step"] for point in any_longer["curve"]]
    numbers["StepsShared"] = nice(shared_step)
    numbers["StepsLate"] = nice(steps_counted[-3])  # two measurements before the end of a counted run
    numbers["FirstSteps"] = ", ".join(nice(step) for step in steps_counted[:4])
    numbers["StepRatio"] = f"{steps_counted[12] / steps_counted[11]:.3f}"
    numbers["StepRatioShort"] = f"{steps_counted[12] / steps_counted[11]:.1f}"
    numbers["CountedLastSteps"] = in_a_row(steps_counted[-3:])
    numbers["LongerLastSteps"] = in_a_row([step for step in steps_longer if step >= shared_step])
    # The word table. At 8 bits it has a row for each of the 1,024 names and 31 more: 30 words
    # and marks and one filler row. Every further kind of fact adds the same number of words.
    for creatures, name in ((16, "Small"), (1024, "Large")):
        numbers[f"Rows{name}"] = nice(table_rows(counted, 8, creatures))
    for bits, name in STAGE_NAMES.items():
        numbers[f"TableRows{name}"] = nice(table_rows(counted, bits, 1024))
    # The rows in use that training never changed (Appendix A names them: the filler row, the
    # full stop, "yes" and "no").
    never = rows_in_use_never_changed()
    assert sorted(never) == sorted([".", "<pad>", "no", "yes"])
    numbers["RowsNeverChangedWordCap"] = in_words(len(never)).capitalize()
    numbers["OtherWords"] = table_rows(counted, 8, 1024) - 1024 - 1
    more_rows, more_kinds = table_rows(counted, 64, 1024) - table_rows(counted, 8, 1024), 32 - 4
    assert more_rows % more_kinds == 0
    numbers["WordsPerKind"] = in_words(more_rows // more_kinds)
    # The size of the models (8 bits): the trained numbers outside the word table, and the table.
    for body, name in ((8, "Narrow"), (128, "Wide")):
        numbers[f"Outside{name}"] = nice(pick(counted, bits=8, body=body)[0]["parameters"]["trained_without_table"])
    largest = pick(counted, bits=8, creatures=1024, body=128)[0]["parameters"]
    numbers["TableNumbersLarge"] = nice(largest["table_numbers"])
    assert largest["table_numbers"] > largest["trained_without_table"]  # "more than the rest of the widest model"
    # How much larger the whole model is with the widest body than with the narrowest (8 bits,
    # 1,024 creatures): the word table is the same in both, so far less than the body widths suggest.
    smallest = pick(counted, bits=8, creatures=1024, body=8)[0]["parameters"]
    numbers["ModelNarrow"], numbers["ModelWide"] = nice(smallest["trained"]), nice(largest["trained"])
    numbers["ModelGrowthPercent"] = whole(100 * (largest["trained"] / smallest["trained"] - 1))
    numbers["BodyTimesWordCap"] = in_words(BODIES[-1] // BODIES[0]).capitalize()  # "Sixteen times the neurons"
    # The questions put aside for later: one heading "## Q..." for each in research_questions.md.
    later = sum(line.startswith("## Q") for line in (ROOT / "research_questions.md").read_text().splitlines())
    numbers["FurtherQuestionsWordCap"] = in_words(later).capitalize()
    return numbers


def numbers_of_the_predictions(counted):
    """The verdict at 8 bits, and the same measures at the later stages."""
    numbers = {"VerdictEight": verdict_at_eight_bits(counted), "VerdictSixtyFour": verdict_from_middles(counted, 64)}
    held = predictions_at_eight_bits(counted, "equal")
    assert held == predictions_at_eight_bits(counted, "unequal")  # the text gives one verdict for both uses
    for name, macro in (("P1", "HeldOne"), ("P2", "HeldTwo"), ("P3", "HeldThree"), ("P4", "HeldFour")):
        numbers[macro] = "held" if held[name] else "did not hold"
    assert held["P2"] == held["P3"]  # the text says "P2 and P3" in one breath
    for use, use_name in USE_NAMES.items():
        measures = prediction_measures(counted, 64, use)
        numbers[f"Gap{use_name}SixtyFour"] = one(measures["P2"])       # 16 creatures minus 1,024, body width 8
        numbers[f"BodyGain{use_name}SixtyFour"] = one(measures["P3"])  # body width 128 minus 8, 1,024 creatures
    numbers["LargestGapUpToThirtyTwo"] = one(max(prediction_measures(counted, bits, use)["P4"]
                                                 for bits in (8, 16, 32) for use in USES))
    # The 64-bit stage has two repeats, and with equal use they fall on opposite sides of the 10
    # points that P2, P3 and P4 turn on. So the text gives both repeats, not only their middle.
    facts = {(run["creatures"], run["body"], run["repeat"]): run["facts"] for run in pick(counted, bits=64, use="equal")}
    repeats = sorted({repeat for _, _, repeat in facts})
    gap = {r: facts[16, 8, r] - facts[1024, 8, r] for r in repeats}             # what P2 is about
    gain = {r: facts[1024, 128, r] - facts[1024, 8, r] for r in repeats}        # what P3 is about
    largest = {r: max(abs(facts[16, body, r] - facts[1024, body, r]) for body in BODIES_64) for r in repeats}  # P4
    high, low = sorted(repeats, key=lambda r: -gap[r])
    assert gap[high] >= 10 and largest[high] > 10    # in one repeat P2 would hold and P4 would not,
    assert gap[low] < 10 and largest[low] <= 10      # in the other P4 would hold and P2 would not
    assert max(gain.values()) >= 10 > min(gain.values())  # and one repeat reaches the 10 points of P3
    numbers["GapEqualRepeatHigh"], numbers["GapEqualRepeatLow"] = one(gap[high]), one(gap[low])
    numbers["BodyGainEqualRepeatHigh"], numbers["BodyGainEqualRepeatLow"] = one(max(gain.values())), one(min(gain.values()))
    return numbers


def numbers_of_learning_time(counted):
    """How long learning took (learning_time): what 64 times the creatures cost, and whether the
    body width changed it."""
    numbers = {}
    for bits, name in ((8, "Eight"), (16, "Sixteen"), (32, "ThirtyTwo")):
        for use, use_name in (("equal", ""), ("unequal", "Unequal")):
            needed = {creatures: learning_time(pick(counted, bits=bits, use=use, creatures=creatures))
                      for creatures in (16, 1024)}
            numbers[f"LearnTimes{name}{use_name}"] = whole(needed[1024] / needed[16])
            if (bits, use) == (8, "equal"):
                numbers["LearnSmallEight"], numbers["LearnLargeEight"] = nice(int(needed[16])), nice(int(needed[1024]))
    # The lines of Figure 8 up to 32 bits: the narrowest body against the widest.
    same = faster = slower = 0
    for bits in (8, 16, 32):
        for use in USES:
            for creatures in CREATURES:
                narrow, wide = (learning_time(pick(counted, bits=bits, use=use, creatures=creatures, body=body))
                                for body in (BODIES[0], BODIES[-1]))
                same, faster, slower = same + (wide == narrow), faster + (wide < narrow), slower + (wide > narrow)
    numbers["TimeLines"] = same + faster + slower
    numbers["TimeSame"], numbers["TimeFaster"], numbers["TimeSlower"] = same, faster, slower
    return numbers


def numbers_of_the_scores(counted):
    """How many runs of a stage passed the bar on each part of the test, how the vocabulary
    shows at 32 bits, and new combinations against combinations that were read."""
    numbers = {}
    for bits, name in STAGE_NAMES.items():
        stage = pick(counted, bits=bits)
        for score, score_name in (("facts", "Facts"), ("questions_one", "Questions"), ("questions_new", "New"),
                                  ("not_known", "Not")):
            numbers[f"{score_name}Known{name}"] = sum(run[score] >= KNOWN for run in stage)
        numbers[f"FactsLowest{name}"] = one(min(run["facts"] for run in stage))
    assert all(run["facts"] == 100 for run in pick(counted, bits=8) + pick(counted, bits=16))  # "ended at 100"
    # 32 bits, "not" known: the smallest vocabulary against the larger ones.
    stage = pick(counted, bits=32)
    small, larger = pick(stage, creatures=16), [run for run in stage if run["creatures"] > 16]
    passed = lambda runs: sum(run["not_known"] >= KNOWN for run in runs)
    assert passed(small) == len(small)  # "all 30 runs with 16 creatures"
    numbers["RunsSmallThirtyTwo"] = len(small)
    numbers["RunsLargerThirtyTwo"], numbers["NotLargerThirtyTwo"] = len(larger), passed(larger)
    larger_unequal = pick(larger, use="unequal")
    numbers["RunsLargerUnequalThirtyTwo"], numbers["NotLargerUnequalThirtyTwo"] = len(larger_unequal), passed(larger_unequal)
    for body, name in ((8, "Narrow"), (128, "Wide")):
        numbers[f"Not{name}UnequalThirtyTwo"] = whole(middle(run["not_known"] for run in pick(larger_unequal, body=body)))
    # How many runs stand behind one point of Figure 9.
    behind = {bits: len(pick(counted, bits=bits, use="equal", creatures=16)) for bits in BITS}
    assert behind[16] == behind[32]
    numbers["RunsPerPointEight"], numbers["RunsPerPointLater"] = behind[8], behind[16]
    numbers["RunsPerPointSixtyFour"] = behind[64]
    # New combinations against combinations that were read in training.
    differences = []
    for bits in BITS:
        for use in USES:
            for creatures in CREATURES[1:]:  # from 64 creatures up
                group = pick(counted, bits=bits, use=use, creatures=creatures)
                if group:
                    differences.append(abs(middle(run["questions_new"] for run in group)
                                           - middle(run["questions_seen"] for run in group)))
    numbers["NewSeenLargestDifference"] = whole(np.ceil(max(differences)))
    for use, use_name in USE_NAMES.items():
        small = pick(counted, bits=64, creatures=16, use=use)
        numbers[f"NewSmallSixtyFour{use_name}"] = whole(middle(run["questions_new"] for run in small))
        numbers[f"SeenSmallSixtyFour{use_name}"] = whole(middle(run["questions_seen"] for run in small))
    return numbers


def low_groups(counted):
    """The vocabularies at 64 bits in which some setting ended below the bar, as (word use, creatures)."""
    return [(use, creatures) for use in USES for creatures in CREATURES_64
            if any(setting_score(counted, 64, use, creatures, body) < KNOWN for body in BODIES_64)]


def numbers_at_sixty_four_bits(counted, longer):
    """64 bits: every setting of the vocabularies that fell short, what a wider body did, and
    which settings ended below the bar."""
    numbers = {}
    stage = pick(counted, bits=64)
    for use, use_name in USE_NAMES.items():
        for creatures, name in ((256, "Mid"), (1024, "Large")):
            for body, body_name in ((8, "Narrow"), (32, "Middle"), (128, "Wide")):
                numbers[f"Facts{use_name}{name}{body_name}"] = one(setting_score(counted, 64, use, creatures, body))
    assert all(run["facts"] == 100 for run in pick(stage, creatures=16))  # "with 16 creatures every run ended at 100"
    assert all(exact(run["facts"]) == "100" for run in pick(stage, creatures=256, use="equal"))  # "256 creatures, equal use: 100"
    settings = [(use, creatures, body) for use in USES for creatures in CREATURES_64 for body in BODIES_64]
    below = [setting for setting in settings if setting_score(counted, 64, *setting) < KNOWN]
    assert sorted(below) == sorted(longer), "the longer runs are exactly the settings below the bar"
    numbers["SettingsSixtyFour"] = len(settings)
    numbers["SettingsBelow"] = len(below)
    numbers["SettingsBelowWord"] = in_words(len(below))
    runs_below = [run for run in stage if run["facts"] < KNOWN]
    numbers["RunsBelowSixtyFour"] = len(runs_below)
    late = stage[0]["curve"][-3]["step"]
    late_gain = [run["facts"] - run["at"][late]["facts"] for run in runs_below]
    numbers["LateGainLow"], numbers["LateGainHigh"] = one(min(late_gain)), one(max(late_gain))
    spread = [abs(a["facts"] - b["facts"]) for setting in below
              for a, b in [pick(stage, use=setting[0], creatures=setting[1], body=setting[2])]]
    numbers["RepeatSpreadHigh"] = one(max(spread))  # how far apart the two repeats of a low setting ended
    # What a wider body did in the vocabularies that fell short: each step from one body width to
    # the next, in the counted runs (middle of the repeats) and in the longer runs.
    groups = low_groups(counted)
    # "1,024 creatures with equal use, 256 and 1,024 with unequal use"
    assert groups == [("equal", 1024), ("unequal", 256), ("unequal", 1024)]
    counted_steps, longer_steps, rises = [], [], []
    for use, creatures in groups:
        scores = [setting_score(counted, 64, use, creatures, body) for body in BODIES_64]
        counted_steps += [b - a for a, b in zip(scores, scores[1:])]
        rises.append(scores[-1] - scores[0])
        ends = [longer[use, creatures, body]["facts"] for body in BODIES_64 if (use, creatures, body) in longer]
        longer_steps += [b - a for a, b in zip(ends, ends[1:])]
    assert min(counted_steps) > 0 and min(longer_steps) > 0  # "every step up in body width raised the score"
    numbers["LowGroupsWord"] = in_words(len(groups))
    numbers["BodyStepsCounted"], numbers["BodyStepsLonger"] = len(counted_steps), len(longer_steps)
    numbers["BodyStepsAll"] = len(counted_steps) + len(longer_steps)
    numbers["BodyRiseLow"], numbers["BodyRiseHigh"] = one(min(rises)), one(max(rises))
    numbers["BodyRiseLowWhole"], numbers["BodyRiseHighWhole"] = whole(min(rises)), whole(max(rises))
    return numbers


def numbers_of_the_longer_runs(counted, longer, shared_step):
    """The eight longer runs: where they ended, what they gained, and how the pace changed."""
    numbers = {}
    stage = pick(counted, bits=64)
    runs = list(longer.values())
    ends = [run["facts"] for run in runs]
    gains = [run["facts"] - run["at"][shared_step]["facts"] for run in runs]
    assert min(gains) > 0  # "every one gained"
    numbers["LongerReached"] = sum(end >= KNOWN for end in ends)
    numbers["LongerReachedWord"] = in_words(numbers["LongerReached"])
    numbers["LongerShortWord"] = in_words(len(runs) - numbers["LongerReached"])
    # The settings that reached 95 are the ones whose counted runs had been nearest to it: their
    # middle was within half a point, and no other setting's was.
    reached = [setting for setting in longer if longer[setting]["facts"] >= KNOWN]
    short = [setting for setting in longer if longer[setting]["facts"] < KNOWN]
    before = {setting: setting_score(counted, 64, *setting) for setting in longer}
    assert all(KNOWN - before[setting] < 0.5 for setting in reached)
    assert all(KNOWN - before[setting] > 0.5 for setting in short)
    numbers["ReachedBeforeLow"] = one(min(before[setting] for setting in reached))
    numbers["ReachedBeforeHigh"] = one(max(before[setting] for setting in reached))
    numbers["ShortEndLow"] = one(min(longer[setting]["facts"] for setting in short))
    numbers["ShortEndHigh"] = one(max(longer[setting]["facts"] for setting in short))
    numbers["LongerLowest"], numbers["LongerLowestExact"] = whole(min(ends)), one(min(ends))
    numbers["LongerGainLow"], numbers["LongerGainHigh"] = one(min(gains)), one(max(gains))
    # The pace: points gained over two doublings of the training that were both measured.
    measured = sorted(runs[0]["at"])
    nearest = lambda target: min(measured, key=lambda step: abs(step - target))
    second = nearest(2 * shared_step)
    third = nearest(2 * second)
    earlier = [run["at"][second]["facts"] - run["at"][shared_step]["facts"] for run in runs]
    later = [run["at"][third]["facts"] - run["at"][second]["facts"] for run in runs]
    numbers["DoublingSecond"], numbers["DoublingThird"] = nice(second), nice(third)
    numbers["DoublingEarlierLow"], numbers["DoublingEarlierHigh"] = one(min(earlier)), one(max(earlier))
    numbers["DoublingLaterLow"], numbers["DoublingLaterHigh"] = one(min(later)), one(max(later))
    numbers["LongerSlowing"] = sum(b < a for a, b in zip(earlier, later))
    # The last stretch, from the last measurement before the end to the end: who was still gaining.
    numbers["LastStretch"] = nice(measured[-1] - measured[-2])
    last = {setting: run["facts"] - run["at"][measured[-2]]["facts"] for setting, run in longer.items()}
    level = sorted(setting for setting, gain in last.items() if abs(gain) < 0.1)
    still = [gain for gain in last.values() if gain >= 0.1]
    assert len(level) + len(still) == len(runs)  # no run was falling
    assert level == [("equal", 1024, 8), ("unequal", 256, 32)]  # the two runs the text names as level
    numbers["LongerLevelWord"], numbers["LongerStillWord"] = in_words(len(level)), in_words(len(still))
    numbers["LongerStillWordCap"] = in_words(len(still)).capitalize()
    numbers["LastGainLow"], numbers["LastGainHigh"] = one(min(still)), one(max(still))
    # The lowest run, and how far it is from the bar at the pace of its last doubling.
    lowest = min(runs, key=lambda run: run["facts"])
    pace = lowest["at"][third]["facts"] - lowest["at"][second]["facts"]
    doublings_needed = int(np.floor((KNOWN - lowest["facts"]) / pace))
    assert (KNOWN - lowest["facts"]) / pace > doublings_needed  # so "more than" is true
    numbers["LowestShort"] = one(KNOWN - lowest["facts"])
    numbers["LowestPace"] = one(pace)
    numbers["LowestDoublingsNeeded"] = in_words(doublings_needed)
    numbers["LowestTimesNeeded"] = nice(2 ** doublings_needed)
    numbers["LongerGapSmall"] = one(setting_score(counted, 64, "unequal", 16, 8) - longer["unequal", 1024, 8]["facts"])
    # The harder parts of the test at the end of the longer runs.
    for score, name in (("questions_one", "Questions"), ("questions_new", "New"), ("not_known", "Not")):
        numbers[f"Longer{name}Low"] = whole(min(run[score] for run in runs))
        numbers[f"Longer{name}High"] = whole(max(run[score] for run in runs))
    # A longer run against the counted repeat it shares its start with (repeat 0 of its setting).
    same_start = {setting: pick(stage, use=setting[0], creatures=setting[1], body=setting[2], repeat=0)[0]
                  for setting in longer}
    assert max(abs(longer[s]["at"][400]["facts"] - same_start[s]["at"][400]["facts"]) for s in longer) < 0.01  # "begin alike"
    # How far each had drifted from that repeat at the last step at which both were measured.
    drift = {s: longer[s]["at"][shared_step]["facts"] - same_start[s]["at"][shared_step]["facts"] for s in longer}
    furthest = max(drift, key=lambda s: abs(drift[s]))
    assert furthest == ("equal", 1024, 8) and drift[furthest] < 0  # the setting the text names: "3.0 points below"
    numbers["SharedDifferenceHigh"] = one(abs(drift[furthest]))
    numbers["SharedDifferenceOthers"] = one(max(abs(d) for s, d in drift.items() if s != furthest))
    numbers["OtherLongerWord"] = in_words(len(longer) - 1)
    numbers["LongerThreadsDifferWord"] = in_words(sum(longer[s]["threads"] != same_start[s]["threads"] for s in longer))
    # The example in the text: the setting whose longer run ended between its two counted repeats.
    example = ("equal", 1024, 8)
    pair = sorted(run["facts"] for run in pick(stage, use=example[0], creatures=example[1], body=example[2]))
    assert pair[0] < longer[example]["facts"] < pair[1]
    numbers["ExampleRepeatLow"], numbers["ExampleRepeatHigh"] = one(pair[0]), one(pair[1])
    numbers["ExampleLongerShared"] = one(longer[example]["at"][shared_step]["facts"])
    numbers["ExampleLongerLate"] = one(longer[example]["at"][measured[-2]]["facts"])
    numbers["ExampleLongerEnd"] = one(longer[example]["facts"])
    return numbers


def table_of_sixty_four_bits(counted, longer, shared_step):
    """Appendix C: one row for each of the 18 settings at 64 bits, counted and longer side by side."""
    stage = pick(counted, bits=64)
    rows = []
    for use in USES:
        for creatures in CREATURES_64:
            for body in BODIES_64:
                group = sorted(pick(stage, use=use, creatures=creatures, body=body), key=lambda run: run["repeat"])
                first, second = (run["facts"] for run in group)
                # The counted runs at their end; then repeat 1 and the longer run at the last step
                # at which both were measured; then the longer run at its end.
                cells = [use if (creatures, body) == (16, 8) else "", nice(creatures) if body == 8 else "", str(body),
                         one(first), one(second), one((first + second) / 2),
                         whole(middle(run["questions_one"] for run in group)),
                         one(group[0]["at"][shared_step]["facts"])]
                run = longer.get((use, creatures, body))
                if run:
                    start, end = run["at"][shared_step]["facts"], run["facts"]
                    reached = "\\reached{" + one(end) + "}" if end >= KNOWN else one(end)
                    cells += [one(start), reached, "+" + one(end - start), whole(run["questions_one"])]
                else:
                    cells += ["\\none"] * 4
                rows.append(" & ".join(cells) + " \\\\")
                if body == BODIES_64[-1] and not (use == USES[-1] and creatures == CREATURES_64[-1]):
                    rows.append("\\addlinespace[3pt]" if creatures != CREATURES_64[-1] else "\\midrule")
    return rows


# ---------------------------------------------------------------------------------------------
# The sentences of the report that state a fact without a number
# ---------------------------------------------------------------------------------------------

def check_the_claims(counted, longer, shared_step):
    """Stops with an error if the data no longer supports a sentence of the report."""
    setting = lambda bits, use, creatures, body: setting_score(counted, bits, use, creatures, body)
    stage_64 = pick(counted, bits=64)

    # "Every setting learned the facts, except where many creatures meet 64 bits."
    for bits in (8, 16, 32):
        assert all(setting(bits, use, creatures, body) >= KNOWN
                   for use in USES for creatures in CREATURES for body in BODIES)
    assert all(creatures >= 256 for _, creatures, _ in longer)

    # "All runs at 32 bits still knew 95 or more."
    assert all(run["facts"] >= KNOWN for run in pick(counted, bits=32))

    # "Up to 16 bits every score stays at the top for every vocabulary."
    for bits in (8, 16):
        for use in USES:
            for creatures in CREATURES:
                group = pick(counted, bits=bits, use=use, creatures=creatures)
                assert all(middle(run[score] for run in group) >= 99 for score, _, _ in SCORE_PANELS)

    # "First the 'not' sentences and the questions about several creatures, then the questions
    # about one creature, the plain facts last": fewer runs pass the bar the earlier a part gives way.
    passed = lambda score: sum(run[score] >= KNOWN for run in pick(counted, bits=32))
    assert max(passed("not_known"), passed("questions_new")) < passed("questions_one") < passed("facts")

    # "The harder parts of the test began to slip at 32 bits, and more with more creatures": with
    # 16 creatures every score is still at the top there; with 1,024 the harder ones are lower.
    for use in USES:
        small, large = (pick(counted, bits=32, use=use, creatures=creatures) for creatures in (16, 1024))
        for score in ("questions_one", "questions_new", "not_known"):
            assert middle(run[score] for run in small) >= 99.9 > middle(run[score] for run in large)

    # "P2 and P3 failed at 8, 16 and 32 bits; at 64 bits the gap that P2 predicts opens."
    for bits in (8, 16, 32):
        for use in USES:
            measures = prediction_measures(counted, bits, use)
            assert measures["P2"] < 10 and measures["P3"] < 10 and measures["P4"] <= 10
    # "With unequal use P2 would hold. With equal use the gap is just short of 10 and just inside
    # P4. P3 would not hold. Read by the rule, this stage would be mixed."
    equal, unequal = (prediction_measures(counted, 64, use) for use in USES)
    assert unequal["P2"] >= 10 and 9 < equal["P2"] < 10 and 9 < equal["P4"] < 10
    assert equal["P3"] < 10 and unequal["P3"] < 10
    assert verdict_from_middles(counted, 64) == "mixed"

    # "Unequal use made it worse": never better than equal use, in any of the nine pairs of settings.
    assert all(setting(64, "unequal", creatures, body) <= setting(64, "equal", creatures, body)
               for creatures in CREATURES_64 for body in BODIES_64)

    # "With 1,024 creatures and equal use only the widest body passed 95": the two narrower
    # settings ended below it, and both repeats of the widest at 95 or more.
    assert setting(64, "equal", 1024, 8) < KNOWN and setting(64, "equal", 1024, 32) < KNOWN
    assert all(run["facts"] >= KNOWN for run in pick(stage_64, use="equal", creatures=1024, body=128))

    # Figure 8 puts a 64-bit setting in the lane "not yet" exactly where it was trained longer.
    for use in USES:
        for creatures in CREATURES_64:
            for body in BODIES_64:
                group = pick(stage_64, use=use, creatures=creatures, body=body)
                assert (learning_time(group) == NEVER) == ((use, creatures, body) in longer)

    # "Every one gained, but more slowly as it went on": each longer run gained over each of the
    # last two doublings of its training, and gained less for each step of training in the later one.
    measured = sorted(next(iter(longer.values()))["at"])
    nearest = lambda target: min(measured, key=lambda step: abs(step - target))
    second = nearest(2 * shared_step)
    third = nearest(2 * second)
    for run in longer.values():
        earlier = run["at"][second]["facts"] - run["at"][shared_step]["facts"]
        later = run["at"][third]["facts"] - run["at"][second]["facts"]
        assert earlier > 0 and later > 0
        assert later / (third - second) < earlier / (second - shared_step)

    # "The order did not change": after the longer training, 1,024 creatures still below 256 at
    # every body width, and unequal use still below equal use. (That a wider body still scored
    # higher is tested where the numbers are made, in numbers_at_sixty_four_bits.)
    for body in BODIES_64:
        assert longer["unequal", 1024, body]["facts"] < longer["unequal", 256, body]["facts"]
    for use, creatures, body in longer:
        if use == "equal":
            assert longer["unequal", creatures, body]["facts"] < longer[use, creatures, body]["facts"]

    # Appendix B: the two runs still under way when the longer runs were started "ended at 100".
    assert all(exact(run["facts"]) == "100" for run in pick(stage_64, use="equal", creatures=256, body=128))

    # "The lowest (1,024 creatures, unequal use, body width 8)."
    assert min(longer, key=lambda key: longer[key]["facts"]) == ("unequal", 1024, 8)
    # "The 14 runs below 95 were all still rising."
    late = stage_64[0]["curve"][-3]["step"]
    assert all(run["facts"] > run["at"][late]["facts"] for run in stage_64 if run["facts"] < KNOWN)
    # "A wider body helped there too" (32 bits, unequal use, "not" known, 64 creatures or more).
    larger = [run for run in pick(counted, bits=32, use="unequal") if run["creatures"] > 16]
    assert (middle(run["not_known"] for run in pick(larger, body=8)) + 10
            < middle(run["not_known"] for run in pick(larger, body=128)))


if __name__ == "__main__":
    counted_runs = read_counted()
    longer_runs = read_longer()
    shared = last_shared_step(counted_runs, longer_runs)
    check_examples()
    check_the_claims(counted_runs, longer_runs, shared)
    figure_result_map(counted_runs, longer_runs, shared)
    figure_language()
    figure_scoring()
    figure_model(counted_runs)
    figure_runs_map(counted_runs, longer_runs)
    figure_predictions(counted_runs)
    figure_learning_curves(counted_runs)
    figure_learning_time(counted_runs)
    figure_scores_by_bits(counted_runs)
    figure_sixty_four(counted_runs, longer_runs, shared)
    figure_longer_curves(counted_runs, longer_runs)
    # Figures left from an earlier version of the report, under other names, are removed.
    for old in OUT.glob("*.pdf"):
        if old.stem not in WRITTEN:
            old.unlink()
    for name, value in write_numbers(counted_runs, longer_runs, shared).items():
        print(f"{name:32} {value}")

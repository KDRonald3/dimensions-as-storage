"""Draw every run of the counted stage: all three repeats of every configuration.

    .venv/bin/python -m v3.repeats_picture     # writes logs/v3/repeats_columns_8.png, _16, _32, _64

One picture for each number of table columns. In a picture: the left half is equal word use,
the right half unequal use; across are the bits per creature; down are the numbers of creatures
(four blocks) and, inside each, the body widths. Each place holds three small squares: the three
repeats of that configuration, each with the facts it knew at its end, out of 100. White means
every fact was known; the darker, the closer to a guess (25). It only reads.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from v3 import grid

ROOT = Path(__file__).resolve().parent.parent
INK, SOFT_INK, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
# facts missed: white when everything is known, dark when a model only guesses (as in scores_picture)
MISSED = LinearSegmentedColormap.from_list("missed", ["#ffffff", "#fde4cf", "#f8a56b", "#d9480f", "#8c2d04"])
BITS = [2 * k for k in grid.KINDS]
BODIES = sorted(grid.BODIES)
USES = [("equal", "Equal use: every creature appears equally often"),
        ("zipf", "Unequal use: a few creatures appear far more often")]


def read(stage="counted"):
    """{(columns, body, bits, creatures, use): [facts of repeat 1, 2, 3]} (in the order of the seeds)."""
    found = {}
    for config in grid.counted(grid.read_settings()):
        summary = ROOT / "runs_v3" / stage / config["name"] / "summary.json"
        if summary.exists():
            key = (config["columns"], config["body"], 2 * config["kinds"], config["creatures"], config["usage"])
            found.setdefault(key, {})[config["seed"]] = json.loads(summary.read_text())["final"]["facts"]
    return {key: [seeds[s] for s in sorted(seeds)] for key, seeds in found.items()}


def draw(runs, columns, out):
    rows = len(grid.CREATURES) * len(BODIES)
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 9.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.10})
    gap = 0.45  # white space between two blocks of creatures
    for axis, (use, use_label) in zip(axes, USES):
        axis.set_facecolor(SURFACE)
        ticks, labels = [], []
        for block, creatures in enumerate(grid.CREATURES):
            top = block * (len(BODIES) + gap)
            axis.text(-1.02, top + len(BODIES) / 2, f"{creatures:,}\ncreatures", ha="center", va="center",
                      fontsize=9, color=INK, rotation=0, linespacing=1.3)
            for line, body in enumerate(BODIES):
                y = top + line
                ticks.append(y + 0.5); labels.append(str(body))
                for x, bits in enumerate(BITS):
                    for place, facts in enumerate(runs.get((columns, body, bits, creatures, use), [])):
                        missed = min(max((100 - facts) / 75, 0), 1)
                        left = x + 0.035 + place * 0.31
                        axis.add_patch(plt.Rectangle((left, y + 0.07), 0.30, 0.86, facecolor=MISSED(missed),
                                                     edgecolor=LINE, linewidth=0.5))
                        axis.text(left + 0.15, y + 0.52, f"{facts:.0f}", ha="center", va="center", fontsize=7.6,
                                  color="white" if missed > 0.5 else INK)
        axis.set_xlim(-1.45, len(BITS)); axis.set_ylim(len(grid.CREATURES) * (len(BODIES) + gap) - gap, -0.1)
        axis.set_xticks([i + 0.5 for i in range(len(BITS))], [f"{b} bits" for b in BITS])
        axis.xaxis.set_ticks_position("top")
        axis.set_yticks(ticks, labels)
        axis.tick_params(length=0, labelsize=9, colors=SOFT_INK, pad=3)
        for label in axis.get_yticklabels():
            label.set_x(0.175)
        axis.set_title(use_label, fontsize=11, color=INK, pad=24, loc="left", x=0.17)
        for side in axis.spines.values():
            side.set_visible(False)
        axis.text(-0.2, -0.62, "body\nwidth", ha="center", va="center", fontsize=8, color=SOFT_INK, linespacing=1.2)
    figure.suptitle(f"Word table with {columns} columns: facts known, out of 100, by every run (three repeats of each configuration)",
                    x=0.03, y=0.985, ha="left", fontsize=13.5, color=INK)
    figure.text(0.03, 0.948, "100: knows every fact", ha="left", va="center", fontsize=9, color=SOFT_INK)
    key = figure.add_axes([0.135, 0.940, 0.10, 0.017])
    key.imshow([[i / 99 for i in range(100)]], aspect="auto", cmap=MISSED, extent=(100, 25, 0, 1))
    key.set_xticks([]); key.set_yticks([])
    for side in key.spines.values():
        side.set_edgecolor(LINE)
    figure.text(0.24, 0.948, "25: only guessing", ha="left", va="center", fontsize=9, color=SOFT_INK)
    figure.text(0.36, 0.948, "The three squares in a place are the three repeats of one configuration: "
                "the same settings, started from different random numbers.", ha="left", va="center", fontsize=9,
                color=SOFT_INK)
    figure.subplots_adjust(left=0.03, right=0.99, top=0.865, bottom=0.015)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    found = read()
    for columns in grid.COLUMNS:
        out = ROOT / "logs" / "v3" / f"repeats_columns_{columns}.png"
        draw(found, columns, out)
        print(out)
    print(len(found), "configurations,", sum(len(v) for v in found.values()), "runs")

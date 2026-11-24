"""Draw the scores of a stage in the plain terms of the experiment: table columns, bits per
creature and number of creatures.

    .venv/bin/python -m v3.scores_picture     # writes logs/v3/scores_tiles.png and scores_lines.png

For every combination of table columns, bits and creatures there are 36 runs (6 body widths,
2 word uses, 3 repeats). The pictures show the facts known by the middle one of those 36: half
score higher and half lower. A run that has ended counts with its final score, a run still
training with its latest score; how many runs of a combination are still training is shown.
It only reads.

Three readers who knew nothing about the project tested the first versions on 23 November
2026. What they asked for is built in: the scale is written on the picture (25 is a guess),
"not final" is a count and not a flag, lines are named where they end, points that fall on
each other are set side by side, and the two pictures no longer use the same blue for
opposite things (here blue is the number of table columns, orange-brown is facts missed).
"""
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from v3 import grid

ROOT = Path(__file__).resolve().parent.parent
INK, SOFT_INK, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
# facts missed: white when everything is known, dark when a model only guesses
MISSED = LinearSegmentedColormap.from_list("missed", ["#ffffff", "#fde4cf", "#f8a56b", "#d9480f", "#8c2d04"])
# the number of table columns: one blue, light (8) to dark (64)
COLUMN_COLOURS = {8: "#86b6ef", 16: "#3987e5", 32: "#1c5cab", 64: "#0d366b"}
BITS = [2 * k for k in grid.KINDS]
RUNS = 36  # runs of one combination


def read(stage="counted"):
    """{(columns, bits, creatures): (middle of facts known, runs still training)}."""
    scores, training = {}, {}
    for config in grid.counted(grid.read_settings()):
        key = (config["columns"], 2 * config["kinds"], config["creatures"])
        summary = ROOT / "runs_v3" / stage / config["name"] / "summary.json"
        so_far = ROOT / "runs_v3" / stage / "_so_far" / f"{config['name']}.json"
        if summary.exists():
            scores.setdefault(key, []).append(json.loads(summary.read_text())["final"]["facts"])
        elif so_far.exists():
            scores.setdefault(key, []).append(json.loads(so_far.read_text())["curve"][-1]["facts"])
            training[key] = training.get(key, 0) + 1
    return {key: (statistics.median(found), training.get(key, 0)) for key, found in scores.items()}


def status(cells):
    left = sum(t for _, t in cells.values())
    when = f"{time.strftime('%H:%M')} Pacific on {time.strftime('%d %B')}"
    return when + (f"; {left:,} of 2,880 runs still training" if left else "; every run has ended")


def tiles(cells, out):
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 4.5), facecolor=SURFACE, gridspec_kw={"wspace": 0.16})
    for axis, columns in zip(axes, grid.COLUMNS):
        axis.set_facecolor(SURFACE)
        for y, creatures in enumerate(grid.CREATURES):
            for x, bits in enumerate(BITS):
                score, left = cells[(columns, bits, creatures)]
                missed = min(max((100 - score) / 75, 0), 1)
                axis.add_patch(plt.Rectangle((x + 0.04, y + 0.05), 0.92, 0.9, facecolor=MISSED(missed),
                                             edgecolor=LINE, linewidth=0.6))
                ink = "white" if missed > 0.5 else INK
                axis.text(x + 0.5, y + (0.42 if left else 0.52), f"{score:.0f}", ha="center", va="center",
                          fontsize=13, color=ink)
                if left:  # how many of its 36 runs are still training
                    axis.text(x + 0.5, y + 0.78, f"{left} left", ha="center", va="center", fontsize=7.5,
                              color="white" if missed > 0.5 else SOFT_INK)
        axis.set_xlim(0, len(BITS)); axis.set_ylim(len(grid.CREATURES), 0)
        axis.set_xticks([i + 0.5 for i in range(len(BITS))], [str(b) for b in BITS])
        axis.set_yticks([i + 0.5 for i in range(len(grid.CREATURES))], [f"{c:,}" for c in grid.CREATURES])
        axis.tick_params(length=0, labelsize=9.5, colors=SOFT_INK)
        axis.set_title(f"word table with {columns} columns", fontsize=11.5, color=INK, pad=8)
        axis.set_xlabel("bits per creature", fontsize=9.5, color=SOFT_INK)
        for side in axis.spines.values():
            side.set_visible(False)
        axis.set_aspect("equal")
    axes[0].set_ylabel("creatures in the language", fontsize=9.5, color=SOFT_INK)
    open_runs = sum(t for _, t in cells.values())
    figure.suptitle("Facts known" + (" so far" if open_runs else "") + ", out of 100, by the typical model",
                    x=0.045, y=0.985, ha="left", fontsize=14, color=INK)
    figure.text(0.985, 0.985, status(cells), ha="right", va="top", fontsize=9, color=SOFT_INK)
    # the scale, on the picture itself
    figure.text(0.047, 0.895, "100: knows every fact", ha="left", va="center", fontsize=9, color=SOFT_INK)
    key = figure.add_axes([0.158, 0.88, 0.11, 0.03])
    key.imshow([[i / 99 for i in range(100)]], aspect="auto", cmap=MISSED, extent=(100, 25, 0, 1))
    key.set_yticks([]); key.set_xticks([])
    for side in key.spines.values():
        side.set_edgecolor(LINE)
    figure.text(0.275, 0.895, "25: only guessing", ha="left", va="center", fontsize=9, color=SOFT_INK)
    if open_runs:
        figure.text(0.40, 0.895, "Small print in a square: how many of its 36 runs are still training; "
                    "its number can still rise.", ha="left", va="center", fontsize=9, color=SOFT_INK)
    else:
        figure.text(0.40, 0.895, "Each number is the middle one of the 36 runs with that combination.",
                    ha="left", va="center", fontsize=9, color=SOFT_INK)
    figure.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.08)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def lines(cells, out):
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 4.9), facecolor=SURFACE, sharey=True,
                                gridspec_kw={"wspace": 0.16, "width_ratios": [1, 1, 1, 1.36]})
    for axis, creatures in zip(axes, grid.CREATURES):
        axis.set_facecolor(SURFACE)
        for place, columns in enumerate(grid.COLUMNS):
            colour = COLUMN_COLOURS[columns]
            # the four lines stand a little apart sideways, so that points at the same height
            # (often 100) do not hide each other
            xs = [x + (place - 1.5) * 0.07 for x in range(len(BITS))]
            ys = [cells[(columns, bits, creatures)][0] for bits in BITS]
            still = [cells[(columns, bits, creatures)][1] > 0 for bits in BITS]
            for i in range(len(BITS) - 1):  # dashed where an end of the piece is not final
                axis.plot(xs[i:i + 2], ys[i:i + 2], color=colour, linewidth=2.2, zorder=2,
                          linestyle=(0, (3, 2)) if still[i] or still[i + 1] else "-")
            for x, y, open_ in zip(xs, ys, still):
                axis.plot([x], [y], "o", markersize=6.5, color=colour, zorder=3,
                          markerfacecolor=SURFACE if open_ else colour, markeredgewidth=1.6)
            if creatures == grid.CREATURES[-1]:  # the lines are named where they end
                axis.text(xs[-1] + 0.16, ys[-1], f"{columns} columns",
                          ha="left", va="center", fontsize=9.5, color=INK)
        axis.axhline(25, color=SOFT_INK, linewidth=0.9, linestyle=(0, (1, 2)))
        axis.set_ylim(18, 104)
        axis.set_xlim(-0.35, len(BITS) - 0.65 + (1.45 if creatures == grid.CREATURES[-1] else 0))
        axis.set_yticks([25, 50, 75, 100])
        axis.set_xticks(range(len(BITS)), [str(b) for b in BITS])
        axis.tick_params(length=0, labelsize=9.5, colors=SOFT_INK, labelleft=True)
        axis.grid(axis="y", color=LINE, linewidth=0.6); axis.set_axisbelow(True)
        axis.set_title(f"language with {creatures:,} creatures", fontsize=11.5, color=INK, pad=8, loc="left")
        axis.set_xlabel("bits per creature", fontsize=9.5, color=SOFT_INK, loc="left")
        for side in axis.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("facts known, out of 100", fontsize=9.5, color=SOFT_INK)
    axes[0].text(-0.3, 26.5, "25: only guessing", fontsize=8.5, color=SOFT_INK, va="bottom")
    handles = [plt.Line2D([], [], color=SOFT_INK, linewidth=2.2, marker="o", markersize=6.5,
                          label="final: every run of the point has ended"),
               plt.Line2D([], [], color=SOFT_INK, linewidth=2.2, linestyle=(0, (3, 2)), marker="o", markersize=6.5,
                          markerfacecolor=SURFACE, markeredgewidth=1.6,
                          label="not final: some of its runs are still training, the point can still rise")]
    figure.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.04, 0.925),
                  columnspacing=2.0, handlelength=3.2)
    figure.suptitle("Facts known" + ("" if not sum(t for _, t in cells.values()) else " so far")
                    + " by the typical model, for each number of table columns",
                    x=0.045, y=0.985, ha="left", fontsize=14, color=INK)
    figure.text(0.985, 0.985, status(cells), ha="right", va="top", fontsize=9, color=SOFT_INK)
    figure.subplots_adjust(left=0.055, right=0.99, top=0.78, bottom=0.115)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    found = read()
    tiles(found, ROOT / "logs" / "v3" / "scores_tiles.png")
    lines(found, ROOT / "logs" / "v3" / "scores_lines.png")
    print(len(found), "combinations;", sum(t for _, t in found.values()), "runs still training")

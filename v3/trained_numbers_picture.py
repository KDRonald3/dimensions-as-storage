"""Draw facts known against the number of trained numbers, for one language of the counted stage.

    .venv/bin/python -m v3.trained_numbers_picture     # writes logs/v3/trained_numbers.png

The question behind it: when more table columns help, is it the wider space or simply the larger
number of trained numbers? The picture shows why the counted stage cannot tell: every model with
more columns also has more trained numbers, so the four column counts never meet on the size
axis. Every run is a dot (three repeats for each body width); the line joins the middle one of
the three. It only reads.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v3 import grid
from v3.scores_picture import COLUMN_COLOURS, INK, LINE, SOFT_INK, SURFACE

ROOT = Path(__file__).resolve().parent.parent
CREATURES, BITS, USE = 1024, 128, "equal"
GAP = "#f1f0ec"  # sizes at which no model was run


def read():
    """{columns: [(body, trained numbers, [facts of the three repeats])], smallest body first}."""
    found = {}
    for summary in (ROOT / "runs_v3" / "counted").glob("*/summary.json"):
        run = json.loads(summary.read_text())
        config = run["config"]
        if (config["creatures"], 2 * config["kinds"], config["usage"]) != (CREATURES, BITS, USE):
            continue
        key = (config["columns"], config["body"])
        found.setdefault(key, [run["parameters"]["trained"], []])[1].append(run["final"]["facts"])
    lines = {}
    for (columns, body), (trained, facts) in sorted(found.items()):
        lines.setdefault(columns, []).append((body, trained, sorted(facts)))
    return lines


def draw(lines, out):
    figure, axis = plt.subplots(figsize=(10.5, 5.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.set_xscale("log")
    columns_in_order = sorted(lines)
    # the sizes between two column counts: no model was run there
    for narrow, wide in zip(columns_in_order, columns_in_order[1:]):
        axis.axvspan(lines[narrow][-1][1] * 1.035, lines[wide][0][1] / 1.035, color=GAP, zorder=0)
    for columns in columns_in_order:
        colour = COLUMN_COLOURS[columns]
        sizes = [trained for _, trained, _ in lines[columns]]
        middles = [statistics.median(facts) for _, _, facts in lines[columns]]
        axis.plot(sizes, middles, color=colour, linewidth=2.2, zorder=2)
        for _, trained, facts in lines[columns]:
            axis.plot([trained] * len(facts), facts, "o", color=colour, markersize=6.5, zorder=3,
                      markeredgecolor=SURFACE, markeredgewidth=1.0)
        low = min(min(facts) for _, _, facts in lines[columns])
        high = max(max(facts) for _, _, facts in lines[columns])
        middle_size = (sizes[0] * sizes[-1]) ** 0.5
        if high > 95:   # no room above the top line
            axis.text(middle_size, low - 10, f"{columns} columns", ha="center", va="top", fontsize=11.5, color=INK)
        else:
            axis.text(middle_size, high + 4.5, f"{columns} columns", ha="center", va="bottom", fontsize=11.5, color=INK)
        # the two ends of a line: the smallest and the largest body
        for place, nudge in ((0, 1 / 1.03), (-1, 1.03)):  # the labels sit under the lowest repeat
            body, trained, facts = lines[columns][place]
            axis.text(trained * nudge, min(facts) - 3.0, f"body {body}", ha="center", va="top",
                      fontsize=8.5, color=SOFT_INK)
    first_gap = (lines[columns_in_order[0]][-1][1] * lines[columns_in_order[1]][0][1]) ** 0.5
    axis.text(first_gap, 86, "grey: no model\nof this size\nwas run", ha="center", va="center", fontsize=9,
              color=SOFT_INK, linespacing=1.35)
    axis.axhline(25, color=SOFT_INK, linewidth=0.9, linestyle=(0, (1, 2)))
    axis.text(13200, 26.3, "25: only guessing", fontsize=9, color=SOFT_INK, va="bottom")
    axis.set_ylim(18, 106)
    axis.set_xlim(12800, 205000)
    axis.set_yticks([25, 50, 75, 100])
    ticks = [15000, 30000, 60000, 120000]
    axis.set_xticks(ticks, [f"{t:,}" for t in ticks])
    axis.minorticks_off()
    axis.tick_params(length=0, labelsize=10, colors=SOFT_INK)
    axis.grid(axis="y", color=LINE, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlabel("trained numbers in the whole model (each mark is twice the one before)", fontsize=10,
                    color=SOFT_INK, loc="left")
    axis.set_ylabel("facts known, out of 100", fontsize=10, color=SOFT_INK)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.suptitle("Every model with more columns also had more trained numbers",
                    x=0.045, y=0.975, ha="left", fontsize=14.5, color=INK)
    figure.text(0.045, 0.895, f"Each dot is one run: {CREATURES:,} creatures, {BITS} bits per creature, words used "
                "equally. Three repeats for each body width; the line joins the middle one.",
                ha="left", va="center", fontsize=9.5, color=SOFT_INK)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.115)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    found = read()
    draw(found, ROOT / "logs" / "v3" / "trained_numbers.png")
    for columns, points in found.items():
        print(columns, "columns:", "; ".join(f"body {b}: {t:,} -> {'/'.join(f'{x:.0f}' for x in f)}" for b, t, f in points))

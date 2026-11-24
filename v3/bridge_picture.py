"""Draw facts known against trained numbers for every bit size, with the bridging runs.

    .venv/bin/python -m v3.bridge_picture      # writes logs/v3/bridge.png

One panel per bit size (1,024 creatures, words used equally). In each, every counted run is a dot
(three repeats for each body width from 4 to 128) and the line joins the middle of the three. The
bridging runs (v3/grid.py, bridge) are diamonds: the narrower model with a body so wide that it
has as many trained numbers as the model with twice the columns and a body of 4, so a diamond
stands straight below or above the first dot of the next column count. A dotted line joins it to
its own column count: no model between the two was run. A bridging run still training is drawn
hollow, at its latest score. It only reads.
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
DODGE = 1.09  # a diamond stands this factor to the left of its true size


def read_stage(stage, wanted):
    """{(columns, kinds, body): [(trained numbers, facts, final?)]} for the runs of a stage with
    1,024 creatures and equal use; runs still training count with their latest score."""
    found = {}
    folder = ROOT / "runs_v3" / stage
    for config in wanted:
        key = (config["columns"], config["kinds"], config["body"])
        summary = folder / config["name"] / "summary.json"
        so_far = folder / "_so_far" / f"{config['name']}.json"
        if summary.exists():
            run = json.loads(summary.read_text())
            found.setdefault(key, []).append((run["parameters"]["trained"], run["final"]["facts"], True))
        elif so_far.exists():
            run = json.loads(so_far.read_text())
            found.setdefault(key, []).append((None, run["curve"][-1]["facts"], False))
    return found


def read():
    settings = grid.read_settings()
    counted = [c for c in grid.counted(settings)
               if c["creatures"] == grid.BRIDGE_CREATURES and c["usage"] == grid.BRIDGE_USAGE]
    return read_stage("counted", counted), read_stage("bridge", grid.bridge(settings))


def size_of(points, fallback):
    sizes = [size for size, _, _ in points if size]
    return sizes[0] if sizes else fallback


def panel(axis, kinds, counted, bridged):
    axis.set_facecolor(SURFACE)
    axis.set_xscale("log")
    for columns in grid.COLUMNS:
        colour = COLUMN_COLOURS[columns]
        bodies = [b for b in grid.BODIES if (columns, kinds, b) in counted]
        sizes = [size_of(counted[(columns, kinds, b)], None) for b in bodies]
        middles = [statistics.median(f for _, f, _ in counted[(columns, kinds, b)]) for b in bodies]
        axis.plot(sizes, middles, color=colour, linewidth=2.0, zorder=2)
        for size, body in zip(sizes, bodies):
            facts = [f for _, f, _ in counted[(columns, kinds, body)]]
            axis.plot([size] * len(facts), facts, "o", color=colour, markersize=4.6, zorder=3,
                      markeredgecolor=SURFACE, markeredgewidth=0.7)
        # the line's name, above its right end (below it when the line runs along the top); its
        # left end is where the diamond of the column count before it stands
        facts = [f for b in bodies for _, f, _ in counted[(columns, kinds, b)]]
        middle_size = sizes[-1]
        if max(facts) < 93:
            axis.text(middle_size, max(facts) + 3, f"{columns} col.", ha="center", va="bottom",
                      fontsize=8.5, color=INK)
        else:
            axis.text(middle_size * 0.97, min(facts) - 3, f"{columns} col.", ha="left", va="top",
                      fontsize=8.5, color=INK)
        if columns == grid.COLUMNS[-1]:
            continue
        body = grid.BRIDGE_BODIES[(kinds, columns)]
        points = bridged.get((columns, kinds, body), [])
        if not points:
            continue
        # the bridge stands at the size of the first model with twice the columns
        target = size_of(counted[(2 * columns, kinds, 4)], None)
        # (drawn a little to the left, so that it does not hide the dots of the wider model of
        # the same size, which are the ones to compare it with)
        size = size_of(points, target) / DODGE
        middle = statistics.median(f for _, f, _ in points)
        axis.plot([sizes[-1], size], [middles[-1], middle], color=colour, linewidth=1.6,
                  linestyle=(0, (1, 2)), zorder=2)
        for _, facts, final in points:
            axis.plot([size], [facts], "D", color=colour, markersize=6.0, zorder=4,
                      markerfacecolor=colour if final else SURFACE, markeredgewidth=1.3)
    axis.axhline(25, color=SOFT_INK, linewidth=0.8, linestyle=(0, (1, 2)))
    axis.set_ylim(18, 104)
    axis.set_xlim(5000, 260000)
    ticks = [10000, 30000, 100000]
    axis.set_xticks(ticks, ["10,000", "30,000", "100,000"])
    axis.minorticks_off()
    axis.set_yticks([25, 50, 75, 100])
    axis.tick_params(length=0, labelsize=8.5, colors=SOFT_INK)
    axis.grid(axis="y", color=LINE, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_title(f"{2 * kinds} bits per creature", fontsize=11, color=INK, loc="left", pad=19)
    widths = ", ".join(f"{grid.BRIDGE_BODIES[(kinds, c)]}" for c in grid.COLUMNS[:-1])
    axis.text(0, 1.035, f"bridging bodies (8, 16, 32 col.): {widths}", transform=axis.transAxes,
              fontsize=8, color=SOFT_INK, va="bottom")
    for side in axis.spines.values():
        side.set_visible(False)


def draw(counted, bridged, out):
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.4), facecolor=SURFACE,
                                gridspec_kw={"wspace": 0.18, "hspace": 0.38})
    for axis, kinds in zip(axes.flat, grid.KINDS):
        panel(axis, kinds, counted, bridged)
    for axis in axes[:, 0]:
        axis.set_ylabel("facts known, out of 100", fontsize=9, color=SOFT_INK)
    for axis in axes[1, :2]:
        axis.set_xlabel("trained numbers in the whole model", fontsize=9, color=SOFT_INK, loc="left")
    axes[0, 2].set_xlabel("trained numbers in the whole model", fontsize=9, color=SOFT_INK, loc="left")
    # the last place holds the key
    key = axes[1, 2]
    key.set_axis_off()
    lines = [
        ("o", "a counted run (body 4 to 128 neurons);", "the line joins the middle of three repeats"),
        ("D", "a bridging run: as many trained numbers as", "the next column count with body 4 (set a\n"
         "little to the left so as not to hide those dots)"),
    ]
    y = 0.92
    for marker, first, second in lines:
        key.plot([0.04], [y], marker, color=SOFT_INK, markersize=7, transform=key.transAxes, clip_on=False)
        key.text(0.11, y, first, transform=key.transAxes, fontsize=9.5, color=INK, va="center")
        key.text(0.11, y - 0.075, second, transform=key.transAxes, fontsize=9.5, color=INK, va="center")
        y -= 0.21
    key.plot([0.01, 0.07], [y, y], color=SOFT_INK, linewidth=1.6, linestyle=(0, (1, 2)),
             transform=key.transAxes)
    key.text(0.11, y, "no model of a size in between was run", transform=key.transAxes, fontsize=9.5,
             color=INK, va="center")
    y -= 0.13
    key.text(0.01, y, "Colour: the number of table columns (col.)", transform=key.transAxes,
             fontsize=9.5, color=INK, va="center")
    y -= 0.03
    for columns in grid.COLUMNS:
        key.plot([0.04 + 0.16 * grid.COLUMNS.index(columns)], [y - 0.07], "o", color=COLUMN_COLOURS[columns],
                 markersize=8, transform=key.transAxes)
        key.text(0.065 + 0.16 * grid.COLUMNS.index(columns), y - 0.07, str(columns), transform=key.transAxes,
                 fontsize=9.5, color=INK, va="center")
    open_runs = sum(1 for points in bridged.values() for _, _, final in points if not final)
    if open_runs:
        key.text(0.01, y - 0.19, f"Hollow diamond: still training ({open_runs} runs);\nits score can still rise.",
                 transform=key.transAxes, fontsize=9.5, color=SOFT_INK, va="center")
    figure.suptitle("At the same number of trained numbers, the model with more columns knew more",
                    x=0.04, y=0.975, ha="left", fontsize=14.5, color=INK)
    figure.text(0.04, 0.93, f"{grid.BRIDGE_CREATURES:,} creatures, words used equally. Every dot and diamond is "
                "one run. Compare each diamond with the dots just to its right: the same size, twice the columns.",
                ha="left", va="center", fontsize=9.5, color=SOFT_INK)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.07)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    counted, bridged = read()
    draw(counted, bridged, ROOT / "logs" / "v3" / "bridge.png")
    done = sum(1 for points in bridged.values() for _, _, final in points if final)
    print(f"bridging runs ended: {done} of 45")

"""Draw what a lower learning rate changed: each follow-up run against the main run it repeats.

    .venv/bin/python -m v3.rate_check_picture [rate_check_third]   # writes logs/v3/<stage>.png

A follow-up run trains the best of a configuration's three repeats again, from the same starting
numbers and on the same text, at a lower learning rate. Each square is one combination of table
columns (the panel), bits per creature (across) and creatures (down); it holds 12 configurations
(6 body widths, 2 word uses). The number is the middle change in facts known, lower rate minus
the rate of the experiment: blue is better with the lower rate, red worse. The small print says
how many of the 12 have finished and are compared. It only reads.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from v3 import grid
from v3.orchestrate import stage_groups

ROOT = Path(__file__).resolve().parent.parent
INK, SOFT_INK, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
# worse (red) ... no change (grey) ... better (blue)
CHANGE = LinearSegmentedColormap.from_list("change", ["#b02a2a", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
BITS = [2 * k for k in grid.KINDS]
LIMIT = 10.0  # a change of this many points (or more) gets the full colour


def read(stage):
    """{(columns, bits, creatures): [change in facts known for each finished configuration]},
    the same against the middle of the three main repeats, and how many runs are still training."""
    main = {}
    for config in grid.counted(grid.read_settings()):
        key = (config["columns"], config["creatures"], config["body"], config["usage"], config["kinds"])
        facts = json.loads((ROOT / "runs_v3" / "counted" / config["name"] / "summary.json").read_text())["final"]["facts"]
        main.setdefault(key, {})[config["seed"]] = facts
    same_run, middle, left = {}, {}, 0
    for config in (c for group in stage_groups(stage) for c in group):
        summary = ROOT / "runs_v3" / stage / config["name"] / "summary.json"
        if not summary.exists():
            left += 1
            continue
        facts = json.loads(summary.read_text())["final"]["facts"]
        repeats = main[(config["columns"], config["creatures"], config["body"], config["usage"], config["kinds"])]
        cell = (config["columns"], 2 * config["kinds"], config["creatures"])
        same_run.setdefault(cell, []).append(facts - repeats[config["seed"]])
        middle.setdefault(cell, []).append(facts - statistics.median(repeats.values()))
    return same_run, middle, left


def draw(stage, out):
    same_run, middle, left = read(stage)
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 4.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.16})
    for axis, columns in zip(axes, grid.COLUMNS):
        axis.set_facecolor(SURFACE)
        for y, creatures in enumerate(grid.CREATURES):
            for x, bits in enumerate(BITS):
                found = same_run.get((columns, bits, creatures), [])
                if not found:
                    axis.add_patch(plt.Rectangle((x + 0.04, y + 0.05), 0.92, 0.9, facecolor=SURFACE, edgecolor=LINE, linewidth=0.6))
                    axis.text(x + 0.5, y + 0.5, "none\nyet", ha="center", va="center", fontsize=7.5, color=SOFT_INK)
                    continue
                change = statistics.median(found)
                shade = 0.5 + max(-0.5, min(0.5, change / LIMIT / 2))
                axis.add_patch(plt.Rectangle((x + 0.04, y + 0.05), 0.92, 0.9, facecolor=CHANGE(shade), edgecolor=LINE, linewidth=0.6))
                ink = "white" if abs(shade - 0.5) > 0.3 else INK
                shown = "0" if abs(change) < 0.05 else f"{change:+.1f}"
                axis.text(x + 0.5, y + (0.42 if len(found) < 12 else 0.52), shown, ha="center", va="center", fontsize=12, color=ink)
                if len(found) < 12:
                    axis.text(x + 0.5, y + 0.78, f"{len(found)} of 12", ha="center", va="center", fontsize=7.5,
                              color="white" if abs(shade - 0.5) > 0.3 else SOFT_INK)
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
    rate = {"rate_check_third": "a third", "rate_check_tenth": "a tenth"}.get(stage, "a lower share")
    figure.suptitle(f"Change in facts known when the same run is trained at {rate} of the learning rate",
                    x=0.045, y=0.985, ha="left", fontsize=14, color=INK)
    when = f"{time.strftime('%H:%M')} Pacific on {time.strftime('%d %B')}"
    figure.text(0.985, 0.985, when + (f"; {left} of 960 runs still training" if left else "; every run has ended"),
                ha="right", va="top", fontsize=9, color=SOFT_INK)
    figure.text(0.047, 0.895, "worse by 10 points or more", ha="left", va="center", fontsize=9, color=SOFT_INK)
    key = figure.add_axes([0.186, 0.88, 0.13, 0.03])
    key.imshow([[i / 99 for i in range(100)]], aspect="auto", cmap=CHANGE)
    key.set_xticks([]); key.set_yticks([])
    for side in key.spines.values():
        side.set_edgecolor(LINE)
    figure.text(0.322, 0.895, "better by 10 or more", ha="left", va="center", fontsize=9, color=SOFT_INK)
    figure.text(0.45, 0.895, "Each number: the middle change over the square's 12 configurations, in points of 100. "
                "0 means no change.", ha="left", va="center", fontsize=9, color=SOFT_INK)
    figure.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.08)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return same_run, middle, left


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "rate_check_third"
    same_run, middle, left = draw(stage, ROOT / "logs" / "v3" / f"{stage}.png")
    a = [x for v in same_run.values() for x in v]; b = [x for v in middle.values() for x in v]
    print(f"compared {len(a)}; still training {left}")
    for label, d in (("against the same run at the chosen rate (the best of three)", a), ("against the middle of the three repeats", b)):
        print(f"  {label}: mean {statistics.mean(d):+.2f}, middle {statistics.median(d):+.2f}; better by 1+: {sum(x >= 1 for x in d)}, within 1: {sum(abs(x) < 1 for x in d)}, worse by 1+: {sum(x <= -1 for x in d)}")

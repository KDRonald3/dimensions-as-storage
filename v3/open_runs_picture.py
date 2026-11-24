"""Draw the latest score of every run of a stage that is still training.

    .venv/bin/python -m v3.open_runs_picture            # writes logs/v3/open_runs.png

One dot is one run that has not ended yet, at the score of its last measurement (from the
"so far" files the machines send home). Across: how many bits a creature carries per column of
the word table. Up: facts known so far. The colour is the number of creatures. It only reads.
"""
import glob
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v3 import grid

ROOT = Path(__file__).resolve().parent.parent
INK, SOFT_INK, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
# the number of creatures: one blue, light to dark
CREATURES = {16: "#86b6ef", 64: "#3987e5", 256: "#1c5cab", 1024: "#0d366b"}


def read(stage="counted"):
    ended = {os.path.basename(os.path.dirname(p)) for p in glob.glob(str(ROOT / "runs_v3" / stage / "*" / "summary.json"))}
    rows = []
    for config in grid.counted(grid.read_settings()):
        path = ROOT / "runs_v3" / stage / "_so_far" / f"{config['name']}.json"
        if config["name"] in ended or not path.exists():
            continue
        last = json.loads(path.read_text())["curve"][-1]
        rows.append({"ratio": 2 * config["kinds"] / config["columns"], "creatures": config["creatures"],
                     "facts": last["facts"], "questions": last["questions_one"]})
    return rows, len(ended)


def draw(out):
    rows, ended = read()
    ratios = sorted({r["ratio"] for r in rows})
    figure, axis = plt.subplots(figsize=(9.6, 5.3), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    scatter = random.Random(0)
    for x, ratio in enumerate(ratios):
        mine = [r for r in rows if r["ratio"] == ratio]
        for place, creatures in enumerate(CREATURES):
            points = [r for r in mine if r["creatures"] == creatures]
            # each number of creatures in a narrow column of its own, so the colours do not hide each other
            xs = [x + (place - 1.5) * 0.19 + scatter.uniform(-0.07, 0.07) for _ in points]
            axis.scatter(xs, [r["facts"] for r in points], s=13, color=CREATURES[creatures], alpha=0.75,
                         linewidths=0)
        axis.text(x, 15.5, f"{len(mine)} runs", ha="center", va="top", fontsize=9, color=SOFT_INK)
    axis.axhline(99, color=INK, linewidth=0.9)
    axis.text(len(ratios) - 0.42, 99.8, "99: counts as learned (with the questions at 99 too)", ha="right",
              va="bottom", fontsize=8.5, color=INK)
    axis.axhline(25, color=SOFT_INK, linewidth=0.9, linestyle=(0, (4, 3)))
    axis.text(len(ratios) - 0.42, 25.8, "25: a guess", ha="right", va="bottom", fontsize=8.5, color=SOFT_INK)
    axis.set_xlim(-0.6, len(ratios) - 0.4)
    axis.set_ylim(12, 106)
    axis.set_yticks([25, 50, 75, 100])
    axis.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    axis.tick_params(length=0, labelsize=10, colors=SOFT_INK)
    axis.set_xlabel("bits per creature for each column of the word table", fontsize=10, color=SOFT_INK, labelpad=8)
    axis.set_ylabel("facts known so far", fontsize=10, color=SOFT_INK)
    axis.grid(axis="y", color=LINE, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=colour, label=f"{creatures:,}")
               for creatures, colour in CREATURES.items()]
    legend = axis.legend(handles=handles, title="creatures in the language", loc="lower left", ncol=4,
                         frameon=False, fontsize=9, title_fontsize=9, bbox_to_anchor=(0.0, 1.0),
                         columnspacing=1.2, handletextpad=0.2)
    legend._legend_box.align = "left"
    figure.suptitle(f"The {len(rows):,} runs still training, by their latest score", x=0.075, y=0.985,
                    ha="left", fontsize=13.5, color=INK)
    figure.text(0.985, 0.955, f"{time.strftime('%H:%M')} Pacific on {time.strftime('%d %B')}; {ended:,} runs have ended",
                ha="right", fontsize=9, color=SOFT_INK)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.12)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return len(rows)


if __name__ == "__main__":
    out = ROOT / "logs" / "v3" / "open_runs.png"
    print(draw(out), out)

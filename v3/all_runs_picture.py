"""Draw every run of a stage at its score: the runs that have ended and the runs still training.

    .venv/bin/python -m v3.all_runs_picture            # writes logs/v3/all_runs.png

One dot is one run. Left: the runs that have ended, at their final score. Right: the runs still
training, at the score of their last measurement (from the "so far" files the machines send
home). Across: how many bits a creature carries per column of the word table. Up: facts known.
The colour is the number of creatures. A ring marks a run that ended at the limit of 640,000
steps. It only reads.
"""
import json
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
    ended, training = [], []
    for config in grid.counted(grid.read_settings()):
        row = {"ratio": 2 * config["kinds"] / config["columns"], "creatures": config["creatures"]}
        summary = ROOT / "runs_v3" / stage / config["name"] / "summary.json"
        so_far = ROOT / "runs_v3" / stage / "_so_far" / f"{config['name']}.json"
        if summary.exists():
            found = json.loads(summary.read_text())
            ended.append(dict(row, facts=found["final"]["facts"], reason=found["ended"]["reason"]))
        elif so_far.exists():
            training.append(dict(row, facts=json.loads(so_far.read_text())["curve"][-1]["facts"], reason=None))
    return ended, training


def panel(axis, rows, ratios, title, first):
    axis.set_facecolor(SURFACE)
    scatter = random.Random(0)
    for x, ratio in enumerate(ratios):
        mine = [r for r in rows if r["ratio"] == ratio]
        for place, creatures in enumerate(CREATURES):
            points = [r for r in mine if r["creatures"] == creatures]
            # each number of creatures in a narrow column of its own, so the colours do not hide each other
            xs = [x + (place - 1.5) * 0.19 + scatter.uniform(-0.07, 0.07) for _ in points]
            axis.scatter(xs, [r["facts"] for r in points], s=11, color=CREATURES[creatures], alpha=0.7, linewidths=0)
            ringed = [(a, r["facts"]) for a, r in zip(xs, points) if r["reason"] == "limit"]
            if ringed:
                axis.scatter([a for a, _ in ringed], [b for _, b in ringed], s=30, facecolors="none",
                             edgecolors=INK, linewidths=0.8)
        axis.text(x, 15.5, f"{len(mine)}" if mine else "none", ha="center", va="top", fontsize=8.5,
                  color=SOFT_INK)
    axis.axhline(99, color=INK, linewidth=0.9)
    axis.axhline(25, color=SOFT_INK, linewidth=0.9, linestyle=(0, (4, 3)))
    axis.set_xlim(-0.6, len(ratios) - 0.4)
    axis.set_ylim(12, 106)
    axis.set_yticks([25, 50, 75, 100])
    axis.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    axis.tick_params(length=0, labelsize=9.5, colors=SOFT_INK)
    axis.grid(axis="y", color=LINE, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10)
    if first:
        axis.set_ylabel("facts known", fontsize=10, color=SOFT_INK)
        axis.text(-0.55, 99.8, "99: learned", ha="left", va="bottom", fontsize=8.5, color=INK)
        axis.text(-0.55, 25.8, "25: a guess", ha="left", va="bottom", fontsize=8.5, color=SOFT_INK)
        axis.text(-0.62, 15.5, "runs", ha="right", va="top", fontsize=8.5, color=SOFT_INK)
    else:
        axis.set_yticklabels([])


def draw(out):
    ended, training = read()
    ratios = sorted({r["ratio"] for r in ended + training})
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), facecolor=SURFACE, gridspec_kw={"wspace": 0.04})
    panel(axes[0], ended, ratios, f"The {len(ended):,} runs that have ended, at their final score", True)
    panel(axes[1], training, ratios, f"The {len(training):,} runs still training, at their latest score", False)
    figure.supxlabel("bits per creature for each column of the word table", fontsize=10, color=SOFT_INK, y=0.02)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=colour, label=f"{creatures:,} creatures")
               for creatures, colour in CREATURES.items()]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="none",
                              markeredgecolor=INK, label="ringed: ended at the limit of 640,000 steps"))
    figure.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.045, 0.925),
                  columnspacing=1.3, handletextpad=0.2)
    figure.suptitle("Every run of the counted stage, by how many bits a creature carries per table column",
                    x=0.05, y=0.985, ha="left", fontsize=13.5, color=INK)
    figure.text(0.985, 0.985, f"{time.strftime('%H:%M')} Pacific on {time.strftime('%d %B')}", ha="right", va="top",
                fontsize=9, color=SOFT_INK)
    figure.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.115)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return len(ended), len(training)


if __name__ == "__main__":
    out = ROOT / "logs" / "v3" / "all_runs.png"
    print(*draw(out), out)

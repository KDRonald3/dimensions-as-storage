"""Draw the measured training speed of each kind of card, by the size of the model.

    .venv/bin/python -m v3.speed_picture            # writes logs/v3/card_speeds.png

The launcher measures a speed every time a group finishes a round on a machine (member-steps a
second for the whole card: one step of one model counts as one). This picture takes, for each
kind of card and each number of table columns, the middle of those measurements. It only reads
the launcher's state file.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, SOFT_INK, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
# one colour per kind of card, the same in this order everywhere (fastest first)
CARDS = {"G4": "#2a78d6", "A100": "#eb6834", "L4": "#1baf7a", "T4": "#eda100"}
COLUMNS = (8, 16, 32, 64)


def read(stage="counted"):
    state = json.loads((ROOT / "logs" / "v3" / f"fleet_{stage}.json").read_text())
    found = {}
    for card, rates in state.get("speeds", {}).items():
        for shape, rate in rates.items():
            found.setdefault((card, int(shape.split("x")[0])), []).append(rate)
    return found


def draw(out):
    found = read()
    figure, axis = plt.subplots(figsize=(9.2, 4.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    width = 0.2
    for place, card in enumerate(CARDS):
        for x, columns in enumerate(COLUMNS):
            rates = found.get((card, columns), [])
            if not rates:
                continue
            middle = statistics.median(rates)
            position = x + (place - 1.5) * (width + 0.015)
            axis.bar(position, middle, width=width, color=CARDS[card])
            axis.text(position, middle + 180, f"{middle:,.0f}", ha="center", va="bottom", fontsize=8.5,
                      color=INK)
            axis.text(position, -420, f"{len(rates)}", ha="center", va="top", fontsize=7.5, color=SOFT_INK)
    for place, card in enumerate(CARDS):  # the key, in the order of the bars
        axis.bar([-9], [0], color=CARDS[card], label=card)
    axis.legend(loc="upper right", ncol=4, frameon=False, fontsize=9.5, handlelength=1.1,
                columnspacing=1.4)
    axis.set_xlim(-0.55, len(COLUMNS) - 0.45)
    axis.set_ylim(-1250, 15500)
    axis.set_xticks(range(len(COLUMNS)), [f"{c} table columns" for c in COLUMNS])
    axis.set_yticks([0, 5000, 10000, 15000], ["0", "5,000", "10,000", "15,000"])
    axis.tick_params(length=0, labelsize=9.5, colors=SOFT_INK)
    axis.xaxis.set_tick_params(pad=18)
    axis.grid(axis="y", color=LINE, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.axhline(0, color=LINE, linewidth=1)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.set_ylabel("model-steps trained a second, whole card", fontsize=9.5, color=SOFT_INK)
    figure.suptitle("Measured training speed of each kind of card", x=0.095, y=0.975, ha="left",
                    fontsize=13.5, color=INK)
    figure.subplots_adjust(left=0.095, right=0.985, top=0.88, bottom=0.14)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    out = ROOT / "logs" / "v3" / "card_speeds.png"
    draw(out)
    print(out)

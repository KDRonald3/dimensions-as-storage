"""Draw the size-matched pairs of the bridging stage, one row per pair.

    .venv/bin/python -m v3.matched_pairs_picture [OUT.png]     # default logs/v3/matched_pairs.png

Each row is ONE number of trained numbers and two models that both have it: the model with fewer
table columns and a very wide body (a bridging run), and the model with twice the columns and a
body of 4 (a counted run). So "the same size" is the row itself, and nothing has to be read off a
size axis. Every repeat is a dot; the number printed is each model's best repeat. Only the pairs that are not both at 100 are drawn; the caption
of whoever shows the picture names the rest. The wording is for outside readers (dimensions, MLP,
parameters), because the picture is on the public results page. It only reads.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v3 import grid

ROOT = Path(__file__).resolve().parent.parent
INK, SOFT_INK, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
FEWER, MORE = "#e69f00", "#1c5cab"  # parameters spent on a wide MLP / on dimensions


def runs(stage, columns, body, kinds):
    found = []
    for seed in grid.SEEDS:
        name = f"d{columns}_c{grid.BRIDGE_CREATURES}_b{body}_{grid.BRIDGE_USAGE}_k{kinds}_s{seed}"
        found.append(json.loads((ROOT / "runs_v3" / stage / name / "summary.json").read_text()))
    return found


def read():
    """[(kinds, columns, body, size, facts of the narrower model, facts of the wider model)] for
    the pairs that are not both at 100, most bits first."""
    pairs = []
    for kinds in reversed(grid.KINDS):
        for columns in grid.COLUMNS[:-1]:
            body = grid.BRIDGE_BODIES[(kinds, columns)]
            narrow, wide = runs("bridge", columns, body, kinds), runs("counted", 2 * columns, 4, kinds)
            fewer = sorted(r["final"]["facts"] for r in narrow)
            more = sorted(r["final"]["facts"] for r in wide)
            if round(statistics.median(fewer)) == 100 and round(statistics.median(more)) == 100:
                continue
            pairs.append((kinds, columns, body, narrow[0]["parameters"]["trained"], fewer, more))
    return pairs


def draw(pairs, out):
    row_height, header_height = 1.0, 0.8
    places, y = [], 0.0  # (kind of line, y, content)
    last_kinds = None
    for pair in pairs:
        if pair[0] != last_kinds:
            y += header_height
            places.append(("header", y, pair[0]))
            last_kinds = pair[0]
        y += row_height
        places.append(("row", y, pair))
    figure, axis = plt.subplots(figsize=(10.0, 0.52 * y + 1.9), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.set_xlim(25, 101.5)
    axis.set_ylim(y + 0.6, 0.1)
    for tick in (25, 50, 75, 100):
        axis.axvline(tick, color=LINE, linewidth=0.7, zorder=0)
    left = axis.get_yaxis_transform()  # x in axes units, y in data units
    for kind, at, content in places:
        if kind == "header":
            axis.text(-0.62, at + 0.12, f"{2 * content} bits of facts per creature", transform=left,
                      fontsize=11, color=INK, fontweight="bold", va="center")
            continue
        kinds, columns, body, size, fewer, more = content
        axis.text(-0.62, at - 0.2, f"{round(size, -2):,} parameters each", transform=left, fontsize=10.5,
                  color=INK, va="center")
        axis.text(-0.62, at + 0.22, f"{columns} dims, MLP {body}   vs   {2 * columns} dims, MLP 4",
                  transform=left, fontsize=9, color=SOFT_INK, va="center")
        # The number shown for a model is its best repeat (the owner, 24 November 2026: "use the
        # largest scores in the graph, unless it does not support the claims"; it does: see the
        # check printed at the end). The best dot is outlined; the others stay visible.
        middle_fewer, middle_more = max(fewer), max(more)
        axis.plot([middle_fewer, middle_more], [at, at], color="#b9b8b2", linewidth=2.2, zorder=1)
        for facts, colour in ((fewer, FEWER), (more, MORE)):
            axis.plot(facts[:-1], [at] * (len(facts) - 1), "o", color=colour, markersize=8, zorder=3,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, alpha=0.55)
            axis.plot(facts[-1:], [at], "o", color=colour, markersize=10, zorder=4,
                      markeredgecolor=INK, markeredgewidth=1.3)
        axis.text(min(fewer) - 1.4, at, f"{middle_fewer:.1f}", ha="right", va="center", fontsize=10.5, color=INK)
        axis.text(max(more) + 2.1, at, f"{middle_more:.1f}", ha="left", va="center", fontsize=10.5, color=INK)
        axis.text(1.115, at, f"+{round(middle_more, 1) - round(middle_fewer, 1):.1f}", transform=left, ha="left", va="center",
                  fontsize=11, color=INK, fontweight="bold")
    axis.text(1.115, 0.35, "gain", transform=left, ha="left", va="center", fontsize=9, color=SOFT_INK)
    axis.set_yticks([])
    axis.set_xticks([25, 50, 75, 100], ["25\n(guessing)", "50", "75", "100"])
    axis.tick_params(length=0, labelsize=10, colors=SOFT_INK)
    axis.set_xlabel("facts known, out of 100", fontsize=10, color=SOFT_INK)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.suptitle("Same number of parameters: the model with more dimensions knew more",
                    x=0.03, y=0.985, ha="left", fontsize=14.5, color=INK)
    figure.text(0.03, 0.935, "Each row is two models with the same parameter count (within 0.07%).\n"
                "Every dot is one training run (3 seeds each); the number is the best of the three (outlined).",
                ha="left", va="center", fontsize=9.5, color=SOFT_INK, linespacing=1.4)
    key_y = 0.885
    figure.text(0.03, key_y, "●", color=FEWER, fontsize=13, va="center")
    figure.text(0.05, key_y, "half the dimensions, parameters spent on a wide MLP", fontsize=10, color=INK, va="center")
    figure.text(0.50, key_y, "●", color=MORE, fontsize=13, va="center")
    figure.text(0.52, key_y, "twice the dimensions, MLP of 4 neurons", fontsize=10, color=INK, va="center")
    top = 1 - 1.3 / figure.get_figheight()
    figure.subplots_adjust(left=0.37, right=0.86, top=top, bottom=0.75 / figure.get_figheight())
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "v3" / "matched_pairs.png"
    found = read()
    draw(found, target)
    print(len(found), "pairs drawn ->", target)
    for kinds, columns, body, size, fewer, more in found:
        print(f"{2 * kinds} bits, {columns} vs {2 * columns} dims: best {max(fewer):.1f} vs {max(more):.1f} "
              f"(+{round(max(more), 1) - round(max(fewer), 1):.1f}); middle {statistics.median(fewer):.1f} vs "
              f"{statistics.median(more):.1f}")

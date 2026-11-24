"""Draw how many of the runs that have ended fall short of 100 on facts or on questions.

    .venv/bin/python -m v3.shortfall_picture            # writes logs/v3/shortfall.png

Two bars over the same runs. The upper one counts a score as short of 100 when it misses even one
answer; the lower one only when it prints as 99 or less (below 99.5). "Questions" is the score
on questions about one creature. It only reads.
"""
import glob
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, SOFT_INK, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
PARTS = [("facts and questions both at 100", "#2a78d6"), ("facts at 100, questions short", "#f8a56b"),
         ("facts short", "#d9480f")]


def read(stage="counted"):
    rows = []
    for path in glob.glob(str(ROOT / "runs_v3" / stage / "*" / "summary.json")):
        final = json.loads(Path(path).read_text())["final"]
        rows.append((final["facts"], final["questions_one"]))
    return rows


def split(rows, bar):
    """(both at the bar, facts at the bar but questions under it, facts under it)."""
    facts_short = sum(1 for facts, _ in rows if facts < bar)
    questions_short = sum(1 for facts, questions in rows if facts >= bar and questions < bar)
    return [len(rows) - facts_short - questions_short, questions_short, facts_short]


def draw(out):
    rows = read()
    total = len(rows)
    figure, axis = plt.subplots(figsize=(10.4, 3.3), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    bars = [("short of 100 by any amount,\neven one answer", 100.0), ("prints as 99 or less\n(below 99.5)", 99.5)]
    for row, (label, bar) in enumerate(bars):
        start = 0.0
        for count, (_, colour) in zip(split(rows, bar), PARTS):
            share = 100 * count / total
            axis.barh(row, share, left=start, height=0.56, color=colour, edgecolor=SURFACE, linewidth=2)
            if share >= 4:
                axis.text(start + share / 2, row, f"{share:.1f} %\n{count:,} runs", ha="center", va="center",
                          fontsize=9.5, color="white" if colour != "#f8a56b" else INK, linespacing=1.25)
            start += share
        axis.text(-1.2, row, label, ha="right", va="center", fontsize=9.5, color=INK, linespacing=1.25)
    axis.set_xlim(0, 100); axis.set_ylim(1.5, -0.62)
    axis.set_xticks([]); axis.set_yticks([])
    for side in axis.spines.values():
        side.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour, label=label) for label, colour in PARTS]
    axis.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(-0.01, 0.97),
                columnspacing=1.6, handlelength=1.0, handleheight=1.0)
    figure.suptitle(f"The {total:,} runs that have ended: how many fall short of 100", x=0.02, y=0.975, ha="left",
                    fontsize=13.5, color=INK)
    figure.text(0.985, 0.975, f"{time.strftime('%H:%M')} Pacific on {time.strftime('%d %B')}", ha="right", va="top",
                fontsize=9, color=SOFT_INK)
    figure.subplots_adjust(left=0.215, right=0.985, top=0.74, bottom=0.03)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return total


if __name__ == "__main__":
    out = ROOT / "logs" / "v3" / "shortfall.png"
    print(draw(out), out)

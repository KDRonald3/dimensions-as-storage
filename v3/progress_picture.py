"""Draw where a stage of the third experiment stands, as one picture.

Top: one square per group of 24 runs (same table columns, body width and bits per creature; the
24 are 4 creature counts x 2 word uses x 3 seeds). The darker the square, the more of its runs
have ended. A square carries the name of the card that trains it now or that trained it (two
names when it moved from one kind of card to another).
Bottom left: how many runs had ended at each moment, with the launcher's own estimate of the end.
Bottom right: how fast each machine trains, measured, as a multiple of a T4 (with the number of
groups it still holds; a fast machine holds heavier groups, so their number says little).

    .venv/bin/python -m v3.progress_picture            # writes logs/v3/progress.png

It only reads: the launcher's state file, its log and the results already downloaded.
"""
import datetime
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from v3 import fleet_plan, grid

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "v3" / "fleet.log"
INK, SOFT_INK, SURFACE, EMPTY, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe7", "#d9d8d2"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # what has happened; what is only estimated
# one hue, light to dark: the share of a group's runs that have ended
BLUES = LinearSegmentedColormap.from_list("ended", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
CARD_ORDER = ["H100", "G4", "A100", "L4", "T4"]  # fastest first


def card_of(machine):
    """The card in a machine's name: air-a100-090039-0 is an A100."""
    return machine.split("-")[1].upper()


def read_log(stage="counted"):
    """From the launcher's log: which cards each group was given to, in order (lines like
    "machine air-g4-090328-0: given groups [115]"); how many runs had ended at each look; and
    the launcher's latest estimate of the hours left."""
    given, ended, hours_left = {}, [], None
    for line in LOG.read_text().splitlines() if LOG.exists() else []:
        found = re.search(r"machine (\S+): given groups \[([\d, ]*)\]", line)
        if found:
            for number in found.group(2).split(","):
                if number.strip():
                    given.setdefault(int(number), []).append(card_of(found.group(1)))
        found = re.match(r"(\S+ \S+)  stage " + stage + r": (\d+) of \d+ runs here", line)  # this stage's lines only
        if found:
            moment = datetime.datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S")
            ended.append((moment, int(found.group(2))))
            estimate = re.search(r"should end in about ([\d.]+) h", line)
            hours_left = (moment, float(estimate.group(1))) if estimate else hours_left
    return given, ended, hours_left


def read(stage):
    """For every group: its settings, how many of its runs have ended, and the card (or cards)
    that train or trained it. Also the launcher's state."""
    from v3.orchestrate import stage_groups
    groups = stage_groups(stage)  # (24 runs a group in the counted stage, 8 in a rate check)
    state_file = ROOT / "logs" / "v3" / f"fleet_{stage}.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"machines": {}, "groups": {}}
    held = {}
    for record in state["groups"].values():
        if record.get("machine"):
            held[record["machine"]] = held.get(record["machine"], 0) + 1
    given, ended_at, hours_left = read_log(stage)
    cells = []
    for index, members in enumerate(groups):
        ended = sum((ROOT / "runs_v3" / stage / m["name"] / "summary.json").exists() for m in members)
        record = state["groups"].get(str(index), {})
        machine = record.get("machine")
        # a machine with a long queue is only training the groups that have reached a look
        training = machine and ended < len(members) and (held[machine] <= 4 or record.get("reached"))
        if training:
            card = card_of(machine)
        elif ended:
            # Finished, or partly finished and waiting for its next round: the cards that had it.
            # The first machine held every group at the start without training most of them, so
            # for a group that was handed on, the first holder is left out. Since 10:04 on
            # 23 November the launcher itself notes the machines that trained a group
            # ("compiled_on"); where it has, that is exact.
            had = given.get(index, [])
            had = had[1:] if len(had) > 1 else had
            noted = [card_of(m) for m in record.get("compiled_on", [])]
            card = "+".join(dict.fromkeys((noted or had)[-2:])) or None
        else:
            card = None
        first = members[0]
        cells.append({"columns": first["columns"], "body": first["body"], "kinds": first["kinds"],
                      "runs": len(members), "ended": ended, "card": card})
    return cells, state, held, ended_at, hours_left


def draw_groups(axes, cells):
    """The four panels of squares, one panel per number of table columns."""
    bodies, kinds = sorted(grid.BODIES), sorted(grid.KINDS)
    for axis, columns in zip(axes, grid.COLUMNS):
        axis.set_facecolor(SURFACE)
        for cell in (c for c in cells if c["columns"] == columns):
            x, y = kinds.index(cell["kinds"]), bodies.index(cell["body"])
            share = cell["ended"] / cell["runs"]
            colour = EMPTY if cell["ended"] == 0 else BLUES(share)
            axis.add_patch(plt.Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, color=colour, lw=0))
            ink = "white" if share > 0.55 else INK
            if cell["ended"] or cell["card"]:
                axis.text(x + 0.5, y + (0.36 if cell["card"] else 0.5), str(cell["ended"]),
                          ha="center", va="center", fontsize=10, color=ink)
            if cell["card"]:
                # the card's name in the same ink as the number, so that it reads on any shade
                axis.text(x + 0.5, y + 0.73, cell["card"], ha="center", va="center", fontsize=7.5,
                          color=ink if cell["ended"] else SOFT_INK)
        axis.set_xlim(0, len(kinds)); axis.set_ylim(len(bodies), 0)
        axis.set_xticks([i + 0.5 for i in range(len(kinds))], [str(2 * k) for k in kinds])
        axis.set_yticks([i + 0.5 for i in range(len(bodies))], [str(b) for b in bodies])
        axis.tick_params(length=0, labelsize=9, colors=SOFT_INK)
        axis.set_title(f"{columns} table columns", fontsize=11, color=INK, pad=8)
        axis.set_xlabel("bits per creature", fontsize=9, color=SOFT_INK)
        for side in axis.spines.values():
            side.set_visible(False)
        axis.set_aspect("equal")
    axes[0].set_ylabel("body width", fontsize=9, color=SOFT_INK)
    for axis in axes[1:]:
        axis.set_yticklabels([])


def draw_time(axis, ended_at, hours_left, total):
    """Runs ended at each moment (what happened), and the launcher's estimate of the end."""
    axis.set_facecolor(SURFACE)
    if ended_at:
        axis.plot([m for m, _ in ended_at], [n for _, n in ended_at], color=BLUE, linewidth=2)
        last_moment, last_count = ended_at[-1]
        axis.plot([last_moment], [last_count], "o", color=BLUE, markersize=6,
                  markeredgecolor=SURFACE, markeredgewidth=1.5)
        axis.annotate(f"{last_count:,} ended at {last_moment:%H:%M}", (last_moment, last_count),
                      xytext=(-8, 6), textcoords="offset points", ha="right", fontsize=9, color=INK)
        if hours_left:
            end = hours_left[0] + datetime.timedelta(hours=hours_left[1])
            # Not a forecast of the path: only "about here", by the launcher's own arithmetic.
            axis.plot([last_moment, end], [last_count, total], color=ORANGE, linewidth=1.6,
                      linestyle=(0, (4, 3)))
            axis.plot([end], [total], "D", color=ORANGE, markersize=6, markeredgecolor=SURFACE,
                      markeredgewidth=1.5)
            # (well below the dashed line, so that the words do not sit on it)
            axis.annotate(f"the launcher's estimate\nof the end: about {end:%H:%M}", (end, total),
                          xytext=(0, -70), textcoords="offset points", ha="right", va="top",
                          fontsize=9, color=INK)
    axis.axhline(total, color=LINE, linewidth=1)
    axis.set_ylim(0, total * 1.08)
    axis.set_yticks([total * i // 4 for i in range(5)], [f"{total * i // 4:,}" for i in range(5)])
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.xaxis.set_major_locator(mdates.HourLocator())
    axis.tick_params(length=0, labelsize=9, colors=SOFT_INK)
    axis.grid(axis="y", color=LINE, linewidth=0.6)
    axis.set_title("Runs that have ended, by time of day (Pacific)", fontsize=11, color=INK,
                   loc="left", pad=8)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(LINE)


def speed_factor(rates):
    """How fast a machine (or a card type) is, from its measured groups: each group's speed
    against the same yardstick for its number of columns, and of those the middle one. Groups
    of different sizes train at different speeds, so raw speeds cannot be compared directly."""
    ratios = sorted(rate / fleet_plan.a100_guess(shape) for shape, rate in rates.items())
    return ratios[len(ratios) // 2] if ratios else None


def draw_machines(axis, state, held):
    """One bar per machine: its measured training speed as a multiple of a T4's."""
    axis.set_facecolor(SURFACE)
    machines = sorted(state["machines"].items(),
                      key=lambda item: (CARD_ORDER.index(item[1].get("card", "T4"))
                                        if item[1].get("card") in CARD_ORDER else 9, item[0]))
    by_card = {card: speed_factor(rates) for card, rates in state.get("speeds", {}).items()}
    unit = by_card.get("T4") or 0.22 * (by_card.get("A100") or 1.0)  # a T4 = 1
    speeds = []
    for name, machine in machines:
        factor = speed_factor(state.get("machine_speeds", {}).get(name, {})) or by_card.get(machine.get("card"))
        speeds.append(factor / unit if factor else 0.0)
    longest = max(speeds + [1.0])
    for row, ((name, machine), speed) in enumerate(zip(machines, speeds)):
        axis.barh(row, speed, height=0.62, color=BLUE)
        groups = held.get(name, 0)
        label = f"{speed:.1f} x an average T4" if speed else "not measured yet"
        axis.text(speed + longest * 0.015, row, f"{label}  ({groups} group{'s' if groups != 1 else ''})",
                  va="center", fontsize=8.5, color=INK)
    axis.set_yticks(range(len(machines)), [m.get("card", "?") for _, m in machines])
    axis.set_ylim(len(machines) - 0.4, -0.6)
    axis.set_xlim(0, longest * 2.25)
    axis.set_xticks([])
    axis.tick_params(length=0, labelsize=9.5, colors=INK)
    axis.set_title("How fast each machine trains (measured)", fontsize=11, color=INK, loc="left",
                   pad=8)
    for side in axis.spines.values():
        side.set_visible(False)


def draw(stage, out):
    cells, state, held, ended_at, hours_left = read(stage)
    total, ended = sum(c["runs"] for c in cells), sum(c["ended"] for c in cells)
    cards = [m.get("card", "?") for m in state["machines"].values()]
    figure = plt.figure(figsize=(11.5, 8.1), facecolor=SURFACE)
    top = figure.add_gridspec(1, len(grid.COLUMNS), left=0.045, right=0.99, top=0.865, bottom=0.44,
                              wspace=0.06)
    bottom = figure.add_gridspec(1, 2, left=0.06, right=0.985, top=0.33, bottom=0.055,
                                 wspace=0.16, width_ratios=[1.7, 1])
    draw_groups([figure.add_subplot(top[0, i]) for i in range(len(grid.COLUMNS))], cells)
    draw_time(figure.add_subplot(bottom[0, 0]), ended_at, hours_left, total)
    draw_machines(figure.add_subplot(bottom[0, 1]), state, held)

    machines = ", ".join(f"{cards.count(c)} {c}" for c in CARD_ORDER if c in cards) or "none"
    names = {"counted": "the counted stage", "rate_check_third": "the follow-up at a third of the learning rate",
             "rate_check_tenth": "the follow-up at a tenth of the learning rate"}
    figure.suptitle(f"{ended:,} of {total:,} runs of {names.get(stage, stage)} have ended",
                    x=0.045, y=0.985, ha="left", fontsize=14, color=INK)
    figure.text(0.045, 0.932, f"{time.strftime('%H:%M')} Pacific on {time.strftime('%d %B')}.  "
                f"Machines training: {machines}.", ha="left", fontsize=9.5, color=SOFT_INK)
    # the key of the squares
    bar = figure.add_axes([0.62, 0.936, 0.2, 0.014])
    size = max(c["runs"] for c in cells)
    bar.imshow([[i / 99 for i in range(100)]], aspect="auto", cmap=BLUES, extent=(0, size, 0, 1))
    bar.set_yticks([]); bar.set_xticks([1, size / 2, size], ["1", str(size // 2), str(size)])
    bar.tick_params(length=0, labelsize=8, colors=SOFT_INK, pad=2)
    for side in bar.spines.values():
        side.set_visible(False)
    figure.text(0.615, 0.943, "none", ha="right", va="center", fontsize=8, color=SOFT_INK)
    figure.add_artist(plt.Rectangle((0.553, 0.936), 0.022, 0.014, color=EMPTY, lw=0,
                                    transform=figure.transFigure))
    figure.text(0.83, 0.943, f"of a group's {size} runs ended", ha="left", va="center", fontsize=8,
                color=SOFT_INK)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return ended, total


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "counted"
    out = ROOT / "logs" / "v3" / ("progress.png" if stage == "counted" else f"progress_{stage}.png")
    print(*draw(stage, out), out)

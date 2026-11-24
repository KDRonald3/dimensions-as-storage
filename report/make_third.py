"""Builds the charts of the third report (report/third.tex): what was found after the first report
and is in neither that report nor the addendum, shown as pictures.

    .venv/bin/python report/make_third.py

It reads the saved runs (runs/, runs_long/, runs_more/), and for runs that are still training
their logs in logs/complete/, so the charts can be redrawn while the last runs go on. It writes
report/figures/t_*.png and report/third_facts.json (every number the charts show).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

import make_assets as M  # noqa: E402  (also sets the charts up without titles, cropped)
import make_addendum as A  # noqa: E402
import report as R  # noqa: E402

# Both modules above wrap R.save_figure to cut off the empty title strip. Keep exactly one wrap.
R.save_figure = M.save_figure_cropped

OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
WIDTHS = [8, 16, 32]
VOCABULARIES = [16, 64, 256, 1024]
SEASON_BLIND = 37.2  # the score of a model that takes no notice of the season


# ------------------------------------------------------------------------------------------------
# Reading runs, finished or still training
# ------------------------------------------------------------------------------------------------


def summary_of(folder, name):
    path = ROOT / folder / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def points_from_train_log(path):
    """Gradings printed by train.py while a run is still training: [(sets, facts, questions)]."""
    points = []
    if path.exists():
        for line in path.read_text().splitlines():
            words = line.replace(",", "").split()
            if len(words) == 10 and "/" in words[-1] and words[0].isdigit():
                points.append((int(words[0]), float(words[7]), float(words[8])))
    return points


def points_from_more_log(path):
    """Gradings printed by more_training.py: [(extra sets, facts, questions)]."""
    points = []
    if path.exists():
        for line in path.read_text().splitlines():
            words = line.replace(",", "").split()
            if len(words) == 6 and words[0].isdigit():
                points.append((int(words[0]), float(words[1]), float(words[2])))
    return points


def new_run(name):
    """A run made from the start (by train.py), finished or not. None if it has not begun."""
    for folder in ("runs", "runs_long"):
        summary = summary_of(folder, name)
        if summary:
            points = [(p["sets"], p["facts_known"], p["questions_right"]) for p in summary["curve"]]
            state = {"solved": "solved", "plateau": "flat", "fixed": "set length"}.get(
                summary["stop_reason"], "not finished")
            return {"points": points, "state": state, "joined_at": None, "folder": folder}
    points = points_from_train_log(ROOT / "logs" / "complete" / f"{name}.log")
    if points:
        return {"points": points, "state": "training", "joined_at": None, "folder": None}
    return None


def continued(run, name):
    """Add the extra training of a run that was picked up again (more_training.py --until
    statements): its points follow the original ones, and joined_at marks where."""
    if run is None:
        return None
    before = run["points"][-1][0]
    summary = summary_of("runs_more", f"{name}_more")
    if summary and summary.get("until") == "statements":
        more = [(p["extra_sets"], p["facts_known"], p["questions_right"]) for p in summary["curve"]]
        state = "solved" if summary["stop_reason"] == "statements solved" else "not finished"
    else:
        more = points_from_more_log(ROOT / "logs" / "complete" / f"{name}.log")
        state = "training" if more else run["state"]
    if not more:
        return run
    # After a pick-up the first gradings come one set apart and jump about; for the picture keep
    # one every doubling (16, 32, 64, ... extra sets) and the last one.
    kept = [point for point in more if point[0] in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)]
    if more[-1] not in kept:
        kept.append(more[-1])
    points = run["points"] + [(before + extra, facts, questions) for extra, facts, questions in kept]
    return {"points": points, "state": state, "joined_at": before, "folder": run["folder"]}


def round32():
    """Every run of the 32-bit round as it stands now: a list of dictionaries with animals,
    width, repeat, what it is called, its points, its state and where it was picked up again."""
    runs = []

    def add(animals, width, repeat, called, run, name):
        if run:
            runs.append({"animals": animals, "width": width, "repeat": repeat, "called": called,
                         "name": name, **run})

    for width in WIDTHS:
        for seed in (0, 1):
            name = f"a16_w{width}_equal_x4_s{seed}"
            add(16, width, seed + 1, "first night", new_run(name), name)
            name = f"a64_w{width}_equal_x4_s{seed}_long"
            add(64, width, seed + 1, "new", new_run(name), name)
    # 256 animals
    for name, repeat, called in (("a256_w8_equal_x4_s0", 1, "first run, continued"),
                                 ("a256_w8_equal_x4_s0_long", 1, "run again, continued"),
                                 ("a256_w8_equal_x4_s1_long", 2, "continued")):
        add(256, 8, repeat, called, continued(new_run(name), name), name)
    add(256, 16, 1, "first run, retraced to 512 sets", new_run("a256_w16_equal_x4_s0_f512"),
        "a256_w16_equal_x4_s0_f512")
    add(256, 16, 2, "first night", new_run("a256_w16_equal_x4_s1_long"), "a256_w16_equal_x4_s1_long")
    add(256, 32, 1, "first run, ended as flat", new_run("a256_w32_equal_x4_s0"), "a256_w32_equal_x4_s0")
    add(256, 32, 1, "run again", new_run("a256_w32_equal_x4_s0_long"), "a256_w32_equal_x4_s0_long")
    # 1,024 animals
    for width in (16, 32):
        name = f"a1024_w{width}_equal_x4_s0_long"
        add(1024, width, 1, "continued", continued(new_run(name), name), name)
    add(1024, 8, 1, "new", new_run("a1024_w8_equal_x4_s0_long"), "a1024_w8_equal_x4_s0_long")
    return runs


# ------------------------------------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------------------------------------

END_LOOK = {"solved": "finished", "not finished": "budget", "training": "budget", "flat": "budget",
            "set length": "fixed"}


def end_mark(panel, x, y, color, state):
    """How a line ends: a filled dot if the run learned everything, a hollow dot if its time ran
    out, a hollow triangle if it was still training when stopped by hand, a square if the stopping test ended it."""
    if state == "training":
        panel.plot([x], [y], linestyle="none", marker=">", markersize=R.HOLLOW_DOT + 1,
                   markerfacecolor=R.SURFACE, markeredgecolor=color, markeredgewidth=R.LINE,
                   zorder=3.6, clip_on=False)
    elif state == "flat":
        panel.plot([x], [y], linestyle="none", marker="s", markersize=R.DOT - 2 * R.PX,
                   markerfacecolor=color, markeredgecolor=R.SURFACE, markeredgewidth=R.LINE,
                   zorder=3.6, clip_on=False)
    else:
        R.draw_dot(panel, x, y, color, END_LOOK[state])


def state_keys():
    gray = R.INK_MUTED
    return [R.key_dot(gray, "finished", "learned every statement"),
            R.key_dot(gray, "budget", "time ran out"),
            R.key_dot(gray, "fixed", "stopped after a set length"),
            Line2D([], [], linestyle="none", marker=">", markersize=R.HOLLOW_DOT + 1,
                   markerfacecolor=R.SURFACE, markeredgecolor=gray, markeredgewidth=R.LINE,
                   label="stopped by hand"),
            Line2D([], [], linestyle="none", marker="s", markersize=R.DOT - 3 * R.PX,
                   markerfacecolor=gray, markeredgecolor=gray, label="ended as flat (wrongly)"),
            Line2D([], [], linestyle="none", marker="|", markersize=10, markeredgewidth=1.3,
                   color=R.INK, label="picked up again here")]


def grid_of_panels(n_panels, legend, per_row, width, height, gap_x=0.5):
    """An empty picture with panels of the given size, per_row to a row."""
    saved = (R.PANELS_PER_ROW, R.PANEL_WIDTH, R.WIDE_PANEL, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT)
    R.PANELS_PER_ROW, R.PANEL_WIDTH, R.WIDE_PANEL, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT = (
        per_row, width, width, height, gap_x, 0.3)
    try:
        figure, panels = R.new_figure(n_panels, "", "", legend)
    finally:
        (R.PANELS_PER_ROW, R.PANEL_WIDTH, R.WIDE_PANEL, R.PANEL_HEIGHT, R.GAP_X,
         R.MARGIN_RIGHT) = saved
    return figure, panels, per_row


def label_grid(panel, number, per_row, x_label, y_label):
    panel.set_xlabel(x_label, fontsize=9, color=R.INK_SOFT, labelpad=7)
    if number % per_row == 0:
        panel.set_ylabel(y_label, fontsize=9, color=R.INK_SOFT, labelpad=7)


def chart_round32(runs, path):
    """The 32-bit round: the statement score of every run against the sets it has read, one
    panel for each size of vocabulary."""
    legend = [[R.key_line(R.width_color(width), f"width {width}") for width in WIDTHS],
              state_keys()[:3], state_keys()[3:]]
    figure, panels, per_row = grid_of_panels(4, legend, 2, 3.25, 2.35)
    longest = max(run["points"][-1][0] for run in runs)
    for number, (panel, animals) in enumerate(zip(panels, VOCABULARIES)):
        here = [run for run in runs if run["animals"] == animals]
        for run in here:
            color = R.width_color(run["width"])
            xs = [sets for sets, _, _ in run["points"]]
            ys = [facts for _, facts, _ in run["points"]]
            panel.plot(xs, ys, color=color, linewidth=R.LINE * 0.9, zorder=2.5,
                       solid_capstyle="round", solid_joinstyle="round")
            if run["joined_at"]:
                at = dict(zip(xs, ys))[run["joined_at"]]
                panel.plot([run["joined_at"]], [at], linestyle="none", marker="|", markersize=10,
                           markeredgewidth=1.3, color=R.INK, zorder=3.4)
            end_mark(panel, xs[-1], ys[-1], color, run["state"])
        R.log_axis(panel, 4, longest)
        panel.set_xlim(3.4, longest * 1.4)
        panel.set_xticks([10, 100, 1000])
        panel.set_xticklabels(["10", "100", "1,000"])
        if animals > 16:
            panel.axhline(SEASON_BLIND, color=R.INK_MUTED, linewidth=R.HAIR, linestyle=(0, (2, 2)),
                          zorder=1.2)
        if animals == 64:
            panel.annotate("37: a model that\nignores the season", xy=(longest * 1.3, SEASON_BLIND),
                           xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                           fontsize=8.5, color=R.INK_SOFT, linespacing=1.3)
        missing = [width for width in WIDTHS if not any(run["width"] == width for run in here)]
        title = f"{animals:,} animals"
        if missing:
            title += "   (width " + " and ".join(str(width) for width in missing) + ": not started yet)"
        R.panel_title(panel, title)
        label_grid(panel, number, per_row, "sets of text read (log scale)", "facts known (0 to 100)")
    R.save_figure(figure, path)


def chart_where_they_stand(runs, path):
    """Where every model of the 32-bit round stands now: its statement score and its question
    score, by size of vocabulary. One dot for each run."""
    legend = [[R.key_line(R.width_color(width), f"width {width}") for width in WIDTHS],
              [R.key_dot(R.INK_MUTED, "finished", "run has ended"),
               Line2D([], [], linestyle="none", marker=">", markersize=R.HOLLOW_DOT + 1,
                      markerfacecolor=R.SURFACE, markeredgecolor=R.INK_MUTED,
                      markeredgewidth=R.LINE, label="stopped by hand (score so far)")]]
    figure, panels, per_row = grid_of_panels(2, legend, 2, 3.25, 2.5)
    shown = []
    for number, (panel, which, name) in enumerate(zip(panels, (1, 2), (
            "What they know: statements", "What they can answer: yes/no questions"))):
        which_name = "facts_known" if which == 1 else "questions"
        for place, animals in enumerate(VOCABULARIES):
            for at, width in enumerate(WIDTHS):
                here = [run for run in runs if run["animals"] == animals and run["width"] == width]
                for k, run in enumerate(here):
                    x = place + (at - 1) * 0.24 + (k - (len(here) - 1) / 2) * 0.07
                    y = run["points"][-1][which]
                    color = R.width_color(width)
                    if run["state"] == "training":
                        end_mark(panel, x, y, color, "training")
                    else:
                        R.draw_dot(panel, x, y, color, "finished")
                    shown.append({"name": run["name"], "state": run["state"], which_name: y})
        panel.set_xlim(-0.6, len(VOCABULARIES) - 0.4)
        panel.set_xticks(range(len(VOCABULARIES)))
        panel.set_xticklabels([f"{animals:,}" for animals in VOCABULARIES])
        for place in range(1, len(VOCABULARIES)):
            panel.axvline(place - 0.5, color=R.GRID, linewidth=R.HAIR, zorder=1)
        R.panel_title(panel, name)
        label_grid(panel, number, per_row, "animals in the vocabulary",
                   "score at the end of the run (0 to 100)")
    R.save_figure(figure, path)


def chart_kinds(path):
    """Width 8 on 256 animals: how many facts of each kind each of its runs has right."""
    kinds = ["colors", "foods", "homes", "waking\ntimes"]
    wanted = [("a256_w8_equal_x4_s0", "runs", "repeat 1, first run"),
              ("a256_w8_equal_x4_s0_long", "runs_long", "repeat 1, run again"),
              ("a256_w8_equal_x4_s1_long", "runs_long", "repeat 2")]
    rows, facts = [], {}
    for name, folder, called in wanted:
        before = M.by_kind_of_fact(ROOT / folder / name)["right_by_kind"]
        after_dir = ROOT / "runs_more" / f"{name}_more"
        after = None
        if (after_dir / "summary.json").exists() and (after_dir / "model.pt").exists():
            after = M.by_kind_of_fact(after_dir)["right_by_kind"]
        rows.append((called, before, after))
        facts[name] = {"when_the_night_ended": before, "after_the_extra_training": after}
    both = any(after for _, _, after in rows)
    legend = [[Line2D([], [], linestyle="none", marker="s", markersize=8, markerfacecolor="#b9b8ae",
                      markeredgecolor="none", label="when the night ended")]
              + ([Line2D([], [], linestyle="none", marker="s", markersize=8,
                         markerfacecolor=R.width_color(8), markeredgecolor="none",
                         label="after the extra training")] if both else []),
              [Line2D([], [], color=R.INK_MUTED, linewidth=R.HAIR, linestyle=(0, (2, 2)),
                      label="upper dotted line (53): a model that ignores the season"),
               Line2D([], [], color=R.INK_MUTED, linewidth=R.HAIR, linestyle=(0, (2, 2)),
                      label="lower dotted line (25): pure guessing")]]
    figure, panels, per_row = grid_of_panels(3, legend, 3, 2.05, 2.2, gap_x=0.35)
    for number, (panel, (called, before, after)) in enumerate(zip(panels, rows)):
        for place in range(4):
            if after:
                panel.bar(place - 0.2, 100 * before[place], width=0.36, color="#b9b8ae", zorder=2)
                panel.bar(place + 0.2, 100 * after[place], width=0.36, color=R.width_color(8),
                          zorder=2)
            else:
                panel.bar(place, 100 * before[place], width=0.6, color="#b9b8ae", zorder=2)
        for level in (25, 53):
            panel.axhline(level, color=R.INK_MUTED, linewidth=R.HAIR, linestyle=(0, (2, 2)),
                          zorder=2.4)
        panel.set_ylim(0, 105)
        panel.set_xticks(range(4))
        panel.set_xticklabels(kinds, fontsize=8.5)
        R.panel_title(panel, called)
        label_grid(panel, number, per_row, "", "facts right, out of 100")
    R.save_figure(figure, path)
    return facts


def chart_before_after(path):
    """The question score of every model of the two planned rounds, before the extra training
    and after it: one line for each model, one panel for each width and round."""
    legend = [[R.key_line(A.VOCABULARY_COLORS[a], f"{a:,} animals") for a in VOCABULARIES]]
    figure, panels, per_row = grid_of_panels(6, legend, 3, 2.05, 2.0, gap_x=0.35)
    facts = {}
    order = [(usage, width) for usage in ("equal", "zipf") for width in WIDTHS]
    tables = {usage: A.more_runs(usage) for usage in ("equal", "zipf")}
    for number, (panel, (usage, width)) in enumerate(zip(panels, order)):
        answered = total = 0
        for animals in VOCABULARIES:
            for k, run in enumerate(tables[usage].get((width, animals), [])):
                start, end = run["curve"][0][1], run["curve"][-1][1]
                shift = (VOCABULARIES.index(animals) - 1.5) * 0.045 + (k - 1) * 0.012
                color = A.VOCABULARY_COLORS[animals]
                panel.plot([0 + shift, 1 + shift], [start, end], color=color,
                           linewidth=R.LINE * 0.75, alpha=0.9, zorder=2.5, solid_capstyle="round")
                panel.plot([0 + shift, 1 + shift], [start, end], linestyle="none", marker="o",
                           markersize=4.2, markerfacecolor=color, markeredgecolor=R.SURFACE,
                           markeredgewidth=0.6, zorder=3)
                total += 1
                answered += end >= 90
        facts[f"{usage}_w{width}"] = {"models": total, "answer_after": answered}
        panel.set_xlim(-0.35, 1.35)
        panel.set_xticks([0, 1])
        panel.set_xticklabels(["when the\nexperiment\nstopped", "after 256\nmore sets"], fontsize=8.5)
        round_name = "first round" if usage == "equal" else "unequal use"
        R.panel_title(panel, f"width {width}, {round_name}")
        label_grid(panel, number, per_row, "", "question score (0 to 100)")
    R.save_figure(figure, path)
    return facts


def chart_when(path):
    """When each model learned the statements and when it learned the questions."""
    legend = [[R.key_line(R.width_color(width), f"width {width}") for width in WIDTHS],
              [R.key_dot(R.INK_MUTED, "finished", "questions answered"),
               Line2D([], [], linestyle="none", marker="^", markersize=R.HOLLOW_DOT,
                      markerfacecolor=R.SURFACE, markeredgecolor=R.INK_MUTED,
                      markeredgewidth=R.LINE, label="not answered yet (drawn at the sets read so far)")]]
    figure, panels, per_row = grid_of_panels(2, legend, 2, 3.25, 2.9)
    facts = {}
    for number, (panel, usage) in enumerate(zip(panels, ("equal", "zipf"))):
        seen = {}
        for (width, _), runs in sorted(A.more_runs(usage).items()):
            for run in runs:
                course = A.whole_course(run)
                known = A.first_reached(course, 90, which=2)
                answered = A.first_reached(course, 90, which=1)
                color = R.width_color(width)
                seen[(known, answered)] = seen.get((known, answered), 0) + 1
                known_at = known * (1 + 0.07 * (seen[(known, answered)] - 1))
                if answered:
                    panel.plot([known_at], [answered], linestyle="none", marker="o", markersize=6,
                               markerfacecolor=color, markeredgecolor=R.SURFACE,
                               markeredgewidth=0.8, zorder=3, alpha=0.95)
                else:
                    panel.plot([known_at], [course[-1][0]], linestyle="none", marker="^",
                               markersize=R.HOLLOW_DOT, markerfacecolor=R.SURFACE,
                               markeredgecolor=color, markeredgewidth=R.LINE, zorder=3)
                facts[run["name"]] = {"statements_at": known, "questions_at": answered,
                                      "sets_read": course[-1][0]}
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(8, 2500)
        panel.set_ylim(8, 2500)
        panel.plot([8, 2500], [8, 2500], color=R.AXIS, linewidth=R.HAIR, zorder=1.5)
        panel.annotate("same time", xy=(900, 900), xytext=(4, -12), textcoords="offset points",
                       fontsize=8, color=R.INK_MUTED, rotation=39)
        for axis in (panel.xaxis, panel.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            axis.set_minor_locator(NullLocator())
        panel.set_xticks([10, 100, 1000])
        panel.set_yticks([10, 100, 1000])
        panel.grid(axis="x", color=R.GRID, linewidth=R.HAIR)
        R.panel_title(panel, "first round" if usage == "equal" else "unequal use")
        label_grid(panel, number, per_row, "sets until the statements were known",
                   "sets until the questions were answered")
    R.save_figure(figure, path)
    return facts


def chart_effort(groups, path):
    """Sets read until every statement was known, against the size of the vocabulary: the first
    round and the unequal-use round side by side on the same scale."""
    import statistics

    legend = [[R.key_line(R.width_color(width), f"width {width}") for width in WIDTHS],
              [R.key_dot(R.INK_MUTED, "finished", "one run"),
               R.key_line(R.INK_MUTED, "line: the middle one of the three repeats")]]
    figure, panels, per_row = grid_of_panels(2, legend, 2, 3.25, 2.6)
    facts = {}
    for number, (panel, usage) in enumerate(zip(panels, ("equal", "zipf"))):
        group = {key: run for key, run in groups[(usage, 1)].items() if key[0] in WIDTHS}
        for width in WIDTHS:
            color, middles = R.width_color(width), []
            for place, animals in enumerate(VOCABULARIES):
                done = sorted(run["sets"] for (w, a, _), run in group.items()
                              if w == width and a == animals and run["stop_reason"] == "solved")
                for k, sets in enumerate(done):
                    panel.plot([place + (k - 1) * 0.07], [sets], linestyle="none", marker="o",
                               markersize=5.5, markerfacecolor=color, markeredgecolor=R.SURFACE,
                               markeredgewidth=0.8, zorder=3)
                if done:
                    middles.append((place, statistics.median(done)))
                facts[f"{usage}_w{width}_a{animals}"] = done
            panel.plot([x for x, _ in middles], [y for _, y in middles], color=color,
                       linewidth=R.LINE, zorder=2)
        panel.set_yscale("log")
        panel.set_ylim(10, 2500)
        panel.set_yticks([10, 30, 100, 300, 1000])
        panel.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        panel.yaxis.set_minor_locator(NullLocator())
        panel.set_xlim(-0.4, 3.4)
        panel.set_xticks(range(4))
        panel.set_xticklabels([f"{animals:,}" for animals in VOCABULARIES])
        R.panel_title(panel, "first round (equal use)" if usage == "equal" else "unequal use")
        label_grid(panel, number, per_row, "animals in the vocabulary",
                   "sets read until every statement was known")
    R.save_figure(figure, path)
    return facts


def chart_text(path):
    """How much variety each language has: its different sentences, and how often one set
    repeats each fact."""
    figure, panels, per_row = grid_of_panels(2, [], 2, 3.25, 2.1)
    values = [("different sentences in the language", [20 * a for a in VOCABULARIES]),
              ("times one set shows each fact", [61440 // (4 * a) for a in VOCABULARIES])]
    for number, (panel, (name, numbers)) in enumerate(zip(panels, values)):
        bars = panel.bar(range(4), numbers, width=0.6, color="#2a78d6", zorder=2)
        for bar, value in zip(bars, numbers):
            panel.annotate(f"{value:,}", xy=(bar.get_x() + bar.get_width() / 2, value),
                           xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                           fontsize=8.5, color=R.INK_SOFT)
        panel.set_ylim(0, max(numbers) * 1.15)
        panel.set_yticks([])
        panel.set_xticks(range(4))
        panel.set_xticklabels([f"{a:,}" for a in VOCABULARIES])
        R.panel_title(panel, name)
        label_grid(panel, number, per_row, "animals in the vocabulary", "")
    R.save_figure(figure, path)


def main():
    FIGURES.mkdir(exist_ok=True)
    runs = round32()
    chart_round32(runs, FIGURES / "t_round32.png")
    chart_where_they_stand(runs, FIGURES / "t_stand.png")
    facts = {"round32": [{key: run[key] for key in ("animals", "width", "repeat", "called",
                                                    "name", "state", "joined_at")}
                         | {"sets": run["points"][-1][0],
                            "facts_known": round(run["points"][-1][1], 2),
                            "questions": round(run["points"][-1][2], 2),
                            "curve": [(s, round(f, 1)) for s, f, _ in run["points"]]}
                         for run in runs]}
    facts["kinds"] = chart_kinds(FIGURES / "t_kinds.png")
    facts["before_after"] = chart_before_after(FIGURES / "t_before_after.png")
    facts["when"] = chart_when(FIGURES / "t_when.png")
    chart_text(FIGURES / "t_text.png")
    A.course_chart(A.more_runs("zipf"), FIGURES / "t_course_zipf.png")
    runs_all = R.load_runs(ROOT / "runs")
    groups, _ = R.comparison_groups(runs_all)
    facts["effort"] = chart_effort(groups, FIGURES / "t_effort.png")
    (OUT / "third_facts.json").write_text(json.dumps(facts, indent=1))
    # The time the charts were drawn, for the line under the report's title.
    import datetime
    now = datetime.datetime.now()
    hour = now.strftime("%I:%M").lstrip("0") + (" am" if now.hour < 12 else " pm")
    training = sum(run["state"] == "training" for run in runs)
    (OUT / "third_status.tex").write_text(
        f"\\newcommand{{\\asof}}{{{hour}}}\n\\newcommand{{\\stilltraining}}{{{training}}}\n")
    print("third report: charts drawn;", {state: sum(run["state"] == state for run in runs)
                                         for state in ("solved", "not finished", "training", "flat")})


if __name__ == "__main__":
    main()

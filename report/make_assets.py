"""Builds the tables and figures of the written report (report/report.tex) from the saved runs.

How to run it, from the project folder:

    .venv/bin/python report/make_assets.py

It reads every runs/*/summary.json through report.py, so the report shows exactly what
report.py shows. It writes:

    report/figures/*.png       the charts
    report/tables/*.tex        the tables, ready to be included in the LaTeX text
    report/rule_output.txt     the table of runs and the decision rule, as report.py prints them
    report/facts.json          every number the text quotes, for checking the text against
"""

import contextlib
import io
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

import report as R  # noqa: E402  (the project's report.py)

OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
TABLES = OUT / "tables"

ENDING_MARK = {"finished": "", "budget": "*", "fixed": "$^{\\circ}$"}

# In the written report every chart has a caption below it, so the title and the explanation
# that report.py draws inside the picture are left out here, and the empty strip they leave at
# the top is cut off. That way the charts can be printed larger.
TOP_STRIP = 0.52  # inches of empty space above the legend once title and subtitle are gone
_new_figure = R.new_figure
_save_figure = R.save_figure


def new_figure_without_title(n_panels, title, subtitle, legend_rows, panel_titles=True):
    return _new_figure(n_panels, "", "", legend_rows, panel_titles)


def save_figure_cropped(figure, path):
    import matplotlib.image as image

    _save_figure(figure, path)
    picture = image.imread(path)
    image.imsave(path, picture[int(TOP_STRIP * R.SAVE_DPI):])


R.new_figure = new_figure_without_title
R.save_figure = save_figure_cropped


def ended(run):
    """'solved', 'plateau', 'budget' or 'fixed'."""
    return run["stop_reason"]


def cell(runs, key="facts_known"):
    """The scores of the repeats at one place, as LaTeX: '100' if all the same, else '91--97'.

    A dagger marks a place where at least one repeat ended flat (plateau), a star one where at
    least one repeat was not finished.
    """
    if not runs:
        return "--"
    scores = [run["final"][key] for run in runs]
    low, high = min(scores), max(scores)
    text = f"{round(low)}" if round(low) == round(high) else f"{round(low)}--{round(high)}"
    endings = {ended(run) for run in runs}
    if "plateau" in endings:
        text += "$^{\\dagger}$"
    if "budget" in endings:
        text += "*"
    return text


def runs_at(group, width, animals):
    return [run for (w, a, _), run in sorted(group.items()) if w == width and a == animals]


def score_table(group, path, key="facts_known"):
    widths = sorted({w for w, _, _ in group})
    vocabularies = sorted({a for _, a, _ in group})
    lines = ["\\begin{tabular}{@{}r" + "r" * len(vocabularies) + "@{}}", "\\toprule",
             " & \\multicolumn{" + str(len(vocabularies)) + "}{c}{Animals in the vocabulary} \\\\",
             "Width & " + " & ".join(f"{a:,}".replace(",", "{,}") for a in vocabularies) + " \\\\",
             "\\midrule"]
    for width in widths:
        cells = [cell(runs_at(group, width, animals), key) for animals in vocabularies]
        lines.append(f"{width} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def bits_per_number_in_use(run):
    """How full a model would be if it knew every fact: the bits of its language divided by the
    numbers it actually works with (the body, and the rows of the words that occur in its text).
    Rows of animals it never reads are left out."""
    c, language, parameters = run["config"], run["language"], run["parameters"]
    other_words = language["vocab_size"] - c["table_animals"]
    in_use = parameters["body"] + (other_words + c["animals"]) * c["width"]
    return language["bits_total"] / in_use


def load_table(group, path):
    widths = sorted({w for w, _, _ in group})
    vocabularies = sorted({a for _, a, _ in group})
    lines = ["\\begin{tabular}{@{}r" + "r" * len(vocabularies) + "@{}}", "\\toprule",
             " & \\multicolumn{" + str(len(vocabularies)) + "}{c}{Animals in the vocabulary} \\\\",
             "Width & " + " & ".join(f"{a:,}".replace(",", "{,}") for a in vocabularies) + " \\\\",
             "\\midrule"]
    for width in widths:
        cells = []
        for animals in vocabularies:
            runs = runs_at(group, width, animals)
            cells.append(f"{50 * bits_per_number_in_use(runs[0]):.0f}\\%" if runs else "--")
        lines.append(f"{width} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def effort_table(group, path):
    """Sets of training text until the run was solved: the three repeats, smallest first."""
    widths = sorted({w for w, _, _ in group})
    vocabularies = sorted({a for _, a, _ in group})
    lines = ["\\begin{tabular}{@{}r" + "r" * len(vocabularies) + "@{}}", "\\toprule",
             " & \\multicolumn{" + str(len(vocabularies)) + "}{c}{Animals in the vocabulary} \\\\",
             "Width & " + " & ".join(f"{a:,}".replace(",", "{,}") for a in vocabularies) + " \\\\",
             "\\midrule"]
    for width in widths:
        cells = []
        for animals in vocabularies:
            runs = runs_at(group, width, animals)
            done = sorted(run["sets"] for run in runs if ended(run) == "solved")
            if not runs:
                cells.append("--")
            elif not done:
                cells.append("*")
            else:
                mark = "*" if len(done) < len(runs) else ""
                cells.append(", ".join(f"{sets:,}".replace(",", "{,}") for sets in done) + mark)
        lines.append(f"{width} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def effort_chart(group, path, title, subtitle):
    """Sets of training until a run was solved, against vocabulary size, one line per width."""
    widths = sorted({w for w, _, _ in group})
    vocabularies = sorted({a for _, a, _ in group})
    legend = [[R.key_line(R.width_color(width), f"width {width}") for width in widths],
              [R.key_dot(R.INK_MUTED, "finished", "one run that was solved"),
               R.key_line(R.INK_MUTED, "line: the middle one of the repeats")]]
    figure, panels = R.new_figure(1, title, subtitle, legend, panel_titles=False)
    panel = panels[0]
    panel.set_xscale("log", base=2)
    panel.set_yscale("log")
    panel.set_xlim(vocabularies[0] / 1.6, vocabularies[-1] * 1.6)
    lowest, highest = [], []
    for place, width in enumerate(widths):
        color = R.width_color(width)
        middles = []
        for animals in vocabularies:
            done = [run["sets"] for run in runs_at(group, width, animals) if ended(run) == "solved"]
            for number, steps in enumerate(sorted(done)):
                x = animals * (1 + 0.035 * (number - (len(done) - 1) / 2))
                panel.plot([x], [steps], "o", color=color, markersize=R.DOT * 0.72,
                           markeredgecolor=R.SURFACE, markeredgewidth=1.0, zorder=3)
            if done:
                middles.append((animals, statistics.median(done)))
                lowest.append(min(done))
                highest.append(max(done))
        if middles:
            panel.plot([a for a, _ in middles], [s for _, s in middles], "-", color=color,
                       linewidth=R.LINE, zorder=2)
            panel.annotate(f"width {width}", xy=middles[-1], xytext=(9, 0),
                           textcoords="offset points", ha="left", va="center", fontsize=9,
                           color=R.INK_SOFT, annotation_clip=False)
    panel.set_ylim(min(lowest) / 1.5, max(highest) * 1.5)
    panel.set_xticks(vocabularies)
    panel.set_xticklabels([f"{animals:,}" for animals in vocabularies])
    panel.xaxis.set_minor_locator(NullLocator())
    panel.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5]))
    panel.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    panel.yaxis.set_minor_locator(NullLocator())
    R.label_axes(panel, 0, "animal words in use (log scale)",
                 "sets of training text until solved (log scale)")
    R.save_figure(figure, path)


def probe_numbers(run_dir):
    """Run probe.py on one finished run and read its numbers back from what it prints.

    Returns ({attribute: held-out percent}, facts shared with the nearest row, out of 4) and
    leaves the picture in report/figures/.
    """
    import probe  # the project's probe.py

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        probe.main([str(run_dir), "--results-dir", str(FIGURES)])
    held_out, nearest = {}, None
    for line in printed.getvalue().splitlines():
        words = line.split()
        for name in ("color", "food", "home", "waking time"):
            if line.strip().startswith(name) and "%" in line:
                held_out[name] = float(line.strip()[len(name):].split("%")[0])
        if line.strip().startswith("an animal and its nearest row"):
            nearest = float(words[-1])
    return held_out, nearest


def probe_table(group, path):
    """How well a simple rule reads each fact from the word-table rows, largest vocabulary,
    averaged over the repeats that were solved. One line per width."""
    largest = max(a for _, a, _ in group)
    widths = sorted({w for w, a, _ in group if a == largest})
    names = ["color", "food", "home", "waking time"]
    lines = ["\\begin{tabular}{@{}rrrrrr@{}}", "\\toprule",
             "Width & Color & Food & Home & Waking time & Facts shared with \\\\",
             " & & & & & the nearest row (of 4) \\\\", "\\midrule"]
    numbers = {}
    for width in widths:
        solved = [run for run in runs_at(group, width, largest) if ended(run) == "solved"]
        if not solved:
            continue
        results = [probe_numbers(ROOT / "runs" / run["name"]) for run in solved]
        means = [statistics.mean(held[name] for held, _ in results) for name in names]
        shared = statistics.mean(nearest for _, nearest in results)
        numbers[width] = {"runs": len(solved), **dict(zip(names, means)), "nearest": shared}
        lines.append(f"{width} & " + " & ".join(f"{m:.0f}\\%" for m in means)
                     + f" & {shared:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return numbers


def season_use(run_dir):
    """For a model trained with seasons: does it use the season word?

    Returns a dictionary with
      same_in_all_seasons: the share of animal-and-fact pairs for which the model gives the same
                           answer in every season;
      usual_value:         how often its answer is a value that is right in the most seasons;
      blind_score:         the score of a model that ignores the season and always gives the
                           value that is right in the most seasons.
    """
    import numpy as np
    import train
    from score import model_predictor, most_likely_next

    config, language, model = train.load_run(run_dir)
    probes = language.fact_probes()
    said = most_likely_next(model_predictor(model), probes)
    answer = np.zeros((config.animals, config.seasons, 4), dtype=int)
    truth = np.zeros_like(answer)
    for probe, word in zip(probes, said):
        answer[probe.animal, probe.season, probe.attribute] = word
        truth[probe.animal, probe.season, probe.attribute] = probe.answer
    same = (answer == answer[:, :1, :]).all(axis=1).mean()
    usual, blind = [], []
    for animal in range(config.animals):
        for fact in range(4):
            values, counts = np.unique(truth[animal, :, fact], return_counts=True)
            most = set(values[counts == counts.max()])
            usual.append(np.mean([answer[animal, season, fact] in most
                                  for season in range(config.seasons)]))
            blind.append(counts.max() / config.seasons)
    return {"same_in_all_seasons": float(same), "usual_value": float(np.mean(usual)),
            "blind_score": float((np.mean(blind) - 0.25) / 0.75 * 100)}


def by_kind_of_fact(run_dir):
    """For a model trained with seasons: how many facts of each kind (color, food, home, waking
    time) it has right, and for each kind how often it gives the same value in all four
    seasons. Shows which part of the language a model that stands still has not learned."""
    import numpy as np
    import train
    from score import model_predictor, most_likely_next

    config, language, model = train.load_run(run_dir)
    probes = language.fact_probes()
    said = most_likely_next(model_predictor(model), probes)
    answer = np.zeros((config.animals, config.seasons, 4), dtype=int)
    truth = np.zeros_like(answer)
    for probe, word in zip(probes, said):
        answer[probe.animal, probe.season, probe.attribute] = word
        truth[probe.animal, probe.season, probe.attribute] = probe.answer
    right = answer == truth
    return {"right_by_kind": [round(float(right[:, :, kind].mean()), 4) for kind in range(4)],
            "right_by_season": [round(float(right[:, season, :].mean()), 4)
                                for season in range(config.seasons)],
            "same_in_all_seasons_by_kind": [
                round(float((answer[:, :, kind] == answer[:, :1, kind]).all(axis=1).mean()), 4)
                for kind in range(4)],
            "animals_with_every_fact_right": int(right.all(axis=(1, 2)).sum())}


def season_table(group, path):
    """One line for every run on the largest vocabulary of a round with seasons."""
    largest = max(a for _, a, _ in group)
    lines = ["\\begin{tabular}{@{}rrrr@{}}", "\\toprule",
             "Width & Repeat & Facts known & Same value given \\\\",
             " & & & in all four seasons \\\\", "\\midrule"]
    numbers = []
    for (width, animals, seed), run in sorted(group.items()):
        if animals != largest:
            continue
        use = season_use(ROOT / "runs" / run["name"])
        mark = {"plateau": "$^{\\dagger}$", "budget": "*", "solved": "", "fixed": ""}[ended(run)]
        lines.append(f"{width} & {seed + 1} & {run['final']['facts_known']:.0f}{mark} & "
                     f"{100 * use['same_in_all_seasons']:.0f}\\% \\\\")
        numbers.append({"name": run["name"], "ended": ended(run),
                        "facts_known": run["final"]["facts_known"], **use})
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return numbers


def know_nothing_gap(run):
    """The gap of a made-up model that knows the sentence shapes and how common each value is,
    but nothing about any animal, on the evaluation rows of this run's language."""
    import score
    import train

    known = {field.name for field in __import__("dataclasses").fields(train.RunConfig)}
    config = train.RunConfig(**{k: v for k, v in run["config"].items() if k in known})
    language, _ = train.build(config)
    rows = train.evaluation_rows(language, config)
    return score.score(score.blind_predictor(language), language, rows)["gap"]


BLIND_GAPS = {}  # filled by width4_table: the know-nothing gap on each language, for facts.json


def width4_table(groups, path):
    """Width 4 on every vocabulary: its score and how much avoidable error it leaves."""
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             " & \\multicolumn{4}{c}{Animals in the vocabulary} \\\\",
             " & 16 & 64 & 256 & 1{,}024 \\\\", "\\midrule"]
    numbers = {}
    for usage, label in (("equal", "equal use"), ("zipf", "unequal use")):
        group = groups.get((usage, 1), {})
        four = {key: run for key, run in group.items() if key[0] == 4}
        if not four:
            continue
        vocabularies = [16, 64, 256, 1024]
        scores, gaps, blind = [], [], []
        for animals in vocabularies:
            runs = runs_at(four, 4, animals)
            scores.append(cell(runs) if runs else "--")
            gaps.append(f"{statistics.mean(r['final']['gap'] for r in runs):.2f}" if runs else "--")
            if runs:
                BLIND_GAPS[f"{usage}_{animals}"] = round(know_nothing_gap(runs[0]), 4)
            blind.append(f"{BLIND_GAPS[f'{usage}_{animals}']:.2f}" if runs else "--")
            numbers[f"{usage}_{animals}"] = [
                {"name": r["name"], "ended": ended(r), "facts_known": r["final"]["facts_known"],
                 "gap": r["final"]["gap"], "sets": r["sets"]} for r in runs]
        lines.append(f"\\emph{{With {label}}} & & & & \\\\")
        lines.append("Facts known & " + " & ".join(scores) + " \\\\")
        lines.append("Avoidable error left & " + " & ".join(gaps) + " \\\\")
        lines.append("\\quad the same for a model that knows only the sentence shapes & "
                     + " & ".join(blind) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return numbers


# ---------------------------------------------------------------------------------------------
# The runs on the two large languages of the 32-bit round (256 and 1,024 animals).
#
# "first" runs were made under the agreed stopping test. "later" runs were trained until solved
# or until their time was up, with the "flat" ending switched off (--started-at 101); they are
# saved in runs_long/, apart from the runs that the agreed rule judges. The first run of width 16
# is shown by its repeat of a set length of 512 sets, which followed exactly the same path and
# went on past set 128, where the stopping test had ended the first run as flat.
LARGE_RUNS = [  # (name, folder, animals, width, kind, how the table names it)
    ("a256_w8_equal_x4_s0", "runs", 256, 8, "first", "first run"),
    ("a256_w8_equal_x4_s0_long", "runs_long", 256, 8, "later", "later run"),
    ("a256_w8_equal_x4_s1_long", "runs_long", 256, 8, "later", "later run, second repeat"),
    ("a256_w16_equal_x4_s0_f512", "runs", 256, 16, "first", "first run, trained on"),
    ("a256_w16_equal_x4_s1_long", "runs_long", 256, 16, "later", "later run, second repeat"),
    ("a256_w32_equal_x4_s0", "runs", 256, 32, "first", "first run"),
    ("a256_w32_equal_x4_s0_long", "runs_long", 256, 32, "later", "later run"),
    ("a1024_w16_equal_x4_s0_long", "runs_long", 1024, 16, "later", "later run"),
    ("a1024_w32_equal_x4_s0_long", "runs_long", 1024, 32, "later", "later run"),
]
ENDED_IN_WORDS = {"solved": "solved", "plateau": "ended as flat", "fixed": "set length reached",
                  "budget": "time ran out", "running": "still training"}


def curve_from_log(path):
    """The gradings of a run that is still training, read from the table it prints as it goes."""
    curve = []
    if not path.exists():
        return curve
    for line in path.read_text().splitlines():
        words = line.replace(",", "").split()
        if len(words) == 10 and "/" in words[-1] and words[0].isdigit():
            curve.append({"sets": int(words[0]), "steps": int(words[1]),
                          "minutes": float(words[2]), "gap": float(words[6]),
                          "facts_known": float(words[7])})
    return curve


def flat_by_the_agreed_test(curve):
    """Where the agreed stopping test would have ended a run, had it been switched on: the
    stopping rule of train.py, with the settings of the planned rounds, asked after each
    grading in turn. Returns (ending, sets) for the first grading where it says "solved" or
    "plateau", or (None, None) if it never does."""
    import train

    config = train.RunConfig()
    for last in range(1, len(curve) + 1):
        ending = train.decide_stop(curve[:last], config)
        if ending:
            return ending, curve[last - 1]["sets"]
    return None, None


def read_large(name, folder, animals, width, kind, label):
    """One run on a large language: its gradings and how it ended ('running' if it has not
    ended yet; then the gradings come from its log)."""
    summary = ROOT / folder / name / "summary.json"
    threads = None
    if summary.exists():
        run = json.loads(summary.read_text())
        curve, ending, threads = run["curve"], run["stop_reason"], run["config"]["threads"]
    else:
        curve, ending = curve_from_log(ROOT / "logs" / f"long_{name}.log"), "running"
    if not curve:
        return None
    test_says, test_at = (None, None)
    if summary.exists():
        test_says, test_at = flat_by_the_agreed_test(curve)
    return {"name": name, "folder": folder, "animals": animals, "width": width, "kind": kind,
            "label": label, "ended": ending, "threads": threads, "sets": curve[-1]["sets"],
            "facts_known": curve[-1]["facts_known"],
            "agreed_test_would_say": test_says, "agreed_test_at_set": test_at,
            "bits_known": run["final"]["bits_known"] if summary.exists() else None,
            "curve": [(point["sets"], point["facts_known"]) for point in curve]}


def large_chart(large_runs, path, blind_score, called_flat):
    """Every run on the large languages: score against sets of training (log scale). One panel
    for each language, one line for each run. called_flat = (sets, score) where the stopping
    test ended the first run of width 16 as flat."""
    from matplotlib.lines import Line2D

    dashes = (0, (4, 2.2))
    languages = sorted({run["animals"] for run in large_runs})
    widths = sorted({run["width"] for run in large_runs})
    legend = [[R.key_line(R.width_color(width), f"width {width}") for width in widths],
              [R.key_line(R.INK_MUTED, "first runs ('flat' ending on)"),
               Line2D([], [], color=R.INK_MUTED, linewidth=R.LINE, linestyle=dashes,
                      label="other runs ('flat' ending off)")],
              [R.key_dot(R.INK_MUTED, "finished", "solved"),
               Line2D([], [], linestyle="none", marker="s", markersize=R.DOT - 3 * R.PX,
                      markerfacecolor=R.INK_MUTED, markeredgecolor=R.INK_MUTED,
                      label="ended as flat"),
               R.key_dot(R.INK_MUTED, "fixed", "fixed length reached"),
               R.key_dot(R.INK_MUTED, "budget", "time ran out")]]
    # Smaller panels than report.py draws, so that the lettering stays readable when the
    # picture is fitted to the width of a printed page.
    sizes = (R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT)
    R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT = 2.35, 2.3, 0.45, 0.25
    try:
        figure, panels = R.new_figure(len(languages), "", "", legend)
    finally:
        R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT = sizes
    last = max(run["sets"] for run in large_runs)
    for number, (panel, animals) in enumerate(zip(panels, languages)):
        for run in large_runs:
            if run["animals"] != animals:
                continue
            color = R.width_color(run["width"])
            xs = [sets for sets, _ in run["curve"]]
            ys = [score for _, score in run["curve"]]
            panel.plot(xs, ys, color=color, linewidth=R.LINE, zorder=2.5,
                       linestyle="-" if run["kind"] == "first" else dashes,
                       solid_capstyle="round", solid_joinstyle="round")
            if run["ended"] == "plateau":
                panel.plot([xs[-1]], [ys[-1]], linestyle="none", marker="s",
                           markersize=R.DOT - 2 * R.PX, markerfacecolor=color,
                           markeredgecolor=R.SURFACE, markeredgewidth=R.LINE, zorder=3.6)
            else:
                how = {"solved": "finished", "fixed": "fixed"}.get(run["ended"], "budget")
                R.draw_dot(panel, xs[-1], ys[-1], color, how)
        R.log_axis(panel, 4, last)
        panel.set_xlim(3.4, last * 1.3)
        panel.set_xticks([5, 10, 20, 50, 100, 200, 500])
        if animals == 256:
            panel.axhline(blind_score, color=R.INK_MUTED, linewidth=R.HAIR,
                          linestyle=(0, (2, 2)), zorder=1.2)
            panel.annotate(f"{blind_score:.0f}: a model that\nignores the season",
                           xy=(3.7, blind_score), xytext=(0, 4), textcoords="offset points",
                           ha="left", va="bottom", fontsize=8.5, color=R.INK_SOFT,
                           linespacing=1.3, zorder=1.3)
            for sets, score in called_flat:
                panel.plot([sets], [score], linestyle="none", marker="s",
                           markersize=R.DOT - 2 * R.PX, markerfacecolor=R.width_color(16),
                           markeredgecolor=R.SURFACE, markeredgewidth=R.LINE, zorder=3.6)
        R.panel_title(panel, f"{animals:,} animals")
        R.label_axes(panel, number, "sets of training text (log scale)", "facts known (0 to 100)")
    R.save_figure(figure, path)


LARGE_TABLE = [  # (name, folder, animals, width, how the table names the run, stopping test)
    ("a256_w8_equal_x4_s0", "runs", 256, 8, "repeat 1, first run", "on"),
    ("a256_w8_equal_x4_s0_long", "runs_long", 256, 8, "repeat 1, run again", "off"),
    ("a256_w8_equal_x4_s1_long", "runs_long", 256, 8, "repeat 2", "off"),
    ("a256_w16_equal_x4_s0", "runs", 256, 16, "repeat 1, first run", "on"),
    ("a256_w16_equal_x4_s0_f512", "runs", 256, 16, "repeat 1, run again", "off"),
    ("a256_w16_equal_x4_s1_long", "runs_long", 256, 16, "repeat 2", "off"),
    ("a256_w32_equal_x4_s0", "runs", 256, 32, "repeat 1, first run", "on"),
    ("a256_w32_equal_x4_s0_long", "runs_long", 256, 32, "repeat 1, run again", "off"),
    ("a1024_w16_equal_x4_s0_long", "runs_long", 1024, 16, "repeat 1", "off"),
    ("a1024_w32_equal_x4_s0_long", "runs_long", 1024, 32, "repeat 1", "off"),
]


def large_table(path):
    """Every run on the large languages of the 32-bit round, one line each: which repeat it
    is, whether the stopping test was on, how long it was trained, its last score and how it
    ended. Returns the rows for facts.json."""
    ending = {"solved": "solved", "plateau": "ended as flat", "fixed": "fixed length reached",
              "budget": "time ran out"}
    lines = ["\\begin{tabular}{@{}rrlcrrl@{}}", "\\toprule",
             "Animals & Width & Run & ``Flat'' ending & Sets & Facts known & How it ended \\\\",
             "\\midrule"]
    rows, before = [], None
    for name, folder, animals, width, label, test in LARGE_TABLE:
        run = read_large(name, folder, animals, width, "", label)
        if run is None or run["ended"] == "running":
            continue
        if before is not None and (animals, width) != before:
            lines.append("\\addlinespace[2pt]")
        before = (animals, width)
        lines.append(f"{animals:,}".replace(",", "{,}") + f" & {width} & {label} & {test} & "
                     + f"{run['sets']:,}".replace(",", "{,}")
                     + f" & {run['facts_known']:.1f} & {ending[run['ended']]} \\\\")
        rows.append({"name": name, "animals": animals, "width": width, "run": label,
                     "stopping_test": test, "sets": run["sets"],
                     "facts_known": round(run["facts_known"], 2), "ended": run["ended"]})
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return rows


def probe_strip(source, path):
    """The four panels of a probe picture set side by side, without the lettering of the
    picture (which is too small to read once printed): the report names the panels itself."""
    import matplotlib.image as image
    import numpy as np

    picture = image.imread(source)
    height, width = picture.shape[:2]
    across = [(0.1016, 0.5159), (0.5571, 0.9714)]
    down = [(0.1764, 0.5273), (0.5838, 0.9347)]
    panels = [picture[int(top * height):int(bottom * height), int(left * width):int(right * width)]
              for top, bottom in down for left, right in across]
    tall = min(panel.shape[0] for panel in panels)
    wide = min(panel.shape[1] for panel in panels)
    gap = np.ones((tall, int(0.06 * wide), picture.shape[2]), dtype=picture.dtype)
    gap[..., :3] = picture[2, 2, :3]  # the background color of the picture
    parts = []
    for number, panel in enumerate(panels):
        parts.append(panel[:tall, :wide])
        if number < len(panels) - 1:
            parts.append(gap)
    image.imsave(path, np.concatenate(parts, axis=1))


def fullness_table(path):
    """How full each width would be on 16, 256 and 1,024 animals with 32 bits each, against the
    rough limit of 2 bits per number in use. Counted like bits_per_number_in_use."""
    lines = ["\\begin{tabular}{@{}rrrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{Animals in the vocabulary} \\\\",
             "Width & 16 & 256 & 1{,}024 \\\\", "\\midrule"]
    numbers = {}
    for width in (8, 16, 32):
        summary = ROOT / "runs" / f"a16_w{width}_equal_x4_s0" / "summary.json"
        if not summary.exists():
            continue
        run = json.loads(summary.read_text())
        other_words = run["language"]["vocab_size"] - run["config"]["table_animals"]
        cells = []
        for animals in (16, 256, 1024):
            in_use = run["parameters"]["body"] + (other_words + animals) * width
            share = 50 * (32 * animals) / in_use
            numbers[f"w{width}_a{animals}"] = round(share, 1)
            cells.append(f"{share:.0f}\\%")
        lines.append(f"{width} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return numbers


def large_facts(large_runs):
    """The runs on the large languages for facts.json: every grading, and for a run that has
    ended whether its final model uses the season."""
    rows = []
    for run in large_runs:
        row = {key: run[key] for key in ("name", "folder", "animals", "width", "kind", "label",
                                         "ended", "threads", "sets", "facts_known",
                                         "agreed_test_would_say", "agreed_test_at_set",
                                         "bits_known")}
        row["curve"] = run["curve"]
        if (ROOT / run["folder"] / run["name"] / "model.pt").exists():
            row.update(season_use(ROOT / run["folder"] / run["name"]))
            row.update(by_kind_of_fact(ROOT / run["folder"] / run["name"]))
        rows.append(row)
    return rows


def side_checks():
    """The small checks of the night, for facts.json: the run graded without the steadied copy,
    the runs with a word table of 16 rows, and the long run of width 4."""
    facts = {}
    def brief(folder, name):
        summary = ROOT / folder / name / "summary.json"
        if not summary.exists():
            return None
        run = json.loads(summary.read_text())
        return {"name": name, "ended": run["stop_reason"], "sets": run["sets"],
                "minutes": round(run["minutes"], 1),
                "facts_known": round(run["final"]["facts_known"], 2),
                "gap": round(run["final"]["gap"], 4),
                "curve": [(p["sets"], round(p["facts_known"], 1)) for p in run["curve"]]}
    facts["no_average"] = brief("runs_extra", "a1024_w8_equal_s0_noaverage")
    facts["with_average"] = brief("runs", "a1024_w8_equal_s0")
    facts["table_16"] = [brief("runs", f"a16_w32_equal_t16_s{seed}") for seed in range(3)]
    facts["table_1024"] = [brief("runs", f"a16_w32_equal_s{seed}") for seed in range(3)]
    facts["width4_long"] = brief("runs_long", "a16_w4_equal_s0_long")
    if facts["width4_long"] is None:
        curve = curve_from_log(ROOT / "logs" / "long_a16_w4_equal_s0_long.log")
        if curve:
            facts["width4_long"] = {"name": "a16_w4_equal_s0_long", "ended": "running",
                                    "sets": curve[-1]["sets"],
                                    "facts_known": curve[-1]["facts_known"],
                                    "gap": curve[-1]["gap"],
                                    "curve": [(p["sets"], p["facts_known"]) for p in curve]}
    return facts


# ---------------------------------------------------------------------------------------------
# The comparison at an equal amount of text: the score of every run after the same number of
# sets, read from the learning curves that were saved anyway. No new training.
EQUAL_TEXT_SETS = [32, 64]


def score_after(run, sets):
    """The score of a run at the grading after `sets` sets. A run that was solved before then
    keeps its final score; any other run that ended earlier has no score there (None)."""
    at = {point["sets"]: point["facts_known"] for point in run["curve"]}
    if sets in at:
        return at[sets]
    if run["sets"] < sets and run["stop_reason"] == "solved":
        return run["final"]["facts_known"]
    return None


def equal_text_table(group, path):
    """Mean score of the repeats after 32 and after 64 sets, by width and vocabulary."""
    widths = sorted({w for w, _, _ in group})
    vocabularies = sorted({a for _, a, _ in group})
    columns = "r" * len(vocabularies)
    lines = ["\\begin{tabular}{@{}r" + "@{\\qquad}".join([columns] * len(EQUAL_TEXT_SETS)) + "@{}}",
             "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{len(vocabularies)}}}{{c}}{{After {sets} sets}}"
                                for sets in EQUAL_TEXT_SETS) + " \\\\",
             "Width & " + " & ".join(" & ".join(f"{a:,}".replace(",", "{,}") for a in vocabularies)
                                     for _ in EQUAL_TEXT_SETS) + " \\\\", "\\midrule"]
    numbers = {}
    for width in widths:
        cells = []
        for sets in EQUAL_TEXT_SETS:
            for animals in vocabularies:
                scores = [score_after(run, sets) for run in runs_at(group, width, animals)]
                if not scores or None in scores:
                    cells.append("--")
                    continue
                cells.append(f"{statistics.mean(scores):.0f}")
                numbers[f"w{width}_a{animals}_after{sets}"] = [round(score, 1) for score in scores]
        lines.append(f"{width} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return numbers


def collect_facts(runs, groups):
    """Every run in a flat list, plus counts, for checking the text against."""
    rows = []
    for run in runs:
        c = run["config"]
        rows.append({
            "name": run["name"], "usage": c["usage"], "seasons": c["seasons"],
            "table_animals": c["table_animals"], "animals": c["animals"], "width": c["width"],
            "seed": c["seed"], "ended": run["stop_reason"], "sets": run["sets"],
            "steps": run["steps"], "minutes": round(run["minutes"], 2),
            "sightings": round(run["sightings"], 1),
            "facts_known": round(run["final"]["facts_known"], 2),
            "facts_known_by_use": round(run["final"]["facts_known_by_use"], 2),
            "questions_right": round(run["final"]["questions_right"], 2),
            "gap": round(run["final"]["gap"], 4),
            "words_known": run["final"]["words_known"],
            "parameters": run["parameters"]["total"],
            "bits_total": round(run["language"]["bits_total"], 1),
            "fixed_sets": c.get("fixed_sets", 0), "learning_rate": c["learning_rate"],
        })
    return {"runs": sorted(rows, key=lambda r: (r["usage"], r["seasons"], r["width"],
                                                 r["animals"], r["seed"], r["name"])),
            "groups": {f"{usage}_x{seasons}": len(group) for (usage, seasons), group in
                       groups.items()}}


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    runs = R.load_runs(ROOT / "runs")
    groups, notes = R.comparison_groups(runs)

    # The table of runs and the decision rule, exactly as report.py prints them.
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        R.print_table(runs)
        R.print_means(groups)
        R.print_rule(R.rule_met(runs))
        for note in notes:
            print("NOTE:", note)
    (OUT / "rule_output.txt").write_text(printed.getvalue())

    for (usage, seasons), whole_group in sorted(groups.items()):
        tag = f"{usage}_x{seasons}"
        group = whole_group
        # Width 4 has a table of its own (width4_table); the main tables are widths 8 and up.
        without_4 = {key: run for key, run in whole_group.items() if key[0] != 4}
        if without_4:
            score_table(without_4, TABLES / f"scores_{tag}.tex")
            effort_table(without_4, TABLES / f"effort_{tag}.tex")
            load_table(without_4, TABLES / f"load_{tag}.tex")
            if usage == "zipf":
                score_table(without_4, TABLES / f"scores_by_use_{tag}.tex",
                            key="facts_known_by_use")
        one = {(usage, seasons): group}
        R.chart_more_words(one, FIGURES / f"words_{tag}.png")
        R.chart_learning(one, FIGURES / f"learning_{tag}.png")
        if usage == "zipf":
            R.chart_common_and_rare(one, FIGURES / f"rare_{tag}.png")
    R.chart_knowledge(runs, FIGURES / "knowledge.png")
    if ("equal", 1) in groups:
        without_width_4 = {key: run for key, run in groups[("equal", 1)].items() if key[0] != 4}
        effort_chart(
            without_width_4, FIGURES / "effort_equal_x1.png",
            "More words take longer to learn; smaller models take longer",
            "Every run in this chart learned every fact. What differs is how many sets of text "
            "it needed. Each dot is one run; both scales are log scales, so equal distances "
            "mean equal multiples.")

    facts = collect_facts(runs, groups)
    if ("equal", 1) in groups:
        facts["probe_equal_x1"] = probe_table(groups[("equal", 1)], TABLES / "probe_equal_x1.tex")
        shown = FIGURES / "probe_a1024_w32_equal_s0_color.png"
        if shown.exists():
            probe_strip(shown, FIGURES / "probe_strip.png")
    for (usage, seasons), group in sorted(groups.items()):
        if seasons > 1:
            tag = f"{usage}_x{seasons}"
            facts[f"season_use_{tag}"] = season_table(group, TABLES / f"season_use_{tag}.tex")
    if any(key[0] == 4 for group in groups.values() for key in group):
        facts["width4"] = width4_table(groups, TABLES / "width4.tex")
        facts["width4_know_nothing_gap"] = BLIND_GAPS
    for (usage, seasons), group in sorted(groups.items()):
        if seasons == 1:
            without_4 = {key: run for key, run in group.items() if key[0] != 4}
            facts[f"equal_text_{usage}_x1"] = equal_text_table(
                without_4, TABLES / f"equal_text_{usage}_x1.tex")
    large_runs = [run for run in (read_large(*entry) for entry in LARGE_RUNS) if run]
    if large_runs:
        season = facts.get("season_use_equal_x4", [])
        blind = season[0]["blind_score"] if season else 37.2
        flat = [(run["sets"], run["final"]["facts_known"])
                for run in groups.get(("equal", 4), {}).values()
                if run["stop_reason"] == "plateau" and run["config"]["width"] == 16]
        large_chart(large_runs, FIGURES / "large_equal_x4.png", blind, flat)
        facts["large_table"] = large_table(TABLES / "large_equal_x4.tex")
        facts["large_runs"] = large_facts(large_runs)
    facts["fullness_x4"] = fullness_table(TABLES / "fullness_x4.tex")
    facts["side_checks"] = side_checks()
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1))
    print(f"{len(runs)} runs; groups: " + ", ".join(f"{u} x{s}: {len(g)}" for (u, s), g in
                                                    sorted(groups.items())))
    print("wrote", ", ".join(sorted(p.name for p in FIGURES.iterdir())))


if __name__ == "__main__":
    main()

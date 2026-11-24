"""Builds the tables and the chart of the addendum (report/addendum.tex): can the models answer
the yes/no questions, and does more training help?

    .venv/bin/python report/make_addendum.py

It reads
    results/questions_now.json     every saved model tested on the questions (questions.py)
    runs_more/*/summary.json       the saved models trained further (more_training.py); for a job
                                   that is still running, its log in logs/more/ is read instead
    runs_more_exact/*/summary.json three runs made again from the start with a longer fixed length
and writes report/tables/q_*.tex, report/figures/q_more.png and report/addendum_facts.json.
"""

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402

import report as R  # noqa: E402  (the project's report.py, for the look of the charts)

OUT = Path(__file__).resolve().parent
TABLES, FIGURES = OUT / "tables", OUT / "figures"

# The charts have a caption below them in the addendum, so the empty strip that report.py
# leaves at the top for a title is cut off (as report/make_assets.py does).
TOP_STRIP = 0.52  # inches
_save_figure = R.save_figure


def save_figure_cropped(figure, path):
    import matplotlib.image as image

    _save_figure(figure, path)
    picture = image.imread(path)
    image.imsave(path, picture[int(TOP_STRIP * R.SAVE_DPI):])


R.save_figure = save_figure_cropped
WIDTHS, VOCABULARIES = [8, 16, 32], [16, 64, 256, 1024]
VOCABULARY_COLORS = {16: "#2a78d6", 64: "#eb6834", 256: "#1baf7a", 1024: "#eda100"}
MARKS = [16, 32, 64, 128, 256]  # extra sets at which the table of more training reads the score


def number(value):
    return f"{value:,}".replace(",", "{,}")


def by_place(rows, usage, seasons=1):
    """{(width, animals): [the rows of the repeats, by seed]} for the planned rounds."""
    table = {}
    for row in rows:
        if (row["usage"] == usage and row["seasons"] == seasons and row["table_animals"] == 1024
                and row["folder"] == "runs" and row["width"] in WIDTHS
                and not row["name"].endswith("_f512")):
            table.setdefault((row["width"], row["animals"]), []).append(row)
    return {key: sorted(group, key=lambda r: r["seed"]) for key, group in table.items()}


def now_table(rows, usage, path):
    """The question score of every model as the experiment left it: three repeats per cell."""
    table = by_place(rows, usage)
    lines = ["\\begin{tabular}{@{}r" + "c" * len(VOCABULARIES) + "@{}}", "\\toprule",
             " & \\multicolumn{" + str(len(VOCABULARIES)) + "}{c}{Animals in the vocabulary} \\\\",
             "Width & " + " & ".join(number(a) for a in VOCABULARIES) + " \\\\", "\\midrule"]
    for width in WIDTHS:
        cells = [", ".join(f"{r['questions_right']:.0f}" for r in table.get((width, animals), []))
                 for animals in VOCABULARIES]
        lines.append(f"{width} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return {f"w{w}_a{a}": [round(r["questions_right"], 1) for r in group]
            for (w, a), group in sorted(table.items())}


def read_more(name):
    """One continued model: its scores at the start and at every grading since. From its summary
    if the job has ended, otherwise from its log (then 'ended' is 'running')."""
    summary = ROOT / "runs_more" / f"{name}_more" / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        curve = [(0, s["start"]["questions_right"], s["start"]["facts_known"])]
        curve += [(p["extra_sets"], p["questions_right"], p["facts_known"]) for p in s["curve"]]
        return {"name": name, "ended": s["stop_reason"], "sets_before": s["sets_before"],
                "curve": curve}
    log = ROOT / "logs" / "more" / f"{name}.log"
    if not log.exists():
        return None
    text = log.read_text()
    start = re.search(r"(\S+): ([\d,]+) sets read before\. Start: facts known ([\d.]+), "
                      r"questions ([\d.]+)", text)
    if not start:
        return None
    curve = [(0, float(start.group(4)), float(start.group(3)))]
    for line in text.splitlines():
        words = line.replace(",", "").split()
        if len(words) == 6 and words[0].isdigit():
            curve.append((int(words[0]), float(words[2]), float(words[1])))
    return {"name": name, "ended": "running",
            "sets_before": int(start.group(2).replace(",", "")), "curve": curve}


def whole_course(run):
    """The question score of one model against ALL the sets it has read: the gradings of its
    original run, then those of its extra training. Returns [(sets, question score, facts)]."""
    original = json.loads((ROOT / "runs" / run["name"] / "summary.json").read_text())
    course = [(p["sets"], p["questions_right"], p["facts_known"]) for p in original["curve"]]
    course += [(run["sets_before"] + extra, questions, facts)
               for extra, questions, facts in run["curve"] if extra > 0]
    return course


def first_reached(course, level, which=1):
    """The number of sets at the first grading where the score (which=1: questions, which=2:
    statements) was at least `level`; None if it never was."""
    for point in course:
        if point[which] >= level:
            return point[0]
    return None


def score_at(run, extra):
    """The question score after `extra` more sets. A model whose questions were solved before
    then keeps its last score; one that has not got that far yet has none (None)."""
    at = {sets: questions for sets, questions, _ in run["curve"]}
    if extra in at:
        return at[extra]
    last = run["curve"][-1]
    if last[0] < extra and run["ended"] == "questions solved":
        return last[1]
    return None


def more_runs(usage, seasons=1):
    """{(width, animals): [continued models of the three repeats]}."""
    table = {}
    for width in WIDTHS:
        for animals in VOCABULARIES:
            for seed in range(3):
                tag = "" if seasons == 1 else f"_x{seasons}"
                run = read_more(f"a{animals}_w{width}_{usage}{tag}_s{seed}")
                if run:
                    table.setdefault((width, animals), []).append(run)
    return table


def more_table(table, path, marks):
    """The question score of the three repeats at the start and after more training."""
    lines = ["\\begin{tabular}{@{}rr" + "c" * (1 + len(marks)) + "@{}}", "\\toprule",
             " & & At the start & \\multicolumn{" + str(len(marks))
             + "}{c}{After this many more sets} \\\\",
             "Width & Animals & & " + " & ".join(str(mark) for mark in marks) + " \\\\",
             "\\midrule"]
    facts = {}
    for width in WIDTHS:
        for animals in VOCABULARIES:
            runs = table.get((width, animals), [])
            if not runs:
                continue
            cells = []
            for extra in [0] + list(marks):
                scores = [score_at(run, extra) for run in runs]
                cells.append(", ".join("--" if s is None else f"{s:.0f}" for s in scores))
                facts[f"w{width}_a{animals}_plus{extra}"] = [None if s is None else round(s, 1)
                                                            for s in scores]
            lines.append(f"{width} & {number(animals)} & " + " & ".join(cells) + " \\\\")
        if width != WIDTHS[-1]:
            lines.append("\\addlinespace[3pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return facts


def when_table(table, path, level=90):
    """For every model: the sets it had read when it first knew the statements, and when it
    first answered the questions (score at least `level`). '>n' = not yet after n sets."""
    lines = ["\\begin{tabular}{@{}rrcc@{}}", "\\toprule",
             "Width & Animals & Sets until the statements were known & "
             "Sets until the questions were answered \\\\", "\\midrule"]
    facts = {}
    for width in WIDTHS:
        for animals in VOCABULARIES:
            runs = table.get((width, animals), [])
            if not runs:
                continue
            statements, questions = [], []
            for run in runs:
                course = whole_course(run)
                known = first_reached(course, level, which=2)
                answered = first_reached(course, level, which=1)
                statements.append(number(known) if known else f"$>${number(course[-1][0])}")
                questions.append(number(answered) if answered else f"$>${number(course[-1][0])}")
                facts[run["name"]] = {"statements_at": known, "questions_at": answered,
                                      "sets_read": course[-1][0], "ended": run["ended"]}
            lines.append(f"{width} & {number(animals)} & " + ", ".join(statements) + " & "
                         + ", ".join(questions) + " \\\\")
        if width != WIDTHS[-1]:
            lines.append("\\addlinespace[3pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return facts


def course_chart(table, path):
    """The question score against all the sets a model has read (its original run and the
    extra training together): one panel for each width, one line for each model, colored by the
    size of its vocabulary. A small tick marks where the original run was ended."""
    legend = [[R.key_line(VOCABULARY_COLORS[a], f"{a:,} animals") for a in VOCABULARIES],
              [Line2D([], [], linestyle="none", marker="|", markersize=12, markeredgewidth=1.5,
                      color=R.INK, label="where the experiment had ended the run"),
               R.key_dot(R.INK_MUTED, "finished", "questions answered"),
               R.key_dot(R.INK_MUTED, "budget", "not yet")]]
    sizes = (R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT)
    R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT = 2.2, 2.1, 0.35, 0.25
    try:
        figure, panels = R.new_figure(len(WIDTHS), "", "", legend)
    finally:
        R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT = sizes
    longest = max(whole_course(run)[-1][0] for runs in table.values() for run in runs)
    for place, (panel, width) in enumerate(zip(panels, WIDTHS)):
        for animals in VOCABULARIES:
            for run in table.get((width, animals), []):
                course = [point for point in whole_course(run) if point[0] >= 8]
                xs, ys = [p[0] for p in course], [p[1] for p in course]
                color = VOCABULARY_COLORS[animals]
                panel.plot(xs, ys, color=color, linewidth=R.LINE * 0.8, alpha=0.9, zorder=2.5,
                           solid_capstyle="round", solid_joinstyle="round")
                at_end = dict((p[0], p[1]) for p in course).get(run["sets_before"])
                if at_end is not None:
                    panel.plot([run["sets_before"]], [at_end], linestyle="none", marker="|",
                               markersize=12, markeredgewidth=1.5, color=R.INK, zorder=3.2)
                how = "finished" if ys[-1] >= 90 else "budget"
                R.draw_dot(panel, xs[-1], ys[-1], color, how)
        R.log_axis(panel, 8, longest)
        panel.set_xlim(7, longest * 1.35)
        panel.set_xticks([10, 30, 100, 300, 1000])
        panel.set_xticklabels(["10", "30", "100", "300", "1,000"])
        R.panel_title(panel, f"width {width}")
        R.label_axes(panel, place, "sets of text read in all (log scale)",
                     "question score (0 to 100)")
    R.save_figure(figure, path)


def more_chart(table, path):
    """The question score against extra sets of training: one panel for each width, one line
    for each model, colored by the size of its vocabulary."""
    legend = [[R.key_line(VOCABULARY_COLORS[a], f"{a:,} animals") for a in VOCABULARIES],
              [R.key_dot(R.INK_MUTED, "finished", "questions solved"),
               R.key_dot(R.INK_MUTED, "budget", "not solved when the extra training ended")]]
    sizes = (R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT)
    R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT = 2.2, 2.1, 0.35, 0.25
    try:
        figure, panels = R.new_figure(len(WIDTHS), "", "", legend)
    finally:
        R.PANEL_WIDTH, R.PANEL_HEIGHT, R.GAP_X, R.MARGIN_RIGHT = sizes
    longest = max((run["curve"][-1][0] for runs in table.values() for run in runs), default=256)
    for place, (panel, width) in enumerate(zip(panels, WIDTHS)):
        for animals in VOCABULARIES:
            for run in table.get((width, animals), []):
                # The start (0 extra sets) is drawn at 0.7 so that it fits on a log scale.
                xs = [max(extra, 0.7) for extra, _, _ in run["curve"]]
                ys = [questions for _, questions, _ in run["curve"]]
                color = VOCABULARY_COLORS[animals]
                panel.plot(xs, ys, color=color, linewidth=R.LINE * 0.8, alpha=0.9, zorder=2.5,
                           solid_capstyle="round", solid_joinstyle="round")
                how = "finished" if run["ended"] == "questions solved" else "budget"
                R.draw_dot(panel, xs[-1], ys[-1], color, how)
        R.log_axis(panel, 0.7, longest)
        panel.set_xlim(0.6, longest * 1.35)
        panel.set_xticks([0.7, 2, 8, 32, 128])
        panel.set_xticklabels(["start", "2", "8", "32", "128"])
        R.panel_title(panel, f"width {width}")
        R.label_axes(panel, place, "more sets of training (log scale)", "question score (0 to 100)")
    R.save_figure(figure, path)


def exact_checks():
    """The three runs made again from the start with a longer fixed length, set beside the same
    models continued from their saved state."""
    rows = []
    for name, original in (("a16_w8_equal_s0_f320", "a16_w8_equal_s0"),
                           ("a64_w16_equal_s0_f320", "a64_w16_equal_s0"),
                           ("a256_w32_equal_s1_f320", "a256_w32_equal_s1")):
        summary = ROOT / "runs_more_exact" / name / "summary.json"
        if not summary.exists():
            continue
        s = json.loads(summary.read_text())
        exact = {p["sets"]: (round(p["questions_right"], 1), round(p["facts_known"], 1))
                 for p in s["curve"]}
        continued = read_more(original)
        rows.append({"name": name, "original": original, "sets_before":
                     continued["sets_before"] if continued else None, "exact": exact,
                     "continued": continued["curve"] if continued else None})
    return rows


X4_MODELS = [  # the models of the 32-bit round that knew the statements: (name, animals, width, repeat)
    ("a16_w8_equal_x4_s0", 16, 8, 1), ("a16_w8_equal_x4_s1", 16, 8, 2),
    ("a16_w16_equal_x4_s0", 16, 16, 1), ("a16_w16_equal_x4_s1", 16, 16, 2),
    ("a16_w32_equal_x4_s0", 16, 32, 1), ("a16_w32_equal_x4_s1", 16, 32, 2),
    ("a256_w16_equal_x4_s0_f512", 256, 16, 1), ("a256_w16_equal_x4_s1_long", 256, 16, 2),
    ("a256_w32_equal_x4_s0_long", 256, 32, 1),
]


def x4_table(path, marks=(16, 32, 64)):
    """The nine models of the 32-bit round that knew the statements: their question score at the
    start and after more sets, and their statement score before and after."""
    lines = ["\\begin{tabular}{@{}rrr" + "r" * (1 + len(marks)) + "rr@{}}", "\\toprule",
             " & & & \\multicolumn{" + str(1 + len(marks)) + "}{c}{Question score} & "
             "\\multicolumn{2}{c}{Statement score} \\\\",
             "Animals & Width & Repeat & at the start & "
             + " & ".join(f"after {mark}" for mark in marks) + " & at the start & at the end \\\\",
             "\\midrule"]
    rows = []
    for name, animals, width, repeat in X4_MODELS:
        run = read_more(name)
        if run is None:
            continue
        scores = [score_at(run, extra) for extra in (0,) + tuple(marks)]
        cells = ["--" if value is None else f"{value:.0f}" for value in scores]
        lines.append(f"{animals} & {width} & {repeat} & " + " & ".join(cells)
                     + f" & {run['curve'][0][2]:.1f} & {run['curve'][-1][2]:.1f} \\\\")
        rows.append({"name": name, "ended": run["ended"], "extra_sets": run["curve"][-1][0],
                     "questions": [None if v is None else round(v, 1) for v in scores],
                     "facts_start": round(run["curve"][0][2], 2),
                     "facts_end": round(run["curve"][-1][2], 2),
                     "lowest_facts": round(min(f for _, _, f in run["curve"]), 2)})
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    return rows


def exact_table(rows, path):
    """The exact checks: a run that was never stopped, beside the same model continued from its
    saved state, at the two numbers of sets where both were graded."""
    described = {"a16_w8_equal_s0": "width 8, 16 animals", "a64_w16_equal_s0": "width 16, 64 animals",
                 "a256_w32_equal_s1": "width 32, 256 animals"}
    lines = ["\\begin{tabular}{@{}lrcccc@{}}", "\\toprule",
             " & Stopped & \\multicolumn{2}{c}{Never stopped} & "
             "\\multicolumn{2}{c}{Continued from the saved model} \\\\",
             "Model & at set & at set 128 & at set 320 & at set 128 & at set 320 \\\\",
             "\\midrule"]
    for row in rows:
        if not row["continued"]:
            continue
        exact = {int(sets): values[0] for sets, values in row["exact"].items()}
        continued = {row["sets_before"] + extra: questions
                     for extra, questions, _ in row["continued"]}
        lines.append(f"{described[row['original']]} & {row['sets_before']} & "
                     f"{exact[128]:.0f} & {exact[320]:.0f} & "
                     f"{continued[128]:.0f} & {continued[320]:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    now = json.loads((ROOT / "results" / "questions_now.json").read_text())
    facts = {"now_equal": now_table(now, "equal", TABLES / "q_now_equal.tex"),
             "now_zipf": now_table(now, "zipf", TABLES / "q_now_zipf.tex")}
    facts["now_all"] = [{key: row[key] for key in ("name", "folder", "sets", "facts_known",
                                                    "questions_right", "yes_right", "no_right")}
                        for row in now]
    for usage in ("equal", "zipf"):
        table = more_runs(usage)
        if not table:
            continue
        reached = min(run["curve"][-1][0] if run["ended"] != "questions solved" else 10 ** 6
                      for runs in table.values() for run in runs)
        marks = [mark for mark in MARKS if mark <= reached] or MARKS[:1]
        facts[f"more_{usage}"] = more_table(table, TABLES / f"q_more_{usage}.tex", marks)
        more_chart(table, FIGURES / f"q_more_{usage}.png")
        facts[f"when_{usage}"] = when_table(table, TABLES / f"q_when_{usage}.tex")
        course_chart(table, FIGURES / f"q_course_{usage}.png")
        runs = [run for group in table.values() for run in group]
        facts[f"more_{usage}_runs"] = [
            {"name": run["name"], "ended": run["ended"], "sets_before": run["sets_before"],
             "extra_sets": run["curve"][-1][0], "questions_start": run["curve"][0][1],
             "questions_end": run["curve"][-1][1], "facts_end": run["curve"][-1][2],
             "lowest_facts": min(f for _, _, f in run["curve"])} for run in runs]
        ended = [run for run in runs if run["ended"] != "running"]
        facts[f"more_{usage}_summary"] = {
            "models": len(runs), "still_running": len(runs) - len(ended),
            "questions_solved": sum(run["ended"] == "questions solved" for run in runs),
            "marks_shown": marks}
    facts["x4_models"] = x4_table(TABLES / "q_more_x4.tex")
    facts["exact_checks"] = exact_checks()
    exact_table(facts["exact_checks"], TABLES / "q_exact.tex")
    (OUT / "addendum_facts.json").write_text(json.dumps(facts, indent=1))
    print("wrote the addendum's tables and chart;",
          {key: value for key, value in facts.items() if key.endswith("_summary")})


if __name__ == "__main__":
    main()

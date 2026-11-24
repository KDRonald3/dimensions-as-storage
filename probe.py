"""
probe.py - look inside the trained word table.

What this file is for
---------------------
The hypothesis says a language model is a representation of the MEANING of words. In this
experiment the meaning of an animal word is its facts (color, food, home, waking time), and the
model's representation of the word is one row of its word table: `width` numbers. So we can look
at the representation itself and ask whether the meaning can be seen in it. This file takes three
looks at the rows of the animals in use:

  1. Can a fact be read from a row by a simple rule?
  2. Do words with similar meanings sit close together?
  3. A picture of the rows, saved to results/.

It trains nothing and changes nothing in the run. train.py and score.py measure what a model
ANSWERS; this file looks at how the answers are STORED.

How to run it
-------------
    python probe.py runs/a1024_w32_equal_s0              # a finished run (see train.py)
    python probe.py runs/a16_w32_equal_s0 --attribute food
    python probe.py --help                               # every setting is a flag

Terms used in this file
-----------------------
row                   The `width` numbers the word table holds for one word. Also called the
                      word's vector or embedding.
linear rule           A rule that only weighs and adds: for every value it computes one score, a
                      weighted sum of the row's numbers, and names the value with the highest
                      score. It is the simplest kind of reader. If such a rule can read a fact,
                      the fact lies in the row in plain sight.
held-out              Kept aside while the rule is fitted and used only to test it. A rule that
                      is right about animals it never saw has found a pattern, not learned a list.
chance                The accuracy of always naming the most common value, knowing no animal.
luck                  With few animals a number comes out a little different every time for no
                      reason at all. Parts 1 and 2 measure how much: they hand every row to the
                      wrong animal, so that the rows can say nothing, and look how far the
                      number then strays (printed as +/-). A result inside that range is luck.
cosine similarity     A number from -1 to 1 saying whether two rows point the same way: 1 = the
                      same direction, 0 = unrelated, -1 = opposite.
principal components  The directions in which a set of rows differ most from each other. Drawing
                      the rows along the first two gives the most telling flat picture of them.
"""

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight into a file; no window is opened

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from language import ATTRIBUTES, VALUES
from train import load_run

# On the command line an attribute is named by its first word: color, food, home, waking.
ATTRIBUTE_NAMES = [name.split()[0] for name in ATTRIBUTES]


@dataclass
class ProbeConfig:
    results_dir: str = "results"  # the picture is saved here
    attribute: str = "color"  # the attribute the picture is about: color, food, home or waking
    seed: int = 0  # decides the random halves of part 1 and the shuffles of parts 1 and 2
    splits: int = 20  # part 1 draws a fresh random half this many times and averages
    shuffles: int = 20  # how often the rows are handed to the wrong animals, to measure luck
    min_animals: int = 64  # part 1 is skipped for fewer animals in use: too few to test on
    neighbors: int = 3  # nearest rows listed for each animal in part 2
    shown: int = 6  # how many animals part 2 lists (the first ones, which are base animals)
    label_up_to: int = 16  # the picture names its highlighted dots only up to this many animals


# The colors of the picture. One accent color for what a panel is about, gray for the rest, and
# all writing in shades of ink (never in the accent color: colored text is hard to read).
SURFACE = "#fcfcfb"  # the light background
INK = "#0b0b0b"  # titles
INK_SECONDARY = "#52514e"  # explanations and names
INK_MUTED = "#898781"  # axis numbers and axis labels
GRID = "#e1e0d9"  # hairline grid
AXIS = "#c3c2b7"  # the two axis lines
ACCENT = "#2a78d6"  # the highlighted dots
REST = "#c3c2b7"  # all other dots

DOT = 6  # a dot is 6 points across, about 8 pixels on a screen
RING = 1.5  # a thin ring in the background color keeps overlapping dots apart
# Where a name may stand beside its dot, in the order tried: (points sideways, points upward,
# horizontal anchor, vertical anchor).
NAME_PLACES = [(6, 0, "left", "center"), (-6, 0, "right", "center"),
               (0, 6, "center", "bottom"), (0, -6, "center", "top")]


def say(text):
    """Print a paragraph, wrapped to the screen and indented like the rest of the output."""
    print(textwrap.fill(text, width=96, initial_indent="  ", subsequent_indent="  "))


def animal_rows(language, model):
    """The learned rows of the animals in use, as an array [animals, width].

    The word table is TIED (see model.py): the same row is what the model reads when the word
    comes in and what it compares against when it decides whether to say the word. Animals that
    are in the table but not in use are left out: the model never read about them.

    One caveat for the three looks below. They read the raw numbers of a row. The model reads a
    row through a layer norm, which removes the average of the numbers it is given. So a look
    here can pick up something the model itself cannot read off a word, such as anything stored
    in a row's average. At width 4 that leaves the model three directions per row, not four.
    """
    table = model.table.weight.detach().numpy()
    return table[language.animal_ids].astype(np.float64)


# ------------------------------------------------------------------------------------------------
# 1. Can a fact be read from a row by a simple rule?
# ------------------------------------------------------------------------------------------------


def fit_linear_rule(rows, values, n_values):
    """Fit a linear rule on some animals. Returns its weights, [width + 1, n_values].

    For every value the rule computes one score: a weighted sum of the row's numbers plus an
    offset. The weights are found by LEAST SQUARES: over the animals given, the score of the true
    value should come as close to 1, and the scores of the other values as close to 0, as
    weighted sums can get. numpy solves that in one call; nothing is trained step by step.
    """
    with_offset = np.hstack([rows, np.ones((len(rows), 1))])  # a constant 1 carries the offset
    wanted = np.eye(n_values)[values]  # per animal: 1 in the column of its true value, 0 elsewhere
    weights, *_ = np.linalg.lstsq(with_offset, wanted, rcond=None)
    return weights


def apply_linear_rule(weights, rows):
    """The value (its number, 0 to 3) the rule names for each row: the one with the top score."""
    with_offset = np.hstack([rows, np.ones((len(rows), 1))])
    return (with_offset @ weights).argmax(axis=1)


def held_out_accuracy(rows, values, n_values, rng, splits):
    """How often a linear rule names the right value for animals it was NOT fitted on.

    The animals are split at random into two halves. The rule is fitted on one half and tested
    on the other. One split of a few dozen animals is a noisy measurement, so this is repeated
    `splits` times with a fresh random half and the accuracies are averaged. Returns 0 to 1.
    """
    animals = len(rows)
    half = animals // 2
    total = 0.0
    for split in range(splits):
        order = rng.permutation(animals)
        fitted_on, tested_on = order[:half], order[half:]
        weights = fit_linear_rule(rows[fitted_on], values[fitted_on], n_values)
        named = apply_linear_rule(weights, rows[tested_on])
        total += (named == values[tested_on]).mean()
    return float(total / splits)


def chance_level(values, n_values):
    """The accuracy of always naming the most common value. Counted from the facts as they are:
    0.25 when four values are equally common, more when one value is more common than the rest."""
    return float(np.bincount(values, minlength=n_values).max() / len(values))


def shuffled_accuracy(rows, values, n_values, rng, splits, shuffles):
    """What held_out_accuracy gives when the rows say nothing about the fact, and how far that
    number strays by luck. Returns (mean, spread), both 0 to 1.

    Every animal is handed the row of some other animal, so the rows can no longer say anything
    about the facts, and the test is run as before. That is done `shuffles` times, each time
    with a fresh shuffle. The mean of the results is what the procedure scores when there is
    nothing to read. It tends to land a little BELOW chance: a value that happens to be common
    in the fitted half is rare in the other half. The spread (the standard deviation: how far a
    single result usually lies from the mean) is the yardstick for luck.
    """
    results = []
    for shuffle in range(shuffles):
        shuffled_rows = rows[rng.permutation(len(rows))]
        results.append(held_out_accuracy(shuffled_rows, values, n_values, rng, splits))
    return float(np.mean(results)), float(np.std(results))


def read_facts(rows, lexicon, rng, splits, shuffles=ProbeConfig.shuffles):
    """Part 1 for each of the four attributes (first season). Returns one dictionary per
    attribute: the real test (held_out), the same test on shuffled rows with its spread
    (shuffled and shuffled_spread, see shuffled_accuracy) and chance."""
    results = []
    for attribute, name in enumerate(ATTRIBUTES):
        values = lexicon[:, 0, attribute]
        n_values = len(VALUES[attribute])
        held_out = held_out_accuracy(rows, values, n_values, rng, splits)
        shuffled, spread = shuffled_accuracy(rows, values, n_values, rng, splits, shuffles)
        results.append({"attribute": name, "held_out": held_out, "shuffled": shuffled,
                        "shuffled_spread": spread, "chance": chance_level(values, n_values)})
    return results


def print_read_facts(results, animals, width, splits, shuffles):
    half = animals // 2
    say(f"A linear rule is fitted on a random half of the animals ({half:,}) and tested on the "
        f"other half ({animals - half:,}), which it never saw. The numbers are averages over "
        f"{splits} random halves.")
    print(f"  {'attribute':14s} {'held-out':>9s} {'rows shuffled':>17s} {'chance':>8s}")
    for result in results:
        print(f"  {result['attribute']:14s} {100 * result['held_out']:8.1f}% "
              f"{100 * result['shuffled']:7.1f}% +/-{100 * result['shuffled_spread']:5.1f} "
              f"{100 * result['chance']:7.1f}%")
    print("  held-out       how often the rule names the right value for animals it was not")
    print("                 fitted on")
    print(f"  rows shuffled  the same test with every row handed to the wrong animal, {shuffles} "
          f"times over:")
    print("                 what this procedure scores when the rows say nothing about the fact,")
    print("                 and (+/-) how far that number strays from one shuffle to the next")
    print("  chance         always naming the most common value")
    say("Read held-out against rows shuffled, not against chance. A held-out number that is not "
        "at least twice the +/- above the shuffled one shows nothing: a difference of that size "
        "is luck. The fewer animals, the larger the +/-.")
    if half <= width + 1:
        # As many adjustable numbers as animals: the rule can match its half exactly whatever the
        # rows hold, like a straight line through two points. Such a fit says little about others.
        say(f"Caution: the rule has {width + 1} adjustable numbers per value and was fitted on "
            f"only {half} animals. It can then fit its half by rote, so the held-out numbers are "
            f"weak evidence. Use a run with more animals.")


def look_at_reading(rows, language, config):
    """Part 1. Returns the list from read_facts, or None when there are too few animals."""
    print("1. CAN A FACT BE READ FROM A ROW BY A SIMPLE RULE?")
    animals, width = rows.shape
    if animals < config.min_animals:
        say(f"Skipped. This check fits a rule on half of the animals and tests it on the other "
            f"half. With {animals} animals in use only {animals - animals // 2} would be left to "
            f"test on, and then luck alone moves the result by many points. The check is made "
            f"from {config.min_animals} animals in use on, for example for a run made with "
            f".venv/bin/python train.py --animals {config.min_animals}. Even there much is "
            f"luck (the check prints how much); the more animals, the less.")
        return None
    rng = np.random.default_rng(config.seed)
    results = read_facts(rows, language.lexicon[:animals], rng, config.splits, config.shuffles)
    print_read_facts(results, animals, width, config.splits, config.shuffles)
    if language.config.seasons > 1:
        say(f"(Only the facts of the first season, {language.seasons_in_use[0]}, are tested here.)")
    return results


# ------------------------------------------------------------------------------------------------
# 2. Do words with similar meanings sit close together?
# ------------------------------------------------------------------------------------------------


def shared_facts(lexicon):
    """shared[i, j] = how many facts animals i and j have in common.

    A fact is shared when both animals have the same value for the same attribute in the same
    season. lexicon is [animals, seasons, 4]; the result is [animals, animals].
    """
    facts = lexicon.reshape(len(lexicon), -1)  # [animals, seasons x 4]: one column per fact
    return (facts[:, None, :] == facts[None, :, :]).sum(axis=2)


def cosine_similarity(rows):
    """similarity[i, j] says whether rows i and j point the same way (1), are unrelated (0) or
    point in opposite directions (-1).

    The average row is subtracted first. Every animal row has a large part in common with all
    the others (whatever makes a word an animal word). Left in, it would make all animals look
    alike; taken out, the comparison is about what DIFFERS between animals.
    """
    differences = rows - rows.mean(axis=0)
    lengths = np.linalg.norm(differences, axis=1, keepdims=True)
    unit = differences / np.maximum(lengths, 1e-12)  # every row scaled to length 1
    return unit @ unit.T


def nearest(similarity, how_many):
    """For each animal the numbers of its `how_many` most similar OTHER animals, nearest first."""
    similarity = similarity.copy()
    np.fill_diagonal(similarity, -np.inf)  # an animal is not its own neighbor
    return np.argsort(-similarity, axis=1, kind="stable")[:, :how_many]


def neighbor_summary(similarity, shared):
    """Average number of facts shared, for three kinds of pairs. Returns a dictionary.

    nearest  an animal and the animal whose row is most similar to its own
    random   two different animals picked at random (worked out exactly, over all pairs)
    best     an animal and the animal that shares the most facts with it, whatever the rows say:
             the most that `nearest` could reach
    If the table arranges the animals by meaning, `nearest` lies clearly above `random`.
    """
    animals = len(shared)
    first = nearest(similarity, 1)[:, 0]
    others = ~np.eye(animals, dtype=bool)  # every pair except an animal with itself
    return {
        "nearest": float(shared[np.arange(animals), first].mean()),
        "random": float(shared[others].mean()),
        "best": float(np.where(others, shared, -1).max(axis=1).mean()),
    }


def neighbors_by_luck(similarity, shared, rng, shuffles):
    """What `nearest` of neighbor_summary comes to when the rows say nothing about the facts,
    and how far it strays by luck. Returns (mean, spread).

    As in part 1, every row is handed to the wrong animal, `shuffles` times over. Which row is
    nearest to which stays as it is; only the facts that go with a row change. With few animals
    the spread is large: for 16 animals about a quarter of a fact.
    """
    animals = len(shared)
    first = nearest(similarity, 1)[:, 0]
    results = []
    for shuffle in range(shuffles):
        owner = rng.permutation(animals)  # owner[i]: the animal whose facts row i now stands for
        results.append(shared[owner, owner[first]].mean())
    return float(np.mean(results)), float(np.std(results))


def look_at_neighbors(rows, language, config):
    """Part 2. Returns the dictionary from neighbor_summary, with the mean and the spread of
    neighbors_by_luck added as "shuffled" and "shuffled_spread"."""
    print("2. DO WORDS WITH SIMILAR MEANINGS SIT CLOSE TOGETHER?")
    animals = len(rows)
    names = language.animal_names
    n_facts = language.facts_per_animal
    similarity = cosine_similarity(rows)
    shared = shared_facts(language.lexicon[:animals])
    how_many = min(config.neighbors, animals - 1)
    neighbors = nearest(similarity, how_many)
    say(f"For each animal: the {how_many} animals with the most similar rows, their cosine "
        f"similarity (1 = same direction, 0 = unrelated, -1 = opposite) and how many of the "
        f"{n_facts} facts the two share.")
    for animal in range(min(config.shown, animals)):
        cells = [f"{names[other]} {similarity[animal, other]:+.2f} ({shared[animal, other]} shared)"
                 for other in neighbors[animal]]
        print(f"  {names[animal]:10s} " + "   ".join(f"{cell:26s}" for cell in cells).rstrip())
    summary = neighbor_summary(similarity, shared)
    rng = np.random.default_rng(config.seed)
    summary["shuffled"], summary["shuffled_spread"] = neighbors_by_luck(similarity, shared, rng,
                                                                       config.shuffles)
    print(f"  Facts shared on average, out of {n_facts}:")
    print(f"    an animal and its nearest row           {summary['nearest']:5.2f}")
    print(f"    the same, rows handed to wrong animals  {summary['shuffled']:5.2f} "
          f"+/- {summary['shuffled_spread']:.2f}   (what luck alone gives)")
    print(f"    two different animals picked at random  {summary['random']:5.2f}")
    print(f"    an animal and the animal most like it   {summary['best']:5.2f}   "
          f"(the most the first line could reach)")
    say("If the first line is not at least twice the +/- above the second, the table shows no "
        "arrangement by meaning: a difference of that size is luck.")
    return summary


# ------------------------------------------------------------------------------------------------
# 3. A picture
# ------------------------------------------------------------------------------------------------


def two_directions(rows):
    """Lay the rows out flat. Returns (points [animals, 2], share [2]).

    A row has `width` numbers, too many to draw. PRINCIPAL COMPONENTS are the directions in which
    the rows differ most from each other: the first is the single direction with the largest
    spread, the second the largest spread at right angles to the first. points holds each row's
    position along these two. share says how much of all the differences between rows each of
    the two directions holds; what is left over is not in the picture.
    numpy finds the directions with a singular value decomposition (SVD).
    """
    differences = rows - rows.mean(axis=0)  # as in cosine_similarity: look at what differs
    _, sizes, directions = np.linalg.svd(differences, full_matrices=False)
    points = differences @ directions[:2].T
    spread = sizes**2
    if spread.sum() == 0:  # every row is the same: there is nothing to lay out
        return points, np.zeros(2)
    return points, spread[:2] / spread.sum()


def dot_boxes(ax, points):
    """The small square each dot covers in the saved picture, in pixels."""
    radius = DOT / 2 * ax.figure.dpi / 72  # half a dot (a point is 1/72 inch)
    return [Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
            for x, y in ax.transData.transform(points)]


def place_name(ax, point, name, in_the_way):
    """Write a name beside its dot: right of it if there is room, else left, above or below.

    "Room" means inside the panel and not on top of any rectangle of `in_the_way`. Returns the
    rectangle the name now covers, or None if no side had room (then nothing is written).
    """
    renderer = ax.figure.canvas.get_renderer()
    panel = ax.get_window_extent(renderer)
    for sideways, upward, across, updown in NAME_PLACES:
        text = ax.annotate(
            name, point, xytext=(sideways, upward), textcoords="offset points", ha=across,
            va=updown, fontsize=8.5, color=INK_SECONDARY, zorder=4,
            # A thin outline in the background color keeps a name readable on a grid line or dot.
            path_effects=[withStroke(linewidth=2, foreground=SURFACE)],
        )
        box = text.get_window_extent(renderer)
        inside = (panel.x0 <= box.x0 and box.x1 <= panel.x1
                  and panel.y0 <= box.y0 and box.y1 <= panel.y1)
        if inside and not any(box.overlaps(other) for other in in_the_way):
            return box
        text.remove()  # no room on this side: take the name away and try the next side
    return None


def name_dots(ax, points, names, other_points):
    """Name the highlighted dots of one panel without printing anything on top of anything.

    For each name a place is looked for that is clear of every dot and of the names written so
    far. If there is none, a place that only covers gray dots will do. If there is none of those
    either, the name is left out rather than printed over another name.
    """
    dots = dot_boxes(ax, points)
    gray_dots = dot_boxes(ax, other_points)
    written = []
    for point, name in zip(points, names):
        box = place_name(ax, point, name, dots + gray_dots + written)
        if box is None:
            box = place_name(ax, point, name, dots + written)
        if box is not None:
            written.append(box)


def draw_panel(ax, points, highlighted, title, limit):
    """One panel: every animal as a gray dot, the highlighted ones on top in the accent color."""
    dot = dict(s=DOT**2, edgecolors=SURFACE, linewidths=RING)  # s is the dot's area
    ax.scatter(points[~highlighted, 0], points[~highlighted, 1], facecolors=REST, zorder=2, **dot)
    ax.scatter(points[highlighted, 0], points[highlighted, 1], facecolors=ACCENT, zorder=3, **dot)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
    # The same scale in both directions, so that distances in the picture are true distances.
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_box_aspect(1)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)  # the grid stays behind the dots
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.75)


def write_around_panels(figure, attribute, season_note, run_name, animals, share):
    """The title, the explanation, the key to the two dot colors and the axis labels."""
    figure.text(0.10, 0.972, f"Do animals with the same {attribute}{season_note} sit together?",
                fontsize=14, fontweight="bold", color=INK, va="top")
    figure.text(0.10, 0.939,
                f"Run {run_name}: the word-table rows of the {animals:,} animals in use.\n"
                f"They are seen from the two directions in which the rows differ most. Every "
                f"panel shows the same dots.",
                fontsize=9.5, color=INK_SECONDARY, va="top", linespacing=1.4)
    dot = dict(marker="o", linestyle="", markersize=DOT, markeredgecolor=SURFACE)
    keys = [Line2D([], [], color=ACCENT, **dot), Line2D([], [], color=REST, **dot)]
    key_words = [f"animals with the {attribute} named above the panel", "all other animals"]
    figure.legend(keys, key_words, loc="upper left", bbox_to_anchor=(0.10, 0.880), ncol=2,
                  frameon=False, fontsize=9, labelcolor=INK_SECONDARY, borderpad=0,
                  borderaxespad=0, handlelength=1, handletextpad=0.4, columnspacing=2)
    figure.supxlabel(f"direction 1: {share[0]:.0%} of the differences between rows",
                     fontsize=9, color=INK_MUTED, y=0.012)
    figure.supylabel(f"direction 2: {share[1]:.0%} of the differences between rows",
                     fontsize=9, color=INK_MUTED, x=0.015)


def draw_picture(points, share, values, value_names, names, attribute, season_note, run_name,
                 path, name_the_dots):
    """Four small panels of the same dots; each highlights the animals with one value.

    points [animals, 2] and share come from two_directions; values [animals] is the number of
    each animal's true value for the attribute; names are the animal names.
    """
    figure, panels = plt.subplots(2, 2, figsize=(8.4, 9.9), sharex=True, sharey=True)
    figure.set_facecolor(SURFACE)
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.065, top=0.824, wspace=0.10,
                           hspace=0.16)
    limit = 1.15 * np.abs(points).max()
    if limit == 0:
        limit = 1.0
    for value, ax in enumerate(panels.flat):
        highlighted = values == value
        count = int(highlighted.sum())
        title = f"{value_names[value]}: {count:,} animal{'' if count == 1 else 's'}"
        draw_panel(ax, points, highlighted, title, limit)
    write_around_panels(figure, attribute, season_note, run_name, len(points), share)
    if name_the_dots:
        figure.canvas.draw()  # settles where everything is, so that the names can be fitted in
        for value, ax in enumerate(panels.flat):
            highlighted = values == value
            some_names = [names[animal] for animal in np.flatnonzero(highlighted)]
            name_dots(ax, points[highlighted], some_names, points[~highlighted])
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def look_at_picture(rows, language, run_name, config):
    """Part 3. Returns the path of the saved picture."""
    print("3. A PICTURE")
    animals, width = rows.shape
    if width < 2:
        say("No picture. A row of this model is a single number, and a single number cannot be "
            "laid out in two directions.")
        return None
    attribute = ATTRIBUTE_NAMES.index(config.attribute.split()[0])  # "waking time" works too
    # With several seasons an animal has, say, one home per season. The picture shows the first.
    season_note = f" in {language.seasons_in_use[0]}" if language.config.seasons > 1 else ""
    points, share = two_directions(rows)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    picture = results_dir / f"probe_{run_name}_{ATTRIBUTE_NAMES[attribute]}.png"
    draw_picture(points, share, language.lexicon[:animals, 0, attribute], VALUES[attribute],
                 language.animal_names, ATTRIBUTES[attribute], season_note, run_name, picture,
                 name_the_dots=animals <= config.label_up_to)
    first, second = round(100 * share[0]), round(100 * share[1])
    say(f"A row has {width} numbers, too many to draw. The picture shows the rows from the two "
        f"directions in which they differ most. These hold {first}% and {second}% of the "
        f"differences between rows; the other {100 - first - second}% cannot be seen in it. Each "
        f"of its four panels highlights the animals with one {ATTRIBUTES[attribute]}{season_note}.")
    print(f"  Saved {picture}")
    return picture


# ------------------------------------------------------------------------------------------------
# The three looks, one after the other
# ------------------------------------------------------------------------------------------------


def print_run(run_dir, run_config, rows):
    print(f"PROBE of {run_dir}")
    print(f"  {len(rows):,} animals in use; each is one row of {rows.shape[1]} numbers in the "
          f"word table (width {run_config.width}).")
    summary_file = Path(run_dir) / "summary.json"
    if summary_file.exists():
        with open(summary_file) as file:
            summary = json.load(file)
        ended = summary["stop_reason"]  # solved, plateau, budget or fixed (see train.py)
        if ended == "fixed":
            ended = "its fixed number of sets"
        print(f"  The run ended by {ended} with facts known "
              f"{summary['final']['facts_known']:.1f} out of 100.")
        if ended == "budget":
            say("It is NOT FINISHED: what follows is the word table as it stood when the time "
                "ran out, not what a model of this size can do.")
    if run_config.usage == "zipf":
        say("Every animal counts once in the three looks below, however rarely it is used. "
            "Under zipf usage most animals are rare and were hardly practiced, and they are the "
            "many here: low numbers can come from that alone, even when the few common animals "
            "are stored well. (report.py separates common from rare animals: chart 4.)")
    print()


def probe(run_dir, config=None):
    """Take the three looks at one finished run. Prints, saves the picture, and returns the
    numbers as a dictionary: read (None when there are too few animals), neighbors, picture."""
    config = config or ProbeConfig()
    run_config, language, model = load_run(run_dir)
    rows = animal_rows(language, model)
    print_run(run_dir, run_config, rows)
    if len(rows) < 3:
        say("With fewer than 3 animals in use there is nothing to compare. Nothing was done.")
        return {"read": None, "neighbors": None, "picture": None}
    read = look_at_reading(rows, language, config)
    print()
    neighbors = look_at_neighbors(rows, language, config)
    print()
    picture = look_at_picture(rows, language, run_config.name, config)
    return {"read": read, "neighbors": neighbors, "picture": picture}


def main(arguments=None):
    defaults = ProbeConfig()
    parser = argparse.ArgumentParser(description="Look inside the word table of a finished run.")
    parser.add_argument("run_dir",
                        help="the directory of a finished run, for example runs/a16_w32_equal_s0")
    parser.add_argument("--results-dir", default=defaults.results_dir,
                        help="the picture is saved here")
    parser.add_argument("--attribute", choices=ATTRIBUTE_NAMES, default=defaults.attribute,
                        help="the attribute the picture is about")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="decides the random halves of part 1 and the shuffles of parts 1 "
                             "and 2")
    parser.add_argument("--splits", type=int, default=defaults.splits,
                        help="part 1 draws a fresh random half this many times and averages")
    parser.add_argument("--shuffles", type=int, default=defaults.shuffles,
                        help="how often the rows are handed to the wrong animals, to measure "
                             "how much of a result is luck")
    parser.add_argument("--min-animals", type=int, default=defaults.min_animals,
                        help="part 1 is skipped for fewer animals in use")
    parser.add_argument("--neighbors", type=int, default=defaults.neighbors,
                        help="nearest rows listed for each animal in part 2")
    parser.add_argument("--shown", type=int, default=defaults.shown,
                        help="how many animals part 2 lists")
    parser.add_argument("--label-up-to", type=int, default=defaults.label_up_to,
                        help="the picture names its highlighted dots only up to this many animals")
    args = parser.parse_args(arguments)
    if args.splits < 1 or args.shuffles < 1:
        parser.error("--splits and --shuffles must be at least 1")
    if args.seed < 0:
        parser.error("--seed must not be negative")
    if not (Path(args.run_dir) / "model.pt").exists():
        parser.error(f"there is no trained model in {args.run_dir} (no model.pt). Train one "
                     f"first, for example with: .venv/bin/python train.py")
    try:
        probe(args.run_dir, ProbeConfig(
            results_dir=args.results_dir, attribute=args.attribute, seed=args.seed,
            splits=args.splits, shuffles=args.shuffles, min_animals=args.min_animals,
            neighbors=args.neighbors, shown=args.shown, label_up_to=args.label_up_to,
        ))
    except ValueError as problem:  # saved settings this code cannot rebuild a language from
        parser.error(f"the run in {args.run_dir} cannot be opened: {problem}")


if __name__ == "__main__":
    main()

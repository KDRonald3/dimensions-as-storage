"""
Tests of probe.py (look inside the word table) and ask.py (type a prompt, see the next word).

Run them with:    pytest tests/test_probe_ask.py

Nothing is trained here beyond a handful of steps: the runs below read one set of 256 sentences,
which teaches a model nothing. They exist so that the two tools are tried on a run directory
exactly as train.py writes it. Everything is written to temporary directories.

The arithmetic of probe.py is checked in two ways. Small cases are worked out by hand. And the
three looks are tried on word tables built by hand in which the facts ARE readable: each row is
made by adding one fixed random vector for each of the animal's facts. On such a table a linear
rule must read the facts almost perfectly and neighbors must share more facts than random pairs.
On rows that hold nothing, both must come out at chance.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import ask
import probe
from language import Language, LanguageConfig
from probe import ProbeConfig
from score import model_predictor
from train import RunConfig, load_run, run

REPO = Path(__file__).resolve().parent.parent
SECTIONS = ["1. CAN A FACT BE READ FROM A ROW BY A SIMPLE RULE?",
            "2. DO WORDS WITH SIMILAR MEANINGS SIT CLOSE TOGETHER?",
            "3. A PICTURE"]


# ------------------------------------------------------------------------------------------------
# Helpers and shared runs
# ------------------------------------------------------------------------------------------------


def tiny_run(out_dir, **changes):
    """A run of a handful of steps (one set of 256 sentences). Returns its directory.

    256 is a multiple of 16 and of 64, the numbers of animals used below: with equal usage the
    set size has to be a multiple of the number of animals.
    """
    settings = dict(animals=16, table_animals=32, set_sentences=256, width=8, batch=8, context=32,
                    max_sets=1, threads=1, out_dir=str(out_dir))
    settings.update(changes)
    before = torch.get_num_threads()
    summary = run(RunConfig(**settings))
    torch.set_num_threads(before)  # a run sets the number of threads; put it back
    return Path(out_dir) / summary["name"]


def readable_rows(lexicon, width, seed=0, noise=0.05):
    """Hand-built rows in which every fact can be read by a linear rule.

    Each value of each fact (red, brown, ..., night; per season) gets one fixed random vector.
    An animal's row is the sum of the vectors of its facts, plus a little noise.
    """
    rng = np.random.default_rng(seed)
    seasons = lexicon.shape[1]
    vectors = rng.normal(size=(seasons, 4, 4, width))  # [season, attribute, value, width]
    rows = np.zeros((len(lexicon), width))
    for season in range(seasons):
        for attribute in range(4):
            rows += vectors[season, attribute, lexicon[:, season, attribute]]
    return rows + noise * rng.normal(size=rows.shape)


def plant_rows(run_dir, noise=0.05):
    """Replace the saved rows of the animals in use by hand-built readable ones."""
    saved = torch.load(run_dir / "model.pt", weights_only=True)
    config, language, _ = load_run(run_dir)
    rows = readable_rows(language.lexicon[: config.animals], config.width, noise=noise)
    ids = torch.as_tensor(language.animal_ids)
    saved["model"]["table.weight"][ids] = torch.tensor(rows, dtype=torch.float32)
    torch.save(saved, run_dir / "model.pt")


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """16 animals in use (of 32 in the table), width 8."""
    return tiny_run(tmp_path_factory.mktemp("small"))


@pytest.fixture(scope="module")
def larger_run(tmp_path_factory):
    """64 animals in use, width 16: enough animals for part 1 of probe.py."""
    return tiny_run(tmp_path_factory.mktemp("larger"), animals=64, table_animals=64, width=16)


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    """Like larger_run, but with the rows of the animals replaced by readable ones."""
    run_dir = tiny_run(tmp_path_factory.mktemp("planted"), animals=64, table_animals=64, width=16)
    plant_rows(run_dir)
    return run_dir


@pytest.fixture(scope="module")
def seasons_run(tmp_path_factory):
    """16 animals with four seasons: 16 facts per animal."""
    return tiny_run(tmp_path_factory.mktemp("seasons"), seasons=4)


@pytest.fixture(scope="module", autouse=True)
def nothing_is_left_in_the_repository():
    """These tests must not leave a picture in the repository's own results/ directory."""
    before = sorted(REPO.glob("results/probe_*"))
    yield
    assert sorted(REPO.glob("results/probe_*")) == before


def results_in(tmp_path):
    return ProbeConfig(results_dir=str(tmp_path / "results"))


# ------------------------------------------------------------------------------------------------
# probe.py, part 1: reading a fact with a linear rule
# ------------------------------------------------------------------------------------------------


def test_linear_rule_on_a_case_worked_out_by_hand():
    # One number per row. Value 1 goes with the number 1, value 0 with the number 0. The best
    # weighted sums are then: score of value 0 = 1 - number, score of value 1 = number.
    rows = np.array([[0.0], [0.0], [1.0], [1.0]])
    values = np.array([0, 0, 1, 1])
    weights = probe.fit_linear_rule(rows, values, n_values=2)
    assert weights.shape == (2, 2)  # [one weight for the number + the offset, two values]
    assert np.allclose(weights, [[-1, 1], [1, 0]])
    assert probe.apply_linear_rule(weights, np.array([[0.1], [0.9], [0.4]])).tolist() == [0, 1, 0]


def test_chance_is_counted_from_the_facts_as_they_are():
    assert probe.chance_level(np.array([0, 1, 2, 3, 0, 1, 2, 3]), 4) == 0.25
    assert probe.chance_level(np.array([2, 2, 2, 1]), 4) == 0.75  # an unbalanced table


def test_a_linear_rule_reads_facts_that_are_readable():
    lexicon = Language(LanguageConfig(animals=128, table_animals=128)).lexicon
    rows = readable_rows(lexicon, width=16)
    results = probe.read_facts(rows, lexicon, np.random.default_rng(0), splits=5)
    assert [result["attribute"] for result in results] == ["color", "food", "home", "waking time"]
    for result in results:
        assert result["held_out"] > 0.95
        assert result["shuffled"] < 0.45  # the same rows handed to the wrong animals: nothing
        assert result["chance"] == 0.25


def test_a_linear_rule_reads_nothing_from_rows_that_hold_nothing():
    lexicon = Language(LanguageConfig(animals=128, table_animals=128)).lexicon
    rows = np.random.default_rng(1).normal(size=(128, 16))
    for result in probe.read_facts(rows, lexicon, np.random.default_rng(0), splits=5):
        assert result["held_out"] < 0.45
        assert result["shuffled"] < 0.45


def test_the_rule_is_tested_only_on_animals_it_was_not_fitted_on():
    # Every animal has a row of its own that says nothing about its facts: all zeros and a single
    # 1 in its own place. A rule can learn such rows by rote and would be right about every
    # animal it was fitted on. About the held-out animals it can know nothing.
    animals = 64
    values = np.arange(animals) % 4
    rows = np.eye(animals)
    weights = probe.fit_linear_rule(rows, values, 4)
    assert (probe.apply_linear_rule(weights, rows) == values).all()  # by rote: all right
    held_out = probe.held_out_accuracy(rows, values, 4, np.random.default_rng(0), splits=10)
    assert held_out < 0.45


def test_the_same_seed_gives_the_same_numbers():
    lexicon = Language(LanguageConfig(animals=64, table_animals=64)).lexicon
    rows = readable_rows(lexicon, width=8, noise=1.0)
    first = probe.read_facts(rows, lexicon, np.random.default_rng(3), splits=3)
    again = probe.read_facts(rows, lexicon, np.random.default_rng(3), splits=3)
    assert first == again


# ------------------------------------------------------------------------------------------------
# probe.py, part 2: neighbors
# ------------------------------------------------------------------------------------------------


def test_shared_facts_of_the_base_animals():
    language = Language(LanguageConfig(animals=16, table_animals=16))
    shared = probe.shared_facts(language.lexicon)
    number = {name: i for i, name in enumerate(language.animal_names)}
    assert shared.shape == (16, 16)
    assert (shared == shared.T).all()
    assert (np.diag(shared) == 4).all()  # an animal shares all four facts with itself
    assert shared[number["frogs"], number["toads"]] == 4  # the same facts in the base table
    assert shared[number["foxes"], number["deer"]] == 3  # red, woods, dusk; not the food
    assert shared[number["foxes"], number["bears"]] == 0


def test_shared_facts_count_every_season():
    language = Language(LanguageConfig(animals=16, table_animals=16, seasons=4))
    shared = probe.shared_facts(language.lexicon)
    assert (np.diag(shared) == 16).all()
    one_season = probe.shared_facts(language.lexicon[:, :1])
    assert (shared >= one_season).all()


def test_cosine_similarity_on_a_case_worked_out_by_hand():
    rows = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # their average is zero
    similarity = probe.cosine_similarity(rows)
    assert np.allclose(np.diag(similarity), 1)
    assert np.isclose(similarity[0, 1], -1)  # opposite directions
    assert np.isclose(similarity[0, 2], 0)  # at right angles: unrelated
    # What all rows have in common is taken out first, and the length of a row does not count.
    assert np.allclose(probe.cosine_similarity(rows + [10.0, -3.0]), similarity)
    assert np.allclose(probe.cosine_similarity(7.0 * rows), similarity)
    # Only the direction counts: a row three times as long points the same way.
    longer = np.array([[3.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]) + [-0.5, 0.0]
    assert np.isclose(probe.cosine_similarity(longer)[0, 1], -1)


def test_nearest_lists_the_most_similar_other_animals():
    similarity = np.array([[1.0, 0.9, 0.1],
                           [0.9, 1.0, 0.5],
                           [0.1, 0.5, 1.0]])
    assert probe.nearest(similarity, 2).tolist() == [[1, 2], [0, 2], [1, 0]]
    assert probe.nearest(similarity, 1).tolist() == [[1], [0], [1]]
    assert similarity[0, 0] == 1.0  # the table handed in is not changed


def test_neighbor_summary_on_a_case_worked_out_by_hand():
    # Animals 0 and 1 share all four facts, 2 and 3 share three, 0 and 3 (and 1 and 3) share one.
    lexicon = np.array([[[0, 0, 0, 0]], [[0, 0, 0, 0]], [[1, 1, 1, 1]], [[1, 1, 1, 0]]])
    shared = probe.shared_facts(lexicon)
    by_meaning = np.array([[1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1]])  # 0 by 1, 2 by 3
    summary = probe.neighbor_summary(probe.cosine_similarity(by_meaning), shared)
    assert summary["nearest"] == pytest.approx((4 + 4 + 3 + 3) / 4)
    assert summary["random"] == pytest.approx(18 / 12)  # all twelve ordered pairs
    assert summary["best"] == pytest.approx(3.5)
    mixed_up = np.array([[1.0, 0.1], [-1.0, 0.1], [1.0, -0.1], [-1.0, -0.1]])  # 0 by 2, 1 by 3
    summary = probe.neighbor_summary(probe.cosine_similarity(mixed_up), shared)
    assert summary["nearest"] == pytest.approx((0 + 1 + 0 + 1) / 4)
    assert summary["best"] == pytest.approx(3.5)  # does not depend on the rows


def test_neighbors_share_more_facts_when_rows_are_built_from_the_facts():
    lexicon = Language(LanguageConfig(animals=128, table_animals=128)).lexicon
    shared = probe.shared_facts(lexicon)
    built = probe.neighbor_summary(probe.cosine_similarity(readable_rows(lexicon, 16)), shared)
    assert built["nearest"] > built["random"] + 1.0
    assert built["nearest"] <= built["best"]
    noise = np.random.default_rng(1).normal(size=(128, 16))
    empty = probe.neighbor_summary(probe.cosine_similarity(noise), shared)
    assert abs(empty["nearest"] - empty["random"]) < 0.4
    assert empty["random"] == pytest.approx(built["random"])  # a property of the facts alone


def test_the_neighbors_come_with_a_yardstick_for_luck():
    """With 16 animals "nearest" strays from "random" by a quarter of a fact on rows of pure
    noise. The same number with the rows handed to the wrong animals shows how much is luck."""
    small = Language(LanguageConfig(animals=16, table_animals=16)).lexicon
    shared = probe.shared_facts(small)
    apart = []
    for seed in range(40):
        noise = np.random.default_rng(seed).normal(size=(16, 8))
        similarity = probe.cosine_similarity(noise)
        mean, spread = probe.neighbors_by_luck(similarity, shared, np.random.default_rng(0), 50)
        assert mean == pytest.approx(0.8, abs=0.15) and 0.15 < spread < 0.45
        apart.append(abs(probe.neighbor_summary(similarity, shared)["nearest"] - mean) / spread)
    assert np.mean(np.array(apart) < 2) > 0.85  # rows that hold nothing stay inside the yardstick

    # Rows built from the facts stand far outside it.
    lexicon = Language(LanguageConfig(animals=128, table_animals=128)).lexicon
    shared = probe.shared_facts(lexicon)
    similarity = probe.cosine_similarity(readable_rows(lexicon, 16))
    mean, spread = probe.neighbors_by_luck(similarity, shared, np.random.default_rng(0), 20)
    assert probe.neighbor_summary(similarity, shared)["nearest"] > mean + 5 * spread
    # The same seed gives the same yardstick.
    assert (mean, spread) == probe.neighbors_by_luck(similarity, shared,
                                                     np.random.default_rng(0), 20)


def test_the_shuffled_rows_come_with_their_spread():
    lexicon = Language(LanguageConfig(animals=64, table_animals=64)).lexicon
    rows = np.random.default_rng(1).normal(size=(64, 4))
    mean, spread = probe.shuffled_accuracy(rows, lexicon[:, 0, 0], 4, np.random.default_rng(0),
                                           splits=20, shuffles=30)
    # With 64 animals the test strays by a few points from one shuffle to the next, and it
    # lands a little below chance (0.25): that is the level to read a result against.
    assert 0.18 < mean < 0.26 and 0.015 < spread < 0.07
    readable = readable_rows(lexicon, width=8)
    results = probe.read_facts(readable, lexicon, np.random.default_rng(0), splits=5, shuffles=10)
    for result in results:
        assert result["held_out"] > result["shuffled"] + 5 * result["shuffled_spread"]


# ------------------------------------------------------------------------------------------------
# probe.py, part 3: the picture
# ------------------------------------------------------------------------------------------------


def test_two_directions_show_a_flat_table_without_loss():
    # Rows that all lie in one plane of a five-dimensional space: two directions show everything.
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(20, 2)) @ rng.normal(size=(2, 5)) + rng.normal(size=5)
    points, share = probe.two_directions(rows)
    assert points.shape == (20, 2)
    assert share.sum() == pytest.approx(1.0)
    assert share[0] >= share[1] > 0
    assert np.allclose(points.mean(axis=0), 0)
    # Distances between dots are the distances between rows.
    in_the_picture = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    in_the_table = np.linalg.norm(rows[:, None] - rows[None, :], axis=2)
    assert np.allclose(in_the_picture, in_the_table)


def test_two_directions_say_how_much_they_leave_out():
    rows = np.random.default_rng(0).normal(size=(200, 16))
    _, share = probe.two_directions(rows)
    assert 0 < share.sum() < 0.5  # sixteen equally spread directions: two of them show little
    _, share = probe.two_directions(np.ones((5, 4)))  # all rows the same: nothing to show
    assert share.tolist() == [0, 0]


def test_names_do_not_cover_each_other_and_stay_inside_the_panel():
    figure, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    figure.canvas.draw()
    # Five dots almost on top of each other, and one at the right edge of the panel.
    points = np.array([[0.0, 0.0], [0.02, 0.0], [0.0, 0.02], [0.02, 0.02], [0.01, 0.01],
                       [0.97, -0.5]])
    names = ["foxes", "bears", "wolves", "owls", "bats", "frogs"]
    probe.name_dots(ax, points, names, other_points=np.array([[-0.5, -0.5]]))
    renderer = figure.canvas.get_renderer()
    panel = ax.get_window_extent(renderer)
    boxes = [text.get_window_extent(renderer) for text in ax.texts]
    written = [text.get_text() for text in ax.texts]
    plt.close(figure)
    assert 2 <= len(boxes) < len(names)  # some of the crowded names are left out
    assert "frogs" in written
    for i, box in enumerate(boxes):
        assert panel.x0 <= box.x0 and box.x1 <= panel.x1
        assert panel.y0 <= box.y0 and box.y1 <= panel.y1
        assert not any(box.overlaps(other) for other in boxes[i + 1 :])
    # There is no room right of the dot at the edge, so its name stands left of it.
    frogs = ax.texts[written.index("frogs")]
    assert frogs.get_horizontalalignment() == "right"


def draw_test_picture(path, animals, name_the_dots):
    rng = np.random.default_rng(0)
    values = np.arange(animals) % 4
    names = [f"animal{i}" for i in range(animals)]
    probe.draw_picture(rng.normal(size=(animals, 2)), np.array([0.6, 0.3]), values,
                       ["red", "brown", "green", "black"], names, "color", "", "a_test_run",
                       path, name_the_dots)


def red_green_blue(color):
    """ "#2a78d6" -> [42, 120, 214] """
    return [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]


def test_the_picture_is_a_real_image_in_the_agreed_colors(tmp_path):
    open_before = plt.get_fignums()
    path = tmp_path / "picture.png"
    for animals, name_the_dots in [(16, True), (300, False)]:
        draw_test_picture(path, animals, name_the_dots)
        image = matplotlib.image.imread(path)
        assert image.shape[:2] == (1485, 1260)  # 9.9 x 8.4 inches at 150 dots per inch
        pixels = (image[:, :, :3] * 255).round().astype(int)
        assert pixels[0, 0].tolist() == red_green_blue(probe.SURFACE)  # the light background
        assert (pixels == red_green_blue(probe.ACCENT)).all(axis=2).any()  # highlighted dots
        assert (pixels == red_green_blue(probe.REST)).all(axis=2).any()  # the other dots
    assert plt.get_fignums() == open_before  # the figure was closed again


def test_dots_are_named_only_for_few_animals(monkeypatch, small_run, larger_run, tmp_path):
    named = []
    monkeypatch.setattr(probe, "name_dots", lambda ax, points, names, others: named.extend(names))
    probe.probe(small_run, results_in(tmp_path))
    language = load_run(small_run)[1]
    assert sorted(named) == sorted(language.animal_names[:16])  # every animal once, in its panel
    named.clear()
    probe.probe(larger_run, results_in(tmp_path))  # 64 animals: too many names to read
    assert named == []


# ------------------------------------------------------------------------------------------------
# probe.py, end to end
# ------------------------------------------------------------------------------------------------


def test_probe_runs_end_to_end_and_writes_its_picture(larger_run, tmp_path, capsys):
    result = probe.probe(larger_run, results_in(tmp_path))
    output = capsys.readouterr().out
    for section in SECTIONS:
        assert section in output
    assert "64 animals in use" in output
    assert "The run ended by budget" in output  # taken from the run's summary.json
    # Like every other tool, probe.py says that such a run is not a measure of the model.
    assert "It is NOT FINISHED: what follows is the word table as it stood" in output
    assert result["picture"] == tmp_path / "results" / f"probe_{larger_run.name}_color.png"
    assert matplotlib.image.imread(result["picture"]).shape[:2] == (1485, 1260)
    assert f"Saved {result['picture']}" in output
    # A model that has read 256 sentences knows nothing: the rule reads nothing from its rows.
    assert len(result["read"]) == 4
    for line in result["read"]:
        assert line["held_out"] < 0.5
        assert line["chance"] == 0.25
        assert f"{100 * line['held_out']:.1f}%" in output
        # The yardstick for luck stands beside every number: the shuffled test and its spread.
        assert 0 < line["shuffled_spread"] < 0.15
        assert (f"{100 * line['shuffled']:.1f}% +/-{100 * line['shuffled_spread']:5.1f}"
                in output)
        # A model that knows nothing reads like shuffled rows: within three spreads of them.
        assert abs(line["held_out"] - line["shuffled"]) < 3 * line["shuffled_spread"] + 0.02
    assert "at least twice the +/- above the shuffled one shows nothing" in " ".join(output.split())
    assert set(result["neighbors"]) == {"nearest", "random", "best", "shuffled",
                                        "shuffled_spread"}
    assert "rows handed to wrong animals" in output and "(what luck alone gives)" in output
    assert "zipf" not in output  # this run has equal usage: nothing to say about rare animals


def test_probe_says_how_a_run_of_a_fixed_length_ended(tmp_path, capsys):
    # A run trained with --fixed-sets is saved with the stop reason "fixed" (see train.py).
    run_dir = tiny_run(tmp_path, fixed_sets=1)
    assert run_dir.name == "a16_w8_equal_t32_s0_f1"
    config, language, model = load_run(run_dir)
    probe.print_run(run_dir, config, probe.animal_rows(language, model))
    output = capsys.readouterr().out
    assert "The run ended by its fixed number of sets with facts known" in output
    assert "NOT FINISHED" not in output  # that is said of runs that ended by budget only


def test_probe_says_that_every_animal_counts_once_under_zipf_usage(tmp_path, capsys):
    run_dir = tiny_run(tmp_path, usage="zipf")
    config, language, model = load_run(run_dir)
    probe.print_run(run_dir, config, probe.animal_rows(language, model))
    output = " ".join(capsys.readouterr().out.split())
    assert "Every animal counts once in the three looks below, however rarely it is used." in output
    assert "Under zipf usage most animals are rare" in output


def test_probe_looks_at_the_rows_of_the_animals_in_use(small_run):
    config, language, model = load_run(small_run)
    rows = probe.animal_rows(language, model)
    assert rows.shape == (16, 8)  # 16 of the 32 animals in the table are in use
    table = model.table.weight.detach().numpy()
    assert np.allclose(rows[0], table[language.word_id["foxes"]])
    assert np.allclose(rows[15], table[language.word_id["lizards"]])


def test_probe_explains_itself_when_there_are_fewer_than_64_animals(small_run, tmp_path, capsys):
    result = probe.probe(small_run, results_in(tmp_path))
    output = " ".join(capsys.readouterr().out.split())  # the explanation is wrapped over lines
    assert "Skipped." in output
    assert "With 16 animals in use only 8 would be left to test on" in output
    assert "luck alone moves the result by many points" in output
    assert "The check is made from 64 animals in use on" in output
    assert ".venv/bin/python train.py --animals 64" in output  # the interpreter that has torch
    assert "Even there much is luck" in output  # 64 animals are not "enough" either
    assert result["read"] is None
    # The other two looks do not need many animals.
    assert result["neighbors"]["random"] == pytest.approx(0.8)  # 16 balanced animals
    assert result["picture"].exists()
    assert "foxes" in output and "shared)" in output


def test_probe_finds_what_was_planted_in_a_saved_run(planted_run, tmp_path, capsys):
    result = probe.probe(planted_run, results_in(tmp_path))
    output = capsys.readouterr().out
    for line in result["read"]:
        assert line["held_out"] > 0.9
        assert line["shuffled"] < 0.5
    assert "100.0%" in output
    assert result["neighbors"]["nearest"] > result["neighbors"]["random"] + 1.0
    assert result["neighbors"]["nearest"] <= result["neighbors"]["best"]


def test_probe_warns_when_the_rule_could_fit_its_half_by_rote(larger_run, tmp_path, capsys):
    probe.probe(larger_run, results_in(tmp_path))  # width 16: 17 numbers, 32 animals to fit on
    assert "Caution" not in capsys.readouterr().out
    wide = tiny_run(tmp_path, animals=64, table_animals=64, width=32)  # 33 numbers, 32 animals
    probe.probe(wide, results_in(tmp_path))
    output = " ".join(capsys.readouterr().out.split())
    assert "Caution: the rule has 33 adjustable numbers per value" in output


def test_probe_with_four_seasons(seasons_run, tmp_path, capsys):
    config = ProbeConfig(results_dir=str(tmp_path / "results"), attribute="home", min_animals=16)
    result = probe.probe(seasons_run, config)
    output = " ".join(capsys.readouterr().out.split())
    assert "how many of the 16 facts the two share" in output
    assert "Only the facts of the first season, winter, are tested here" in output
    assert "one home in winter" in output
    assert result["picture"].name == f"probe_{seasons_run.name}_home.png"
    assert result["picture"].exists()


def test_probe_changes_nothing_in_the_run(small_run, tmp_path):
    before = {path.name: path.read_bytes() for path in small_run.iterdir()}
    probe.probe(small_run, results_in(tmp_path))
    assert {path.name: path.read_bytes() for path in small_run.iterdir()} == before


def test_probe_flags(small_run, tmp_path, capsys):
    results = tmp_path / "elsewhere"
    probe.main([str(small_run), "--results-dir", str(results), "--attribute", "waking",
                "--min-animals", "16", "--splits", "2", "--neighbors", "2", "--shown", "3",
                "--seed", "1", "--label-up-to", "0"])
    output = capsys.readouterr().out
    assert (results / f"probe_{small_run.name}_waking.png").exists()
    assert "averages over 2 random halves" in " ".join(output.split())
    assert "the 2 animals with the most similar rows" in " ".join(output.split())
    listed = [line for line in output.splitlines() if "shared)" in line]
    assert len(listed) == 3 and listed[0].split()[0] == "foxes"
    assert "one waking time" in " ".join(output.split())


def test_probe_without_a_run_says_what_is_missing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        probe.main([str(tmp_path / "no_such_run")])
    complaint = capsys.readouterr().err
    assert "no model.pt" in complaint and ".venv/bin/python train.py" in complaint
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("flags, complaint", [
    (["--seed", "-1"], "--seed must not be negative"),
    (["--splits", "0"], "--splits and --shuffles must be at least 1"),
    (["--shuffles", "0"], "--splits and --shuffles must be at least 1"),
])
def test_probe_refuses_flags_that_make_no_sense(small_run, tmp_path, capsys, flags, complaint):
    with pytest.raises(SystemExit) as stopped:
        probe.main([str(small_run), "--results-dir", str(tmp_path / "results"), *flags])
    assert stopped.value.code == 2 and complaint in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_probe_of_a_model_with_one_number_per_word_draws_no_picture(tmp_path, capsys):
    """The smallest possible model: width 1. Its rows cannot be laid out in two directions."""
    run_dir = tiny_run(tmp_path, width=1, heads=1)
    result = probe.probe(run_dir, results_in(tmp_path))
    output = " ".join(capsys.readouterr().out.split())
    assert "No picture. A row of this model is a single number" in output
    assert result["picture"] is None and result["neighbors"] is not None
    assert not (tmp_path / "results").exists()


def test_probe_from_the_command_line(small_run, tmp_path):
    # Started in an empty directory: the picture must land in ./results there, nowhere else.
    done = subprocess.run([sys.executable, str(REPO / "probe.py"), str(small_run)],
                          cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr
    for section in SECTIONS:
        assert section in done.stdout
    assert [path.name for path in (tmp_path / "results").iterdir()] == [
        f"probe_{small_run.name}_color.png"]


# ------------------------------------------------------------------------------------------------
# ask.py
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded(small_run):
    """The small run, loaded: (config, language, predict)."""
    config, language, model = load_run(small_run)
    return config, language, model_predictor(model)


def test_ask_answers_a_prompt_and_shows_the_oracle_beside_the_model(small_run, capsys):
    ask.main([str(small_run), "do foxes eat"])
    output = capsys.readouterr().out
    assert 'After "do foxes eat" the next word is:' in output
    lines = {line.split()[0]: line.split() for line in output.splitlines() if line.strip()}
    assert lines["next"] == ["next", "word", "model", "oracle"]
    # Half the questions show the true value (foxes eat meat), the rest one of the three others.
    assert lines["meat"][2] == "50.0%"
    for word in ["plants", "insects", "seeds"]:
        assert lines[word][2] == "16.7%"
    assert lines["(all"][3] == "0.0%"  # the oracle gives nothing to any other word
    assert "The oracle is not certain here: 4 different words can come next" in output


def test_ask_prints_the_models_own_probabilities(loaded, capsys):
    config, language, predict = loaded
    ids = language.encode("foxes live in")
    rows = ask.answer(language, predict, config.context, "foxes live in", top=5)
    output = capsys.readouterr().out
    model = np.exp(predict(np.array([ids]))[0, -1].astype(np.float64))
    assert model.sum() == pytest.approx(1.0, abs=1e-5)
    oracle = language.next_distribution(ids)
    for word, model_share, oracle_share in rows:
        token = language.word_id[word]
        assert model_share == pytest.approx(model[token])
        assert oracle_share == pytest.approx(oracle[token])
    # Listed: the model's five most likely words, and the oracle's one word (woods) if it is not
    # among them. The model's most likely word comes first.
    shown = [word for word, _, _ in rows]
    top_five = [language.words[token] for token in np.argsort(-model)[:5]]
    assert set(shown) == set(top_five) | {"woods"}
    assert len(shown) == len(set(shown))
    assert shown[0] == top_five[0]
    model_shares = [model_share for _, model_share, _ in rows]
    assert model_shares == sorted(model_shares, reverse=True)
    assert rows[shown.index("woods")][2] == 1.0
    assert 'The oracle is certain: the next word is "woods".' in output


def test_ask_words_to_show():
    model = np.array([0.5, 0.3, 0.1, 0.06, 0.04])
    oracle = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    assert ask.words_to_show(model, oracle, top=2) == [0, 1, 3]  # the oracle's word is added
    assert ask.words_to_show(model, None, top=2) == [0, 1]
    assert ask.words_to_show(model, model, top=3) == [0, 1, 2]  # nothing is listed twice


def test_ask_reading_says_whether_the_model_agrees():
    language = Language(LanguageConfig(animals=16, table_animals=16))
    ids = language.encode("foxes eat")
    oracle = language.next_distribution(ids)
    assert 'the next word is "meat". The model\'s first choice is the same.' in ask.reading(
        language, oracle, oracle)
    wrong = np.zeros(language.vocab_size)
    wrong[language.word_id["seeds"]] = 1.0
    assert 'The model\'s first choice is "seeds": wrong.' in ask.reading(language, wrong, oracle)
    open_choice = language.next_distribution(language.encode("foxes"))
    assert "not certain" in ask.reading(language, wrong, open_choice)
    assert ask.reading(language, wrong, None) == 'The model\'s first choice is "seeds".'


def test_ask_handles_an_unknown_word(small_run, capsys):
    ask.main([str(small_run), "do cats eat"])
    output = " ".join(capsys.readouterr().out.split())
    assert '"cats" is not a word of this language.' in output
    assert "Some words it knows: foxes bears wolves" in output  # a few valid words are listed
    assert "After " not in output  # no table for a prompt that cannot be read


def test_ask_explains_an_animal_that_is_in_the_table_but_not_in_use(loaded, capsys):
    config, language, predict = loaded
    stranger = language.animal_names[20]  # the run uses the first 16 of 32 animals
    rows = ask.answer(language, predict, config.context, f"{stranger} eat", top=3)
    output = " ".join(capsys.readouterr().out.split())
    assert f'"{stranger}" has a row in the word table, but this run\'s text never used it' in output
    assert len(rows) == 3  # the model still answers
    assert all(oracle_share is None for _, _, oracle_share in rows)  # the oracle cannot


def test_ask_explains_a_prompt_that_no_sentence_starts_with(loaded, seasons_run, capsys):
    config, language, predict = loaded
    ask.answer(language, predict, config.context, "eat eat", top=3)
    output = " ".join(capsys.readouterr().out.split())
    assert 'The oracle has no answer: no sentence of this language starts with "eat eat".' in output
    assert 'Try for example "foxes eat".' in output
    # With seasons every sentence starts with "in SEASON"; the hint shows how.
    ask.main([str(seasons_run), "foxes eat"])
    output = " ".join(capsys.readouterr().out.split())
    assert 'Try for example "in winter foxes eat".' in output
    ask.main([str(seasons_run), "in winter foxes eat"])
    assert 'The oracle is certain: the next word is "meat".' in capsys.readouterr().out


def test_ask_example_prompts_are_valid(small_run, seasons_run):
    for run_dir in [small_run, seasons_run]:
        language = load_run(run_dir)[1]
        examples = ask.example_prompts(language)
        assert len(examples) == 2
        for example in examples:
            assert language.next_distribution(language.encode(example)).sum() == pytest.approx(1)


def test_ask_refuses_an_empty_or_too_long_prompt_politely(loaded, capsys):
    config, language, predict = loaded
    assert ask.answer(language, predict, config.context, "   ", top=5) == []
    assert "Type at least one word" in capsys.readouterr().out
    too_long = " ".join(["foxes eat meat ."] * 9)  # 36 words; this model reads 32 in one row
    assert ask.answer(language, predict, config.context, too_long, top=5) == []
    assert "This prompt has 36 words; the model reads at most 32" in capsys.readouterr().out


def test_ask_reads_prompts_in_a_loop_until_an_empty_line(small_run, monkeypatch, capsys):
    typed = iter(["foxes eat", "do cats eat", "Are Bears brown ?", "", "never reached"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    ask.main([str(small_run)])
    output = " ".join(capsys.readouterr().out.split())
    assert 'for example "foxes eat" or "are foxes red ?"' in output
    assert 'After "foxes eat" the next word is:' in output
    assert '"cats" is not a word of this language.' in output  # the loop goes on after a mistake
    assert 'After "are bears brown ?" the next word is:' in output  # capitals are fine
    assert next(typed) == "never reached"  # the empty line ended the loop


def test_ask_loop_ends_quietly_when_the_input_ends(small_run, monkeypatch, capsys):
    def no_more_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_more_input)
    ask.main([str(small_run)])
    assert "Traceback" not in capsys.readouterr().out


def test_ask_says_when_a_run_was_not_finished(small_run, tmp_path, capsys):
    ask.main([str(small_run), "foxes eat"])  # the tiny run ended by budget after one set
    output = " ".join(capsys.readouterr().out.split())
    assert "This run ended by budget: it is NOT FINISHED." in output
    # A run that did not end by budget, and a run without a summary: nothing is said.
    fixed = tiny_run(tmp_path, fixed_sets=1)
    ask.main([str(fixed), "foxes eat"])
    assert "NOT FINISHED" not in capsys.readouterr().out
    (fixed / "summary.json").unlink()
    ask.main([str(fixed), "foxes eat"])
    assert "NOT FINISHED" not in capsys.readouterr().out


@pytest.mark.parametrize("top", ["0", "-1"])
def test_ask_refuses_to_show_fewer_than_one_word(small_run, capsys, top):
    with pytest.raises(SystemExit) as stopped:
        ask.main([str(small_run), "foxes eat", "--top", top])
    assert stopped.value.code == 2
    assert "--top must be at least 1" in capsys.readouterr().err


def test_ask_and_probe_open_a_run_whose_set_size_is_refused_today(small_run, tmp_path, capsys):
    """A set size that suited the code when the run was saved, and is refused for new runs
    now (250 sentences for 16 animals), must not make a finished run unreadable."""
    old = tmp_path / "old"
    old.mkdir()
    saved = torch.load(small_run / "model.pt", weights_only=True)
    saved["config"]["set_sentences"] = 250
    torch.save(saved, old / "model.pt")
    ask.main([str(old), "foxes eat"])
    assert 'After "foxes eat" the next word is:' in capsys.readouterr().out
    result = probe.probe(old, results_in(tmp_path))
    assert result["neighbors"] is not None

    # Settings that no language can be built from at all end in a sentence, not in a traceback.
    saved["config"]["animals"] = 64  # more animals in use than the table of 32 holds
    torch.save(saved, old / "model.pt")
    for tool in (ask, probe):
        with pytest.raises(SystemExit) as stopped:
            tool.main([str(old)])
        assert stopped.value.code == 2
        assert "cannot be opened: animals must be between 1 and table_animals" in \
            capsys.readouterr().err


def test_ask_without_a_run_says_what_is_missing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        ask.main([str(tmp_path / "no_such_run"), "foxes eat"])
    assert "no model.pt" in capsys.readouterr().err


def test_ask_from_the_command_line(small_run, tmp_path):
    # The words of the prompt may also be given without quotes.
    done = subprocess.run([sys.executable, str(REPO / "ask.py"), str(small_run), "do", "foxes",
                           "eat", "--top", "3"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr
    assert 'After "do foxes eat" the next word is:' in done.stdout
    assert "oracle" in done.stdout
    assert list(tmp_path.iterdir()) == []  # asking writes nothing

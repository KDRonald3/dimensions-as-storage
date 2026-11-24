"""
Tests of compare.py (one model on two vocabularies) and sweep.py (the named rounds).

Run them with:    pytest tests/test_compare_sweep.py

Most of these tests need no model at all: how two runs are read side by side is checked on
hand-made summaries, and which runs a round holds is checked with --dry-run. The few tests that
do train use runs of a few dozen steps on a tiny language and write only to a temporary
directory. Such runs learn nothing; they check the plumbing: that both runs are trained and
saved, that a finished run is not trained again, and that several runs at a time work.
"""

import dataclasses
import json
import os
import signal
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

import compare
import sweep
import train as T
from train import RunConfig, default_name, grading_sets

# Flags for a run of a few dozen steps: sets of 256 sentences, rows of 32 words, 8 rows a batch.
TINY_FLAGS = ["--table-animals", "64", "--set-sentences", "256", "--width", "8", "--batch", "8",
              "--context", "32", "--threads", "1"]

# The same idea as a round for sweep.py: 2 vocabularies x 2 widths = 4 runs of 2 sets each.
TINY_ROUND = {
    "about": "four very short runs, for the tests",
    "animals": [32, 16],
    "widths": [4, 8],
    "table_animals": [32],
    "seeds": [0],
    "settings": {"usage": "equal", "seasons": 1, "set_sentences": 256, "batch": 8,
                 "context": 32, "max_sets": 2},
}
TINY_NAMES = ["a16_w4_equal_t32_s0", "a16_w8_equal_t32_s0", "a32_w4_equal_t32_s0",
              "a32_w8_equal_t32_s0"]


@pytest.fixture(autouse=True)
def keep_thread_setting():
    """A run sets the number of processor threads; put it back so other tests are not slowed."""
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.fixture
def tiny_round(monkeypatch):
    """Add the round "tiny" to sweep.py for the length of one test."""
    monkeypatch.setitem(sweep.ROUNDS, "tiny", TINY_ROUND)


def one_line(text):
    """Printed text with every line break and run of spaces turned into one space, so a test
    can look for a sentence wherever the printout happened to wrap it."""
    return " ".join(text.split())


# ------------------------------------------------------------------------------------------------
# compare.py: reading two runs side by side, on hand-made summaries
# ------------------------------------------------------------------------------------------------


SCHEDULE = set(islice(grading_sets(), 30))  # 1, 2, 3, 4, 6, 8, 11, 16, ...


def made_up_run(animals, graded_at, stop_reason, floor, usage="equal"):
    """A summary such as train.run returns, with only the keys compare.py reads.

    The gap is 1 / sets and facts known is 100 - 50 / sets, so every grading has numbers of its
    own. One set is 61,440 sentences, as in the real experiment. A grading at a set count that
    is not on the schedule is marked as the extra one, as train.py does.
    """
    curve = []
    for sets in graded_at:
        curve.append({"sets": sets, "sightings": sets * 61440 / (animals * 4), "floor": floor,
                      "gap": 1 / sets, "loss": floor + 1 / sets, "facts_known": 100 - 50 / sets,
                      "scheduled": sets in SCHEDULE})
    return {
        "name": f"a{animals}_w32_{usage}_s0",
        "language": {"animals": animals, "table_animals": 1024, "usage": usage, "floor": floor},
        "parameters": {"body": 28544, "table": 33632, "total": 62176},
        "stop_reason": stop_reason,
        "sets": graded_at[-1],
        "curve": curve,
    }


# The usual case: the small vocabulary is solved early, the large one trains on to a plateau.
SMALL = made_up_run(16, [1, 2, 3, 4, 6], "solved", floor=1.2)
LARGE = made_up_run(1024, [1, 2, 3, 4, 6, 8, 11], "plateau", floor=2.591)


def two_tables(printed):
    """The printout cut in two: reading (1), and reading (2) up to the notes."""
    first, rest = printed.split("(2) AT THE END OF EACH RUN")
    second, notes = rest.split("What to notice")
    assert "(1) AT THE SAME NUMBER OF SETS" in first
    return first, second, notes


def test_the_large_run_is_read_at_the_set_count_where_the_small_run_ended():
    assert compare.last_shared_sets(SMALL, LARGE) == 6 == SMALL["sets"]
    point = compare.grading_at(LARGE, 6)
    assert point["sets"] == 6 and point["gap"] == pytest.approx(1 / 6)
    assert compare.grading_at(LARGE, 7) is None  # never graded there
    assert compare.grading_at(SMALL, 8) is None  # the small run had ended by then


def test_compare_prints_both_readings(capsys):
    compare.print_comparison(SMALL, LARGE)
    first, second, notes = two_tables(capsys.readouterr().out)

    # (1) Both runs after 6 sets. The large run stood at a gap of 1/6 and was not finished.
    assert "both models after 6 sets, so after the same amount of text" in first
    assert "This is where the 16-animal run ended." in first
    assert compare.table_line(SMALL, SMALL["curve"][-1]) in first
    assert compare.table_line(LARGE, compare.grading_at(LARGE, 6)) in first
    at_six = compare.table_line(LARGE, compare.grading_at(LARGE, 6))
    assert at_six.split() == ["1,024", "28,544", "33,632", "6", "90.0", "2.758", "2.591",
                              "0.167", "91.7", "still", "training"]
    # (2) Each run at its own end: the large one after 11 sets, by plateau.
    assert compare.table_line(SMALL, SMALL["curve"][-1]) in second
    at_end = compare.table_line(LARGE, LARGE["curve"][-1])
    assert at_end in second and at_end not in first
    assert at_end.split() == ["1,024", "28,544", "33,632", "11", "165.0", "2.682", "2.591",
                              "0.091", "95.5", "plateau"]
    assert compare.table_line(SMALL, SMALL["curve"][-1]).split()[-1] == "solved"


def test_compare_points_out_what_a_beginner_should_notice(capsys):
    compare.print_comparison(SMALL, LARGE)
    printed = one_line(capsys.readouterr().out)
    # The same number of parameters, with the numbers.
    assert "exactly the same number of parameters: 28,544 in the body + 33,632 in the word " \
           "table = 62,176" in printed
    # The raw loss cannot be compared: the best possible loss differs.
    assert "1.200 bits per word for 16 animals and 2.591 for 1,024" in printed
    assert "even a perfect model has a higher loss on the large vocabulary" in printed
    # Only facts known can be compared between vocabularies; the gap only at 0.
    assert "Only facts known can be compared between the two rows." in printed
    assert "The gap can be compared only at 0, which means perfect at every vocabulary" in printed
    assert "copying from earlier in the row without knowing anything" in printed
    assert "Read the gap instead" not in printed
    # At equal sets the large vocabulary had far less practice per fact: 5,760 against 90.
    assert "seen 5,760 times" in printed and "1,024-animal language 90 times" in printed
    assert "64 times less practice per fact" in printed
    # Nothing here is unfinished or zipf, so those remarks stay away.
    assert "NOT FINISHED" not in printed and "zipf" not in printed


def test_compare_gives_no_verdict(capsys):
    compare.print_comparison(SMALL, LARGE)
    printed = capsys.readouterr().out.lower()
    for word in ["hypothesis", "confirm", "refute", "proves", "supports", "size effect"]:
        assert word not in printed


def test_a_small_run_that_ended_off_the_grading_schedule(capsys):
    # The budget ended the small run after 5 sets, where no other run is ever graded.
    small = made_up_run(16, [1, 2, 3, 4, 5], "budget", floor=1.2)
    large = made_up_run(1024, [1, 2, 3, 4, 6, 8], "plateau", floor=2.591)
    assert compare.last_shared_sets(small, large) == 4
    compare.print_comparison(small, large)
    printed = capsys.readouterr().out
    first, second, notes = two_tables(printed)
    assert "both models after 4 sets" in first
    assert "ended after 5 sets" in one_line(first)
    assert "not a set count at which both runs had a scheduled grading" in one_line(first)
    assert "4 is the last set count at which both had one" in one_line(first)
    # In (1) neither run had ended; in (2) the small one is marked as not finished.
    assert first.count("still training") == 2
    assert compare.table_line(small, small["curve"][-1]).endswith("budget: NOT FINISHED")
    assert compare.table_line(small, small["curve"][-1]) in second
    assert "NOT FINISHED: the 16-animal run ended by budget." in one_line(notes)


def test_two_runs_are_lined_up_only_at_scheduled_gradings(capsys):
    # Both budgets ended after 5 sets, so both runs have an extra grading there. It is not a
    # scheduled one, and reading (1) is taken at set 4, the last grading that every run gets.
    small = made_up_run(16, [1, 2, 3, 4, 5], "budget", floor=1.2)
    large = made_up_run(1024, [1, 2, 3, 4, 5], "budget", floor=2.591)
    assert [point["scheduled"] for point in small["curve"]] == [True, True, True, True, False]
    assert compare.scheduled_sets(small) == {1, 2, 3, 4}
    assert compare.last_shared_sets(small, large) == 4
    compare.print_comparison(small, large)
    first, second, notes = two_tables(capsys.readouterr().out)
    assert "both models after 4 sets" in first
    assert compare.table_line(large, compare.grading_at(large, 4)) in first
    assert compare.table_line(large, large["curve"][-1]) in second  # the end is still shown


def test_a_large_run_that_ended_first(capsys):
    small = made_up_run(16, [1, 2, 3, 4, 6, 8], "solved", floor=1.2)
    large = made_up_run(1024, [1, 2, 3, 4], "budget", floor=2.591)
    assert compare.last_shared_sets(small, large) == 4
    compare.print_comparison(small, large)
    first, second, notes = two_tables(capsys.readouterr().out)
    assert compare.table_line(small, compare.grading_at(small, 4)).endswith("still training")
    assert compare.table_line(small, compare.grading_at(small, 4)) in first
    assert compare.table_line(large, large["curve"][-1]) in first
    assert "NOT FINISHED: the 1,024-animal run ended by budget." in one_line(notes)


def test_runs_that_ended_together_and_the_zipf_remark(capsys):
    small = made_up_run(16, [1, 2, 3], "budget", floor=1.2, usage="zipf")
    large = made_up_run(1024, [1, 2, 3], "budget", floor=2.591, usage="zipf")
    compare.print_comparison(small, large)
    first, second, notes = two_tables(capsys.readouterr().out)
    assert "Both runs ended after the same number of sets" in second
    assert "NOT FINISHED: the 16-animal run and the 1,024-animal run ended by budget." \
           in one_line(notes)
    assert "With zipf usage" in notes and "average" in notes


# ------------------------------------------------------------------------------------------------
# compare.py: two real (tiny) runs
# ------------------------------------------------------------------------------------------------


def saved_summary(run_dir):
    with open(run_dir / "summary.json") as file:
        return json.load(file)


def test_compare_trains_both_vocabularies_and_prints_both_readings(tmp_path, capsys):
    compare.main(["--small", "16", "--large", "64", *TINY_FLAGS, "--max-sets", "3",
                  "--out-dir", str(tmp_path)])
    printed = capsys.readouterr().out
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a16_w8_equal_t64_s0",
                                                               "a64_w8_equal_t64_s0"]
    small = saved_summary(tmp_path / "a16_w8_equal_t64_s0")
    large = saved_summary(tmp_path / "a64_w8_equal_t64_s0")
    assert (tmp_path / "a64_w8_equal_t64_s0" / "model.pt").exists()
    assert small["steps"] <= 100 and large["steps"] <= 100  # smoke checks, not experiments

    # The same model: the same parameters and settings. Only the animals in use differ.
    assert small["parameters"] == large["parameters"]
    for key in small["config"]:
        if key not in ("animals", "name"):
            assert small["config"][key] == large["config"][key], key
    assert (small["config"]["animals"], large["config"]["animals"]) == (16, 64)

    # The small vocabulary is trained first, and both readings are printed.
    assert printed.index("THE SMALL VOCABULARY: 16") < printed.index("THE LARGE VOCABULARY: 64")
    first, second, notes = two_tables(printed)
    assert "both models after 3 sets" in first  # three sets were the whole budget of both
    for summary in [small, large]:
        assert compare.table_line(summary, summary["curve"][-1]) in first
        assert compare.table_line(summary, summary["curve"][-1]) in second
    assert "exactly the same number of parameters" in one_line(notes)
    # Sets of 256 sentences: 3 sets show each of 64 facts 12 times and each of 256 facts 3 times.
    assert small["curve"][-1]["sightings"] == pytest.approx(3 * 256 / 64)
    assert large["curve"][-1]["sightings"] == pytest.approx(3 * 256 / 256)
    assert "4 times less practice per fact" in one_line(notes)


def test_compare_finds_the_large_runs_grading_where_the_small_run_ended(monkeypatch, tmp_path,
                                                                       capsys):
    # Let the real grading run, then pretend that the 16-animal model knows everything. Its run
    # is then "solved" at the second grading, while the 64-animal run goes on to its budget.
    real_score = T.score

    def steered_score(predict, language, eval_rows):
        result = real_score(predict, language, eval_rows)
        if language.config.animals == 16:
            result["facts_known"] = 100.0
        return result

    monkeypatch.setattr(T, "score", steered_score)
    config = RunConfig(table_animals=64, set_sentences=256, width=8, batch=8, context=32,
                       max_sets=5, threads=1, out_dir=str(tmp_path), name="mine")
    small, large = compare.compare(config, small=16, large=64)
    assert (small["name"], large["name"]) == ("mine_a16", "mine_a64")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["mine_a16", "mine_a64"]
    assert (small["stop_reason"], small["sets"]) == ("solved", 2)
    assert (large["stop_reason"], large["sets"]) == ("budget", 5)
    assert [point["sets"] for point in large["curve"]] == [1, 2, 3, 4, 5]

    capsys.readouterr()  # forget what the two runs printed
    assert compare.last_shared_sets(small, large) == 2
    compare.print_comparison(small, large)
    first, second, notes = two_tables(capsys.readouterr().out)
    at_two = compare.table_line(large, compare.grading_at(large, 2))
    at_end = compare.table_line(large, large["curve"][-1])
    assert at_two.split()[3] == "2" and at_two.endswith("still training")
    assert at_end.split()[3] == "5" and at_end.endswith("budget: NOT FINISHED")
    assert at_two in first and at_end not in first
    assert at_end in second and at_two not in second
    # The grading at two sets is the large run's own second grading, taken from its curve.
    assert f"{large['curve'][1]['loss']:7.3f}" in at_two
    assert large["curve"][1]["sightings"] == pytest.approx(2 * 256 / 256)


def summaries_written_at(folder):
    """When each summary.json in a folder was last written, by run name."""
    return {path.parent.name: path.stat().st_mtime_ns for path in folder.glob("*/summary.json")}


def test_compare_uses_a_run_that_is_already_saved_with_the_same_settings(monkeypatch, tmp_path,
                                                                        capsys):
    flags = ["--small", "16", "--large", "64", *TINY_FLAGS, "--max-sets", "3",
             "--out-dir", str(tmp_path)]
    compare.main(flags)
    capsys.readouterr()
    written = summaries_written_at(tmp_path)

    def must_not_train(config):
        raise AssertionError(f"{default_name(config)} is saved and must not be trained again")

    monkeypatch.setattr(compare, "run", must_not_train)
    compare.main(flags)
    printed = capsys.readouterr().out
    assert one_line(printed).count("Already trained with these settings: the run saved in") == 2
    assert "is used as it is. (--again trains it again.) It ended by BUDGET." in one_line(printed)
    first, second, notes = two_tables(printed)  # and both readings are printed as before
    assert "both models after 3 sets" in first
    assert summaries_written_at(tmp_path) == written


def test_compare_does_not_replace_the_runs_of_a_round(tiny_round, monkeypatch, tmp_path, capsys):
    """With its default names compare.py writes where a round of sweep.py writes. A round's
    runs of 3 sets must not be replaced by a quick comparison of 2 sets, or by another recipe."""
    monkeypatch.setitem(sweep.ROUNDS["tiny"], "settings",
                        {**TINY_ROUND["settings"], "max_sets": 3})
    sweep.main(["tiny", "--threads", "1", "--widths", "8", "--out-dir", str(tmp_path)])
    written = summaries_written_at(tmp_path)
    assert sorted(written) == [TINY_NAMES[1], TINY_NAMES[3]]
    flags = ["--small", "16", "--large", "32", "--table-animals", "32", "--set-sentences", "256",
             "--width", "8", "--batch", "8", "--context", "32", "--out-dir", str(tmp_path)]
    capsys.readouterr()

    # A shorter budget: the saved runs went further than this command would, and are used.
    compare.main([*flags, "--max-sets", "2"])
    assert summaries_written_at(tmp_path) == written
    assert "Already trained with these settings" in capsys.readouterr().out

    # Another recipe: refused in a sentence, before anything is trained.
    with pytest.raises(SystemExit) as stopped:
        compare.main([*flags, "--max-sets", "3", "--learning-rate", "0.02"])
    assert stopped.value.code == 2
    complaint = one_line(capsys.readouterr().err)
    assert "already holds a run, and it was trained with other settings" in complaint
    assert "learning_rate 0.01 (asked now: 0.02)" in complaint
    assert "--again to train this run again and replace the saved one" in complaint
    assert summaries_written_at(tmp_path) == written

    # --again says: yes, replace them.
    compare.main([*flags, "--max-sets", "2", "--again"])
    assert all(summaries_written_at(tmp_path)[name] > written[name] for name in written)
    assert saved_summary(tmp_path / TINY_NAMES[1])["sets"] == 2


@pytest.mark.parametrize("flags, complaint", [
    (["--animals", "64"], "give --small and --large instead of --animals"),
    (["--small", "64", "--large", "16"], "--small (64) must be at least 1 and less than"),
    (["--small", "64", "--large", "64"], "less than --large (64)"),
    (["--large", "128"], "more animals than the word table holds"),
    (["--small", "16", "--large", "32", "--width", "5"], "width"),  # 5 is not divisible by 2 heads
    # A set size that suits the small vocabulary but not the large one: refused before the small
    # run is trained, not after.
    (["--small", "16", "--large", "64", "--set-sentences", "240"],
     "must be a multiple of the number of animals (64)"),
])
def test_compare_refuses_settings_that_make_no_sense(tmp_path, capsys, flags, complaint):
    with pytest.raises(SystemExit) as stopped:
        compare.main([*TINY_FLAGS, "--max-sets", "2", "--out-dir", str(tmp_path), *flags])
    assert stopped.value.code == 2
    assert complaint in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []  # nothing was trained or saved


# ------------------------------------------------------------------------------------------------
# sweep.py: which runs a round holds
# ------------------------------------------------------------------------------------------------

ROUND_SIZES = {"quick": 4, "round1-equal": 48, "round1-zipf": 48, "table-check": 12, "scaleup": 48}


def dials(round_name):
    """The values that occur in a round, for each setting the rounds differ in."""
    configs = sweep.planned_runs(round_name)
    return {key: sorted({getattr(config, key) for config in configs})
            for key in ["animals", "width", "usage", "seasons", "table_animals", "seed"]}


def test_the_agreed_rounds_exist():
    assert list(sweep.ROUNDS) == list(ROUND_SIZES)


@pytest.mark.parametrize("round_name, size", ROUND_SIZES.items())
def test_dry_run_lists_every_run_of_a_round_and_trains_nothing(tmp_path, capsys, round_name, size):
    out_dir = tmp_path / "runs"
    sweep.main([round_name, "--dry-run", "--out-dir", str(out_dir)])
    printed = capsys.readouterr().out
    names = [default_name(config) for config in sweep.planned_runs(round_name)]
    assert len(names) == size and len(set(names)) == size  # no two runs share a directory
    assert printed.count("to train") == size
    assert all(f" {name} " in printed for name in names)
    assert f"{size} runs, 1 at a time" in printed
    assert "Dry run: nothing was trained." in printed
    assert list(tmp_path.iterdir()) == []  # not even the output directory was made


def test_the_rounds_hold_the_agreed_runs():
    main_dials = {"animals": [16, 64, 256, 1024], "width": [4, 8, 16, 32], "seasons": [1],
                  "table_animals": [1024], "seed": [0, 1, 2]}
    assert dials("quick") == {"animals": [16, 1024], "width": [4, 32], "usage": ["equal"],
                              "seasons": [1], "table_animals": [1024], "seed": [0]}
    assert dials("round1-equal") == {**main_dials, "usage": ["equal"]}
    assert dials("round1-zipf") == {**main_dials, "usage": ["zipf"]}
    assert dials("table-check") == {"animals": [16], "width": [4, 32], "usage": ["equal"],
                                    "seasons": [1], "table_animals": [16, 1024],
                                    "seed": [0, 1, 2]}
    assert dials("scaleup") == {"animals": [16, 64, 256, 1024], "width": [8, 16, 32, 64],
                                "usage": ["equal"], "seasons": [4], "table_animals": [1024],
                                "seed": [0, 1, 2]}
    # Everything else is train.py's default: the rounds change nothing behind the user's back.
    # The one exception is the set size of "table-check", which is written out (see below).
    for round_name in ROUND_SIZES:
        for config in sweep.planned_runs(round_name):
            same = dataclasses.replace(RunConfig(), animals=config.animals, width=config.width,
                                       usage=config.usage, seasons=config.seasons,
                                       table_animals=config.table_animals, seed=config.seed)
            if round_name == "table-check":
                same = dataclasses.replace(same, set_sentences=61440)
            assert config == same


def test_both_tables_of_the_table_check_round_read_the_same_text():
    # The automatic set size follows the table: 960 sentences for 16 animals, 61,440 for 1,024.
    # With that, the two halves of the round would differ in far more than the table.
    assert T.build(RunConfig(table_animals=16))[0].set_sentences == 960
    assert T.build(RunConfig(table_animals=1024))[0].set_sentences == 61440

    # So the round writes the set size out, and then both halves read exactly the same rows, in
    # training and in grading. Only the word table differs.
    by_table = {config.table_animals: config for config in sweep.planned_runs("table-check")
                if (config.width, config.seed) == (4, 1)}
    assert sorted(by_table) == [16, 1024]
    small_language, small_model = T.build(by_table[16])
    large_language, large_model = T.build(by_table[1024])
    assert small_language.set_sentences == large_language.set_sentences == 61440
    for set_index in [0, 3]:
        rows = [language.pack_rows(language.make_set(set_index, data_seed=1), 96)
                for language in (small_language, large_language)]
        assert rows[0][0].shape[0] > 3000  # several thousand rows: 53 steps of 64 rows per set
        assert all(np.array_equal(small, large) for small, large in zip(*rows))
    graded_on = [T.evaluation_rows(language, config) for language, config in
                 [(small_language, by_table[16]), (large_language, by_table[1024])]]
    assert all(np.array_equal(small, large) for small, large in zip(*graded_on))
    assert small_model.table.weight.shape[0] == 43 and large_model.table.weight.shape[0] == 1051


@pytest.mark.parametrize("round_name", [*ROUND_SIZES, "tiny"])
def test_small_vocabularies_come_first(tiny_round, round_name):
    animals = [config.animals for config in sweep.planned_runs(round_name)]
    assert animals == sorted(animals)


def test_the_four_quick_runs_are_part_of_the_main_round():
    quick = {default_name(config) for config in sweep.planned_runs("quick")}
    main_round = {default_name(config) for config in sweep.planned_runs("round1-equal")}
    assert quick < main_round


def test_the_flags_reach_every_run(tmp_path, capsys):
    for config in sweep.planned_runs("scaleup", max_minutes=3, threads=5, out_dir="elsewhere"):
        assert (config.max_minutes, config.threads, config.out_dir) == (3, 5, "elsewhere")
    assert sweep.planned_runs("quick")[0].max_minutes == RunConfig.max_minutes == 20.0

    def heading(*flags):
        sweep.main(["quick", "--dry-run", "--out-dir", str(tmp_path / "runs"), *flags])
        return capsys.readouterr().out.splitlines()[1]

    # Threads: 8 for one run at a time, 2 each for several, unless --threads says otherwise.
    assert "1 at a time, 8 threads each, a budget of 20 min each" in heading()
    assert "3 at a time, 2 threads each" in heading("--jobs", "3")
    assert "3 at a time, 5 threads each" in heading("--jobs", "3", "--threads", "5")
    assert "a budget of 1.5 min each" in heading("--max-minutes", "1.5")
    assert "1 thread each" in heading("--threads", "1")
    assert (sweep.JOBS, sweep.THREADS_ONE_JOB, sweep.THREADS_MANY_JOBS) == (1, 8, 2)


def test_without_a_round_the_rounds_are_listed(capsys):
    sweep.main([])
    printed = capsys.readouterr().out
    for round_name, size in ROUND_SIZES.items():
        assert any(line.split()[:3] == [round_name, str(size), "runs"]
                   for line in printed.splitlines())


def test_an_unknown_round_and_a_bad_job_count_are_refused(capsys):
    with pytest.raises(SystemExit):
        sweep.main(["round2"])
    with pytest.raises(SystemExit):
        sweep.main(["quick", "--dry-run", "--jobs", "0"])
    assert "--jobs must be at least 1" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------
# sweep.py: part of a round (--widths, --seeds, --animals)
# ------------------------------------------------------------------------------------------------


def test_part_of_a_round_can_be_picked_out(tmp_path, capsys):
    whole = sweep.planned_runs("round1-equal")
    part = sweep.only(whole, widths=[8, 16, 32])
    assert len(part) == 36 and {config.width for config in part} == {8, 16, 32}
    assert part == [config for config in whole if config.width != 4]  # in the round's order
    one = sweep.only(whole, widths=[32], seeds=[0], animals=[16])
    assert [default_name(config) for config in one] == ["a16_w32_equal_s0"]
    assert sweep.only(whole) == whole

    sweep.main(["round1-equal", "--dry-run", "--widths", "8", "16", "32", "--out-dir",
                str(tmp_path / "runs")])
    printed = capsys.readouterr().out
    assert "36 runs, 1 at a time" in printed and printed.count("to train") == 36
    assert "_w4_" not in printed


@pytest.mark.parametrize("flags", [["--widths", "8"], ["--seeds", "7"],
                                   ["--widths", "4", "--animals", "64"]])
def test_a_choice_that_matches_no_run_is_refused_in_a_sentence(tmp_path, capsys, flags):
    with pytest.raises(SystemExit) as stopped:
        sweep.main(["quick", "--dry-run", "--out-dir", str(tmp_path / "runs"), *flags])
    assert stopped.value.code == 2
    complaint = one_line(capsys.readouterr().err)
    assert "no run of the round quick has these widths, seeds and animals." in complaint
    assert "The round has --widths 4 32, --seeds 0 and --animals 16 1024." in complaint
    with pytest.raises(ValueError, match="no run of the round quick"):
        sweep.run_round("quick", dry_run=True, widths=[8])


def test_a_part_of_a_round_is_all_that_is_trained_and_counted(tiny_round, tmp_path, capsys):
    sweep.main(["tiny", "--threads", "1", "--widths", "4", "--out-dir", str(tmp_path)])
    printed = capsys.readouterr().out  # no SystemExit: both runs that were asked for are there
    assert sorted(path.name for path in tmp_path.iterdir()) == [TINY_NAMES[0], TINY_NAMES[2]]
    assert "ROUND tiny: 2 of 2 runs are finished." in printed
    # While the larger vocabulary is still to come, the estimate says that it is likely too low.
    estimates = [line for line in printed.splitlines() if "left: about" in line]
    assert len(estimates) == 1 and "likely too low" in estimates[0]


def test_a_shorter_budget_than_the_saved_runs_of_a_round_had_is_pointed_out(tiny_round, tmp_path,
                                                                             capsys):
    # The width-8 half of the round was trained with 30 minutes for each run.
    for config in sweep.only(sweep.planned_runs("tiny", max_minutes=30, out_dir=str(tmp_path)),
                             widths=[8]):
        pretend_finished(config)
    assert sweep.longer_budget_note("tiny", 30, str(tmp_path)) is None
    assert sweep.longer_budget_note("tiny", 45, str(tmp_path)) is None
    # Now the width-4 half, with the default budget of 20 minutes: no saved run is among the
    # runs chosen, and still the difference must be said.
    sweep.main(["tiny", "--dry-run", "--widths", "4", "--out-dir", str(tmp_path)])
    printed = one_line(capsys.readouterr().out)
    assert ("NOTE: runs of this round were saved with a budget of 30 min each (for example "
            "a16_w8_equal_t32_s0); this command gives 20.") in printed
    assert "add --max-minutes 30." in printed


# ------------------------------------------------------------------------------------------------
# sweep.py: the time left
# ------------------------------------------------------------------------------------------------


def test_time_left_is_an_estimate_and_a_ceiling():
    assert sweep.minutes_text(0.42) == "0.4 min" and sweep.minutes_text(12.4) == "12 min"
    assert sweep.minutes_text(210) == "3.5 hours"
    assert sweep.count(1, "run") == "1 run" and sweep.count(48, "run") == "48 runs"
    assert sweep.time_left_text(0, 5, 10.0, jobs=1, max_minutes=20) == "no runs left"
    # Nothing trained yet: only the ceiling, 10 runs of at most 20 minutes.
    before = sweep.time_left_text(10, 0, 0.0, jobs=1, max_minutes=20)
    assert before.startswith("10 runs left: at most about 3.3 hours")
    # 5 runs took 10 minutes between them, so 10 more would take about 20.
    after = sweep.time_left_text(10, 5, 10.0, jobs=1, max_minutes=20)
    assert after.startswith("10 runs left: about 20 min if they take as long as the runs so far")
    assert "at most about 3.3 hours" in after
    # Two at a time: half as long, for the estimate and for the ceiling.
    two = sweep.time_left_text(10, 5, 10.0, jobs=2, max_minutes=20)
    assert "about 10 min if" in two and "at most about 1.7 hours" in two
    # One run left takes a whole run, however many jobs there are.
    assert "at most about 20 min" in sweep.time_left_text(1, 0, 0.0, jobs=4, max_minutes=20)
    # The estimate never goes above the ceiling.
    assert "about 40 min if" in sweep.time_left_text(2, 1, 30.0, jobs=1, max_minutes=20)
    # Small vocabularies are trained first. While larger ones are still to come, the estimate
    # from the runs so far is likely too low, and the sentence says so.
    assert "likely too low" not in after
    early = sweep.time_left_text(10, 5, 10.0, jobs=1, max_minutes=20, larger_to_come=True)
    assert ("about 20 min if they take as long as the runs so far (likely too low: the runs so "
            "far were on the small vocabularies, which come first), at most about 3.3 "
            "hours") in early


# ------------------------------------------------------------------------------------------------
# sweep.py: finished runs are skipped
# ------------------------------------------------------------------------------------------------


def pretend_finished(config, stop_reason="solved", **saved_with):
    """Write the summary.json of a run that never took place, as train.py would have saved it.

    saved_with: settings that were different when the pretended run was trained.
    """
    name = default_name(config)
    summary = {
        "name": name,
        "config": {**dataclasses.asdict(config), "name": name, **saved_with},
        "stop_reason": stop_reason,
        "sets": 6,
        "minutes": 0.5,
        "final": {"facts_known": 100.0, "gap": 0.01},
    }
    run_dir = Path(config.out_dir) / name
    run_dir.mkdir(parents=True)
    with open(run_dir / "summary.json", "w") as file:
        json.dump(summary, file)
    return summary


def never_train(monkeypatch):
    def refuse(config):
        raise AssertionError(f"{default_name(config)} is finished and must not be trained again")

    monkeypatch.setattr(sweep, "run", refuse)


def test_sweep_trains_a_round_and_skips_what_is_finished(tiny_round, monkeypatch, tmp_path,
                                                         capsys):
    flags = ["tiny", "--threads", "1", "--out-dir", str(tmp_path)]
    sweep.main(flags)
    printed = capsys.readouterr().out
    assert sorted(path.name for path in tmp_path.iterdir()) == TINY_NAMES
    assert printed.count("to train") == 4 and printed.count("RESULT") == 4
    assert printed.count("left: about") == 3 and "1 run left: about" in printed
    assert printed.count("no runs left") == 1
    assert "ROUND tiny: 4 of 4 runs are finished." in printed
    assert f"Next: .venv/bin/python report.py --runs-dir {tmp_path}" in printed
    for name in TINY_NAMES:
        summary = saved_summary(tmp_path / name)
        assert summary["stop_reason"] == "budget" and summary["sets"] == 2
        assert summary["steps"] <= 100 and summary["config"]["threads"] == 1
    # The order of training: small vocabularies first, as listed.
    trained = [line.split()[1] for line in printed.splitlines() if line.startswith("RESULT")]
    assert trained == TINY_NAMES

    # The same command again: every run is finished, so nothing may be trained.
    written = {name: (tmp_path / name / "summary.json").stat().st_mtime_ns for name in TINY_NAMES}
    never_train(monkeypatch)
    sweep.main(flags)
    printed = capsys.readouterr().out
    assert printed.count("already done") == 5  # four runs and the line that counts them
    assert "4 already done. no runs left." in printed
    assert "to train" not in printed and "RESULT" not in printed and "NOTE" not in printed
    assert "ROUND tiny: 4 of 4 runs are finished." in printed
    for name in TINY_NAMES:
        assert (tmp_path / name / "summary.json").stat().st_mtime_ns == written[name]


def test_an_interrupted_round_is_resumed_where_it_stopped(tiny_round, monkeypatch, tmp_path,
                                                         capsys):
    # Two of the four runs were finished before the interruption.
    configs = sweep.planned_runs("tiny", threads=1, out_dir=str(tmp_path))
    assert [default_name(config) for config in configs] == TINY_NAMES
    pretend_finished(configs[0])
    pretend_finished(configs[2], stop_reason="budget")
    asked_for = []
    real_run = sweep.run

    def recording_run(config):
        asked_for.append(default_name(config))
        return real_run(config)

    monkeypatch.setattr(sweep, "run", recording_run)
    finished = sweep.run_round("tiny", threads=1, out_dir=str(tmp_path))
    printed = capsys.readouterr().out
    assert asked_for == [TINY_NAMES[1], TINY_NAMES[3]]
    assert sorted(finished) == TINY_NAMES
    assert "2 already done. 2 runs left" in printed
    # A saved run that ended by budget is shown as not finished, and still skipped.
    skipped = [line for line in printed.splitlines() if "already done  " in line]
    assert len(skipped) == 2 and skipped[1].endswith("budget: NOT FINISHED")


def test_dry_run_shows_which_runs_are_finished(tiny_round, monkeypatch, tmp_path, capsys):
    configs = sweep.planned_runs("tiny", out_dir=str(tmp_path))
    pretend_finished(configs[1])
    never_train(monkeypatch)
    sweep.main(["tiny", "--dry-run", "--out-dir", str(tmp_path)])
    printed = capsys.readouterr().out
    assert printed.count("to train") == 3 and printed.count("already done  ") == 1
    assert "1 already done. 3 runs left: at most about 60 min" in printed
    assert sorted(path.name for path in tmp_path.iterdir()) == [TINY_NAMES[1]]


def test_a_summary_that_was_cut_off_counts_as_not_finished(tmp_path):
    config = RunConfig(out_dir=str(tmp_path))
    assert sweep.load_finished(config) is None
    run_dir = tmp_path / default_name(config)
    run_dir.mkdir()
    (run_dir / "summary.json").write_text('{"name": "a16_w32_eq')
    assert sweep.load_finished(config) is None
    (run_dir / "summary.json").write_text('{"name": "a16_w32_equal_s0"}')
    assert sweep.load_finished(config) == {"name": "a16_w32_equal_s0"}


def test_a_saved_run_with_other_settings_is_pointed_out(tiny_round, monkeypatch, tmp_path, capsys):
    config = RunConfig(out_dir=str(tmp_path), max_minutes=20)
    same = pretend_finished(config)
    assert sweep.settings_that_differ(same, config) == []
    # Where a run was saved and how many threads it had do not make it another run.
    moved = dataclasses.replace(config, out_dir="elsewhere", threads=2)
    assert sweep.settings_that_differ(same, moved) == []
    # Another learning rate does: the name is the same, the run is not.
    other_rate = pretend_finished(dataclasses.replace(config, seed=1), learning_rate=0.002)
    assert sweep.settings_that_differ(other_rate, dataclasses.replace(config, seed=1)) == [
        "learning_rate 0.002 (this round: 0.01)"]
    # A shorter budget matters only if it was the budget that ended the run.
    solved_early = pretend_finished(dataclasses.replace(config, seed=2), max_minutes=2.0)
    assert sweep.settings_that_differ(solved_early, dataclasses.replace(config, seed=2)) == []
    cut_short = pretend_finished(dataclasses.replace(config, seed=3), "budget", max_minutes=2.0)
    assert sweep.settings_that_differ(cut_short, dataclasses.replace(config, seed=3)) == [
        "max_minutes 2.0 (this round: 20)"]
    # ... and only if this round would give the run more: a run that got 60 minutes went further.
    long_run = pretend_finished(dataclasses.replace(config, seed=4), "budget", max_minutes=60.0)
    assert sweep.settings_that_differ(long_run, dataclasses.replace(config, seed=4)) == []
    # A set size that is written out is the same as the automatic size it equals: a run saved
    # with 0 (automatic, 61,440 for the 1,024-animal table) IS the run that asks for 61,440.
    written_out = dataclasses.replace(config, set_sentences=61440)
    assert sweep.settings_that_differ(same, written_out) == []
    assert sweep.settings_that_differ(same, dataclasses.replace(config, set_sentences=30720)) == [
        "set_sentences 0 (this round: 30720)"]
    # With a 16-animal table the automatic size is 960, so there 0 and 61,440 are different runs.
    small_table = dataclasses.replace(config, table_animals=16)
    saved_small = pretend_finished(small_table)
    assert sweep.settings_that_differ(
        saved_small, dataclasses.replace(small_table, set_sentences=61440)) == [
        "set_sentences 0 (this round: 61440)"]

    # A run saved by older code, before a setting existed: that setting cannot be compared, and
    # the note says so. (Such a run may have been ended by another stopping rule.)
    older = pretend_finished(dataclasses.replace(config, seed=5))
    del older["config"]["min_steps"], older["config"]["average"]
    assert sweep.settings_that_differ(older, dataclasses.replace(config, seed=5)) == [
        "average not recorded (saved by older code; this round: 0.998)",
        "min_steps not recorded (saved by older code; this round: 2000)"]

    # In a round, such a run is skipped all the same, with a note that says what differs.
    configs = sweep.planned_runs("tiny", out_dir=str(tmp_path / "round"))
    pretend_finished(configs[0], batch=64)
    never_train(monkeypatch)
    sweep.main(["tiny", "--dry-run", "--out-dir", str(tmp_path / "round")])
    printed = one_line(capsys.readouterr().out)
    assert "NOTE: this saved run was trained with batch 64 (this round: 8)." in printed
    assert f"Delete {tmp_path / 'round' / TINY_NAMES[0]} to train it again." in printed


def test_a_run_that_breaks_down_does_not_stop_the_round(tiny_round, monkeypatch, tmp_path,
                                                       capsys):
    def breaks_at_width_8(config):
        if config.width == 8:
            raise RuntimeError("Training broke down around step 7.")
        return pretend_finished(config)

    monkeypatch.setattr(sweep, "run", breaks_at_width_8)
    with pytest.raises(SystemExit) as stopped:
        sweep.main(["tiny", "--out-dir", str(tmp_path)])
    assert stopped.value.code == 1  # tells a calling script that runs are missing
    printed = capsys.readouterr().out
    assert printed.count("RESULT") == 2
    assert printed.count("FAILED  a") == 2 and "Training broke down around step 7." in printed
    assert "ROUND tiny: 2 of 4 runs are finished." in printed
    assert sorted(path.name for path in tmp_path.iterdir()) == [TINY_NAMES[0], TINY_NAMES[2]]


@pytest.mark.parametrize("error", [ValueError, OSError, KeyError])
def test_any_error_in_one_run_leaves_the_rest_of_the_round_alone(tiny_round, monkeypatch,
                                                                tmp_path, capsys, error):
    def breaks_at_the_second_run(config):
        if default_name(config) == TINY_NAMES[1]:
            raise error("no room left on the disk")
        return pretend_finished(config)

    monkeypatch.setattr(sweep, "run", breaks_at_the_second_run)
    with pytest.raises(SystemExit) as stopped:
        sweep.main(["tiny", "--out-dir", str(tmp_path)])
    assert stopped.value.code == 1  # a run is missing, and the exit code says so
    printed = capsys.readouterr().out
    assert printed.count("RESULT") == 3  # the two runs after the failed one were still trained
    assert f"FAILED  {TINY_NAMES[1]}: {error.__name__}: " in printed
    assert "ROUND tiny: 3 of 4 runs are finished." in printed
    # The closing list keeps the reason, and does not call every failure a breakdown of training.
    (closing,) = [line for line in printed.splitlines() if "FAILED, nothing was saved" in line]
    assert TINY_NAMES[1] in closing and "no room left on the disk" in closing


def test_a_plain_kill_ends_a_round_like_ctrl_c(tiny_round, monkeypatch, tmp_path, capsys):
    """`kill` sends the signal SIGTERM. Left alone it ends the main program at once, and with
    --jobs above 1 the worker processes would go on training. So it is turned into the same
    interruption as Ctrl-C, which stops the workers first."""
    with pytest.raises(KeyboardInterrupt):
        sweep.stop_like_ctrl_c(signal.SIGTERM, None)
    handler_before = signal.getsignal(signal.SIGTERM)
    stopped_workers = []
    monkeypatch.setattr(sweep, "stop_workers", lambda: stopped_workers.append(True))

    def killed_during_the_second_run(config):
        if default_name(config) == TINY_NAMES[1]:
            os.kill(os.getpid(), signal.SIGTERM)  # what `kill PID` does
            time.sleep(5)  # the signal arrives here; the run never gets to its end
        return pretend_finished(config)

    monkeypatch.setattr(sweep, "run", killed_during_the_second_run)
    with pytest.raises(SystemExit) as stopped:
        sweep.main(["tiny", "--out-dir", str(tmp_path)])
    assert stopped.value.code == 130
    assert "Interrupted. Type the same command again" in capsys.readouterr().out
    assert stopped_workers == [True]
    assert sorted(path.name for path in tmp_path.iterdir()) == [TINY_NAMES[0]]
    assert signal.getsignal(signal.SIGTERM) == handler_before  # put back as it was


# ------------------------------------------------------------------------------------------------
# sweep.py: several runs at a time
# ------------------------------------------------------------------------------------------------


def worker_that_dies_at_width_8_on_16_animals(config):
    """Stands in for sweep.train_quietly in a worker process (see the test below): one run ends
    its worker process without a word, as the system does to a process that runs out of memory.
    All other runs "finish" at once, without training."""
    if (config.animals, config.width) == (16, 8):
        first_time = not (Path(config.out_dir) / "died_before").exists()
        (Path(config.out_dir) / "died_before").touch()
        if first_time:
            time.sleep(1.0)  # long enough for the other worker to get through the other runs
        os._exit(1)
    return config, {"name": default_name(config), "minutes": 0.0}, None


def test_a_worker_process_that_dies_does_not_leave_the_round_waiting(tiny_round, tmp_path,
                                                                    capsys):
    """Before this was handled, a round with --jobs above 1 never ended when a worker died: it
    waited for ever for the run that worker had been training."""
    to_train = sweep.planned_runs("tiny", threads=1, out_dir=str(tmp_path))
    started = time.perf_counter()
    results = list(sweep.train_all(to_train, jobs=2,
                                   work=worker_that_dies_at_width_8_on_16_animals))
    assert time.perf_counter() - started < 60
    # Every run is accounted for: three came back, and the one that kills its worker is
    # reported as failed after a second try in a fresh worker.
    assert sorted(default_name(config) for config, _, _ in results) == TINY_NAMES
    failed = {default_name(config): problem for config, summary, problem in results
              if summary is None}
    assert list(failed) == [TINY_NAMES[1]]
    assert "ended without a word, also at a second try" in failed[TINY_NAMES[1]]
    assert "NOTE: a worker process died. 1 run that it left untrained" in capsys.readouterr().out
    assert sweep.multiprocessing.active_children() == []  # no worker is left behind


def test_sweep_with_two_jobs(tiny_round, tmp_path, capsys):
    sweep.main(["tiny", "--jobs", "2", "--out-dir", str(tmp_path)])
    printed = capsys.readouterr().out
    assert "4 runs, 2 at a time, 2 threads each" in printed
    assert "their own tables are not shown" in printed
    assert "RUN a16" not in printed  # the runs' own printouts are hidden
    assert printed.count("RESULT") == 4
    assert "ROUND tiny: 4 of 4 runs are finished." in printed
    assert sorted(path.name for path in tmp_path.iterdir()) == TINY_NAMES
    for name, config in zip(TINY_NAMES, sweep.planned_runs("tiny")):
        summary = saved_summary(tmp_path / name)
        assert (tmp_path / name / "model.pt").exists()
        assert summary["config"]["threads"] == 2
        assert (summary["config"]["animals"], summary["config"]["width"]) == (config.animals,
                                                                              config.width)
        assert summary["stop_reason"] == "budget" and summary["sets"] == 2
        assert summary["steps"] <= 100

    # The runs trained side by side are the runs trained one at a time: same text, same steps.
    alone = T.run(dataclasses.replace(sweep.planned_runs("tiny")[0], threads=2,
                                      out_dir=str(tmp_path / "alone")))
    together = saved_summary(tmp_path / TINY_NAMES[0])
    assert together["steps"] == alone["steps"]
    assert together["final"]["loss"] == pytest.approx(alone["final"]["loss"], abs=1e-4)


def test_two_jobs_with_nothing_left_to_train(tiny_round, monkeypatch, tmp_path, capsys):
    for config in sweep.planned_runs("tiny", threads=2, out_dir=str(tmp_path)):
        pretend_finished(config)
    never_train(monkeypatch)
    sweep.main(["tiny", "--jobs", "2", "--out-dir", str(tmp_path)])
    assert "ROUND tiny: 4 of 4 runs are finished." in capsys.readouterr().out

"""
Tests of report.py: the agreed decision rule, the table, the CSV files and the charts.

Run them with:    pytest tests/test_report.py

Almost everything here works on MADE-UP runs: summary.json files with invented numbers, written
into a temporary directory in the layout train.py uses. No model is needed to check a rule about
numbers, and nothing is written to the repository's runs/ or results/ directories. One test at
the end does two real runs of a few steps each, only to check that report.py can read what
train.py really writes.

The decision rule gets one made-up scenario for every outcome it can have:

  met                  a drop after a plateau, fixed by the next width up, in two of three seeds
  not rescued          a drop after a plateau that the next width up does not fix
  not finished         a drop in a run that ended by budget
  fixed length         a drop in a run that was given a fixed number of sets (no stopping rule)
  one seed only        everything holds, but in a single seed
  no drop              every width scores high everywhere: the models were not stretched
  never high           a width that cannot even learn the 16-animal language

Further down come the things the rule takes for granted and the ways a folder of runs can
mislead it: runs trained with another recipe or under another stopping rule (no verdict), two
runs at the same place, a missing or a stray width, a run that never began to learn facts, and
a score that fell before the run was called flat.
"""

import csv
import dataclasses
import json
import math
from itertools import takewhile

import matplotlib.image
import numpy as np
import pytest
import torch

import report as R
from train import RunConfig, default_name, grading_sets, run

# The size of the model's body for each width (two layers), as counted in model.py.
BODY = {4: 880, 6: 1608, 8: 2528, 16: 8128, 32: 28544, 64: 106240}


@pytest.fixture(autouse=True)
def small_pictures(monkeypatch):
    """Coarse pictures are enough for the tests, and they are much quicker to draw."""
    monkeypatch.setattr(R, "SAVE_DPI", 50)


# ------------------------------------------------------------------------------------------------
# Made-up runs
# ------------------------------------------------------------------------------------------------


def made_up_run(folder, animals, width, seed, facts, ended, usage="equal", seasons=1,
                table_animals=1024, sets=64, name=None, **other_settings):
    """Write folder/<name>/summary.json with INVENTED numbers, in the layout of train.py.

    facts is the final score and ended the stop reason. Everything else is invented to fit:
    a learning curve that rises to the final score, a loss, a list of facts right per animal
    (worse for rare animals under zipf usage). other_settings are further fields of RunConfig,
    for example learning_rate=0.001. Returns the summary dictionary.
    """
    config = RunConfig(animals=animals, table_animals=table_animals, seasons=seasons, usage=usage,
                       width=width, seed=seed, out_dir=str(folder), **other_settings)
    config = dataclasses.replace(config, name=name or default_name(config))
    rng = np.random.default_rng([seed, width, animals, seasons])
    vocab_size = 27 + (seasons if seasons > 1 else 0) + table_animals
    bits_total = 8.0 * seasons * animals
    floor = 1.2 + math.log2(animals) / 8
    seen_per_set = 15 * table_animals / animals

    known = np.full(animals, facts / 100)
    if usage == "zipf":  # rare animals (high numbers) are known less well
        rank = np.arange(1, animals + 1)
        known = np.clip(known * (1.7 - 1.4 * np.log(rank) / math.log(max(animals, 2))), 0, 1)
    right = rng.binomial(4 * seasons, 0.25 + 0.75 * known)

    def score_at(share):  # the invented score after this share of the training
        return {
            "facts_known": facts * share, "facts_known_by_use": min(100.0, facts * share * 1.1),
            "questions_right": facts * share * 0.9,
            "unseen_known": None if table_animals == animals else 0.4,
            "words_known": int((right == 4 * seasons).sum() * share), "n_words": animals,
            "loss": floor + 1.5 * (1 - facts * share / 100), "floor": floor,
            "gap": 1.5 * (1 - facts * share / 100),
            "bits_total": bits_total, "bits_known": bits_total * (facts * share / 100) ** 1.5,
            "headline": facts * share, "headline_name": "facts_known",
        }

    # Graded on the schedule, and once more at the last set if that is not on the schedule.
    on_schedule = list(takewhile(lambda count: count <= sets, grading_sets()))
    graded_at = on_schedule if on_schedule[-1] == sets else on_schedule + [sets]
    curve = []
    for count in graded_at:
        share = (1 - math.exp(-5 * count / sets)) / (1 - math.exp(-5))
        curve.append({"sets": count, "steps": 53 * count, "minutes": 0.02 * count,
                      "sightings": seen_per_set * count, "scheduled": count in on_schedule,
                      **score_at(share)})
    summary = {
        "name": config.name,
        "config": dataclasses.asdict(config),
        "language": {"vocab_size": vocab_size, "animals": animals, "table_animals": table_animals,
                     "seasons": seasons, "usage": usage,
                     "set_sentences": 60 * seasons * table_animals, "bits_total": bits_total,
                     "bits_per_animal": bits_total / animals, "floor": floor},
        "parameters": {"body": BODY[width], "table": vocab_size * width,
                       "total": BODY[width] + vocab_size * width},
        "stop_reason": ended,
        "stop_explained": f"made up: {ended}",
        "sets": sets, "steps": 53 * sets, "minutes": 0.02 * sets,
        "sightings": seen_per_set * sets,
        "final": {**score_at(1.0), "facts_right_by_animal": right.tolist()},
        "curve": curve,
    }
    run_dir = folder / config.name
    run_dir.mkdir(parents=True)
    with open(run_dir / "summary.json", "w") as file:
        json.dump(summary, file)
    return summary


def write_runs(folder, table, **settings):
    """Write one made-up run for every entry of `table`, which reads like a results table:

        {(width, animals): [(score, ending) for seed 0, for seed 1, ...]}
    """
    for (width, animals), per_seed in table.items():
        for seed, (facts, ended) in enumerate(per_seed):
            made_up_run(folder, animals, width, seed, facts, ended, **settings)


def judge(folder):
    """The rule applied to the runs in a folder. Returns (result, the printed text as one line)."""
    result = R.rule_met(R.load_runs(folder))
    return result, " ".join(" ".join(result["lines"]).split())


def verdict_of(result, width):
    (verdict,) = [verdict for verdict in result["verdicts"] if verdict["width"] == width]
    return verdict


SOLVED = [(99, "solved"), (100, "solved"), (98, "solved")]  # a width that knows everything


# ------------------------------------------------------------------------------------------------
# The decision rule, one scenario per outcome
# ------------------------------------------------------------------------------------------------


def test_the_agreed_numbers_are_unchanged():
    assert (R.HIGH, R.DROP, R.RESCUE, R.SEEDS_NEEDED, R.SMALL) == (95, 10, 10, 2, 16)


def test_rule_met_when_a_plateau_drop_is_rescued_in_two_of_three_seeds(tmp_path):
    write_runs(tmp_path, {
        (4, 16): SOLVED,
        (4, 1024): [(60, "plateau"), (62, "plateau"), (97, "solved")],  # seed 2 shows no drop
        (8, 16): SOLVED,
        (8, 1024): SOLVED,
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}

    width_4 = verdict_of(result, 4)
    assert width_4["met"] and width_4["seeds_met"] == [0, 1] and width_4["next_width"] == 8
    assert width_4["held"] == {"a": 3, "b": 2, "c": 2}
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["met", "met", "no_drop"]
    assert "Width 4: the rule is MET. (a), (b) and (c) all hold in 2 of 3 seeds (2 needed)." in text
    assert ("seed 0: (a) yes: 99.0 on 16 animals. (b) yes: 60.0 on 1,024 animals after a plateau, "
            "39.0 points lower. (c) yes: width 8 scores 99.0 there, 39.0 points higher. "
            "All three hold.") in text
    assert "seed 2: (a) yes: 98.0 on 16 animals. (b) no: its lowest score" in text

    # Width 8 is the largest width here and it never dropped.
    width_8 = verdict_of(result, 8)
    assert not width_8["met"] and width_8["next_width"] == 16  # of which there are no runs
    assert "Width 8: the rule is NOT met." in text
    assert "SUMMARY for equal usage, 1 season: the rule is met for width 4." in text
    assert "The models were not stretched" not in text


def test_a_drop_that_the_next_width_does_not_fix_is_not_a_size_effect(tmp_path):
    write_runs(tmp_path, {
        (4, 16): SOLVED,
        (4, 1024): [(60, "plateau"), (62, "plateau"), (61, "plateau")],
        (8, 16): SOLVED,
        (8, 1024): [(64, "plateau"), (63, "plateau"), (70, "plateau")],  # at most 9 points higher
        (16, 16): SOLVED,
        (16, 1024): [(66, "plateau"), (65, "plateau"), (72, "plateau")],
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    width_4 = verdict_of(result, 4)
    assert width_4["held"] == {"a": 3, "b": 3, "c": 0}
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["not_rescued"] * 3
    assert ("(c) no: width 8 scores 64.0 there, 4.0 points higher; a rescue needs 10. A drop "
            "that the next width up does not fix is NOT a size effect.") in text
    assert "SUMMARY for equal usage, 1 season: the rule is met for no width." in text
    assert ("A drop for width 4 on 1,024 animals and width 8 on 1,024 animals was not fixed by "
            "the next width up") in text
    assert "that drop is not a size effect" in text
    # The largest width dropped as well, but there is nothing larger to compare it with.
    assert [seed["outcome"] for seed in verdict_of(result, 16)["seeds"]] == ["no_next_width"] * 3
    assert ("(c) cannot be tested: there are no runs of width 32, the next width up, among these "
            "runs.") in text
    assert "A drop for width 16 on 1,024 animals has no run of the next width up beside it" in text


def test_a_drop_in_a_run_that_ended_by_budget_cannot_count(tmp_path):
    write_runs(tmp_path, {
        (4, 16): SOLVED,
        (4, 1024): [(40, "budget"), (42, "budget"), (41, "budget")],
        (8, 16): SOLVED,
        (8, 1024): SOLVED,  # the next width up would even "rescue" it
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    width_4 = verdict_of(result, 4)
    assert not width_4["met"] and width_4["seeds_met"] == []
    assert width_4["held"] == {"a": 3, "b": 0, "c": 3}
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["not_finished"] * 3
    assert ("(b) no: 40.0 on 1,024 animals, 59.0 points lower, but that run ended by budget: "
            "it is NOT FINISHED and cannot count.") in text
    assert "comes from a run that ended by budget" in text
    assert "not finished and cannot count" in text
    assert "The models were not stretched" not in text


def test_a_drop_in_a_run_of_a_fixed_length_cannot_count(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (8, 16): SOLVED, (8, 1024): SOLVED})
    # Width 4 on 1,024 animals was trained with --fixed-sets 64: the stopping rule was never
    # asked, so nobody has checked that these runs were flat.
    for seed, facts in enumerate([40, 42, 41]):
        fixed = made_up_run(tmp_path, 1024, 4, seed, facts, "fixed", fixed_sets=64)
        assert fixed["name"] == f"a1024_w4_equal_s{seed}_f64"
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    width_4 = verdict_of(result, 4)
    assert not width_4["met"] and width_4["held"] == {"a": 3, "b": 0, "c": 3}
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["not_plateau"] * 3
    assert ("(b) no: 40.0 on 1,024 animals, 59.0 points lower, but that run had a fixed number "
            "of sets: the stopping rule was never asked, so it cannot count as a plateau.") in text
    assert "Only a drop in a run that ended by plateau counts." in text
    assert ("A drop for width 4 on 1,024 animals comes from a run that did not end by "
            "plateau") in text
    assert "The models were not stretched" not in text
    # Runs of a fixed length among the ordinary runs are pointed out as well. That is a note:
    # how long a run trained is not part of the recipe, so the rule is still applied.
    assert "NOTE: this group holds runs of a fixed length (3 runs: " in text
    assert "CAREFUL" not in text and result["refused"] == {}


def test_the_ordinary_run_counts_when_a_run_of_a_fixed_length_sits_beside_it(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(60, "plateau")] * 3,
                          (8, 16): SOLVED, (8, 1024): SOLVED})
    # The check the setting is for: the same model trained much longer, under a name of its own.
    made_up_run(tmp_path, 1024, 4, 0, 61, "fixed", fixed_sets=1448, sets=1448)
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}
    assert verdict_of(result, 4)["seeds_met"] == [0, 1, 2]
    assert "a1024_w4_equal_s0_f1448 (facts known 61.0, ended as fixed) is left out." in text
    assert ("a1024_w4_equal_s0 (facts known 60.0, ended as plateau) has the same vocabulary, width "
            "and seed and is the run that counts, because a run of a fixed length never asked the "
            "stopping rule.") in text
    assert "CAREFUL" not in text  # the run that was left out is not part of the group
    assert "NOTE: this group holds runs of a fixed length" not in text


def test_a_drop_in_one_seed_only_is_not_enough(tmp_path):
    write_runs(tmp_path, {
        (4, 16): SOLVED,
        (4, 1024): [(60, "plateau"), (97, "solved"), (98, "solved")],
        (8, 16): SOLVED,
        (8, 1024): SOLVED,
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    width_4 = verdict_of(result, 4)
    assert not width_4["met"] and width_4["seeds_met"] == [0]
    assert "Width 4: the rule is NOT met. (a), (b) and (c) all hold in 1 of 3 seeds" in text
    assert "For width 4 all three conditions hold in fewer than 2 seeds" in text


def test_no_drop_anywhere_means_the_models_were_not_stretched(tmp_path):
    write_runs(tmp_path, {(width, animals): SOLVED for width in [4, 8, 32]
                          for animals in [16, 256, 1024]})
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    for verdict in result["verdicts"]:
        assert verdict["held"] == {"a": 3, "b": 0, "c": 0}
        assert [seed["outcome"] for seed in verdict["seeds"]] == ["no_drop"] * 3
    assert "SUMMARY for equal usage, 1 season: the rule is met for no width." in text
    assert "The models were not stretched" in text
    assert "neither support for the hypothesis nor a refutation of it" in text
    assert "a drop needs 10. (c) no: width 8 scores 99.0 there, 0.0 points higher" in text
    assert "No drop in this seed." in text


def test_a_width_that_never_reached_high_on_16_animals_cannot_meet_the_rule(tmp_path):
    write_runs(tmp_path, {
        (4, 16): [(80, "plateau"), (82, "plateau"), (94.9, "plateau")],
        (4, 1024): [(40, "plateau"), (41, "plateau"), (42, "plateau")],  # a drop, and rescued,
        (8, 16): SOLVED,
        (8, 1024): SOLVED,  # ... but (a) fails
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    width_4 = verdict_of(result, 4)
    assert width_4["held"] == {"a": 0, "b": 3, "c": 3} and not width_4["met"]
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["not_high"] * 3
    assert "seed 0: (a) no: 80.0 on 16 animals." in text
    assert "the width does not reach 95 even on 16 animals" in text
    assert "A seed of width 4 did not reach 95 on 16 animals" in text


# ------------------------------------------------------------------------------------------------
# The decision rule, the fine print
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("small, large, rescuer, expected", [
    (95.0, 85.0, 95.0, True),  # exactly HIGH, exactly DROP, exactly RESCUE: all count
    (94.9, 80.0, 95.0, False),  # (a) just missed
    (100.0, 90.1, 100.0, False),  # a drop of 9.9
    (100.0, 80.0, 89.9, False),  # a rescue of 9.9
    (100.0, 6.4, 16.4, True),  # a rescue of 10: the computer says 16.4 - 6.4 = 9.999999999999998
])
def test_the_thresholds_are_inclusive(tmp_path, small, large, rescuer, expected):
    write_runs(tmp_path, {
        (4, 16): [(small, "plateau")] * 2,
        (4, 256): [(large, "plateau")] * 2,
        (8, 16): [(100, "solved")] * 2,
        (8, 256): [(rescuer, "plateau")] * 2,
    })
    result, _ = judge(tmp_path)
    assert verdict_of(result, 4)["met"] is expected


def test_drop_and_rescue_must_be_on_the_same_vocabulary(tmp_path):
    # At 256 animals: a drop after a plateau, not rescued. At 1,024: rescued, but not finished.
    write_runs(tmp_path, {
        (4, 16): SOLVED,
        (4, 256): [(80, "plateau")] * 3,
        (4, 1024): [(40, "budget")] * 3,
        (8, 16): SOLVED,
        (8, 256): [(84, "plateau")] * 3,
        (8, 1024): [(90, "plateau")] * 3,
    })
    result, text = judge(tmp_path)
    width_4 = verdict_of(result, 4)
    assert not width_4["met"]
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["not_rescued"] * 3
    assert [seed["animals"] for seed in width_4["seeds"]] == [256, 256, 256]


def test_the_rescuer_must_have_the_same_seed(tmp_path):
    # Width 8 exists at 1,024 animals for seed 2 only; the drops are in seeds 0 and 1.
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(60, "plateau")] * 3, (8, 16): SOLVED})
    made_up_run(tmp_path, 1024, 8, 2, 99, "solved")
    result, text = judge(tmp_path)
    width_4 = verdict_of(result, 4)
    assert width_4["seeds_met"] == [2] and not width_4["met"]
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["no_next_width"] * 2 + ["met"]
    assert "(c) cannot be tested: no run of width 8 on 1,024 animals with this seed." in text


def test_the_next_width_up_is_twice_the_width(tmp_path):
    # The "quick" round has only widths 4 and 32, and a single seed. Width 32 scores 49 points
    # higher, but it is not the next width up of width 4, so (c) cannot be tested.
    write_runs(tmp_path, {
        (4, 16): [(99, "solved")], (4, 1024): [(50, "plateau")],
        (32, 16): [(100, "solved")], (32, 1024): [(99, "solved")],
    })
    result, text = judge(tmp_path)
    width_4 = verdict_of(result, 4)
    assert R.WIDTH_STEP == 2 and width_4["next_width"] == 8
    assert width_4["seeds_met"] == [] and width_4["held"] == {"a": 1, "b": 1, "c": 0}
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["no_next_width"]
    assert ("(c) cannot be tested: there are no runs of width 8, the next width up, among these "
            "runs. Width 32 is not the next width up and cannot stand in for it.") in text
    assert "The next width up of a width is 2 times that width (8 for width 4)." in text


def test_a_missing_width_does_not_hand_the_rescue_to_a_larger_one(tmp_path):
    # Width 8 did not fix the drop of width 4 (and was not finished); width 16 knows everything.
    runs = {(4, 16): SOLVED, (4, 1024): [(70, "plateau")] * 3,
            (8, 16): SOLVED, (8, 1024): [(72, "budget")] * 3,
            (16, 16): SOLVED, (16, 1024): SOLVED}
    write_runs(tmp_path / "all", runs)
    result, _ = judge(tmp_path / "all")
    assert result["met"] == {("equal", 1): []}
    # Take the width-8 runs away, and width 16 must NOT step in as "the next width up".
    del runs[(8, 16)], runs[(8, 1024)]
    write_runs(tmp_path / "no_8", runs)
    result, text = judge(tmp_path / "no_8")
    assert result["met"] == {("equal", 1): []}
    assert [seed["outcome"] for seed in verdict_of(result, 4)["seeds"]] == ["no_next_width"] * 3
    assert "Width 16 is not the next width up and cannot stand in for it." in text


def test_one_stray_run_of_another_width_changes_nothing(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(70, "plateau")] * 3,
                          (8, 16): SOLVED, (8, 1024): SOLVED})
    made_up_run(tmp_path, 16, 6, 0, 100, "solved")  # one try with width 6, on 16 animals only
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}  # width 8 is still the next width up of width 4
    assert "(c) yes: width 8 scores 99.0 there, 29.0 points higher." in text
    assert [seed["outcome"] for seed in verdict_of(result, 6)["seeds"]] == ["no_larger_run"]


def test_a_rescuer_that_scores_lower_is_described_as_lower(tmp_path):
    write_runs(tmp_path, {
        (4, 16): SOLVED, (4, 64): [(70, "plateau")] * 3,
        (8, 16): SOLVED, (8, 64): [(65, "plateau")] * 3,
    })
    _, text = judge(tmp_path)
    assert "(c) no: width 8 scores 65.0 there, 5.0 points lower; a rescue needs 10." in text


def test_each_usage_and_number_of_seasons_is_judged_on_its_own(tmp_path):
    drop = {(4, 16): SOLVED, (4, 1024): [(60, "plateau")] * 3, (8, 16): SOLVED, (8, 1024): SOLVED}
    flat = {(4, 16): SOLVED, (4, 1024): SOLVED, (8, 16): SOLVED, (8, 1024): SOLVED}
    write_runs(tmp_path, drop, usage="equal")
    write_runs(tmp_path, flat, usage="zipf")
    write_runs(tmp_path, drop, usage="equal", seasons=4)
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4], ("zipf", 1): [], ("equal", 4): [4]}
    assert "SUMMARY for zipf usage, 1 season: the rule is met for no width." in text
    assert "SUMMARY for equal usage, 4 seasons: the rule is met for width 4." in text


def test_runs_with_a_smaller_table_and_doubles_are_left_out(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(60, "plateau")] * 3,
                          (8, 16): SOLVED, (8, 1024): SOLVED})
    # The table-check round: 16 animals in a 16-animal table. A poor score that must not be
    # mistaken for the 16-animal score of the main runs.
    write_runs(tmp_path, {(4, 16): [(50, "plateau")] * 3}, table_animals=16)
    # The same place once more under other names, with a score that would break (b). The run
    # with the default name is the one that counts, wherever the other names fall in the alphabet.
    made_up_run(tmp_path, 1024, 4, 0, 99, "solved", name="second_try")
    made_up_run(tmp_path, 1024, 4, 1, 99, "solved", name="a1024_w4_equal_again")
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}
    assert verdict_of(result, 4)["seeds_met"] == [0, 1, 2]
    assert "a16_w4_equal_t16_s0 has a word table of 16 animals, not the 1,024 of its group" in text
    assert "second_try (facts known 99.0, ended as solved) is left out." in text
    assert "a1024_w4_equal_again (facts known 99.0, ended as solved) is left out." in text
    assert ("a1024_w4_equal_s1 (facts known 60.0, ended as plateau) has the same vocabulary, width "
            "and seed and is the run that counts, because it lies in the folder train.py gives a "
            "run of these settings by default.") in text
    # The run left out would have changed the verdict, and the note says so.
    assert "CAREFUL: the two scores are 39.0 points apart, so it matters which" in text


def test_one_run_with_another_table_does_not_push_the_others_out(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(60, "plateau")] * 3,
                          (8, 16): SOLVED, (8, 1024): SOLVED})
    # One try with a larger word table. The twelve runs above stay "the group".
    made_up_run(tmp_path, 2048, 4, 0, 20, "plateau", table_animals=2048)
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}
    assert [seed["animals"] for seed in verdict_of(result, 4)["seeds"]] == [1024] * 3
    assert "a2048_w4_equal_t2048_s0 has a word table of 2,048 animals, not the 1,024" in text
    assert text.count("has a word table of") == 1


def test_a_next_width_that_was_not_finished_leaves_the_drop_open(tmp_path):
    write_runs(tmp_path, {
        (4, 16): SOLVED,
        (4, 1024): [(60, "plateau")] * 3,
        (8, 16): SOLVED,
        (8, 1024): [(64, "budget")] * 3,  # 4 points higher so far, but it ran out of time
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    width_4 = verdict_of(result, 4)
    assert width_4["held"] == {"a": 3, "b": 3, "c": 0}
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["next_not_finished"] * 3
    assert ("(c) no: width 8 scores 64.0 there (that run was NOT FINISHED), 4.0 points higher; "
            "a rescue needs 10. The next width up was not finished") in text
    assert ("A drop for width 4 on 1,024 animals could not be judged: the next width up ended by "
            "budget") in text
    assert "is NOT a size effect" not in text and "is not a size effect" not in text


def test_a_next_width_that_rescues_counts_even_if_it_was_not_finished(tmp_path):
    # More training can only widen the distance, so 30 points higher is a rescue all the same.
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(60, "plateau")] * 3,
                          (8, 16): SOLVED, (8, 1024): [(90, "budget")] * 3})
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}
    assert "(c) yes: width 8 scores 90.0 there (that run was NOT FINISHED), 30.0 points" in text


# ------------------------------------------------------------------------------------------------
# The decision rule: what it takes for granted (the same recipe, the agreed stopping rule)
# ------------------------------------------------------------------------------------------------

# A clear case for the rule: width 4 drops after a plateau, width 8 fixes it, in all three seeds.
CLEAR_CASE = {(4, 16): SOLVED, (4, 1024): [(70, "plateau")] * 3, (8, 16): SOLVED}


def change_summary(folder, name, change):
    """Rewrite folder/name/summary.json after `change` has edited the summary dictionary."""
    path = folder / name / "summary.json"
    with open(path) as file:
        summary = json.load(file)
    change(summary)
    with open(path, "w") as file:
        json.dump(summary, file)


def test_runs_trained_with_another_recipe_get_no_verdict(tmp_path):
    write_runs(tmp_path, CLEAR_CASE)
    for seed in range(3):  # the rescuers were trained with another learning rate and budget
        made_up_run(tmp_path, 1024, 8, seed, 99, "solved", learning_rate=0.001, max_minutes=5)
    result, text = judge(tmp_path)
    # Without the other learning rate this would be "met for width 4". A rescue that was bought
    # with another recipe is not a size effect, so the group gets no verdict at all.
    assert result["met"] == {} and result["verdicts"] == []
    assert list(result["refused"]) == [("equal", 1)]
    assert ("CAREFUL: the runs were not all trained the same way. learning_rate is 0.001 in "
            "3 runs: ") in text
    assert "a1024_w8_equal_s2. In the other runs of this group it is 0.01." in text
    assert "The rule is NOT applied to this group, and no verdict is given for it." in text
    assert "the rule is met" not in text and "clearly holds" not in text.split("---")[-1]
    assert text.count("CAREFUL") == 1  # a different budget (max_minutes) is no reason to refuse


def test_runs_that_read_sets_of_another_size_get_no_verdict(tmp_path):
    write_runs(tmp_path, {**CLEAR_CASE, (8, 1024): SOLVED})
    result, _ = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]} and result["refused"] == {}

    # (c) says "the same seed and so the same training sets". With sets of another size the
    # dropped run and its rescuer did not read the same sets. The size is read as the language
    # resolved it, because 0 in the settings only means "automatic".
    def smaller_sets(summary):
        summary["language"]["set_sentences"] = 1024

    for seed in range(3):
        change_summary(tmp_path, f"a1024_w4_equal_s{seed}", smaller_sets)
    result, text = judge(tmp_path)
    assert result["met"] == {} and ("equal", 1) in result["refused"]
    assert "set_sentences is 1024 in 3 runs" in text and "it is 61440." in text


def test_a_plateau_under_a_looser_stopping_rule_gets_no_verdict(tmp_path):
    # Every run of the group was trained with a stopping rule that calls a plateau at once. The
    # runs agree with each other, but "plateau" does not mean here what was agreed.
    write_runs(tmp_path, {**CLEAR_CASE, (8, 1024): SOLVED}, min_steps=0, min_sightings=0,
               flat_points=100.0)
    result, text = judge(tmp_path)
    assert result["met"] == {} and len(result["refused"][("equal", 1)]) == 3
    assert ("CAREFUL: the stopping rule was not the agreed one. min_steps is 0 in 12 runs: "
            in text)
    assert "The agreed value is 2000." in text
    assert "flat_points is 100 in 12 runs" in text and "min_sightings is 0 in 12 runs" in text
    assert "The rule is NOT applied to this group" in text


def test_runs_saved_before_a_setting_of_the_stopping_rule_existed_get_no_verdict(tmp_path):
    write_runs(tmp_path, {**CLEAR_CASE, (8, 1024): SOLVED})

    def as_older_code_saved_it(summary):
        del summary["config"]["min_steps"], summary["config"]["started_at"]

    for path in sorted(tmp_path.iterdir()):
        change_summary(tmp_path, path.name, as_older_code_saved_it)
    result, text = judge(tmp_path)
    assert result["met"] == {}
    assert "min_steps is not recorded in 12 runs" in text
    assert "started_at is not recorded in 12 runs" in text and "The agreed value is 5." in text


def test_the_rule_names_the_stopping_rule_it_relies_on(tmp_path):
    write_runs(tmp_path, {**CLEAR_CASE, (8, 1024): SOLVED})
    _, text = judge(tmp_path)
    assert ("A plateau is a plateau under the stopping rule of train.py with its agreed settings "
            "(solved_at 97, flat_points 2, flat_gap 0.05, flat_gap_bits 0.002, min_sightings 400, "
            "min_steps 2000, started_at 5).") in text
    assert "The rule is applied only to a group of runs that were all trained the same way" in text


def test_every_setting_of_a_run_has_its_place_in_the_report():
    """A setting that is added to RunConfig must be sorted into one of the lists of report.py by
    someone who has thought about it. Until then it counts as part of the recipe (the safe side),
    and this test fails to say so."""
    every_setting = [field.name for field in dataclasses.fields(RunConfig)]
    apart = R.DIALS + R.GROUP_KEYS + R.LENGTH + R.RUNNING
    assert len(set(apart)) == len(apart) and set(apart) < set(every_setting)
    assert set(R.STOPPING_RULE) < set(R.recipe_settings())
    assert R.recipe_settings() == [
        "set_sentences", "layers", "heads", "batch", "context", "learning_rate", "warmup",
        "average", "solved_at", "flat_points", "flat_gap", "flat_gap_bits", "min_sightings",
        "min_steps", "started_at"]


def test_the_table_of_means_warns_when_the_recipes_are_mixed(tmp_path, capsys):
    write_runs(tmp_path, CLEAR_CASE)
    for seed in range(3):
        made_up_run(tmp_path, 1024, 8, seed, 99, "solved", learning_rate=0.001)
    groups, _ = R.comparison_groups(R.load_runs(tmp_path))
    R.print_means(groups, cautions=[("equal", 1)])
    printed = " ".join(capsys.readouterr().out.split())
    assert "CAREFUL: the runs of this group were not all trained the same way" in printed


# ------------------------------------------------------------------------------------------------
# The decision rule: two runs at the same place
# ------------------------------------------------------------------------------------------------


def copy_run(folder, name, new_folder_name, facts, ended):
    """A second run with the same settings and the SAME name inside, in a folder of another name:
    what copying a run folder leaves behind. Only its score and its ending differ."""
    with open(folder / name / "summary.json") as file:
        summary = json.load(file)
    summary["final"]["facts_known"] = facts
    summary["stop_reason"] = ended
    (folder / new_folder_name).mkdir()
    with open(folder / new_folder_name / "summary.json", "w") as file:
        json.dump(summary, file)


def test_an_ordinary_run_counts_before_a_fixed_run_whatever_their_names(tmp_path):
    write_runs(tmp_path, CLEAR_CASE)
    for seed in range(3):
        # Width 8 on 1,024 animals twice: an ordinary run under a name of its own, which does
        # not fix the drop, and a run of a fixed length under the default name, which would.
        made_up_run(tmp_path, 1024, 8, seed, 72, "plateau", name=f"patient_s{seed}")
        made_up_run(tmp_path, 1024, 8, seed, 99, "fixed", fixed_sets=2000, sets=2000)
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    assert [seed["outcome"] for seed in verdict_of(result, 4)["seeds"]] == ["not_rescued"] * 3
    assert "(c) no: width 8 scores 72.0 there, 2.0 points higher; a rescue needs 10." in text
    assert "a1024_w8_equal_s0_f2000 (facts known 99.0, ended as fixed) is left out." in text
    assert "patient_s0 (facts known 72.0, ended as plateau) has the same vocabulary" in text
    assert "CAREFUL: the two scores are 27.0 points apart, so it matters which" in text


def test_a_copy_of_a_run_folder_does_not_decide_the_verdict(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): SOLVED, (8, 16): SOLVED, (8, 1024): SOLVED})
    for seed in range(3):
        # An older result kept beside the new one. Its folder sorts BEFORE the real one.
        copy_run(tmp_path, f"a1024_w4_equal_s{seed}", f"OLD_a1024_w4_equal_s{seed}", 70, "plateau")
    runs = R.load_runs(tmp_path)
    assert len(runs) == 15 and len({run["folder"] for run in runs}) == 15
    result = R.rule_met(runs)
    text = " ".join(" ".join(result["lines"]).split())
    assert result["met"] == {("equal", 1): []}  # the run in the folder train.py writes counts
    assert [seed["outcome"] for seed in verdict_of(result, 4)["seeds"]] == ["no_drop"] * 3
    assert "OLD_a1024_w4_equal_s0 (facts known 70.0, ended as plateau) is left out." in text
    assert "because it lies in the folder train.py gives a run of these settings by default" in text


def test_two_runs_that_cannot_be_told_apart_leave_their_place_empty(tmp_path):
    for first, second in [("a_first", "z_second"), ("z_first", "a_second")]:
        folder = tmp_path / first
        write_runs(folder, {**CLEAR_CASE, (8, 1024): SOLVED})
        for seed in range(3):  # width 4 on 1,024 animals under two names of the user's own
            (folder / f"a1024_w4_equal_s{seed}").rename(folder / f"{first}_s{seed}")
            copy_run(folder, f"{first}_s{seed}", f"{second}_s{seed}", 99, "solved")
        result, text = judge(folder)
        # Whichever name comes first in the alphabet: neither run is used, so nothing is met.
        assert result["met"] == {("equal", 1): []}
        assert [seed["outcome"] for seed in verdict_of(result, 4)["seeds"]] == ["no_larger_run"] * 3
        assert ("There are 2 runs for width 4 on 1,024 animals with seed 0 and nothing says which "
                "one counts") in text
        assert "NONE of them is used" in text


# ------------------------------------------------------------------------------------------------
# The decision rule: runs of a fixed length, runs that never began, scores that fell
# ------------------------------------------------------------------------------------------------


def test_a_fixed_rescuer_that_does_not_rescue_leaves_the_drop_open(tmp_path):
    write_runs(tmp_path, CLEAR_CASE)
    for seed in range(3):  # the next width up was cut off after 5 sets by --fixed-sets
        made_up_run(tmp_path, 1024, 8, seed, 72, "fixed", fixed_sets=5, sets=5)
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    assert [seed["outcome"] for seed in verdict_of(result, 4)["seeds"]] == ["next_fixed"] * 3
    assert ("(c) no: width 8 scores 72.0 there (that run had a fixed number of sets), 2.0 points "
            "higher; a rescue needs 10. The next width up had a fixed number of sets and never "
            "asked the stopping rule, so nobody knows whether it would fix the drop") in text
    assert "A drop for width 4 on 1,024 animals could not be judged: the next width up had a " \
           "fixed number of sets" in text
    assert "is NOT a size effect" not in text and "is not a size effect" not in text


def test_a_fixed_run_on_16_animals_is_named_as_such(tmp_path):
    write_runs(tmp_path, {(4, 1024): SOLVED, (8, 16): SOLVED, (8, 1024): SOLVED})
    for seed in range(3):
        made_up_run(tmp_path, 16, 4, seed, 100, "fixed", fixed_sets=20, sets=20)
    _, text = judge(tmp_path)
    assert "(a) yes: 100.0 on 16 animals (that run had a fixed number of sets)." in text


def test_a_run_that_never_began_to_learn_facts_is_said_to_be_just_that(tmp_path):
    # Width 4 stays at 2 facts out of 100 on 1,024 animals. Below started_at the stopping rule
    # never calls a plateau, so such a run can only end by budget, however long it trains.
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(2.0, "budget")] * 3,
                          (8, 16): SOLVED, (8, 1024): SOLVED})
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    width_4 = verdict_of(result, 4)
    assert width_4["held"] == {"a": 3, "b": 0, "c": 3}
    assert [seed["outcome"] for seed in width_4["seeds"]] == ["not_begun"] * 3
    assert ("(b) no: 2.0 on 1,024 animals, 97.0 points lower, but that run ended by budget before "
            "it had begun to learn facts (fewer than 5 of 100 known): it is NOT FINISHED and "
            "cannot count.") in text
    assert "More training helps only if this model then begins" in text
    assert ("A drop for width 4 on 1,024 animals comes from a run that ended by budget before it "
            "had begun to learn facts (fewer than 5 facts out of 100 known).") in text
    assert "the agreed rule can reach no verdict from it" in text
    # The advice for an ordinary unfinished run would be wrong here, and is not given.
    assert "train it longer (--max-minutes) before reading anything into the drop" not in text
    # A run out of budget that HAS begun (5 or more) gets the ordinary sentence.
    assert R.outcome_on({"dropped": True, "b": False, "stop_reason": "budget",
                         "below_start": False}) == "not_finished"


def test_a_plateau_far_below_the_runs_own_best_score_is_pointed_out(tmp_path):
    write_runs(tmp_path, {**CLEAR_CASE, (8, 1024): SOLVED})
    _, text = judge(tmp_path)
    assert "fell back" not in text

    def score_fell(summary):  # the run had stood at 95 and ended at 70
        summary["curve"][-3]["facts_known"] = 95.0

    change_summary(tmp_path, "a1024_w4_equal_s0", score_fell)
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}  # the verdict is the rule's; the reader is warned
    assert text.count("CAREFUL: this run had reached 95.0 earlier and fell back.") == 1
    assert "points to unstable training, not to a limit of the model's size" in text


# ------------------------------------------------------------------------------------------------
# The decision rule: what the sentences say
# ------------------------------------------------------------------------------------------------


def test_every_larger_vocabulary_with_a_drop_is_reported(tmp_path):
    # On 64 animals: a small drop after a plateau that width 8 does not fix. On 1,024 animals: a
    # large drop that width 8 does fix, in a run that was not finished. Both must be told.
    write_runs(tmp_path, {
        (4, 16): SOLVED, (4, 64): [(88, "plateau")] * 3, (4, 1024): [(30, "budget")] * 3,
        (8, 16): SOLVED, (8, 64): [(90, "plateau")] * 3, (8, 1024): [(95, "solved")] * 3,
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    seed = verdict_of(result, 4)["seeds"][0]
    assert seed["outcome"] == "not_rescued" and seed["animals"] == 64
    assert [v["outcome"] for v in seed["vocabularies"]] == ["not_rescued", "not_finished"]
    assert ("Also lower on 1,024 animals: 30.0 (ended by budget: NOT FINISHED), 69.0 points "
            "lower; width 8 scores 95.0 there. In short: that run cannot count.") in text
    assert "A drop for width 4 on 64 animals was not fixed by the next width up" in text
    assert "A drop for width 4 on 1,024 animals comes from a run that ended by budget." in text


def test_a_drop_that_does_not_go_on_at_larger_vocabularies_is_said_so(tmp_path):
    write_runs(tmp_path, {
        (4, 16): SOLVED, (4, 64): [(80, "plateau")] * 3, (4, 256): SOLVED, (4, 1024): SOLVED,
        (8, 16): SOLVED, (8, 64): SOLVED, (8, 256): SOLVED, (8, 1024): SOLVED,
    })
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): [4]}  # the letter of the rule: "some larger vocabulary"
    assert ("All three hold. No such drop on the larger vocabularies: 99.0 on 256 animals and "
            "99.0 on 1,024 animals.") in text


def test_a_value_just_below_a_threshold_is_not_printed_as_the_threshold(tmp_path):
    assert R.shown(9.993) == "10.0" and R.shown(9.993, 10) == "9.99"
    assert R.shown(10.0, 10) == "10.0" and R.shown(3.21, 10) == "3.2"
    assert R.shown(94.99, 95) == "94.99" and R.shown(9.9996, 10) == "9.9996"
    assert R.shown(16.4 - 6.4, 10) == "10.0"  # 9.999999999999998 counts as 10 (see at_least)
    # Scores as score.py can give them: 62 of 64 facts, and 3,661 of 4,096.
    small, large = 100 * (62 / 64 - 0.25) / 0.75, 100 * (3661 / 4096 - 0.25) / 0.75
    write_runs(tmp_path, {(4, 16): [(small, "plateau")] * 2, (4, 1024): [(large, "plateau")] * 2,
                          (8, 16): SOLVED[:2], (8, 1024): [(small, "plateau")] * 2})
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    assert "9.99 points lower; a drop needs 10." in text
    assert "9.99 points higher; a rescue needs 10." in text
    assert "10.0 points lower; a drop needs 10" not in text


def test_not_stretched_is_said_only_when_something_was_compared(tmp_path):
    # Width 4 has runs on 16 animals only, width 8 on 1,024 only: nothing can be compared.
    write_runs(tmp_path, {(4, 16): SOLVED, (8, 1024): SOLVED})
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    assert "The models were not stretched" not in text
    assert "Nothing could be compared: no width has, with the same seed, a run on 16" in text
    # When something was compared, the sentence says on which vocabularies.
    write_runs(tmp_path, {(4, 64): SOLVED, (4, 256): SOLVED})
    _, text = judge(tmp_path)
    assert ("(larger vocabularies compared: 64 and 256 animals). The models were not "
            "stretched") in text


def test_what_the_letter_of_the_rule_allows_is_pointed_out(tmp_path):
    # Ten seeds, and the pattern shows in two of them: "at least 2 seeds" is met.
    write_runs(tmp_path / "ten", {
        (4, 16): [(100, "solved")] * 10, (8, 16): [(100, "solved")] * 10,
        (4, 1024): [(70, "plateau")] * 2 + [(99, "solved")] * 8, (8, 1024): [(99, "solved")] * 10})
    result, text = judge(tmp_path / "ten")
    assert result["met"] == {("equal", 1): [4]}
    assert ("Note on width 4: the three conditions hold in 2 of 10 seeds. The agreed rule asks "
            "for 2 seeds, however many were trained") in text
    # Three seeds: seed 0 drops on 64 animals only, seed 1 on 256 only.
    write_runs(tmp_path / "scattered", {
        (4, 16): SOLVED, (8, 16): SOLVED, (8, 64): SOLVED, (8, 256): SOLVED,
        (4, 64): [(80, "plateau"), (99, "solved"), (99, "solved")],
        (4, 256): [(99, "solved"), (80, "plateau"), (99, "solved")]})
    result, text = judge(tmp_path / "scattered")
    assert result["met"] == {("equal", 1): [4]}
    assert ("Note on width 4: no single vocabulary shows the drop and the rescue in 2 seeds; the "
            "seeds that count do so on different ones (64 and 256 animals).") in text
    assert "it does not ask for most of them" not in text  # 2 of 3 seeds is most of them


def test_no_width_high_and_no_drop_means_nothing_was_tested(tmp_path):
    low = [(40, "plateau"), (42, "plateau"), (41, "plateau")]
    write_runs(tmp_path, {(4, 16): low, (4, 1024): low, (8, 16): low, (8, 1024): low})
    result, text = judge(tmp_path)
    assert result["met"] == {("equal", 1): []}
    assert "No width reached 95 on 16 animals" in text
    assert "Nothing was tested: this is neither support for the hypothesis nor a refutation" in text
    assert "The models were not stretched" not in text


def test_the_rule_needs_the_small_vocabulary_and_a_larger_one(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (8, 16): SOLVED})
    write_runs(tmp_path, {(4, 256): SOLVED, (4, 1024): SOLVED}, usage="zipf")
    result, text = judge(tmp_path)
    assert result["met"] == {} and result["verdicts"] == []
    assert "it needs runs on 16 animals and on at least one larger vocabulary (found: 16)" in text
    assert "(found: 256, 1,024)" in text


def test_a_missing_run_on_16_animals_is_said_plainly(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): SOLVED, (8, 16): SOLVED[:1],
                          (8, 1024): SOLVED})
    result, text = judge(tmp_path)
    outcomes = [seed["outcome"] for seed in verdict_of(result, 8)["seeds"]]
    assert outcomes == ["no_drop", "no_small_run", "no_small_run"]
    assert "seed 1: no run on 16 animals, so nothing can be compared." in text


def test_print_rule_prints_the_rule_before_the_verdict(tmp_path, capsys):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): SOLVED, (8, 16): SOLVED, (8, 1024): SOLVED})
    R.print_rule(R.rule_met(R.load_runs(tmp_path)))
    printed = capsys.readouterr().out
    assert max(len(line) for line in printed.splitlines()) <= 100
    text = " ".join(printed.split())
    assert text.startswith("THE AGREED DECISION RULE (fixed before any experiment was run)")
    assert "(a) the width scores 95 or more on the 16-animal language" in text
    assert "scored at least 10 points below its own 16-animal score" in text
    assert "scores at least 10 points higher" in text
    assert "A run that ended by budget is NOT FINISHED and can never satisfy (b)." in text
    assert "Nor can a run that was given a fixed number of sets (--fixed-sets)" in text
    assert text.index("can never satisfy (b)") < text.index("Width 4: the rule is")


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------


def test_rank_groups_are_the_agreed_ones():
    assert R.rank_groups(1024) == [("1-16", 1, 16), ("17-64", 17, 64), ("65-256", 65, 256),
                                   ("257-1024", 257, 1024)]
    assert R.rank_groups(16) == [("1-16", 1, 16)]
    assert R.rank_groups(100) == [("1-16", 1, 16), ("17-64", 17, 64), ("65-100", 65, 100)]


def test_right_by_rank_group_counts_facts(tmp_path):
    summary = made_up_run(tmp_path, 64, 4, 0, 50, "plateau", usage="zipf", table_animals=64)
    summary["final"]["facts_right_by_animal"] = [4] * 16 + [1] * 24 + [3] * 24
    assert R.right_by_rank_group(summary) == {"1-16": 100.0, "17-64": 50.0}


def test_guessing_level_of_the_balanced_table_is_a_quarter(tmp_path):
    summary = made_up_run(tmp_path, 64, 4, 0, 50, "plateau", usage="zipf", table_animals=64)
    assert R.guessing_levels(summary["config"]) == {"1-16": 25.0, "17-64": 25.0}


def test_guessing_level_follows_an_edited_base_table(tmp_path, monkeypatch):
    import language

    browner = [(name, "brown", food, home, time)
               for name, _, food, home, time in language.BASE_ANIMALS]
    monkeypatch.setattr(language, "BASE_ANIMALS", browner)
    summary = made_up_run(tmp_path, 64, 4, 0, 50, "plateau", usage="zipf", table_animals=64)
    levels = R.guessing_levels(summary["config"])
    # Every base animal is brown now: one of its four facts is right without knowing anything.
    assert levels["1-16"] == pytest.approx(100 * (1 + 3 * 0.25) / 4)
    assert levels["17-64"] == 25.0


def test_spread_apart_keeps_order_and_distance():
    assert list(R.spread_apart([10, 50, 90], 6, 0, 100)) == [10, 50, 90]  # nothing to do
    placed = R.spread_apart([100, 99, 100, 20], 6, 0, 100)
    assert list(placed) == [94, 88, 100, 20]
    low = R.spread_apart([0, 1, 2], 6, 0, 100)
    assert list(low) == [0, 6, 12]
    # A crowd in the middle stays centered on where it was: half goes up, half goes down.
    assert list(R.spread_apart([50, 50, 50], 6, 0, 100)) == [44, 50, 56]
    assert list(R.spread_apart([25, 55, 56], 6, 0, 100)) == [25, 52.5, 58.5]


def test_mean_curve_stops_where_the_first_seed_stops(tmp_path):
    short = made_up_run(tmp_path, 1024, 4, 0, 60, "plateau", sets=8)
    long = made_up_run(tmp_path, 1024, 4, 1, 80, "budget", sets=37)  # 37 is not a grading count
    xs, ys = R.mean_curve([short, long])
    assert len(xs) == len(short["curve"]) == 6  # graded after 1, 2, 3, 4, 6 and 8 sets
    assert xs[-1] == pytest.approx(15 * 8)
    both_at_8 = [point["facts_known"] for run in (short, long) for point in run["curve"]
                 if point["sets"] == 8]
    assert ys[-1] == pytest.approx(np.mean(both_at_8))


def test_width_colors_are_the_reference_palette_in_its_order():
    reference = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    assert [R.WIDTH_COLORS[width] for width in [4, 8, 16, 32, 64, 128]] == reference
    assert R.width_color(8) == "#eb6834" and R.width_color(3) == R.OTHER_WIDTH_COLOR


# ------------------------------------------------------------------------------------------------
# The whole report: table, CSV files and charts
# ------------------------------------------------------------------------------------------------


def full_made_up_experiment(folder):
    """Made-up runs for every round of sweep.py, with all three endings among them.

    The numbers are invented to make the charts busy enough to check how they look. They are not
    results and say nothing about what the real experiment will show.
    """
    invented = {  # width -> the score on 16, 64, 256 and 1,024 animals
        "equal": {4: [99, 96, 68, 31], 8: [100, 99, 91, 58], 16: [100, 100, 99, 86],
                  32: [100, 100, 100, 99]},
        "zipf": {4: [98, 90, 61, 33], 8: [100, 97, 80, 52], 16: [100, 99, 93, 70],
                 32: [100, 100, 98, 84]},
        "scaleup": {8: [99, 84, 47, 20], 16: [100, 98, 79, 41], 32: [100, 100, 97, 73],
                    64: [100, 100, 100, 96]},
    }
    for kind, by_width in invented.items():
        usage = "equal" if kind == "scaleup" else kind
        seasons = 4 if kind == "scaleup" else 1
        for width, scores in by_width.items():
            for animals, score in zip([16, 64, 256, 1024], scores):
                for seed in range(3):
                    # Seeds differ a little, except where the score is close to solved.
                    facts = float(score + [0, -3, 2][seed] * (score < 95))
                    ended = "solved" if facts >= 97 else "plateau"
                    sets = 6 * (1024 // animals) if ended == "solved" else 1448
                    if ended == "plateau" and animals == 1024 and seed == 2:
                        ended, sets = "budget", 1200  # ran out of time, off the grading schedule
                    made_up_run(folder, animals, width, seed, facts, ended, usage=usage,
                                seasons=seasons, sets=sets)
    for width in [4, 32]:  # the table-check round
        for seed in range(3):
            made_up_run(folder, 16, width, seed, 99, "solved", table_animals=16, sets=6)
    # One check by training much longer, with --fixed-sets: saved beside the ordinary run.
    made_up_run(folder, 1024, 4, 0, 32, "fixed", fixed_sets=2000, sets=2000)


def is_picture(path):
    """True if the file is a PNG picture on the light chart background."""
    pixels = matplotlib.image.imread(path)
    corner = [round(float(channel) * 255) for channel in pixels[2, 2, :3]]
    return pixels.shape[0] > 100 and pixels.shape[1] > 100 and corner == [0xFC, 0xFC, 0xFB]


def test_report_writes_the_table_the_csv_files_and_every_chart(tmp_path, capsys):
    runs_dir, results_dir = tmp_path / "runs", tmp_path / "results"
    full_made_up_experiment(runs_dir)
    result = R.report(runs_dir, results_dir)
    printed = capsys.readouterr().out
    n_runs = 3 * 4 * 4 * 3 + 6 + 1

    # The table: one line per run, a run that ended by budget marked as not finished.
    assert f"RESULTS OF {n_runs} RUNS" in printed
    assert "equal usage, 4 seasons, word table of 1,024 animals" in printed
    assert "zipf usage, 1 season, word table of 1,024 animals" in printed
    assert "equal usage, 1 season, word table of 16 animals" in printed
    (line,) = [line for line in printed.splitlines() if line.endswith(" a1024_w4_equal_s2")]
    assert "NOT FINISHED" in line and "33.0" in line
    (line,) = [line for line in printed.splitlines() if line.endswith(" a16_w32_equal_s0")]
    assert "solved" in line and "100.0" in line
    # A run of a fixed length is shown as "fixed", and it did not run out of budget.
    (line,) = [line for line in printed.splitlines() if line.endswith(" a1024_w4_equal_s0_f2000")]
    assert " fixed " in line and "NOT FINISHED" not in line and "32.0" in line
    table = printed[: printed.index("FILES WRITTEN")]
    assert max(len(line) for line in table.splitlines()) <= 115

    # The short table: the mean over seeds, with a star where a run was not finished.
    means = table[table.index("FACTS KNOWN, THE MEAN OVER SEEDS"):]
    block = means[means.index("equal usage, 1 season"): means.index("equal usage, 4 seasons")]
    assert block.splitlines()[1].split() == ["width", "16", "64", "256", "1,024", "animal", "words"]
    assert block.splitlines()[2].split() == ["4", "99.0", "96.0", "67.7", "30.7*"]
    assert block.splitlines()[5].split() == ["32", "100.0", "100.0", "100.0", "99.0"]
    assert "word table of 16 animals" not in means  # the table-check runs are not part of it

    # The CSV files: every run, every grading.
    with open(results_dir / "summary.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == n_runs
    by_name = {row["name"]: row for row in rows}
    row = by_name["a1024_w4_equal_s2"]
    assert (row["facts_known"], row["stop_reason"], row["finished"]) == ("33.0", "budget", "no")
    assert int(row["parameters_total"]) == 880 + 1051 * 4
    assert float(row["bits_per_parameter"]) == pytest.approx(8192 / 5084, rel=1e-5)
    assert float(row["share_held"]) == pytest.approx(100 * 0.33**1.5, rel=1e-5)
    small_load = float(by_name["a16_w32_equal_s0"]["bits_per_parameter"])
    assert small_load == pytest.approx(128 / 62176, rel=1e-5)  # small numbers keep their digits
    assert row["unseen_known"] == ""  # every table animal is in use: nothing unseen to ask about
    assert row["pct_right_animals_257_1024"] != ""
    assert by_name["a16_w4_equal_s0"]["pct_right_animals_257_1024"] == ""
    with open(results_dir / "curves.csv", newline="") as file:
        gradings = list(csv.DictReader(file))
    of_run = [row for row in gradings if row["name"] == "a1024_w4_equal_s2"]
    assert [int(row["sets"]) for row in of_run][-3:] == [724, 1024, 1200]
    assert float(of_run[-1]["facts_known"]) == 33.0
    # The extra grading where the budget ended that run (1,200 sets) is marked as such.
    assert [row["scheduled"] for row in of_run][-3:] == ["True", "True", "False"]
    fixed = by_name["a1024_w4_equal_s0_f2000"]
    # A run of a fixed length is neither finished nor out of budget: it has a word of its own.
    assert (fixed["stop_reason"], fixed["finished"], int(fixed["sets"])) == ("fixed", "fixed", 2000)
    assert by_name["a16_w32_equal_s0"]["finished"] == "yes"

    # The charts: all four, and they are named in the printout.
    for name in R.CHART_FILES.values():
        assert is_picture(results_dir / name), name
        assert name in printed
    assert "not drawn" not in printed

    # The rule is printed last, and report() hands back what rule_met found.
    assert printed.index("FILES WRITTEN") < printed.index("THE AGREED DECISION RULE")
    assert set(result["met"]) == {("equal", 1), ("zipf", 1), ("equal", 4)}
    assert result["met"][("equal", 1)] == [4, 8, 16]
    one_line = " ".join(printed.split())  # the notes name whole folders and are wrapped
    assert "a16_w4_equal_t16_s0 has a word table of 16 animals" in one_line
    assert "a1024_w4_equal_s0_f2000 (facts known 32.0, ended as fixed) is left out." in one_line


def test_only_charts_with_runs_are_drawn_and_old_ones_are_removed(tmp_path, capsys):
    runs_dir, results_dir = tmp_path / "runs", tmp_path / "results"
    write_runs(runs_dir, {(4, 16): SOLVED, (32, 16): SOLVED})  # equal usage, one vocabulary
    results_dir.mkdir()
    (results_dir / R.CHART_FILES[4]).write_text("an old chart from other runs")
    R.report(runs_dir, results_dir)
    printed = capsys.readouterr().out
    assert sorted(path.name for path in results_dir.iterdir()) == [
        "chart2_learning.png", "chart3_knowledge_per_parameter.png", "curves.csv", "summary.csv"]
    assert "not drawn: chart 1 needs runs on at least two vocabulary sizes" in printed
    assert "not drawn: chart 4 needs zipf runs on more than 16 animals" in printed

    # With a second vocabulary chart 1 appears; with zipf runs chart 4 does.
    write_runs(runs_dir, {(4, 64): SOLVED})
    R.report(runs_dir, results_dir)
    assert (results_dir / R.CHART_FILES[1]).exists()
    assert not (results_dir / R.CHART_FILES[4]).exists()
    write_runs(runs_dir, {(4, 16): SOLVED, (4, 64): [(70, "plateau")] * 3}, usage="zipf",
               table_animals=64)
    R.report(runs_dir, results_dir)
    assert is_picture(results_dir / R.CHART_FILES[4])


def test_no_runs_is_said_plainly_and_nothing_is_written(tmp_path, capsys):
    assert R.report(tmp_path / "runs", tmp_path / "results") is None
    printed = capsys.readouterr().out
    assert "No finished runs" in printed and "NOTE" not in printed
    assert not (tmp_path / "results").exists()


def test_with_no_runs_the_files_of_an_earlier_report_are_pointed_out(tmp_path, capsys):
    runs_dir, results_dir = tmp_path / "runs", tmp_path / "results"
    write_runs(runs_dir, {(4, 16): SOLVED})
    R.report(runs_dir, results_dir)
    before = sorted(path.name for path in results_dir.iterdir())
    capsys.readouterr()
    assert R.report(tmp_path / "mistyped", results_dir) is None  # no such runs folder
    printed = " ".join(capsys.readouterr().out.split())
    assert "still holds the files of an earlier report (summary.csv, curves.csv, chart2" in printed
    assert "They were made from other runs" in printed
    assert sorted(path.name for path in results_dir.iterdir()) == before  # nothing was removed


def test_a_summary_that_was_cut_off_is_skipped_with_a_message(tmp_path, capsys):
    write_runs(tmp_path, {(4, 16): SOLVED})
    broken = tmp_path / "a64_w4_equal_s0"
    broken.mkdir()
    (broken / "summary.json").write_text('{"name": "a64_w4_equal_s0", "config": {"anim')
    runs = R.load_runs(tmp_path)
    assert [run["config"]["animals"] for run in runs] == [16, 16, 16]
    assert "a64_w4_equal_s0/summary.json: the file is not complete" in capsys.readouterr().out


def test_a_language_with_nothing_to_know_does_not_break_the_report(tmp_path, capsys):
    """One animal in use holds 0 bits of facts (there is nothing to tell apart). Its load is 0,
    which has no place on the log scale of chart 3: the run is left out of that chart."""
    runs_dir, results_dir = tmp_path / "runs", tmp_path / "results"
    write_runs(runs_dir, {(4, 16): SOLVED})
    summary = made_up_run(runs_dir, 1, 4, 0, 0, "budget")
    summary["final"]["bits_total"] = summary["final"]["bits_known"] = 0.0
    with open(runs_dir / summary["name"] / "summary.json", "w") as file:
        json.dump(summary, file)
    R.report(runs_dir, results_dir)
    assert "RESULTS OF 4 RUNS" in capsys.readouterr().out
    assert is_picture(results_dir / R.CHART_FILES[3])


def test_a_run_of_a_fixed_length_has_marks_of_its_own(tmp_path, capsys):
    """Width 4 on 256 animals exists only as runs of a fixed length. Nowhere may such a run look
    like one that reached a limit: not in the table of means, not in the CSV file, not as a dot."""
    runs_dir, results_dir = tmp_path / "runs", tmp_path / "results"
    write_runs(runs_dir, {(4, 16): SOLVED, (8, 16): SOLVED, (8, 256): SOLVED})
    for seed in range(3):
        made_up_run(runs_dir, 256, 4, seed, 50, "fixed", fixed_sets=20, sets=20)
    runs = R.load_runs(runs_dir)
    assert sorted({R.how_ended(run) for run in runs}) == ["finished", "fixed"]
    assert R.how_ended({"stop_reason": "budget"}) == "budget"
    assert R.how_ended({"stop_reason": "plateau"}) == "finished"
    R.report(runs_dir, results_dir)
    printed = capsys.readouterr().out
    means = printed[printed.index("FACTS KNOWN, THE MEAN OVER"): printed.index("FILES WRITTEN")]
    assert means.splitlines()[4].split() == ["4", "99.0", "50.0#"]
    assert means.splitlines()[5].split() == ["8", "99.0", "99.0"]
    assert "# = at least one of these runs had a fixed number of sets" in means
    with open(results_dir / "summary.csv", newline="") as file:
        finished = {row["name"]: row["finished"] for row in csv.DictReader(file)}
    assert finished["a256_w4_equal_s0_f20"] == "fixed" and finished["a16_w4_equal_s0"] == "yes"
    # The legend explains the half-filled dot only where there is one.
    assert [len(row) for row in R.fixed_key(runs)] == [1]
    assert R.fixed_key([run for run in runs if R.how_ended(run) != "fixed"]) == []


def test_the_table_of_means_says_how_many_runs_are_behind_a_mean(tmp_path, capsys):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 1024): [(70, "plateau"), (40, "plateau"),
                                                       (10, "plateau")], (8, 16): SOLVED})
    made_up_run(tmp_path, 1024, 8, 0, 99, "solved")  # one seed where the others have three
    groups, _ = R.comparison_groups(R.load_runs(tmp_path))
    assert R.runs_behind_the_means(groups[("equal", 1)]) == (
        "Each mean is taken over 3 runs, except: width 8 on 1,024 animals (1).")
    R.print_means(groups)
    assert "except: width 8 on 1,024 animals (1)." in capsys.readouterr().out
    del groups[("equal", 1)][(8, 1024, 0)]
    assert R.runs_behind_the_means(groups[("equal", 1)]) == "Each mean is taken over 3 runs."


def count_pixels(picture, color, box):
    """How many pixels inside box (left, top, right, bottom, as shares of the picture) have
    exactly this color, give or take a little."""
    pixels = matplotlib.image.imread(picture)[:, :, :3]
    height, width = pixels.shape[:2]
    left, top, right, bottom = box
    part = pixels[int(top * height): int(bottom * height), int(left * width): int(right * width)]
    wanted = np.array([int(color[i: i + 2], 16) / 255 for i in (1, 3, 5)])
    return int((np.abs(part - wanted).max(axis=2) < 0.06).sum())


def test_dots_of_different_widths_do_not_hide_each_other(tmp_path, monkeypatch):
    """Where every width scores the same, each must still be seen. Width 4 was NOT FINISHED on
    both vocabularies; widths 8 and 16 were solved with the same score of 100. Drawn on one spot,
    the dots of width 16 would cover all the others, hollow ones included."""
    monkeypatch.setattr(R, "SAVE_DPI", 100)
    for width, ended in [(4, "budget"), (8, "solved"), (16, "solved")]:
        write_runs(tmp_path / "runs", {(width, 16): [(100, ended)] * 3,
                                       (width, 64): [(100, ended)] * 3})
    groups, _ = R.comparison_groups(R.load_runs(tmp_path / "runs"))
    picture = tmp_path / "chart1.png"
    assert R.chart_more_words(groups, picture)
    below_the_legend = (0.0, 0.4, 1.0, 1.0)
    for width in [4, 8, 16]:
        assert count_pixels(picture, R.WIDTH_COLORS[width], below_the_legend) > 100, width


def test_every_run_gets_a_slot_of_its_own_beside_its_place_on_the_axis():
    places = [R.sideways(width_place, seed_place, n_widths=4, n_seeds=3, step=6)
              for width_place in range(4) for seed_place in range(3)]
    assert places == sorted(places) and len(set(places)) == 12
    assert places[0] == pytest.approx(-places[-1])  # centered on the place itself
    steps = np.diff(places)
    assert steps.min() == pytest.approx(6)  # seeds of one width: one step apart
    assert steps.max() == pytest.approx(6 * (1 + R.WIDTH_GAP))  # a little more between widths
    # One width and one seed: the dot stands on its place.
    assert R.sideways(0, 0, n_widths=1, n_seeds=1, step=9) == 0


def test_dots_move_closer_together_where_room_is_short():
    figure, (panel,) = R.new_figure(1, "title", "subtitle", [])
    panel.set_xlim(0, 3)
    assert R.dot_step(panel, [1], n_widths=4, n_seeds=3) == R.DOT_STEP  # one place: all the room
    assert R.dot_step(panel, [1, 2], n_widths=1, n_seeds=1) == R.DOT_STEP
    crowded = R.dot_step(panel, [1, 1.1, 2], n_widths=4, n_seeds=3)
    assert 0 < crowded < R.DOT_STEP
    R.plt.close(figure)


def test_a_file_that_is_not_the_summary_of_a_run_is_skipped_with_a_message(tmp_path, capsys):
    write_runs(tmp_path, {(4, 16): SOLVED})
    for name, content in [("empty_braces", "{}"), ("nothing", "null"), ("a_list", "[1, 2]")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(content)
    made_up_run(tmp_path, 64, 4, 0, 99, "solved")

    def without_final(summary):
        del summary["final"]

    change_summary(tmp_path, "a64_w4_equal_s0", without_final)
    runs = R.load_runs(tmp_path)
    printed = capsys.readouterr().out
    assert [run["config"]["animals"] for run in runs] == [16, 16, 16]
    assert printed.count("it is not a complete summary of a run") == 4
    assert "a64_w4_equal_s0/summary.json: it is not a complete summary of a run" in printed
    assert R.report(tmp_path, tmp_path / "results") is not None  # and the report goes on


def test_a_setting_this_code_does_not_know_does_not_stop_the_report(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 64): SOLVED})

    def saved_by_other_code(summary):
        summary["config"]["dropout"] = 0.1

    change_summary(tmp_path, "a64_w4_equal_s1", saved_by_other_code)
    runs = R.load_runs(tmp_path)
    assert R.default_name_of(runs[-2]) == "a64_w4_equal_s1"
    assert R.rule_met(runs)["met"] == {("equal", 1): []}


def test_runs_saved_by_different_versions_of_the_code_are_pointed_out(tmp_path):
    write_runs(tmp_path, {(4, 16): SOLVED, (4, 64): SOLVED})
    _, text = judge(tmp_path)
    assert "not all saved by the same version of the code" not in text

    def newer_code(summary):
        summary["code"] = "0123456789ab"

    change_summary(tmp_path, "a64_w4_equal_s0", newer_code)
    result, text = judge(tmp_path)
    assert ("NOTE: the runs of this group were not all saved by the same version of the code "
            "(mark not recorded in 5 runs: ") in text
    assert "mark 0123456789ab in 1 run: " in text
    assert result["met"] == {("equal", 1): []} and result["refused"] == {}  # a note, no more


def test_report_reads_what_train_really_writes(tmp_path, capsys):
    """Two real runs of about ten steps each: far too short to learn anything. This only checks
    that the summary.json of train.py has everything report.py reads."""
    threads_before = torch.get_num_threads()
    for animals in [16, 64]:
        run(RunConfig(animals=animals, table_animals=64, usage="zipf", set_sentences=240, width=4,
                      batch=8, context=32, max_sets=2, threads=1, out_dir=str(tmp_path / "runs")))
    torch.set_num_threads(threads_before)
    capsys.readouterr()  # drop what the two runs printed

    result = R.report(tmp_path / "runs", tmp_path / "results")
    printed = capsys.readouterr().out
    assert "RESULTS OF 2 RUNS" in printed
    assert printed.count("NOT FINISHED") >= 2  # both ended by budget (max_sets)
    for name in R.CHART_FILES.values():
        assert is_picture(tmp_path / "results" / name), name
    assert result["met"] == {("zipf", 1): []}
    assert "(that run was NOT FINISHED)" in " ".join(printed.split())
    with open(tmp_path / "results" / "summary.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["name"] for row in rows] == ["a16_w4_zipf_t64_s0", "a64_w4_zipf_t64_s0"]
    assert rows[1]["unseen_known"] == "" and rows[0]["unseen_known"] != ""

"""
Tests of train.py: the grading schedule, the stopping rule, and one training run.

Run them with:    pytest tests/test_train.py

The stopping rule is tested with hand-made histories: no model is needed to check a rule about
numbers. The training loop is tested with very short runs (a few dozen steps on a tiny language)
that write only to a temporary directory. These runs are far too short to learn anything; they
check the plumbing, not the model: that a run ends for a valid reason, that everything is saved
and can be loaded back, that models of different widths read exactly the same text, and that a
model is only ever graded at the end of a set.
"""

import argparse
import copy
import dataclasses
import json
import math
from itertools import islice

import numpy as np
import pytest
import torch

import train as T
from score import model_predictor, score
from train import RunConfig, decide_stop, default_name, grading_sets, run

# The default rule: solved_at 97, flat_points 2, flat_gap 0.05, flat_gap_bits 0.002,
# min_sightings 400, min_steps 2000, started_at 5.
RULE = RunConfig()

SCORE_KEYS = {"facts_known", "facts_known_by_use", "questions_right", "unseen_known", "words_known",
              "n_words", "loss", "floor", "gap", "bits_total", "bits_known", "headline",
              "headline_name"}
# What every entry of the learning curve says about where training stood, beside the scores.
PLACE_KEYS = {"sets", "steps", "minutes", "sightings", "scheduled"}


def tiny(out_dir, **changes):
    """Settings for a run of a few dozen steps: 16 animals, sets of 240 sentences, 3 sets."""
    settings = dict(animals=16, table_animals=32, set_sentences=240, width=8, batch=8, context=32,
                    max_sets=3, threads=1, out_dir=str(out_dir))
    settings.update(changes)
    return RunConfig(**settings)


@pytest.fixture(autouse=True)
def keep_thread_setting():
    """A run sets the number of processor threads; put it back so other tests are not slowed."""
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """One tiny run, shared by the tests that only look at what it left behind."""
    out_dir = tmp_path_factory.mktemp("runs")
    before = torch.get_num_threads()
    summary = run(tiny(out_dir))
    torch.set_num_threads(before)
    return summary, out_dir / summary["name"]


# ------------------------------------------------------------------------------------------------
# The grading schedule
# ------------------------------------------------------------------------------------------------


def test_grading_sets_start_as_agreed():
    assert list(islice(grading_sets(), 12)) == [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64]


def test_grading_sets_never_repeat_and_double_every_second_grading():
    sets = list(islice(grading_sets(), 40))
    assert all(later > earlier for earlier, later in zip(sets, sets[1:]))
    assert all(later <= 2 * earlier for earlier, later in zip(sets, sets[1:]))
    # Two gradings on is a doubling of training (within rounding), which the stopping rule uses.
    for earlier, later in zip(sets[1:], sets[3:]):
        assert 1.8 <= later / earlier <= 2.2
    assert sets[19:25] == [1024, 1448, 2048, 2896, 4096, 5793]


# ------------------------------------------------------------------------------------------------
# The stopping rule, on hand-made histories
# ------------------------------------------------------------------------------------------------


def history(facts, gaps, sightings_per_set=100, steps_per_set=500):
    """A made-up learning curve: one entry per grading, at the set counts of the real schedule."""
    sets = list(islice(grading_sets(), len(facts)))  # 1, 2, 3, 4, 6, 8, ...
    return [{"sets": s, "steps": s * steps_per_set, "facts_known": f, "gap": g,
             "sightings": s * sightings_per_set, "scheduled": True}
            for s, f, g in zip(sets, facts, gaps)]


# Five gradings (sets 1, 2, 3, 4, 6; 600 sightings and 3,000 steps at the end) on which nothing
# moves any more: +0.5 points over the last doubling, +1 over the one before, and the gap shrank
# by 2%.
FLAT_FACTS = [60, 60.5, 61, 61, 61.5]
FLAT_GAPS = [0.50, 0.50, 0.50, 0.495, 0.49]


def test_solved_when_the_last_two_gradings_are_at_or_above_solved_at():
    assert decide_stop(history([50, 98, 99], [1.0, 0.1, 0.05]), RULE) == "solved"
    assert decide_stop(history([97.0, 97.0], [0.1, 0.1]), RULE) == "solved"  # "at or above"


def test_one_good_grading_is_not_solved():
    assert decide_stop(history([99], [0.05]), RULE) is None
    assert decide_stop(history([50, 80, 99], [1.0, 0.4, 0.05]), RULE) is None
    assert decide_stop(history([99, 96.9, 99], [0.05, 0.1, 0.05]), RULE) is None
    assert decide_stop(history([99, 99, 96.9], [0.05, 0.05, 0.1]), RULE) is None


def test_a_clear_plateau():
    assert decide_stop(history(FLAT_FACTS, FLAT_GAPS), RULE) == "plateau"
    # A score that drifts down is flat too: it is certainly not gaining.
    assert decide_stop(history([61, 61, 60, 60, 59], FLAT_GAPS), RULE) == "plateau"


def test_still_rising_is_not_a_plateau():
    assert decide_stop(history([40, 45, 50, 55, 60], FLAT_GAPS), RULE) is None
    # Flat over the last doubling only: the doubling before it still gained 10 points.
    assert decide_stop(history([50, 55, 60, 60.5, 61], FLAT_GAPS), RULE) is None
    # Gaining exactly flat_points is not "fewer than flat_points".
    assert decide_stop(history([60, 60, 60, 60, 62.0], FLAT_GAPS), RULE) is None
    assert decide_stop(history([60, 60, 60, 60, 61.9], FLAT_GAPS), RULE) == "plateau"
    # The gain is measured over a DOUBLING (two gradings back), not since the last grading:
    # 1.5 points per grading look small, but they are 3 points per doubling.
    assert decide_stop(history([57, 58.5, 60, 61.5, 63], FLAT_GAPS), RULE) is None
    assert decide_stop(history([60, 60, 60, 61.5, 63], FLAT_GAPS), RULE) is None
    # The doubling before reaches back to the grading four back: 58 -> 60.5 is a gain of 2.5.
    assert decide_stop(history([58, 60, 60.5, 60.5, 61], FLAT_GAPS), RULE) is None


def test_flat_but_too_few_sightings_is_not_a_plateau():
    # 50 sightings per set: 300 at set 6, fewer than the 400 the rule asks for.
    assert decide_stop(history(FLAT_FACTS, FLAT_GAPS, sightings_per_set=50), RULE) is None
    # One grading later (set 8, 400 sightings) the same flat curve does count.
    longer = history(FLAT_FACTS + [61.5], FLAT_GAPS + [0.49], sightings_per_set=50)
    assert longer[-1]["sightings"] == 400
    assert decide_stop(longer, RULE) == "plateau"


def test_flat_but_too_few_steps_is_not_a_plateau():
    # 250 steps per set: 1,500 at set 6, fewer than the 2,000 the rule asks for, although every
    # fact has been seen 600 times. A model can look flat simply because it has hardly started.
    assert decide_stop(history(FLAT_FACTS, FLAT_GAPS, steps_per_set=250), RULE) is None
    # One grading later (set 8, exactly 2,000 steps) the same flat curve does count.
    longer = history(FLAT_FACTS + [61.5], FLAT_GAPS + [0.49], steps_per_set=250)
    assert longer[-1]["steps"] == 2000
    assert decide_stop(longer, RULE) == "plateau"
    # min_steps only ever holds a plateau back. A run that is solved is solved after any number
    # of steps.
    assert decide_stop(history([98, 99], [0.1, 0.05], steps_per_set=1), RULE) == "solved"


def test_a_model_that_has_not_begun_to_learn_facts_is_never_a_plateau():
    # This is a real case from calibration (1,024 animals, width 16): the model had learned the
    # sentence shapes and then stood still at 0.1 to 0.4 facts known, with the gap at the level
    # of a know-nothing model, from step 583 to step 2,385. It was saved as a plateau. Models of
    # other widths left the same stand-still after 600 to 4,800 steps and went on to learn
    # everything, so a stand-still at zero is not a limit.
    stalled = history([0.0, 0.0, 0.1, 0.2, 0.4], [0.35, 0.34, 0.34, 0.33, 0.33])
    assert stalled[-1]["steps"] >= RULE.min_steps and stalled[-1]["sightings"] >= RULE.min_sightings
    assert decide_stop(stalled, RULE) is None
    # Without the guard the very same curve is called a plateau.
    assert decide_stop(stalled, dataclasses.replace(RULE, started_at=0)) == "plateau"
    # Exactly at started_at the model counts as having begun; a flat curve there is a plateau.
    begun = history([5.0, 5.0, 5.0, 5.0, 5.0], [0.33, 0.33, 0.33, 0.33, 0.33])
    assert decide_stop(begun, RULE) == "plateau"
    just_below = history([4.9, 4.9, 4.9, 4.9, 4.9], [0.33, 0.33, 0.33, 0.33, 0.33])
    assert decide_stop(just_below, RULE) is None
    # started_at only ever holds a plateau back; a solved run is solved all the same.
    assert decide_stop(history([98, 99], [0.1, 0.05]), RULE) == "solved"


def test_flat_score_while_the_gap_is_still_falling_is_not_a_plateau():
    # The score stands still but the gap fell from 0.70 to 0.50 over the last doubling: the model
    # is still learning, even though no fact has tipped from wrong to right yet.
    assert decide_stop(history(FLAT_FACTS, [0.9, 0.8, 0.7, 0.6, 0.5]), RULE) is None
    # Only the LAST doubling of the gap is looked at.
    assert decide_stop(history(FLAT_FACTS, [0.9, 0.7, 0.50, 0.495, 0.49]), RULE) == "plateau"
    # It is a doubling (two gradings back): 3% per grading is 6% per doubling, which is a fall.
    assert decide_stop(history(FLAT_FACTS, [0.50, 0.50, 0.50, 0.485, 0.47]), RULE) is None


def test_a_gap_that_shrinks_by_a_large_share_but_by_almost_nothing_is_flat():
    # A run stuck just below solved_at with a tiny gap. Over the last doubling the gap fell from
    # 0.004 to 0.003: a quarter of it, but only 0.001 bits per word, which is less than the best
    # possible loss is accurate to. That is noise, not learning.
    stuck = [96.0] * 5
    tiny_gaps = [0.004, 0.004, 0.004, 0.0035, 0.003]
    assert decide_stop(history(stuck, tiny_gaps), RULE) == "plateau"
    # Without flat_gap_bits such a run could never be called flat.
    without = dataclasses.replace(RULE, flat_gap_bits=0)
    assert decide_stop(history(stuck, tiny_gaps), without) is None
    # A fall that is large both ways (a fifth of the gap, 0.004 bits) is still a fall ...
    assert decide_stop(history(stuck, [0.020, 0.020, 0.020, 0.018, 0.016]), RULE) is None
    # ... and a fall of many bits that is a small share of the gap is flat, as it always was.
    assert decide_stop(history(stuck, [1.00, 1.00, 1.00, 0.99, 0.98]), RULE) == "plateau"


def test_at_an_extra_grading_off_the_schedule_only_solved_is_asked():
    # Six flat gradings: at a scheduled grading that is a plateau.
    flat = history([60] * 6, [0.5] * 6)
    assert decide_stop(flat, RULE) == "plateau"
    # The same numbers when the last grading is the extra one of a run that ran out of budget
    # between two scheduled gradings. It is not a doubling away from the others: no plateau.
    flat[-1]["scheduled"] = False
    assert decide_stop(flat, RULE) is None
    # Solved is asked all the same: the last two gradings are at or above solved_at.
    good = history([50, 60, 70, 80, 98, 99], [0.5] * 6)
    good[-1]["scheduled"] = False
    assert decide_stop(good, RULE) == "solved"
    # One good grading is not enough there either.
    one_good = history([50, 60, 70, 80, 90, 99], [0.5] * 6)
    one_good[-1]["scheduled"] = False
    assert decide_stop(one_good, RULE) is None


def test_a_plateau_needs_five_gradings():
    for length in range(1, 5):
        short = history([60] * length, [0.5] * length, sightings_per_set=1000)
        assert decide_stop(short, RULE) is None
    assert decide_stop(history([60] * 5, [0.5] * 5, sightings_per_set=1000), RULE) == "plateau"


def test_a_gap_of_zero_does_not_break_the_rule():
    assert decide_stop(history(FLAT_FACTS, [0.0] * 5), RULE) == "plateau"
    assert decide_stop(history(FLAT_FACTS, [-0.001] * 5), RULE) == "plateau"


def test_solved_wins_over_plateau():
    assert decide_stop(history([98] * 5, [0.01] * 5), RULE) == "solved"


def test_the_settings_of_the_rule_are_used():
    flat = history(FLAT_FACTS, FLAT_GAPS)
    # With a stricter rule the gain of 0.5 points counts as a gain, and the 2% as a fall.
    assert decide_stop(flat, dataclasses.replace(RULE, flat_points=0.4)) is None
    assert decide_stop(flat, dataclasses.replace(RULE, flat_gap=0.01)) is None
    assert decide_stop(flat, dataclasses.replace(RULE, min_sightings=601)) is None
    assert decide_stop(flat, dataclasses.replace(RULE, min_steps=3001)) is None
    assert decide_stop(flat, dataclasses.replace(RULE, min_steps=3000)) == "plateau"
    assert decide_stop(flat, dataclasses.replace(RULE, solved_at=61)) == "solved"
    # The 2% by which the gap shrank are 0.01 bits: a fall under flat_gap=0.01, unless
    # flat_gap_bits asks for more bits than that.
    asks_for_more_bits = dataclasses.replace(RULE, flat_gap=0.01, flat_gap_bits=0.02)
    assert decide_stop(flat, asks_for_more_bits) == "plateau"


def test_decide_stop_is_pure():
    flat = history(FLAT_FACTS, FLAT_GAPS)
    untouched = copy.deepcopy(flat)
    assert decide_stop(flat, RULE) == decide_stop(flat, RULE) == "plateau"
    assert flat == untouched
    assert decide_stop([], RULE) is None


def test_every_ending_is_explained_in_plain_words():
    flat = history(FLAT_FACTS, FLAT_GAPS)
    assert T.explain_stop("solved", RULE, flat).startswith("SOLVED")
    plateau = T.explain_stop("plateau", RULE, flat)
    assert plateau.startswith("PLATEAU")
    # The sentence gives the practice the run had, and the least the rule asks for.
    assert "seen 600 times on average, in 3,000 training steps" in plateau
    assert "never called before 400 sightings and 2,000 steps" in plateau
    assert "by less than 5% or by less than 0.002 bits per word" in plateau
    assert "earlier in this run" not in plateau  # this score never fell
    budget = T.explain_stop("budget", RULE, flat)
    assert budget.startswith("BUDGET") and "NOT FINISHED" in budget
    assert "had not begun" not in budget  # 61.5 facts known: this run had begun long ago
    fixed = T.explain_stop("fixed", dataclasses.replace(RULE, fixed_sets=6), flat)
    assert fixed.startswith("FIXED") and "exactly 6 sets" in fixed
    assert "the stopping rule was not asked" in fixed and "never count as a plateau" in fixed


def test_a_run_out_of_budget_that_never_began_is_told_apart():
    """Below started_at the rule never calls a plateau (see the test above), so a model that
    never begins to learn facts ends by budget however long it trains. The sentence must say
    so: "give it more minutes" is the wrong advice for such a run."""
    never_begun = history([0, 0.4, 2, 2, 2], [0.5] * 5)
    assert decide_stop(never_begun, RULE) is None
    budget = T.explain_stop("budget", RULE, never_begun)
    assert budget.startswith("BUDGET") and "NOT FINISHED" in budget
    assert "It had not begun to learn facts: fewer than 5 out of 100 were known" in budget
    assert "a model that never begins ends by budget however long it trains" in budget
    # At exactly started_at the run has begun, and the ordinary sentence is given.
    begun = T.explain_stop("budget", RULE, history([0, 1, 3, 4, 5.0], [0.5] * 5))
    assert "had not begun" not in begun


def test_a_plateau_after_a_fall_says_that_the_score_fell():
    """The rule asks whether the score still gains. A score that falls passes that test, and is
    saved as a plateau; the sentence must not let it pass for a model that reached a ceiling."""
    falling = history([80, 70, 60, 50, 40], [0.5] * 5)
    assert decide_stop(falling, RULE) == "plateau"  # the agreed rule, to the letter
    plateau = T.explain_stop("plateau", RULE, falling)
    assert "Note: earlier in this run the score had been higher (80.0)" in plateau
    assert "part of this is a fall, not a ceiling" in plateau


# ------------------------------------------------------------------------------------------------
# Settings, names and flags
# ------------------------------------------------------------------------------------------------


def test_default_names():
    assert default_name(RunConfig()) == "a16_w32_equal_s0"
    zipf = RunConfig(animals=1024, width=4, usage="zipf", seed=2)
    assert default_name(zipf) == "a1024_w4_zipf_s2"
    assert default_name(RunConfig(animals=256, width=16, seasons=4)) == "a256_w16_equal_x4_s0"
    assert default_name(RunConfig(table_animals=16)) == "a16_w32_equal_t16_s0"
    # A run of a fixed length has a name of its own, so it never replaces the ordinary run.
    assert default_name(RunConfig(fixed_sets=500)) == "a16_w32_equal_s0_f500"
    assert default_name(RunConfig(seasons=4, table_animals=64, seed=1, fixed_sets=8)) == \
        "a16_w32_equal_x4_t64_s1_f8"


def test_the_agreed_defaults():
    config = RunConfig()
    assert (config.animals, config.table_animals, config.seasons) == (16, 1024, 1)
    assert (config.usage, config.set_sentences) == ("equal", 0)
    assert (config.width, config.layers, config.heads) == (32, 2, 2)
    assert (config.batch, config.context) == (64, 96)
    assert (config.learning_rate, config.warmup) == (0.01, 100)
    assert (config.solved_at, config.flat_points, config.flat_gap) == (97.0, 2.0, 0.05)
    assert (config.flat_gap_bits, config.min_sightings, config.min_steps) == (0.002, 400, 2000)
    assert (config.started_at, config.average) == (5.0, 0.998)
    assert (config.max_minutes, config.max_sets, config.fixed_sets) == (20.0, 100000, 0)
    assert (config.seed, config.threads, config.out_dir, config.name) == (0, 8, "runs", None)
    assert (T.EVAL_ROWS, T.WEIGHT_DECAY, T.GRADIENT_CLIP) == (256, 0.0, 1.0)


def test_every_setting_is_a_flag():
    parser = argparse.ArgumentParser()
    T.add_run_flags(parser)
    untouched = parser.parse_args([])
    for field in dataclasses.fields(RunConfig):
        assert hasattr(untouched, field.name), f"no flag for {field.name}"
    assert T.config_from_args(untouched) == RunConfig()
    changed = parser.parse_args(
        "--animals 64 --table-animals 128 --seasons 4 --usage zipf --set-sentences 960 --width 8 "
        "--layers 3 --heads 4 --seed 5 --batch 16 --context 48 --learning-rate 0.01 --warmup 7 "
        "--solved-at 90 --flat-points 1 --flat-gap 0.1 --flat-gap-bits 0.01 --min-sightings 50 "
        "--min-steps 300 --max-minutes 1.5 --max-sets 9 --fixed-sets 6 --threads 2 "
        "--out-dir elsewhere --name mine".split()
    )
    assert T.config_from_args(changed) == RunConfig(
        animals=64, table_animals=128, seasons=4, usage="zipf", set_sentences=960, width=8,
        layers=3, heads=4, seed=5, batch=16, context=48, learning_rate=0.01, warmup=7,
        solved_at=90, flat_points=1, flat_gap=0.1, flat_gap_bits=0.01, min_sightings=50,
        min_steps=300, max_minutes=1.5, max_sets=9, fixed_sets=6, threads=2, out_dir="elsewhere",
        name="mine",
    )


def test_the_average_follows_the_trained_model():
    config = RunConfig(average=0.998)
    language, model = T.build(dataclasses.replace(config, animals=16, table_animals=16, width=4))
    averaged = copy.deepcopy(model)
    start = [p.clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # the trained model has moved by 1 everywhere
    # First step: the average is taken over the steps so far, so it keeps only 1/10 of the start.
    T.update_average(averaged, model, 0, config)
    for mean, old in zip(averaged.parameters(), start):
        assert torch.allclose(mean, old + 0.9)
    # Late in a run it keeps 0.998 of itself and takes 0.002 of the trained model.
    T.update_average(averaged, model, 10**6, config)
    for mean, old in zip(averaged.parameters(), start):
        assert torch.allclose(mean, old + 0.9 + 0.002 * 0.1, atol=1e-6)
    # The trained model itself is never touched by the averaging.
    for now, old in zip(model.parameters(), start):
        assert torch.equal(now, old + 1.0)


def test_averaging_changes_what_is_graded_but_not_the_training(monkeypatch, tmp_path):
    def training_losses(average, where):
        losses = []
        real_step = T.train_step

        def recording_step(*arguments):
            losses.append(real_step(*arguments))
            return losses[-1]

        monkeypatch.setattr(T, "train_step", recording_step)
        summary = run(tiny(where, average=average))
        monkeypatch.setattr(T, "train_step", real_step)
        return losses, summary

    with_average, averaged_run = training_losses(0.998, tmp_path / "a")
    without, plain_run = training_losses(0.0, tmp_path / "b")
    assert with_average == without  # every training step saw the same model and the same batch
    assert averaged_run["final"]["loss"] != plain_run["final"]["loss"]  # but another model is graded


def test_learning_rate_warms_up_and_then_stays_constant():
    config = RunConfig(learning_rate=2e-3, warmup=100)
    assert T.learning_rate_at(0, config) == pytest.approx(2e-5)
    assert T.learning_rate_at(49, config) == pytest.approx(1e-3)
    assert T.learning_rate_at(99, config) == pytest.approx(2e-3)
    assert T.learning_rate_at(100, config) == T.learning_rate_at(10**6, config) == 2e-3
    assert T.learning_rate_at(0, dataclasses.replace(config, warmup=0)) == 2e-3


def test_the_seed_decides_the_invented_animals_and_the_model_has_a_row_for_every_word():
    language, model = T.build(RunConfig(animals=16, table_animals=64, seed=3, width=8))
    assert language.config.lexicon_seed == 3
    assert model.table.weight.shape == (language.vocab_size, 8) == (27 + 64, 8)
    assert model.config.context == RunConfig.context


def test_evaluation_rows_are_fixed_for_a_run():
    config = RunConfig(animals=16, table_animals=32, context=32)
    language, _ = T.build(config)
    first = T.evaluation_rows(language, config)
    again = T.evaluation_rows(language, config)
    assert first[0].shape == (T.EVAL_ROWS, 32)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))


# ------------------------------------------------------------------------------------------------
# A tiny run: how it ends and what it leaves behind
# ------------------------------------------------------------------------------------------------


def test_a_run_ends_for_a_valid_reason(smoke):
    summary, run_dir = smoke
    assert summary["stop_reason"] in T.STOP_REASONS
    assert T.STOP_REASONS == ["solved", "plateau", "budget", "fixed"]
    # Three sets are far too few to be solved or flat, so this run can only have run out of budget.
    assert summary["stop_reason"] == "budget"
    assert summary["sets"] == 3
    assert "NOT FINISHED" in summary["stop_explained"]


def test_summary_has_every_agreed_key(smoke):
    summary, run_dir = smoke
    assert set(summary) >= {"name", "config", "language", "parameters", "stop_reason", "sets",
                            "steps", "minutes", "sightings", "final", "curve"}
    assert summary["name"] == run_dir.name == "a16_w8_equal_t32_s0"
    assert set(summary["config"]) == {field.name for field in dataclasses.fields(RunConfig)}
    assert summary["config"]["name"] == summary["name"]
    assert set(summary["language"]) == {
        "vocab_size", "animals", "table_animals", "seasons", "usage", "set_sentences",
        "bits_total", "bits_per_animal", "floor",
    }
    assert summary["language"]["vocab_size"] == 27 + 32
    assert summary["language"]["set_sentences"] == 240
    assert summary["language"]["bits_total"] == pytest.approx(128)
    assert summary["language"]["bits_per_animal"] == pytest.approx(8)
    assert summary["language"]["floor"] == summary["final"]["floor"]
    # Width 8, 2 layers and a context of 32: the body of 2,528 at context 96, less 64 x 8.
    assert summary["parameters"] == {"body": 2528 - 64 * 8, "table": 59 * 8,
                                     "total": 2528 - 64 * 8 + 59 * 8}
    assert set(summary["final"]) == SCORE_KEYS | {"facts_right_by_animal"}
    assert len(summary["final"]["facts_right_by_animal"]) == 16


def test_the_curve_has_one_entry_per_grading(smoke):
    summary, run_dir = smoke
    curve = summary["curve"]
    assert [point["sets"] for point in curve] == [1, 2, 3]
    for point in curve:
        assert set(point) == SCORE_KEYS | PLACE_KEYS
        assert point["scheduled"] is True  # sets 1, 2 and 3 are all on the grading schedule
        assert point["sightings"] == pytest.approx(point["sets"] * 240 / 64)  # 64 facts
        assert math.isfinite(point["loss"]) and point["gap"] == point["loss"] - point["floor"]
    assert [point["steps"] for point in curve] == sorted(point["steps"] for point in curve)
    # The last entry of the curve is the final state of the run.
    last = curve[-1]
    assert all(last[key] == summary["final"][key] for key in SCORE_KEYS)
    assert (summary["sets"], summary["steps"]) == (last["sets"], last["steps"])
    assert (summary["minutes"], summary["sightings"]) == (last["minutes"], last["sightings"])
    assert summary["steps"] <= 100  # a smoke check, not an experiment


def test_summary_is_saved_as_json(smoke):
    summary, run_dir = smoke
    assert sorted(path.name for path in run_dir.iterdir()) == ["model.pt", "summary.json"]
    with open(run_dir / "summary.json") as file:
        assert json.load(file) == summary


def test_the_saved_model_reproduces_the_final_scores(smoke):
    summary, run_dir = smoke
    config, language, model = T.load_run(run_dir)
    assert dataclasses.asdict(config) == summary["config"]
    again = score(model_predictor(model), language, T.evaluation_rows(language, config))
    assert again["loss"] == pytest.approx(summary["final"]["loss"], abs=1e-6)
    assert again["floor"] == pytest.approx(summary["final"]["floor"], abs=1e-9)
    assert again["facts_right_by_animal"] == summary["final"]["facts_right_by_animal"]
    assert again["questions_right"] == summary["final"]["questions_right"]


def test_a_run_is_saved_with_a_mark_of_the_code_that_made_it(smoke, monkeypatch, tmp_path):
    summary, run_dir = smoke
    assert summary["code"] == T.CODE_MARK == T.code_mark()
    assert len(T.CODE_MARK) == 12 and set(T.CODE_MARK) <= set("0123456789abcdef")
    assert T.CODE_FILES == ["language.py", "model.py", "score.py", "train.py"]
    # The mark follows the content of the files: any change gives another mark.
    (tmp_path / "one.py").write_text("x = 1\n")
    monkeypatch.setattr(T, "CODE_FILES", [str(tmp_path / "one.py")])
    before = T.code_mark()
    assert T.code_mark() == before
    (tmp_path / "one.py").write_text("x = 1  # a comment\n")
    assert T.code_mark() != before


def test_a_run_whose_set_size_is_refused_today_can_still_be_opened(smoke, tmp_path):
    """Asking or probing a trained model does not depend on how its training text was cut into
    sets. A run saved when the code still accepted its set size must stay readable."""
    summary, run_dir = smoke
    saved = torch.load(run_dir / "model.pt", weights_only=True)
    saved["config"]["set_sentences"] = 250  # no multiple of 16 animals: refused for a new run
    saved["config"]["a_setting_of_other_code"] = 1  # and a setting this RunConfig does not have
    (tmp_path / "old").mkdir()
    torch.save(saved, tmp_path / "old" / "model.pt")
    with pytest.raises(ValueError, match="must be a multiple of the number of animals"):
        T.build(tiny(tmp_path, set_sentences=250))
    config, language, model = T.load_run(tmp_path / "old")
    assert config.set_sentences == 250 and language.config.animals == 16
    result = score(model_predictor(model), language, T.evaluation_rows(language, config))
    assert result["loss"] == pytest.approx(summary["final"]["loss"], abs=1e-5)


def test_the_same_settings_give_the_same_run(smoke, tmp_path):
    summary, run_dir = smoke
    again = run(tiny(tmp_path))
    assert again["final"]["loss"] == pytest.approx(summary["final"]["loss"], abs=1e-6)
    assert again["steps"] == summary["steps"]
    other_seed = run(tiny(tmp_path, seed=1))
    assert other_seed["final"]["loss"] != pytest.approx(summary["final"]["loss"], abs=1e-6)


# ------------------------------------------------------------------------------------------------
# What a run reads: the same sets at every width, and whole sets only
# ------------------------------------------------------------------------------------------------


def record_training(monkeypatch):
    """Keep a copy of every batch that a run trains on. Returns the list that fills up."""
    batches = []
    real_step = T.train_step

    def recording_step(model, optimizer, inputs, targets, rate):
        batches.append((inputs.numpy().copy(), targets.numpy().copy()))
        return real_step(model, optimizer, inputs, targets, rate)

    monkeypatch.setattr(T, "train_step", recording_step)
    return batches


def same_batches(some, others):
    return len(some) == len(others) and all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(some, others)
    )


@pytest.mark.parametrize("usage", ["equal", "zipf"])
def test_two_widths_read_exactly_the_same_sets(monkeypatch, tmp_path, usage):
    batches = record_training(monkeypatch)
    read = {}
    for name, changes in [("narrow", dict(width=4)), ("wide", dict(width=16)),
                          ("other seed", dict(width=4, seed=1))]:
        batches.clear()
        summary = run(tiny(tmp_path, usage=usage, **changes))
        assert summary["config"]["width"] == changes["width"]
        read[name] = list(batches)
    assert len(read["narrow"]) > 6  # several batches in each of the three sets
    assert same_batches(read["narrow"], read["wide"])  # same token arrays, same order
    assert not same_batches(read["narrow"], read["other seed"])  # the check can tell runs apart

    # And those batches are exactly sets 0, 1, 2 of the language, each packed and cut into batches.
    config = tiny(tmp_path, usage=usage, width=4)
    language, _ = T.build(config)
    expected = []
    for set_index in range(3):
        sentences = language.make_set(set_index, data_seed=config.seed)
        inputs, targets = language.pack_rows(sentences, config.context)
        for first in range(0, len(inputs), config.batch):
            last = first + config.batch
            expected.append((inputs[first:last], targets[first:last]))
    assert same_batches(read["narrow"], expected)


def test_grading_and_stopping_happen_only_at_set_ends(monkeypatch, tmp_path):
    # Zipf usage with an odd set size: the sets differ in length by a few sentences and the last
    # batch of a set is smaller than the others, so a count that is "a whole number of sets" is
    # not reached by accident.
    config = tiny(tmp_path, usage="zipf", set_sentences=250, max_sets=5)
    language, _ = T.build(config)
    ends_a_sentence = [language.word_id["."], language.yes_id, language.no_id]
    batches = record_training(monkeypatch)
    read_at_grading = []  # (sentences, rows, steps) trained on when each grading took place
    real_score = T.score

    def recording_score(predict, language, eval_rows):
        sentences = sum(int(np.isin(targets, ends_a_sentence).sum()) for _, targets in batches)
        rows = sum(len(inputs) for inputs, _ in batches)
        read_at_grading.append((sentences, rows, len(batches)))
        return real_score(predict, language, eval_rows)

    monkeypatch.setattr(T, "score", recording_score)
    summary = run(config)

    # What whole sets hold, counted from the language itself.
    set_sizes, set_rows = [], []
    for set_index in range(5):
        sentences = language.make_set(set_index, data_seed=config.seed)
        set_sizes.append(len(sentences))
        set_rows.append(len(language.pack_rows(sentences, config.context)[0]))
    assert len(set(set_sizes)) > 1 and any(rows % config.batch for rows in set_rows)
    whole_sets = {k: (sum(set_sizes[:k]), sum(set_rows[:k]),
                      sum(math.ceil(rows / config.batch) for rows in set_rows[:k]))
                  for k in range(1, 6)}

    # Graded at sets 1, 2, 3, 4 (the schedule) and at 5, where the budget of 5 sets ended the run.
    graded_at = [point["sets"] for point in summary["curve"]]
    assert graded_at == [1, 2, 3, 4, 5]
    # The curve says which grading was the extra one.
    assert [point["scheduled"] for point in summary["curve"]] == [True, True, True, True, False]
    assert read_at_grading == [whole_sets[k] for k in graded_at]
    assert [point["steps"] for point in summary["curve"]] == [whole_sets[k][2] for k in graded_at]
    assert summary["stop_reason"] == "budget" and summary["sets"] == 5
    # Nothing was trained after the last grading: the saved model is the graded model.
    assert (summary["steps"], len(batches)) == (whole_sets[5][2], whole_sets[5][2])
    assert summary["sightings"] == pytest.approx(language.sightings(5)[1])
    assert summary["steps"] <= 100


# ------------------------------------------------------------------------------------------------
# The endings, reached through the real loop
# ------------------------------------------------------------------------------------------------


def pretend(monkeypatch, facts_by_grading=None, **fixed):
    """Let the real grading run, then overwrite some of its numbers (to steer the stopping rule
    without training a model for hours).

    fixed: numbers that are the same at every grading. facts_by_grading: a list with the
    facts_known to pretend at the first grading, the second, and so on.
    """
    real_score = T.score
    gradings_so_far = []

    def steered_score(predict, language, eval_rows):
        result = real_score(predict, language, eval_rows)
        result.update(fixed)
        if facts_by_grading is not None:
            result["facts_known"] = facts_by_grading[len(gradings_so_far)]
        gradings_so_far.append(result)
        return result

    monkeypatch.setattr(T, "score", steered_score)


def test_a_solved_run_stops_at_the_second_good_grading(monkeypatch, tmp_path):
    pretend(monkeypatch, facts_known=100.0)
    summary = run(tiny(tmp_path, max_sets=50))
    assert summary["stop_reason"] == "solved"
    assert [point["sets"] for point in summary["curve"]] == [1, 2]
    assert summary["stop_explained"].startswith("SOLVED")


def test_a_flat_run_stops_as_a_plateau_at_the_fifth_grading(monkeypatch, tmp_path):
    # The two guards of the rule are set low enough here for a run of a few dozen steps. (The
    # scores are made up, so this checks the loop, not what a model can learn.)
    pretend(monkeypatch, facts_known=50.0, gap=0.3)
    summary = run(tiny(tmp_path, max_sets=50, min_sightings=20, min_steps=0))
    assert summary["stop_reason"] == "plateau"
    assert [point["sets"] for point in summary["curve"]] == [1, 2, 3, 4, 6]
    assert summary["sightings"] == pytest.approx(6 * 240 / 64) and summary["sightings"] >= 20
    assert summary["steps"] <= 100


def test_a_flat_run_is_not_called_a_plateau_before_min_steps(monkeypatch, tmp_path):
    # The same flat run with min_steps=40. At the fifth grading (set 6) it has taken 35 steps,
    # so the rule must wait for the next grading (set 8, 47 steps).
    pretend(monkeypatch, facts_known=50.0, gap=0.3)
    summary = run(tiny(tmp_path, max_sets=50, min_sightings=20, min_steps=40))
    assert summary["stop_reason"] == "plateau"
    assert [point["sets"] for point in summary["curve"]] == [1, 2, 3, 4, 6, 8]
    assert summary["curve"][-2]["steps"] < 40 <= summary["curve"][-1]["steps"] == summary["steps"]
    assert "never called before 20 sightings and 40 steps" in summary["stop_explained"]
    assert summary["steps"] <= 100


def test_a_flat_run_with_too_little_practice_runs_on_to_its_budget(monkeypatch, tmp_path):
    pretend(monkeypatch, facts_known=50.0, gap=0.3)
    # min_sightings 400 and min_steps 2,000; 7 sets give only 26 sightings and about 40 steps.
    summary = run(tiny(tmp_path, max_sets=7))
    assert summary["stop_reason"] == "budget"
    assert [point["sets"] for point in summary["curve"]] == [1, 2, 3, 4, 6, 7]
    assert [point["scheduled"] for point in summary["curve"]] == [True] * 5 + [False]


def test_a_run_out_of_budget_whose_last_two_gradings_are_solved_is_saved_as_solved(monkeypatch,
                                                                                    tmp_path):
    # The budget of 7 sets ends this run between two scheduled gradings (6 and 8), so it is
    # graded once more at set 7. The gradings at sets 6 and 7 are both at or above solved_at:
    # the run is solved, and must not be saved as "budget: NOT FINISHED".
    pretend(monkeypatch, facts_by_grading=[10, 20, 30, 40, 100, 100])
    summary = run(tiny(tmp_path, max_sets=7))
    assert [point["sets"] for point in summary["curve"]] == [1, 2, 3, 4, 6, 7]
    assert summary["stop_reason"] == "solved" and summary["sets"] == 7
    assert summary["stop_explained"].startswith("SOLVED")
    assert decide_stop(summary["curve"], RULE) == "solved"  # the saved curve says the same


def test_one_solved_grading_at_the_end_of_the_budget_is_still_not_finished(monkeypatch, tmp_path):
    pretend(monkeypatch, facts_by_grading=[10, 20, 30, 40, 50, 100])
    summary = run(tiny(tmp_path, max_sets=7))
    assert summary["stop_reason"] == "budget" and "NOT FINISHED" in summary["stop_explained"]


def test_a_flat_run_is_never_called_a_plateau_at_the_extra_grading(monkeypatch, tmp_path):
    # Flat, with every guard of the rule switched off, so a plateau is called as soon as there
    # are five gradings: at set 6 if the run gets that far. With a budget of 5 sets the fifth
    # grading is the extra one at set 5, and there the rule must not call a plateau.
    pretend(monkeypatch, facts_known=50.0, gap=0.3)
    summary = run(tiny(tmp_path, max_sets=5, min_sightings=0, min_steps=0))
    assert [point["sets"] for point in summary["curve"]] == [1, 2, 3, 4, 5]
    assert summary["stop_reason"] == "budget"


def test_fixed_sets_trains_exactly_that_many_sets_and_asks_no_rule(monkeypatch, tmp_path):
    # Every grading reads 100, which would end an ordinary run as "solved" after two sets, and
    # both budgets (2 sets, 0 minutes) would end it even sooner. A run of a fixed length asks
    # neither: it trains for its 5 sets.
    batches = record_training(monkeypatch)
    pretend(monkeypatch, facts_known=100.0)
    config = tiny(tmp_path, fixed_sets=5, max_sets=2, max_minutes=0.0)
    summary = run(config)
    assert summary["stop_reason"] == "fixed" and summary["sets"] == 5
    assert summary["stop_explained"].startswith("FIXED LENGTH")
    assert summary["sightings"] == pytest.approx(5 * 240 / 64)
    # Graded at the scheduled set counts and at the last set, which is marked as the extra one.
    assert [point["sets"] for point in summary["curve"]] == [1, 2, 3, 4, 5]
    assert [point["scheduled"] for point in summary["curve"]] == [True, True, True, True, False]
    # The batches it read are exactly the first five sets of the language.
    language, _ = T.build(config)
    rows_of_five_sets = sum(len(language.pack_rows(language.make_set(k, config.seed),
                                                   config.context)[0]) for k in range(5))
    assert sum(len(inputs) for inputs, _ in batches) == rows_of_five_sets
    assert summary["steps"] == len(batches) <= 100
    # It has a name of its own, so the ordinary run of the same settings is not replaced.
    assert summary["name"] == "a16_w8_equal_t32_s0_f5"
    assert (tmp_path / "a16_w8_equal_t32_s0_f5" / "summary.json").exists()


def test_fixed_sets_that_end_on_the_schedule_add_no_extra_grading(tmp_path):
    summary = run(tiny(tmp_path, fixed_sets=4))
    assert summary["stop_reason"] == "fixed"
    assert [point["sets"] for point in summary["curve"]] == [1, 2, 3, 4]
    assert all(point["scheduled"] for point in summary["curve"])


def test_the_real_loop_calls_no_plateau_while_the_model_has_hardly_started(tmp_path):
    """Nothing is faked here: a real run, graded for real, on a word table of 16 animals.

    With that table one set is 960 sentences, which fit in a single batch: one set is ONE
    training step. Before min_steps existed, this very run was saved as "plateau" after 32
    steps, a third of the way through the warm-up, with a score of 0 and the sentence "the
    model has hit its limit". It had hardly started. (64 steps: plumbing, not an experiment.)
    """
    config = RunConfig(animals=16, table_animals=16, width=4, max_sets=64, threads=1,
                       learning_rate=2e-3, out_dir=str(tmp_path))  # the rate of that first run
    summary = run(config)
    curve = summary["curve"]
    assert [point["steps"] for point in curve] == [point["sets"] for point in curve]
    assert summary["steps"] == 64 < config.warmup < config.min_steps  # still inside the warm-up
    assert summary["stop_reason"] == "budget"  # NOT a plateau

    def first_plateau(rule):
        """The step count at which this rule would first have called a plateau on this curve."""
        for gradings in range(1, len(curve) + 1):
            if decide_stop(curve[:gradings], rule) == "plateau":
                return curve[gradings - 1]["steps"]
        return None

    # The check can tell: on this very curve the rule without its two guards (min_steps and
    # started_at) calls a plateau at step 32, and with min_steps=40 alone at the first grading
    # after step 40. Either guard on its own is enough to hold the plateau back here: the model
    # knows 0 facts, so it has not begun to learn.
    assert first_plateau(config) is None
    assert first_plateau(dataclasses.replace(config, min_steps=0, started_at=0)) == 32
    assert first_plateau(dataclasses.replace(config, min_steps=40, started_at=0)) == 45
    assert first_plateau(dataclasses.replace(config, min_steps=0)) is None


def test_running_out_of_time_ends_the_run_at_the_next_set_end(tmp_path):
    summary = run(tiny(tmp_path, max_sets=50, max_minutes=0.0))
    assert summary["stop_reason"] == "budget"
    assert summary["sets"] == 1 and len(summary["curve"]) == 1


def test_a_run_that_breaks_down_is_an_error_and_saves_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "train_step", lambda *arguments: float("nan"))
    with pytest.raises(RuntimeError, match="learning-rate"):
        run(tiny(tmp_path))
    assert list(tmp_path.iterdir()) == []

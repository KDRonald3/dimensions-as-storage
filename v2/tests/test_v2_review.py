"""Second checks of v2/, written separately from the first tests (a review).

These tests do not use the truth check or the oracle of the other test files. They read every
sentence word by word with their own small reader, and they build their own perfect exam taker
from the facts alone.
"""

import json
import subprocess
import sys
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pytest
import torch

from v2 import lang as L
from v2 import net, run

PROJECT = Path(__file__).resolve().parents[2]

# The kinds of fact as PLAN.md B.1 writes them (only "mice" is replaced, by "gnats").
PLAN_KINDS = (
    "are: red brown green black | eat: meat plants insects seeds | live in: woods fields rivers "
    "caves | wake at: dawn noon dusk night | fear: fire water noise light | like: sun rain snow "
    "wind | carry: sticks stones leaves shells | drink: dew sap milk nectar | climb: trees rocks "
    "walls hills | dig: holes tunnels pits dens | catch: flies moths gnats fish | build: nests "
    "dams webs mounds | follow: tracks scents sounds stars | avoid: roads farms towns camps | "
    "cross: bridges marshes dunes ice | guard: eggs cubs paths pools | smell: smoke rot musk salt "
    "| hear: thunder steps calls wings | chew: bark roots grass bones | watch: clouds shadows "
    "flames waves | dodge: traps nets snares hooks | seek: shade warmth shelter company | gather: "
    "twigs moss feathers pebbles | hoard: nuts berries grain honey | visit: ponds cliffs meadows "
    "ruins | shed: fur skin horns claws | mark: logs stumps boulders fences | raid: hives barns "
    "orchards burrows | ride: currents gusts tides floods | dread: frost drought storms eclipses "
    "| swallow: gravel sand mud clay | trail: herds flocks swarms packs"
)


def plan_kinds():
    kinds = []
    for part in PLAN_KINDS.split("|"):
        link, values = part.split(":")
        kinds.append((link.split(), values.split()))
    return kinds


def test_words_are_those_of_the_plan():
    assert [(list(a), list(b)) for a, b in L.KINDS] == plan_kinds()
    lang = L.Language(1024, 32)
    assert len(set(lang.words)) == len(lang.words)  # every token a different word
    names = set(L.creature_names())
    assert not any(word in names for word in lang.words[:lang.creature_base])


# ------------------------------------------------------------------------------------------------
# A reader of my own, and the truth of a sentence
# ------------------------------------------------------------------------------------------------


def read(lang, row):
    """(type, creatures, kind, value) of a token row, read word by word. Fails on any error."""
    tokens = [int(t) for t in row]
    n = tokens.index(0) if 0 in tokens else len(tokens)
    assert all(t == 0 for t in tokens[n:]), "padding in the middle"
    tokens = tokens[:n]
    words = [lang.words[t] for t in tokens]
    creature = [t - lang.creature_base if t >= lang.creature_base else None for t in tokens]
    value_of = {int(lang.value_ids[k, v]): (k, v) for k in range(lang.kinds) for v in range(4)}
    question = words[-1] in ("yes", "no")
    end = n - 2 if question else n - 1
    assert words[end] == ("?" if question else ".")
    kind, value = value_of[tokens[end - 1]]
    link = plan_kinds()[kind][0]
    places = [i for i in range(n) if creature[i] is not None]
    listed = [creature[i] for i in places]
    assert places == list(range(places[0], places[0] + 2 * len(places) - 1, 2))
    between = [words[i] for i in range(places[0] + 1, places[-1], 2)]
    assert between == ([","] * (len(listed) - 2) + ["and"] if len(listed) > 1 else [])
    assert 1 <= len(listed) <= 4 and len(set(listed)) == len(listed)
    before, after = words[:places[0]], words[places[-1] + 1:end - 1]
    if question:
        assert before == (["are"] if kind == 0 else ["do"])
        assert after == ([] if kind == 0 else link)
        return words[-1], listed, kind, value
    assert before == []
    if after == link:
        return "statement", listed, kind, value
    assert after == (["are", "not"] if kind == 0 else ["do", "not"] + link), words
    return "not", listed, kind, value


def true(lang, sort, listed, kind, value):
    has = [lang.facts[c, kind] == value for c in listed]
    return {"statement": all(has), "yes": all(has), "not": not any(has),
            "no": not all(has)}[sort]


@pytest.mark.parametrize("creatures,kinds,usage", [(16, 4, "zipf"), (16, 32, "equal"),
                                                   (64, 8, "zipf"), (1024, 4, "zipf")])
def test_every_training_sentence_is_true_and_avoids_held_pairs(creatures, kinds, usage):
    lang = L.Language(creatures, kinds, usage)
    held = {tuple(pair) for pair in lang.held_pairs.tolist()}
    seen = Counter()
    for step in range(30):
        for row in lang.sample_batch(7, step):
            sort, listed, kind, value = read(lang, row)
            assert true(lang, sort, listed, kind, value), lang.decode(row)
            for a, b in combinations(sorted(listed), 2):
                assert (a, b) not in held, lang.decode(row)
            seen[sort] += 1
    total = sum(seen.values())
    for sort, chance in (("statement", 0.4), ("not", 0.2), ("yes", 0.2), ("no", 0.2)):
        assert abs(seen[sort] / total - chance) < 0.02


# ------------------------------------------------------------------------------------------------
# The sampler's chances
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("redraw_rounds", [3, 0])
def test_different_creatures_by_use_have_the_exact_chances(monkeypatch, redraw_rounds):
    """Drawing k different creatures by use = one after another, each from the ones not yet
    taken, in proportion to the weights. Checked against the exact chance of every order,
    also for the short cut taken after REDRAW_ROUNDS rounds."""
    monkeypatch.setattr(L, "REDRAW_ROUNDS", redraw_rounds)
    weights = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    groups = L.WeightedGroups([np.arange(5)], weights)
    rng = np.random.default_rng(0)
    draws = 200_000
    for k in (2, 3):
        chosen = L.draw_different(rng, groups, np.zeros((draws, k), dtype=np.int64))
        counts = Counter(map(tuple, chosen.tolist()))
        for order in permutations(range(5), k):
            chance, left = 1.0, 1.0
            for c in order:
                chance *= weights[c] / left
                left -= weights[c]
            assert abs(counts[order] / draws - chance) < 5 * np.sqrt(chance / draws) + 1e-4


@pytest.mark.parametrize("usage", ["equal", "zipf"])
def test_sampler_with_16_creatures(usage):
    """With 16 creatures some long lists cannot avoid a held-back pair. Only those are made
    shorter: one-creature lists keep their 40%, and no list is made shorter for nothing."""
    lang = L.Language(16, 4, usage)
    drawn = lang.draw_sentences(np.random.default_rng(3), 100_000)
    lengths, types = drawn["lengths"], drawn["types"]
    assert abs((lengths == 1).mean() - 0.4) < 0.006
    assert abs(np.bincount(types, minlength=4) / len(types) - L.TYPE_CHANCES).max() < 0.006
    # A "yes" list of 4 needs the 4 creatures that share one value. Count how many of the 16
    # (kind, value) groups contain a held-back pair: those can never give a "yes" list of 4.
    bad = sum(lang.has_held_pair(np.nonzero(lang.facts[:, kind] == value)[0][None])[0]
              for kind in range(4) for value in range(4))
    yes = types == L.YES_QUESTION
    expected_four = 0.2 * (1 - bad / 16)
    assert abs((lengths[yes] == 4).mean() - expected_four) < 0.015
    # Kinds stay at equal chance.
    assert abs(np.bincount(drawn["kinds"], minlength=4) / len(types) - 0.25).max() < 0.006


# ------------------------------------------------------------------------------------------------
# The exam, with an exam taker of my own
# ------------------------------------------------------------------------------------------------


def perfect_taker(lang, spoil_no_one=False):
    """Knows every fact. Reads each prompt word by word. With spoil_no_one it answers "yes"
    to the "no" questions in which exactly one creature lacks the value."""
    link_of = {tuple(link): k for k, (link, _) in enumerate(plan_kinds()[:lang.kinds])}
    value_of = {int(lang.value_ids[k, v]): (k, v) for k in range(lang.kinds) for v in range(4)}

    def predict(prompts, last):
        scores = np.random.default_rng(1).random((len(prompts), lang.n_out))  # noise elsewhere
        for i, (row, end) in enumerate(zip(prompts, last)):
            tokens = [int(t) for t in row[:end + 1]]
            assert 0 not in tokens
            words = [lang.words[t] for t in tokens]
            creatures = [t - lang.creature_base for t in tokens if t >= lang.creature_base]
            if words[-1] == "?":
                kind, value = value_of[tokens[-2]]
                lacking = sum(lang.facts[c, kind] != value for c in creatures)
                say_yes = lacking == 0 or (spoil_no_one and lacking == 1)
                scores[i, lang.yes_id - 1] = 9.0 if say_yes else 2.0
                scores[i, lang.no_id - 1] = 2.0 if say_yes else 9.0
                continue
            assert len(creatures) == 1 and tokens[0] >= lang.creature_base
            negated = "not" in words
            kind = link_of[tuple(w for w in words[1:] if w not in ("do", "not"))]
            for value in range(4):
                if_true = lang.facts[creatures[0], kind] == value
                scores[i, lang.value_ids[kind, value] - 1] = (
                    (-9.0 if if_true else 2.0) if negated else (9.0 if if_true else 2.0))
        return scores

    return predict


@pytest.mark.parametrize("creatures,kinds", [(16, 4), (16, 32), (256, 8)])
def test_exam_scores(creatures, kinds):
    lang = L.Language(creatures, kinds)
    best = L.grade(lang, perfect_taker(lang))
    for key in ("facts", "questions_one", "questions_new", "questions_seen", "not_known"):
        assert best[key] == 100.0, key
    assert best["facts_by_kind"] == [100.0] * kinds

    def negative(prompts, last):
        return -perfect_taker(lang)(prompts, last)

    worst = L.grade(lang, negative)
    for key in ("facts", "questions_one", "questions_new", "questions_seen", "not_known"):
        assert worst[key] == 0.0, key

    def always(word):
        def predict(prompts, last):
            scores = np.zeros((len(prompts), lang.n_out))
            scores[:, lang.word_id[word] - 1] = 1.0
            return scores
        return predict

    for word in ("yes", "no"):
        constant = L.grade(lang, always(word))
        assert constant["questions_one"] == 0.0
        assert constant["questions_new"] == 0.0 and constant["questions_seen"] == 0.0
    flat = L.grade(lang, lambda prompts, last: np.zeros((len(prompts), lang.n_out)))
    assert flat["facts"] == 25.0 and flat["not_known"] == 25.0

    # The split by f: spoiling only the f = 1 "no" questions shows up only there.
    spoilt = L.grade(lang, perfect_taker(lang, spoil_no_one=True))
    for detail in ("questions_new_detail", "questions_seen_detail"):
        assert spoilt[detail]["yes"] == 100.0 and spoilt[detail]["no_more"] == 100.0
        assert spoilt[detail]["no_one"] == 0.0


@pytest.mark.parametrize("creatures,kinds", [(16, 4), (64, 4), (1024, 4)])
def test_list_questions(creatures, kinds):
    lang = L.Language(creatures, kinds)
    exam = lang.exam()
    for name, with_held in (("questions_new", True), ("questions_seen", False)):
        part = exam[name]
        n = len(part["answer"])
        assert n == 2 * int(part["answer"].sum())  # half "yes", half "no"
        keys = set()
        for i in range(n):
            listed = part["lists"][i][part["lists"][i] >= 0]
            assert 2 <= len(listed) <= 4
            assert lang.has_held_pair(listed[None])[0] == with_held
            lacking = sum(lang.facts[c, part["kinds"][i]] != part["values"][i] for c in listed)
            assert lacking == part["wrong"][i]
            assert (lacking == 0) == bool(part["answer"][i])
            sort, read_list, kind, value = read(lang, part["sentences"][i])
            assert sorted(read_list) == sorted(listed.tolist())
            assert true(lang, sort, read_list, kind, value)
            keys.add((kind, value, tuple(sorted(read_list))))
        assert len(keys) == n  # no question twice
        if creatures >= 64:
            assert n == 2000
        if creatures == 1024:  # with 64 creatures short "yes" lists are scarce
            lengths = (part["lists"] >= 0).sum(axis=1)
            for answer in (0, 1):
                shares = np.bincount(lengths[part["answer"] == answer], minlength=5)[2:]
                assert shares.max() - shares.min() <= 1
            # f of the "no" questions: equal chance 1 to k, before removing repeats.
            for k in (2, 3, 4):
                fs = part["wrong"][(part["answer"] == 0) & (lengths == k)]
                assert (np.abs(np.bincount(fs, minlength=k + 1)[1:] / len(fs) - 1 / k)
                        < 4 * np.sqrt(1 / k / len(fs))).all()
        if creatures == 16:
            # With 16 creatures: every possible "yes" question is used (they are the scarce
            # side), counted here by trying every list.
            possible = 0
            for k in (2, 3, 4):
                for listed in combinations(range(creatures), k):
                    if lang.has_held_pair(np.array([listed]))[0] != with_held:
                        continue
                    for kind in range(kinds):
                        possible += len(set(lang.facts[list(listed), kind])) == 1
            assert int(part["answer"].sum()) == min(1000, possible)


def test_exam_and_facts_do_not_depend_on_the_run():
    first = L.Language(64, 8, "equal")
    first_exam = first.exam()
    # Draw training text and set other seeds before building the second exam.
    np.random.seed(5)
    torch.manual_seed(5)
    second = L.Language(64, 8, "zipf")
    for step in range(5):
        second.sample_batch(9, step)
    second_exam = second.exam()
    assert np.array_equal(first.facts, second.facts)
    assert np.array_equal(first.held_pairs, second.held_pairs)
    for name in first_exam:
        for key in first_exam[name]:
            assert np.array_equal(first_exam[name][key], second_exam[name][key]), (name, key)


# ------------------------------------------------------------------------------------------------
# The model and the two forms of the word table
# ------------------------------------------------------------------------------------------------


def test_trained_and_frozen_differ_only_in_the_table():
    lang = L.Language(64, 4)
    torch.manual_seed(11)
    trained = net.make_net(lang, 16, table_form="trained")
    torch.manual_seed(11)
    frozen = net.make_net(lang, 16, table_form="frozen")
    a, b = trained.state_dict(), frozen.state_dict()
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key
    tokens = torch.from_numpy(lang.sample_batch(0, 0, 8))
    assert torch.equal(trained(tokens), frozen(tokens))
    assert trained.word_table.requires_grad and not frozen.word_table.requires_grad
    # The starting table is the fixed one, whatever the seed.
    torch.manual_seed(12)
    other = net.make_net(lang, 16)
    assert torch.equal(other.word_table, trained.word_table)
    assert torch.equal(trained.word_table, torch.from_numpy(net.word_table(lang)))


SMALL = {"creatures": 16, "kinds": 4, "usage": "zipf", "body": 8, "seed": 3, "steps": 150,
         "lr": 0.01, "threads": 1, "batch": 64}


def test_trained_table_moves_only_rows_in_use(tmp_path):
    summary = run.train(SMALL, tmp_path, quiet=True)
    lang = L.Language(16, 4, "zipf")
    start = torch.from_numpy(net.word_table(lang))
    saved = torch.load(tmp_path / summary["name"] / "model.pt", weights_only=False)["model"]
    table = saved["word_table"]
    # Creatures 16 to 1,023 are not in use: exactly unchanged in the saved (averaged) copy.
    unused = slice(lang.creature_base + 16, None)
    assert torch.equal(table[unused], start[unused])
    # Rows of the creatures in use and of the words did train.
    used = slice(lang.creature_base, lang.creature_base + 16)
    assert not torch.equal(table[used], start[used])
    assert not torch.equal(table[1:lang.creature_base], start[1:lang.creature_base])
    assert summary["parameters"]["trained"] == (summary["parameters"]["trained_without_table"]
                                                + 32 * (lang.creature_base + 16))


def test_frozen_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CHECKPOINT_EVERY", 50)
    config = dict(SMALL, table="frozen")
    first = run.train(config, tmp_path / "a", quiet=True)
    assert first["name"] == "c16_b8_zipf_k4_s3_frozen"
    assert first["parameters"]["trained"] == first["parameters"]["trained_without_table"]
    lang = L.Language(16, 4, "zipf")
    saved = torch.load(tmp_path / "a" / first["name"] / "model.pt", weights_only=False)
    assert torch.equal(saved["model"]["word_table"], torch.from_numpy(net.word_table(lang)))
    # Stopped and picked up again: exactly the same.
    assert run.train(config, tmp_path / "b", stop_after=100, quiet=True) is None
    resumed = run.run_one(config, str(tmp_path / "b"))
    again = torch.load(tmp_path / "b" / first["name"] / "model.pt", weights_only=False)
    for key in saved["model"]:
        assert torch.equal(saved["model"][key], again["model"][key]), key
    assert resumed["final"] == first["final"] and resumed["curve"] == first["curve"]


# ------------------------------------------------------------------------------------------------
# Training, grading and the summary
# ------------------------------------------------------------------------------------------------


def test_loss_is_only_on_graded_words():
    lang = L.Language(64, 4)
    tokens = lang.sample_batch(2, 0, 256)
    inputs, targets = run.inputs_and_targets(lang, tokens)
    torch.manual_seed(0)
    model = net.make_net(lang, 8)
    scores = model(inputs)
    # By hand: only places whose next token is a word (not <pad>, not a creature).
    nxt = tokens[:, 1:]
    keep = torch.from_numpy((nxt != 0) & (nxt < lang.creature_base))
    log_chances = torch.log_softmax(scores, dim=-1)
    picked = log_chances[keep].gather(1, torch.from_numpy(nxt - 1)[keep][:, None])
    by_hand = -picked.mean()
    loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]),
                                             targets.reshape(-1), ignore_index=run.IGNORE)
    assert torch.allclose(loss, by_hand)


def test_the_averaged_copy_is_graded_and_saved(tmp_path):
    summary = run.run_one(dict(SMALL, steps=100), str(tmp_path))
    lang = L.Language(16, 4, "zipf")
    saved = torch.load(tmp_path / summary["name"] / "model.pt", weights_only=False)
    model = net.make_net(lang, 8)
    model.load_state_dict(saved["model"])
    model.eval()
    again = L.grade(lang, run.predictor(model))
    final = {k: v for k, v in summary["final"].items() if k != "loss"}
    assert again == final


def test_summary_has_exactly_the_plan_format(tmp_path):
    summary = run.run_one(dict(SMALL, steps=150), str(tmp_path))
    text = json.loads((tmp_path / summary["name"] / "summary.json").read_text())
    assert text == summary
    assert list(text) == ["name", "config", "parameters", "steps", "sentences", "minutes",
                          "steps_per_second", "final", "curve"]
    assert list(text["parameters"]) == ["trained", "trained_without_table",
                                        "table_rows_in_use", "table_numbers", "body_middle"]
    assert all(isinstance(v, int) for v in text["parameters"].values())
    final = text["final"]
    assert list(final) == ["facts", "facts_by_kind", "questions_one", "questions_new",
                           "questions_new_detail", "questions_seen", "questions_seen_detail",
                           "not_known", "loss"]
    assert len(final["facts_by_kind"]) == 4
    for detail in ("questions_new_detail", "questions_seen_detail"):
        assert list(final[detail]) == ["yes", "no_one", "no_more", "count"]
        assert isinstance(final[detail]["count"], int)
    for key in ("facts", "questions_one", "questions_new", "questions_seen", "not_known",
                "loss"):
        assert isinstance(final[key], float)
    assert [point["step"] for point in text["curve"]] == run.grading_steps(150)
    assert list(text["curve"][0]) == ["step", "sentences", "facts", "questions_one",
                                      "questions_new", "questions_seen", "not_known", "loss"]
    assert text["steps"] == 150 and text["sentences"] == 150 * 64
    assert isinstance(text["minutes"], float) and isinstance(text["steps_per_second"], float)
    # A summary of another config under the same name is not returned as if it were this run.
    with pytest.raises(ValueError):
        run.run_one(dict(SMALL, steps=150, lr=0.02, name=summary["name"]), str(tmp_path))


def test_command_line_with_frozen_table(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "v2.run", "--creatures", "16", "--body", "8", "--usage", "equal",
         "--kinds", "32", "--seed", "1", "--steps", "100", "--lr", "0.01", "--threads", "1",
         "--batch", "32", "--table", "frozen", "--out-root", str(tmp_path)],
        cwd=PROJECT, capture_output=True, text=True, check=True)
    summary = json.loads((tmp_path / "c16_b8_equal_k32_s1_frozen" / "summary.json").read_text())
    assert summary["config"]["table"] == "frozen" and summary["config"]["kinds"] == 32
    assert "facts" in result.stdout

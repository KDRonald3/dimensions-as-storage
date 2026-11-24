"""Tests of v2/lang.py: words, facts, held-back pairs, training sentences, word use, exam."""

import hashlib
from collections import Counter

import numpy as np
import pytest

from v2 import lang as L

# sha256 of the 1,024 animal names of the first experiment (language.animal_names(1024)), one
# per line. Worked out once from language.py, so this test does not need to import it.
FIRST_EXPERIMENT_NAMES = "c9ae590f674ae8793c609313f71702647b5b8d1e90315a76c323a594776c50dc"

SENTENCES = 20_000


@pytest.fixture(scope="module")
def languages():
    """Languages are slow-ish to build (the exam), so each is made once."""
    made = {}

    def get(creatures, kinds=4, usage="equal"):
        key = (creatures, kinds, usage)
        if key not in made:
            made[key] = L.Language(creatures, kinds, usage)
        return made[key]

    return get


# ------------------------------------------------------------------------------------------------
# Words
# ------------------------------------------------------------------------------------------------


def test_creature_names_are_those_of_the_first_experiment():
    names = L.creature_names()
    assert len(names) == 1024 and len(set(names)) == 1024
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == FIRST_EXPERIMENT_NAMES


@pytest.mark.parametrize("kinds", [4, 8, 16, 32])
def test_token_order(languages, kinds):
    lang = languages(16, kinds)
    assert lang.words[0] == "<pad>"
    assert lang.words[1:15] == L.FIXED_WORDS
    assert lang.words[lang.creature_base:] == L.creature_names()
    assert lang.n_tokens == lang.creature_base + 1024
    # Words of unused kinds are not in the vocabulary.
    if kinds < 16:
        assert "guard" not in lang.words and "pools" not in lang.words
    if kinds < 32:
        assert "trail" not in lang.words and "packs" not in lang.words
    # Every token is a different word (B.1), so no value is spelled like a creature.
    assert len(set(lang.words)) == len(lang.words)
    # Every non-creature word keeps its token id when more kinds are used.
    big = languages(16, 32)
    assert big.words[:lang.creature_base] == lang.words[:lang.creature_base]


# ------------------------------------------------------------------------------------------------
# Facts (test 5) and held-back pairs
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("creatures", [16, 64, 256, 1024])
def test_facts_are_balanced_and_do_not_change_with_kinds(creatures):
    facts = {kinds: L.make_facts(creatures, kinds) for kinds in (4, 8, 16, 32)}
    for kind in range(32):
        counts = np.bincount(facts[32][:, kind], minlength=4)
        assert counts.tolist() == [creatures // 4] * 4
    assert np.array_equal(facts[4], facts[8][:, :4])
    assert np.array_equal(facts[4], facts[16][:, :4])
    assert np.array_equal(facts[8], facts[16][:, :8])
    assert np.array_equal(facts[16], facts[32][:, :16])


@pytest.mark.parametrize("creatures", [16, 64, 256, 1024])
def test_held_back_pairs(creatures):
    pairs = L.make_held_pairs(creatures)
    all_pairs = creatures * (creatures - 1) // 2
    assert len(pairs) == max(1, all_pairs // 10)
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert len({tuple(p) for p in pairs.tolist()}) == len(pairs)
    assert np.array_equal(pairs, L.make_held_pairs(creatures))  # fixed seed


# ------------------------------------------------------------------------------------------------
# Training sentences (test 3)
# ------------------------------------------------------------------------------------------------


def wrong_count(lang, read):
    """How many creatures of a question's list do NOT have the value."""
    return sum(lang.facts[c, read["kind"]] != read["value"] for c in read["creatures"])


@pytest.mark.parametrize("creatures", [16, 256])
@pytest.mark.parametrize("usage", ["equal", "zipf"])
@pytest.mark.parametrize("kinds", [4, 32])
def test_training_sentences(languages, creatures, usage, kinds):
    lang = languages(creatures, kinds, usage)
    rng = np.random.default_rng([99, creatures, kinds])
    tokens = lang.draw_sentences(rng, SENTENCES)["tokens"]
    assert tokens.shape == (SENTENCES, 13)

    reads = []
    for row in tokens:
        read = L.read_sentence(lang, row)
        assert read is not None, lang.decode(row)
        assert L.is_true(lang, row), lang.decode(row)
        found = read["creatures"]
        for a in range(len(found)):
            for b in range(a + 1, len(found)):
                assert not lang.held[found[a], found[b]], lang.decode(row)
        reads.append(read)

    def shares(values, size):
        counts = np.bincount(values, minlength=size)
        return counts / counts.sum()

    types = np.array([r["type"] for r in reads])
    assert np.allclose(shares(types, 4), L.TYPE_CHANCES, atol=0.015)
    assert np.allclose(shares(np.array([r["kind"] for r in reads]), kinds), 1 / kinds,
                       atol=0.015)
    # Values have equal chance in statements and questions (drawn first there).
    drawn = np.array([r["value"] for r in reads if r["type"] != L.NOT_STATEMENT])
    assert np.allclose(shares(drawn, 4), 0.25, atol=0.02)
    # In "not" statements the value is drawn after the creatures, among the values none of them
    # has. With equal use that makes every value equally likely. (With zipf use it does not: a
    # common creature's own value is then left out more often, which is right.)
    if usage == "equal":
        negated = np.array([r["value"] for r in reads if r["type"] == L.NOT_STATEMENT])
        assert np.allclose(shares(negated, 4), 0.25, atol=0.03)

    lengths = np.array([len(r["creatures"]) for r in reads])
    length_shares = shares(lengths - 1, 4)
    if creatures >= 256:
        assert np.allclose(length_shares, L.LENGTH_CHANCES, atol=0.015)
        # In "no" questions, f (the creatures without the value) has equal chance 1 to k.
        for k in (2, 3, 4):
            fs = [wrong_count(lang, r) for r in reads
                  if r["type"] == L.NO_QUESTION and len(r["creatures"]) == k]
            # About 800 questions per k: allow 4 times the usual random wobble.
            wobble = 4 * np.sqrt((1 / k) * (1 - 1 / k) / len(fs))
            assert np.allclose(shares(np.array(fs) - 1, k), 1 / k, atol=wobble)
    else:
        # With 16 creatures some long lists cannot avoid a held-back pair and are made shorter.
        assert abs(length_shares[0] - 0.4) < 0.015
        assert length_shares[3] < 0.2 + 0.015
    # Every "no" question really has at least one creature without the value.
    assert all(wrong_count(lang, r) >= 1 for r in reads if r["type"] == L.NO_QUESTION)


def test_batches_depend_only_on_seed_and_step(languages):
    lang = languages(64, 4, "zipf")
    assert np.array_equal(lang.sample_batch(3, 17), lang.sample_batch(3, 17))
    assert not np.array_equal(lang.sample_batch(3, 17), lang.sample_batch(3, 18))
    assert not np.array_equal(lang.sample_batch(3, 17), lang.sample_batch(4, 17))
    assert lang.sample_batch(0, 0).shape == (512, 13)


# ------------------------------------------------------------------------------------------------
# Word use (test 9)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("usage", ["equal", "zipf"])
def test_word_use_follows_the_weights(usage):
    lang = L.Language(64, 4, usage)
    if usage == "equal":
        assert np.allclose(lang.weights, 1 / 64)
    else:
        assert np.allclose(lang.weights, (1 / np.arange(1, 65)) / (1 / np.arange(1, 65)).sum())
    drawn = lang.draw_sentences(np.random.default_rng(5), 200_000)
    one = drawn["lengths"] == 1
    # "not" statements of one creature: the creature is drawn by use from all creatures.
    picked = drawn["lists"][one & (drawn["types"] == L.NOT_STATEMENT), 0]
    observed = np.bincount(picked, minlength=64) / len(picked)
    assert np.all(np.abs(observed - lang.weights) < 4 * np.sqrt(lang.weights / len(picked)))
    # Statements of one creature: drawn by use among the creatures that have the value.
    for kind in range(4):
        for value in range(4):
            chosen = one & (drawn["types"] == L.STATEMENT) & (drawn["kinds"] == kind) \
                & (drawn["values"] == value)
            picked = drawn["lists"][chosen, 0]
            members = np.nonzero(lang.facts[:, kind] == value)[0]
            expected = lang.weights[members] / lang.weights[members].sum()
            observed = np.array([(picked == m).mean() for m in members])
            assert np.all(np.abs(observed - expected) < 4 * np.sqrt(expected / len(picked)) + 1e-3)
    if usage == "zipf":
        counts = Counter(drawn["lists"][drawn["lists"] >= 0].tolist())
        assert counts.most_common(1)[0][0] == 0  # creature of rank 1 is the most used


# ------------------------------------------------------------------------------------------------
# The exam (tests 4 and 6)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("creatures,kinds", [(16, 4), (16, 16), (256, 4)])
def test_exam_sets(languages, creatures, kinds):
    lang = languages(creatures, kinds)
    exam = lang.exam()
    assert len(exam["facts"]["prompts"]) == creatures * kinds
    assert len(exam["not_known"]["prompts"]) == creatures * kinds
    assert len(exam["questions_one"]["prompts"]) == creatures * kinds * 4
    for name, should_have_held in (("questions_new", True), ("questions_seen", False)):
        part = exam[name]
        assert 0 < len(part["sentences"]) <= 2000
        lists = part["lists"]
        for row, sentence in zip(lists, part["sentences"]):
            found = row[row >= 0]
            assert 2 <= len(found) <= 4
            held = any(lang.held[a, b] for i, a in enumerate(found) for b in found[i + 1:])
            assert held == should_have_held
            assert L.is_true(lang, sentence), lang.decode(sentence)
        yes = int(part["answer"].sum())
        assert yes == len(part["answer"]) - yes  # half "yes", half "no"
        # The prompt is the question without its answer, read at the "?".
        last_words = part["prompts"][np.arange(len(lists)), part["last"]]
        assert (last_words == lang.word_id["?"]).all()
        if creatures >= 256:
            assert len(part["sentences"]) == 2000
            sizes = np.bincount((lists >= 0).sum(axis=1), minlength=5)[2:]
            assert sizes.max() - sizes.min() <= 2
    for sentence in exam["questions_one"]["sentences"]:
        assert L.is_true(lang, sentence)


def oracle(lang):
    """A predictor that knows every fact: the best possible exam taker."""
    yes, no = lang.yes_id, lang.no_id

    def predict(prompts, last):
        scores = np.zeros((len(prompts), lang.n_out))
        for i, (row, end) in enumerate(zip(prompts, last)):
            row = [int(t) for t in row[:end + 1]]
            if lang.words[row[-1]] == "?":
                read = L.read_sentence(lang, row + [yes])
                every = all(lang.facts[c, read["kind"]] == read["value"]
                            for c in read["creatures"])
                scores[i, (yes if every else no) - 1] = 1.0
                continue
            creature = row[0] - lang.creature_base
            words = [lang.words[t] for t in row[1:]]
            negated = "not" in words
            link = [w for w in words if w not in ("do", "not")]
            kind = [k for k in range(lang.kinds) if L.KINDS[k][0] == link][0]
            true_value = lang.value_ids[kind, lang.facts[creature, kind]]
            if negated:  # every value high except the true one
                scores[i, lang.value_ids[kind] - 1] = 1.0
                scores[i, true_value - 1] = 0.0
            else:
                scores[i, true_value - 1] = 1.0
        return scores

    return predict


@pytest.mark.parametrize("creatures,kinds", [(16, 4), (16, 16), (64, 8), (16, 32)])
def test_oracle_scores_100_and_a_flat_predictor_is_at_chance(languages, creatures, kinds):
    lang = languages(creatures, kinds)
    best = L.grade(lang, oracle(lang))
    for key in ("facts", "questions_one", "questions_new", "questions_seen", "not_known"):
        assert best[key] == 100.0, key
    assert best["facts_by_kind"] == [100.0] * kinds

    def flat(prompts, last):
        return np.zeros((len(prompts), lang.n_out))

    blind = L.grade(lang, flat)
    assert blind["facts"] == 25.0 and blind["not_known"] == 25.0
    assert blind["questions_one"] == 0.0
    assert blind["questions_new"] == 0.0 and blind["questions_seen"] == 0.0

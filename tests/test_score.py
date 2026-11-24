"""
Tests of score.py: the measuring instrument, checked before anything is trained.

Run them with:    pytest tests/test_score.py

A score is only worth something if we know what it says about predictors we understand. So the
instrument is pointed at fake models whose right scores are known in advance:

  the oracle           knows every fact                  -> 100 everywhere, gap 0, all bits
  the blind model      knows no single animal            -> 0 on every score that is corrected
                                                            for chance, 0 bits
  a half-knowing one   knows every other animal          -> numbers worked out by hand
  a color-knowing one  knows one fact of every animal    -> numbers worked out by hand
  a leaky one          knows animals it never read about -> the leak check must say so
  an untrained model   has read nothing                  -> 0, and nothing about unseen animals

All of it is repeated for equal and zipf usage, for one and four seasons, and for a base table
that was edited to be unbalanced. Nothing here trains a model and nothing is written to disk.
"""

import dataclasses
import json
import math

import numpy as np
import pytest
import torch

import language as L
import score as S
from language import Language, LanguageConfig
from model import ModelConfig, TinyGPT
from score import blind_predictor, model_predictor, oracle_predictor, score

ROWS = 12  # evaluation rows in these tests (the fake models are slow)
CONTEXT = 40  # words per row


def make(animals=16, table_animals=32, seasons=1, usage="equal", lexicon_seed=0):
    return Language(LanguageConfig(animals, table_animals, seasons, usage, lexicon_seed))


def rows_for(language):
    return language.sample_rows(np.random.default_rng(5), ROWS, CONTEXT)


def untrained_model(language, width=16, seed=0):
    torch.manual_seed(seed)
    return TinyGPT(ModelConfig(language.vocab_size, width=width, context=CONTEXT))


# The four languages every check is repeated on. 32 animals are two blocks of 16; the table holds
# 16 more that are never in use, for the leak check.
CASES = [("equal", 1), ("zipf", 1), ("equal", 4), ("zipf", 4)]


CASE_NAMES = [f"{usage}-{seasons}" for usage, seasons in CASES]


@pytest.fixture(scope="module", params=CASES, ids=CASE_NAMES)
def graded(request):
    """One language with the scores of the oracle and of the blind model (computed once)."""
    usage, seasons = request.param
    language = make(animals=32, table_animals=48, seasons=seasons, usage=usage)
    eval_rows = rows_for(language)
    return {
        "language": language,
        "eval_rows": eval_rows,
        "oracle": score(oracle_predictor(language), language, eval_rows),
        "blind": score(blind_predictor(language), language, eval_rows),
    }


def unbalanced_base_table():
    """The base table with some facts changed, so that values are no longer equally common."""
    edited = []
    for number, (name, color, food, home, wake) in enumerate(L.BASE_ANIMALS):
        if color == "black":
            color = "red"  # now 8 red, 4 brown, 4 green, no black
        if number < 6:
            food = "meat"  # now 7 animals eat meat (4 did before; 3 of the first 6 are new)
        if number == 0:
            home = "caves"  # 5 caves, 3 woods
        edited.append((name, color, food, home, wake))
    return edited


# ------------------------------------------------------------------------------------------------
# The oracle gets full marks
# ------------------------------------------------------------------------------------------------


def test_the_oracle_gets_100_and_a_gap_of_zero(graded):
    language, result = graded["language"], graded["oracle"]
    assert result["facts_known"] == pytest.approx(100)
    assert result["facts_known_by_use"] == pytest.approx(100)
    assert result["questions_right"] == pytest.approx(100)
    assert result["gap"] == pytest.approx(0, abs=1e-5)
    assert result["loss"] == pytest.approx(result["floor"], abs=1e-5)
    assert result["bits_known"] == pytest.approx(result["bits_total"])
    assert result["bits_total"] == pytest.approx(language.knowledge_bits())
    assert result["words_known"] == result["n_words"] == 32
    assert result["facts_right_by_animal"] == [language.facts_per_animal] * 32
    assert result["headline"] == result["facts_known"]
    assert result["headline_name"] == "facts_known"


def test_the_oracle_knows_nothing_about_animals_that_are_not_in_use(graded):
    assert graded["oracle"]["unseen_known"] == 0


# ------------------------------------------------------------------------------------------------
# The blind model gets zero
# ------------------------------------------------------------------------------------------------


def test_the_blind_model_gets_0(graded):
    result = graded["blind"]
    assert result["facts_known"] == pytest.approx(0, abs=1e-9)
    assert result["facts_known_by_use"] == pytest.approx(0, abs=1e-9)
    assert result["questions_right"] == pytest.approx(0, abs=1e-9)
    assert result["bits_known"] == pytest.approx(0, abs=1e-9)
    assert result["unseen_known"] == pytest.approx(0, abs=1e-9)
    # It still predicts the sentence forms, so its loss is far below an even guess over all words,
    # but it pays for every fact it does not know.
    assert result["floor"] == pytest.approx(graded["oracle"]["floor"])
    assert 0.05 < result["gap"] < 2
    assert result["loss"] < math.log2(graded["language"].vocab_size)


def test_the_blind_model_gives_every_animal_the_same_answer(graded):
    language = graded["language"]
    predict = blind_predictor(language)
    probes = language.fact_probes()
    said = S.most_likely_next(predict, probes)
    answers = {}  # (season, attribute) -> the words said for it
    for probe, word in zip(probes, said):
        answers.setdefault((probe.season, probe.attribute), set()).add(int(word))
    assert all(len(words) == 1 for words in answers.values())
    # ... and each answer is a value of the right attribute, not some other word.
    for (season, attribute), words in answers.items():
        assert words.pop() in language.value_ids[attribute]


def test_the_blind_model_answers_by_how_common_each_value_is():
    language = make(animals=16, usage="zipf")
    predict = blind_predictor(language)
    prefix = language.encode("foxes eat")
    probabilities = np.exp(predict(np.array([prefix]))[0, -1])
    for value, word in enumerate(L.VALUES[1]):
        eaters = language.lexicon[:16, 0, 1] == value
        assert probabilities[language.word_id[word]] == pytest.approx(language.usage[eaters].sum())
    # After a question it gives the odds that follow from the design alone: the true value is
    # shown half the time, so with a balanced table and equal usage "yes" and "no" are even.
    balanced = make(animals=16)
    question = np.array([balanced.encode("do foxes eat meat ?")])
    odds = np.exp(blind_predictor(balanced)(question)[0, -1])
    assert odds[balanced.yes_id] == pytest.approx(0.5)
    assert odds[balanced.no_id] == pytest.approx(0.5)


# ------------------------------------------------------------------------------------------------
# An edited, unbalanced base table
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("usage", ["equal", "zipf"])
@pytest.mark.parametrize("seasons", [1, 4])
def test_an_unbalanced_table_still_gives_100_and_0(monkeypatch, usage, seasons):
    monkeypatch.setattr(L, "BASE_ANIMALS", unbalanced_base_table())
    language = make(animals=16, table_animals=32, seasons=seasons, usage=usage)
    assert np.bincount(language.lexicon[:16, 0, 0], minlength=4).tolist() == [8, 4, 4, 0]
    assert language.knowledge_bits() < 16 * 8 * seasons  # the edited facts are worth less
    eval_rows = rows_for(language)

    oracle = score(oracle_predictor(language), language, eval_rows)
    assert oracle["facts_known"] == pytest.approx(100)
    assert oracle["facts_known_by_use"] == pytest.approx(100)
    assert oracle["questions_right"] == pytest.approx(100)
    assert oracle["gap"] == pytest.approx(0, abs=1e-5)
    assert oracle["bits_known"] == pytest.approx(language.knowledge_bits())
    assert oracle["bits_total"] == pytest.approx(language.knowledge_bits())

    blind = score(blind_predictor(language), language, eval_rows)
    assert blind["facts_known"] == pytest.approx(0, abs=1e-9)
    assert blind["facts_known_by_use"] == pytest.approx(0, abs=1e-9)
    assert blind["questions_right"] == pytest.approx(0, abs=1e-9)
    assert blind["bits_known"] == pytest.approx(0, abs=1e-9)


def test_chance_is_computed_from_the_table_as_it_is(monkeypatch):
    balanced = make(animals=16)
    assert S.chance_accuracy(balanced.lexicon[:16], np.full(16, 1 / 16)) == pytest.approx(0.25)
    monkeypatch.setattr(L, "BASE_ANIMALS", unbalanced_base_table())
    edited = make(animals=16)
    # Most common value: 8 red, 7 meat, 5 caves, 4 of each waking time, out of 16 animals.
    expected = (8 + 7 + 5 + 4) / (4 * 16)
    assert S.chance_accuracy(edited.lexicon[:16], np.full(16, 1 / 16)) == pytest.approx(expected)
    # Weighted by usage, an animal that is used more counts for more.
    zipf = make(animals=16, usage="zipf")
    by_hand = 0.0
    for attribute in range(4):
        shares = [zipf.usage[zipf.lexicon[:16, 0, attribute] == value].sum() for value in range(4)]
        by_hand += max(shares) / 4
    assert S.chance_accuracy(zipf.lexicon[:16], zipf.usage) == pytest.approx(by_hand)
    assert by_hand > 0.25


# ------------------------------------------------------------------------------------------------
# A model that knows half of the animals: numbers worked out by hand
# ------------------------------------------------------------------------------------------------


def half_knowing_predictor(language, known):
    """An oracle that has forgotten some animals: whatever comes after one of them, it says "no"."""
    forgotten = {int(language.table_animal_ids[a]) for a in range(language.config.animals)
                 if a not in known}
    says_no = np.full(language.vocab_size, 1e-6)  # nearly all of the probability on "no"
    says_no[language.no_id] = 1 - 1e-6 * (language.vocab_size - 1)

    def next_distribution(prefix):
        if forgotten & set(prefix):
            return says_no
        return language.next_distribution(prefix)

    return S.fake_model(next_distribution, language.vocab_size)


@pytest.mark.parametrize("usage", ["equal", "zipf"])
def test_knowing_half_of_the_animals(usage):
    language = make(animals=16, usage=usage)
    known = set(range(0, 16, 2))  # every other animal: 8 of 16
    result = score(half_knowing_predictor(language, known), language, rows_for(language))

    # Half of the facts are right. Chance is 1/4, so the corrected score is (0.5 - 0.25) / 0.75.
    assert result["facts_right_by_animal"] == [4, 0] * 8
    assert result["words_known"] == 8 and result["n_words"] == 16
    assert result["facts_known"] == pytest.approx(100 * (0.5 - 0.25) / 0.75)

    # By use: the known animals' share of all usage, against the chance level by usage.
    accuracy = language.usage[sorted(known)].sum()
    chance = S.chance_accuracy(language.lexicon[:16], language.usage)
    assert result["facts_known_by_use"] == pytest.approx(100 * (accuracy - chance) / (1 - chance))
    if usage == "equal":
        assert result["facts_known_by_use"] == pytest.approx(result["facts_known"])

    # Bits: for every attribute the accuracy is a = 8/16. With four values that needs at least
    # 2 - h(0.5) - 0.5 x log2(3) = 0.2075 bits per fact. Far less than half of the 2 bits.
    per_fact = 2 - 1 - 0.5 * math.log2(3)
    assert result["bits_known"] == pytest.approx(64 * per_fact)
    assert result["bits_known"] < 0.5 * result["bits_total"]

    # Questions: it answers right for the animals it knows and "no" for the others. So every no
    # item is right and half of the yes items are: balanced accuracy (1 + 0.5) / 2 = 0.75, and
    # the score is 200 x (0.75 - 0.5) = 50. Plain accuracy would have said 87.5%.
    assert result["questions_right"] == pytest.approx(50)
    # The same number counted the long way round, kind of question by kind of question.
    balanced = []
    for attribute in range(4):
        for value in range(4):
            has_it = language.lexicon[:16, 0, attribute] == value
            yes_right = np.mean([animal in known for animal in np.flatnonzero(has_it)])
            balanced.append((yes_right + 1.0) / 2)
    assert result["questions_right"] == pytest.approx(200 * (np.mean(balanced) - 0.5))
    assert 0 < result["gap"] < 20


def test_bits_known_rises_with_accuracy_and_is_zero_at_chance():
    language = make(animals=16)
    lexicon = language.lexicon[:16]
    previous = 0.0
    for known in [4, 8, 12, 16]:  # with 4 of 16 right the accuracy is exactly 1/4: chance
        right = np.zeros((16, 1, 4), dtype=bool)
        right[:known] = True
        bits = S.bits_known(right, lexicon)
        assert (bits == 0) if known == 4 else (bits > previous)
        previous = bits
    assert previous == pytest.approx(128)
    assert S.bits_known(np.zeros((16, 1, 4), dtype=bool), lexicon) == 0  # always wrong: no credit


# ------------------------------------------------------------------------------------------------
# A model that knows one fact of every animal: facts are counted, not whole animals
# ------------------------------------------------------------------------------------------------


def color_knowing_predictor(language):
    """A model that knows the color of every animal and nothing else.

    While the sentence under way is about color (it has the word "are" in it) it answers like the
    oracle. Everywhere else it answers like a blind model that takes every animal to be used
    equally often: it names the value that most animals have (the first one if several values
    are equally common), however often each animal is really used.
    """
    oracle = oracle_predictor(language)
    blind = blind_predictor(Language(dataclasses.replace(language.config, usage="equal")))
    are = language.word_id["are"]
    ends_a_sentence = [language.word_id["."], language.yes_id, language.no_id]

    def predict(tokens):
        knows, guesses = oracle(tokens), blind(tokens)
        answers = guesses.copy()
        for row in range(tokens.shape[0]):
            about_color = False
            for place in range(tokens.shape[1]):
                about_color = about_color or tokens[row, place] == are
                if about_color:
                    answers[row, place] = knows[row, place]
                if tokens[row, place] in ends_a_sentence:
                    about_color = False  # the next sentence starts afresh
        return answers

    return predict


@pytest.mark.parametrize("usage, seasons", CASES)
def test_knowing_one_fact_of_every_animal(usage, seasons):
    language = make(animals=32, table_animals=32, seasons=seasons, usage=usage)
    lexicon = language.lexicon[:32]
    result = score(color_knowing_predictor(language), language, rows_for(language))

    # By hand: every color is right. For food, home and waking time the model names the value
    # that most animals have. In a balanced table all four are equally common, so that is the
    # first value: meat, woods, dawn.
    right = np.zeros((32, seasons, 4), dtype=bool)
    right[:, :, 0] = True
    for season in range(seasons):
        for attribute in [1, 2, 3]:
            true_values = lexicon[:, season, attribute]
            assert np.bincount(true_values, minlength=4).tolist() == [8, 8, 8, 8]
            right[:, season, attribute] = true_values == 0
    assert result["facts_right_by_animal"] == right.sum(axis=(1, 2)).tolist()

    # The headline counts FACTS. The value the model names is right for 8 of the 32 animals, so
    # the accuracy is (1 + 3 x 1/4) / 4 = 7/16. Chance is 1/4, so the score is
    # (7/16 - 1/4) / (3/4) = 25. Had it counted whole animals, it would read about 0 here.
    assert result["facts_known"] == pytest.approx(25.0)
    assert result["headline"] == result["facts_known"]
    # A WORD is known only if every fact of the animal is right.
    assert result["words_known"] == int(right.all(axis=(1, 2)).sum())
    assert result["words_known"] == 0  # no animal of this table has meat, woods and dawn together
    # Bits: all 2 bits of every color (32 animals x 2 bits in each season), nothing else. The
    # other attributes are right a quarter of the time, which shows no knowledge at all.
    assert result["bits_known"] == pytest.approx(64.0 * seasons)
    # Questions: of the 16 kinds in a season, the 4 about color are answered perfectly and the
    # other 12 always the same way (a balanced accuracy of 1/2): 200 x (10/16 - 1/2) = 25.
    assert result["questions_right"] == pytest.approx(25.0)

    if usage == "equal":
        assert result["facts_known_by_use"] == pytest.approx(25.0)
    else:
        # By use, an animal counts by how often it is used, and so does the chance level: a
        # know-nothing would name the values of the much-used animals, which this model does
        # not. So the score by use is another number, and here a clearly lower one. The
        # headline is facts_known, not this.
        accuracy = float((language.usage[:, None, None] * right).sum() / (seasons * 4))
        chance = S.chance_accuracy(lexicon, language.usage)
        by_use = 100 * (accuracy - chance) / (1 - chance)
        assert result["facts_known_by_use"] == pytest.approx(by_use)
        assert result["facts_known_by_use"] < result["facts_known"] - 5
        assert result["headline"] != pytest.approx(result["facts_known_by_use"])


def test_knowing_one_fact_of_every_animal_knows_no_word():
    # The plain case once more with its three numbers written out: 32 animals, equal usage.
    language = make(animals=32, table_animals=32)
    result = score(color_knowing_predictor(language), language, rows_for(language))
    assert result["facts_known"] == pytest.approx(25.0)
    assert result["words_known"] == 0
    assert result["bits_known"] == pytest.approx(64.0)


# ------------------------------------------------------------------------------------------------
# The leak check must be able to fire
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("usage, seasons", CASES)
def test_the_leak_check_fires_for_a_model_that_knows_unseen_animals(usage, seasons):
    # 16 animals in use and 48 in the table: the leak check asks about the 32 unseen ones.
    language = make(animals=16, table_animals=48, seasons=seasons, usage=usage)
    eval_rows = rows_for(language)
    # A leaky model: the oracle of the SAME language with all 48 animals in use. The two
    # languages share their words, their token ids and their facts, so this model knows every
    # fact of the 32 animals that the first language never writes about.
    everything = make(animals=48, table_animals=48, seasons=seasons, usage=usage)
    assert everything.words == language.words
    assert np.array_equal(everything.lexicon, language.lexicon)
    leaky = score(oracle_predictor(everything), language, eval_rows)
    assert leaky["unseen_known"] == pytest.approx(100)
    assert leaky["facts_known"] == pytest.approx(100)  # it knows the animals in use as well

    # Half-leaky: it has forgotten every other unseen animal, and says "no" after one of those,
    # which is never a value. So 16 of the 32 unseen animals are right: an accuracy of 1/2. The
    # unseen animals are two balanced blocks, so chance is 1/4, and the leak check must read
    # (1/2 - 1/4) / (1 - 1/4) = one third of 100.
    known = set(range(16)) | set(range(16, 48, 2))
    half_leaky = score(half_knowing_predictor(everything, known), language, eval_rows)
    assert half_leaky["unseen_known"] == pytest.approx(100 * (0.5 - 0.25) / 0.75)
    assert half_leaky["facts_known"] == pytest.approx(100)


# ------------------------------------------------------------------------------------------------
# What exactly counts as right
# ------------------------------------------------------------------------------------------------


def test_the_true_value_must_beat_every_word_of_the_vocabulary():
    language = make(animals=16)
    oracle = oracle_predictor(language)

    def distracted(tokens):
        """The oracle, except that it always finds "." a little more likely than anything else."""
        log_probabilities = oracle(tokens).copy()
        log_probabilities[:, :, language.word_id["."]] = 1.0
        return log_probabilities

    result = score(distracted, language, rows_for(language))
    assert result["facts_known"] == 0
    assert result["questions_right"] == 0
    assert result["words_known"] == 0


def test_an_animal_word_that_is_not_in_use_is_a_wrong_answer_too():
    # The winner is taken over the WHOLE vocabulary, and that includes the animal words that are
    # in the word table but never in the text.
    language = make(animals=16)  # the table holds 32 animals
    oracle = oracle_predictor(language)
    unused = int(language.table_animal_ids[20])
    assert unused not in language.animal_ids

    def distracted_by(log_probability):
        """The oracle, except that it gives one unused animal word this log-probability
        everywhere. The oracle gives a true value log 1 = 0, so anything above 0 beats it."""
        def predict(tokens):
            log_probabilities = oracle(tokens).copy()
            log_probabilities[:, :, unused] = log_probability
            return log_probabilities
        return predict

    result = score(distracted_by(1.0), language, rows_for(language))
    assert result["facts_known"] == 0 and result["facts_known_by_use"] == 0
    assert result["facts_right_by_animal"] == [0] * 16
    assert result["questions_right"] == 0
    assert result["words_known"] == 0 and result["bits_known"] == 0
    # Just below the true value the same word changes nothing: every fact is still right.
    result = score(distracted_by(-0.5), language, rows_for(language))
    assert result["facts_known"] == pytest.approx(100) and result["words_known"] == 16


def test_always_saying_no_scores_zero_on_questions():
    language = make(animals=16)

    def always(word):
        def predict(tokens):
            log_probabilities = np.full(tokens.shape + (language.vocab_size,), -20.0)
            log_probabilities[:, :, language.word_id[word]] = 0.0
            return log_probabilities
        return predict

    probes = language.question_probes()
    plain_accuracy = np.mean([probe.answer == language.no_id for probe in probes])
    assert plain_accuracy == pytest.approx(0.75)  # three of four questions have the answer no ...
    assert S.questions_right(always("no"), language) == 0  # ... and yet always-no knows nothing
    assert S.questions_right(always("yes"), language) == 0


def test_padding_a_batch_of_probes_does_not_change_the_answers(monkeypatch):
    language = make(animals=16, seasons=4)
    predict = model_predictor(untrained_model(language))
    probes = language.fact_probes() + language.question_probes()
    assert len(probes) > S.PROBE_BATCH  # more than one batch
    together = S.most_likely_next(predict, probes)
    one_by_one = np.concatenate([S.most_likely_next(predict, [probe]) for probe in probes[::7]])
    assert np.array_equal(together[::7], one_by_one)
    assert len({len(probe.prefix) for probe in probes}) > 2  # the prompts really differ in length
    # The answers are the model's own: computed here directly, one prompt at a time.
    for probe, word in list(zip(probes, together))[::97]:
        assert word == predict(np.array([probe.prefix]))[0, -1].argmax()
    # Cutting the probes into batches differently changes nothing either.
    monkeypatch.setattr(S, "PROBE_BATCH", 37)
    assert np.array_equal(S.most_likely_next(predict, probes), together)


def test_loss_floor_and_gap_use_only_real_target_words(monkeypatch):
    language = make(animals=16)
    inputs, targets, bits, kind = eval_rows = rows_for(language)
    predict = model_predictor(untrained_model(language))
    loss, floor = S.loss_and_floor(predict, eval_rows)
    real = targets != L.IGNORE
    assert (~real).any()  # there is padding in these rows
    assert floor == pytest.approx(bits[real].mean())
    # The same loss, computed one position at a time.
    log_probabilities = predict(inputs)
    by_hand = [-log_probabilities[row, place, targets[row, place]] / math.log(2)
               for row, place in zip(*np.nonzero(real))]
    assert loss == pytest.approx(np.mean(by_hand), rel=1e-5)
    # An untrained model spreads its guess almost evenly over all words.
    assert loss == pytest.approx(math.log2(language.vocab_size), abs=0.2)
    # Splitting the rows into batches differently must not change anything.
    assert S.ROW_BATCH > 5
    monkeypatch.setattr(S, "ROW_BATCH", 5)
    assert S.loss_and_floor(predict, eval_rows)[0] == pytest.approx(loss)


# ------------------------------------------------------------------------------------------------
# A real model, before any training
# ------------------------------------------------------------------------------------------------


def test_model_predictor_returns_log_probabilities():
    language = make(animals=16)
    predict = model_predictor(untrained_model(language))
    tokens = np.array([language.encode("foxes eat meat ."), language.encode("do bears eat plants")])
    log_probabilities = predict(tokens)
    assert isinstance(log_probabilities, np.ndarray)
    assert log_probabilities.shape == (2, 4, language.vocab_size)
    assert np.exp(log_probabilities).sum(axis=2) == pytest.approx(np.ones((2, 4)), abs=1e-5)


@pytest.mark.parametrize("usage, seasons", CASES)
def test_an_untrained_model_knows_nothing(usage, seasons):
    language = make(animals=16, table_animals=64, seasons=seasons, usage=usage)
    eval_rows = rows_for(language)
    for seed in range(3):
        result = score(model_predictor(untrained_model(language, seed=seed)), language, eval_rows)
        assert result["unseen_known"] is not None and result["unseen_known"] < 5
        assert result["facts_known"] < 5
        assert result["questions_right"] < 5
        assert result["gap"] > 1
        assert result["bits_total"] == pytest.approx(16 * 8 * seasons)


def test_unseen_known_is_none_when_every_table_animal_is_in_use():
    language = make(animals=32, table_animals=32)
    result = score(model_predictor(untrained_model(language)), language, rows_for(language))
    assert result["unseen_known"] is None


def test_the_leak_check_probes_at_most_256_unseen_animals():
    assert S.UNSEEN_ANIMALS == 256
    language = make(animals=16, table_animals=320)
    right = S.grade_facts(model_predictor(untrained_model(language)), language, "unseen",
                          S.UNSEEN_ANIMALS)
    assert right.shape == (256, 1, 4)


def test_score_returns_plain_numbers_that_can_be_saved():
    language = make(animals=16, usage="zipf")
    result = score(model_predictor(untrained_model(language)), language, rows_for(language))
    expected = {"facts_known", "facts_known_by_use", "questions_right", "unseen_known",
                "words_known", "n_words", "facts_right_by_animal", "loss", "floor", "gap",
                "bits_total", "bits_known", "headline", "headline_name"}
    assert set(result) == expected
    assert json.loads(json.dumps(result)) == result
    assert len(result["facts_right_by_animal"]) == 16
    assert result["gap"] == pytest.approx(result["loss"] - result["floor"])

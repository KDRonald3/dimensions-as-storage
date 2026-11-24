"""
Tests of language.py: the toy language, the exact training sets and the oracle.

Run them with:    pytest tests/test_language.py

language.py is the measuring instrument of the whole experiment, so these tests check it from the
outside. Wherever possible a test reads the generated TEXT with its own small parser
(read_sentence below) instead of trusting the code that wrote the text. Nothing here trains a
model, and nothing is written to disk.
"""

from collections import Counter

import numpy as np
import pytest

import language as L
from language import Language, LanguageConfig

TABLE = 64  # a small word table keeps the tests fast; the real one has 1,024 animals


def make(animals=16, table_animals=TABLE, seasons=1, usage="equal", lexicon_seed=0, set_sentences=0):
    return Language(LanguageConfig(animals, table_animals, seasons, usage, lexicon_seed, set_sentences))


# ------------------------------------------------------------------------------------------------
# Helpers: an independent reader of sentences, and checks that several tests share
# ------------------------------------------------------------------------------------------------

ATTRIBUTE_OF_VERB = {"are": 0, "eat": 1, "live": 2, "wake": 3}
LINK_WORD = {0: [], 1: [], 2: ["in"], 3: ["at"]}  # "live in ...", "wake at ..."


def read_sentence(lang, ids):
    """Read one sentence from its token ids WITHOUT using language.py's sentence writer.

    Returns (animal number, season, attribute, value word, answer word). The answer is None for a
    statement. Fails if the sentence does not have one of the agreed forms.
    """
    words = lang.decode(ids).split()
    season = 0
    if lang.config.seasons > 1:
        assert words[0] == "in"
        season = L.SEASONS.index(words[1])
        words = words[2:]
    if words[-1] == ".":  # NOUN VERB [in/at] VALUE .
        noun, value, answer = words[0], words[-2], None
        attribute = ATTRIBUTE_OF_VERB[words[1]]
        assert words[2:-2] == LINK_WORD[attribute]
    elif words[0] == "are":  # are NOUN VALUE ? ANSWER
        assert len(words) == 5 and words[3] == "?"
        noun, value, answer, attribute = words[1], words[2], words[4], 0
    else:  # do NOUN VERB [in/at] VALUE ? ANSWER
        assert words[0] == "do" and words[-2] == "?"
        noun, value, answer = words[1], words[-3], words[-1]
        attribute = ATTRIBUTE_OF_VERB[words[2]]
        assert attribute != 0 and words[3:-3] == LINK_WORD[attribute]
    assert answer in (None, "yes", "no")
    assert value in L.VALUES[attribute]
    return lang.animal_names.index(noun), season, attribute, value, answer


def true_value(lang, animal, season, attribute):
    return L.VALUES[attribute][lang.lexicon[animal, season, attribute]]


def read_set(lang, numbers):
    """Count the sentences of a set: {(animal, season, attribute, value, answer): how many}."""
    tally = Counter()
    for number, count in zip(*np.unique(numbers, return_counts=True)):
        tally[read_sentence(lang, lang.sentence_ids(number))] += int(count)
    return tally


def row_tokens(inputs, targets, row):
    """The real tokens of one packed row: its first word, then every target that is not padding."""
    return [int(inputs[row, 0])] + [int(t) for t in targets[row] if t != L.IGNORE]


def check_truth(lang, numbers):
    """Every statement is true and every question has the right answer."""
    for (animal, season, attribute, value, answer), count in read_set(lang, numbers).items():
        is_true = value == true_value(lang, animal, season, attribute)
        if answer is None:
            assert is_true, "a statement in the training text is false"
        else:
            assert answer == ("yes" if is_true else "no"), "a question has the wrong answer"


def check_exact_composition(lang, numbers, times):
    """Every fact has exactly 9 TELL, 3 ASK true and 1 ASK for each wrong value, `times` over."""
    tally = read_set(lang, numbers)
    for animal in range(lang.config.animals):
        for season in range(lang.config.seasons):
            for attribute in range(4):
                true = true_value(lang, animal, season, attribute)
                assert tally[(animal, season, attribute, true, None)] == 9 * times
                assert tally[(animal, season, attribute, true, "yes")] == 3 * times
                for value in L.VALUES[attribute]:
                    if value != true:
                        assert tally[(animal, season, attribute, value, "no")] == 1 * times
    facts = lang.config.animals * lang.config.seasons * 4
    assert sum(tally.values()) == len(numbers) == 15 * times * facts  # and nothing else


def check_bits_against_reference(lang, seed=0, n_rows=8, context=48):
    """sample_rows (fast, step by step) and next_distribution (slow, by listing every sentence)
    are two separate implementations of the oracle. They must agree at every position."""
    inputs, targets, bits, kind = lang.sample_rows(np.random.default_rng(seed), n_rows, context)
    checked = 0
    for row in range(n_rows):
        tokens = row_tokens(inputs, targets, row)
        for t in range(len(tokens) - 1):
            distribution = lang.next_distribution(tokens[: t + 1])
            assert abs(distribution.sum() - 1) < 1e-9 and distribution.min() >= 0
            probability = distribution[targets[row, t]]
            assert probability > 0
            assert bits[row, t] == pytest.approx(-np.log2(probability), abs=1e-5)
            checked += 1
    assert checked >= 200
    return inputs, targets, bits, kind


# ------------------------------------------------------------------------------------------------
# The words and the lexicon
# ------------------------------------------------------------------------------------------------


def test_base_language_has_43_words_in_the_agreed_order():
    lang = make(animals=16, table_animals=16)
    assert lang.vocab_size == len(lang.words) == 43
    assert lang.words[:11] == [".", "?", "are", "do", "eat", "live", "in", "wake", "at", "yes", "no"]
    assert lang.words[11:27] == [word for row in L.VALUES for word in row]
    assert lang.words[27:] == [row[0] for row in L.BASE_ANIMALS]
    assert lang.seasons_in_use == []
    assert list(lang.animal_ids) == list(lang.table_animal_ids) == list(range(27, 43))


def test_word_table_covers_every_table_animal_at_every_vocabulary_size():
    small, large = make(animals=16), make(animals=64)
    assert small.words == large.words and small.vocab_size == 27 + TABLE
    assert list(small.table_animal_ids) == list(large.table_animal_ids) == list(range(27, 27 + TABLE))
    assert list(small.animal_ids) == list(range(27, 27 + 16))
    assert list(large.animal_ids) == list(large.table_animal_ids)


def test_season_words_are_present_only_with_more_than_one_season():
    lang = make(seasons=4)
    assert lang.seasons_in_use == ["winter", "spring", "summer", "fall"]
    assert lang.words[27:31] == lang.seasons_in_use
    assert lang.vocab_size == 31 + TABLE and lang.table_animal_ids[0] == 31
    assert "winter" not in make(seasons=1).words


def test_base_table_is_balanced():
    assert len(L.BASE_ANIMALS) == 16
    for attribute in range(4):
        used = Counter(row[1 + attribute] for row in L.BASE_ANIMALS)
        assert used == {value: 4 for value in L.VALUES[attribute]}


@pytest.mark.parametrize("animals", [16, 64, 256])
@pytest.mark.parametrize("seasons", [1, 4])
def test_every_animal_carries_8_bits_per_season(animals, seasons):
    lang = make(animals=animals, table_animals=256, seasons=seasons)
    assert lang.lexicon.shape == (256, seasons, 4)
    for start in range(0, 256, 16):  # every block of 16 animals, in use or not
        for season in range(seasons):
            for attribute in range(4):
                used = np.bincount(lang.lexicon[start : start + 16, season, attribute], minlength=4)
                assert list(used) == [4, 4, 4, 4]
    assert lang.knowledge_bits() == pytest.approx(8 * seasons * animals)


def test_knowledge_bits_follow_the_actual_frequencies():
    # 24 animals are not a whole number of balanced blocks, so they hold a little less than 8 bits each.
    lang = make(animals=24)
    expected = 0.0
    for attribute in range(4):
        share = np.bincount(lang.lexicon[:24, 0, attribute], minlength=4) / 24
        expected += 24 * -(share[share > 0] * np.log2(share[share > 0])).sum()
    assert lang.knowledge_bits() == pytest.approx(expected)
    assert lang.knowledge_bits() <= 8 * 24


def test_invented_names():
    names = L.animal_names(6000)  # more than there are two-syllable names
    assert names[:16] == [row[0] for row in L.BASE_ANIMALS]
    assert names[16] == "bakos"
    assert len(set(names)) == len(names)
    other_words = set(L.SENTENCE_WORDS + L.SEASONS + [word for row in L.VALUES for word in row])
    assert not other_words & set(names)
    for name in names[16:]:
        body = name[:-1]
        assert name.endswith("s") and len(body) % 2 == 0 and len(body) >= 4
        assert all(letter in L.NAME_CONSONANTS for letter in body[0::2])
        assert all(letter in L.NAME_VOWELS for letter in body[1::2])
        assert body[-1] in L.NAME_LAST_VOWELS
        assert not any(piece in name for piece in L.NAME_SKIP)  # nothing rude or confusing
    assert all(len(name) == 5 for name in names[16:1024])  # the default table: two syllables and "s"
    assert L.animal_names(100) == names[:100]  # a name depends only on the animal's number
    assert len({L.invented_name(number) for number in range(20000)}) == 20000  # candidates never repeat


def test_lexicon_is_nested():
    reference = make(animals=64, table_animals=256)
    for animals, table_animals, usage in [(16, 16, "equal"), (16, 64, "zipf"), (64, 64, "equal"), (48, 100, "zipf")]:
        lang = make(animals=animals, table_animals=table_animals, usage=usage)
        assert lang.animal_names == reference.animal_names[:table_animals]
        assert np.array_equal(lang.lexicon, reference.lexicon[:table_animals])
    # The first season of a four-season language is the one-season language.
    assert np.array_equal(make(seasons=4, table_animals=256).lexicon[:, 0], reference.lexicon[:, 0])


def test_base_animals_are_the_same_for_every_seed():
    first, second = make(lexicon_seed=0), make(lexicon_seed=1)
    hand_written = [[L.VALUES[a].index(row[1 + a]) for a in range(4)] for row in L.BASE_ANIMALS]
    assert first.lexicon[:16, 0].tolist() == second.lexicon[:16, 0].tolist() == hand_written
    assert first.animal_names == second.animal_names
    assert not np.array_equal(first.lexicon[16:], second.lexicon[16:])  # invented facts do differ


def test_usage():
    assert make(usage="equal").usage.tolist() == [1 / 16] * 16
    small, large = make(animals=16, usage="zipf").usage, make(animals=64, usage="zipf").usage
    assert small.sum() == pytest.approx(1) and large.sum() == pytest.approx(1)
    assert large[0] / large[1] == pytest.approx(2) and large[0] / large[9] == pytest.approx(10)
    # The small vocabulary is the common end of the large one: same proportions among the first 16.
    assert np.allclose(large[:16] / large[:16].sum(), small)


def test_settings_that_make_no_sense_are_refused():
    for bad in [dict(animals=65), dict(animals=0), dict(seasons=5), dict(usage="often"), dict(set_sentences=-1)]:
        with pytest.raises(ValueError):
            make(**bad)


def test_a_set_size_that_does_not_suit_the_number_of_animals_is_refused():
    # Under equal usage all animals round up or down together, so a set size that is not a multiple
    # of the number of animals gives sets of different sizes. The message says what to use instead.
    with pytest.raises(ValueError, match="multiple of the number of animals .64.*use 64 or 128, or 0"):
        make(animals=64, set_sentences=100)
    with pytest.raises(ValueError, match="use 49,984 or 50,048"):
        make(animals=64, set_sentences=50000)
    # A set smaller than the number of animals would leave some sets empty, under any usage.
    for usage in ["equal", "zipf"]:
        with pytest.raises(ValueError, match="smaller than the number of animals .64.*use at least 64"):
            make(animals=64, usage=usage, set_sentences=60)
    # What is allowed: a multiple under equal usage, any size from the number of animals up under
    # zipf (its sets are only nearly the nominal size anyway), and the automatic size always.
    for size in [64, 128, 1920]:
        assert all(len(make(animals=64, set_sentences=size).make_set(k, 0)) == size for k in range(5))
    assert make(animals=64, usage="zipf", set_sentences=100).set_sentences == 100
    assert make(animals=64, usage="zipf", set_sentences=64).set_sentences == 64
    for animals in [16, 24, 64]:
        for usage in ["equal", "zipf"]:
            assert make(animals=animals, usage=usage).set_sentences == L.automatic_set_sentences(TABLE, 1)


def test_the_tour_works_with_a_set_size_given_by_hand(capsys):
    # 240 suits the 16 animals of this language but not 64. The last table of "python language.py"
    # shows other vocabulary sizes too, and must not stumble over the set size.
    L.print_best_possible(LanguageConfig(animals=16, table_animals=TABLE, set_sentences=240))
    printed = capsys.readouterr().out
    assert "<- this language" in printed and "compare the share of facts known" in " ".join(printed.split())
    assert [line.split()[0] for line in printed.splitlines()[2:4]] == ["16", "64"]


def test_the_automatic_set_size_follows_the_table_and_is_exact_for_the_planned_vocabularies():
    assert L.automatic_set_sentences(1024, 1) == 61440 and L.automatic_set_sentences(1024, 4) == 245760
    assert L.automatic_set_sentences(16, 1) == 960
    # 16, 64, 256 and 1,024 animals in a table of 1,024: a whole number of every animal's cycle.
    for seasons in [1, 4]:
        for animals in [16, 64, 256, 1024]:
            assert L.automatic_set_sentences(1024, seasons) % (60 * seasons * animals) == 0


def test_encode_and_decode_round_trip():
    lang = make(seasons=4)
    for text in ["in winter foxes eat meat .", "in fall do bakos live in woods ? no", ""]:
        assert lang.decode(lang.encode(text)) == text
    assert lang.encode(" ".join(lang.words)) == list(range(lang.vocab_size))
    assert lang.decode(np.arange(3)) == ". ? are"
    assert lang.decode([27, L.IGNORE]) == "winter _"  # a target that is not graded
    with pytest.raises(ValueError, match="cats"):
        lang.encode("cats eat meat .")
    for not_a_token in [-1, lang.vocab_size]:
        with pytest.raises(ValueError):
            lang.decode([not_a_token])


# ------------------------------------------------------------------------------------------------
# The sentences and the exact training sets
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seasons", [1, 4])
def test_sentence_numbers_cover_every_form_of_every_fact_once(seasons):
    lang = make(animals=16, seasons=seasons)
    assert lang.n_sentences == 16 * seasons * 4 * 5
    seen = set()
    for number in range(lang.n_sentences):
        animal, season, attribute, value, answer = read_sentence(lang, lang.sentence_ids(number))
        seen.add((animal, season, attribute, value, answer is None))
        # sentence_parts is the code's own account of the same sentence; the two must agree.
        assert tuple(int(part) for part in lang.sentence_parts(number))[:3] == (animal, season, attribute)
        assert (lang.sentence_parts(number)[3] == L.TELL) == (answer is None)
    assert len(seen) == lang.n_sentences


@pytest.mark.parametrize("usage", ["equal", "zipf"])
@pytest.mark.parametrize("seasons", [1, 4])
def test_every_tell_is_true_and_every_ask_answer_is_correct(usage, seasons):
    lang = make(animals=64, seasons=seasons, usage=usage, lexicon_seed=3)
    for set_index in range(2):
        check_truth(lang, lang.make_set(set_index, data_seed=1))


@pytest.mark.parametrize("animals", [16, 64])
@pytest.mark.parametrize("seasons", [1, 4])
def test_exact_composition_under_equal_usage(animals, seasons):
    lang = make(animals=animals, seasons=seasons)
    assert lang.set_sentences == 15 * 4 * seasons * TABLE
    for set_index in [0, 5]:
        numbers = lang.make_set(set_index, data_seed=0)
        check_exact_composition(lang, numbers, times=TABLE // animals)


@pytest.mark.parametrize("seasons", [2, 3])
def test_two_and_three_seasons_also_give_exact_sets(seasons):
    lang = make(animals=16, seasons=seasons, usage="equal")
    assert lang.seasons_in_use == L.SEASONS[:seasons] and lang.vocab_size == 27 + seasons + TABLE
    numbers = lang.make_set(2, data_seed=1)
    check_truth(lang, numbers)
    check_exact_composition(lang, numbers, times=TABLE // 16)
    assert lang.knowledge_bits() == pytest.approx(8 * seasons * 16)
    check_bits_against_reference(lang)


@pytest.mark.parametrize("seasons", [1, 4])
def test_each_animals_cycle_is_spread_evenly(seasons):
    lang = make(animals=16, seasons=seasons)
    facts = seasons * 4
    for animal in range(16):
        cycle = lang.cycles[animal]
        assert len(cycle) == facts * 15
        who, season, attribute, form = lang.sentence_parts(cycle)
        assert (who == animal).all()
        fact = season * 4 + attribute
        true = lang.lexicon[animal, season, attribute]
        is_tell, is_ask_true = form == L.TELL, form == 1 + true
        around = np.arange(2 * len(cycle)) % len(cycle)  # to look at stretches that wrap around the end
        for start in range(len(cycle)):
            # Any stretch as long as the number of facts holds every fact once ...
            assert len(set(fact[around[start : start + facts]].tolist())) == facts
            # ... and any five sentences in a row are 3 TELL, 1 ASK true and 1 ASK wrong.
            five = around[start : start + 5]
            assert is_tell[five].sum() == 3 and is_ask_true[five].sum() == 1
        for one_fact in range(facts):
            mine = fact == one_fact  # this fact's 15 sentences, in the order they are used
            assert mine.sum() == 15 and is_tell[mine].sum() == 9 and is_ask_true[mine].sum() == 3
            assert len(set(form[mine].tolist())) == 5  # the TELL and all four ASKs occur
            tells = np.concatenate([is_tell[mine], is_tell[mine]])
            assert all(tells[start : start + 5].sum() == 3 for start in range(15))  # no bunching


def test_zipf_counts_stay_within_one_sentence_of_exact():
    lang = make(animals=64, usage="zipf")
    seen = np.zeros(64, dtype=np.int64)
    for k in range(1, 9):
        numbers = lang.make_set(k - 1, data_seed=0)
        per_animal = Counter()  # counted by reading the sentences themselves
        for (animal, _, _, _, _), count in read_set(lang, numbers).items():
            per_animal[animal] += count
        seen += np.array([per_animal[animal] for animal in range(64)])
        exact = k * lang.set_sentences * lang.usage
        assert np.abs(seen - exact).max() <= 1.0
        assert abs(len(numbers) - lang.set_sentences) <= 64  # the set size is only nearly nominal
    assert seen[0] / seen[1] == pytest.approx(2, rel=0.01)  # animal 1 twice as often as animal 2
    assert seen[0] > 40 * seen[63]  # and the last animal is rare


@pytest.mark.parametrize("set_sentences", [0, 100, 1000])
def test_running_counts_never_go_down(set_sentences):
    # With a small set most animals get no sentence at all in most sets. Still no count is negative.
    lang = make(animals=64, usage="zipf", set_sentences=set_sentences)
    previous = lang.animal_counts(0)
    assert previous.sum() == 0
    for k in list(range(1, 300)) + [99_999, 100_000]:
        now = lang.animal_counts(k)
        assert (now >= previous).all()
        assert np.abs(now - k * lang.set_sentences * lang.usage).max() <= 0.5 + 1e-6
        previous = now
    assert len(lang.make_set(7, 0)) == (lang.animal_counts(8) - lang.animal_counts(7)).sum()


def test_make_set_is_deterministic():
    for usage in ["equal", "zipf"]:
        one, two = make(usage=usage, seasons=4), make(usage=usage, seasons=4)
        assert np.array_equal(one.make_set(3, data_seed=7), two.make_set(3, data_seed=7))
        assert np.array_equal(one.make_set(3, data_seed=7), one.make_set(3, data_seed=7))
        assert not np.array_equal(one.make_set(3, data_seed=7), one.make_set(3, data_seed=8))
        assert not np.array_equal(one.make_set(3, data_seed=7), one.make_set(4, data_seed=7))
        # The data seed only decides the ORDER inside a set, never what is in it.
        assert np.array_equal(np.sort(one.make_set(3, data_seed=7)), np.sort(one.make_set(3, data_seed=8)))


def test_an_animal_gets_the_same_sentences_in_the_same_order_at_every_vocabulary_size():
    for seasons in [1, 4]:
        small, large = make(animals=16, seasons=seasons), make(animals=64, seasons=seasons, usage="zipf")
        assert np.array_equal(small.cycles, large.cycles[:16])
        for number in small.cycles[3, :10]:
            assert small.sentence_ids(number) == large.sentence_ids(number)


def test_every_set_has_a_new_order_but_the_same_composition_under_equal_usage():
    lang = make(animals=16)
    first = lang.make_set(0, data_seed=0)
    for set_index in [1, 2, 10]:
        other = lang.make_set(set_index, data_seed=0)
        assert not np.array_equal(first, other)
        assert np.array_equal(np.sort(first), np.sort(other))


@pytest.mark.parametrize("usage", ["equal", "zipf"])
def test_animals_outside_the_vocabulary_in_use_never_appear(usage):
    lang = make(animals=16, usage=usage)
    not_in_use = lang.table_animal_ids[16:]
    for set_index in range(3):
        inputs, targets = lang.pack_rows(lang.make_set(set_index, data_seed=2), context=96)
        assert not np.isin(inputs, not_in_use).any() and not np.isin(targets, not_in_use).any()
    inputs, targets, _, _ = lang.sample_rows(np.random.default_rng(0), 64, 96)
    assert not np.isin(inputs, not_in_use).any() and not np.isin(targets, not_in_use).any()


@pytest.mark.parametrize("usage", ["equal", "zipf"])
@pytest.mark.parametrize("seasons", [1, 4])
def test_sightings_agree_with_a_direct_count(usage, seasons):
    # 1,920 = 30 sentences per animal and set: a multiple of the 64 animals, as equal usage needs,
    # but not a whole number of an animal's cycle (60 or 240 sentences), so the sets differ.
    lang = make(animals=64, usage=usage, seasons=seasons, set_sentences=1920)
    per_fact = Counter()
    for k in range(1, 6):
        for (animal, season, attribute, _, _), count in read_set(lang, lang.make_set(k - 1, 0)).items():
            per_fact[(animal, season, attribute)] += count
        counted = np.zeros(64)
        for (animal, _, _), count in per_fact.items():
            counted[animal] += count / (seasons * 4)
        per_animal, mean = lang.sightings(k)
        assert np.allclose(per_animal, counted)
        assert mean == pytest.approx(sum(per_fact.values()) / (64 * seasons * 4))
    assert lang.sightings(0)[1] == 0


def test_under_equal_usage_every_fact_is_seen_exactly_the_same_number_of_times():
    lang = make(animals=16, seasons=4)
    per_fact = Counter()
    for (animal, season, attribute, _, _), count in read_set(lang, lang.make_set(0, 0)).items():
        per_fact[(animal, season, attribute)] += count
    per_animal, mean = lang.sightings(1)
    assert len(per_fact) == 16 * 16 and set(per_fact.values()) == {mean} == {15 * TABLE // 16}
    assert per_animal.tolist() == [mean] * 16


# ------------------------------------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seasons, usage, context", [(1, "equal", 96), (4, "zipf", 32), (1, "zipf", 6), (4, "equal", 8)])
def test_pack_rows(seasons, usage, context):
    lang = make(animals=64, seasons=seasons, usage=usage, set_sentences=3008)  # 47 for each animal
    numbers = lang.make_set(1, data_seed=5)
    sentences = [lang.sentence_ids(number) for number in numbers]
    inputs, targets = lang.pack_rows(numbers, context)
    assert inputs.shape == targets.shape == (len(inputs), context)
    assert inputs.dtype == np.int64 and targets.dtype == np.int64
    next_sentence = 0
    for row in range(len(inputs)):
        real = targets[row] != L.IGNORE
        n_real = int(real.sum())
        assert real[:n_real].all()  # padding sits only at the end of a row
        assert (inputs[row, n_real + 1 :] == L.PAD_ID).all()
        assert (inputs[row, 1 : n_real + 1] == targets[row, : context - 1][: n_real]).all()  # shifted by one
        # The row is made of whole sentences, in the order of the set, starting at a sentence start.
        tokens, rebuilt = row_tokens(inputs, targets, row), []
        while len(rebuilt) < len(tokens):
            rebuilt += sentences[next_sentence]
            next_sentence += 1
        assert rebuilt == tokens and len(tokens) <= context + 1
        if next_sentence < len(sentences):  # the row stopped only because the next sentence did not fit
            assert len(tokens) + len(sentences[next_sentence]) > context + 1
    assert next_sentence == len(sentences)  # every sentence of the set exactly once


def test_pack_rows_also_takes_lists_of_token_ids():
    lang = make(animals=16, seasons=4)
    numbers = lang.make_set(0, data_seed=0)[:500]
    from_numbers = lang.pack_rows(numbers, 40)
    from_lists = lang.pack_rows([lang.sentence_ids(number) for number in numbers], 40)
    assert np.array_equal(from_numbers[0], from_lists[0]) and np.array_equal(from_numbers[1], from_lists[1])
    inputs, targets = lang.pack_rows([lang.encode("foxes eat meat ."), lang.encode("are owls brown ? yes")], 6)
    assert lang.decode(inputs[0]) == "foxes eat meat . . ." and targets[0].tolist()[3:] == [L.IGNORE] * 3
    assert lang.decode(inputs[1]) == "are owls brown ? yes ."
    empty_inputs, empty_targets = lang.pack_rows(numbers[:0], 40)
    assert empty_inputs.shape == empty_targets.shape == (0, 40)


def test_pack_rows_refuses_rows_shorter_than_a_sentence():
    with pytest.raises(ValueError):
        make(seasons=1).pack_rows(make(seasons=1).make_set(0, 0), context=5)  # longest sentence: 7 words
    with pytest.raises(ValueError):
        make(seasons=4).pack_rows(make(seasons=4).make_set(0, 0), context=7)  # longest sentence: 9 words


# ------------------------------------------------------------------------------------------------
# The oracle
# ------------------------------------------------------------------------------------------------


def test_next_distribution_gives_the_numbers_of_the_design():
    lang = make(animals=16)
    fox = lang.word_id["foxes"]

    def after(text):
        distribution = lang.next_distribution(lang.encode(text))
        return {lang.words[token]: p for token, p in enumerate(distribution) if p > 0}

    start = after("")
    assert start.pop("are") == pytest.approx(0.4 * 1 / 4) and start.pop("do") == pytest.approx(0.4 * 3 / 4)
    assert set(start) == set(lang.animal_names[:16])
    assert all(p == pytest.approx(0.6 / 16) for p in start.values())
    assert after("foxes") == pytest.approx({"are": 1 / 4, "eat": 1 / 4, "live": 1 / 4, "wake": 1 / 4})
    assert after("foxes live") == {"in": 1} and after("foxes wake") == {"at": 1}
    assert after("foxes live in") == {"woods": 1} and after("foxes live in woods") == {".": 1}
    assert after("do foxes") == pytest.approx({"eat": 1 / 3, "live": 1 / 3, "wake": 1 / 3})
    assert after("do foxes eat") == pytest.approx({"meat": 1 / 2, "plants": 1 / 6, "insects": 1 / 6, "seeds": 1 / 6})
    assert after("are foxes") == pytest.approx({"red": 1 / 2, "brown": 1 / 6, "green": 1 / 6, "black": 1 / 6})
    assert after("do foxes eat meat") == {"?": 1} and after("do foxes eat meat ?") == {"yes": 1}
    assert after("do foxes eat seeds ?") == {"no": 1} and after("are foxes red ?") == {"yes": 1}
    assert after("do")[lang.words[fox]] == pytest.approx(1 / 16) and len(after("do")) == 16
    # A finished sentence tells nothing about the next one.
    assert after("foxes eat meat . do owls wake at dawn ? no") == pytest.approx(after(""))
    assert after("foxes eat meat . do owls") == pytest.approx(after("do owls"))


def test_next_distribution_with_zipf_usage_and_seasons():
    lang = make(animals=64, usage="zipf", seasons=4)
    assert lang.next_distribution([])[lang.word_id["in"]] == 1
    seasons = lang.next_distribution(lang.encode("in"))
    assert all(seasons[lang.word_id[word]] == pytest.approx(1 / 4) for word in L.SEASONS)
    start = lang.next_distribution(lang.encode("in fall"))
    assert np.allclose(start[lang.animal_ids], 0.6 * lang.usage)
    assert start[lang.word_id["are"]] == pytest.approx(0.1) and start[lang.word_id["do"]] == pytest.approx(0.3)
    assert np.allclose(lang.next_distribution(lang.encode("in fall do"))[lang.animal_ids], lang.usage)


def test_next_distribution_sums_to_one_after_many_prefixes():
    for lang in [make(animals=16), make(animals=64, usage="zipf", seasons=4)]:
        inputs, targets, _, _ = lang.sample_rows(np.random.default_rng(1), 6, 40)
        prefixes = [[]]
        for row in range(6):
            tokens = row_tokens(inputs, targets, row)
            prefixes += [tokens[:cut] for cut in range(1, len(tokens) + 1)]  # cut everywhere
        assert len(prefixes) > 150
        for prefix in prefixes:
            distribution = lang.next_distribution(prefix)
            assert distribution.shape == (lang.vocab_size,) and distribution.dtype == np.float64
            assert abs(distribution.sum() - 1) < 1e-9 and distribution.min() >= 0


def test_next_distribution_refuses_text_that_cannot_occur():
    lang = make(animals=16)
    for impossible in ["foxes foxes", "foxes eat .", "bakos", "do bakos eat", ". ."]:
        with pytest.raises(ValueError):
            lang.next_distribution(lang.encode(impossible))  # "bakos" is in the table but not in use


@pytest.mark.parametrize("usage", ["equal", "zipf"])
@pytest.mark.parametrize("seasons", [1, 4])
def test_sample_rows_bits_equal_the_reference_oracle(usage, seasons):
    lang = make(animals=64, usage=usage, seasons=seasons, lexicon_seed=2)
    inputs, targets, bits, kind = check_bits_against_reference(lang, seed=3)
    assert inputs.shape == targets.shape == bits.shape == kind.shape == (8, 48)
    assert bits.dtype == np.float32 and kind.dtype == np.uint8
    padding = targets == L.IGNORE
    assert ((kind == L.KIND_PADDING) == padding).all() and (bits[padding] == 0).all()
    # The value of a TELL and the answer of an ASK are certain for the oracle (0 bits), and they
    # are marked: one value for every "." and one answer for every "?".
    value_words, answers = lang.value_ids.reshape(-1), [lang.yes_id, lang.no_id]
    assert np.isin(targets[kind == L.KIND_TELL_VALUE], value_words).all()
    assert ((kind == L.KIND_ASK_ANSWER) == np.isin(targets, answers)).all()
    assert (bits[kind == L.KIND_TELL_VALUE] == 0).all() and (bits[kind == L.KIND_ASK_ANSWER] == 0).all()
    assert (kind == L.KIND_TELL_VALUE).sum() == (targets == lang.word_id["."]).sum() > 0
    assert (kind == L.KIND_ASK_ANSWER).sum() == (targets == lang.word_id["?"]).sum() > 0


def test_sample_rows_is_repeatable_and_packs_whole_sentences():
    lang = make(animals=64, usage="zipf")
    one = lang.sample_rows(np.random.default_rng(5), 32, 96)
    two = lang.sample_rows(np.random.default_rng(5), 32, 96)
    other = lang.sample_rows(np.random.default_rng(6), 32, 96)
    assert all(np.array_equal(a, b) for a, b in zip(one, two))
    assert not np.array_equal(one[0], other[0])
    whole = {tuple(lang.sentence_ids(number)) for number in range(lang.n_sentences)}
    for row in range(32):
        tokens, sentence = row_tokens(one[0], one[1], row), []
        assert len(tokens) > 96 + 1 - lang.longest_sentence  # the row is as full as it can be
        for token in tokens:
            sentence.append(token)
            if token in (lang.word_id["."], lang.yes_id, lang.no_id):
                assert tuple(sentence) in whole
                sentence = []
        assert sentence == []  # the row ends at the end of a sentence


def test_sample_rows_draws_the_designed_mix():
    lang = make(animals=64, usage="zipf")
    inputs, targets, _, _ = lang.sample_rows(np.random.default_rng(0), 600, 96)
    tells = (targets == lang.word_id["."]).sum()
    yes, no = (targets == lang.yes_id).sum(), (targets == lang.no_id).sum()
    sentences = tells + yes + no
    assert sentences > 8000
    assert tells / sentences == pytest.approx(0.6, abs=0.03)
    assert yes / sentences == pytest.approx(0.2, abs=0.03)
    words = np.concatenate([inputs[:, :1].reshape(-1), targets[targets != L.IGNORE]])
    share = np.bincount(words, minlength=lang.vocab_size)[lang.animal_ids] / sentences
    assert np.abs(share - lang.usage).max() < 0.02  # animals are named as often as usage says


@pytest.mark.parametrize("usage", ["equal", "zipf"])
@pytest.mark.parametrize("seasons", [1, 4])
def test_step_probabilities_multiply_to_the_sentence_probability(usage, seasons):
    lang = make(animals=64, usage=usage, seasons=seasons)
    from_steps = 2.0 ** -lang.sentence_bits.sum(axis=1)
    assert np.allclose(from_steps, lang.sentence_probability, rtol=1e-9)
    assert lang.sentence_probability.sum() == pytest.approx(1)


@pytest.mark.parametrize("seasons", [1, 4])
def test_composition_of_zipf_sets_matches_the_oracle_in_the_long_run(seasons):
    lang = make(animals=64, usage="zipf", seasons=seasons)
    sets = 20
    counted = np.zeros(lang.n_sentences)
    for set_index in range(sets):
        counted += np.bincount(lang.make_set(set_index, data_seed=0), minlength=lang.n_sentences)
    expected = sets * lang.set_sentences * lang.sentence_probability
    # Each animal is within half a sentence of its exact share, and inside an animal's cycle a
    # sentence is used at most 9 times per round, so no sentence can be further off than this.
    assert np.abs(counted - expected).max() <= 9.5
    assert np.abs(counted - expected).mean() < 1.0


@pytest.mark.parametrize("seasons", [1, 4])
def test_composition_of_an_equal_usage_set_matches_the_oracle(seasons):
    lang = make(animals=16, seasons=seasons)
    numbers = lang.make_set(0, data_seed=0)
    counted = np.bincount(numbers, minlength=lang.n_sentences) / len(numbers)
    for number in range(lang.n_sentences):
        ids = lang.sentence_ids(number)
        probability = 1.0  # the chain rule: multiply the probability of each word given the words before
        for place, token in enumerate(ids):
            probability *= lang.next_distribution(ids[:place])[token]
        assert counted[number] == pytest.approx(probability, rel=1e-9)


def test_best_possible_loss_matches_hand_arithmetic():
    # Per sentence the oracle is surprised by: the animal, the attribute (2 bits) and the form.
    form_bits = -(0.6 * np.log2(0.6) + 0.2 * np.log2(0.2) + 3 * (1 / 15) * np.log2(1 / 15))
    mean_words = 0.6 * (4 + 4 + 5 + 5) / 4 + 0.4 * (5 + 6 + 7 + 7) / 4
    for animals in [16, 64]:
        expected = (np.log2(animals) + 2 + form_bits) / mean_words
        assert make(animals=animals).best_possible_loss() == pytest.approx(expected)
    # A bigger vocabulary has a higher best possible loss; a season word adds 2 bits and 2 words.
    assert make(animals=64).best_possible_loss() > make(animals=16).best_possible_loss()
    expected = (np.log2(16) + 2 + form_bits + 2) / (mean_words + 2)
    assert make(animals=16, seasons=4).best_possible_loss() == pytest.approx(expected)


# ------------------------------------------------------------------------------------------------
# Probes
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("usage", ["equal", "zipf"])
@pytest.mark.parametrize("seasons", [1, 4])
def test_fact_probes_for_animals_in_use(usage, seasons):
    lang = make(animals=16, usage=usage, seasons=seasons)
    probes = lang.fact_probes()
    assert len(probes) == 16 * seasons * 4
    assert [(p.animal, p.season, p.attribute) for p in probes] == [
        (animal, season, attribute) for animal in range(16) for season in range(seasons) for attribute in range(4)
    ]
    for probe in probes:
        distribution = lang.next_distribution(probe.prefix)  # the probe starts at the start of a row
        assert distribution[probe.answer] == pytest.approx(1)
        assert lang.words[probe.answer] == true_value(lang, probe.animal, probe.season, probe.attribute)
        assert lang.animal_ids[probe.animal] in probe.prefix
    assert lang.decode(probes[1].prefix) == ("foxes eat" if seasons == 1 else "in winter foxes eat")
    assert len(lang.fact_probes(max_animals=3)) == 3 * seasons * 4


@pytest.mark.parametrize("seasons", [1, 4])
def test_fact_probes_for_unseen_animals(seasons):
    small, everything = make(animals=16, seasons=seasons), make(animals=TABLE, seasons=seasons)
    probes = small.fact_probes(which="unseen")
    assert len(probes) == (TABLE - 16) * seasons * 4
    assert {probe.animal for probe in probes} == set(range(16, TABLE))
    for probe in probes:
        # The language that does use these animals says what the right answer is ...
        assert everything.next_distribution(probe.prefix)[probe.answer] == pytest.approx(1)
        assert small.words[probe.answer] == true_value(small, probe.animal, probe.season, probe.attribute)
    # ... while the small language never says anything about them.
    with pytest.raises(ValueError):
        small.next_distribution(probes[0].prefix)
    few = small.fact_probes(which="unseen", max_animals=8)
    assert {probe.animal for probe in few} == set(range(16, 24))
    assert make(animals=TABLE).fact_probes(which="unseen") == []
    with pytest.raises(ValueError):
        small.fact_probes(which="everything")


@pytest.mark.parametrize("seasons", [1, 4])
def test_question_probes(seasons):
    lang = make(animals=16, seasons=seasons, usage="zipf")
    probes = lang.question_probes()
    assert len(probes) == 16 * seasons * 4 * 4
    assert sum(probe.answer == lang.yes_id for probe in probes) == 16 * seasons * 4  # one yes per fact
    assert sum(probe.answer == lang.no_id for probe in probes) == 16 * seasons * 4 * 3
    for probe in probes:
        assert lang.words[probe.prefix[-1]] == "?"
        assert lang.next_distribution(probe.prefix)[probe.answer] == pytest.approx(1)
        shown = lang.words[probe.prefix[-2]]
        is_true = shown == true_value(lang, probe.animal, probe.season, probe.attribute)
        assert probe.answer == (lang.yes_id if is_true else lang.no_id)
    some = lang.question_probes(max_items=40, rng=np.random.default_rng(0))
    assert len(some) == 40 and all(probe in probes for probe in some)
    assert lang.question_probes(max_items=40) == lang.question_probes(max_items=40)  # repeatable
    unseen = lang.question_probes(which="unseen", max_items=20)
    assert len(unseen) == 20 and all(probe.animal >= 16 for probe in unseen)


# ------------------------------------------------------------------------------------------------
# An edited base table
# ------------------------------------------------------------------------------------------------


def test_an_unbalanced_base_table_still_works(monkeypatch):
    # Paint every black animal red: now 8 are red, 4 brown, 4 green and none is black.
    edited = [(name, "red" if color == "black" else color, food, home, wake)
              for name, color, food, home, wake in L.BASE_ANIMALS]
    monkeypatch.setattr(L, "BASE_ANIMALS", edited)
    lang = make(animals=16)
    assert np.bincount(lang.lexicon[:16, 0, 0], minlength=4).tolist() == [8, 4, 4, 0]
    # Color is now worth -(1/2 log2 1/2 + 2 x 1/4 log2 1/4) = 1.5 bits instead of 2.
    assert lang.knowledge_bits() == pytest.approx(16 * (1.5 + 2 + 2 + 2))
    numbers = lang.make_set(0, data_seed=0)
    check_truth(lang, numbers)
    check_exact_composition(lang, numbers, times=TABLE // 16)
    check_bits_against_reference(lang)
    for probe in lang.fact_probes():
        assert lang.next_distribution(probe.prefix)[probe.answer] == pytest.approx(1)
    # Invented animals are untouched, so a larger vocabulary mixes the edited block with balanced ones.
    larger = make(animals=64, usage="zipf")
    assert np.bincount(larger.lexicon[:64, 0, 0], minlength=4).tolist() == [20, 16, 16, 12]
    share = np.array([20, 16, 16, 12]) / 64
    assert larger.knowledge_bits() == pytest.approx(64 * (-(share * np.log2(share)).sum() + 6))
    check_truth(larger, larger.make_set(0, data_seed=0))
    check_bits_against_reference(larger)


def test_a_wrong_word_in_the_base_table_is_reported(monkeypatch):
    edited = [("foxes", "purple", "meat", "woods", "dusk")] + L.BASE_ANIMALS[1:]
    monkeypatch.setattr(L, "BASE_ANIMALS", edited)
    with pytest.raises(ValueError, match="purple"):
        make()

"""
language.py - the toy language about animals, its exact training sets, and the oracle.

What this file is for
---------------------
To test whether a bigger vocabulary needs a bigger model, we need a language in which we know
EXACTLY how much there is to know about every word. No real language is like that, so we invent
one. Every animal word carries four facts (color, food, home, waking time). Each fact has one of
four values, so each fact is worth 2 bits and each animal is worth 8 bits, at every vocabulary
size. A bigger vocabulary means more animal words and nothing else.

This file makes three things:
  1. the words and the table of facts (the "lexicon");
  2. the training text, in "sets" with exact counts of every animal, fact and sentence form;
  3. the "oracle": the program that knows every right answer and the exact probability of every
     next word. A model is graded against it.

How to run it
-------------
    python language.py                       # the base language: 16 animals, 43 words
    python language.py --animals 1024        # the largest vocabulary
    python language.py --usage zipf          # some animals are used far more often than others
    python language.py --seasons 4           # 32 bits per animal instead of 8

Terms used in this file
-----------------------
token    One word, stored as a whole number (its "token id"). A model never sees letters, only
         these numbers. In this language one word is always exactly one token.
fact     One true thing about one animal: (animal, season, attribute) -> value.
bit      The unit of knowledge. Choosing one of four equally likely values takes two yes/no
         questions, so such a fact is worth 2 bits.
oracle   The designed process the text is drawn from. Because we wrote it ourselves we can
         compute the exact probability of each next word. No model can predict better than this.
loss     How surprised a predictor is by the next word, on average, in bits per word. Some words
         can never be predicted (which animal comes next is a free choice), so even the oracle has
         a loss above zero. That number is the "best possible loss".
usage    How often each animal is written about. "equal": all the same. "zipf": the second animal
         half as often as the first, the third a third as often, and so on.
set      A fixed number of training sentences with exact counts of every animal, fact and sentence
         form, in shuffled order. Training text is one set after another.
row      What a model reads in one go: a few whole sentences, one after another.
"""

import argparse
import textwrap
from dataclasses import dataclass, replace

import numpy as np

# ------------------------------------------------------------------------------------------------
# The words
# ------------------------------------------------------------------------------------------------

# The eleven words that hold sentences together.
SENTENCE_WORDS = [".", "?", "are", "do", "eat", "live", "in", "wake", "at", "yes", "no"]

# The four things that can be known about an animal, and the four possible values of each.
ATTRIBUTES = ["color", "food", "home", "waking time"]
VALUES = [
    ["red", "brown", "green", "black"],
    ["meat", "plants", "insects", "seeds"],
    ["woods", "fields", "rivers", "caves"],
    ["dawn", "noon", "dusk", "night"],
]

# Season words. They are part of the language only when seasons > 1.
SEASONS = ["winter", "spring", "summer", "fall"]

# How a fact is stated (TELL) and how it is asked about (ASK), one form for each attribute.
# NOUN, VALUE and ANSWER are blanks to fill in. With seasons > 1 every sentence also starts with
# "in SEASON", for example "in winter foxes eat meat ."
TELL_FORMS = [
    "NOUN are VALUE .",
    "NOUN eat VALUE .",
    "NOUN live in VALUE .",
    "NOUN wake at VALUE .",
]
ASK_FORMS = [
    "are NOUN VALUE ? ANSWER",
    "do NOUN eat VALUE ? ANSWER",
    "do NOUN live in VALUE ? ANSWER",
    "do NOUN wake at VALUE ? ANSWER",
]
VERBS = ["are", "eat", "live", "wake"]  # the word that reveals which attribute a sentence is about

# The base animals: name, color, food, home, waking time. The table is BALANCED: every value is
# used by exactly four of the sixteen animals, which is what makes each fact worth exactly 2 bits.
# You may edit it. Nothing below assumes it is balanced: chance levels and bits are always
# computed from the table as it actually is.
BASE_ANIMALS = [
    ("foxes", "red", "meat", "woods", "dusk"),
    ("bears", "brown", "plants", "caves", "noon"),
    ("wolves", "black", "meat", "caves", "night"),
    ("owls", "brown", "meat", "woods", "night"),
    ("bats", "black", "insects", "caves", "night"),
    ("frogs", "green", "insects", "rivers", "dusk"),
    ("snakes", "green", "insects", "fields", "noon"),
    ("crows", "black", "seeds", "fields", "dawn"),
    ("hens", "red", "seeds", "fields", "dawn"),
    ("ducks", "brown", "plants", "rivers", "dawn"),
    ("deer", "red", "plants", "woods", "dusk"),
    ("mice", "brown", "seeds", "fields", "night"),
    ("toads", "green", "insects", "rivers", "dusk"),
    ("robins", "red", "seeds", "woods", "dawn"),
    ("otters", "black", "meat", "rivers", "noon"),
    ("lizards", "green", "plants", "caves", "noon"),
]

# Invented animals get their facts in blocks of this many animals. Inside a block every value is
# used equally often, so invented animals are worth exactly as many bits as the base animals.
BLOCK = 16

# Letters for the names of invented animals: consonant-vowel syllables plus a final "s" ("bakos").
NAME_CONSONANTS = "bdfgklmnprtvz"
NAME_VOWELS = "aeiou"
NAME_LAST_VOWELS = "aou"  # names end in -as, -os or -us, which keeps them unlike English plurals
# A made-up name is skipped when it contains one of these pieces, because it would then read like
# a rude word or like a real word that would confuse (a real animal, "bonus", "virus", ...).
NAME_SKIP = (
    "anus bad bat ber bob bof bog bol bon boz bub bud bug bul bum bur but dag deg dem dev did dik "
    "dil din div dod dof dog dop dov duf duk dum dup fag fak fal fap fat feg fek fen fet fig fik "
    "fod fok fot fud fuk fup fut gat gaz gek gen get gez gil gip git giz god gof gon guk gul gur "
    "kaf kag kak kap kat kik kil kok kon kot kud kuf kuk kul kum kun kut lam lap leb lez lob lol "
    "lor lot lub lul lup mag mak mam mem men min mof mol mon mor muf muk mul muz nad naz neg nek "
    "nig nin nip nob nog nud nuk nun nut pak pap pat pav ped peg pek pen per pet pig pik pim pin "
    "pit piz pof pol pom pop pot poz pub pud puf pug puk pul pum pup put puz rab ram ran rap rat "
    "red ret rez rim rin rom tab tap tat teg tet tif tig tik tit tod tof tok tor tuk tum tun tup "
    "vag ved ven vet vir vol vom vul zeb zer zet zez zid zik zit ziz zob zog zor zul"
).split()

# ------------------------------------------------------------------------------------------------
# How often each kind of sentence is used
# ------------------------------------------------------------------------------------------------

# Of every 15 sentences about one fact:
CYCLE = 15  # sentences about one fact in one full cycle
TELLS = 9  # statements: "foxes eat meat ."                                  (60%)
ASKS_TRUE = 3  # questions that show the true value: "do foxes eat meat ? yes"  (20%)
ASKS_WRONG = 1  # questions for EACH of the three wrong values: "... plants ? no" (20% / 3 each)

# The order in which the forms come round in the training text: of any five sentences in a row,
# three are statements, one asks about the true value and one about a wrong value. Three times
# this pattern is one cycle of 15. If you change the numbers above, change this pattern to match;
# a test checks that the training sets and the oracle agree.
FORM_PATTERN = ["tell", "ask true", "tell", "ask wrong", "tell"]

P_TELL = TELLS / CYCLE  # the chance that a sentence is a statement
P_ASK = (CYCLE - TELLS) / CYCLE  # the chance that it is a question
P_ASK_TRUE = ASKS_TRUE / CYCLE  # ... a question showing the true value
P_ASK_WRONG = ASKS_WRONG / CYCLE  # ... a question showing one particular wrong value

# Each fact has five distinct sentences, called its forms:
# form 0 is the TELL, and form 1 + v is the ASK that shows value number v (one of them is true).
FORMS = 5
TELL = 0

# ------------------------------------------------------------------------------------------------
# Small technical constants
# ------------------------------------------------------------------------------------------------

PAD_ID = 0  # rows are filled up with this token id after their last sentence (it is never graded)
IGNORE = -100  # target value that means "do not grade this position" (PyTorch's convention)

# What kind of word a graded position is (returned by sample_rows).
KIND_STRUCTURE = 0  # sentence structure, or a free choice nobody can know in advance
KIND_TELL_VALUE = 1  # the value in a TELL: predicting it needs the fact
KIND_ASK_ANSWER = 2  # the yes or no after a question: predicting it needs the fact
KIND_PADDING = 255  # not a word at all

# Three separate streams of random numbers. Each random generator below is seeded with a list
# that starts with one of these numbers, so the facts, the order inside an animal's cycle and the
# shuffling of a set can never accidentally use the same random numbers.
LEXICON_STREAM = 1
CYCLE_STREAM = 2
SHUFFLE_STREAM = 3


@dataclass(frozen=True)
class LanguageConfig:
    animals: int = 16  # animal words in use: the vocabulary-size dial
    table_animals: int = 1024  # animal rows in the word table, shared by every model of a round
    seasons: int = 1  # 1 = 8 bits per animal, 4 = 32 bits per animal
    usage: str = "equal"  # "equal" or "zipf"
    lexicon_seed: int = 0  # decides the facts of the invented animals
    set_sentences: int = 0  # 0 = automatic (15 x 4 x seasons x table_animals)


@dataclass
class Probe:
    """A short prompt with one right next word. It starts at the beginning of a row, so the
    answer cannot be copied from earlier text: it has to come from what the model has learned."""

    prefix: list  # token ids of the prompt, for example "foxes eat" or "do foxes eat meat ?"
    answer: int  # token id of the right next word
    animal: int  # the animal's number: 0 is the first animal (a row of the lexicon), not a token id
    season: int  # 0 when there is only one season
    attribute: int  # 0 color, 1 food, 2 home, 3 waking time


# ------------------------------------------------------------------------------------------------
# Names and facts
# ------------------------------------------------------------------------------------------------


def invented_name(number):
    """Candidate name number `number` (0, 1, 2, ...) for an invented animal. Number 0 is "bakos"."""
    first = [c + v for c in NAME_CONSONANTS for v in NAME_VOWELS]  # ba be bi bo bu da ...: 65
    last = [c + v for c in NAME_CONSONANTS for v in NAME_LAST_VOWELS]  # ba bo bu da do ...: 39
    pairs = len(first) * len(last)  # 2,535 different two-syllable names
    more, pair = divmod(number, pairs)
    # A fixed scramble, so that neighbors do not get look-alike names. It visits every pair once.
    pair = (pair * 1597 + 13) % pairs
    name = first[pair // len(last)] + last[pair % len(last)] + "s"
    while more > 0:  # only when the two-syllable names run out: put more syllables in front
        more, extra = divmod(more - 1, len(first))
        name = first[extra] + name
    return name


def animal_names(count):
    """The names of the first `count` animals: the base animals, then invented ones.

    The name of animal number i is always the same, however many animals are asked for.
    """
    names = [row[0] for row in BASE_ANIMALS]
    taken = set(SENTENCE_WORDS + SEASONS + names)
    for row in VALUES:
        taken.update(row)
    number = 0
    while len(names) < count:
        name = invented_name(number)
        number += 1
        if name in taken:  # never reuse a word that already means something
            continue
        if any(piece in name for piece in NAME_SKIP):
            continue
        names.append(name)
    return names[:count]


def balanced_facts(rng, animals):
    """Random facts for a group of animals in which every value is used equally often.

    Picture a deck with as many cards as animals, a quarter of them for each value. For every
    attribute the deck is shuffled and each animal is dealt one card.
    Returns [animals, 4]: for each animal and attribute, the number (0 to 3) of the true value.
    """
    deck = np.arange(animals) % 4
    return np.stack([rng.permutation(deck) for attribute in ATTRIBUTES], axis=1)


def build_lexicon(table_animals, seasons, lexicon_seed):
    """The table of what is true: lexicon[animal, season, attribute] = number of the true value.

    The lexicon is NESTED: the facts of animal number i depend only on (lexicon_seed, i). They do
    not change when more animals are put in use or the table is made bigger. That is what lets us
    say the small vocabulary is a part of the large one.
    """
    n_base = len(BASE_ANIMALS)
    lexicon = np.zeros((max(table_animals, n_base), seasons, 4), dtype=np.int64)
    for season in range(seasons):
        # Block 0 is the base animals. Blocks 1, 2, 3, ... are the invented animals, 16 at a time.
        rng = np.random.default_rng([LEXICON_STREAM, lexicon_seed, season, 0])
        lexicon[:n_base, season] = balanced_facts(rng, n_base)
        for start in range(n_base, table_animals, BLOCK):
            block = 1 + (start - n_base) // BLOCK
            rng = np.random.default_rng([LEXICON_STREAM, lexicon_seed, season, block])
            stop = min(start + BLOCK, table_animals)
            lexicon[start:stop, season] = balanced_facts(rng, BLOCK)[: stop - start]
    # The first season of the base animals is not random: it is the hand-written table.
    for animal, row in enumerate(BASE_ANIMALS):
        for attribute, value in enumerate(row[1:]):
            if value not in VALUES[attribute]:
                raise ValueError(
                    f'BASE_ANIMALS: "{value}" is not a {ATTRIBUTES[attribute]}. '
                    f"Choose one of: {', '.join(VALUES[attribute])}."
                )
            lexicon[animal, 0, attribute] = VALUES[attribute].index(value)
    return lexicon[:table_animals]


def usage_shares(animals, usage):
    """The probability that a sentence is about each animal. Returns [animals], summing to 1."""
    if usage == "equal":
        return np.full(animals, 1.0 / animals)
    # "zipf": the 2nd animal is used half as often as the 1st, the 3rd a third as often, and so
    # on. Real languages look roughly like this: a few very common words and a long tail of rare
    # ones. New animals join at the rare end, so the small vocabulary is exactly the most common
    # words of the large one.
    weights = 1.0 / np.arange(1, animals + 1)
    return weights / weights.sum()


def check_config(config):
    """Stop early, with a plain message, if the settings make no sense."""
    if not 1 <= config.animals <= config.table_animals:
        raise ValueError(
            f"animals must be between 1 and table_animals ({config.table_animals}); "
            f"got {config.animals}. The word table has to hold every animal in use."
        )
    if not 1 <= config.seasons <= len(SEASONS):
        raise ValueError(f"seasons must be between 1 and {len(SEASONS)}; got {config.seasons}.")
    if config.usage not in ("equal", "zipf"):
        raise ValueError(f'usage must be "equal" or "zipf"; got "{config.usage}".')
    if config.set_sentences < 0 or config.lexicon_seed < 0:
        raise ValueError("set_sentences and lexicon_seed must not be negative.")
    # A set size given by hand must suit the number of animals, or "a set" stops being a fixed
    # amount of text (make_set says why). 0, the automatic size, is not checked here.
    size, animals = config.set_sentences, config.animals
    if 0 < size < animals:
        raise ValueError(
            f"set_sentences ({size:,}) is smaller than the number of animals ({animals:,}), so "
            f"some sets would be empty: use at least {animals:,}, or 0 for the automatic size."
        )
    if size > 0 and config.usage == "equal" and size % animals != 0:
        below = size - size % animals
        raise ValueError(
            f"with equal usage set_sentences ({size:,}) must be a multiple of the number of "
            f"animals ({animals:,}), or the sets differ in size: use {below:,} or "
            f"{below + animals:,}, or 0 for the automatic size."
        )


def automatic_set_sentences(table_animals, seasons):
    """The set size used when set_sentences is 0: 15 sentences for every fact of every animal
    of the TABLE. It follows the size of the table, not the number of animals in use, so every
    vocabulary of a round gets sets of the same size (61,440 for one season and 1,024 animals)."""
    return CYCLE * 4 * seasons * table_animals


# ------------------------------------------------------------------------------------------------
# Packing sentences into rows
# ------------------------------------------------------------------------------------------------


def padded_table(lists, empty, dtype):
    """Stack lists of different lengths into one array, filling the unused ends with `empty`."""
    width = max(len(row) for row in lists)
    return np.array([list(row) + [empty] * (width - len(row)) for row in lists], dtype=dtype)


def lay_out(lengths, row_tokens):
    """Decide where each sentence goes when whole sentences are packed into rows.

    A row takes sentences, in order, for as long as the next one still fits. Returns
    (row_of, start_of, n_rows): the row of each sentence and the position where it starts.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if len(lengths) > 0 and lengths.max() > row_tokens:
        raise ValueError(f"a row of {row_tokens} words is too short for the longest sentence.")
    ends = np.cumsum(lengths)  # where each sentence would end if all were written on one line
    first_of_row, used_before_row = [], []
    first, used = 0, 0
    while first < len(lengths):
        first_of_row.append(first)
        used_before_row.append(used)
        # The first sentence that would end beyond this row starts the next row.
        first = int(np.searchsorted(ends, used + row_tokens, side="right"))
        used = int(ends[first - 1])
    n_rows = len(first_of_row)
    per_row = np.diff(first_of_row + [len(lengths)])
    row_of = np.repeat(np.arange(n_rows), per_row)
    start_of = ends - lengths - np.array(used_before_row, dtype=np.int64)[row_of]
    return row_of, start_of, n_rows


def write_rows(table, numbers, lengths, row_of, start_of, shape, empty):
    """Copy row `numbers[i]` of `table` into the place lay_out chose for sentence i."""
    rows = np.full(shape, empty, dtype=table.dtype)
    for place in range(table.shape[1]):
        has = lengths > place  # the sentences that have a word at this place
        rows[row_of[has], start_of[has] + place] = table[numbers[has], place]
    return rows


def inputs_and_targets(rows):
    """Split rows of context + 1 tokens into what the model reads and what it should predict.

    The model reads a row left to right and, at every position, predicts the NEXT word. So the
    targets are the same row shifted by one. Positions after the last sentence are filled with
    PAD_ID on the input side and IGNORE on the target side, so they are never graded.
    """
    inputs = np.where(rows[:, :-1] < 0, PAD_ID, rows[:, :-1])
    targets = np.where(rows[:, 1:] < 0, IGNORE, rows[:, 1:])
    return inputs, targets


# ------------------------------------------------------------------------------------------------
# The language
# ------------------------------------------------------------------------------------------------


class Language:
    """One toy language: its words, its facts, its training sets and its oracle.

    The main things it holds (all made in __init__):
      words             every word; the position of a word in this list is its token id
      vocab_size        how many words there are, counting every animal of the table
      animal_ids        token ids of the animals in use
      table_animal_ids  token ids of every animal in the table
      lexicon           [table_animals, seasons, 4]: the number (0 to 3) of each fact's true value
      usage             [animals]: the probability that a sentence is about each animal in use
      set_sentences     the nominal number of sentences in one training set
    """

    def __init__(self, config):
        check_config(config)
        self.config = config
        self.attributes = ATTRIBUTES
        self.values = VALUES
        # The season words in use. With one season no season word is ever said, so this is empty.
        self.seasons_in_use = SEASONS[: config.seasons] if config.seasons > 1 else []

        # Token ids are handed out in this order: sentence words, value words, season words (only
        # when seasons > 1), then the animal words of the WHOLE table. A model trained on 16
        # animals still has a row for every animal of the table; it just never reads about them.
        self.animal_names = animal_names(config.table_animals)
        value_words = [word for row in VALUES for word in row]
        self.words = SENTENCE_WORDS + value_words + self.seasons_in_use + self.animal_names
        self.vocab_size = len(self.words)
        self.word_id = {word: token for token, word in enumerate(self.words)}
        if len(self.word_id) != len(self.words):
            raise ValueError("two words of the language are spelled the same; rename one.")
        first_animal = self.vocab_size - config.table_animals
        self.table_animal_ids = np.arange(first_animal, self.vocab_size)
        self.animal_ids = self.table_animal_ids[: config.animals]
        self.value_ids = np.array([[self.word_id[word] for word in row] for row in VALUES])
        self.yes_id = self.word_id["yes"]
        self.no_id = self.word_id["no"]

        self.lexicon = build_lexicon(config.table_animals, config.seasons, config.lexicon_seed)
        self.usage = usage_shares(config.animals, config.usage)
        self.facts_per_animal = config.seasons * 4
        self.cycle_length = self.facts_per_animal * CYCLE
        # The automatic set size holds every fact's 15-sentence cycle a whole number of times when
        # usage is equal and the number of animals divides table_animals: exact counts, no rounding.
        automatic = automatic_set_sentences(config.table_animals, config.seasons)
        self.set_sentences = config.set_sentences or automatic

        self._build_sentence_tables()
        self.cycles = np.stack([self._build_cycle(animal) for animal in range(config.animals)])
        self._whole_sentences = None  # made on first use by next_distribution
        self._probe_cache = {}  # probes never change, so each list is built only once

    # --------------------------------------------------------------------------------------------
    # Sentences
    # --------------------------------------------------------------------------------------------

    def sentence_words(self, animal, season, attribute, shown=None):
        """The words of one sentence about one fact.

        shown=None gives the statement (TELL). shown=v gives the question (ASK) that shows value
        number v; its answer is "yes" only when v is the true value.
        """
        true = self.lexicon[animal, season, attribute]
        value = true if shown is None else shown
        form = TELL_FORMS[attribute] if shown is None else ASK_FORMS[attribute]
        blanks = {
            "NOUN": self.animal_names[animal],
            "VALUE": VALUES[attribute][value],
            "ANSWER": "yes" if value == true else "no",
        }
        words = [blanks.get(word, word) for word in form.split()]
        if self.config.seasons > 1:
            words = ["in", SEASONS[season]] + words
        return words

    def oracle_steps(self, animal, season, attribute, shown=None):
        """How the oracle sees one sentence, word by word, reading left to right.

        Returns (probabilities, kinds), one entry per word: the probability of that word given
        the words before it, and what kind of word it is (see the KIND_ constants).
        """
        is_tell = shown is None
        true = self.lexicon[animal, season, attribute]
        probabilities, kinds = [], []
        if self.config.seasons > 1:
            probabilities += [1.0, 1.0 / self.config.seasons]  # "in" is certain; the season is not
            kinds += [KIND_STRUCTURE, KIND_STRUCTURE]
        form = TELL_FORMS[attribute] if is_tell else ASK_FORMS[attribute]
        for place, word in enumerate(form.split()):
            probability, kind = 1.0, KIND_STRUCTURE  # "in", "at", "." and "?" are certain
            if word == "NOUN":
                # Which animal. A TELL starts with the animal, so there the word also says "this is
                # a statement". In an ASK that was already said by the first word.
                probability = self.usage[animal] * (P_TELL if is_tell else 1.0)
            elif word == "VALUE" and is_tell:
                kind = KIND_TELL_VALUE  # certain for the oracle, because it knows the fact
            elif word == "VALUE":
                # A question shows the true value half the time and each wrong value 1/6 of it.
                probability = (P_ASK_TRUE if shown == true else P_ASK_WRONG) / P_ASK
            elif word == "ANSWER":
                kind = KIND_ASK_ANSWER  # certain for the oracle
            elif place == 0 and word == "are":
                probability = P_ASK * 1 / 4  # a question, and about color (one attribute of four)
            elif place == 0 and word == "do":
                probability = P_ASK * 3 / 4  # a question about one of the other three attributes
            elif word in VERBS:
                # The verb says which attribute: one of four in a statement, and one of the three
                # that are left after "do" in a question.
                probability = 1 / 4 if is_tell else 1 / 3
            probabilities.append(float(probability))
            kinds.append(kind)
        return probabilities, kinds

    def sentence_share(self, animal, season, attribute, shown=None):
        """The designed probability of one whole sentence. This IS the oracle's definition:
        pick a season (all equally likely), an animal (by usage), an attribute (all equally
        likely) and a sentence form (60% TELL, 20% ASK true, 20%/3 for each wrong ASK)."""
        true = self.lexicon[animal, season, attribute]
        if shown is None:
            form_share = P_TELL
        elif shown == true:
            form_share = P_ASK_TRUE
        else:
            form_share = P_ASK_WRONG
        return float((1 / self.config.seasons) * self.usage[animal] * (1 / 4) * form_share)

    def sentence_number(self, animal, season, attribute, form):
        """Every distinct sentence has a number. The five forms of one fact sit side by side."""
        return ((animal * self.config.seasons + season) * 4 + attribute) * FORMS + form

    def sentence_parts(self, numbers):
        """The reverse of sentence_number: returns (animal, season, attribute, form)."""
        fact, form = np.divmod(numbers, FORMS)
        animal_season, attribute = np.divmod(fact, 4)
        animal, season = np.divmod(animal_season, self.config.seasons)
        return animal, season, attribute, form

    def sentence_ids(self, number):
        """The token ids of sentence `number`, as a list."""
        return self.sentence_tokens[number, : self.sentence_length[number]].tolist()

    def _build_sentence_tables(self):
        """Write out every distinct sentence about the animals in use, once.

        Afterwards a training set is just an array of sentence numbers, which keeps data
        preparation fast. For every sentence we store its token ids, the oracle's probability of
        each word, the kind of each word, and the probability of the whole sentence.
        """
        tokens, steps, kinds, shares = [], [], [], []
        for animal in range(self.config.animals):
            for season in range(self.config.seasons):
                for attribute in range(4):
                    for form in range(FORMS):
                        shown = None if form == TELL else form - 1
                        words = self.sentence_words(animal, season, attribute, shown)
                        chances, word_kinds = self.oracle_steps(animal, season, attribute, shown)
                        tokens.append([self.word_id[word] for word in words])
                        steps.append(chances)
                        kinds.append(word_kinds)
                        shares.append(self.sentence_share(animal, season, attribute, shown))
        self.n_sentences = len(tokens)
        self.sentence_length = np.array([len(row) for row in tokens], dtype=np.int64)
        self.longest_sentence = int(self.sentence_length.max())
        self.sentence_tokens = padded_table(tokens, -1, np.int64)
        self.sentence_kind = padded_table(kinds, KIND_PADDING, np.uint8)
        # Surprise in bits: -log2(probability). A certain word (probability 1) costs 0 bits, a
        # one-in-four word costs 2 bits. Added up over a sentence it is the news in that sentence.
        self.sentence_bits = np.log2(1.0 / padded_table(steps, 1.0, np.float64))
        self.sentence_probability = np.array(shares, dtype=np.float64)

    # --------------------------------------------------------------------------------------------
    # Exact training sets
    # --------------------------------------------------------------------------------------------

    def _build_cycle(self, animal):
        """The fixed order in which one animal's sentences are used, over and over.

        The cycle holds 15 sentences for each of the animal's facts. Two repeating patterns run
        side by side in it. The FACT follows a shuffled order of the animal's facts, repeated 15
        times, so any stretch of the cycle is spread evenly over the facts. The FORM follows
        FORM_PATTERN, so any five sentences in a row are three statements and two questions.
        The number of facts (4, 8, 12 or 16) shares no factor with 5, so the two patterns drift
        against each other like gears with 4 and 5 teeth. After one full cycle every fact has met
        every step of the form pattern exactly three times: 9 TELL, 3 ASK true, 3 ASK wrong.
        """
        rng = np.random.default_rng([CYCLE_STREAM, self.config.lexicon_seed, animal])
        n_facts = self.facts_per_animal
        fact_order = rng.permutation(n_facts).tolist()
        start = int(rng.integers(len(FORM_PATTERN)))
        wrong_left = {}  # for each fact: the wrong values not asked about yet, in shuffled order
        for fact in range(n_facts):
            true = self.lexicon[animal, fact // 4, fact % 4]
            wrong_left[fact] = rng.permutation([v for v in range(4) if v != true]).tolist()
        cycle = []
        for i in range(n_facts * CYCLE):
            fact = fact_order[i % n_facts]
            season, attribute = divmod(fact, 4)
            step = FORM_PATTERN[(start + i) % len(FORM_PATTERN)]
            if step == "tell":
                form = TELL
            elif step == "ask true":
                form = 1 + int(self.lexicon[animal, season, attribute])
            else:
                form = 1 + wrong_left[fact].pop()  # each wrong value is asked about once
            cycle.append(self.sentence_number(animal, season, attribute, form))
        if any(wrong_left.values()):
            raise ValueError("FORM_PATTERN does not give every fact each wrong value exactly once.")
        return np.array(cycle, dtype=np.int64)

    def animal_counts(self, sets_done):
        """How many sentences about each animal the first `sets_done` sets hold, added up.

        The exact share would be sets_done x set_sentences x usage, which is usually not a whole
        number. Rounding the RUNNING TOTAL (not each set) keeps every animal within half a
        sentence of its exact share for ever, and a running total can never go down, so no set
        gets a negative count.
        """
        exact = sets_done * self.set_sentences * self.usage
        return np.floor(exact + 0.5).astype(np.int64)

    def make_set(self, set_index, data_seed):
        """The sentences of training set number `set_index` (0 is the first), in shuffled order.

        Returns an array of sentence numbers; sentence_ids(number) gives the token ids of one, and
        pack_rows turns the whole array into rows for training. The result depends only on the
        language settings, data_seed and set_index. It does not depend on the model in any way,
        so models of different sizes trained with the same seed read exactly the same text.

        How exact is a set? Every animal's running total is rounded to a whole sentence (see
        animal_counts), so it depends on the set size:
          - Under "equal" usage every set has exactly the same composition when set_sentences is
            a multiple of 60 x seasons x animals: a whole number of every animal's cycle. The
            automatic size is such a multiple for 16, 64, 256 and 1,024 animals in a table of
            1,024. A multiple of the number of animals alone gives sets of exactly set_sentences
            sentences whose composition evens out only over several sets; check_config refuses
            a hand-given size that is not even that. (With the automatic size and a number of
            animals that does not divide it, such as 100, all animals round up or down together
            and the sets take turns between two sizes, 61,400 and 61,500.)
          - Under "zipf" usage the real size of a set strays from set_sentences by about 0.1%
            at most. That is a measurement, not a limit: at 1,024 animals the sets were within
            37 sentences of 61,440 over the first 300 sets and within 54 over 20,000 sets. The
            more sets are looked at, the larger the largest stray; with fewer animals it is
            smaller.
        sightings() always counts the sentences that were really made.
        """
        before = self.animal_counts(set_index)
        counts = self.animal_counts(set_index + 1) - before
        animal = np.repeat(np.arange(self.config.animals), counts)
        # Every animal carries on in its own cycle where the set before stopped.
        first = np.cumsum(counts) - counts  # where each animal's stretch begins in this set
        step = before[animal] + np.arange(len(animal)) - first[animal]
        sentences = self.cycles[animal, step % self.cycle_length]
        rng = np.random.default_rng([SHUFFLE_STREAM, data_seed, set_index])
        return rng.permutation(sentences)

    def sightings(self, sets_done):
        """How much practice the first `sets_done` sets gave. Returns (per_animal, mean).

        per_animal[i] is how many times each fact of animal i has been seen so far, on average
        over its facts. Every sentence about a fact counts as one sighting, statement or question.
        mean is the average over all facts of all animals in use.
        """
        per_animal = self.animal_counts(sets_done) / self.facts_per_animal
        return per_animal, float(per_animal.mean())

    def pack_rows(self, sentences, context):
        """Pack WHOLE sentences into rows of context + 1 tokens. Returns (inputs, targets).

        Both have shape [rows, context]. A row is filled with sentences for as long as the next
        one fits; the rest of the row is padding. So every row starts at the start of a sentence,
        no sentence is cut in two, and every sentence is used exactly once.
        `sentences` is what make_set returns (sentence numbers). A list of token id lists also
        works.
        """
        if len(sentences) > 0 and not np.isscalar(sentences[0]):
            table = padded_table(sentences, -1, np.int64)  # a list of token id lists
            lengths = np.array([len(sentence) for sentence in sentences], dtype=np.int64)
            numbers = np.arange(len(sentences))
        else:
            table = self.sentence_tokens
            numbers = np.asarray(sentences, dtype=np.int64)
            lengths = self.sentence_length[numbers]
        row_of, start_of, n_rows = lay_out(lengths, context + 1)
        shape = (n_rows, context + 1)
        rows = write_rows(table, numbers, lengths, row_of, start_of, shape, empty=-1)
        return inputs_and_targets(rows)

    # --------------------------------------------------------------------------------------------
    # The oracle
    # --------------------------------------------------------------------------------------------

    def sample_rows(self, rng, n_rows, context):
        """Rows for evaluation: sentences drawn at random from the oracle, packed like pack_rows.

        rng is a numpy random generator (or a whole number to seed one).
        Returns (inputs, targets, bits, kind), all of shape [n_rows, context]:
          bits   float32, the oracle's surprise at each target word, -log2 P(word | the words
                 before it in the row); 0 at padding. Its average is the best possible loss.
          kind   uint8, what each target word is: one of the KIND_ constants.

        One small imperfection, for the record. The oracle does not look at how much room is left
        in the row. Near the end of a row only short sentences still fit, so a model that keeps
        track of its position could guess the sentence form there slightly better than the oracle.
        The possible gain has nothing to do with facts and is tiny at the default row length:
        about 0.002 bits per word at context 96. Every row has one such end, so the gain is
        larger for shorter rows: about 0.01 bits per word at context 24 and 0.02 at context 12.
        In other words, the best possible loss is accurate to about 0.002 bits per word at
        context 96, and a gap that close to 0 (on either side) means "as good as the oracle".
        """
        rng = np.random.default_rng(rng)
        # Draw as many sentences as could ever fit, then keep the first n_rows rows.
        most_per_row = (context + 1) // int(self.sentence_length.min())
        n_draws = n_rows * most_per_row
        numbers = rng.choice(self.n_sentences, size=n_draws, p=self.sentence_probability)
        lengths = self.sentence_length[numbers]
        row_of, start_of, _ = lay_out(lengths, context + 1)
        keep = row_of < n_rows
        numbers, lengths = numbers[keep], lengths[keep]
        row_of, start_of = row_of[keep], start_of[keep]
        shape = (n_rows, context + 1)
        tokens = write_rows(self.sentence_tokens, numbers, lengths, row_of, start_of, shape, -1)
        bits = write_rows(self.sentence_bits, numbers, lengths, row_of, start_of, shape, 0.0)
        kind = write_rows(
            self.sentence_kind, numbers, lengths, row_of, start_of, shape, KIND_PADDING
        )
        inputs, targets = inputs_and_targets(tokens)
        # The first word of a row is read but never predicted, so it has no target, bits or kind.
        return inputs, targets, bits[:, 1:].astype(np.float32), kind[:, 1:]

    def next_distribution(self, prefix):
        """The exact probability of every possible next word after `prefix`.

        prefix is a list of token ids from the start of a row (it may be empty). Returns
        float64 [vocab_size], summing to 1.

        This is the slow, obviously correct version of the oracle, written in a completely
        different way from oracle_steps so that tests can check one against the other. It does no
        clever reasoning: it lists every sentence that could still be under way, weighs each by
        its designed probability, and adds up what each one would say next.
        Raises ValueError when the prefix cannot occur in this language (for example it names an
        animal that is not in use, or it contains padding).
        """
        if self._whole_sentences is None:
            self._whole_sentences = {tuple(self.sentence_ids(n)) for n in range(self.n_sentences)}
        # Sentences are drawn independently, so finished sentences tell us nothing about the next
        # word. Drop them; only the unfinished sentence at the end matters.
        unfinished = []
        for token in prefix:
            unfinished.append(int(token))
            if tuple(unfinished) in self._whole_sentences:
                unfinished = []
        # Which sentences begin with exactly these words?
        could_be = np.ones(self.n_sentences, dtype=bool)
        if len(unfinished) >= self.longest_sentence:
            could_be[:] = False
        else:
            for place, token in enumerate(unfinished):
                could_be &= self.sentence_tokens[:, place] == token
        if not could_be.any():
            known = range(self.vocab_size)
            said = " ".join(self.words[t] if t in known else f"<{t}>" for t in unfinished)
            raise ValueError(f'no sentence of this language starts with "{said}".')
        next_words = self.sentence_tokens[could_be, len(unfinished)]
        weights = self.sentence_probability[could_be]
        distribution = np.bincount(next_words, weights=weights, minlength=self.vocab_size)
        return distribution / distribution.sum()

    def best_possible_loss(self):
        """The oracle's own loss on a long stretch of text, in bits per word.

        No model can score lower on average. It is above zero because some words are free choices:
        nobody can know which animal the next sentence will be about.
        (The floor measured on rows differs a little from this number, because the first word of
        every row is read but never predicted. Grading uses the bits from sample_rows.)
        """
        bits_per_sentence = self.sentence_bits.sum(axis=1)
        mean_bits = (self.sentence_probability * bits_per_sentence).sum()
        mean_words = (self.sentence_probability * self.sentence_length).sum()
        return float(mean_bits / mean_words)

    def knowledge_bits(self):
        """Total bits of facts held by the animals in use.

        Computed from how often each value really occurs among the animals in use, never assumed.
        If all four values of an attribute are equally common, each fact is worth 2 bits. If one
        value is more common than the others the facts are easier to guess and worth less.
        """
        in_use = self.lexicon[: self.config.animals]
        total = 0.0
        for season in range(self.config.seasons):
            for attribute in range(4):
                counts = np.bincount(in_use[:, season, attribute], minlength=4)
                share = counts[counts > 0] / self.config.animals
                bits_per_fact = -(share * np.log2(share)).sum()  # the entropy of this attribute
                total += self.config.animals * bits_per_fact
        return float(total)

    # --------------------------------------------------------------------------------------------
    # Probes: short prompts used for grading
    # --------------------------------------------------------------------------------------------

    def _probe_animals(self, which, max_animals):
        """The animal numbers a probe list covers: the first max_animals of the chosen group."""
        if which == "in_use":
            animals = range(self.config.animals)
        elif which == "unseen":  # in the word table, but never in any training text
            animals = range(self.config.animals, self.config.table_animals)
        else:
            raise ValueError(f'which must be "in_use" or "unseen"; got "{which}".')
        return animals if max_animals is None else animals[:max_animals]

    def fact_probes(self, which="in_use", max_animals=None):
        """One probe per fact: the start of its TELL ("foxes eat") and the id of the true value.

        The order is by animal, then season, then attribute. which="unseen" gives probes about
        animals that are in the word table but not in use, for the leak check.
        """
        key = ("fact", which, max_animals)
        if key not in self._probe_cache:
            probes = []
            for animal in self._probe_animals(which, max_animals):
                for season in range(self.config.seasons):
                    for attribute in range(4):
                        words = self.sentence_words(animal, season, attribute)
                        ids = [self.word_id[word] for word in words]
                        # The last two words of a TELL are the value and the "."
                        probes.append(Probe(ids[:-2], ids[-2], animal, season, attribute))
            self._probe_cache[key] = probes
        return list(self._probe_cache[key])

    def question_probes(self, which="in_use", max_items=None, rng=None):
        """ASK prompts ending in "?", for every fact and every shown value; the answer is the id
        of "yes" or "no". Each fact gives one yes item and three no items.

        The order is by animal, season, attribute, then shown value. With max_items a random
        selection of that many is returned (chosen with rng; the same every time if rng is None).
        """
        key = ("question", which)
        if key not in self._probe_cache:
            probes = []
            for animal in self._probe_animals(which, None):
                for season in range(self.config.seasons):
                    for attribute in range(4):
                        for shown in range(4):
                            words = self.sentence_words(animal, season, attribute, shown)
                            ids = [self.word_id[word] for word in words]
                            probes.append(Probe(ids[:-1], ids[-1], animal, season, attribute))
            self._probe_cache[key] = probes
        probes = self._probe_cache[key]
        if max_items is not None and max_items < len(probes):
            rng = np.random.default_rng(0 if rng is None else rng)
            chosen = np.sort(rng.choice(len(probes), size=max_items, replace=False))
            return [probes[i] for i in chosen]
        return list(probes)

    # --------------------------------------------------------------------------------------------
    # Text in and out
    # --------------------------------------------------------------------------------------------

    def encode(self, text):
        """Turn text into a list of token ids. Words are separated by spaces, also "." and "?"."""
        ids = []
        for word in text.lower().split():
            if word not in self.word_id:
                # "?" and "." come before the last words, so the list does not end in ". ."
                some = self.animal_names[:3] + ["are", "do", "eat", "live", "in", "wake", "at"]
                some += ["?", "."] + [row[0] for row in VALUES] + self.seasons_in_use[:1]
                raise ValueError(
                    f'"{word}" is not a word of this language. Some words it knows: '
                    f'{" ".join(some)}. Put a space between all words, also before "." and "?".'
                )
            ids.append(self.word_id[word])
        return ids

    def decode(self, ids):
        """Turn token ids back into text, words separated by spaces.

        A target that is not graded (IGNORE, -100) is shown as "_".
        """
        words = []
        for token in ids:
            token = int(token)
            if token == IGNORE:
                words.append("_")
            elif 0 <= token < self.vocab_size:
                words.append(self.words[token])
            else:
                raise ValueError(f"{token} is not a token id of this language.")
        return " ".join(words)


# ------------------------------------------------------------------------------------------------
# Running this file: a guided tour of the language
# ------------------------------------------------------------------------------------------------


def print_words(language):
    config = language.config
    names = language.animal_names[: config.animals]
    if len(names) > 22:
        names = names[:18] + ["..."] + names[-3:]
    print("THE WORDS (one word is one token: a model sees each word as a whole number)")
    print(f"  sentence words ({len(SENTENCE_WORDS)}): {' '.join(SENTENCE_WORDS)}")
    for attribute, row in zip(ATTRIBUTES, VALUES):
        print(f"  {attribute} ({len(row)}): {' '.join(row)}")
    if language.seasons_in_use:
        print(f"  seasons ({config.seasons}): {' '.join(language.seasons_in_use)}")
    print(f"  animals in use ({config.animals:,}): {' '.join(names)}")
    unused = config.table_animals - config.animals
    in_use = language.vocab_size - unused
    print(f"  words in use: {in_use:,}")
    print(f"  rows in the word table of every model: {language.vocab_size:,} "
          f"({unused:,} animals are in the table but never appear in the text)")
    print(f'  for example, the token id of "{names[0]}" is {language.word_id[names[0]]}')
    print()


def print_lexicon(language, invented=4):
    config = language.config
    n_base = min(len(BASE_ANIMALS), config.table_animals)
    shown = list(range(min(n_base + invented, config.table_animals)))
    title = "THE LEXICON (what is true about each animal"
    print(title + (", in the first season)" if config.seasons > 1 else ")"))
    print(f"  {'animal':10s} " + " ".join(f"{attribute:12s}" for attribute in ATTRIBUTES))
    for animal in shown:
        facts = [VALUES[a][language.lexicon[animal, 0, a]] for a in range(4)]
        note = "" if animal < n_base else "   invented"
        note += "" if animal < config.animals else ", not in use"
        name = language.animal_names[animal]
        print(f"  {name:10s} " + " ".join(f"{value:12s}" for value in facts) + note)
    if config.table_animals > len(shown):
        more = config.table_animals - len(shown)
        print(f"  ... and {more:,} more invented animals with random facts")
    if config.seasons > 1:
        print(f"  With {config.seasons} seasons every animal has four facts per season. "
              f"{language.animal_names[0]}:")
        for season in range(config.seasons):
            facts = [VALUES[a][language.lexicon[0, season, a]] for a in range(4)]
            print(f"  {'in ' + SEASONS[season]:10s} " + " ".join(f"{v:12s}" for v in facts))
    print()


def print_set(language, data_seed):
    config = language.config
    numbers = language.make_set(0, data_seed)
    animal, season, attribute, form = language.sentence_parts(numbers)
    true = language.lexicon[animal, season, attribute]
    per_animal = np.bincount(animal, minlength=config.animals)
    tells = int((form == TELL).sum())
    asks_yes = int((form == 1 + true).sum())
    some = sorted(set(range(config.animals)[:3]) | set(range(config.animals)[-3:]))
    print("ONE TRAINING SET (the first one; every set is shuffled afresh)")
    print(f"  sentences: {len(numbers):,} (nominal set size {language.set_sentences:,})")
    print(f"  statements: {tells:,}   questions answered yes: {asks_yes:,}   "
          f"questions answered no: {len(numbers) - tells - asks_yes:,}")
    print("  sentences about some animals, most used first: "
          + ", ".join(f"{language.animal_names[a]} {per_animal[a]:,}" for a in some))
    per_fact = per_animal / language.facts_per_animal
    print(f"  times each fact is seen in this set: {per_fact.max():,.1f} for the most used animal, "
          f"{per_fact.min():,.1f} for the least used")
    print(f"  distinct sentences in the whole language: {language.n_sentences:,} "
          f"(5 for each of {config.animals * language.facts_per_animal:,} facts)")
    print()


def print_rows(language, data_seed, context, how_many=2):
    inputs, targets = language.pack_rows(language.make_set(0, data_seed), context)
    print(f"SAMPLE ROWS (whole sentences packed into rows of {context + 1} words; "
          f"this set makes {len(inputs):,} rows)")
    for row in range(min(how_many, len(inputs))):
        ids = [inputs[row, 0]] + [t for t in targets[row] if t != IGNORE]
        text = textwrap.fill(language.decode(ids), width=96, initial_indent=f"  row {row}: ",
                             subsequent_indent="         ")
        print(text)
    print()


def word_note(language, token, kind, probability):
    """A few plain words on why the oracle gives a word the probability it does."""
    if kind == KIND_TELL_VALUE:
        return "a fact: certain for whoever knows it"
    if kind == KIND_ASK_ANSWER:
        return "an answer: certain for whoever knows the fact"
    if token in language.value_ids:
        return "the value a question shows: the true one half the time"
    if probability < 1:
        return "a free choice: nobody can know it in advance"
    return "fixed by the form of the sentence"


def print_annotated_row(language, data_seed, context=19):
    inputs, targets, bits, kind = language.sample_rows(data_seed, 1, context)
    ids = [int(inputs[0, 0])] + [int(t) for t in targets[0] if t != IGNORE]
    kinds = [KIND_STRUCTURE] + [int(k) for k in kind[0] if k != KIND_PADDING]
    print(f"ONE ROW, WORD BY WORD (a short row of {context + 1} words so it fits on the screen)")
    print("  The oracle reads left to right and says how likely each word is before seeing it.")
    print(f"  {'word':10s} {'probability':>11s} {'surprise in bits':>17s}")
    for place, token in enumerate(ids):
        probability = language.next_distribution(ids[:place])[token]
        note = word_note(language, token, kinds[place], probability)
        surprise = np.log2(1 / probability)
        print(f"  {language.words[token]:10s} {probability:11.4f} {surprise:17.2f}   {note}")
    print("  Surprise is -log2(probability). Its average over the words is the best possible loss.")
    print()


def print_best_possible(config):
    print("KNOWLEDGE AND THE BEST POSSIBLE LOSS")
    print(f"  {'animal words':>12s} {'words in use':>13s} {'bits of facts':>14s} "
          f"{'bits per animal':>16s} {'best possible loss (bits per word)':>36s}")
    sizes = sorted({16, 64, 256, 1024, config.animals})
    for animals in [size for size in sizes if size <= config.table_animals]:
        # The same language, resized. The set size plays no part in these numbers, so the
        # automatic one is used: a size given by hand need not suit every number of animals.
        language = Language(replace(config, animals=animals, set_sentences=0))
        in_use = language.vocab_size - (config.table_animals - animals)
        bits = language.knowledge_bits()
        mark = "   <- this language" if animals == config.animals else ""
        print(f"  {animals:12,d} {in_use:13,d} {bits:14,.0f} {bits / animals:16.2f} "
              f"{language.best_possible_loss():36.3f}{mark}")
    print(textwrap.fill(
        "The best possible loss rises with the number of animal words even for a perfect model: "
        "with more animals to choose among, nobody can guess which animal the next sentence is "
        "about. That part of the loss is not missing knowledge. The bits per animal stay the same. "
        "So never compare the raw loss of two vocabularies: compare the share of facts known. (The "
        "gap to the best possible loss can be compared only at 0, where it means perfect at "
        "every vocabulary size.)",
        width=96, initial_indent="  ", subsequent_indent="  "))


def main():
    defaults = LanguageConfig()
    parser = argparse.ArgumentParser(
        description="Show the toy language, one training set and the oracle."
    )
    parser.add_argument("--animals", type=int, default=defaults.animals,
                        help="animal words in use: the vocabulary-size dial")
    parser.add_argument("--table-animals", type=int, default=defaults.table_animals,
                        help="animal rows in the word table, shared by every model of a round")
    parser.add_argument("--seasons", type=int, default=defaults.seasons,
                        help="1 = 8 bits per animal, 4 = 32 bits per animal")
    parser.add_argument("--usage", choices=["equal", "zipf"], default=defaults.usage,
                        help="how use is spread over the animals")
    parser.add_argument("--lexicon-seed", type=int, default=defaults.lexicon_seed,
                        help="decides the facts of the invented animals")
    parser.add_argument("--set-sentences", type=int, default=defaults.set_sentences,
                        help="sentences per training set; 0 = automatic")
    parser.add_argument("--data-seed", type=int, default=0,
                        help="decides the shuffling of the set that is shown")
    parser.add_argument("--context", type=int, default=96,
                        help="words a model reads per row, for the sample rows")
    args = parser.parse_args()

    config = LanguageConfig(
        animals=args.animals, table_animals=args.table_animals, seasons=args.seasons,
        usage=args.usage, lexicon_seed=args.lexicon_seed, set_sentences=args.set_sentences,
    )
    try:
        language = Language(config)
    except ValueError as problem:
        parser.error(str(problem))  # print the plain message and stop
    print_words(language)
    print_lexicon(language)
    print_set(language, args.data_seed)
    print_rows(language, args.data_seed, args.context)
    print_annotated_row(language, args.data_seed)
    print_best_possible(config)


if __name__ == "__main__":
    main()

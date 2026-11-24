"""
v2/lang.py - the language of the second experiment (PLAN.md, part B.1 to B.5).

What is in this file
--------------------
  1. The words and their token ids (B.1).
  2. The facts: which value each creature has for each kind of fact (B.2).
  3. The held-back pairs: pairs of creatures never named together in training (B.3).
  4. How often each creature is used: "equal" or "zipf" (B.4).
  5. The training sentences, drawn a whole batch at a time (B.3).
  6. A separate, simple truth check, used by the tests to check the sentence maker.
  7. The exam: five fixed sets of prompts and their scores (B.5).

Nothing here depends on the seed of a run except the training sentences.

Terms
-----
token     One word, stored as a whole number (its token id).
kind      One kind of fact, for example colour ("are red") or food ("eat meat").
value     One of the four answers of a kind, for example "red".
list      The one to four creatures a sentence is about: "a", "a and b", "a , b and c", ...
by use    Drawn with the creature weights of B.4, so common creatures come up more often.
"""

import numpy as np

# ------------------------------------------------------------------------------------------------
# The words (B.1)
# ------------------------------------------------------------------------------------------------

PAD = "<pad>"  # fills a row after the end of its sentence; always token 0
PAD_ID = 0

# The 14 fixed words, in this order: token ids 1 to 14.
FIXED_WORDS = [".", "?", ",", "and", "not", "are", "do", "eat", "live", "in", "wake", "at",
               "yes", "no"]

# The 32 kinds of fact: the linking phrase, then the four values.
# One change from PLAN.md B.1: the plan's value "mice" (kind 11, "catch") is also the name of
# creature 11, and every token must be a different word. So, as B.1 asks, that value is replaced
# by "gnats", which is no creature name and no other word of the language.
KINDS = [
    (["are"], ["red", "brown", "green", "black"]),
    (["eat"], ["meat", "plants", "insects", "seeds"]),
    (["live", "in"], ["woods", "fields", "rivers", "caves"]),
    (["wake", "at"], ["dawn", "noon", "dusk", "night"]),
    (["fear"], ["fire", "water", "noise", "light"]),
    (["like"], ["sun", "rain", "snow", "wind"]),
    (["carry"], ["sticks", "stones", "leaves", "shells"]),
    (["drink"], ["dew", "sap", "milk", "nectar"]),
    (["climb"], ["trees", "rocks", "walls", "hills"]),
    (["dig"], ["holes", "tunnels", "pits", "dens"]),
    (["catch"], ["flies", "moths", "gnats", "fish"]),
    (["build"], ["nests", "dams", "webs", "mounds"]),
    (["follow"], ["tracks", "scents", "sounds", "stars"]),
    (["avoid"], ["roads", "farms", "towns", "camps"]),
    (["cross"], ["bridges", "marshes", "dunes", "ice"]),
    (["guard"], ["eggs", "cubs", "paths", "pools"]),
    (["smell"], ["smoke", "rot", "musk", "salt"]),
    (["hear"], ["thunder", "steps", "calls", "wings"]),
    (["chew"], ["bark", "roots", "grass", "bones"]),
    (["watch"], ["clouds", "shadows", "flames", "waves"]),
    (["dodge"], ["traps", "nets", "snares", "hooks"]),
    (["seek"], ["shade", "warmth", "shelter", "company"]),
    (["gather"], ["twigs", "moss", "feathers", "pebbles"]),
    (["hoard"], ["nuts", "berries", "grain", "honey"]),
    (["visit"], ["ponds", "cliffs", "meadows", "ruins"]),
    (["shed"], ["fur", "skin", "horns", "claws"]),
    (["mark"], ["logs", "stumps", "boulders", "fences"]),
    (["raid"], ["hives", "barns", "orchards", "burrows"]),
    (["ride"], ["currents", "gusts", "tides", "floods"]),
    (["dread"], ["frost", "drought", "storms", "eclipses"]),
    (["swallow"], ["gravel", "sand", "mud", "clay"]),
    (["trail"], ["herds", "flocks", "swarms", "packs"]),
]
# The 32 kinds added for the third experiment (v3/PLAN.md, Part B.1), used only with 64 kinds
# (128 bits per creature). The same pattern as kinds 5 to 32: one linking word and four values,
# all plain words that differ from every other token and from every creature name (a test checks
# this). KINDS itself is left as it was, so everything with 4 to 32 kinds stays exactly the same.
MORE_KINDS = [
    (["hunt"], ["hares", "crabs", "worms", "beetles"]),
    (["fight"], ["rivals", "giants", "ghosts", "kings"]),
    (["sing"], ["songs", "hymns", "tunes", "chants"]),
    (["wear"], ["hats", "boots", "coats", "rings"]),
    (["paint"], ["doors", "masks", "pots", "carts"]),
    (["sniff"], ["flowers", "spices", "herbs", "mint"]),
    (["taste"], ["sugar", "lemons", "pepper", "vinegar"]),
    (["love"], ["music", "games", "poems", "dances"]),
    (["hate"], ["dust", "smog", "crowds", "bells"]),
    (["keep"], ["coins", "keys", "gems", "maps"]),
    (["throw"], ["balls", "spears", "darts", "discs"]),
    (["pull"], ["ropes", "sleds", "wagons", "chains"]),
    (["push"], ["rafts", "barrels", "crates", "boxes"]),
    (["fill"], ["jars", "bowls", "cups", "buckets"]),
    (["open"], ["gates", "locks", "lids", "shutters"]),
    (["break"], ["glass", "plates", "bricks", "tiles"]),
    (["cook"], ["soup", "stew", "bread", "rice"]),
    (["sell"], ["cloth", "wool", "silk", "lace"]),
    (["buy"], ["tools", "nails", "pins", "needles"]),
    (["count"], ["ants", "bees", "geese", "swans"]),
    (["sweep"], ["floors", "stairs", "porches", "yards"]),
    (["wash"], ["dishes", "shirts", "socks", "windows"]),
    (["mend"], ["tents", "quilts", "sacks", "hammocks"]),
    (["blow"], ["whistles", "bubbles", "kisses", "candles"]),
    (["lick"], ["spoons", "paws", "stamps", "sweets"]),
    (["pick"], ["apples", "plums", "pears", "cherries"]),
    (["plant"], ["oaks", "pines", "elms", "firs"]),
    (["sail"], ["seas", "lakes", "bays", "gulfs"]),
    (["fly"], ["kites", "gliders", "balloons", "blimps"]),
    (["read"], ["books", "letters", "signs", "notes"]),
    (["draw"], ["circles", "squares", "lines", "arrows"]),
    (["wrap"], ["gifts", "parcels", "scarves", "blankets"]),
]
ALL_KINDS = KINDS + MORE_KINDS  # kinds 1 to 64; the first K are used with --kinds K
ALLOWED_KINDS = (4, 8, 16, 32, 64)
MAX_KINDS = 32  # the full vocabulary of the second experiment's starting table (see below)
NESTED_KINDS = 64  # the full vocabulary of the third experiment's "nested64" starting table
MAX_CREATURES = 1024

# The longest sentence has 13 tokens ("do a , b , c and d live in woods ? yes").
SENTENCE_LENGTH = 13

# ------------------------------------------------------------------------------------------------
# Creature names: copied from the first experiment (language.py), so that creature number i has
# the same name as animal number i there. The rule below must not be changed.
# ------------------------------------------------------------------------------------------------

BASE_NAMES = ["foxes", "bears", "wolves", "owls", "bats", "frogs", "snakes", "crows", "hens",
              "ducks", "deer", "mice", "toads", "robins", "otters", "lizards"]
# The words of the first experiment. A made-up name equal to one of these was skipped there,
# so it is skipped here too (this keeps the list identical).
FIRST_EXPERIMENT_WORDS = (
    [".", "?", "are", "do", "eat", "live", "in", "wake", "at", "yes", "no"]
    + ["winter", "spring", "summer", "fall"]
    + ["red", "brown", "green", "black", "meat", "plants", "insects", "seeds",
       "woods", "fields", "rivers", "caves", "dawn", "noon", "dusk", "night"]
)
NAME_CONSONANTS = "bdfgklmnprtvz"
NAME_VOWELS = "aeiou"
NAME_LAST_VOWELS = "aou"
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


def invented_name(number):
    """Candidate name number `number` for an invented creature (copied from the first
    experiment). Number 0 is "bakos"."""
    first = [c + v for c in NAME_CONSONANTS for v in NAME_VOWELS]
    last = [c + v for c in NAME_CONSONANTS for v in NAME_LAST_VOWELS]
    pairs = len(first) * len(last)
    more, pair = divmod(number, pairs)
    pair = (pair * 1597 + 13) % pairs  # a fixed scramble, so neighbours do not look alike
    name = first[pair // len(last)] + last[pair % len(last)] + "s"
    while more > 0:
        more, extra = divmod(more - 1, len(first))
        name = first[extra] + name
    return name


def creature_names(count=MAX_CREATURES):
    """The names of the first `count` creatures, the same list and order as the first
    experiment: the 16 base animals, then invented names."""
    names = list(BASE_NAMES)
    taken = set(FIRST_EXPERIMENT_WORDS + names)
    number = 0
    while len(names) < count:
        name = invented_name(number)
        number += 1
        if name in taken or any(piece in name for piece in NAME_SKIP):
            continue
        names.append(name)
    return names[:count]


def kind_words(kinds):
    """The words that the kinds 1 to `kinds` add, in token order: for each kind, its linking
    words that are not fixed words already, then its four values."""
    words = []
    for link, values in ALL_KINDS[:kinds]:
        words += [word for word in link if word not in FIXED_WORDS]
        words += values
    return words


# The full vocabulary (32 kinds, 1,024 creatures). The word table is drawn for it once, and
# every smaller vocabulary takes its starting rows from it (see Language.full_ids).
FULL_NON_CREATURE = 1 + len(FIXED_WORDS) + len(kind_words(MAX_KINDS))
FULL_TOKENS = FULL_NON_CREATURE + MAX_CREATURES
# The same for the third experiment's starting table, which is drawn for 64 kinds.
NESTED_NON_CREATURE = 1 + len(FIXED_WORDS) + len(kind_words(NESTED_KINDS))
NESTED_TOKENS = NESTED_NON_CREATURE + MAX_CREATURES

# Separate streams of random numbers, so that facts, held-back pairs, the training text and the
# exam can never use the same random numbers by accident. (13 is used by v2/net.py.)
FACT_STREAM = 11
HELD_STREAM = 12
TEXT_STREAM = 14
EXAM_STREAM = 15

# ------------------------------------------------------------------------------------------------
# How sentences are drawn (B.3)
# ------------------------------------------------------------------------------------------------

STATEMENT, NOT_STATEMENT, YES_QUESTION, NO_QUESTION = 0, 1, 2, 3
TYPE_NAMES = ["statement", "not statement", "yes question", "no question"]
TYPE_CHANCES = [0.4, 0.2, 0.2, 0.2]
LENGTH_CHANCES = [0.4, 0.2, 0.2, 0.2]  # list length 1, 2, 3, 4
MAX_TRIES = 50  # tries for a list without a held-back pair before the list is made shorter
# The 50 tries are made in three bulks: most sentences succeed in the first two tries, so only
# a few go on to the next bulks. The total is exactly MAX_TRIES.
TRY_BULKS = (2, 8, 40)
assert sum(TRY_BULKS) == MAX_TRIES
REDRAW_ROUNDS = 3  # see draw_different

# Exam sizes (B.5).
EXAM_QUESTIONS = 2000  # up to this many questions in S3, and in S4
EXAM_CANDIDATES = 30000  # random candidate lists drawn per list length and answer
EXAM_BATCH = 4096  # prompts handed to the model at once


class WeightedGroups:
    """Several groups of creatures, each with weights, from which one creature can be drawn
    per entry of an array of group numbers, all at once.

    How it works: the weights of each group are stacked one after another on a line, group g
    taking the stretch from g to g + 1. A random number u between 0 and 1 is put at g + u, and
    the creature whose stretch holds that point is the one drawn.
    """

    def __init__(self, members, weights):
        # members: list of arrays of creature numbers; weights: one weight per creature.
        self.members = np.concatenate(members)
        self.start = np.cumsum([0] + [len(m) for m in members])[:-1]
        self.end = self.start + np.array([len(m) for m in members])
        edges = []
        for group, creatures in enumerate(members):
            share = np.cumsum(weights[creatures])
            share = share / share[-1]
            share[-1] = 1.0  # exactly 1, whatever the rounding
            edges.append(group + share)
        self.edges = np.concatenate(edges)
        # The same groups as a table, one group per row, padded with -1 (weight 0).
        widest = max(len(m) for m in members)
        self.table = np.full((len(members), widest), -1, dtype=np.int64)
        self.table_weights = np.zeros((len(members), widest))
        for group, creatures in enumerate(members):
            self.table[group, :len(creatures)] = creatures
            self.table_weights[group, :len(creatures)] = weights[creatures]

    def draw_excluding(self, rng, groups, taken):
        """One creature (by weight) from each group in `groups` ([rows]), leaving out the
        creatures in the same row of `taken` ([rows, n])."""
        creatures = self.table[groups]  # [rows, widest]
        weights = self.table_weights[groups].copy()
        weights[(creatures[:, :, None] == taken[:, None, :]).any(axis=2)] = 0.0
        edges = np.cumsum(weights, axis=1)
        points = rng.random(len(groups)) * edges[:, -1]
        places = (edges <= points[:, None]).sum(axis=1)
        # Never land on a left-out or padding place through rounding: at most the last place
        # that has weight.
        last_with_weight = weights.shape[1] - 1 - np.argmax(weights[:, ::-1] > 0, axis=1)
        places = np.minimum(places, last_with_weight)
        return creatures[np.arange(len(groups)), places]

    def draw(self, rng, groups):
        """One creature (by weight) from each group named in `groups` (any shape)."""
        groups = np.asarray(groups)
        points = groups + rng.random(groups.shape)
        places = np.searchsorted(self.edges, points.ravel(), side="right").reshape(groups.shape)
        # Guard against the point rounding up onto the next group's stretch (extremely rare).
        places = np.minimum(places, self.end[groups] - 1)
        return self.members[places]


def draw_different(rng, groups, slots):
    """Draw one creature for every place of `slots` ([rows, k] group numbers), with no
    creature twice in a row.

    Every place is drawn, then each place is drawn again for as long as it repeats a creature
    of an earlier place. That is the same as drawing the creatures one after another, each
    from the ones not yet taken, with chances in proportion to the weights.

    In a small group with one very common creature, drawing again can take many rounds. So
    after REDRAW_ROUNDS rounds the few places still repeating are drawn directly from the
    creatures not yet taken, which gives exactly the same chances.
    """
    chosen = groups.draw(rng, slots)
    for place in range(1, slots.shape[1]):
        clash = (chosen[:, :place] == chosen[:, place:place + 1]).any(axis=1)
        for _ in range(REDRAW_ROUNDS):
            if not clash.any():
                break
            rows = np.nonzero(clash)[0]
            chosen[rows, place] = groups.draw(rng, slots[rows, place])
            clash[rows] = (chosen[rows, :place] == chosen[rows, place:place + 1]).any(axis=1)
        rows = np.nonzero(clash)[0]
        if rows.size:
            chosen[rows, place] = groups.draw_excluding(rng, slots[rows, place],
                                                        chosen[rows, :place])
    return chosen


def shuffle_rows(rng, table):
    """Put the entries of each row of `table` in a random order."""
    order = np.argsort(rng.random(table.shape), axis=1)
    return np.take_along_axis(table, order, axis=1)


class Language:
    """One language: N creatures, K kinds of fact, one kind of word use.

    Main attributes
      words          the word of every token id (every token is a different word)
      n_tokens       number of tokens (rows of the word table)
      creature_base  token id of creature 0; creature i is creature_base + i
      n_out          number of words the model can say: every token except <pad> and the
                     creatures. Output number j is the score of token id j + 1.
      link_ids       for each kind, the token ids of its linking phrase
      value_ids      [K, 4] token ids of the values
      facts          [N, K] the true value (0 to 3) of every creature and kind
      held           [N, N] True where the two creatures are a held-back pair
      weights        [N] how often each creature is used, summing to 1
      full_ids       for every token, its row in the full vocabulary (32 kinds, 1,024 creatures)
      rows_in_use    number of tokens a run can meet: <pad>, the other non-creature words and
                     the first N creatures (the other creature rows are never read)
    """

    def __init__(self, creatures, kinds=4, usage="equal"):
        if kinds not in ALLOWED_KINDS:
            raise ValueError(f"kinds must be one of {ALLOWED_KINDS}; got {kinds}.")
        if creatures % 4 != 0 or not 16 <= creatures <= MAX_CREATURES:
            raise ValueError(f"creatures must be a multiple of 4 from 16 to {MAX_CREATURES}; "
                             f"got {creatures}.")
        if usage not in ("equal", "zipf"):
            raise ValueError(f'usage must be "equal" or "zipf"; got "{usage}".')
        self.creatures = creatures
        self.kinds = kinds
        self.usage = usage

        # --- words and token ids (B.1) ---
        self.names = creature_names(MAX_CREATURES)
        non_creature = [PAD] + FIXED_WORDS + kind_words(kinds)
        self.words = non_creature + self.names  # always all 1,024 creature rows
        self.n_tokens = len(self.words)
        self.creature_base = len(non_creature)
        self.n_out = self.creature_base - 1
        self.rows_in_use = self.creature_base + creatures
        if len(set(self.words)) != len(self.words):
            raise RuntimeError("two tokens are spelled the same; every token must differ")
        # Token id of every non-creature word (these are all spelled differently).
        self.word_id = {word: token for token, word in enumerate(non_creature)}
        self.link_ids = [[self.word_id[w] for w in link] for link, _ in ALL_KINDS[:kinds]]
        self.value_ids = np.array([[self.word_id[w] for w in values]
                                   for _, values in ALL_KINDS[:kinds]])
        self.yes_id = self.word_id["yes"]
        self.no_id = self.word_id["no"]
        # Rows of the full table: the non-creature words come first in the same order in every
        # vocabulary, so only the creatures move.
        # (With 64 kinds there is no such row for the extra words: the second experiment's
        # table has only 32 kinds. Such a language must use the "nested64" table.)
        if kinds <= MAX_KINDS:
            self.full_ids = np.concatenate([np.arange(self.creature_base),
                                            FULL_NON_CREATURE + np.arange(MAX_CREATURES)])
        else:
            self.full_ids = None
        # Rows of the third experiment's nested table (64 kinds, 1,024 creatures), in the same
        # way.
        self.nested_ids = np.concatenate([np.arange(self.creature_base),
                                          NESTED_NON_CREATURE + np.arange(MAX_CREATURES)])

        self.facts = make_facts(creatures, kinds)
        self.held_pairs = make_held_pairs(creatures)
        self.held = np.zeros((creatures, creatures), dtype=bool)
        self.held[self.held_pairs[:, 0], self.held_pairs[:, 1]] = True
        self.held[self.held_pairs[:, 1], self.held_pairs[:, 0]] = True
        self.weights = creature_weights(creatures, usage)

        self.groups = self._make_groups(self.weights)
        self._exam = None

    # --------------------------------------------------------------------------------------------
    # Groups of creatures to draw from
    # --------------------------------------------------------------------------------------------

    def _make_groups(self, weights):
        """Groups of creatures: those that have each (kind, value), those that do not, and all.

        Group numbers: have(kind, value) = 4 * kind + value; not-have = 4 * K + that; all = 8 * K.
        """
        have, not_have = [], []
        for kind in range(self.kinds):
            for value in range(4):
                has_it = self.facts[:, kind] == value
                have.append(np.nonzero(has_it)[0])
                not_have.append(np.nonzero(~has_it)[0])
        everyone = [np.arange(self.creatures)]
        return WeightedGroups(have + not_have + everyone, weights)

    def group_have(self, kind, value):
        return 4 * kind + value

    def group_not_have(self, kind, value):
        return 4 * self.kinds + 4 * kind + value

    def group_all(self):
        return 8 * self.kinds

    def has_held_pair(self, lists):
        """For lists [rows, k] of creatures: True where some two of them are a held-back pair."""
        found = np.zeros(lists.shape[0], dtype=bool)
        for a in range(lists.shape[1]):
            for b in range(a + 1, lists.shape[1]):
                found |= self.held[lists[:, a], lists[:, b]]
        return found

    # --------------------------------------------------------------------------------------------
    # Turning drawn sentences into tokens
    # --------------------------------------------------------------------------------------------

    def tokens_of(self, types, kinds, lists, values):
        """Token rows [rows, 13] for sentences given by their parts.

        types   sentence type (STATEMENT, NOT_STATEMENT, YES_QUESTION, NO_QUESTION)
        kinds   kind of fact (0 is the first kind, "are")
        lists   [rows, 4] creature numbers, -1 where the list is shorter
        values  value number (0 to 3)

        Every sentence is first laid out in 16 fixed places, some of them empty, and the empty
        places are then squeezed out:
          question word | a , b , c and d | do not | link link | not | value | . or ? | answer
        """
        rows = len(types)
        k = (lists >= 0).sum(axis=1)
        first_kind = kinds == 0
        question = (types == YES_QUESTION) | (types == NO_QUESTION)
        negated = types == NOT_STATEMENT
        link = np.full((self.kinds, 2), -1)
        for kind, ids in enumerate(self.link_ids):
            link[kind, :len(ids)] = ids
        empty = -1
        word = self.word_id

        places = np.full((rows, 16), empty, dtype=np.int64)
        places[:, 0] = np.where(question, np.where(first_kind, word["are"], word["do"]), empty)
        for i in range(4):  # the list: creature, separator, creature, ...
            places[:, 1 + 2 * i] = np.where(lists[:, i] >= 0, self.creature_base + lists[:, i],
                                            empty)
            if i < 3:
                # After creature i comes "and" if creature i + 1 is the last one, "," if more
                # follow, nothing if creature i is the last.
                separator = np.where(k == i + 2, word["and"],
                                     np.where(k > i + 2, word[","], empty))
                places[:, 2 + 2 * i] = separator
        places[:, 8] = np.where(negated & ~first_kind, word["do"], empty)
        places[:, 9] = np.where(negated & ~first_kind, word["not"], empty)
        say_link = ~(question & first_kind)  # "are LIST VAL ?" has its linking word in front
        places[:, 10] = np.where(say_link, link[kinds, 0], empty)
        places[:, 11] = np.where(say_link, link[kinds, 1], empty)
        places[:, 12] = np.where(negated & first_kind, word["not"], empty)
        places[:, 13] = self.value_ids[kinds, values]
        places[:, 14] = np.where(question, word["?"], word["."])
        places[:, 15] = np.where(types == YES_QUESTION, word["yes"],
                                 np.where(types == NO_QUESTION, word["no"], empty))

        # Squeeze out the empty places, keeping the order of the others.
        order = np.argsort(places == empty, axis=1, kind="stable")
        packed = np.take_along_axis(places, order, axis=1)
        if (packed[:, SENTENCE_LENGTH:] != empty).any():
            raise RuntimeError("a sentence came out longer than 13 tokens")
        packed = packed[:, :SENTENCE_LENGTH]
        packed[packed == empty] = PAD_ID
        return packed

    # --------------------------------------------------------------------------------------------
    # Drawing training sentences (B.3)
    # --------------------------------------------------------------------------------------------

    def draw_sentences(self, rng, size):
        """Draw `size` true training sentences. Returns a dictionary of arrays:
        tokens [size, 13], types, kinds, values, lists [size, 4] (-1 = no creature), lengths."""
        types = rng.choice(4, size=size, p=TYPE_CHANCES)
        lengths = 1 + rng.choice(4, size=size, p=LENGTH_CHANCES)
        kinds = rng.integers(self.kinds, size=size)
        values = rng.integers(4, size=size)  # for "not" statements it is drawn later instead
        # For "no" questions: how many creatures of the list do NOT have the value.
        wrong = rng.integers(1, lengths + 1)
        lists = np.full((size, 4), -1, dtype=np.int64)

        bulk = np.zeros(size, dtype=np.int64)  # which bulk of tries each sentence is at
        todo = np.arange(size)
        while todo.size:
            failed = []
            # Sentences with the same list length and bulk are tried together.
            for k in range(1, 5):
                for stage in range(len(TRY_BULKS)):
                    rows = todo[(lengths[todo] == k) & (bulk[todo] == stage)]
                    if rows.size == 0:
                        continue
                    ok = self._try_lists(rng, rows, k, TRY_BULKS[stage], types, kinds, values,
                                         wrong, lists)
                    failed.append(rows[~ok])
            todo = np.sort(np.concatenate(failed)) if failed else np.zeros(0, dtype=np.int64)
            # A sentence that used up all its tries gets a list one shorter, and starts afresh.
            bulk[todo] += 1
            out_of_tries = todo[bulk[todo] == len(TRY_BULKS)]
            lengths[out_of_tries] -= 1
            bulk[out_of_tries] = 0
            no_rows = out_of_tries[types[out_of_tries] == NO_QUESTION]
            wrong[no_rows] = rng.integers(1, lengths[no_rows] + 1)

        tokens = self.tokens_of(types, kinds, lists, values)
        return {"tokens": tokens, "types": types, "kinds": kinds, "values": values,
                "lists": lists, "lengths": lengths}

    def _try_lists(self, rng, rows, k, tries, types, kinds, values, wrong, lists):
        """Make `tries` attempts at a list of k creatures for each sentence in `rows`. The first
        good attempt is written into `lists` (and, for "not" statements, `values`). Returns a
        True/False array: which sentences got a list."""
        # Which group each place of the list is drawn from.
        slots = np.empty((rows.size, k), dtype=np.int64)
        kind, value, kind_of_type = kinds[rows], values[rows], types[rows]
        for place in range(k):
            have = self.group_have(kind, value)
            not_have = self.group_not_have(kind, value)
            # "no" questions: the last `wrong` places are creatures without the value.
            no_place = (kind_of_type == NO_QUESTION) & (place >= k - wrong[rows])
            slots[:, place] = np.where(no_place, not_have, have)
            slots[kind_of_type == NOT_STATEMENT, place] = self.group_all()
        # All attempts at once: [rows, tries, k].
        attempts = draw_different(rng, self.groups, np.repeat(slots, tries, axis=0))
        good = ~self.has_held_pair(attempts)
        # "not" statements also need a value that none of the creatures has.
        is_not = np.repeat(kind_of_type == NOT_STATEMENT, tries)
        their_values = self.facts[attempts, np.repeat(kind, tries)[:, None]]  # [rows*tries, k]
        free = ~(their_values[:, :, None] == np.arange(4)).any(axis=1)  # [rows*tries, 4]
        good &= ~is_not | free.any(axis=1)

        good = good.reshape(rows.size, tries)
        found = good.any(axis=1)
        first = np.argmax(good, axis=1)  # the first good attempt
        pick = np.arange(rows.size) * tries + first
        chosen = shuffle_rows(rng, attempts[pick])  # every list in a random order
        lists[rows[found], :k] = chosen[found]
        # The value of a "not" statement: drawn with equal chance from the free values.
        new_not = found & (kind_of_type == NOT_STATEMENT)
        if new_not.any():
            score = rng.random((rows.size, 4))
            score[~free[pick]] = -1.0
            values[rows[new_not]] = np.argmax(score, axis=1)[new_not]
        return found

    def sample_batch(self, seed, step, size=512):
        """The training batch of step `step` of a run with seed `seed`: [size, 13] token ids.
        It depends only on (seed, step), so a run can be picked up again exactly."""
        rng = np.random.default_rng([TEXT_STREAM, seed, step])
        return self.draw_sentences(rng, size)["tokens"]

    # --------------------------------------------------------------------------------------------
    # The exam (B.5)
    # --------------------------------------------------------------------------------------------

    def exam(self):
        """The five fixed exam sets (built once, then kept)."""
        if self._exam is None:
            self._exam = build_exam(self)
        return self._exam

    def decode(self, ids):
        """Token ids as text, without the padding."""
        return " ".join(self.words[i] for i in ids if i != PAD_ID)


# ------------------------------------------------------------------------------------------------
# Facts, held-back pairs, weights (B.2, B.3, B.4)
# ------------------------------------------------------------------------------------------------


def make_facts(creatures, kinds):
    """[creatures, kinds]: the true value of each creature for each kind. For each kind exactly a
    quarter of the creatures have each value. The shuffle of kind number j depends only on
    (creatures, j), so the first kinds do not change when more kinds are used."""
    facts = np.zeros((creatures, kinds), dtype=np.int64)
    deck = np.arange(creatures) % 4
    for kind in range(kinds):
        rng = np.random.default_rng([FACT_STREAM, creatures, kind])
        facts[:, kind] = rng.permutation(deck)
    return facts


def make_held_pairs(creatures):
    """[pairs, 2]: one tenth of all pairs of creatures (rounded down, at least 1), first < second.
    The choice depends only on the number of creatures."""
    first, second = np.triu_indices(creatures, k=1)
    count = max(1, len(first) // 10)
    rng = np.random.default_rng([HELD_STREAM, creatures])
    chosen = np.sort(rng.choice(len(first), size=count, replace=False))
    return np.stack([first[chosen], second[chosen]], axis=1)


def creature_weights(creatures, usage):
    """How often each creature is used; sums to 1. "zipf" is the rule of the first experiment:
    creature 2 half as often as creature 1, creature 3 a third as often, and so on."""
    if usage == "equal":
        return np.full(creatures, 1.0 / creatures)
    weights = 1.0 / np.arange(1, creatures + 1)
    return weights / weights.sum()


# ------------------------------------------------------------------------------------------------
# A separate, simple truth check (used by the tests)
# ------------------------------------------------------------------------------------------------


def read_sentence(lang, ids):
    """Read one sentence word by word, without using the sentence maker above.

    Returns a dictionary (type, creatures, kind, value, answer), or None if the tokens are not
    a correct sentence of the language."""
    ids = [int(i) for i in ids]
    while ids and ids[-1] == PAD_ID:
        ids.pop()
    if PAD_ID in ids or len(ids) < 3:
        return None
    word = [lang.words[i] for i in ids]
    is_creature = [i >= lang.creature_base for i in ids]

    def read_list(start):
        """The list starting at `start`: (creatures, place after it), or None."""
        found, place = [], start
        while True:
            if place >= len(ids) or not is_creature[place]:
                return None
            found.append(ids[place] - lang.creature_base)
            place += 1
            if place < len(ids) and not is_creature[place] and word[place] in (",", "and"):
                separator = word[place]
                place += 1
                if separator == "and":
                    if place >= len(ids) or not is_creature[place]:
                        return None
                    found.append(ids[place] - lang.creature_base)
                    place += 1
                    break
            else:
                break
        # Only "a", "a and b", "a , b and c", "a , b , c and d" are right.
        expected = []
        for i in range(len(found)):
            expected.append("creature")
            if i < len(found) - 2:
                expected.append(",")
            elif i == len(found) - 2:
                expected.append("and")
        got = ["creature" if is_creature[p] else word[p] for p in range(start, place)]
        if got != expected or len(found) > 4 or len(set(found)) != len(found):
            return None
        return found, place

    def kind_of_value(token):
        """(kind, value) of a value token, or None."""
        for kind in range(lang.kinds):
            for value in range(4):
                if lang.value_ids[kind, value] == token:
                    return kind, value
        return None

    def link_words(kind):
        return ALL_KINDS[kind][0]

    if word[-1] in ("yes", "no"):
        if word[-2] != "?":
            return None
        answer = word[-1]
        middle = ids[:-2]
        found = kind_of_value(middle[-1]) if middle else None
        if found is None:
            return None
        kind, value = found
        opener = "are" if kind == 0 else "do"
        if word[0] != opener:
            return None
        read = read_list(1)
        if read is None:
            return None
        creatures, place = read
        rest = word[place:len(middle) - 1]
        if rest != ([] if kind == 0 else link_words(kind)):
            return None
        kind_of_sentence = YES_QUESTION if answer == "yes" else NO_QUESTION
        return {"type": kind_of_sentence, "creatures": creatures, "kind": kind, "value": value,
                "answer": answer}

    if word[-1] != ".":
        return None
    found = kind_of_value(ids[-2])
    if found is None:
        return None
    kind, value = found
    read = read_list(0)
    if read is None:
        return None
    creatures, place = read
    rest = word[place:len(ids) - 2]
    if kind == 0:
        plain, negated = ["are"], ["are", "not"]
    else:
        plain, negated = link_words(kind), ["do", "not"] + link_words(kind)
    if rest == plain:
        kind_of_sentence = STATEMENT
    elif rest == negated:
        kind_of_sentence = NOT_STATEMENT
    else:
        return None
    return {"type": kind_of_sentence, "creatures": creatures, "kind": kind, "value": value,
            "answer": None}


def is_true(lang, ids):
    """True when the tokens are a correct sentence of the language that is true (or a question
    with the right answer)."""
    read = read_sentence(lang, ids)
    if read is None:
        return False
    has = [lang.facts[c, read["kind"]] == read["value"] for c in read["creatures"]]
    if read["type"] == STATEMENT:
        return all(has)
    if read["type"] == NOT_STATEMENT:
        return not any(has)
    right = "yes" if all(has) else "no"
    return read["answer"] == right


# ------------------------------------------------------------------------------------------------
# The exam (B.5)
# ------------------------------------------------------------------------------------------------
#
# Every exam set is a dictionary with
#   prompts  [rows, length] token ids, padded with <pad> at the end
#   last     [rows] the place of the last word of each prompt: the score is read there
#   options  [rows, n] the token ids among which the model has to choose
#   right    [rows] which option is right (for S5: which option should score LOWEST)
# Question sets also have answer ("yes" = 1, "no" = 0), wrong (the f of B.3, 0 for "yes"),
# kinds, values, lists and sentences (the whole question with its answer, for the tests).


def pad_rows(rows):
    """A list of token lists -> (array padded with <pad>, place of each row's last token)."""
    width = max(len(row) for row in rows)
    table = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    for i, row in enumerate(rows):
        table[i, :len(row)] = row
    return table, np.array([len(row) - 1 for row in rows])


def fact_prompts(lang, negated):
    """S1 ("c are", "c live in") or, with negated, S5 ("c are not", "c do not live in")."""
    rows, options, right, kinds = [], [], [], []
    for creature in range(lang.creatures):
        for kind in range(lang.kinds):
            c = lang.creature_base + creature
            if not negated:
                rows.append([c] + lang.link_ids[kind])
            elif kind == 0:
                rows.append([c, lang.word_id["are"], lang.word_id["not"]])
            else:
                rows.append([c, lang.word_id["do"], lang.word_id["not"]] + lang.link_ids[kind])
            options.append(lang.value_ids[kind])
            right.append(lang.facts[creature, kind])
            kinds.append(kind)
    prompts, last = pad_rows(rows)
    return {"prompts": prompts, "last": last, "options": np.array(options),
            "right": np.array(right), "kinds": np.array(kinds)}


def question_set(lang, types, kinds, values, lists, wrong):
    """An exam set of questions from their parts. The prompt is the question without its
    answer; the score is read at the "?"."""
    sentences = lang.tokens_of(types, kinds, lists, values)
    lengths = (sentences != PAD_ID).sum(axis=1)
    prompts = sentences.copy()
    prompts[np.arange(len(prompts)), lengths - 1] = PAD_ID  # remove the answer
    width = int(lengths.max()) - 1
    answer = (types == YES_QUESTION).astype(np.int64)
    options = np.tile([lang.yes_id, lang.no_id], (len(types), 1))
    return {"prompts": prompts[:, :width], "last": lengths - 2, "options": options,
            "right": 1 - answer, "answer": answer, "wrong": wrong, "kinds": kinds,
            "values": values, "lists": lists, "sentences": sentences}


def one_creature_questions(lang):
    """S2: for every creature and kind, the "yes" question on the true value and the "no"
    questions on the three other values."""
    n = lang.creatures * lang.kinds * 4
    creature = np.repeat(np.arange(lang.creatures), lang.kinds * 4)
    kinds = np.tile(np.repeat(np.arange(lang.kinds), 4), lang.creatures)
    values = np.tile(np.arange(4), lang.creatures * lang.kinds)
    true = lang.facts[creature, kinds] == values
    types = np.where(true, YES_QUESTION, NO_QUESTION)
    lists = np.full((n, 4), -1, dtype=np.int64)
    lists[:, 0] = creature
    return question_set(lang, types, kinds, values, lists, np.where(true, 0, 1))


def spread_evenly(total, available):
    """Split `total` over the places of `available` as evenly as possible, never giving a place
    more than it has. Returns the counts."""
    counts = [0] * len(available)
    left = total
    while left > 0:
        open_places = [i for i in range(len(available)) if counts[i] < available[i]]
        if not open_places:
            break
        share = max(1, left // len(open_places))
        for i in open_places:
            give = min(share, available[i] - counts[i], left)
            counts[i] += give
            left -= give
            if left == 0:
                break
    return counts


def list_questions(lang):
    """S3 and S4: questions about lists of 2, 3 or 4 creatures, with (S3) and without (S4) a
    held-back pair. Creatures are drawn with equal weights. Returns (S3, S4)."""
    rng = np.random.default_rng([EXAM_STREAM, lang.creatures, lang.kinds])
    equal = lang._make_groups(np.full(lang.creatures, 1.0 / lang.creatures))
    # found[held][answer][k] = list of questions (kind, value, list, wrong), no repeats.
    found = {held: {answer: {} for answer in (1, 0)} for held in (True, False)}
    for k in (2, 3, 4):
        for answer in (1, 0):
            m = EXAM_CANDIDATES
            kinds = rng.integers(lang.kinds, size=m)
            values = rng.integers(4, size=m)
            wrong = rng.integers(1, k + 1, size=m) if answer == 0 else np.zeros(m, dtype=np.int64)
            slots = np.empty((m, k), dtype=np.int64)
            for place in range(k):
                slots[:, place] = np.where(place >= k - wrong,
                                           lang.group_not_have(kinds, values),
                                           lang.group_have(kinds, values))
            lists = shuffle_rows(rng, draw_different(rng, equal, slots))
            held = lang.has_held_pair(lists)
            for with_held in (True, False):
                seen, kept = set(), []
                for i in np.nonzero(held == with_held)[0]:
                    key = (int(kinds[i]), int(values[i]), tuple(sorted(lists[i].tolist())))
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append((kinds[i], values[i], lists[i], wrong[i]))
                    if len(kept) == EXAM_QUESTIONS // 2:
                        break
                found[with_held][answer][k] = kept

    sets = []
    for with_held in (True, False):
        # Half "yes", half "no", as far as possible; list lengths in equal shares.
        available = {a: [len(found[with_held][a][k]) for k in (2, 3, 4)] for a in (1, 0)}
        per_answer = min(EXAM_QUESTIONS // 2, sum(available[1]), sum(available[0]))
        chosen = []
        for answer in (1, 0):
            counts = spread_evenly(per_answer, available[answer])
            for k, count in zip((2, 3, 4), counts):
                chosen += [(answer,) + q for q in found[with_held][answer][k][:count]]
        n = len(chosen)
        lists = np.full((n, 4), -1, dtype=np.int64)
        for i, q in enumerate(chosen):
            lists[i, :len(q[3])] = q[3]
        answers = np.array([q[0] for q in chosen])
        types = np.where(answers == 1, YES_QUESTION, NO_QUESTION)
        sets.append(question_set(lang, types, np.array([q[1] for q in chosen]),
                                 np.array([q[2] for q in chosen]), lists,
                                 np.array([q[4] for q in chosen])))
    return sets[0], sets[1]


def build_exam(lang):
    """All five exam sets. They depend only on the creatures and kinds, never on a run."""
    new_lists, seen_lists = list_questions(lang)
    return {"facts": fact_prompts(lang, negated=False),
            "questions_one": one_creature_questions(lang),
            "questions_new": new_lists,
            "questions_seen": seen_lists,
            "not_known": fact_prompts(lang, negated=True)}


def chosen_options(predict, exam_set, lowest=False):
    """Which option the predictor picks for every prompt (in batches).

    predict(prompts, last) must return scores [rows, n_out]: one score per word the model can
    say, where column j is token id j + 1. A tie goes to the first option."""
    picks = []
    for start in range(0, len(exam_set["prompts"]), EXAM_BATCH):
        part = slice(start, start + EXAM_BATCH)
        scores = predict(exam_set["prompts"][part], exam_set["last"][part])
        option_scores = np.take_along_axis(scores, exam_set["options"][part] - 1, axis=1)
        picks.append(np.argmin(option_scores, axis=1) if lowest
                     else np.argmax(option_scores, axis=1))
    return np.concatenate(picks)


def percent(right):
    return float(100.0 * right.mean()) if right.size else 0.0


def question_score(predict, exam_set):
    """% of "yes" right + % of "no" right - 100, not below 0; and the details."""
    return score_of_picks(chosen_options(predict, exam_set), exam_set)


def score_of_picks(picks, exam_set):
    """question_score from the options already chosen (picks: one per question)."""
    right = picks == exam_set["right"]
    yes = exam_set["answer"] == 1
    no_one = (exam_set["answer"] == 0) & (exam_set["wrong"] == 1)
    no_more = (exam_set["answer"] == 0) & (exam_set["wrong"] > 1)
    score = max(0.0, percent(right[yes]) + percent(right[~yes]) - 100.0)
    detail = {"yes": percent(right[yes]), "no_one": percent(right[no_one]),
              "no_more": percent(right[no_more]), "count": int(len(right))}
    return score, detail


# The exam sets, and for each whether the LOWEST scoring option is the one chosen.
EXAM_SETS = {"facts": False, "questions_one": False, "questions_new": False,
             "questions_seen": False, "not_known": True}


def grade(lang, predict):
    """All exam scores (0 to 100) for one predictor. Returns a dictionary."""
    exam = lang.exam()
    picks = {name: chosen_options(predict, exam[name], lowest)
             for name, lowest in EXAM_SETS.items()}
    return grade_picks(lang, picks)


def grade_picks(lang, picks):
    """All exam scores from the options already chosen in every exam set ({set name: picks}),
    as grade computes them. (The group trainer chooses them for many models at once.)"""
    exam = lang.exam()
    facts = exam["facts"]
    right = picks["facts"] == facts["right"]
    by_kind = [percent(right[facts["kinds"] == kind]) for kind in range(lang.kinds)]
    one, _ = score_of_picks(picks["questions_one"], exam["questions_one"])
    new, new_detail = score_of_picks(picks["questions_new"], exam["questions_new"])
    seen, seen_detail = score_of_picks(picks["questions_seen"], exam["questions_seen"])
    negated = exam["not_known"]
    not_right = picks["not_known"] == negated["right"]
    return {"facts": percent(right), "facts_by_kind": by_kind, "questions_one": one,
            "questions_new": new, "questions_new_detail": new_detail,
            "questions_seen": seen, "questions_seen_detail": seen_detail,
            "not_known": percent(not_right)}

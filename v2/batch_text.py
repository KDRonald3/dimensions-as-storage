"""
v2/batch_text.py - the training sentences of many models at once, made directly as tensors on
the device (PLAN.md, parts B.3 and B.4).

v2/lang.py has the reference sentence maker (NumPy, one model at a time). This file makes the
same KIND of sentences, with the same chances, for every member of a group in one go:

    maker = BatchText(languages, seeds, batch=512, device="cuda")
    tokens = maker.make(step)      # [members, 512, 13] token ids on the device

The sentences are not the same ones the NumPy maker would give (the random numbers are drawn
differently), but every choice is made with the same chances, so the training text has the same
distribution. The tests in v2/tests/test_v2_batch.py compare the two makers.

The random numbers of a member depend only on (its seed, the step), as PLAN.md B.7 asks: never
on the other members of its group. So a run gets the same sentences whichever runs share its
group (see "Random numbers" below).

How a sentence is drawn (the same steps as B.3, and as in v2/lang.py)
----------------------------------------------------------------------
  1. type, list length k, kind and value, each with the chances of B.3;
     for "no" questions also f, the number of creatures in the list that do NOT have the value.
  2. the creatures:
       statement and "yes" question  k different creatures that have the value
       "no" question                 k - f that have it and f that do not
       "not" statement               k different creatures of all, then a value none of them
                                     has (if there is none, the creatures are drawn again)
     always "by use", that is with the creature weights of B.4.
  3. a list with a held-back pair is drawn again. After 50 tries the list is made one shorter
     (and f is drawn again for a "no" question), as in v2/lang.py.
  4. the list is put in random order and the sentence is laid out as tokens.

Drawing different creatures "by use"
------------------------------------
The creatures of every group (those that have a value, those that do not, everyone) are laid
out on a line, as in v2/lang.py (WeightedGroups): group g takes the stretch from g to g + 1, and
each creature in it a piece as long as its share of the group's weight. A random point on the
stretch picks the creature whose piece holds it (a "lookup in the running totals").

A list must not name a creature twice. The second creature is drawn from the group WITHOUT the
first one: the point is drawn on a stretch shortened by the first creature's piece, and then
moved past that piece if it lies at or beyond it. With two or three creatures taken already,
the point is moved past their pieces one after another, from the left. This gives exactly the
chances of drawing one creature after another, each from those not yet taken, in proportion to
the weights (as in v2/lang.py), with one random number per creature and no drawing again.

A "no" question takes k - f creatures from those that have the value and f from those that do
not. The two groups have no creature in common, so a creature only has to be left out of the
group it was drawn from.

The members of a group of models can have different numbers of creatures. Every language has
its own groups on the same line (language number m starts at m x (8 x kinds + 1)).

Many steps at once
------------------
make_steps(steps) makes the sentences of several steps in one pass (the trainer makes 16 at a
time). Each sentence still depends only on (its seed, its step, its language), so a step's
sentences are the same whether it is made alone or with others.

Random numbers
--------------
One shared random generator would give a member different numbers depending on which other
members share its group. Instead every random number is worked out from a label: (the member's
seed and the step, what the number is for, the round of tries, the sentence, the try, the
creature). The label is scrambled by a fixed mixing rule (`mix`), which turns it into a number
that looks random, and the same label always gives the same number. So a member's numbers do
not depend on the other members, nor on how the work is cut into pieces.
"""

import math

import numpy as np
import torch

from v2.lang import (LENGTH_CHANCES, MAX_TRIES, NO_QUESTION, NOT_STATEMENT, PAD_ID,
                     SENTENCE_LENGTH, STATEMENT, TYPE_CHANCES, YES_QUESTION)

# Tags the random numbers of this sentence maker, so they never coincide with another stream
# (v2/lang.py uses 11 to 15).
BATCH_TEXT_STREAM = 16

# The 50 tries for a list are made in three bulks, as in v2/lang.py; the total is exactly 50.
# How the tries are split does not change the chances: each try is a fresh draw and the first
# good one is kept. The first bulk (every sentence takes part) has 4 tries, so that almost every
# sentence is done after it, and the processor rarely has to wait for the card to see which
# sentences are left.
TRY_BULKS = (4, 6, 40)
assert sum(TRY_BULKS) == MAX_TRIES

# What a random number is for (part of its label; each use has its own number).
FOR_TYPE, FOR_LENGTH, FOR_KIND, FOR_VALUE, FOR_F, FOR_CREATURES, FOR_ORDER, FOR_NOT_VALUE, \
    FOR_F_AGAIN = range(9)
MAX_TRIES_IN_BULK = 64  # room for the try number in a label (a bulk has at most 40 tries)
LOW_32 = 0xFFFFFFFF


def times_low_32(x, constant):
    """The last 32 bits of x * constant, for whole numbers x from 0 to 2^32 - 1 (int64). The
    constant is split into two halves of 16 bits, so no number on the way gets bigger than about
    2^49 and the int64 sums never overflow (on any device)."""
    low, high = constant & 0xFFFF, constant >> 16
    return (x * low + (((x * high) & 0xFFFF) << 16)) & LOW_32


def mix(x):
    """Scramble 32-bit whole numbers (int64 tensors holding 0 to 2^32 - 1): every input gives a
    different output, and inputs that differ in one bit give outputs that look unrelated. (The
    "lowbias32" rule of C. Wellons: shifts, exclusive-or and two multiplications.)"""
    x = x ^ (x >> 16)
    x = times_low_32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = times_low_32(x, 0x846CA68B)
    return x ^ (x >> 16)


def member_keys(seed, step):
    """Two 32-bit numbers that depend only on (seed, step): the key of one member's random
    numbers at that step. NumPy's SeedSequence scrambles the three numbers well."""
    return [int(v) for v in
            np.random.SeedSequence([BATCH_TEXT_STREAM, seed, step]).generate_state(2)]


#: The line of all groups of all members (see "Drawing different creatures" above) is kept on a
#: grid of steps of 2^-36. Every number on it is then held exactly in float64 (the line is
#: shorter than 2^16), so adding, subtracting and comparing give exactly the same result
#: wherever a member's groups lie on the line. Without the grid, a member placed further along
#: the line (in a larger group of models) would be rounded more coarsely, and once in a great
#: while a point next to the end of a piece would pick the neighbouring creature: a run's text
#: would then depend, very rarely, on the other members of its group.
GRID = 2.0 ** 36


def on_grid(x):
    """NumPy numbers from 0 to 2^16, rounded to the nearest step of the grid."""
    return np.round(np.asarray(x, dtype=np.float64) * GRID) / GRID


def on_grid_floor(x):
    """Torch float64 numbers from 0 to 1, rounded down to a step of the grid."""
    return torch.floor(x * GRID) / GRID


class BatchText:
    """The sentence maker of one group. `languages` is one v2.lang.Language per member; they
    must all have the same kinds (so the same words and token ids). `seeds` is the seed of
    every member: with the step, it sets that member's sentences."""

    def __init__(self, languages, seeds, batch=512, device="cpu"):
        kinds = {lang.kinds for lang in languages}
        if len(kinds) != 1:
            raise ValueError("all members of a group must have the same kinds")
        if len(seeds) != len(languages):
            raise ValueError("give one seed per member")
        self.seeds = [int(seed) for seed in seeds]
        self.device = torch.device(device)
        self.batch = batch
        self.members = len(languages)
        if batch * MAX_TRIES_IN_BULK >= 1 << 32:
            raise ValueError("batch too large for the labels of the random numbers")
        first = languages[0]
        self.kinds = first.kinds
        self.creature_base = first.creature_base
        widest = max(lang.creatures for lang in languages)
        self.widest = widest
        put = lambda array: torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

        # Per member, padded to the widest member:
        #   facts   [members, widest, kinds] the value of each creature (-1 where not in use)
        #   held    [members, widest, widest] True for a held-back pair
        facts = np.full((self.members, widest, self.kinds), -1, dtype=np.int8)
        held = np.zeros((self.members, widest, widest), dtype=bool)
        for m, lang in enumerate(languages):
            facts[m, :lang.creatures] = lang.facts
            held[m, :lang.creatures, :lang.creatures] = lang.held
        self.facts = put(facts)
        self.held = put(held)

        # The line of all groups of all members (see "Drawing different creatures" above), from
        # the groups of v2/lang.py. Group g of member m is number m * per_member + g. For every
        # piece of the line: its creature, where it starts (low) and ends (high); for every
        # group: its first and last piece.
        self.per_member = 8 * self.kinds + 1
        creature, low, high, first_piece, last_piece = [], [], [], [], []
        pieces = 0
        if len(languages) * self.per_member >= 1 << 16:
            raise ValueError("too many members for exact sums on the line (see on_grid)")
        for m, lang in enumerate(languages):
            groups = lang.groups
            offset = m * self.per_member
            for g in range(self.per_member):
                begin, end = groups.start[g], groups.end[g]
                # g + share, put on the grid (see on_grid), then shifted to member m.
                edges = on_grid(groups.edges[begin:end]) + offset
                starts = np.concatenate([[g + offset], edges[:-1]])
                creature.append(groups.members[begin:end])
                low.append(starts)
                high.append(edges)
                first_piece.append(pieces)
                last_piece.append(pieces + end - begin - 1)
                pieces += end - begin
        self.piece_creature = put(np.concatenate(creature).astype(np.int64))
        self.piece_low = put(np.concatenate(low).astype(np.float64))
        self.piece_high = put(np.concatenate(high).astype(np.float64))
        self.first_piece = put(np.array(first_piece, dtype=np.int64))
        self.last_piece = put(np.array(last_piece, dtype=np.int64))

        # The words, the same for every member.
        self.value_ids = put(first.value_ids.astype(np.int64))  # [kinds, 4]
        link = np.full((self.kinds, 2), -1, dtype=np.int64)
        for kind, ids in enumerate(first.link_ids):
            link[kind, :len(ids)] = ids
        self.link_ids = put(link)
        self.word = first.word_id
        self.type_edges = torch.tensor(np.cumsum(TYPE_CHANCES)[:-1], dtype=torch.float32,
                                       device=self.device)
        self.length_edges = torch.tensor(np.cumsum(LENGTH_CHANCES)[:-1], dtype=torch.float32,
                                         device=self.device)

    # --------------------------------------------------------------------------------------------

    def make(self, step):
        """The batch of step `step`: [members, batch, 13] token ids on the device. Member i's
        sentences depend only on (seeds[i], step) and its language."""
        return self.make_steps([step])[0]

    def make_steps(self, steps):
        """The batches of several steps: [len(steps), members, batch, 13]. The same as calling
        make for each step, but in one pass (much faster on a graphics card)."""
        tokens = self.draw_steps(steps)["tokens"]
        return tokens.view(len(steps), self.members, self.batch, SENTENCE_LENGTH)

    def draw(self, step):
        """The sentences of one step, with their parts (all flat over members x batch):
        tokens [members, batch, 13], member, types, kinds, values, lists [rows, 4] (-1 = no
        creature), lengths."""
        found = self.draw_steps([step])
        found["tokens"] = found["tokens"].view(self.members, self.batch, SENTENCE_LENGTH)
        return found

    def draw_steps(self, steps):
        """The sentences of the steps `steps`, flat over (step, member, sentence). A "stream" is
        one member at one step: its random numbers come from the key of (its seed, the step)."""
        keys = torch.tensor([member_keys(seed, step) for step in steps for seed in self.seeds],
                            dtype=torch.long)
        self.keys = keys.to(self.device)  # [streams, 2]
        streams = len(steps) * self.members
        rows = streams * self.batch
        stream = torch.arange(streams, device=self.device).repeat_interleave(self.batch)
        self.stream = stream  # the stream of every sentence of the flat batch
        self.member = stream % self.members  # and its member (its language)
        self.row_in_member = torch.arange(rows, device=self.device) % self.batch
        everyone = torch.arange(rows, device=self.device)
        rand = lambda purpose: self.uniform(everyone, purpose, 0)[:, 0]

        # 1. Type, list length, kind, value, and f (for "no" questions), with the chances of B.3.
        types = torch.bucketize(rand(FOR_TYPE), self.type_edges, right=True)
        lengths = 1 + torch.bucketize(rand(FOR_LENGTH), self.length_edges, right=True)
        kinds = (rand(FOR_KIND) * self.kinds).long().clamp_(max=self.kinds - 1)
        values = (rand(FOR_VALUE) * 4).long().clamp_(max=3)  # "not" statements: drawn later
        wrong = self._draw_f(rand(FOR_F), lengths)
        lists = torch.full((rows, 4), -1, dtype=torch.long, device=self.device)

        # 2 and 3. The creatures. Only the sentences still without a list take part in each
        # round. All sentences start together, so every sentence still waiting is at the same
        # bulk of tries in the same round.
        todo = everyone
        round_number = 0
        while True:
            tries = TRY_BULKS[round_number % len(TRY_BULKS)]
            ok = self._try_lists(todo, tries, round_number, types, kinds, values, wrong,
                                 lengths, lists)
            todo = todo[~ok]  # the only place where the processor waits for the card
            round_number += 1
            if not todo.numel():
                break
            if round_number % len(TRY_BULKS) == 0:
                # All 50 tries used up: a list one shorter, and f again for "no" questions.
                lengths[todo] -= 1
                is_no = types[todo] == NO_QUESTION
                again = self.uniform(todo, FOR_F_AGAIN, round_number)[:, 0]
                new_f = self._draw_f(again, lengths[todo])
                wrong[todo] = torch.where(is_no, new_f, wrong[todo])

        tokens = self.tokens_of(types, kinds, lists, values)
        return {"tokens": tokens, "member": self.member, "types": types, "kinds": kinds,
                "values": values, "lists": lists, "lengths": lengths}

    def scrambled(self, rows, purpose, round_number, columns=1, tries=1):
        """32-bit random whole numbers: [len(rows) * tries, columns], one for each (sentence in
        `rows`, try, column), worked out from its label (see the top of the file). `rows` are
        numbers of sentences in the flat batch."""
        stream = self.stream[rows]
        sentence = self.row_in_member[rows]
        if tries > 1:
            stream = stream.repeat_interleave(tries)
            sentence = sentence.repeat_interleave(tries)
            attempt = torch.arange(tries, device=self.device).repeat(rows.numel())
        else:
            attempt = torch.zeros_like(sentence)
        # Label, part 1 (the same for every sentence): column, purpose and round, mixed with
        # the first key of each stream -> [streams, columns].
        column = torch.arange(columns, device=self.device)
        label_1 = column + (purpose << 12) + (round_number << 16)
        part_1 = mix(label_1[None, :] ^ self.keys[:, 0:1])
        # Label, part 2: sentence and try, mixed with the second key of the stream. Both parts
        # are scrambled before they are put together, so no pattern of the labels survives.
        label_2 = sentence * MAX_TRIES_IN_BULK + attempt
        part_2 = mix(label_2 ^ self.keys[stream, 1])
        return mix(part_1[stream] ^ part_2[:, None])

    def uniform(self, rows, purpose, round_number, columns=1, tries=1):
        """Random numbers from 0 to 1 (never 1), with 24 binary digits (as many as torch.rand
        gives in float32): [len(rows) * tries, columns]."""
        bits = self.scrambled(rows, purpose, round_number, columns, tries)
        return (bits >> 8).to(torch.float32) * (1.0 / (1 << 24))

    def fine_uniform(self, rows, purpose, round_number, columns=1, tries=1):
        """Random numbers from 0 to 1 (never 1) with 48 binary digits, in float64, made from two
        numbers of 24 digits: fine enough for the smallest creature weight."""
        bits = self.scrambled(rows, purpose, round_number, 2 * columns, tries) >> 8
        whole = (bits[:, 0::2] << 24) | bits[:, 1::2]
        return whole.to(torch.float64) * (1.0 / (1 << 48))

    @staticmethod
    def _draw_f(u, lengths):
        """f with equal chance from 1 to k (k = the list length)."""
        return torch.minimum(1 + (u * lengths).long(), lengths)

    def _try_lists(self, rows, tries, round_number, types, kinds, values, wrong, lengths, lists):
        """Make `tries` attempts at a list for each sentence in `rows`. The first good attempt is
        written into `lists` (and, for "not" statements, `values`). Returns True/False for each
        sentence: did it get a list."""
        n = rows.numel()
        repeat = lambda x: x[rows].repeat_interleave(tries)  # one entry per attempt
        mem, kind, value = repeat(self.member), repeat(kinds), repeat(values)
        kind_of_type, k, f = repeat(types), repeat(lengths), repeat(wrong)
        is_not = kind_of_type == NOT_STATEMENT
        is_no = kind_of_type == NO_QUESTION

        # The group each place of the list is drawn from: those that have the value; for a "no"
        # question the last f places from those that do not; for a "not" statement everyone.
        have = mem * self.per_member + 4 * kind + value
        place = torch.arange(4, device=self.device)[None, :]
        slot = torch.where(is_no[:, None] & (place >= (k - f)[:, None]),
                           (have + 4 * self.kinds)[:, None], have[:, None])
        slot = torch.where(is_not[:, None], (mem * self.per_member + 8 * self.kinds)[:, None],
                           slot)
        used = place < k[:, None]

        # The creatures, place after place, each from its group without those taken already
        # (see "Drawing different creatures" at the top of the file).
        u = self.fine_uniform(rows, FOR_CREATURES, round_number, 4, tries)
        pieces = torch.full((n * tries, 4), -1, dtype=torch.long, device=self.device)
        for p in range(4):
            group = slot[:, p]
            if p == 0:
                point = group + on_grid_floor(u[:, 0])
            else:
                taken = pieces[:, :p].clamp(min=0)
                counts = (slot[:, :p] == group[:, None]) & used[:, :p]
                width = torch.where(counts, self.piece_high[taken] - self.piece_low[taken], 0.0)
                start = torch.where(counts, self.piece_low[taken], math.inf)
                start, order = start.sort(dim=1)
                width = width.gather(1, order)
                point = group + on_grid_floor(u[:, p] * (1.0 - width.sum(dim=1)))
                for j in range(p):  # move past the taken pieces, from the left
                    point = torch.where(point >= start[:, j], point + width[:, j], point)
            found_piece = torch.searchsorted(self.piece_high, point, right=True)
            # Rounding can never carry the point into the next group.
            found_piece = torch.minimum(torch.maximum(found_piece, self.first_piece[group]),
                                        self.last_piece[group])
            pieces[:, p] = torch.where(used[:, p], found_piece, -1)
        attempt = torch.where(pieces >= 0, self.piece_creature[pieces.clamp(min=0)], -1)

        # Is the attempt good? No held-back pair in the list, no creature twice (impossible but
        # for rounding, so it is checked anyway) ...
        there = attempt >= 0
        safe = attempt.clamp(min=0)
        bad = torch.zeros(n * tries, dtype=torch.bool, device=self.device)
        for a in range(4):
            for b in range(a + 1, 4):
                both = there[:, a] & there[:, b]
                bad |= both & (self.held[mem, safe[:, a], safe[:, b]]
                               | (safe[:, a] == safe[:, b]))
        # ... and, for a "not" statement, a value that none of the creatures has.
        list_values = torch.where(there, self.facts[mem[:, None], safe, kind[:, None]].long(),
                                  torch.full_like(safe, -1))
        free = ~(list_values[:, :, None] == torch.arange(4, device=self.device)).any(dim=1)
        good = ~bad & (~is_not | free.any(dim=1))

        # Keep the first good attempt of each sentence.
        good = good.view(n, tries)
        found = good.any(dim=1)
        first = good.to(torch.int32).argmax(dim=1)  # the first True (0 if there is none)
        pick = torch.arange(n, device=self.device) * tries + first
        chosen = attempt[pick]
        # Every list in a random order (the empty places stay at the end).
        order_keys = self.uniform(rows, FOR_ORDER, round_number, 4).masked_fill(chosen < 0, 2.0)
        chosen = chosen.gather(1, order_keys.argsort(dim=1))
        # Only sentences that found a list are written (the others keep -1 and try again).
        lists[rows] = torch.where(found[:, None], chosen, lists[rows])
        # The value of a "not" statement: equal chance among the free values.
        new_not = found & (types[rows] == NOT_STATEMENT)
        value_keys = self.uniform(rows, FOR_NOT_VALUE, round_number, 4).masked_fill(~free[pick],
                                                                                     -1.0)
        values[rows] = torch.where(new_not, value_keys.argmax(dim=1), values[rows])
        return found

    # --------------------------------------------------------------------------------------------

    def tokens_of(self, types, kinds, lists, values):
        """Token rows [rows, 13]: the same layout as Language.tokens_of in v2/lang.py. Every
        sentence is first laid out in 16 fixed places, some of them empty, and the empty places
        are then squeezed out:
          question word | a , b , c and d | do not | link link | not | value | . or ? | answer
        """
        rows = types.numel()
        dev = self.device
        word = self.word
        empty = -1
        k = (lists >= 0).sum(dim=1)
        first_kind = kinds == 0
        question = (types == YES_QUESTION) | (types == NO_QUESTION)
        negated = types == NOT_STATEMENT
        nothing = torch.full((rows,), empty, dtype=torch.long, device=dev)
        const = lambda w: torch.full((rows,), word[w], dtype=torch.long, device=dev)

        places = torch.full((rows, 16), empty, dtype=torch.long, device=dev)
        places[:, 0] = torch.where(question, torch.where(first_kind, const("are"), const("do")),
                                   nothing)
        for i in range(4):  # the list: creature, separator, creature, ...
            places[:, 1 + 2 * i] = torch.where(lists[:, i] >= 0, self.creature_base + lists[:, i],
                                               nothing)
            if i < 3:
                # "and" before the last creature, "," between earlier ones.
                places[:, 2 + 2 * i] = torch.where(k == i + 2, const("and"),
                                                   torch.where(k > i + 2, const(","), nothing))
        places[:, 8] = torch.where(negated & ~first_kind, const("do"), nothing)
        places[:, 9] = torch.where(negated & ~first_kind, const("not"), nothing)
        say_link = ~(question & first_kind)  # "are LIST VAL ?" has its linking word in front
        places[:, 10] = torch.where(say_link, self.link_ids[kinds, 0], nothing)
        places[:, 11] = torch.where(say_link, self.link_ids[kinds, 1], nothing)
        places[:, 12] = torch.where(negated & first_kind, const("not"), nothing)
        places[:, 13] = self.value_ids[kinds, values]
        places[:, 14] = torch.where(question, const("?"), const("."))
        places[:, 15] = torch.where(types == YES_QUESTION, const("yes"),
                                    torch.where(types == NO_QUESTION, const("no"), nothing))

        # Squeeze out the empty places, keeping the order of the others.
        order = torch.sort((places == empty).to(torch.int32), dim=1, stable=True).indices
        packed = places.gather(1, order)
        # No sentence is longer than 13 tokens (checked by the tests; a check here would make
        # the processor wait for the graphics card at every step).
        packed = packed[:, :SENTENCE_LENGTH]
        return torch.where(packed == empty, torch.full_like(packed, PAD_ID), packed)


# Checked once when the file is loaded: the sentence types are numbered as in v2/lang.py.
assert (STATEMENT, NOT_STATEMENT, YES_QUESTION, NO_QUESTION) == (0, 1, 2, 3)

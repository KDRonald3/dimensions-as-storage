"""How fast each machine is, when the stage will end, and which moves make it end sooner.

Plain arithmetic, no Colab here: v3/colab_fleet.py calls it at every look.

Speeds
------
A speed is in member-steps a second for the whole card (both workers together), for one
group shape (columns, kinds, body). Where it comes from, best first:
  1. measured on this machine for this shape (each round a group finishes tells how many
     member-steps it trained in how many seconds);
  2. measured on another machine with the same card for this shape;
  3. measured on another card type for this shape, scaled by how fast the two cards are;
  4. the A100 guess for the shape (measured on Modal A100s, v3/speed_report_2.md), scaled by how
     fast the card is (measured once machines of that card have trained, else a guess).

Work
----
A group of the doubling rule trains round after round (to 40,000 steps, 80,000, ...). After
each look some members end. The work still to do is the rest of the present round for all its
members, then each later round for the share of members expected to go on (SURVIVAL: a guess at
first, then measured from the groups that have passed looks).

The plan
--------
A machine ends when it has trained all the groups it holds: (work of each group / its speed),
added up. The stage ends when the last machine ends. A move of a group from one machine to
another costs MOVE_SECONDS on the new machine (the checkpoint is uploaded and the group's shape is
compiled there: 60 to 165 s). The planner moves groups one at a time, each time the move that
brings the end of the stage (and then that of the next-latest machine, and so on) earliest, as long
as a move gains more than MARGIN_SECONDS.
"""

import math

# Member-steps a second of a whole A100 with 2 workers, by the number of columns
# (v3/speed_report_2.md: 8 columns 6,611; 16: 5,977; 32: 2,904 to 3,655; 64: 1,857 to 2,825).
A100_GUESS = {8: 6600, 16: 6000, 32: 3300, 64: 2300}
# How fast each card is against an A100. T4: measured on Colab, about a quarter. The others are
# guesses from the cards' published figures (they are replaced by measurements as they come).
CARD_GUESS = {"H100": 1.7, "G4": 1.3, "A100": 1.0, "L4": 0.45, "T4": 0.25}
# Compute units an hour, from `colab usage` on 23 November 2026 as machines joined (A100 6.73,
# T4 1.15, L4 1.94, G4 about 10); the H100 is a guess. Only a tie-breaker.
UNITS_AN_HOUR = {"H100": 11.0, "G4": 10.0, "A100": 6.73, "L4": 1.94, "T4": 1.15}

MOVE_SECONDS = 150.0        # take a group to another machine: upload, then compile its shape
STOP_SECONDS = 90.0         # stop a machine in the middle of a round: save, download, release
NEW_MACHINE_SECONDS = 60.0  # start the job on a new machine (before any group can train)
MARGIN_SECONDS = 60.0       # a move must bring the end at least this much earlier
SURVIVAL = 0.6              # share of members of a group expected to go on after a look


def shape_of(configs):
    """The shape of a group: columns x kinds x body (each group of the counted stage has its own)."""
    first = configs[0]
    return f"{first['columns']}x{first['kinds']}x{first['body']}"


def a100_guess(shape):
    columns = int(shape.split("x")[0])
    nearest = min(A100_GUESS, key=lambda c: abs(math.log(c) - math.log(columns)))
    return A100_GUESS[nearest]


def next_look(config, step):
    """The step at which a group at `step` pauses next (as v3/machine_job.next_pause)."""
    if config.get("length_rule") != "doubling":
        return config["steps"]
    look = config["first_look"]
    while look <= step:
        look *= 2
    return min(look, config["steps"])


# The share of a group's members that goes on after a look depends almost only on how many bits a
# creature carries per column of the word table. Seen on 23 November in the first 13 groups that
# had finished: at half a bit per column or less every run ended at the first look (40,000
# steps); at 1 bit per column a run trained 95,000 steps on average; at 4 and 8 bits per column
# 430,000 to 480,000 (15 of 24 ran to the limit). The shares below give those averages; the
# value for 2 bits per column lies between its neighbours and was not measured.
SURVIVAL_BY_BITS_PER_COLUMN = ((0.5, 0.05), (1.0, 0.42), (2.0, 0.70), (4.0, 0.88), (8.0, 0.90),
                               (16.0, 0.92))


def survival_guess(config, fallback=SURVIVAL):
    """The share of members expected to go on after each look, for a group with this setting
    (`fallback` for a group that does not say its kinds and columns)."""
    if not config.get("kinds") or not config.get("columns"):
        return fallback
    ratio = 2.0 * config["kinds"] / config["columns"]  # a kind of fact is 2 bits
    if ratio <= SURVIVAL_BY_BITS_PER_COLUMN[0][0]:
        return SURVIVAL_BY_BITS_PER_COLUMN[0][1]
    nearest = min(SURVIVAL_BY_BITS_PER_COLUMN, key=lambda pair: abs(math.log(pair[0] / ratio)))
    return nearest[1]


def work_left(config, reached, members, survival=SURVIVAL, done_in_round=0.0):
    """Expected member-steps still to train for a group (see the top)."""
    total, step, expected = 0.0, reached, float(members)
    while step < config["steps"] and expected > 0.01:
        look = next_look(config, step)
        total += expected * (look - step)
        step = look
        expected *= survival
    return max(total - done_in_round, 0.05 * total)


class Speeds:
    """What is known about the speeds (see the top). `measured` is
    {card: {shape: member-steps a second}} over all machines, `by_machine` is
    {machine: {shape: member-steps a second}}."""

    def __init__(self, measured=None, by_machine=None):
        self.measured = measured if measured is not None else {}
        self.by_machine = by_machine if by_machine is not None else {}

    def card_factor(self, card):
        """How fast the card is against the A100 guesses: measured if possible."""
        found = self.measured.get(card, {})
        if not found:
            return CARD_GUESS.get(card, 0.25)
        ratios = sorted(rate / a100_guess(shape) for shape, rate in found.items())
        return ratios[len(ratios) // 2]

    def rate(self, card, shape, machine=None):
        """Member-steps a second of a whole card of this type for a group of this shape."""
        if machine is not None and shape in self.by_machine.get(machine, {}):
            return self.by_machine[machine][shape]
        if shape in self.measured.get(card, {}):
            return self.measured[card][shape]
        for other, found in self.measured.items():
            if shape in found:
                return found[shape] * self.card_factor(card) / self.card_factor(other)
        return a100_guess(shape) * self.card_factor(card)

    def record(self, machine, card, shape, member_steps, seconds, sharing):
        """A round has ended: `member_steps` trained in `seconds` while `sharing` groups trained on
        the card at once. Stored as the speed of the whole card (an average with what was known)."""
        if not member_steps or not seconds or seconds <= 0:
            return None
        whole = member_steps / seconds * max(1, sharing)
        for table, key in ((self.by_machine.setdefault(machine, {}), shape),
                           (self.measured.setdefault(card, {}), shape)):
            old = table.get(key)
            table[key] = whole if old is None else 0.5 * old + 0.5 * whole
        return whole


# ------------------------------------------------------------------------------------------------
# The plan
# ------------------------------------------------------------------------------------------------
#
# machines: {name: {"card": .., "ready": seconds from now until it can train (0 for a running
#                   machine), "rate": function shape -> member-steps a second}}
# groups:   {index: {"work": member-steps left, "shape": .., "machine": name or None,
#                    "movable": may it be moved now (between rounds, checkpoint here),
#                    "compiled_on": machines where its shape is compiled already}}


def seconds_on(group, name, machines):
    """Seconds a group needs on a machine, moving cost included."""
    cost = 0.0 if name in group["compiled_on"] else MOVE_SECONDS
    return cost + group["work"] / machines[name]["rate"](group["shape"])


def ends_of(machines, groups, where):
    """Seconds from now until each machine has trained the groups `where` gives it."""
    ends = {name: m["ready"] for name, m in machines.items()}
    for index, name in where.items():
        if name in ends:  # (groups on a machine outside the plan, e.g. one that is leaving)
            ends[name] += seconds_on(groups[index], name, machines)
    return ends


def stage_end(ends):
    return max(ends.values()) if ends else 0.0


def earlier(new, old, margin=MARGIN_SECONDS):
    """Is the list of machine ends `new` better than `old`? Both are compared from the latest
    machine down: the latest may not get later, and the first that differs by more than the margin
    must be earlier."""
    new, old = sorted(new, reverse=True), sorted(old, reverse=True)
    if new and old and new[0] > old[0] + 1e-6:
        return False
    for a, b in zip(new, old):
        if a < b - margin:
            return True
        if a > b + margin:
            return False
    return False


def improve(machines, groups, where, movable, limit=500):
    """Move groups (those in `movable`) one at a time while a move makes the stage end earlier
    (see `earlier`). Free groups (where None) are first given to the machine that ends them
    earliest, largest first. Returns the new hand-out."""
    where = dict(where)
    if not machines:
        return where
    ends = ends_of(machines, groups, where)
    for index in sorted((i for i, n in where.items() if n is None),
                        key=lambda i: -groups[i]["work"]):
        best = min(machines, key=lambda n: (ends[n] + seconds_on(groups[index], n, machines), n))
        where[index] = best
        ends[best] += seconds_on(groups[index], best, machines)
    for _ in range(limit):
        best = None
        values = list(ends.values())
        for index in movable:
            here = where[index]
            if here is None or here not in machines:
                continue
            leaving = seconds_on(groups[index], here, machines)
            for name in machines:
                if name == here:
                    continue
                trial = dict(ends)
                trial[here] -= leaving
                trial[name] += seconds_on(groups[index], name, machines)
                if earlier(list(trial.values()), values):
                    key = sorted(trial.values(), reverse=True)
                    if best is None or key < best[0]:
                        best = (key, index, name, trial)
        if best is None:
            break
        _, index, name, ends = best
        where[index] = name
    return where


def without(machines, groups, where, movable, leaving):
    """The hand-out when the machine `leaving` is given back: all its groups go elsewhere (its
    groups in the middle of a round after STOP_SECONDS)."""
    others = {n: m for n, m in machines.items() if n != leaving}
    if not others:
        return None
    trial_groups = dict(groups)
    trial_where = dict(where)
    freed = [i for i, n in where.items() if n == leaving]
    for index in freed:
        group = dict(groups[index])
        if index not in movable:  # stopped in the middle of its round first
            group["work"] = group["work"] + STOP_SECONDS * others[min(others)]["rate"](
                group["shape"])
        trial_groups[index] = group
        trial_where[index] = None
    trial = improve(others, trial_groups, trial_where, list(movable) + freed)
    return trial, ends_of(others, trial_groups, trial), trial_groups


def cost_an_hour(machines, ends):
    """Units an hour of the machines that still have work (for the tie-breaker)."""
    return sum(UNITS_AN_HOUR.get(m["card"], 7.0) for n, m in machines.items() if ends[n] > 0)

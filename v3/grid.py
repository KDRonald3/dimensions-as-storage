"""
v3/grid.py - the runs of the third experiment (v3/PLAN.md, Part B.2 and B.3).

This file uses only the Python standard library: the program on the owner's computer that
sends the sessions to Kaggle imports it without torch or numpy.

    practice()           the practice runs: 240 runs (40 groups of 6)
    counted(settings)    the counted runs: 2,880 runs (120 groups of 24), in the order of B.3;
                         settings is the content of v3/settings.json (the one chosen rate)
    groups(configs)      the runs cut into groups that the group trainer trains together

    python -m v3.grid practice        # print the configs as JSON
    python -m v3.grid counted

Every run has text_seed 0 (one fixed text per language) and the nested 64-column starting table.
"""

import json
import sys
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "settings.json"

COLUMNS = (8, 16, 32, 64)
KINDS = (4, 8, 16, 32, 64)  # 8, 16, 32, 64 and 128 bits per creature
BODIES = (4, 8, 16, 32, 64, 128)
CREATURES = (16, 64, 256, 1024)
USAGES = ("equal", "zipf")
SEEDS = (0, 1, 2)

PRACTICE_BODIES = (4, 128)
PRACTICE_CREATURES = (16, 1024)
PRACTICE_RATES = (0.003, 0.01, 0.03)
PRACTICE_SEED = 100
PRACTICE_STEPS = 40000

LAYERS = 2
HEADS = 2

FIRST_LOOK = 40000  # the doubling rule: first look ...
LIMIT = 640000  # ... and the limit (40,000 doubled four times)

# B.3: the order of the counted groups, so that an early stop still gives a balanced picture.
BODY_ORDER = (128, 4, 32, 8, 64, 16)


def name_of(config, rate_in_name=False):
    """d{columns}_c{creatures}_b{body}_{usage}_k{kinds}_s{seed}, with _lr{rate} for practice."""
    name = (f"d{config['columns']}_c{config['creatures']}_b{config['body']}_{config['usage']}"
            f"_k{config['kinds']}_s{config['seed']}")
    if rate_in_name:
        # Four significant digits: 0.003, 0.01 and 0.03 as they are, a third of 0.01 as 0.003333.
        name += f"_lr{config['lr']:.4g}"
    return name


def make_config(columns, creatures, body, usage, kinds, seed, lr, steps, practice):
    """One run. threads is only used by the one-model trainer (v2/run.py), not on Kaggle."""
    config = {"creatures": creatures, "kinds": kinds, "usage": usage, "body": body,
              "seed": seed, "steps": steps, "lr": lr, "threads": 1, "table": "trained",
              "columns": columns, "text_seed": 0, "table_start": "nested64",
              # B.2: 2 layers and 2 attention heads at every column count (a head has half the
              # columns, so it grows with them).
              "layers": LAYERS, "heads": HEADS}
    if not practice:
        config["length_rule"] = "doubling"
        config["first_look"] = FIRST_LOOK
    config["name"] = name_of(config, rate_in_name=practice)
    return config


def practice():
    """The practice runs (seed 100, never counted): for every columns and kinds, body 4 and 128,
    creatures 16 and 1,024, three rates, equal use, a fixed 40,000 steps. Groups of 6 follow
    each other: kinds, then columns, then body; inside a group creatures, then rate."""
    return [make_config(columns, creatures, body, "equal", kinds, PRACTICE_SEED, lr,
                        PRACTICE_STEPS, practice=True)
            for kinds in KINDS
            for columns in COLUMNS
            for body in PRACTICE_BODIES
            for creatures in PRACTICE_CREATURES
            for lr in PRACTICE_RATES]


CARRY_ON_EVERY = 10  # B.2: every tenth counted run (positions 0, 10, 20, ...) is carried on


def read_settings(path=SETTINGS_FILE):
    with open(path) as file:
        return json.load(file)


def counted(settings):
    """The counted runs, in the order of B.3: body widths 128, 4, 32, 8, 64, 16; inside a body
    width, kinds from 4 to 64; inside those, columns from 8 to 64. Each group of 24 (creatures,
    then usage, then seed) follows the one before. Every run has the ONE rate settings["rate"]
    (written by v3/choose_rates.py). Every tenth run of this fixed list (positions 0, 10, 20,
    ...) has carry_on: true."""
    if "rate" not in settings or settings["rate"] not in PRACTICE_RATES:
        raise ValueError("settings has no chosen rate; run v3/choose_rates.py")
    runs = [make_config(columns, creatures, body, usage, kinds, seed, settings["rate"], LIMIT,
                        practice=False)
            for body in BODY_ORDER
            for kinds in KINDS
            for columns in COLUMNS
            for creatures in CREATURES
            for usage in USAGES
            for seed in SEEDS]
    for position in range(0, len(runs), CARRY_ON_EVERY):
        runs[position]["carry_on"] = True
    return runs


# B.2, the rate check: the best seed of every setting again, at a third and at a tenth of the
# chosen rate. Each is its own stage, with its own result folder (runs_v3/<stage>).
RATE_CHECKS = {"rate_check_third": 3, "rate_check_tenth": 10}


def best_seed(summaries):
    """The seed of the best of a setting's counted runs ({seed: summary}): most facts known; a tie
    goes to the better questions about one creature, then to the lower seed."""
    return min(summaries, key=lambda seed: (-summaries[seed]["final"]["facts"],
                                            -summaries[seed]["final"]["questions_one"], seed))


def rate_check(settings, counted_summaries, divisor):
    """The runs of a rate-check stage (B.2): for each of the 960 settings, the seed of its best
    counted run, trained again from the same start at the chosen rate divided by `divisor`
    (3 or 10), with the doubling rule. counted_summaries is {name: summary} of the counted runs;
    every counted run must be there (ValueError otherwise). Names end in _lr{rate}. In the order
    of the counted runs."""
    runs = counted(settings)
    missing = [c["name"] for c in runs if c["name"] not in counted_summaries]
    if missing:
        raise ValueError(f"{len(missing)} counted runs have not ended yet, for example "
                         f"{', '.join(missing[:3])}")
    by_setting = {}
    for config in runs:
        key = tuple(config[k] for k in ("columns", "creatures", "body", "usage", "kinds"))
        by_setting.setdefault(key, {})[config["seed"]] = counted_summaries[config["name"]]
    rate = settings["rate"] / divisor
    checks = []
    for (columns, creatures, body, usage, kinds), found in by_setting.items():
        config = make_config(columns, creatures, body, usage, kinds, best_seed(found), rate,
                             LIMIT, practice=False)
        config["name"] = name_of(config, rate_in_name=True)
        checks.append(config)
    return checks


# The bridging stage (asked for by the owner on 23 November 2026, in the evening).
# In the counted stage every model with more table columns also had more trained numbers: on an
# axis of trained numbers the four column counts never meet, so the stage cannot tell whether more
# columns help because they give the words more room or simply because they add trained numbers.
# These runs bridge the gaps. For every bit size and every pair of neighbouring column counts, the
# narrower model gets a body so wide that it has as many trained numbers as the model with twice
# the columns and the narrowest body (4). At 128 bits, for example, 8 columns with a body of 457
# against 16 columns with a body of 4: both about 30,100 trained numbers. Everything else is as in
# the counted stage: the chosen rate, the doubling rule, seeds 0, 1 and 2, text seed 0, the nested
# starting table. Only the hardest language: 1,024 creatures, words used equally. Every run is
# carried on (carry_on): a run the rule ends as "no more gain" before the limit trains two more
# doublings, and its result stays where the rule ended it. At 8 bits and 8 columns, 5 of the 18
# counted runs of this language stopped at 40,000 steps on 81.2 (three of the four kinds of fact
# learned); the one of them that was carried on reached 100. So the carried-on scores show
# whether a bridging run was stopped on such a plateau.
BRIDGE_CREATURES = 1024
BRIDGE_USAGE = "equal"
# (kinds, columns): the body width. Worked out with v2/net.count_numbers on the model itself: the
# whole number that brings the trained numbers closest to those of the model with twice the columns
# and a body of 4 (all within 0.07 %). v3/tests/test_v3_bridge.py checks every one again.
BRIDGE_BODIES = {
    (4, 8): 316, (4, 16): 372, (4, 32): 472,
    (8, 8): 325, (8, 16): 382, (8, 32): 482,
    (16, 8): 344, (16, 16): 401, (16, 32): 502,
    (32, 8): 382, (32, 16): 440, (32, 32): 541,
    (64, 8): 457, (64, 16): 517, (64, 32): 620,
}


def bridge(settings):
    """The runs of the bridging stage: 15 models (5 bit sizes x 3 column counts), each with the
    seeds 0, 1 and 2: 45 runs, all carried on. The longest first: kinds from 64 down to 4, inside them columns
    from 32 down to 8. Same rate as the counted runs (settings["rate"])."""
    if "rate" not in settings or settings["rate"] not in PRACTICE_RATES:
        raise ValueError("settings has no chosen rate; run v3/choose_rates.py")
    runs = [make_config(columns, BRIDGE_CREATURES, BRIDGE_BODIES[(kinds, columns)], BRIDGE_USAGE,
                        kinds, seed, settings["rate"], LIMIT, practice=False)
            for kinds in reversed(KINDS)
            for columns in reversed(COLUMNS[:-1])
            for seed in SEEDS]
    for run in runs:
        run["carry_on"] = True
    return runs


def groups(configs):
    """The runs cut into groups of the group trainer: runs with the same columns, kinds and body
    width (and every other shared setting) train together, whatever their rate, creatures, word
    use and seed. In the order in which each group first appears."""
    found = {}
    for config in configs:
        key = (config["columns"], config["kinds"], config["body"], config["steps"],
               config.get("length_rule"), config.get("first_look"), config["table"],
               config.get("table_start"))
        found.setdefault(key, []).append(config)
    return list(found.values())


def test_stage():
    """Four groups, only for trying the programs on Kaggle (never a result; names start with
    "test_"). Groups 1 and 2 are counted groups as they will be run (24 members, body width 128;
    kinds 4 with 8 columns, and the heaviest: kinds 64 with 64 columns), to measure the speed
    and to be stopped by the session's budget. Groups 3 and 4 are tiny (2 members, 16
    creatures, two rates, the doubling rule with a first look at step 400 and a limit of 800),
    so they finish within a session."""
    configs = []
    for kinds, columns in ((4, 8), (64, 64)):
        for creatures in CREATURES:
            for usage in USAGES:
                for seed in SEEDS:
                    config = make_config(columns, creatures, 128, usage, kinds, seed, 0.01,
                                         LIMIT, practice=False)
                    config["name"] = "test_" + config["name"]
                    configs.append(config)
    for columns in (8, 16):
        for seed, lr in ((0, 0.01), (1, 0.03)):
            config = make_config(columns, 16, 4, "equal", 4, seed, lr, 800, practice=False)
            config["first_look"] = 400
            config["batch"] = 128
            config["name"] = "test_" + name_of(config, rate_in_name=True)
            configs.append(config)
    return configs


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("practice", "counted"):
        sys.exit("usage: python -m v3.grid practice|counted")
    chosen = practice() if sys.argv[1] == "practice" else counted(read_settings())
    print(json.dumps(chosen, indent=1))

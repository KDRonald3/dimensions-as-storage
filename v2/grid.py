"""
v2/grid.py - the list of runs of each stage of the second experiment (PLAN.md, part B.8).

This file uses only the Python standard library: a launcher without torch or numpy imports it.

    python -m v2.grid main8        # print the configs of a stage as JSON

Stages
------
smoke    one short run, to see that everything works
pilot    the practice runs (seed 100) that decide the learning rate and the training length
main8    the counted runs at 8 bits per creature (4 kinds of fact): 200 runs
bits16   the same grid with 8 kinds of fact, seeds 0 to 2
bits32   the same grid with 16 kinds of fact, seeds 0 to 2
bits64   the same grid with 32 kinds of fact, seeds 0 to 2
long     the runs four times as long (empty until the counted runs are in)
frozen8  the main8 grid with a frozen word table (a later test, not counted)

Every stage except frozen8 uses a trained word table ("table": "trained").

The training length and learning rate of the counted stages come from v2/settings.json, which is
filled in from the pilot.
"""

import json
import sys
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "settings.json"
STAGES = ["smoke", "pilot", "main8", "bits16", "bits32", "bits64", "long", "frozen8"]

CREATURES = (16, 64, 256, 1024)
BODIES = (8, 16, 32, 64, 128)
USAGES = ("equal", "zipf")
def threads_for(body):
    """Processor threads for a run. A small body trains almost as fast on 1 thread as on 2
    (measured on the cloud machines on 22 November), so it gets 1 and costs half."""
    return 1 if body <= 64 else 2


def settings():
    """The settings fixed from the pilot: steps, lr, pilot_steps."""
    with open(SETTINGS_FILE) as file:
        return json.load(file)


def name_of(config):
    """The default run name (the same rule as v2/run.py): "_frozen" is added at the end when the
    word table is frozen."""
    name = (f"c{config['creatures']}_b{config['body']}_{config['usage']}"
            f"_k{config['kinds']}_s{config['seed']}")
    if config.get("table", "trained") == "frozen":
        name += "_frozen"
    return name


def make_config(creatures, body, usage, kinds, seed, steps, lr, suffix="", table="trained"):
    config = {"creatures": creatures, "kinds": kinds, "usage": usage, "body": body,
              "seed": seed, "steps": steps, "lr": lr, "threads": threads_for(body), "table": table}
    config["name"] = name_of(config) + suffix
    return config


def full_grid(kinds, seeds, table="trained", stage_settings=None, creatures=CREATURES,
              bodies=BODIES):
    """Every creatures x body x usage x seed. steps and lr are the 8-bit ones unless the stage
    has its own entry in settings.json (each stage's settings come from its own practice runs)."""
    fixed = stage_settings or settings()
    return [make_config(n, body, usage, kinds, seed, fixed["steps"], fixed["lr"], table=table)
            for n in creatures
            for body in bodies
            for usage in USAGES
            for seed in seeds]


def grid(stage: str) -> list[dict]:
    """The configs of one stage."""
    if stage == "smoke":
        return [make_config(16, 32, "equal", 4, 100, 300, 0.01)]
    if stage == "pilot":
        # Practice runs, never counted. Three learning rates, at the corners of the grid.
        # 8 bits (4 kinds): 15 runs. 64 bits (32 kinds), the hardest setting: 6 runs, longer.
        fixed = settings()
        runs = [make_config(creatures, body, "equal", 4, 100, fixed["pilot_steps"], lr,
                            suffix=f"_lr{lr}")
                for lr in (0.003, 0.01, 0.03)
                for creatures, body in ((16, 8), (256, 8), (1024, 8), (256, 128), (1024, 128))]
        runs += [make_config(1024, body, "equal", 32, 100, fixed["pilot_steps_64bits"], lr,
                             suffix=f"_lr{lr}")
                 for lr in (0.003, 0.01, 0.03)
                 for body in (8, 128)]
        return runs
    if stage == "main8":
        return full_grid(kinds=4, seeds=range(5))
    if stage == "bits16":
        return full_grid(kinds=8, seeds=range(3), stage_settings=settings().get("bits16"))
    if stage == "bits32":
        return full_grid(kinds=16, seeds=range(3), stage_settings=settings().get("bits32"))
    if stage == "bits64":
        # Cut down on 22 November to fit the cloud credit: 3 vocabularies, 3 body widths and
        # 2 repeats, because one 64-bit run is about ten times as long as an 8-bit run.
        return full_grid(kinds=32, seeds=range(2), stage_settings=settings().get("bits64"),
                         creatures=(16, 256, 1024), bodies=(8, 32, 128))
    if stage == "pilot2":
        # Practice runs for 16 and 32 bits (8 and 16 kinds), never counted.
        return [make_config(1024, body, "equal", kinds, 100, 40000, lr, suffix=f"_lr{lr}")
                for kinds in (8, 16) for lr in (0.003, 0.01) for body in (8, 128)]
    if stage == "long":
        # The plan's control for every setting that ended below 95 facts known: one run (seed 0)
        # four times as long, to tell "cannot hold" from "not learned yet". Only 64-bit settings
        # ended below 95; this list was read from the results on 22 November 2026.
        fixed = settings()["bits64"]
        below_95 = [("equal", 1024, 8), ("equal", 1024, 32), ("zipf", 256, 8), ("zipf", 256, 32),
                    ("zipf", 256, 128), ("zipf", 1024, 8), ("zipf", 1024, 32), ("zipf", 1024, 128)]
        return [make_config(creatures, body, usage, 32, 0, 4 * fixed["steps"], fixed["lr"],
                            suffix="_long")
                for usage, creatures, body in below_95]
    if stage == "speed":
        # Not an experiment: four short runs that only measure how fast a cloud machine trains
        # with 1 and with 2 threads, for the smallest and the largest body.
        return [dict(make_config(1024, body, "equal", 4, 100, 1500, 0.01, suffix=f"_t{threads}"),
                     threads=threads)
                for body in (8, 128) for threads in (1, 2)]
    if stage == "frozen8":
        return full_grid(kinds=4, seeds=range(5), table="frozen")
    raise ValueError(f"unknown stage {stage!r}; the stages are: {', '.join(STAGES)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: python -m v2.grid STAGE   (one of: {', '.join(STAGES)})")
    print(json.dumps(grid(sys.argv[1]), indent=1))

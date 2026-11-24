"""Writes MADE-UP results, only to test the analysis program (analyze.py) and the report layout.

    .venv/bin/python v2/make_fake_runs.py                 # the 200 runs of stage main8
    .venv/bin/python v2/make_fake_runs.py --extra         # also made-up runs for bits16, bits32, frozen8

Everything goes to /tmp/air_v2_fake (never into the project). The numbers are invented by a simple
rule: a model of a given size "holds" a certain number of creatures, and the score falls off once
there are more creatures than that. None of this says anything about the real experiment.
A marker file FAKE_DATA is written, so that analyze.py puts "MADE-UP DATA" on every picture.
"""

import argparse
import json
import math
import random
from pathlib import Path

OUT = Path("/tmp/air_v2_fake")
CREATURES = [16, 64, 256, 1024]
BODIES = [8, 16, 32, 64, 128]
USAGES = ["equal", "zipf"]
SEEDS = [0, 1, 2, 3, 4]
STEPS = 20000
BATCH = 512


def grading_steps(steps):
    """Steps 100 x 2^(i/2), rounded, and the last step (as in the plan, B.7)."""
    points, i = [], 0
    while round(100 * 2 ** (i / 2)) <= steps:
        points.append(round(100 * 2 ** (i / 2)))
        i += 1
    if points[-1] != steps:
        points.append(steps)
    return points


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def made_up_score(creatures, body, usage, kinds, rng, table):
    """The final 'facts known' score: 100 while the model holds the creatures, falling to 25."""
    holds = 130 * (body / 8) ** 1.1  # how many creatures this body size can hold (made up)
    if table == "trained":
        holds *= 4  # a trained table gives every creature its own room: it holds more (made up)
    holds /= (kinds / 4) ** 0.8  # more kinds of fact: more to store
    if usage == "zipf":
        holds *= 0.8
    x = 3.0 * (math.log2(holds / creatures) + 0.3) + rng.gauss(0, 0.5)
    return 25 + 75 * sigmoid(x)


def parameters(creatures, body, kinds, table):
    """Counts of numbers in the model, as the plan's summary.json lists them (made-up rule)."""
    rows_in_use = 1 + 14 + 5 * kinds + creatures
    without_table = 2 * (65 * body + 32) + 13 * 32 + 32 * (14 + 4 * kinds)
    table_numbers = rows_in_use * 32
    return {"trained": without_table + (table_numbers if table == "trained" else 0),
            "trained_without_table": without_table, "table_rows_in_use": rows_in_use,
            "table_numbers": table_numbers, "body_middle": 2 * (65 * body + 32)}


def make_summary(creatures, body, usage, kinds, seed, table="trained"):
    rng = random.Random(f"{creatures}-{body}-{usage}-{kinds}-{seed}-{table}")
    facts = made_up_score(creatures, body, usage, kinds, rng, table)
    learned = (facts - 25) / 75  # 0 = nothing learned, 1 = all learned
    questions_one = max(0.0, 100 * learned ** 1.5 - rng.uniform(0, 4))
    questions_new = max(0.0, 100 * learned ** 2.5 - rng.uniform(0, 8))
    questions_seen = max(0.0, 100 * learned ** 2.2 - rng.uniform(0, 5))
    not_known = min(100.0, 25 + 75 * learned ** 1.8 - rng.uniform(0, 3))
    loss = 0.02 + 1.5 * (1 - learned) ** 1.5
    by_kind = [min(100.0, max(0.0, facts + rng.gauss(0, 2))) for _ in range(kinds)]
    detail = lambda q: {"yes": min(100.0, q + 3), "no_one": min(100.0, q + 5),
                        "no_more": max(0.0, q - 4), "count": 2000}
    curve = []
    tau = 1500 * (1 + 0.4 * math.log2(creatures / 16)) / (body / 32) ** 0.3
    for step in grading_steps(STEPS):
        progress = 1 - math.exp(-step / tau)
        scale = lambda v, chance: chance + (v - chance) * progress + rng.gauss(0, 0.6)
        curve.append({
            "step": step, "sentences": step * BATCH,
            "facts": round(min(100, max(0, scale(facts, 25))), 2),
            "questions_one": round(min(100, max(0, scale(questions_one, 0))), 2),
            "questions_new": round(min(100, max(0, scale(questions_new, 0))), 2),
            "questions_seen": round(min(100, max(0, scale(questions_seen, 0))), 2),
            "not_known": round(min(100, max(0, scale(not_known, 25))), 2),
            "loss": round(loss + 2.5 * (1 - progress), 4)})
    curve[-1].update(facts=round(facts, 2), questions_one=round(questions_one, 2),
                     questions_new=round(questions_new, 2),
                     questions_seen=round(questions_seen, 2), not_known=round(not_known, 2))
    name = f"c{creatures}_b{body}_{usage}_k{kinds}_s{seed}" + ("_frozen" if table == "frozen" else "")
    return {
        "name": name,
        "config": {"creatures": creatures, "kinds": kinds, "usage": usage, "body": body,
                   "seed": seed, "table": table, "steps": STEPS, "lr": 0.01, "threads": 2, "columns": 32,
                   "layers": 2, "heads": 2, "batch": BATCH, "warmup": 100, "average": 0.998,
                   "name": name},
        "parameters": parameters(creatures, body, kinds, table),
        "steps": STEPS, "sentences": STEPS * BATCH, "minutes": round(20 + body / 8, 1),
        "steps_per_second": 12.0,
        "final": {"facts": round(facts, 2), "facts_by_kind": [round(v, 2) for v in by_kind],
                  "questions_one": round(questions_one, 2),
                  "questions_new": round(questions_new, 2),
                  "questions_new_detail": detail(questions_new),
                  "questions_seen": round(questions_seen, 2),
                  "questions_seen_detail": detail(questions_seen),
                  "not_known": round(not_known, 2), "loss": round(loss, 4)},
        "curve": curve}


def write_stage(stage, creatures_list, bodies, usages, kinds, table="trained"):
    count = 0
    for creatures in creatures_list:
        for body in bodies:
            for usage in usages:
                for seed in SEEDS:
                    summary = make_summary(creatures, body, usage, kinds, seed, table)
                    folder = OUT / stage / summary["name"]
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "summary.json").write_text(json.dumps(summary))
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--extra", action="store_true",
                        help="also write made-up runs for bits16, bits32 and frozen8")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FAKE_DATA").write_text("These results are made up. Never use them as results.\n")
    total = write_stage("main8", CREATURES, BODIES, USAGES, 4)
    if args.extra:
        total += write_stage("bits16", [16, 64, 256], [8, 32, 128], ["equal"], 8)
        total += write_stage("bits32", [16, 64, 256], [8, 32, 128], ["equal"], 16)
        total += write_stage("frozen8", CREATURES, BODIES, USAGES, 4, "frozen")
    print(f"wrote {total} made-up runs under {OUT}")


if __name__ == "__main__":
    main()

"""
v3/card_check.py - a reviewer's check of the group trainer ON A GRAPHICS CARD, with the card
settings switched on (compiler search, recorded step, small attention program, 256 steps of
sentences at once). The tests in v2/tests and v3/tests run on the processor only, where none of
these is used, so this is the only check of them against the reference trainer.

    ~/.local/bin/modal run v3/modal_card_check.py --columns 32   # on one A100 on Modal
    python -m v3.card_check                                 # on any machine with a card

It prints one line per check ("OK" or "PROBLEM") and the differences it measured.

  1. Sentences: those made 256 steps at once on the card equal those made one step at a time on
     the processor, and a member's sentences do not depend on the other members.
  2. 64-bit numbers (so that rounding cannot hide a real difference): a group with one member
     that ends early, one marked member that is carried on and one ended by the real rule, with
     the card settings on (all but the recorded step, see as_64_bits), ends with the same averaged copies (model.pt) as separate
     trainings by the reference code of v2/run.py on the same batches. The card's "TF32" must
     be off.
  3. 64-bit numbers: the same group trained breadth first (paused at every look and picked up
     again, as v3/machine_job.py does) ends as the group trained straight through; the "so far"
     files are right at every pause.
  4. 32-bit numbers (as the counted stage runs): the group trainer against the reference; the
     recorded step on and off; rows of unused creatures never move; the scores of model.pt,
     graded by the reference code on the processor, equal the summary's.
"""

import copy
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from v2 import batch, run
from v2.batch_text import BatchText
from v2.lang import Language, grade
from v3 import machine_job

DEVICE = os.environ.get("CHECK_DEVICE", "cuda")
COLUMNS = int(os.environ.get("CHECK_COLUMNS", "32"))  # 8: heads of 4 columns (the "add up" way)
STARTED = time.time()
PROBLEMS = []


def report(ok, what, detail=""):
    if not ok:
        PROBLEMS.append(what)
    print(f"{'OK     ' if ok else 'PROBLEM'} {what} {detail}  [{time.time() - STARTED:.0f} s]",
          flush=True)


def member(name, creatures, usage, seed, **more):
    config = {"creatures": creatures, "kinds": 4, "usage": usage, "body": 8, "seed": seed,
              "steps": 160, "lr": 0.01, "threads": 1, "batch": 64, "text_seed": 0, "columns": COLUMNS,
              "table_start": "nested64", "length_rule": "doubling", "first_look": 40,
              "name": name}
    config.update(more)
    return run.complete_config(config)


# E ends at its first look (40) ("learned", made up), A is ended there as "no more gain" but is
# carried on to 160 (the limit), B follows the real rule (to 160 at most; 64-bit rounding
# grows to about 1e-5 by 320 steps, so the runs are kept short).
GROUP = [member("E", 16, "zipf", 2), member("A", 16, "equal", 0, carry_on=True),
         member("B", 64, "zipf", 1)]
REAL_END_REASON = run.end_reason


def made_up_rule(config, step, curve):
    if config["name"] == "E" and step == 40:
        return "learned"
    if config["name"] == "A" and step == 40:
        return "no more gain"
    return REAL_END_REASON(config, step, curve)


batch.end_reason = made_up_rule


def languages(configs):
    return [Language(c["creatures"], c["kinds"], c["usage"]) for c in configs]


def reference_batches(configs, steps):
    langs = languages(configs)
    return [np.stack([lang.sample_batch(c["seed"], step, c["batch"])
                      for lang, c in zip(langs, configs)]) for step in range(steps)]


def reference_averaged(config, lang, batches, member_index, steps, dtype):
    """The averaged copy after `steps` steps of the reference code (v2/run.py) on the card."""
    torch.manual_seed(config["seed"])
    model = batch.make_net(lang, config["body"], config["columns"], config["layers"],
                           config["heads"], config["table"],
                           table_start=config["table_start"]).to(dtype).to(DEVICE)
    averaged = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=run.WEIGHT_DECAY)
    for step in range(steps):
        inputs, targets = run.inputs_and_targets(lang, batches[step][member_index])
        run.train_step(model, optimizer, inputs.to(DEVICE), targets.to(DEVICE),
                       run.learning_rate_at(step, config))
        run.update_average(averaged, model, step, config)
    return {k: v.detach().cpu() for k, v in averaged.state_dict().items()}


def model_of(folder, name):
    return torch.load(Path(folder) / name / "model.pt", weights_only=False)["model"]


def largest_difference(a, b):
    """The largest difference over all tensors, leaving out the bias of the attention keys:
    its true gradient is exactly zero (it changes no attention weight), so it moves only by
    rounding noise that AdamW blows up, in the reference as much as in the group trainer (see
    v2/tests/test_v2_batch.py key_bias_part). Also names the tensor with that difference."""
    worst, where = 0.0, ""
    for k in a:
        gap = (a[k].double() - b[k].double()).abs()
        if k.endswith("attention.query_key_value.bias"):
            columns = gap.shape[0] // 3
            gap = torch.cat([gap[:columns], gap[2 * columns:]])
        if gap.max().item() > worst:
            worst, where = gap.max().item(), k
    return worst, where


def end_step(summary):
    return summary["ended"]["step"] if "ended" in summary else summary["steps"]


# ------------------------------------------------------------------------------------------------


def check_text():
    langs = languages(GROUP)
    seeds = [run.text_seed_of(c) for c in GROUP]
    on_card = BatchText(langs, seeds, 64, DEVICE).make_steps(range(0, 256)).cpu()
    on_cpu = BatchText(langs, seeds, 64, "cpu")
    same = all(torch.equal(on_card[s], on_cpu.make(s)) for s in (0, 1, 63, 64, 200, 255))
    alone = BatchText(langs[2:], seeds[2:], 64, DEVICE).make_steps(range(100, 356)).cpu()
    report(same and torch.equal(alone[:156], on_card[100:, 2:]),
           "1. sentences: card (256 at once) = processor (one at a time); member alone = in group")


def as_64_bits():
    """Both trainers in 64-bit numbers. The recorded step is off here: torch's fused AdamW
    takes its learning rate as a tensor only in 32-bit numbers. (It is checked in 32 bits in
    part 4.) Returns a function that puts everything back."""
    plain_make_net = run.make_net
    batch.make_net = lambda *a, **k: plain_make_net(*a, **k).double()
    batch.GRAPH_ON_CARD = False
    return lambda: (setattr(batch, "make_net", plain_make_net),
                    setattr(batch, "GRAPH_ON_CARD", True))


def check_64_bits(work):
    undo = as_64_bits()
    try:
        batches = reference_batches(GROUP, 160)
        out = work / "group64"
        summaries, _ = batch.train_group(GROUP, out, DEVICE, quiet=True,
                                         batch_maker=lambda s: torch.from_numpy(batches[s]))
        report(not torch.backends.cuda.matmul.allow_tf32, "2. TF32 is off by default")
        ends = {s["name"]: end_step(s) for s in summaries}
        print("   ended:", {s["name"]: s.get("ended") for s in summaries}, flush=True)
        report(ends["E"] == 40 and ends["A"] == 40 and "carried_on" in summaries[1],
               "2. E ended at 40, A carried on and its result is that of step 40")
        langs = languages(GROUP)
        for i, (config, lang) in enumerate(zip(GROUP, langs)):
            reference = reference_averaged(config, lang, batches, i, ends[config["name"]],
                                           torch.float64)
            difference, where = largest_difference(model_of(out, config["name"]), reference)
            report(difference < 1e-9, f"2. 64 bits, member {config['name']} (to step "
                   f"{ends[config['name']]}): model.pt = reference", f"largest difference "
                   f"{difference:.2e} ({where})")
    finally:
        undo()


def check_breadth_first(work):
    """The group with its own sentences (the real sentence maker), straight through and
    breadth first."""
    undo = as_64_bits()
    try:
        straight_out = work / "straight64"
        straight, _ = batch.train_group(GROUP, straight_out, DEVICE, quiet=True)
        out, so_far = work / "parts64", work / "so_far"
        step, rounds = 0, 0
        while True:
            pause = machine_job.next_pause(GROUP[0], step)
            result = batch.run_group_until(GROUP, out, DEVICE, groups_root=work / "groups",
                                           pause_at=pause, so_far_root=so_far, quiet=True)
            rounds += 1
            if not result["unfinished"]:
                break
            step = result["step"]
            notes = {p.stem: json.loads(p.read_text()) for p in so_far.glob("*.json")}
            right = (set(notes) == set(result["unfinished"])
                     and all(n["step"] == step == pause and n["curve"][-1]["step"] == step
                             for n in notes.values()))
            report(right, f"3. paused at {step}: so-far files of {sorted(notes)}")
        report(not any(so_far.glob("*.json")), "3. no so-far file left at the end")
        for one in straight:
            two = json.loads((out / one["name"] / "summary.json").read_text())
            same = (one["ended"] == two["ended"] and one.get("carried_on") == two.get("carried_on")
                    and [p["facts"] for p in one["curve"]] == [p["facts"] for p in two["curve"]]
                    and max(abs(a["loss"] - b["loss"]) for a, b in zip(one["curve"], two["curve"]))
                    < 1e-9)
            difference, where = largest_difference(model_of(out, one["name"]),
                                                   model_of(straight_out, one["name"]))
            report(same and difference < 1e-9, f"3. {one['name']}: {rounds} rounds = straight "
                   f"through", f"largest difference {difference:.2e} ({where})")
    finally:
        undo()


def check_32_bits(work):
    configs = [dict(c, steps=120, length_rule="fixed") for c in GROUP]
    configs = [run.complete_config({k: v for k, v in c.items()
                                    if k not in ("length_rule", "first_look", "carry_on")})
               for c in configs]
    batches = reference_batches(configs, 120)
    feed = lambda s: torch.from_numpy(batches[s])
    found = {}
    for graph in (True, False):
        batch.GRAPH_ON_CARD = graph
        out = work / f"group32_{graph}"
        summaries, state = batch.train_group(configs, out, DEVICE, quiet=True, batch_maker=feed)
        found[graph] = (summaries, out, state)
    batch.GRAPH_ON_CARD = True
    langs = languages(configs)
    for i, (config, lang) in enumerate(zip(configs, langs)):
        reference = reference_averaged(config, lang, batches, i, 120, torch.float32)
        # 32-bit rounding grows over 120 steps (on the processor too), so this is a coarse
        # check only; the 64-bit checks above are the exact ones.
        on = largest_difference(model_of(found[True][1], config["name"]), reference)[0]
        off = largest_difference(model_of(found[False][1], config["name"]), reference)[0]
        both = largest_difference(model_of(found[True][1], config["name"]),
                                  model_of(found[False][1], config["name"]))[0]
        report(on < 0.1, f"4. 32 bits, member {config['name']}: group vs reference",
               f"recorded step on {on:.1e}, off {off:.1e}, on vs off {both:.1e}")
        # Unused creature rows never move (exactly).
        start = run.make_net(lang, config["body"], config["columns"], table_start=config["table_start"]).word_table
        saved = model_of(found[True][1], config["name"])["word_table"]
        unused = slice(lang.rows_in_use, None)
        report(torch.equal(saved[unused], start.detach()[unused]),
               f"4. {config['name']}: unused rows of the averaged table never moved")
        # The summary's scores = the reference grading of model.pt on the processor.
        model = run.make_net(lang, config["body"], config["columns"], table_start=config["table_start"])
        model.load_state_dict(model_of(found[True][1], config["name"]))
        model.eval()
        graded = grade(lang, run.predictor(model))
        summary = found[True][0][i]
        worst = max(abs(summary["final"][k] - v) for k, v in graded.items()
                    if k != "loss" and isinstance(v, (int, float)))
        report(worst < 1e-9, f"4. {config['name']}: scores of model.pt = summary's",
               f"largest difference {worst}")


def main():
    print(torch.__version__, torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE, flush=True)
    work = Path(tempfile.mkdtemp())
    for check in (lambda: check_text(),
                  lambda: check_64_bits(work),
                  lambda: check_breadth_first(work),
                  lambda: check_32_bits(work)):
        try:
            check()
        except Exception:
            report(False, "a check stopped with an error", traceback.format_exc()[-2500:])
    print(f"{len(PROBLEMS)} problems", flush=True)
    sys.exit(1 if PROBLEMS else 0)


if __name__ == "__main__":
    main()

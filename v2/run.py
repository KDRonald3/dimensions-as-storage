"""
v2/run.py - one training run of the second experiment (PLAN.md, parts B.7 and B.8).

How to run it
-------------
    python -m v2.run --creatures 64 --body 32 --usage equal --kinds 4 --seed 0 \
        --steps 20000 --lr 0.01 --threads 2 --out-root runs_v2

The run writes OUT_ROOT/NAME/summary.json (all the scores, with the training curve) and
OUT_ROOT/NAME/model.pt (the averaged copy of the weights, with the config).

Picking up again
----------------
Every 1,000 steps the run saves a checkpoint (OUT_ROOT/NAME/checkpoint.pt). A run that finds a
checkpoint goes on from it, and ends exactly as if it had never stopped, because the batch of
step t depends only on (seed, t). A run that finds a finished summary.json returns it at once.
The checkpoint is removed when the run is finished.

Terms
-----
step           One adjustment of the weights, after one batch of 512 sentences.
learning rate  How big each adjustment is (lr).
warm-up        The first 100 steps, in which the learning rate rises from almost 0 to lr.
averaged copy  A running average of the weights over roughly the last 500 steps. It wobbles
               less than the weights themselves; it is the copy that is graded and saved.
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v2.lang import Language, grade
from v2.net import count_numbers, make_net

# Fixed parts of the recipe, as in the first experiment.
GRADIENT_CLIP = 1.0  # a step larger than this is scaled down to it
WEIGHT_DECAY = 0.0
IGNORE = -100  # target value meaning "no loss here" (PyTorch's convention)
CHECKPOINT_EVERY = 1000  # steps between checkpoints

REQUIRED_KEYS = ["creatures", "kinds", "usage", "body", "seed", "steps", "lr", "threads"]
OPTIONAL_DEFAULTS = {"table": "trained", "columns": 32, "layers": 2, "heads": 2, "batch": 512,
                     "warmup": 100, "average": 0.998}
# Settings added for the third experiment (v3/PLAN.md, Part B.1). Unlike the keys above, they
# are NOT filled in when they are left out, so a config without them is exactly a config of the
# second experiment (and its summary is unchanged). Left out, each means the old behaviour:
#   text_seed    the training text depends on (text_seed, step) instead of (seed, step); seed
#                then sets the starting weights only
#   table_start  "nested64": the third experiment's starting word table (see v2/net.py)
#   length_rule  "doubling": the run may end early by the doubling rule (see end_reason);
#                "fixed" (the old behaviour): it trains for exactly `steps`
#   first_look   with the doubling rule: the step of the first look (default 40,000)
#   carry_on     with the doubling rule: true for the fixed sample of runs that are trained on
#                anyway (v3/PLAN.md B.2). If such a run ends as "no more gain", it is trained for
#                two more doublings (or to the limit). Its result (final, ended, steps and
#                model.pt) stays that of the point where the rule ended it; the later gradings
#                are kept in its curve and summed up under "carried_on" in its summary.
NEW_KEYS = ["text_seed", "table_start", "length_rule", "first_look", "carry_on"]
FIRST_LOOK = 40000
# The doubling rule's marks (v3/PLAN.md B.1).
LEARNED_MARK = 99.0  # facts known and questions about one creature both at least this
GAIN_MARK = 1.0  # "no more gain": facts known rose by less than this over each of two doublings
# Scores are sums of fractions, so a score of exactly 99, or a gain of exactly 1 point, can come
# out a hair below it (98.99999999999999). Such a hair is not counted against the run.
HAIR = 1e-9
CARRY_DOUBLINGS = 2  # a carried-on run is trained this many doublings further


def default_name(config):
    """c{creatures}_b{body}_{usage}_k{kinds}_s{seed}, with "_frozen" at the end when the word
    table is frozen."""
    name = (f"c{config['creatures']}_b{config['body']}_{config['usage']}"
            f"_k{config['kinds']}_s{config['seed']}")
    if config.get("table", "trained") == "frozen":
        name += "_frozen"
    return name


def complete_config(config):
    """The config with every key filled in (defaults for the optional ones, and the name)."""
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"the config lacks these keys: {', '.join(missing)}")
    unknown = (set(config) - set(REQUIRED_KEYS) - set(OPTIONAL_DEFAULTS) - set(NEW_KEYS)
               - {"name"})
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    full = {key: config[key] for key in REQUIRED_KEYS}
    for key, value in OPTIONAL_DEFAULTS.items():
        full[key] = config.get(key, value)
    if full["table"] not in ("trained", "frozen"):
        raise ValueError(f'table must be "trained" or "frozen"; got "{full["table"]}".')
    # The third experiment's settings: only those that are given.
    for key in NEW_KEYS:
        if key in config:
            full[key] = config[key]
    if "table_start" in full and full["table_start"] != "nested64":
        raise ValueError(f'table_start must be "nested64" (or left out); got '
                         f'"{full["table_start"]}".')
    if "text_seed" in full and not isinstance(full["text_seed"], int):
        raise ValueError("text_seed must be a whole number")
    rule = full.get("length_rule", "fixed")
    if rule not in ("fixed", "doubling"):
        raise ValueError(f'length_rule must be "fixed" or "doubling"; got "{rule}".')
    if rule == "doubling":
        full.setdefault("first_look", FIRST_LOOK)
        first, limit = full["first_look"], full["steps"]
        # The limit must be the first look doubled a whole number of times, and a quarter of
        # the first look a whole step, so that every look has the two earlier points it needs.
        doublings = limit // first if first > 0 else 0
        if (first % 4 != 0 or limit % first != 0 or doublings < 1
                or doublings & (doublings - 1) != 0):
            raise ValueError(f"with the doubling rule, steps ({limit}) must be first_look "
                             f"({first}) times 1, 2, 4, 8, ..., and first_look a multiple of 4")
    elif "first_look" in full:
        raise ValueError("first_look is only used with length_rule doubling")
    if "carry_on" in full:
        if not isinstance(full["carry_on"], bool):
            raise ValueError("carry_on must be true or false")
        if rule != "doubling":
            raise ValueError("carry_on is only used with length_rule doubling")
    full["name"] = config.get("name") or default_name(full)
    return full


def doubling(config):
    """True when the run (a complete config) ends by the doubling rule."""
    return config.get("length_rule", "fixed") == "doubling"


def text_seed_of(config):
    """The number that, with the step, sets the training text: text_seed when given, else the
    seed (as in the second experiment)."""
    return config["text_seed"] if "text_seed" in config else config["seed"]


def looks(config):
    """The steps at which a run with the doubling rule is looked at: first_look, and each time
    its length has doubled, up to the limit `steps`. Empty for a fixed length."""
    if not doubling(config):
        return []
    points, look = [], config["first_look"]
    while look <= config["steps"]:
        points.append(look)
        look *= 2
    return points


def grading_points(config):
    """All steps at which a run is graded: the usual ones (grading_steps) and, with the doubling
    rule, also a quarter and a half of the first look and every look."""
    points = set(grading_steps(config["steps"]))
    if doubling(config):
        points |= {config["first_look"] // 4, config["first_look"] // 2}
        points |= set(looks(config))
    return sorted(points)


def end_reason(config, step, curve):
    """The doubling rule (v3/PLAN.md B.1), applied after the grading of step `step`.

    curve is the run's list of graded points so far (each with "step", "facts" and
    "questions_one"). Returns why the run ends here ("learned", "no more gain" or "limit"), or
    None when it goes on. It is only ever decided at a look."""
    if step not in looks(config):
        return None
    at = {point["step"]: point for point in curve}
    now, half, quarter = at[step], at[step // 2], at[step // 4]
    if now["facts"] >= LEARNED_MARK - HAIR and now["questions_one"] >= LEARNED_MARK - HAIR:
        return "learned"
    # "No more gain": less than 1 point over the last doubling AND over the one before it.
    if (now["facts"] - half["facts"] < GAIN_MARK - HAIR
            and half["facts"] - quarter["facts"] < GAIN_MARK - HAIR):
        return "no more gain"
    if step >= config["steps"]:
        return "limit"  # reached the limit; the run is "still rising"
    return None


def carries_on(config, step, reason):
    """True when a run that the rule ends here with `reason` is to be trained on anyway: it is
    one of the marked runs (carry_on), it ended as "no more gain", and it is not at the limit."""
    return (reason == "no more gain" and config.get("carry_on", False)
            and step < config["steps"])


def carry_until(config, step):
    """The step at which a run carried on from `step` stops: two more doublings, or the limit."""
    return min(step * 2 ** CARRY_DOUBLINGS, config["steps"])


def carried_on_summary(final, at_end, step, why):
    """The "carried_on" part of a summary: the scores where the carrying on stopped, and how
    much facts known gained after the rule ended the run (at_end: the scores there)."""
    return {"step": step, "why": why,
            "facts": final["facts"], "questions_one": final["questions_one"],
            "questions_new": final["questions_new"], "questions_seen": final["questions_seen"],
            "not_known": final["not_known"], "loss": final["loss"],
            "gain": final["facts"] - at_end["facts"]}


def add_stop_evidence(summary):
    """For a run that ended as "no more gain" (v3/PLAN.md, "Showing that a run was not stopped
    too early"): its facts known at a quarter, a half and the whole of its length, and how far
    its training loss fell over the last doubling (the training loss of the grading at half the
    length minus that at the end; each is the average since the grading before it). Added to
    the summary as "stop_evidence". Other summaries are left as they are."""
    ended = summary.get("ended")
    if not ended or ended["reason"] != "no more gain":
        return
    step = ended["step"]
    at = {point["step"]: point for point in summary["curve"]}
    quarter, half, whole = at[step // 4], at[step // 2], at[step]
    summary["stop_evidence"] = {
        "facts_quarter": quarter["facts"], "facts_half": half["facts"],
        "facts_whole": whole["facts"], "loss_half": half["loss"], "loss_whole": whole["loss"],
        "loss_fall": half["loss"] - whole["loss"]}


def same_run(a, b):
    """True when two complete configs describe the same run. The thread count is left out: it
    changes only the speed (and the last digits of the sums), not what is trained."""
    return ({k: v for k, v in a.items() if k != "threads"}
            == {k: v for k, v in b.items() if k != "threads"})


def grading_steps(steps):
    """The steps after which the model is graded: 100 x 2^(i/2), rounded (100, 141, 200, 283,
    ...), up to `steps`, and at `steps` itself."""
    points, i = set(), 0
    while round(100 * 2 ** (i / 2)) <= steps:
        points.add(round(100 * 2 ** (i / 2)))
        i += 1
    points.add(steps)
    return sorted(points)


def learning_rate_at(step, config):
    """Rises in a straight line over the warm-up, then stays at lr (as in the first
    experiment). step 0 is the first step."""
    if config["warmup"] <= 0:
        return config["lr"]
    return config["lr"] * min(1.0, (step + 1) / config["warmup"])


def inputs_and_targets(lang, tokens):
    """The model reads the first 12 tokens of every row; the target at each place is the next
    token. Where that next token is <pad> or a creature name there is no loss (IGNORE).
    Otherwise the target is the number of the word among the model's outputs (token id - 1)."""
    inputs = tokens[:, :-1]
    following = tokens[:, 1:]
    graded = (following != 0) & (following < lang.creature_base)
    targets = np.where(graded, following - 1, IGNORE)
    return torch.from_numpy(np.ascontiguousarray(inputs)), torch.from_numpy(targets)


def train_step(model, optimizer, inputs, targets, rate):
    """Adjust the weights once. Returns the loss on the batch, in bits per graded word."""
    for group in optimizer.param_groups:
        group["lr"] = rate
    scores = model(inputs)
    loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    return loss.item() / math.log(2)  # natural-log units / ln 2 = bits


def update_average(averaged, model, steps_before, config):
    """Move the averaged copy a little toward the trained weights (as in the first experiment).
    Early in a run the average is over the steps so far, so the starting values do not hold it
    back for hundreds of steps."""
    keep = min(config["average"], (1 + steps_before) / (10 + steps_before))
    with torch.no_grad():
        for mean, now in zip(averaged.parameters(), model.parameters()):
            # mean + (1 - keep) * (now - mean): the same as keep * mean + (1 - keep) * now,
            # but a number that has not moved (a word-table row never used) stays exactly as
            # it was. The other way of writing it changes such numbers by rounding.
            mean.lerp_(now, 1 - keep)


def predictor(model):
    """The model as an exam taker: predict(prompts, last) -> scores [rows, n_out] at the place
    `last` of every prompt."""

    def predict(prompts, last):
        with torch.no_grad():
            scores = model(torch.from_numpy(np.ascontiguousarray(prompts)))
            rows = torch.arange(len(last))
            return scores[rows, torch.from_numpy(np.asarray(last))].numpy()

    return predict


def write_safely(path, write):
    """Write a file under a temporary name first, then rename it, so that a run stopped in the
    middle never leaves half a file."""
    temporary = Path(str(path) + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def make_summary(config, numbers, steps, seconds, curve, final):
    return {
        "name": config["name"],
        "config": config,
        "parameters": numbers,
        "steps": steps,
        "sentences": steps * config["batch"],
        "minutes": seconds / 60,
        "steps_per_second": steps / seconds if seconds > 0 else 0.0,
        "final": final,
        "curve": curve,
    }


def run_one(config: dict, out_root: str) -> dict:
    """Train one model. Writes out_root/<name>/summary.json and model.pt. Returns the summary."""
    return train(config, out_root)


def train(config, out_root, stop_after=None, quiet=False):
    """The training run behind run_one. stop_after (for tests only) stops the run right after
    the checkpoint of that step, as if the computer had been switched off."""
    config = complete_config(config)
    run_dir = Path(out_root) / config["name"]
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if summary_path.exists():
        with open(summary_path) as file:
            summary = json.load(file)
        if not same_run(summary["config"], config):
            raise ValueError(f"{summary_path} belongs to another config; use another out_root "
                             f"or name.")
        return summary
    run_dir.mkdir(parents=True, exist_ok=True)
    say = (lambda *args: None) if quiet else (lambda *args: print(*args, flush=True))

    torch.set_num_threads(config["threads"])
    lang = Language(config["creatures"], config["kinds"], config["usage"])
    torch.manual_seed(config["seed"])  # the starting values
    model = make_net(lang, config["body"], config["columns"], config["layers"], config["heads"],
                     config["table"], config.get("table_start", "v2"))
    averaged = copy.deepcopy(model)  # the copy that is graded and saved
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=WEIGHT_DECAY)
    predict = predictor(averaged)
    numbers = count_numbers(model, lang)

    step, curve, loss_sum, loss_count, seconds = 0, [], 0.0, 0, 0.0
    # A marked run that is carried on (carry_on): where the rule ended it, its scores and
    # averaged weights there, and the step at which carrying on stops. None otherwise.
    carry = None
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=False)
        if not same_run(saved["config"], config):
            raise ValueError(f"{checkpoint_path} belongs to another config; remove it first.")
        model.load_state_dict(saved["model"])
        averaged.load_state_dict(saved["averaged"])
        optimizer.load_state_dict(saved["optimizer"])
        step, curve = saved["step"], saved["curve"]
        loss_sum, loss_count, seconds = saved["loss_sum"], saved["loss_count"], saved["seconds"]
        carry = saved.get("carry")
        say(f"{config['name']}: going on from the checkpoint at step {step:,}")
    else:
        say(f"{config['name']}: {numbers['trained']:,} trained numbers "
            f"({numbers['body_middle']:,} in the middle parts), word table {config['table']} "
            f"({numbers['table_numbers']:,} numbers in use)")

    gradings = set(grading_points(config))
    text_seed = text_seed_of(config)
    started = time.perf_counter()
    time_batches = time_grading = 0.0
    final = None
    ended = None  # with the doubling rule: {"step": ..., "reason": ...} when the run ends
    while step < config["steps"]:
        clock = time.perf_counter()
        tokens = lang.sample_batch(text_seed, step, config["batch"])
        inputs, targets = inputs_and_targets(lang, tokens)
        time_batches += time.perf_counter() - clock

        model.train()
        loss = train_step(model, optimizer, inputs, targets, learning_rate_at(step, config))
        if not math.isfinite(loss):
            raise RuntimeError(f"Training broke down at step {step:,}: the loss is no longer a "
                               f"number. This is a failed run, not a result. Try a smaller lr.")
        update_average(averaged, model, step, config)
        step += 1
        loss_sum += loss
        loss_count += 1

        if step in gradings:
            clock = time.perf_counter()
            averaged.eval()
            final = grade(lang, predict)
            final["loss"] = loss_sum / loss_count  # average training loss since last grading
            loss_sum, loss_count = 0.0, 0
            time_grading += time.perf_counter() - clock
            curve.append({"step": step, "sentences": step * config["batch"],
                          "facts": final["facts"], "questions_one": final["questions_one"],
                          "questions_new": final["questions_new"],
                          "questions_seen": final["questions_seen"],
                          "not_known": final["not_known"], "loss": final["loss"]})
            say(f"  step {step:>7,}  loss {final['loss']:.3f}  facts {final['facts']:5.1f}  "
                f"one {final['questions_one']:5.1f}  new {final['questions_new']:5.1f}  "
                f"seen {final['questions_seen']:5.1f}  not {final['not_known']:5.1f}")
            if carry is not None:
                if step >= carry["until"]:
                    say(f"  carried on to step {step:,}")
                    break
            else:
                reason = end_reason(config, step, curve)
                if reason is not None and carries_on(config, step, reason):
                    # Keep the result of this point, then train on (it is not the result).
                    carry = {"ended": {"step": step, "reason": reason},
                             "final": copy.deepcopy(final),
                             "averaged": copy.deepcopy(averaged.state_dict()),
                             "until": carry_until(config, step)}
                    say(f"  ended at step {step:,}: {reason}; carried on to step "
                        f"{carry['until']:,} (a marked run)")
                elif reason is not None:
                    ended = {"step": step, "reason": reason}
                    say(f"  ended at step {step:,}: {reason}")
                    break

        if step % CHECKPOINT_EVERY == 0 and step < config["steps"]:
            state = {"config": config, "model": model.state_dict(),
                     "averaged": averaged.state_dict(), "optimizer": optimizer.state_dict(),
                     "step": step, "curve": curve, "loss_sum": loss_sum,
                     "loss_count": loss_count,
                     "seconds": seconds + time.perf_counter() - started}
            if carry is not None:
                state["carry"] = carry
            write_safely(checkpoint_path, lambda path: torch.save(state, path))
            if stop_after is not None and step >= stop_after:
                return None

    seconds += time.perf_counter() - started
    spent = time.perf_counter() - started
    if spent > 0:
        say(f"  time: making batches {100 * time_batches / spent:.0f}%, grading "
            f"{100 * time_grading / spent:.0f}%, training "
            f"{100 * (spent - time_batches - time_grading) / spent:.0f}%")
    if final is None:  # only when a finished checkpoint is picked up again: grade once more
        final = grade(lang, predict)
        final["loss"] = curve[-1]["loss"] if curve else 0.0

    saved_weights = averaged.state_dict()
    if carry is not None:
        # A carried-on run: its result is that of the point where the rule ended it.
        summary = make_summary(config, numbers, carry["ended"]["step"], seconds, curve,
                               carry["final"])
        summary["ended"] = carry["ended"]
        why = "limit" if step >= config["steps"] else "two more doublings"
        summary["carried_on"] = carried_on_summary(final, carry["final"], step, why)
        saved_weights = carry["averaged"]
    else:
        summary = make_summary(config, numbers, step, seconds, curve, final)
    if ended is not None:
        summary["ended"] = ended
    add_stop_evidence(summary)
    write_safely(run_dir / "model.pt",
                 lambda path: torch.save({"config": config, "model": saved_weights}, path))
    # The summary is written last: if summary.json exists, the run is complete.
    write_safely(summary_path,
                 lambda path: path.write_text(json.dumps(summary, indent=1)))
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    say(f"  saved {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train one model of the second experiment.")
    parser.add_argument("--creatures", type=int, required=True)
    parser.add_argument("--body", type=int, required=True)
    parser.add_argument("--usage", choices=["equal", "zipf"], required=True)
    parser.add_argument("--kinds", type=int, choices=[4, 8, 16, 32, 64], default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--table", choices=["trained", "frozen"], default="trained")
    for key, value in OPTIONAL_DEFAULTS.items():
        if key != "table":
            parser.add_argument(f"--{key}", type=type(value), default=value)
    # The third experiment's settings (left out: the second experiment's behaviour).
    parser.add_argument("--text-seed", dest="text_seed", type=int, default=None)
    parser.add_argument("--table-start", dest="table_start", choices=["nested64"], default=None)
    parser.add_argument("--length-rule", dest="length_rule", choices=["fixed", "doubling"],
                        default=None)
    parser.add_argument("--first-look", dest="first_look", type=int, default=None)
    parser.add_argument("--carry-on", dest="carry_on", action="store_const", const=True,
                        default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-root", default="runs_v2")
    args = vars(parser.parse_args())
    out_root = args.pop("out_root")
    for key in ["name"] + NEW_KEYS:
        if args[key] is None:
            del args[key]
    summary = run_one(args, out_root)
    final = summary["final"]
    print(f"{summary['name']}: facts {final['facts']:.1f}, questions one "
          f"{final['questions_one']:.1f}, new {final['questions_new']:.1f}, seen "
          f"{final['questions_seen']:.1f}, not {final['not_known']:.1f}, "
          f"{summary['minutes']:.1f} minutes")


if __name__ == "__main__":
    main()

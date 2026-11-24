"""Go on training a saved model on the same kind of text, and watch its question score.

    python more_training.py runs/a16_w8_equal_s0 --extra-sets 256

Why: the runs of the experiment were ended as soon as the model knew the statements ("foxes are
red ."). Many of them could not yet answer the questions ("are foxes green ? no"). This program
takes such a saved model and simply trains it further, on new sets of the same text (statements
and questions, in the same mix as before), and grades it as it goes.

How it continues, and where it differs from never having stopped:

  * The text continues where the run left off: the first new set is set number (sets already
    read), drawn the same way as before, so nothing about the text changes.
  * The starting point is the SAVED model, which is the steadied (averaged) copy that was graded,
    not the raw model of the last training step. They are nearly the same.
  * The optimizer starts afresh: its memory of earlier steps was not saved. To keep that fresh
    start gentle, the learning rate rises to its normal value over the first `warmup` steps, as
    it does at the start of every run.
  * The graded model is again a steadied copy, as in the experiment.

It stops when the question score has been at least `solved_at` at two gradings in a row while the
statements are still known ("questions solved"), or after `extra_sets` sets ("cap").

With --until statements it is used to FINISH a run that ran out of time before its model knew
the statements: it then stops when the statement score has been at least `solved_at` at two
gradings in a row ("statements solved"). --max-minutes adds a time limit ("time").

It saves OUT_DIR/NAME/summary.json (every grading) and OUT_DIR/NAME/model.pt (the final model,
in the same form as the runs of the experiment, so questions.py, ask.py and probe.py can open it).
"""

import argparse
import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import questions
import score
import train

# The gradings, counted in sets of extra training. About two gradings for every doubling.
GRADINGS = [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024,
            1448, 2048, 2896, 4096, 5793, 8192]


def grade(model, language, eval_rows):
    """Everything that is graded: the scores of score.py, plus yes and no questions apart."""
    predict = score.model_predictor(model)
    result = score.score(predict, language, eval_rows)
    result.update(questions.test_questions(predict, language))
    return {key: result[key] for key in ("facts_known", "questions_right", "yes_right",
                                         "no_right", "yes_chance_when_yes", "yes_chance_when_no",
                                         "says_yes", "loss", "gap")}


def continue_run(run_dir, extra_sets, solved_at, threads, out_dir, until="questions",
                 max_minutes=0.0):
    started = time.perf_counter()
    run_dir = Path(run_dir)
    before = json.loads((run_dir / "summary.json").read_text())
    torch.set_num_threads(min(threads, os.cpu_count()))
    config, language, model = train.load_run(run_dir)
    torch.manual_seed(config.seed)
    eval_rows = train.evaluation_rows(language, config)
    sets_before, steps_before = before["sets"], before["steps"]
    # load_run rebuilds the language with the automatic set size. A run that was given another
    # set size by hand would be continued in sets of a different length: refuse, do not mislead.
    if before["language"]["set_sentences"] != language.set_sentences:
        raise SystemExit(f"{before['name']} was trained in sets of "
                         f"{before['language']['set_sentences']:,} sentences, not the automatic "
                         f"{language.set_sentences:,}. This program cannot continue it.")

    # The graded model is a steadied copy of the trained one; both start as the saved model.
    graded = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=train.WEIGHT_DECAY)

    start = grade(graded, language, eval_rows)
    print(f"{before['name']}: {sets_before:,} sets read before. Start: facts known "
          f"{start['facts_known']:.1f}, questions {start['questions_right']:.1f} "
          f"(yes right {start['yes_right']:.0f}%, no right {start['no_right']:.0f}%)")
    print(f"{'extra sets':>10} {'facts':>7} {'questions':>10} {'yes right':>10} {'no right':>9}"
          f" {'minutes':>8}")

    curve, stop_reason, new_steps = [], None, 0
    for extra in range(1, extra_sets + 1):
        # The same text as before, continued: set numbers go on from where the run stopped.
        sentences = language.make_set(sets_before + extra - 1, data_seed=config.seed)
        inputs, targets = language.pack_rows(sentences, config.context)
        inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
        model.train()
        for first in range(0, len(inputs), config.batch):
            # A fresh optimizer gets the same gentle start as a new run: the warm-up.
            rate = train.learning_rate_at(new_steps, config)
            loss = train.train_step(model, optimizer, inputs[first:first + config.batch],
                                    targets[first:first + config.batch], rate)
            train.check_not_broken(loss, new_steps)
            # The steadied copy keeps its normal pace (steps counted over the whole training).
            train.update_average(graded, model, steps_before + new_steps, config)
            new_steps += 1
        out_of_time = max_minutes > 0 and (time.perf_counter() - started) / 60 >= max_minutes
        if extra in GRADINGS or extra == extra_sets or out_of_time:
            point = grade(graded, language, eval_rows)
            point.update({"extra_sets": extra, "sets": sets_before + extra,
                          "extra_steps": new_steps,
                          "minutes": (time.perf_counter() - started) / 60})
            curve.append(point)
            print(f"{extra:>10,} {point['facts_known']:>7.1f} {point['questions_right']:>10.1f} "
                  f"{point['yes_right']:>10.1f} {point['no_right']:>9.1f} "
                  f"{point['minutes']:>8.1f}", flush=True)
            last_two = curve[-2:]
            known = len(last_two) == 2 and all(p["facts_known"] >= solved_at for p in last_two)
            answered = known and all(p["questions_right"] >= solved_at for p in last_two)
            if until == "questions" and answered:
                stop_reason = "questions solved"
            elif until == "statements" and known:
                stop_reason = "statements solved"
            elif out_of_time:
                stop_reason = "time"
            if stop_reason:
                break
    if stop_reason is None:
        stop_reason = "cap"

    name = before["name"] + "_more"
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": name, "continued_from": str(run_dir), "config": asdict(config),
        "sets_before": sets_before, "steps_before": steps_before, "start": start,
        # True if the model already answered the questions when it was saved: then the two
        # sets of extra training only confirm it, and "questions solved" is not a gain.
        "solved_at_start": bool(start["questions_right"] >= solved_at),
        "sets": sets_before + curve[-1]["extra_sets"],
        "extra_sets": curve[-1]["extra_sets"], "extra_sets_allowed": extra_sets,
        "stop_reason": stop_reason, "until": until, "final": curve[-1], "curve": curve,
        "minutes": (time.perf_counter() - started) / 60,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # Saved like the runs of the experiment, so the other tools can open it.
    saved_config = {**asdict(config), "name": name, "out_dir": str(out_dir)}
    torch.save({"model": graded.state_dict(), "config": saved_config}, out / "model.pt")
    final = curve[-1]
    print(f"Ended: {stop_reason} after {final['extra_sets']:,} extra sets. Facts known "
          f"{final['facts_known']:.1f}, questions {final['questions_right']:.1f}. Saved {out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train a saved model further and grade its "
                                                 "answers to the yes/no questions.")
    parser.add_argument("run_dir", help="the folder of a saved run, such as runs/a16_w8_equal_s0")
    parser.add_argument("--extra-sets", type=int, default=256, help="at most this many more sets")
    parser.add_argument("--solved-at", type=float, default=97.0,
                        help="question score that counts as solved (twice in a row)")
    parser.add_argument("--threads", type=int, default=1, help="processor threads")
    parser.add_argument("--out-dir", default="runs_more", help="results go to OUT_DIR/NAME_more/")
    parser.add_argument("--until", choices=["questions", "statements"], default="questions",
                        help="stop when the questions are answered (default), or when the "
                             "statements are known (to finish a run that ran out of time)")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="also stop after this many minutes (0 = no time limit)")
    args = parser.parse_args()
    continue_run(args.run_dir, args.extra_sets, args.solved_at, args.threads, args.out_dir,
                 args.until, args.max_minutes)


if __name__ == "__main__":
    main()

"""Test saved models on the yes/no questions of the toy language.

    python questions.py runs                 every saved run in runs/
    python questions.py runs/a16_w8_equal_s0 one run

A question looks like "are foxes green ?" and the right continuation is "yes" or "no". The model
is given the question up to and including "?" and nothing before it, and the word it rates most
likely to come next is its answer. (If that word is neither "yes" nor "no", the answer is wrong.)

For each run this prints, and saves in results/questions.json:

  facts known      the report's score on statements, 0 to 100 (0 = guessing)
  questions        the question score of score.py, 0 to 100: balanced accuracy, so that always
                   answering "no" (right three times in four, with no knowledge) reads 0
  yes right        of the questions whose answer is yes, the percentage answered yes
  no right         of the questions whose answer is no, the percentage answered no
  chance of yes    the average chance the model gives to "yes", where the right answer is yes
                   and where it is no (see test_questions for why this is the clearer picture)

Nothing is trained here. Questions are asked at the start of a row, like the facts. The question
score counts every animal once, also in the unequal-use runs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import score
import train


def chance_of_yes(predict, probes, language):
    """For each question, the chance (0 to 1) the model gives to "yes" as the next word."""
    chances = np.zeros(len(probes))
    for first in range(0, len(probes), score.PROBE_BATCH):
        some = probes[first:first + score.PROBE_BATCH]
        lengths = np.array([len(probe.prefix) for probe in some])
        tokens = np.full((len(some), lengths.max()), score.PAD_ID, dtype=np.int64)
        for row, probe in enumerate(some):
            tokens[row, :lengths[row]] = probe.prefix
        after_last_word = predict(tokens)[np.arange(len(some)), lengths - 1]
        chances[first:first + len(some)] = np.exp(after_last_word[:, language.yes_id])
    return chances


def test_questions(predict, language):
    """The question score, and the plain accuracy on yes questions and on no questions.

    yes_right and no_right count only which of the two words came out on top. A model that
    cannot tell true from false typically gives "yes" about a 50% chance on EVERY question
    (half of the questions in the training text are answered yes), so which word comes out on
    top is decided by a hair and says little. yes_chance_when_yes and yes_chance_when_no show
    that directly: the average chance given to "yes" where the right answer is yes, and where it
    is no. A model that knows the answers has them near 100 and near 0; a model that does not
    has them about equal.
    """
    probes = language.question_probes("in_use")
    said = score.most_likely_next(predict, probes)
    answer = np.array([probe.answer for probe in probes])
    is_yes = answer == language.yes_id
    right = said == answer
    yes_chance = chance_of_yes(predict, probes, language)
    yes_right, no_right = float(100 * right[is_yes].mean()), float(100 * right[~is_yes].mean())
    return {
        "questions_right": score.questions_right(predict, language),
        "questions_unfloored": yes_right + no_right - 100,  # can be below 0: worse than a coin
        "yes_right": yes_right,
        "no_right": no_right,
        "yes_chance_when_yes": float(100 * yes_chance[is_yes].mean()),
        "yes_chance_when_no": float(100 * yes_chance[~is_yes].mean()),
        "says_yes": float(100 * (said == language.yes_id).mean()),
        "says_no": float(100 * (said == language.no_id).mean()),
        "questions": int(len(probes)),
    }


def test_run(run_dir):
    """Load one saved run and test its model. Returns a dictionary of what was found."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    config, language, model = train.load_run(run_dir)
    model.eval()
    predict = score.model_predictor(model)
    result = test_questions(predict, language)
    result.update({
        "name": summary["name"], "folder": str(run_dir.parent), "usage": config.usage,
        "seasons": config.seasons, "animals": config.animals, "width": config.width,
        "seed": config.seed, "table_animals": config.table_animals,
        "ended": summary["stop_reason"],
        "sets": summary.get("sets", summary["final"].get("sets")),
        "facts_known": summary["final"]["facts_known"],
        "questions_right_when_saved": summary["final"]["questions_right"],
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Test saved models on the yes/no questions.")
    parser.add_argument("where", nargs="+", help="a run folder, or a folder of run folders")
    parser.add_argument("--out", default="results/questions.json", help="where to save the table")
    args = parser.parse_args()

    run_dirs = []
    for where in map(Path, args.where):
        if (where / "model.pt").exists():
            run_dirs.append(where)
        else:
            run_dirs += sorted(d for d in where.iterdir() if (d / "model.pt").exists())
    rows = []
    print(f"{'run':34} {'sets':>6} {'facts':>6} {'questions':>10} {'yes right':>10} {'no right':>9}"
          f" {'chance of yes when the answer is yes / no':>44}")
    for run_dir in run_dirs:
        row = test_run(run_dir)
        rows.append(row)
        print(f"{row['name']:34} {row['sets']:>6,} {row['facts_known']:>6.1f} "
              f"{row['questions_right']:>10.1f} {row['yes_right']:>10.1f} {row['no_right']:>9.1f}"
              f" {row['yes_chance_when_yes']:>34.0f}% / {row['yes_chance_when_no']:.0f}%")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} runs tested; saved {out}")


if __name__ == "__main__":
    main()

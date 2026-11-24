"""
train.py - one training run: one model, one vocabulary, trained until one of three endings.

What this file is for
---------------------
It builds one toy language (language.py) and one tiny GPT (model.py), lets the model read the
training text set after set, grades it at the end of some sets (score.py), and stops for exactly
one of three reasons:

  solved    the model knows practically every fact. It is big enough; more training is pointless.
  plateau   more training has stopped helping. The model has hit its limit under this recipe.
  budget    time (or the number of sets) ran out first. The run is NOT FINISHED and says nothing
            about the model's limit.

There is a fourth kind of run, for checking a result: with --fixed-sets the stopping rule is
switched off and the run trains for exactly as many sets as you say. It is saved as "fixed".

How to run it
-------------
    python train.py                                  # 16 animals, width 32
    python train.py --animals 1024 --width 4         # many words, a very small model
    python train.py --usage zipf --max-minutes 5     # some animals far more common than others
    python train.py --fixed-sets 500                 # exactly 500 sets, no stopping rule
    python train.py --help                           # every setting is a flag

The result goes to runs/<name>/summary.json (all the numbers, with the learning curve) and
runs/<name>/model.pt (the trained model, for ask.py and probe.py). The name is built from the
settings, for example a16_w32_equal_s0. Only the animals, the width, the usage, the seasons, the
table size and the seed are part of it (and fixed_sets, when it is used): if you change another
setting, pass --name as well, or the new run replaces the old one.

Terms used in this file
-----------------------
parameter      One of the numbers inside the model that training adjusts.
batch          The rows the model reads before its parameters are adjusted once.
step           One adjustment of the parameters, after one batch.
learning rate  How big each adjustment is.
warm-up        The first steps, during which the learning rate rises from almost zero to its full
               value. A fresh model's first adjustments are wild; starting gently keeps it stable.
set            A fixed number of training sentences with exact counts of every animal and fact.
grading        Measuring the model with score.py. It happens only at the end of a set.
sighting       One training sentence about a fact. "Each fact was seen 400 times" means 400
               sightings per fact on average.
doubling       Training for twice as many sets as before. The stopping rule asks what the last
               doubling of training bought.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import textwrap
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from language import IGNORE, Language, LanguageConfig
from model import ModelConfig, TinyGPT, count_parameters
from score import model_predictor, most_likely_next, score


@dataclass
class RunConfig:
    # --- the language ---
    animals: int = 16  # animal words in use: the vocabulary-size dial
    table_animals: int = 1024  # animal rows in the word table, the same for every model of a round
    seasons: int = 1  # 1 = 8 bits per animal, 4 = 32 bits per animal
    usage: str = "equal"  # "equal": every animal equally often; "zipf": the r-th 1/r as often
    set_sentences: int = 0  # sentences per training set; 0 = 15 x 4 x seasons x table_animals
    # --- the model ---
    width: int = 32  # numbers per word: the model-size dial
    layers: int = 2  # blocks stacked on top of each other
    heads: int = 2  # attention heads per block
    # --- training ---
    seed: int = 0  # decides the model's starting values, the invented animals' facts, the shuffling
    batch: int = 64  # rows read before each adjustment of the parameters
    context: int = 96  # words per row
    # The learning rate was chosen in calibration on the 16-animal language only, before any run
    # of the experiment: 0.01 is the highest rate at which every width that can learn those 16
    # animals does so smoothly (width 8 needed about 2,400 steps at 0.01 against 13,600 at 0.002;
    # 0.02 gained nothing and 0.03 was erratic). It is the same for every run of the experiment.
    learning_rate: float = 0.01  # how big each adjustment is
    warmup: int = 100  # steps over which the learning rate rises to its full value
    # The model that is graded (and saved) is not the model of the last training step but a
    # running average of its parameters over roughly the last 1 / (1 - average) = 500 steps.
    # With a constant learning rate the parameters never settle: they keep wobbling around a
    # center, and in calibration the score of a small model on a large vocabulary wobbled by
    # several points from one grading to the next (92, then 100). Averaging removes the wobble
    # and changes nothing about the training itself. 0 = grade the trained model as it is.
    average: float = 0.998  # share of the old average that is kept at each training step
    # --- the stopping rule ---
    solved_at: float = 97.0  # facts_known that counts as solved
    flat_points: float = 2.0  # gaining fewer points than this over a doubling of training is flat
    flat_gap: float = 0.05  # the gap shrinking by less than this fraction over a doubling is flat
    # The best possible loss is itself only accurate to about 0.002 bits per word (see sample_rows
    # in language.py), so a smaller movement of the gap is noise, however large a share of a tiny
    # gap it is. Without this setting a run just below solved_at, whose gap is only a few
    # thousandths of a bit, could never be called flat.
    flat_gap_bits: float = 0.002  # the gap shrinking by fewer bits per word than this is flat too
    min_sightings: float = 400  # no plateau before each fact was seen this often, on average
    # min_steps is there so that a run cannot be called flat during the warm-up, or before the
    # model has begun to learn: at the very start the score stands at 0 and the gap hardly moves,
    # which looks just like a plateau. It makes the stopping rule stricter, never looser.
    min_steps: int = 2000  # no plateau before this many training steps
    # A model first learns the sentence shapes, then stands still for a while with no facts known,
    # and only then starts on the facts. In calibration that stand-still lasted between about 600
    # and 4,800 steps depending on width and vocabulary, and one run (1,024 animals, width 16) was
    # wrongly called a plateau after 2,385 steps with 0.4 facts known. No fixed number of steps
    # is safe against that. So a run that has not begun to learn facts is never called a plateau:
    # if it never begins, it ends by budget and is reported as not finished. Like min_steps, this
    # makes the stopping rule stricter, never looser.
    started_at: float = 5.0  # no plateau while facts_known is below this: learning has not begun
    max_minutes: float = 20.0  # budget: stop at the first set end after this much time
    max_sets: int = 100000  # budget: never train on more sets than this
    # fixed_sets is for checking a result by training much longer than the stopping rule would.
    # Above 0, the run trains for exactly that many sets: the rule and the budget are not asked,
    # and the run is saved as "fixed" under a name of its own (see default_name).
    fixed_sets: int = 0  # 0 = off; above 0: train exactly this many sets, with no stopping rule
    # --- running ---
    threads: int = 8  # processor threads; on this machine one run is fastest with 8
    out_dir: str = "runs"  # results go to out_dir/name/
    name: str | None = None  # None = built from the settings, see default_name


# Fixed parts of the training recipe. They are the same for every run of the experiment, so they
# are not flags; change them here if you want to try something else.
EVAL_ROWS = 256  # rows of fresh text on which loss, floor and gap are measured
WEIGHT_DECAY = 0.0  # no pull of the parameters toward zero
GRADIENT_CLIP = 1.0  # an adjustment that would be larger than this is scaled down to it
GRADINGS_PER_DOUBLING = 2  # the model is graded twice for every doubling of the training text
EVAL_STREAM = 4  # tags the random numbers of the evaluation rows (language.py uses 1, 2 and 3)

# How a run can end. The first three are the endings of the stopping rule; "fixed" is a run that
# was given a fixed number of sets (fixed_sets) and never asked the rule.
STOP_REASONS = ["solved", "plateau", "budget", "fixed"]

# The files that decide what a run does. A short mark of their content is saved with every run
# (see code_mark), so that it can be seen later which runs were made by the same code.
CODE_FILES = ["language.py", "model.py", "score.py", "train.py"]


def code_mark():
    """A short mark of the code that makes the runs: 12 characters worked out from every
    character of CODE_FILES (a hash: the same files always give the same mark, and any change
    gives another one).

    Every setting of a run is saved in its summary, but a change of the code itself would not
    show there. With the mark it does: two runs with the same mark were made by exactly the same
    code. An edit that changes no result (a comment, say) changes the mark too, so two different
    marks mean "check what changed in between", not "these runs are wrong".
    """
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:12]


# Worked out once, when this file is loaded: the code a program runs is the code as it was then,
# also if a file is edited while a long round of runs is still training.
CODE_MARK = code_mark()


def default_name(config):
    """The run's name, built from its settings: a16_w32_equal_s0, a256_w16_equal_x4_s0, ...

    a = animals in use, w = width, then the usage, x = seasons (only if more than one),
    t = animals in the table (only if not the usual 1,024), s = seed, f = fixed_sets (only if
    it is used: a16_w32_equal_s0_f500, so that such a run never replaces the ordinary one).
    """
    name = f"a{config.animals}_w{config.width}_{config.usage}"
    if config.seasons > 1:
        name += f"_x{config.seasons}"
    if config.table_animals != RunConfig.table_animals:
        name += f"_t{config.table_animals}"
    name += f"_s{config.seed}"
    if config.fixed_sets > 0:
        name += f"_f{config.fixed_sets}"
    return name


# ------------------------------------------------------------------------------------------------
# When to grade and when to stop
# ------------------------------------------------------------------------------------------------


def grading_sets():
    """The set counts at which the model is graded: 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, ...

    Each is about 1.41 times the last, so every second one is a doubling. Grading costs time and
    learning slows down as it goes on, so there is no point in grading after every set for ever.
    The list never ends; the run takes numbers from it for as long as it needs.

    One grading can fall outside this list: a run that its budget (or fixed_sets) ends at another
    set count is graded once more there, so that the saved scores belong to the saved model. In
    the learning curve that grading is marked "scheduled": False, all others "scheduled": True.
    """
    last = 0
    k = 0
    while True:
        sets = round(2 ** (k / GRADINGS_PER_DOUBLING))
        if sets > last:  # at the start the rounded numbers repeat (1, 1, 2): skip repeats
            yield sets
            last = sets
        k += 1


def decide_stop(history, config):
    """THE STOPPING RULE. Returns "solved", "plateau" or None (None = keep training).

    history has one entry per grading so far, oldest first. Each entry is a dictionary with at
    least: sets, steps, facts_known, gap, sightings, scheduled. This function only looks at those
    numbers; it does not touch the model. (The endings "budget" and "fixed" are decided by the
    training loop.)
    """
    # SOLVED: the score is at or above solved_at at the last two gradings. Twice, so that one
    # lucky grading does not end the run.
    last_two = history[-2:]
    if len(last_two) == 2 and all(point["facts_known"] >= config.solved_at for point in last_two):
        return "solved"

    # PLATEAU needs five gradings: now, a doubling of training ago, and two doublings ago.
    if len(history) < 1 + 2 * GRADINGS_PER_DOUBLING:
        return None
    now = history[-1]
    one_doubling_ago = history[-1 - GRADINGS_PER_DOUBLING]
    two_doublings_ago = history[-1 - 2 * GRADINGS_PER_DOUBLING]

    # 0. Only at a scheduled grading. The extra grading where a budget ended a run is not a
    #    doubling of training away from the gradings before it, so "what did the last doubling
    #    buy?" cannot be asked there. Only the solved half of the rule, above, is asked.
    if not now["scheduled"]:
        return None
    # 1. Enough practice: a model that has hardly seen the facts is not at its limit. Nor is one
    #    that has taken only a few steps: during the warm-up, and before a model has begun to
    #    learn, the score stands still and the gap hardly moves, which looks just like a plateau.
    if now["sightings"] < config.min_sightings or now["steps"] < config.min_steps:
        return None
    #    And the model must have begun to learn facts at all (see started_at in RunConfig): a
    #    model that knows the sentence shapes and nothing else stands still for a long time
    #    before it starts on the facts, and that stand-still is not a limit.
    if now["facts_known"] < config.started_at:
        return None
    # 2. The score gained fewer than flat_points over EACH of the last two doublings.
    last_gain = now["facts_known"] - one_doubling_ago["facts_known"]
    gain_before = one_doubling_ago["facts_known"] - two_doublings_ago["facts_known"]
    if last_gain >= config.flat_points or gain_before >= config.flat_points:
        return None
    # 3. The gap to the best possible loss has stopped shrinking too. The score only moves when
    #    a fact tips from wrong to right; the gap already moves while the model is still
    #    getting closer. A falling gap means it is still learning, so that is not a plateau.
    #    "Falling" means by at least the fraction flat_gap AND by at least flat_gap_bits: the
    #    best possible loss is only accurate to about 0.002 bits per word, so a smaller fall is
    #    noise, even when it is a large share of a tiny gap.
    shrunk_bits = one_doubling_ago["gap"] - now["gap"]
    if one_doubling_ago["gap"] > 0:
        shrunk_share = shrunk_bits / one_doubling_ago["gap"]
        if shrunk_share >= config.flat_gap and shrunk_bits >= config.flat_gap_bits:
            return None
    return "plateau"


def explain_stop(stop_reason, config, curve):
    """The ending of a run as one plain sentence or two."""
    now = curve[-1]
    known = f"facts known {now['facts_known']:.1f} out of 100"
    if stop_reason == "solved":
        return (f"SOLVED. The score was at or above {config.solved_at:g} at the last two gradings "
                f"({known}). This model is big enough for this vocabulary; more training is "
                f"pointless.")
    if stop_reason == "plateau":
        text = (f"PLATEAU. Each fact has been seen {now['sightings']:,.0f} times on average, in "
                f"{now['steps']:,} training steps (a plateau is never called before "
                f"{config.min_sightings:,g} sightings and {config.min_steps:,} steps, nor while "
                f"fewer than {config.started_at:g} facts out of 100 are known). Over each "
                f"of the last two doublings of training the score gained fewer than "
                f"{config.flat_points:g} points, and over the last one the gap to the best "
                f"possible loss shrank by less than {config.flat_gap:.0%} or by less than "
                f"{config.flat_gap_bits:g} bits per word. The gain from more training has become "
                f"minimal: under this training recipe the model has hit its limit ({known}).")
        # The rule asks whether the score still GAINS. A score that falls passes that test too,
        # and a fall is not a limit of the model, so it is said here.
        best = max(point["facts_known"] for point in curve)
        if best - now["facts_known"] >= config.flat_points:
            text += (f" Note: earlier in this run the score had been higher ({best:.1f}), so "
                     f"part of this is a fall, not a ceiling: look at the learning curve.")
        return text
    if stop_reason == "fixed":
        return (f"FIXED LENGTH. The run was trained for exactly {config.fixed_sets:,} sets, as "
                f"asked with fixed_sets, and the stopping rule was not asked ({known}). This "
                f"shows where the model stands after that much training. It is not one of the "
                f"three endings and can never count as a plateau.")
    text = (f"BUDGET. The run reached its limit of {config.max_minutes:g} minutes or "
            f"{config.max_sets:,} sets before it was solved or flat ({known}). It is NOT FINISHED "
            f"and must not be read as a limit of the model.")
    if now["facts_known"] < config.started_at:
        # See started_at in RunConfig. This is the one case in which a longer budget may not
        # help, and the reader should know it before giving the run more minutes.
        text += (f" It had not begun to learn facts: fewer than {config.started_at:g} out of 100 "
                 f"were known, and there the stopping rule never calls a plateau. More time "
                 f"helps only if the model then begins to learn; a model that never begins ends "
                 f"by budget however long it trains.")
    return text


# ------------------------------------------------------------------------------------------------
# Building the pieces
# ------------------------------------------------------------------------------------------------


def build(config):
    """The language and a fresh, untrained model for one run. Returns (language, model).

    The invented animals' facts are decided by the run's seed. The model's word table has a row
    for every word of the language, which includes every animal of the table, in use or not.
    """
    language = Language(LanguageConfig(
        animals=config.animals, table_animals=config.table_animals, seasons=config.seasons,
        usage=config.usage, lexicon_seed=config.seed, set_sentences=config.set_sentences,
    ))
    model = TinyGPT(ModelConfig(
        vocab_size=language.vocab_size, width=config.width, layers=config.layers,
        heads=config.heads, context=config.context,
    ))
    return language, model


def evaluation_rows(language, config):
    """The rows on which loss, floor and gap are measured: the same at every grading of a run.

    They are drawn at random from the oracle with their own random generator, so they do not
    depend on the model and are not the training sets.
    """
    rng = np.random.default_rng([EVAL_STREAM, config.seed])
    return language.sample_rows(rng, EVAL_ROWS, config.context)


def load_run(run_dir):
    """Rebuild a finished run from run_dir/model.pt. Returns (config, language, model).

    The language is rebuilt with the automatic set size. The set size only decides how the
    training text is cut into sets, and nothing that is done with a trained model (asking it,
    grading it, looking at its word table) depends on that. So a run whose set size an older
    version of this code accepted, and this version refuses, can still be opened.
    """
    saved = torch.load(Path(run_dir) / "model.pt", weights_only=True)
    # Only the settings this RunConfig knows: a model saved by other code may hold another one.
    known = {field.name for field in fields(RunConfig)}
    config = RunConfig(**{key: value for key, value in saved["config"].items() if key in known})
    language, model = build(replace(config, set_sentences=0))
    model.load_state_dict(saved["model"])
    return config, language, model


# ------------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------------


def learning_rate_at(step, config):
    """The learning rate for step number `step` (0 is the first).

    It rises in a straight line during the warm-up and is CONSTANT afterwards. Many recipes
    lower the rate toward the end of training, but then the model at step 1,000 of a long run
    would differ from the end of a 1,000-step run. With a constant rate every grading along the
    way is a fair picture of a shorter training run.
    """
    if config.warmup <= 0:
        return config.learning_rate
    return config.learning_rate * min(1.0, (step + 1) / config.warmup)


def train_step(model, optimizer, inputs, targets, rate):
    """Adjust the parameters once, from one batch. Returns the loss on the batch (bits per word).

    inputs and targets are [rows, context]. The target at each position is the word that comes
    next; the loss (cross-entropy) is the model's average surprise at those words. Positions
    after a row's last sentence have the target IGNORE and are left out.
    """
    for group in optimizer.param_groups:
        group["lr"] = rate
    logits = model(inputs)  # [rows, context, vocab_size]: a score for every possible next word
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE)
    optimizer.zero_grad()
    # The gradient says, for every parameter, which way to move it to lower the loss.
    loss.backward()
    # If the gradient is very large, shrink it: one odd batch should not throw the model around.
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    return loss.item() / math.log(2)  # PyTorch counts in natural-log units; / ln 2 gives bits


def update_average(averaged, model, steps, config):
    """Move the averaged model a little toward the trained model (see average in RunConfig).

    steps is the number of training steps taken before this one. Early in a run the average is
    taken over the steps so far: keeping 0.998 of an average that still consists of the starting
    values would hold the graded model back for hundreds of steps.
    """
    keep = min(config.average, (1 + steps) / (10 + steps))
    with torch.no_grad():
        for mean, now in zip(averaged.parameters(), model.parameters()):
            mean.mul_(keep).add_(now, alpha=1 - keep)


def train_on_set(model, optimizer, inputs, targets, config, steps, averaged=None):
    """Train on every row of one set, exactly once, in the set's own order. Returns the step count.

    The last batch of a set may be smaller than the others: nothing is borrowed from the next
    set, so a grading at the end of a set has seen whole sets and nothing else. If an averaged
    model is given, it follows the trained model after every step (see update_average).
    """
    model.train()
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    for first in range(0, len(inputs), config.batch):
        rate = learning_rate_at(steps, config)
        loss = train_step(model, optimizer, inputs[first : first + config.batch],
                          targets[first : first + config.batch], rate)
        check_not_broken(loss, steps)
        if averaged is not None:
            update_average(averaged, model, steps, config)
        steps += 1
    return steps


def check_not_broken(loss, steps):
    """Stop with a plain message if training has broken down.

    With too large a learning rate the parameters can grow until the arithmetic overflows. The
    loss then turns into "nan" (not a number) and never recovers. That is a failed run, not one
    of the three endings, so it must not be saved as a result.
    """
    if not math.isfinite(loss):
        raise RuntimeError(
            f"Training broke down around step {steps:,}: the loss is no longer a number. "
            f"This is a failed run, not a result. Try a smaller --learning-rate."
        )


# ------------------------------------------------------------------------------------------------
# Printing
# ------------------------------------------------------------------------------------------------


def print_start(config, language, model, eval_rows):
    counts = count_parameters(model)
    bits = language.knowledge_bits()
    words_in_use = language.vocab_size - (config.table_animals - config.animals)
    floor = float(eval_rows[2][eval_rows[1] != IGNORE].mean())
    seen_per_set = language.sightings(1)[1]
    print(f"RUN {config.name}")
    print(f"  vocabulary in use   {config.animals:,} animal words ({words_in_use:,} words in all), "
          f"usage {config.usage}")
    print(f"  word table          {config.table_animals:,} animal rows ({language.vocab_size:,} "
          f"rows in all), the same at every vocabulary size")
    print(f"  knowledge           {bits:,.0f} bits of facts "
          f"({bits / config.animals:.2f} per animal)")
    print(f"  model               width {config.width}, {config.layers} layers, "
          f"{config.heads} heads")
    print(f"  parameters          body {counts['body']:,} + word table {counts['table']:,} "
          f"= {counts['total']:,}")
    print(f"  load                {bits / counts['total']:.4f} bits of facts per parameter "
          f"({bits / counts['body']:.4f} per body parameter)")
    print(f"  training text       sets of {language.set_sentences:,} sentences; one set shows "
          f"each fact {seen_per_set:,.1f} times on average")
    print(f"  best possible loss  {floor:.3f} bits per word on the {EVAL_ROWS} evaluation rows")
    if config.fixed_sets > 0:
        print(f"  length              exactly {config.fixed_sets:,} sets (fixed_sets): no "
              f"stopping rule and no budget, on {torch.get_num_threads()} threads")
    else:
        print(f"  budget              {config.max_minutes:g} minutes or {config.max_sets:,} sets, "
              f"on {torch.get_num_threads()} threads")
    print()
    print("  Graded at the end of a set. seen = times each fact was seen so far (on average);")
    print("  loss, best (possible) and gap = loss - best, in bits per word; facts and questions:")
    print("  0 = knows no animal, 100 = knows all; words = animals with every fact right.")
    by_use = f"{'by use':>8s}" if config.usage == "zipf" else ""
    print(f"{'sets':>6s} {'steps':>9s} {'minutes':>8s} {'seen':>10s} {'loss':>8s} {'best':>8s} "
          f"{'gap':>8s} {'facts':>8s}{by_use} {'questions':>10s} {'words':>12s}")


def print_grading(point, config):
    """One line of the table: where the model stands after this many sets."""
    by_use = f"{point['facts_known_by_use']:8.1f}" if config.usage == "zipf" else ""
    words = f"{point['words_known']:,}/{point['n_words']:,}"
    print(f"{point['sets']:6,d} {point['steps']:9,d} {point['minutes']:8.1f} "
          f"{point['sightings']:10,.1f} {point['loss']:8.3f} {point['floor']:8.3f} "
          f"{point['gap']:8.3f} {point['facts_known']:8.1f}{by_use} "
          f"{point['questions_right']:10.1f} {words:>12s}", flush=True)


def spread(items, how_many):
    """`how_many` items taken at even distances from a list (fewer if the list is short)."""
    places = np.linspace(0, len(items) - 1, min(how_many, len(items)))
    return [items[int(place)] for place in places]


def print_prompts(predict, language):
    """About a dozen prompts with the model's most likely next word beside the true one."""
    questions = language.question_probes("in_use")
    yes = [probe for probe in questions if probe.answer == language.yes_id]
    no = [probe for probe in questions if probe.answer == language.no_id]
    probes = spread(language.fact_probes("in_use"), 8) + spread(yes, 2) + spread(no, 2)
    said = most_likely_next(predict, probes)
    print(f"  {'prompt':38s} {'model says':12s} {'true':10s}")
    for probe, word in zip(probes, said):
        mark = "" if word == probe.answer else "   <- wrong"
        print(f"  {language.decode(probe.prefix):38s} {language.words[word]:12s} "
              f"{language.words[probe.answer]:10s}{mark}")


# ------------------------------------------------------------------------------------------------
# One run
# ------------------------------------------------------------------------------------------------


def grading_point(result, sets, steps, minutes, sightings, scheduled):
    """One entry of the learning curve: where training stands, plus every score.

    scheduled says whether this grading is one of grading_sets() (True) or the extra one at the
    set where a run was ended off the schedule (False). Other tools use it to tell the two apart.
    The per-animal list is left out here; it is kept only in the final result of a run.
    """
    point = {"sets": sets, "steps": steps, "minutes": minutes, "sightings": sightings,
             "scheduled": scheduled}
    for key, value in result.items():
        if key != "facts_right_by_animal":
            point[key] = value
    return point


def make_summary(config, language, model, stop_reason, curve, final):
    """Everything worth keeping about a finished run, as plain numbers and words."""
    last = curve[-1]
    return {
        "name": config.name,
        "config": asdict(config),
        "code": CODE_MARK,  # which version of the code made this run (see code_mark)
        "language": {
            "vocab_size": language.vocab_size,
            "animals": config.animals,
            "table_animals": config.table_animals,
            "seasons": config.seasons,
            "usage": config.usage,
            "set_sentences": language.set_sentences,
            "bits_total": final["bits_total"],
            "bits_per_animal": final["bits_total"] / config.animals,
            "floor": final["floor"],
        },
        "parameters": count_parameters(model),
        "stop_reason": stop_reason,
        "stop_explained": explain_stop(stop_reason, config, curve),
        "sets": last["sets"],
        "steps": last["steps"],
        "minutes": last["minutes"],
        "sightings": last["sightings"],
        "final": final,
        "curve": curve,
    }


def save_run(config, model, summary):
    """Write model.pt and summary.json to out_dir/name/. Returns the directory."""
    run_dir = Path(config.out_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "model": model.state_dict()}, run_dir / "model.pt")
    # The summary is written last: if summary.json exists, the run is complete.
    with open(run_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=1)
    return run_dir


def run(config):
    """Train one model until it is solved, flat or out of budget (or, with fixed_sets, for exactly
    that many sets). Returns the summary dictionary, which is also saved to
    out_dir/name/summary.json."""
    started = time.perf_counter()
    if config.name is None:
        config = replace(config, name=default_name(config))
    torch.set_num_threads(min(config.threads, os.cpu_count()))
    torch.manual_seed(config.seed)  # the model's starting values
    np.random.seed(config.seed)
    language, model = build(config)
    eval_rows = evaluation_rows(language, config)
    # AdamW is the usual rule for turning gradients into adjustments of the parameters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=WEIGHT_DECAY)
    # The model that is graded and saved: a running average of the trained model's parameters
    # (see average in RunConfig). With average 0 it is the trained model itself.
    graded = copy.deepcopy(model) if config.average > 0 else model
    predict = model_predictor(graded)
    print_start(config, language, model, eval_rows)

    curve, final, stop_reason, steps = [], None, None, 0
    schedule = grading_sets()
    next_grading = next(schedule)
    set_index = 0
    while stop_reason is None:
        # The text of a set depends on the seed and the set's number, never on the model.
        sentences = language.make_set(set_index, data_seed=config.seed)
        inputs, targets = language.pack_rows(sentences, config.context)
        steps = train_on_set(model, optimizer, inputs, targets, config, steps,
                             graded if config.average > 0 else None)
        set_index += 1  # = the number of whole sets trained on so far
        minutes = (time.perf_counter() - started) / 60
        scheduled = set_index == next_grading
        if config.fixed_sets > 0:  # a run of a fixed length: no stopping rule and no budget
            last_set = set_index >= config.fixed_sets
        else:
            last_set = minutes >= config.max_minutes or set_index >= config.max_sets
        # Graded at the scheduled set counts, and once more where the run has to end if that is
        # not one of them: the saved scores must belong to the saved model.
        if scheduled or last_set:
            final = score(predict, language, eval_rows)
            check_not_broken(final["loss"], steps)
            minutes = (time.perf_counter() - started) / 60  # grading takes time too
            sightings = language.sightings(set_index)[1]
            curve.append(grading_point(final, set_index, steps, minutes, sightings, scheduled))
            print_grading(curve[-1], config)
            if scheduled:
                next_grading = next(schedule)
            if config.fixed_sets > 0:
                stop_reason = "fixed" if last_set else None
            else:
                # At the extra grading of a run that is out of budget, decide_stop only asks
                # whether the run is solved; it never calls a plateau there.
                stop_reason = decide_stop(curve, config)
                if stop_reason is None and last_set:
                    stop_reason = "budget"

    summary = make_summary(config, language, model, stop_reason, curve, final)
    print()
    print(textwrap.fill("Ended by " + summary["stop_explained"], width=96))
    print()
    print_prompts(predict, language)
    run_dir = save_run(config, graded, summary)  # the saved scores belong to the graded model
    print()
    print(f"Saved {run_dir / 'summary.json'} and {run_dir / 'model.pt'}")
    return summary


# ------------------------------------------------------------------------------------------------
# Command-line flags (compare.py reuses them)
# ------------------------------------------------------------------------------------------------


def add_run_flags(parser):
    """One flag for every field of RunConfig, with the defaults written in RunConfig."""
    d = RunConfig()
    add = parser.add_argument
    add("--animals", type=int, default=d.animals,
        help="animal words in use: the vocabulary-size dial")
    add("--table-animals", type=int, default=d.table_animals,
        help="animal rows in the word table, the same for every model of a round")
    add("--seasons", type=int, default=d.seasons,
        help="1 = 8 bits per animal, 4 = 32 bits per animal")
    add("--usage", choices=["equal", "zipf"], default=d.usage,
        help="how use is spread over the animals")
    add("--set-sentences", type=int, default=d.set_sentences,
        help="sentences per training set; 0 = automatic")
    add("--width", type=int, default=d.width, help="numbers per word: the model-size dial")
    add("--layers", type=int, default=d.layers, help="blocks stacked on top of each other")
    add("--heads", type=int, default=d.heads, help="attention heads per block")
    add("--seed", type=int, default=d.seed,
        help="decides the model's starting values, the invented animals' facts and the shuffling")
    add("--batch", type=int, default=d.batch,
        help="rows read before each adjustment of the parameters")
    add("--context", type=int, default=d.context, help="words per row")
    add("--learning-rate", type=float, default=d.learning_rate, help="how big each adjustment is")
    add("--warmup", type=int, default=d.warmup,
        help="steps over which the learning rate rises to its full value")
    add("--average", type=float, default=d.average,
        help="the graded model is a running average of the trained model: the share of the old "
             "average kept at each step (0 = grade the trained model as it is)")
    add("--solved-at", type=float, default=d.solved_at, help="facts_known that counts as solved")
    add("--flat-points", type=float, default=d.flat_points,
        help="gaining fewer points than this over a doubling of training counts as flat")
    add("--flat-gap", type=float, default=d.flat_gap,
        help="the gap shrinking by less than this fraction over a doubling counts as flat")
    add("--flat-gap-bits", type=float, default=d.flat_gap_bits,
        help="the gap shrinking by fewer bits per word than this over a doubling counts as "
             "flat too")
    add("--min-sightings", type=float, default=d.min_sightings,
        help="never call a plateau before each fact was seen this often on average")
    add("--min-steps", type=int, default=d.min_steps,
        help="never call a plateau before this many training steps")
    add("--started-at", type=float, default=d.started_at,
        help="never call a plateau while facts_known is below this: learning has not begun")
    add("--max-minutes", type=float, default=d.max_minutes,
        help="budget: stop at the first set end after this much time")
    add("--max-sets", type=int, default=d.max_sets,
        help="budget: never train on more sets than this")
    add("--fixed-sets", type=int, default=d.fixed_sets,
        help="0 = off; above 0: train exactly this many sets and do not ask the stopping rule "
             "(for checking a result by training much longer)")
    add("--threads", type=int, default=d.threads, help="processor threads")
    add("--out-dir", default=d.out_dir, help="results go to OUT_DIR/NAME/")
    add("--name", default=d.name, help="name of the run; by default built from the settings")


def config_from_args(args):
    """The RunConfig that the flags of add_run_flags describe."""
    return RunConfig(
        animals=args.animals, table_animals=args.table_animals, seasons=args.seasons,
        usage=args.usage, set_sentences=args.set_sentences,
        width=args.width, layers=args.layers, heads=args.heads,
        seed=args.seed, batch=args.batch, context=args.context,
        learning_rate=args.learning_rate, warmup=args.warmup, average=args.average,
        solved_at=args.solved_at, flat_points=args.flat_points, flat_gap=args.flat_gap,
        flat_gap_bits=args.flat_gap_bits, min_sightings=args.min_sightings,
        min_steps=args.min_steps, started_at=args.started_at,
        max_minutes=args.max_minutes, max_sets=args.max_sets,
        fixed_sets=args.fixed_sets, threads=args.threads, out_dir=args.out_dir, name=args.name,
    )


def main():
    parser = argparse.ArgumentParser(description="Train one tiny GPT on one toy language.")
    add_run_flags(parser)
    config = config_from_args(parser.parse_args())
    try:
        run(config)
    except ValueError as problem:  # settings that make no sense: print the plain message and stop
        parser.error(str(problem))


if __name__ == "__main__":
    main()

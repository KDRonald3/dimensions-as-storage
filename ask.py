"""
ask.py - type a prompt, see what a trained model predicts next, beside the oracle.

What this file is for
---------------------
A language model does one thing: given some words, it says how likely every word of its
vocabulary is to come next. This tool lets you watch that directly. You type the start of a
sentence and it prints the model's most likely next words with their probabilities. Beside them
it prints the ORACLE's probabilities: the exact chances in the designed language (language.py),
which no model can beat. Where the two columns agree, the model has learned that piece.

How to run it
-------------
    python ask.py runs/a16_w32_equal_s0 "do foxes eat"       # one prompt
    python ask.py runs/a16_w32_equal_s0 "are bears brown ?"  # put a space before ? and .
    python ask.py runs/a16_w32_equal_s0                      # no prompt: asks you for prompts
    python ask.py runs/a16_w32_equal_s0 "foxes" --top 10     # show more words

The directory is one written by train.py. Put the prompt in quotes: a bare ? means something
else to the command line.

A prompt is read as the start of a row, so the model has nothing to copy an answer from: what it
says has to come from what it learned in training.
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

from score import model_predictor
from train import load_run

TOP_WORDS = 5  # how many of the model's most likely next words are shown


def say(text):
    """Print a paragraph, wrapped to the screen and indented like the rest of the output."""
    print(textwrap.fill(text, width=96, initial_indent="  ", subsequent_indent="  "))


def example_prompts(language):
    """Two prompts that are certainly valid in this language: the start of a statement and a
    whole question, both about the first animal. With more than one season they start with
    "in winter", as every sentence of such a language does."""
    statement = language.sentence_words(0, 0, 1)  # animal 0, first season, food: "foxes eat meat ."
    question = language.sentence_words(0, 0, 0, shown=0)  # color, asked: "are foxes red ? yes"
    return [" ".join(statement[:-2]), " ".join(question[:-1])]  # "foxes eat" and "are foxes red ?"


def model_probabilities(predict, ids):
    """The model's probability for every word of the vocabulary to come after the prompt."""
    # predict wants a batch of rows and answers at every position; we give it one row and keep
    # the answer after the last word. It returns logarithms of probabilities; exp undoes the log.
    log_probabilities = predict(np.array([ids]))[0, -1]
    return np.exp(log_probabilities.astype(np.float64))


def oracle_probabilities(language, ids):
    """The exact probability of every next word, or the reason why there is none.

    Returns (probabilities, None) or (None, a plain sentence). The oracle has no answer when no
    sentence of the language can start with the prompt.
    """
    try:
        return language.next_distribution(ids), None
    except ValueError as problem:
        not_in_use = set(language.table_animal_ids[language.config.animals :].tolist())
        never_read = [language.words[token] for token in ids if token in not_in_use]
        if never_read:
            return None, (
                f'The oracle has no answer: "{never_read[0]}" has a row in the word table, but '
                f"this run's text never used it (only the first {language.config.animals:,} "
                f"animals are in use). The model has never read this word, so what it says "
                f"about it is not knowledge."
            )
        return None, (f"The oracle has no answer: {problem} "
                      f'Try for example "{example_prompts(language)[0]}".')


def words_to_show(model, oracle, top):
    """The token ids to list: the model's `top` most likely words, and the oracle's `top` most
    likely words if they are not among them already. Ordered by the model's probability."""
    shown = np.argsort(-model, kind="stable")[:top].tolist()
    if oracle is not None:
        for token in np.argsort(-oracle, kind="stable")[:top].tolist():
            if oracle[token] > 0 and token not in shown:
                shown.append(token)
    return sorted(shown, key=lambda token: -model[token])


def reading(language, model, oracle):
    """One plain sentence on how to read the two columns for this prompt."""
    says = language.words[int(model.argmax())]
    if oracle is None:
        return f'The model\'s first choice is "{says}".'
    true = language.words[int(oracle.argmax())]
    if oracle.max() > 0.999:
        verdict = "The model's first choice is the same." if says == true else (
            f'The model\'s first choice is "{says}": wrong.')
        return f'The oracle is certain: the next word is "{true}". {verdict}'
    choices = int((oracle > 0).sum())
    return (f"The oracle is not certain here: {choices:,} different words can come next, and "
            f"which one comes is a free choice. The best any model can do is to give them the "
            f"oracle's probabilities.")


def answer(language, predict, context, text, top):
    """Print what comes after one prompt. Returns the rows printed, as a list of
    (word, model probability, oracle probability or None); empty if the prompt cannot be used."""
    try:
        ids = language.encode(text)
    except ValueError as problem:  # an unknown word: the message lists some valid words
        say(str(problem))
        return []
    if len(ids) == 0:
        say("Type at least one word: the model predicts what comes AFTER the words it is given.")
        return []
    if len(ids) > context:
        say(f"This prompt has {len(ids)} words; the model reads at most {context} in one row.")
        return []
    model = model_probabilities(predict, ids)
    oracle, why_not = oracle_probabilities(language, ids)
    shown = words_to_show(model, oracle, top)

    print(f'  After "{language.decode(ids)}" the next word is:')
    print(f"    {'next word':12s} {'model':>8s} {'oracle':>8s}")
    rows = []
    for token in shown:
        exact = None if oracle is None else float(oracle[token])
        rows.append((language.words[token], float(model[token]), exact))
        cell = "-" if exact is None else f"{100 * exact:.1f}%"
        print(f"    {language.words[token]:12s} {100 * model[token]:7.1f}% {cell:>8s}")
    # Whatever probability is left belongs to the words that are not listed.
    rest = "-" if oracle is None else f"{100 * max(0.0, 1 - oracle[shown].sum()):.1f}%"
    print(f"    {'(all others)':12s} {100 * max(0.0, 1 - model[shown].sum()):7.1f}% {rest:>8s}")
    if why_not is not None:
        say(why_not)
    say(reading(language, model, oracle))
    return rows


def ask_in_a_loop(language, predict, context, top):
    """Read prompts from the keyboard until an empty line."""
    examples = example_prompts(language)
    print(f'Type the start of a sentence and press Enter, for example "{examples[0]}" or '
          f'"{examples[1]}".')
    print('Put a space between all words, also before "?" and ".". An empty line ends.')
    while True:
        try:
            text = input("prompt> ")
        except (EOFError, KeyboardInterrupt):  # the input ended, or Ctrl-C was pressed
            print()
            return
        if text.strip() == "":
            return
        answer(language, predict, context, text, top)
        print()


def say_if_not_finished(run_dir):
    """One line if the run ended by budget: its answers are then no measure of the model.

    How a run ended is written in its summary.json (see train.py). Without that file nothing
    is said.
    """
    try:
        with open(Path(run_dir) / "summary.json") as file:
            ended = json.load(file)["stop_reason"]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return
    if ended == "budget":
        say("This run ended by budget: it is NOT FINISHED. The answers below show where the "
            "model stood when the time ran out, not what a model of this size can learn.")


def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="Show what a trained model predicts after a prompt, beside the oracle."
    )
    parser.add_argument("run_dir",
                        help="the directory of a finished run, for example runs/a16_w32_equal_s0")
    parser.add_argument("prompt", nargs="*",
                        help='the start of a sentence, for example "do foxes eat"; '
                             "leave it out to type prompts one after another")
    parser.add_argument("--top", type=int, default=TOP_WORDS,
                        help="how many of the model's most likely next words are shown")
    args = parser.parse_args(arguments)
    if args.top < 1:
        parser.error("--top must be at least 1: it is the number of words shown")
    if not (Path(args.run_dir) / "model.pt").exists():
        parser.error(f"there is no trained model in {args.run_dir} (no model.pt). Train one "
                     f"first, for example with: .venv/bin/python train.py")

    try:
        config, language, model = load_run(args.run_dir)
    except ValueError as problem:  # saved settings this code cannot rebuild a language from
        parser.error(f"the run in {args.run_dir} cannot be opened: {problem}")
    predict = model_predictor(model)
    print(f"Run {config.name}: {config.animals:,} animal words in use, width {config.width}.")
    say_if_not_finished(args.run_dir)
    print("model  = the probability this trained model gives each next word")
    print("oracle = the exact probability in the designed language; no model can do better")
    print()
    if args.prompt:
        answer(language, predict, config.context, " ".join(args.prompt), args.top)
    else:
        ask_in_a_loop(language, predict, config.context, args.top)


if __name__ == "__main__":
    main()

"""
score.py - grading a model against the oracle.

What this file is for
---------------------
A trained model is a pile of numbers. To say how much it knows we ask it things whose right
answers we know (the oracle in language.py knows them all) and count. This file is the measuring
instrument of the experiment. It trains nothing. The main function is

    score(predict, language, eval_rows)  ->  a dictionary of numbers

`predict` is anything that reads rows of words and says, at every position, how likely each word
of the vocabulary is to come next. A trained model is one such thing. So are the two fake models
in this file: the oracle (knows everything) and the blind model (knows no animal). They exist to
test the instrument itself: before trusting a score of 60 we check that a know-it-all gets 100
and a know-nothing gets 0.

How to run it
-------------
    python score.py                            # grade three predictors whose scores we know
    python score.py --usage zipf --seasons 4
    pytest tests/test_score.py                 # the same checks and more, as tests

Run directly, it grades the oracle, the blind model and a model that has not been trained.

Terms used in this file
-----------------------
probe            A short prompt with one right next word: "foxes eat" -> "meat". It starts at the
                 beginning of a row, so the answer cannot be copied from earlier text.
most likely      The one word to which a model gives the highest probability. A fact counts as
                 known only if the true value beats EVERY other word of the vocabulary.
chance           The score reachable without knowing any animal, by always naming the most
                 common value. Scores are corrected for it, so "knows nothing" reads as 0.
log-probability  The logarithm of a probability. Models compute with logarithms because tiny
                 probabilities stay manageable that way. log 1 = 0; the less likely, the more
                 negative.
loss             A predictor's average surprise at the word that really came next, in bits per
                 word: -log2 of the probability it gave that word. Probability 1 costs 0 bits,
                 1/2 costs 1 bit, 1/4 costs 2 bits.
floor            The oracle's loss on the same words: the best possible loss. It is not zero,
                 because some words are free choices (which animal comes next?). It is accurate
                 to about 0.002 bits per word at context 96: a model that keeps track of how much
                 room is left in a row can beat it by that much (see sample_rows in language.py).
gap              loss - floor. A gap within about 0.002 of zero, on either side, means "as good
                 as the oracle", at every vocabulary size. The raw loss cannot be compared between
                 vocabularies, and a gap above zero cannot either: with few animals a row often
                 repeats a fact, and copying it closes part of the gap without any knowledge.
                 Only facts_known can be compared between vocabulary sizes.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from language import IGNORE, PAD_ID, Language, LanguageConfig
from model import ModelConfig, TinyGPT

# These three numbers only affect speed, memory and the size of a check. They are not part of the
# experiment's design.
PROBE_BATCH = 512  # probes handed to the model in one go
ROW_BATCH = 32  # evaluation rows handed to the model in one go
UNSEEN_ANIMALS = 256  # at most this many never-read animals are probed for the leak check


# ------------------------------------------------------------------------------------------------
# Three kinds of predictor: a real model and two fake ones
# ------------------------------------------------------------------------------------------------


def model_predictor(model):
    """Wrap a TinyGPT so that it can be graded. Returns the function predict(tokens).

    predict takes token ids [batch, time] and returns natural-log probabilities
    [batch, time, vocab_size]: at every position, one number for every word that could come next.
    """

    def predict(tokens):
        model.eval()  # "we are grading now, not training" (this model behaves the same in both)
        with torch.no_grad():  # no learning here, so PyTorch need not keep notes for it
            logits = model(torch.as_tensor(np.asarray(tokens), dtype=torch.long))
            # The model gives every word a score (a logit). Softmax turns scores into
            # probabilities that sum to 1; log_softmax gives their logarithms directly.
            return F.log_softmax(logits, dim=-1).numpy()

    return predict


def fake_model(next_distribution, vocab_size):
    """Turn a function "words so far -> probability of every next word" into a predict function.

    Where that function raises ValueError (text that cannot occur in the language, such as the
    padding at the end of a row, or an animal that is not in use) the fake model has no idea and
    spreads its guess evenly over all words.
    """
    no_idea = np.full(vocab_size, 1.0 / vocab_size)

    def predict(tokens):
        tokens = np.asarray(tokens)
        probabilities = np.empty(tokens.shape + (vocab_size,))
        for row in range(tokens.shape[0]):
            lost = False  # True from the first impossible word on: what follows is lost as well
            for place in range(tokens.shape[1]):
                if not lost:
                    try:
                        words_so_far = tokens[row, : place + 1].tolist()
                        probabilities[row, place] = next_distribution(words_so_far)
                        continue
                    except ValueError:
                        lost = True
                probabilities[row, place] = no_idea
        with np.errstate(divide="ignore"):  # log(0) is minus infinity: "this word is impossible"
            return np.log(probabilities)

    return predict


def oracle_predictor(language):
    """A fake model that knows everything: it answers with the oracle's exact probabilities.

    It gets 100 on every score and a gap of 0. Slow (it lists every sentence of the language at
    every position), so it is for tests and for ask.py, not for training runs.
    """
    return fake_model(language.next_distribution, language.vocab_size)


def blind_predictor(language):
    """A fake model that knows the sentence forms and how common each value is among the animals
    in use, but nothing about any single animal. It must get 0 on every score that is corrected
    for chance: facts_known, facts_known_by_use, questions_right and bits_known.

    words_known is NOT corrected for chance, and there the blind model can read above 0: it
    names the most common value of every attribute, and an animal that happens to have all of
    those values is counted as known (a word or two with 16 animals, a handful with 1,024).
    Under zipf usage that is as a rule the most used animal, whose values are the most common
    ones by use.

    It works like Language.next_distribution (list every sentence that could be under way, weigh
    each by its designed probability, add up what each would say next) with one change: the
    animal word is not looked at. After "foxes eat" it therefore answers for an average animal:
    each food as often as it is eaten among the animals in use, counted by usage.
    """
    # Animal words have the highest token ids, so "the smaller of the word and the first animal"
    # writes every animal as the first animal. In this copy of the sentences all animals look
    # alike.
    first_animal = int(language.table_animal_ids[0])
    hidden = np.minimum(language.sentence_tokens, first_animal)
    whole = {tuple(row[:length]) for row, length in zip(hidden.tolist(), language.sentence_length)}

    def next_distribution(prefix):
        unfinished = []  # the words of the sentence under way; finished sentences are dropped
        for token in prefix:
            unfinished.append(min(int(token), first_animal))
            if tuple(unfinished) in whole:
                unfinished = []
        could_be = np.ones(language.n_sentences, dtype=bool)
        for place, token in enumerate(unfinished[: language.longest_sentence]):
            could_be &= hidden[:, place] == token
        if len(unfinished) >= language.longest_sentence or not could_be.any():
            raise ValueError("no sentence of this language starts like this.")
        # The next word is read from the REAL sentences: where an animal comes next, the blind
        # model still knows how often each animal is used.
        next_words = language.sentence_tokens[could_be, len(unfinished)]
        weights = language.sentence_probability[could_be]
        distribution = np.bincount(next_words, weights=weights, minlength=language.vocab_size)
        return distribution / distribution.sum()

    return fake_model(next_distribution, language.vocab_size)


# ------------------------------------------------------------------------------------------------
# Asking probes
# ------------------------------------------------------------------------------------------------


def most_likely_next(predict, probes):
    """The predictor's single most likely next word after each probe's prompt.

    The winner is taken over the WHOLE vocabulary, not just among the sensible answers.
    Returns an array of token ids, one per probe.
    """
    said = np.zeros(len(probes), dtype=np.int64)
    for first in range(0, len(probes), PROBE_BATCH):
        some = probes[first : first + PROBE_BATCH]
        # Prompts differ in length ("foxes eat", "foxes live in"). The shorter ones are filled up
        # at the end. That cannot change an answer: a model reads left to right, so what it
        # predicts after the last real word does not depend on anything that comes later.
        lengths = np.array([len(probe.prefix) for probe in some])
        tokens = np.full((len(some), lengths.max()), PAD_ID, dtype=np.int64)
        for row, probe in enumerate(some):
            tokens[row, : lengths[row]] = probe.prefix
        log_probabilities = predict(tokens)
        after_last_word = log_probabilities[np.arange(len(some)), lengths - 1]  # [probes, vocab]
        said[first : first + len(some)] = after_last_word.argmax(axis=1)
    return said


def grade_facts(predict, language, which, max_animals=None):
    """Ask one group of animals ("in_use" or "unseen") for every one of their facts.

    Returns a table right[animal, season, attribute] of True and False, with the group's first
    animal in row 0.
    """
    probes = language.fact_probes(which, max_animals)
    said = most_likely_next(predict, probes)
    n_animals = len(probes) // language.facts_per_animal
    first_animal = probes[0].animal
    right = np.zeros((n_animals, language.config.seasons, 4), dtype=bool)
    for probe, word in zip(probes, said):
        right[probe.animal - first_animal, probe.season, probe.attribute] = word == probe.answer
    return right


# ------------------------------------------------------------------------------------------------
# From right and wrong answers to scores
# ------------------------------------------------------------------------------------------------


def chance_accuracy(lexicon, weights):
    """The accuracy of knowing no animal and always naming the most common value.

    lexicon is [animals, seasons, 4], the true values of the animals being graded; weights is how
    much each animal counts (summing to 1). The level is computed from the table as it actually
    is: 1/4 for a balanced table, higher when some value is more common than the others.
    """
    seasons = lexicon.shape[1]
    total = 0.0
    for season in range(seasons):
        for attribute in range(4):
            true_values = lexicon[:, season, attribute]
            share_of_value = np.bincount(true_values, weights=weights, minlength=4)
            total += share_of_value.max()
    return total / (seasons * 4)


def facts_known(right, lexicon, weights):
    """The share of facts known, 0 to 100, corrected for chance.

    100 x (accuracy - chance) / (1 - chance): naming the most common value everywhere gives 0,
    knowing everything gives 100. Never below 0.
    """
    per_animal = right.mean(axis=(1, 2))  # the share of its facts each animal has right
    accuracy = float((weights * per_animal).sum())
    chance = chance_accuracy(lexicon, weights)
    if chance >= 1:  # every animal has the same facts: there is nothing to know
        return 0.0
    return float(np.clip(100 * (accuracy - chance) / (1 - chance), 0, 100))


def questions_right(predict, language):
    """How well yes/no questions are answered, 0 to 100, corrected for chance.

    Three of every four question probes have the answer "no", so a model that always says "no"
    is right 75% of the time while knowing nothing. BALANCED accuracy repairs that: it is the
    average of the accuracy on the yes questions and the accuracy on the no questions, which is
    one half for always-no, for always-yes and for a coin. The score is 200 x (balanced - 0.5).

    The balancing is done separately for each kind of question (season and shown value, such as
    "do ... eat meat ?") and then averaged over the kinds. With the balanced base table that
    gives exactly the plain balanced accuracy. If you edit the table so that most animals are,
    say, brown, then answering "yes" to every "are ... brown ?" would beat a coin without
    knowing any animal; balancing each kind separately keeps such a guesser at exactly 0.

    A kind of question with only one possible answer is not graded at all. (Edit the table so
    that no animal is black, and every "are ... black ?" has the answer "no": there is no yes
    to weigh against the no.) Wrong answers to such questions cost nothing.
    """
    probes = language.question_probes("in_use")
    said = most_likely_next(predict, probes)
    answer = np.array([probe.answer for probe in probes])
    right = said == answer
    is_yes = answer == language.yes_id
    # prefix[-2] is the value shown in the question (the word before "?"). A value word belongs
    # to one attribute only, so season and shown value together name the kind of question.
    kind = np.array([probe.season * language.vocab_size + probe.prefix[-2] for probe in probes])
    _, kind = np.unique(kind, return_inverse=True)  # renumber the kinds 0, 1, 2, ...
    n_yes = np.bincount(kind, weights=is_yes)
    n_no = np.bincount(kind, weights=~is_yes)
    yes_right = np.bincount(kind, weights=is_yes & right)
    no_right = np.bincount(kind, weights=~is_yes & right)
    both = (n_yes > 0) & (n_no > 0)  # a kind with only one answer has nothing to tell apart
    if not both.any():
        return 0.0
    balanced = (yes_right[both] / n_yes[both] + no_right[both] / n_no[both]) / 2
    return float(np.clip(200 * (balanced.mean() - 0.5), 0, 100))


def binary_entropy(share):
    """The bits in one yes/no outcome that is "yes" with this probability (1 bit at one half)."""
    if share <= 0 or share >= 1:
        return 0.0
    return float(-share * np.log2(share) - (1 - share) * np.log2(1 - share))


def bits_known(right, lexicon):
    """A LOWER BOUND on the bits of facts a model must be holding to be this accurate.

    This is not simply accuracy x bits, because accuracy can be bought with fewer bits than one
    might think. Take a fact with four equally likely values (2 bits). To be right half the time
    one need not hold half of those 2 bits: about 0.21 bits per fact can be enough, if the
    knowledge is stored cleverly across many facts at once. The price list is known as Fano's
    inequality: to be right a share `a` of the time on a fact with k values that is worth
    `entropy` bits, one must hold at least
        entropy - h(a) - (1 - a) x log2(k - 1)      bits per fact,
    where h is the binary entropy. At a = 1 that is all of the fact's bits; at chance it is 0.
    The bound is worked out for each season and attribute from that attribute's actual value
    frequencies and its accuracy (not corrected for chance), and added up over the facts.
    A model may well hold more than this bound; it cannot hold less.

    This is the entropy-based form of the bound: it starts from what a fact is really worth. On
    a balanced table (every value equally common, 2 bits per fact) it is identical to the form
    with log2(k) in place of the entropy. On an edited, unbalanced table only the entropy-based
    form reads 0 for a model that just names the most common value, so it is the one used.
    """
    animals = len(lexicon)
    k = 4  # the number of values an attribute can have
    total = 0.0
    for season in range(lexicon.shape[1]):
        for attribute in range(4):
            accuracy = right[:, season, attribute].mean()
            if accuracy <= 1 / k:  # no better than a blind guess among k values: no knowledge shown
                continue
            counts = np.bincount(lexicon[:, season, attribute], minlength=k)
            share = counts[counts > 0] / animals
            entropy = -(share * np.log2(share)).sum()  # what one fact of this attribute is worth
            per_fact = entropy - binary_entropy(accuracy) - (1 - accuracy) * math.log2(k - 1)
            total += animals * max(0.0, per_fact)
    return float(total)


def loss_and_floor(predict, eval_rows):
    """The predictor's loss and the best possible loss on the evaluation rows, in bits per word.

    Only real target words count; the padding at the end of a row is skipped.
    """
    inputs, targets, bits, _ = eval_rows
    graded = targets != IGNORE
    surprise = 0.0  # added up in natural-log units ("nats")
    for first in range(0, len(inputs), ROW_BATCH):
        log_probabilities = predict(inputs[first : first + ROW_BATCH])
        target = targets[first : first + ROW_BATCH]
        real = graded[first : first + ROW_BATCH]
        # Pick out, at every position, the log-probability given to the word that really came.
        which_word = np.where(real, target, 0)[:, :, None]
        of_target = np.take_along_axis(log_probabilities, which_word, axis=2)[:, :, 0]
        surprise -= float(of_target[real].sum(dtype=np.float64))
    loss = surprise / graded.sum() / math.log(2)  # dividing by ln 2 turns nats into bits
    floor = float(bits[graded].mean(dtype=np.float64))  # the oracle's surprise at the same words
    return loss, floor


# ------------------------------------------------------------------------------------------------
# The grade
# ------------------------------------------------------------------------------------------------


def score(predict, language, eval_rows):
    """Grade a predictor. eval_rows is the tuple from language.sample_rows, fixed for a run.

    Returns a dictionary:
      facts_known            0 to 100, the headline: every fact of the animals in use counts once
      facts_known_by_use     the same, with every animal weighted by how often it is used
      questions_right        0 to 100: yes/no questions, balanced accuracy. It is balanced
                             separately for each kind of question, so that a guesser reads 0;
                             kinds with only one possible answer are not graded
      unseen_known           facts_known for animals that are in the word table but were never
                             read about. It should be near 0. More would mean knowledge leaks in
                             from somewhere other than the training text. None if there are none.
      words_known, n_words   animals with every fact right, out of how many animals. NOT corrected
                             for chance: a model that knows nothing can have a word or two right
                             (a handful of 1,024; under zipf usage as a rule the most used animal)
      facts_right_by_animal  for each animal in use, how many of its facts are right
      loss, floor, gap       bits per word on the evaluation rows; gap = loss - floor. The floor
                             can be beaten by about 0.002 at context 96, so a gap within that of
                             zero means "as good as the oracle"
      bits_total, bits_known the bits of facts in the language, and a lower bound on the bits held
                             (the entropy-based Fano bound, see bits_known)
      headline, headline_name
    """
    config = language.config
    lexicon = language.lexicon[: config.animals]
    all_alike = np.full(config.animals, 1 / config.animals)  # every animal counts the same

    right = grade_facts(predict, language, "in_use")
    right_by_animal = right.sum(axis=(1, 2))
    headline = facts_known(right, lexicon, all_alike)

    unseen_known = None
    if config.table_animals > config.animals:
        unseen_right = grade_facts(predict, language, "unseen", UNSEEN_ANIMALS)
        n_unseen = len(unseen_right)
        unseen_lexicon = language.lexicon[config.animals : config.animals + n_unseen]
        unseen_known = facts_known(unseen_right, unseen_lexicon, np.full(n_unseen, 1 / n_unseen))

    loss, floor = loss_and_floor(predict, eval_rows)
    return {
        "facts_known": headline,
        "facts_known_by_use": facts_known(right, lexicon, language.usage),
        "questions_right": questions_right(predict, language),
        "unseen_known": unseen_known,
        "words_known": int((right_by_animal == language.facts_per_animal).sum()),
        "n_words": int(config.animals),
        "facts_right_by_animal": right_by_animal.tolist(),
        "loss": loss,
        "floor": floor,
        "gap": loss - floor,
        "bits_total": language.knowledge_bits(),
        "bits_known": bits_known(right, lexicon),
        "headline": headline,
        "headline_name": "facts_known",
    }


# ------------------------------------------------------------------------------------------------
# Running this file: the instrument checked on three predictors whose scores we know in advance
# ------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Grade the oracle, the blind model and an untrained model. Trains nothing."
    )
    parser.add_argument("--animals", type=int, default=16, help="animal words in use")
    parser.add_argument("--table-animals", type=int, default=32,
                        help="animal rows in the word table (small here: the oracle is slow)")
    parser.add_argument("--seasons", type=int, default=1,
                        help="1 = 8 bits per animal, 4 = 32 bits per animal")
    parser.add_argument("--usage", choices=["equal", "zipf"], default="equal",
                        help="how use is spread over the animals")
    parser.add_argument("--seed", type=int, default=0,
                        help="decides the invented animals' facts, the rows, the untrained model")
    parser.add_argument("--rows", type=int, default=16, help="evaluation rows")
    parser.add_argument("--context", type=int, default=48, help="words per row")
    parser.add_argument("--width", type=int, default=32, help="width of the untrained model")
    args = parser.parse_args()

    language = Language(LanguageConfig(args.animals, args.table_animals, args.seasons, args.usage,
                                       lexicon_seed=args.seed))
    eval_rows = language.sample_rows(args.seed, args.rows, args.context)
    torch.manual_seed(args.seed)
    untrained = TinyGPT(ModelConfig(language.vocab_size, width=args.width, context=args.context))
    results = {
        "oracle": score(oracle_predictor(language), language, eval_rows),
        "blind": score(blind_predictor(language), language, eval_rows),
        "untrained": score(model_predictor(untrained), language, eval_rows),
    }

    print(f"{args.animals} animals in use, {language.vocab_size} rows in the word table, "
          f"{args.rows} evaluation rows of {args.context} words")
    print(f"  {'':20s}" + "".join(f"{name:>12s}" for name in results))
    keys = ["facts_known", "facts_known_by_use", "questions_right", "unseen_known", "words_known",
            "loss", "floor", "gap", "bits_known", "bits_total"]
    for key in keys:
        cells = ""
        for result in results.values():
            cells += f"{'none':>12s}" if result[key] is None else f"{result[key]:12.3f}"
        print(f"  {key:20s}{cells}")
    print("The oracle knows every fact; the blind model knows the sentence forms and how common")
    print("each value is, but no single animal; the untrained model has not read anything yet.")
    print("words_known is not corrected for chance: the blind model can have a word right by luck.")


if __name__ == "__main__":
    main()

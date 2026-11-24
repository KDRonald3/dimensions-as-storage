<img src="assets/icon.svg" width="64" height="64" alt="">

# air: does a bigger vocabulary need a bigger model?

This repository tests one hypothesis: *a language model is a high-dimensional representation of
the meaning of words, so a tiny vocabulary should need only a tiny model, and a large vocabulary
with the same model size should perform worse.* "Perform" has one narrow meaning here:
**recalling facts about words**. The model reads `foxes eat` and has to continue with `meat`.
Fluency and reasoning are not measured.

> **Status (2026-11-22): the experiment was run on the night of 21 to 22 November 2026. The
> results are in [`report/report.pdf`](report/report.pdf).** Read the report for what was found,
> what was changed along the way and what is left to do; this README describes the programs.
> In one line: by the decision rule fixed in advance the outcome is *undecided*. Every model of
> width 8 or more learned every fact of every language with 8 bits per animal, and the round
> with 32 bits per animal showed that a run can stand still for hundreds of sets and then go
> on, so the "plateau" ending of the stopping rule cannot be trusted as a limit.
> The one lead for the hypothesis: width 8 did not learn 256 animals at 32 bits each in three
> runs, while widths 16 and 32 did.
>
> Where the runs are (none of these folders is in git): the planned rounds in `runs/`, the long
> runs of the 32-bit round in `runs_long/` (trained with `--started-at 101`, which switches the
> plateau ending off), and small side checks in `runs_extra/`. `report/make_assets.py` rebuilds
> every table and chart of the report from them; `tectonic report.tex`, run inside `report/`,
> rebuilds the PDF. Every run was made with a copy of the programs set aside before the first
> run; `train.py`, `language.py`, `model.py` and `score.py` here train and grade exactly as
> that copy does (`logs/frozen_code.sha256`).
>
> Before any run of the experiment, short calibration runs were made to set three defaults of
> the training recipe: the learning rate (now 0.01), the running average of the model that is
> graded (`average`), and the rule that no plateau is called before a model has begun to learn
> facts (`started_at`). What those runs showed is written in the comments above these three
> settings in `train.py`; rule 6 below describes them. The thresholds of the decision rule were
> not touched.

Commands are written as `.venv/bin/python train.py` and are run from this folder. That is the
interpreter everything here was tested with. `uv run --no-sync python train.py` is the same thing.
Do not use plain `uv run`: it was never tried here, and it would first work out the package
versions afresh, because `pyproject.toml` pins none of them. The text at the top of each file
writes its examples as `python train.py` for short: type `.venv/bin/python train.py`. On this
machine plain `python` is another interpreter, without PyTorch.

## Changes made after the code review

The code was reviewed before any experiment, and every finding was checked by a second reviewer.
Six things changed. The most important is A: before it, a run on a small word table could be
saved as `plateau` after 32 training steps, before the model had begun to learn.

- **A. The stopping rule was made stricter against a plateau that is called too early**: never
  before 2,000 training steps (`min_steps`). (That alone turned out not to be enough: see G.)
  Also, a fall of the gap by less than 0.002 bits per word is noise, not learning
  (`flat_gap_bits`); a run out of budget is still asked whether it is solved; and `--fixed-sets`
  trains without the rule (rule 6 below).
- **B. The `table-check` round** gives both halves the same sets, so only the word table differs.
- **C. The model's body gets its starting values before the word table**, so the same seed gives
  the same body whatever the size of the table.
- **D. A set size that does not suit the number of animals is refused**, with a plain message.
- **E. The grader's descriptions were corrected and its tests extended**: what is corrected for
  chance and what is not, and tests that fail if the leak check or the headline score breaks.
- **F. Only facts known is compared between vocabulary sizes**; the gap only at 0.

Then the recipe was calibrated (see the status above), and the whole project was checked a second
time. What changed then:

- **G. Set in calibration:** the learning rate is 0.01; the model that is graded and saved is a
  running average of the trained model (`average`); no plateau is called while a model knows
  fewer than 5 facts out of 100 (`started_at`). Rule 6 explains all three.
- **H. The decision rule is applied only where it can be trusted.** A group of runs that were not
  all trained the same way, or not under the agreed stopping rule, gets no verdict. "The next
  width up" is twice the width and nothing else. Of two runs at the same place, which one counts
  no longer depends on their names. See "How the code applies it" below.
- **I. A run of a fixed length is marked as such everywhere** (a half-filled dot, `fixed` in the
  tables), and a run that ran out of budget before it began to learn facts gets a sentence of
  its own: more minutes may not help it.
- **J. The report tells more:** every larger vocabulary with a drop, not just one per seed; how
  many runs are behind a mean; and in chart 1 every run has a place of its own, so that equal
  scores no longer hide each other.
- **K. `sweep.py`** can train part of a round (`--widths`, `--seeds`, `--animals`), goes on
  after any error in one run, and with `--jobs` above 1 no longer leaves workers behind when it
  is killed or waits for ever when a worker dies.
- **L. `compare.py` no longer replaces a saved run** without being asked (`--again`).
- **M. `probe.py` prints how much of its numbers is luck**, and says so when a run was not
  finished or used zipf.
- **N. Every run is saved with a mark of the code that made it** (`code` in `summary.json`).

## The idea in five sentences

1. We invent a small language about animals in which we know exactly how much there is to know
   about every word: four facts per animal, worth 8 bits.
2. A bigger vocabulary is more animal words, each carrying the same 8 bits, and nothing else
   changes.
3. A tiny GPT of one fixed size is trained on 16, 64, 256 and 1,024 animal words, and is then
   asked for every fact.
4. This is repeated for four model sizes (widths 4, 8, 16 and 32), three times each.
5. The hypothesis predicts that a model which learns the 16-animal language gets stuck at a lower
   score on a larger one, and that the next model size up, reading exactly the same text, does
   clearly better there.

## The language

The base language has 43 words. One word is one **token**: the model never sees letters, only a
whole number for each word.

```
sentence words (11): . ? are do eat live in wake at yes no
color (4): red brown green black
food (4): meat plants insects seeds
home (4): woods fields rivers caves
waking time (4): dawn noon dusk night
animals (16): foxes bears wolves owls bats frogs snakes crows hens ducks deer mice toads robins otters lizards
```

Every animal has four **facts**: a color, a food, a home and a waking time. Each fact is stated
and asked about in a fixed form:

```
foxes are red .            are foxes red ? yes
foxes eat meat .           do foxes eat meat ? yes
foxes live in woods .      do foxes live in woods ? yes
foxes wake at dusk .       do foxes wake at dusk ? yes
```

A fact has five distinct sentences. Of every 15 training sentences about it, 9 are the statement,
3 are the question that shows the true value, and 3 are questions that show a wrong value, one
for each:

```
foxes eat meat .             statement                       9 of 15
do foxes eat meat ? yes      question, true value            3 of 15
do foxes eat plants ? no     question, a wrong value         1 of 15 for each of the three wrong values
```

The facts of the 16 base animals are a hand-written table at the top of `language.py`
(`BASE_ANIMALS`). In it every value is used by exactly four animals. Animals beyond the sixteenth
are invented: they get made-up names (`bakos`, `gigas`, `vofos`, ...) and random facts, dealt in
blocks of 16 animals so that every value stays equally common.

The language comes with an **oracle**: the program that wrote the text. It knows every right
answer and the exact probability of every next word, so a model can be graded against it.

## The model

`model.py` is a small GPT with 2 layers. It reads a row of words and, at every position, guesses
the next word. Its **parameters** are the numbers that training adjusts. They come in two parts:

- the **word table**: one row of numbers for every word. The same table is used to read a word
  and to score it as the next word (this is called a tied table);
- the **body**: everything else.

The **width** is how many numbers describe one word. It is the model-size dial of the experiment.

## What is measured

- **facts known**: the headline, 0 to 100. For every fact the model gets the start of its
  statement (`foxes eat`) at the beginning of a row, so there is nothing to copy from. The fact
  counts only if the true value is the model's single most likely next word among all the words
  it has. The share is rescaled so that always naming the most common value reads 0 and knowing
  everything reads 100.
- **by use**: facts known with every animal weighted by how often it is used. Shown for zipf
  usage only.
- **questions right**: 0 to 100, on prompts like `do foxes eat seeds ?`. It is balanced, so
  always answering `no` scores 0. The balancing is done separately for each kind of question
  (`do ... eat seeds ?` is one kind), so that a guesser reads 0 whatever the table looks like.
- **words known**: the animals with every one of their facts right. This number is not corrected
  for chance: a model that knows nothing can have a word or two right, or a handful of 1,024
  (under zipf usage as a rule the most used animal, whose values are the most common ones).
- **loss**: the model's average surprise at the next word, in bits per word. Lower is better.
- **best possible**: the oracle's loss on the same words. It is not 0, because nobody can know
  which animal the next sentence is about.
- **gap**: loss minus best possible. 0 means the model predicts as well as anyone could. The best
  possible loss is accurate to about 0.002 bits per word, so a gap that close to 0, on either
  side, means "as good as the oracle". The gap measures prediction, not only memory: inside a row
  a model can also copy a fact that was stated earlier. **Between vocabulary sizes only facts
  known can be compared. The gap can be compared only at 0** (see "What is NOT a size effect").
- **times each fact was seen**: training sentences about one fact so far, on average over all
  facts.
- **unseen known**: facts known for animals that are in the word table but never appeared in the
  text. It should be near 0: a check that no knowledge leaks in from elsewhere. (In
  `summary.json` and `summary.csv`.)
- **bits known**: a lower bound on the bits of facts the model must be holding to be as accurate
  as it is. (In `summary.json` and `summary.csv`.)

## The rules of the experiment

These were agreed before anything was run. Each has a reason.

1. **Knowledge per word is held constant.** Every animal carries four facts, each one of four
   equally common values. Choosing one of four takes two yes/no questions, so a fact is worth
   2 bits and an animal 8 bits, at every vocabulary size.
   *Why:* if words in a bigger vocabulary also had more or less to know about them, a lower score
   could not be blamed on the number of words.

2. **A bigger vocabulary is more animal words and nothing else.** The sentence forms, the 27
   other words and the possible answers stay the same. The vocabularies are nested: the 16
   animals are the first 16 of the 64, with the same facts, and so on.
   *Why:* only one dial moves, so a difference between two runs can only come from the number of
   animal words.

3. **Every model shares one word table.** Normally a bigger vocabulary means more rows in the
   word table, and so a bigger model. Here every model has a row for each of the 1,024 animals
   (1,051 rows in all), whatever vocabulary it is trained on. A model trained on 16 animals simply
   never reads about the other 1,008.
   *Why:* "the same model size" is then literally true. A model of a given width has exactly the
   same number of parameters at every vocabulary size.

4. **Training text comes in exact sets, and a run is graded and can stop only at the end of a
   set.** A set is 61,440 sentences (15 x 4 facts x 1,024 animals) with exact, not random, counts
   of every animal, fact and sentence form, shuffled afresh every time. The sets depend on the
   **seed** (the number that decides everything random in a run), never on the model.
   *Why:* no model is lucky or unlucky in what it happened to read. Two models of different width
   with the same seed read exactly the same sentences in the same order, so only the model
   differs.

5. **Use is spread over the animals in two ways, run separately.** `equal`: every animal is used
   equally often. `zipf`: the second animal is used half as often as the first, the third a third
   as often, and so on. New animals join at the rare end, so the small vocabulary is exactly the
   most common words of the large one.
   *Why:* equal use is the clean case. Real languages look more like zipf, and words getting
   rarer is part of what a bigger vocabulary is, so it is measured and not removed.

6. **A run ends in one of three ways, and the ending is saved with the result.**
   - `solved`: facts known was 97 or more at the last two gradings. The model is big enough.
   - `plateau`: more training has stopped helping. All of this must hold: each fact has been seen
     at least 400 times on average (`min_sightings`); the model has taken at least 2,000 training
     steps (`min_steps`); the model knows at least 5 facts out of 100 (`started_at`); facts known
     gained fewer than 2 points over each of the last two doublings of training (`flat_points`);
     and over the last doubling the gap shrank by less than 5% or by less than 0.002 bits per
     word (`flat_gap`, `flat_gap_bits`).
   - `budget`: 20 minutes (or 100,000 sets) ran out first. The run is **not finished**.

   *Why:* a low score says something about a model's limit only if the model had stopped
   improving. A run that ran out of time shows where it stood, not what it can do.
   *Why the steps and the bits:* early in training the score stands at 0 and the gap hardly
   moves, which looks just like a plateau. `min_steps` keeps a run from being called flat during
   the warm-up or before the model has begun to learn; it makes the rule stricter, never looser.
   And the best possible loss is only accurate to about 0.002 bits per word, so a smaller
   movement of the gap is noise and does not count as "still shrinking".
   *Why the 5 facts:* a model first learns the sentence forms, then stands still for a while with
   no facts known, and only then starts on the facts. In calibration the length of that
   stand-still varied so much with width and vocabulary that no fixed number of steps was safe,
   and one run was called a plateau with 0.4 facts known. So a run that has not begun to learn
   facts is never called a plateau.
   **What follows from it:** a model that never begins (one that is simply too small for a
   vocabulary, say) can only end by `budget`, however long it trains. It is reported as not
   finished and can never count for (b) of the decision rule. `train.py` and `report.py` say so
   in a sentence of their own for such a run, because "give it more minutes" may not help it.
   How such a case is to be read (for example by a `--fixed-sets` check of a stated length) has
   not been agreed yet. It should be, before results are read.

   Two more parts of the recipe were set in calibration, the same for every run. The **learning
   rate** is 0.01 (`learning_rate`): the comment in `train.py` gives the reason. And the model
   that is graded and saved is not the model of the last training step but a **running average**
   of its parameters over roughly the last 500 steps (`average`, 0.998). With a constant learning
   rate the parameters never settle but keep wobbling, and the score wobbled with them from one
   grading to the next; the average removes that and changes nothing about the training itself.
   So `ask.py` and `probe.py` open the averaged model. `--average 0` grades the model as it is.

7. **A size effect needs a plateau and a rescue.** A lower score on a larger vocabulary counts as
   an effect of model size only if the run ended by plateau and the next width up, trained on the
   same sets, does clearly better.
   *Why:* a model can score lower for reasons that have nothing to do with its size, such as too
   little practice or a training recipe that does not suit it. If a bigger model on the same text
   fixes the drop, size was what was missing. If not, it was something else.

Three details of rules 4 and 6. A model is graded at set counts 1, 2, 3, 4, 6, 8, 11, 16, 23, 32,
45, 64, ..., each about 1.41 times the last. A run that runs out of budget between two of them
is graded once more where it stops; at that extra grading only `solved` is asked (the last two
gradings at 97 or more), never a plateau, and in the saved learning curve it is marked
`"scheduled": false`. And under zipf usage a set can be a little longer or shorter than 61,440,
because counts are rounded to whole sentences: by about 0.1% at most. That is a measurement, not
a limit: with 1,024 animals the sets were within 37 sentences of 61,440 over the first 300 sets
and within 54 over 20,000 sets; with fewer animals they are closer.

**A fourth kind of run, for checking.** `--fixed-sets 500` switches the stopping rule and the
budget off: the run trains for exactly 500 sets, is graded at the scheduled set counts and at the
last set, and is saved with the ending `fixed` under a name of its own
(`a1024_w4_equal_s0_f500`), so it never replaces the ordinary run. It is for checking a result
by training much longer than the rule would. A `fixed` run can never count as a plateau.

## The arithmetic of load

**Load** is the bits of facts in the language divided by the parameters of the model. These
numbers are for 1,024 animals. They come from building every model and language with the code
(`count_parameters` in `model.py`, `knowledge_bits` in `language.py`), not from training.

| width | body | word table | total parameters | bits per parameter at 8 bits per animal (8,192 bits) | bits per parameter at 32 bits per animal (32,768 bits) |
|---:|---:|---:|---:|---:|---:|
| 4 | 880 | 4,204 | 5,084 | 1.61 | 6.43 |
| 8 | 2,528 | 8,408 | 10,936 | 0.75 | 2.99 |
| 16 | 8,128 | 16,816 | 24,944 | 0.33 | 1.31 |
| 32 | 28,544 | 33,632 | 62,176 | 0.13 | 0.53 |
| 64 | 106,240 | 67,264 | 173,504 | 0.047 | 0.19 |

The parameter counts are for the 8-bit language (1,051 words). The 32-bit language has four more
words, the seasons, so its word table is 4 x width parameters larger; the last column uses those
totals. Round 1 uses widths 4 to 32, the scale-up round widths 8 to 64. With 16 animals there are
only 128 bits of facts, so every load is 64 times smaller (0.025 at width 4).

Is 1.6 bits per parameter a lot? Nobody knows in advance. A parameter is stored as a 32-bit
number, but how many bits of facts a trained parameter can hold in this setting is exactly what
is not known. That is why the decision rule below has an outcome for "the models were not
stretched".

**Why a separate 32-bit round exists.** In round 1 the number of words and the amount of
knowledge always rise together: four times the animals is four times the bits. So round 1 alone
cannot tell whether a limit is about the number of words or about the total amount to know. The
`scaleup` round separates the two. It gives every animal the same four facts in each of four
seasons (`in winter foxes eat meat .`), which is 32 bits per animal, again identical at every
vocabulary size. Now 256 animals hold exactly as many bits (8,192) as 1,024 animals did in
round 1. Chart 3 puts both rounds on one axis, bits of facts per parameter, and asks whether the
runs fall on one curve. In the agreed plan this round comes after round 1, and only if the rule
is met there.

## The decision rule, fixed before any result

The rule lives at the top of `report.py` as constants: `HIGH = 95`, `DROP = 10`, `RESCUE = 10`,
`SEEDS_NEEDED = 2`, `SMALL = 16`, and `WIDTH_STEP = 2` for "the next width up". They are
deliberately not command-line flags: a rule that is adjusted after the results are in proves
nothing.

For one width, one usage and one number of seasons, the hypothesis **clearly holds** when, in at
least 2 seeds, all three are true:

- **(a)** the width scores 95 or more (facts known) on the 16-animal language;
- **(b)** on some larger vocabulary its run **ended by plateau** and scored at least 10 points
  below its own 16-animal score with the same seed;
- **(c)** on that same vocabulary the next width up, with the same seed and so the same training
  sets, scores at least 10 points higher.

How the code applies it:

- (b) and (c) must hold on the same vocabulary, within one seed. "At least" includes the
  threshold itself.
- A run that ended by budget is not finished and can never satisfy (b). It can count for (a), or
  as the rescuer in (c), if its score is already high enough. The printed sentence then says
  `(that run was NOT FINISHED)`.
- A run that ended by budget with fewer than 5 facts out of 100 known had not begun to learn
  facts (rule 6, `started_at`). It cannot satisfy (b), and a longer budget changes that only if
  the model then begins. The report says so instead of "train it longer".
- A run trained with `--fixed-sets` (ending `fixed`) never asked the stopping rule, so it can
  never satisfy (b) either. As a rescuer it counts if it scores high enough; if it does not,
  nobody knows whether it would have, and the report says the drop cannot be judged.
- "The next width up" is twice the width: 8 for width 4, 16 for width 8, and so on
  (`WIDTH_STEP = 2` in `report.py`). If there is no run of that width, (c) cannot be tested; a
  still larger width does not stand in for it. So in the `quick` round, which has only widths 4
  and 32, (c) is never tested, and the rule can never be met for the largest width of a round.
- **One run per place.** If two runs have the same vocabulary, width and seed, one counts: an
  ordinary run before a run of a fixed length, and then the run that lies in the folder
  `train.py` gives it by default before a run in any other folder. The run left out is listed
  with its score and its ending. If nothing says which of two runs counts (two names of your
  own), neither is used.
- **The rule takes two things for granted, and checks them.** Every run of a group must have
  been trained with the same recipe (every setting except the vocabulary, the width, the seed
  and the length of the run), and under the agreed stopping rule, which is the stopping rule
  with the default settings of `train.py`. A group that fails gets no verdict: the report prints
  `CAREFUL`, names the runs, and says "The rule is NOT applied to this group". Train those runs
  again or move them out of the folder.
- If no width drops by 10 points although some width reached 95 on 16 animals, the report says:
  "The models were not stretched: this is neither support for the hypothesis nor a refutation of
  it." If no width reached 95 either, it says that nothing was tested; and if no width has both
  a 16-animal run and a larger one, that nothing could be compared.
- If a drop is not fixed by the next width up, it says: "that drop is not a size effect." If the
  next width up was itself not finished, it says the drop cannot be judged yet.
- **Every drop is reported.** A seed's sentence is about one larger vocabulary (the first on
  which (b) and (c) hold, else the first with a drop after a plateau, else the largest drop).
  Every other vocabulary with a drop is named after it with its own answer, and so are larger
  vocabularies on which the drop does not go on.
- The rule is applied separately to each combination of usage and seasons. Inside one
  combination only runs with the same word-table size are compared, so the 16-animal-table runs
  of the `table-check` round are not judged by it.

`report.py` prints one verdict per width, one line per seed with the numbers and which of (a),
(b), (c) held, and one summary line per group, for example "the rule is met for no width."

Two things the letter of the rule allows, which the report points out when they happen. "At least
2 seeds" is a fixed number: with ten seeds, two that show the pattern are enough. And "some
larger vocabulary" is asked of each seed on its own: two seeds may show the drop on two different
vocabularies. The planned rounds have three seeds, so the first cannot happen unless seeds are
added later. Whether the rule should be tightened here is an open question; it has not been
changed.

## The path, one command at a time

Each step adds one idea. Run them in this order.

**1. Meet the language.** Trains nothing.

```
.venv/bin/python language.py
```

It prints the words, the facts, one training set, sample rows, and the oracle at work on one row.
The start of that row, from the real output:

```
  word       probability  surprise in bits
  are             0.1000              3.32   a free choice: nobody can know it in advance
  deer            0.0625              4.00   a free choice: nobody can know it in advance
  red             0.5000              1.00   the value a question shows: the true one half the time
  ?               1.0000              0.00   fixed by the form of the sentence
  yes             1.0000              0.00   an answer: certain for whoever knows the fact
  bats            0.0375              4.74   a free choice: nobody can know it in advance
  eat             0.2500              2.00   a free choice: nobody can know it in advance
  insects         1.0000              0.00   a fact: certain for whoever knows it
  .               1.0000              0.00   fixed by the form of the sentence
```

And its last table, the first lesson of the experiment:

```
KNOWLEDGE AND THE BEST POSSIBLE LOSS
  animal words  words in use  bits of facts  bits per animal   best possible loss (bits per word)
            16            43            128             8.00                                1.478   <- this language
            64            91            512             8.00                                1.863
           256           283          2,048             8.00                                2.248
         1,024         1,051          8,192             8.00                                2.632
```

The best possible loss rises with the vocabulary even for a perfect model, because there are more
animals to choose among. The bits per animal stay the same. So never compare the raw loss of two
vocabularies. Try `--animals 1024`, `--usage zipf` and `--seasons 4`.

**2. Check the measuring instrument.** Trains almost nothing.

```
.venv/bin/python -m pytest
```

Before any score is trusted, the tests push two fake models through the grader. The oracle, which
knows everything, must get 100 and a gap of 0. A "blind" model, which knows the sentence forms but
no single animal, must get 0 on every score that is corrected for chance. The tests also check
the exact parameter counts and that two runs of different width read exactly the same sets.
(About 390 tests. They took a little over a minute on this machine while a round was training
beside them.)
`.venv/bin/python score.py` shows the same check as a small table, and
`.venv/bin/python model.py` shows where a model's parameters are.

**3. Train one model.** This is the first base run: 16 animals, width 32.

```
.venv/bin/python train.py
```

It prints one line per grading (sets, steps, minutes, times each fact was seen, loss, best
possible, gap, facts known, questions right, words known), then how the run ended as a plain
sentence, then a dozen prompts with the model's answer beside the true one. The result goes to
`runs/a16_w32_equal_s0/`: `summary.json` (every number, with the learning curve) and `model.pt`
(the trained model). Everything else rests on this run: part (a) of the rule needs a model that
reaches 95 on 16 animals. If this run does not, that has to be understood first.
`.venv/bin/python train.py --help` lists every setting.

**4. Ask the model yourself.**

```
.venv/bin/python ask.py runs/a16_w32_equal_s0 "do foxes eat"
```

It prints the model's most likely next words with their probabilities, beside the oracle's exact
probabilities. Put the prompt in quotes and a space before `?` and `.`. Without a prompt it asks
for prompts in a loop.

**5. The experiment in one command.**

```
.venv/bin/python compare.py
```

It trains the same width-32 model on 16 and on 1,024 animals, each until its own run ends, and
prints the two results twice: (1) at the same number of sets, (2) at the end of each run. Then it
points out what to notice. It gives no verdict: one pair of runs cannot settle anything.
`--small`, `--large` and every flag of `train.py` work here. The 16-animal run of step 3 is
already saved with these settings, so it is used as it is and not trained again; `--again` trains
both runs afresh.

**6. A first look at more than one model size.** Four runs: 16 and 1,024 animals, widths 4 and 32.

```
.venv/bin/python sweep.py quick --dry-run     # list the runs and stop
.venv/bin/python sweep.py quick
```

The two width-32 runs were already trained in step 5 and are skipped, so only the two width-4
runs are new.

**7. Read the results.** Trains nothing.

```
.venv/bin/python report.py
```

It reads every `runs/*/summary.json`, prints a results table and the means over seeds, writes
`results/summary.csv` (one row per run), `results/curves.csv` (one row per grading) and the
charts, and applies the decision rule. A chart is drawn only when there are runs for it.

**8. The main round.** 48 runs: 16, 64, 256 and 1,024 animals; widths 4, 8, 16, 32; seeds 0, 1, 2.

```
.venv/bin/python sweep.py round1-equal
.venv/bin/python report.py
```

**9. The same round with zipf usage.** 48 more runs.

```
.venv/bin/python sweep.py round1-zipf
.venv/bin/python report.py
```

**10. Look inside the model.**

```
.venv/bin/python probe.py runs/a1024_w32_equal_s0
```

The hypothesis says a model is a representation of the meaning of words, so this tool looks at
the representation itself: the rows of the word table. It asks whether a fact can be read from a
row by a simple rule, whether animals with similar facts have similar rows, and it saves a
picture to `results/`. The first check is made from 64 animals in use on. Beside its numbers it
prints how far they stray by luck alone (the same check with every row handed to the wrong
animal, written `+/-`): a result inside that range shows nothing. With 16 or 64 animals luck is
large; with 1,024 it is small. Nothing it prints is a verdict on the hypothesis.

Two more rounds exist: `sweep.py table-check` (16 animals with a 16-animal word table and with
the 1,024-animal one: do the unused rows change anything?) and `sweep.py scaleup` (the 32-bit
round). `.venv/bin/python sweep.py` lists all rounds.

One thing to know about `table-check`. Left to itself, the size of a set follows the size of the
word table: 960 sentences with the 16-animal table, 61,440 with the 1,024-animal one. So the
round writes the set size out: both halves get sets of 61,440 sentences. With this setting the
table is the only difference: both halves read exactly the same text in the same batches, are
graded at the same points, and the model's body starts from the same values. (The rows of the
word table itself start from other random values, because a table of another size is drawn.) If
you try a small table by hand with `train.py --table-animals 16`, add `--set-sentences 61440`
as well; otherwise one set is a single training step.

### Things to know about rounds and run names

- **How long?** Only the speed of training has been measured: on this machine, on 8 threads, a
  model of width 4 to 32 takes about 60 to 78 training steps per second, so one set of the
  8-bit language (53 steps) takes a little under a second. How many sets a run needs is not
  known before the first runs. So the only firm figure for a whole run is the ceiling set by the
  budget: 20 minutes per run, so about 80 minutes for `quick` and 16 hours for a round of 48
  runs. (A run can overshoot its budget by one set and one grading.) `sweep.py` prints this
  ceiling, and after each run an estimate from the runs so far. `--max-minutes` changes the
  budget of every run.
- **One run at a time.** By default a round trains one run at a time on 8 threads. `--jobs 4`
  trains four at a time with 2 threads each. The budget is time on the clock, so slower runs may
  then end not finished.
- **Rounds resume.** `sweep.py` skips a run whose `summary.json` exists. An interrupted round is
  continued by typing the same command again, and the runs of steps 3, 5 and 6 are not trained
  again in step 8.
- **Part of a round.** `--widths 8 16 32` trains only those widths of a round, `--seeds 0` only
  that seed, `--animals 16 1024` only those vocabularies; the rest can be trained later with
  another command. Give every part the same `--max-minutes`: the budget is not part of a run's
  name, and `sweep.py` prints a NOTE when runs of the round were saved with a larger one.
- **A run that fails does not stop the round.** Whatever goes wrong in one run, it is listed as
  FAILED with the reason, the other runs are still trained, and the command ends with an error
  code. Typing the command again tries the failed run again.
- **Names hold six settings only:** animals, width, usage, seasons, table size and seed, as in
  `a16_w32_equal_s0`, `a1024_w4_zipf_s2`, `a256_w16_equal_x4_s0` (four seasons) and
  `a16_w32_equal_t16_s0` (a 16-animal table). (A run with `--fixed-sets 500` gets `_f500` at the
  end as well.) `train.py` always trains, and replaces a run of the same name. `compare.py` uses
  a saved run that was trained with the same settings, and stops if a run of that name was
  trained with other settings; `--again` makes it train and replace. If you change any other
  setting, such as `--learning-rate`, add `--name` or `--out-dir`.
- **A stale run is skipped too.** If a run of the right name was saved with other settings, or
  ended by budget under a shorter `--max-minutes`, `sweep.py` still skips it and prints a NOTE.
  After trying a round with a short budget, delete the folders of the runs that ended by budget,
  or use another `--out-dir`, before the real round.

`runs/` and `results/` are generated. `runs/` is ignored by git, and so is `tries/`, the folder
used below for side experiments; `results/` is not, so charts and tables can be committed if you
want to keep them.

## Reading the charts

Every dot belongs to one run. A filled dot is a run that the stopping rule ended (solved or
plateau). A **hollow** dot is a run that ended by budget: not finished. A **half-filled** dot is
a run of a fixed length (`--fixed-sets`): the stopping rule was never asked, so it is no limit
either. Every value in a chart can also be read as a number in `results/summary.csv` or
`results/curves.csv`; there the column `finished` says `yes`, `no` or `fixed`. In the table of
means `*` marks a mean with a run that was not finished and `#` one with a run of a fixed length,
and a line under each table says how many runs are behind a mean.

**Chart 1, "More words, same model"** (`chart1_more_words.png`). Across: the number of animal
words. Up: facts known. One line per width, through the means over seeds; one panel per usage and
number of seasons. Every model on one line has exactly the same number of parameters. A line that
falls to the right means more words, fewer facts known. Then look at the line of the next width
up at the same vocabulary: does it stay clearly higher? The runs on one vocabulary are set side
by side, the widths from left to right and for each width one dot per seed, so that runs with
the same score do not hide each other. Lines of widths with the same means still lie on top of
each other; the names at their ends and the table tell them apart.

**Chart 2, "Learning against practice"** (`chart2_learning.png`). For the largest vocabulary.
Across: how many times each fact has been seen. Up: facts known. A thick line per width is the
mean over seeds, thin lines are single seeds, and a dot marks where each run ended. A line that
has gone flat below 100 has stopped gaining from practice. A line that is still rising where it
ends has not shown its limit.

**Chart 3, "Knowledge per parameter"** (`chart3_knowledge_per_parameter.png`). Across: the load,
bits of facts per parameter. Up: the share of the knowledge the model holds (bits known out of
all bits, a lower bound). One dot per run, colored and shaped by bits per animal (8 or 32). The
question it answers: do the dots fall on one curve, so that load alone decides?

**Chart 4, "Common and rare words"** (`chart4_common_and_rare.png`), zipf runs only. The animals
of the largest vocabulary in groups by rank of use (1-16, 17-64, 65-256, 257-1024). Up: the share
of facts right in each group, not corrected for guessing; a thin gray line shows what guessing
gives. One line per width.

### What is NOT a size effect

- **A run that was not finished.** It ended by budget: a hollow dot, `NOT FINISHED` in every
  table. It shows where the run stood when time ran out, not a limit of the model.
- **A difference that appears only at equal sets.** One set shows each fact of the 16-animal
  language 960 times, and each fact of the 1,024-animal language 15 times. So in table (1) of
  `compare.py` the large vocabulary has had 64 times less practice per fact. Read table (2) and
  chart 2.
- **A drop that the next width up does not fix.** If a bigger model on the same text is no
  better, size was not what was missing.
- **Under zipf, rare words that are simply under-practiced.** In one set of the 1,024-animal
  language each fact of the most used animal is seen about 2,045 times and each fact of the least
  used animal twice. "Times each fact was seen" is an average that hides this. Chart 4 and the
  `by use` score separate common from rare.
- **A higher raw loss.** The best possible loss itself rises with the vocabulary (step 1).
- **A larger gap.** Only facts known can be compared between vocabulary sizes. The gap can be
  compared only at 0. With 16 animals a row of about 18 sentences often repeats a fact, so part
  of the gap can be closed by copying from earlier in the row without knowing anything. The code
  review measured it for a reader that knows no fact at all: with equal usage it closes about 11%
  of the gap at 16 animals, against 0.1% at 1,024. Facts known is asked without earlier text and
  does not have this problem.

## Experiments to try, one thing at a time

Change one thing, rerun, compare. Keep side experiments out of the main results with their own
folder (`--out-dir tries`), and give a run a `--name` when the changed setting is not part of the
name.

```
.venv/bin/python train.py --out-dir tries                                 # the baseline: 16 animals, width 32
.venv/bin/python train.py --out-dir tries --animals 1024 --width 4        # many words, a very small model
.venv/bin/python train.py --out-dir tries --animals 1024 --width 8        # the next width up, same sets
.venv/bin/python train.py --out-dir tries --animals 256 --usage zipf      # common and rare animals
.venv/bin/python train.py --out-dir tries --seed 1                        # other starting values, facts and shuffling
.venv/bin/python train.py --out-dir tries --table-animals 16 --set-sentences 61440   # no unused rows, same sets
.venv/bin/python train.py --out-dir tries --seasons 4 --width 16          # 32 bits per animal
.venv/bin/python train.py --out-dir tries --layers 1 --name one_layer     # a shallower model
.venv/bin/python train.py --out-dir tries --learning-rate 0.001 --name slow_learning
.venv/bin/python train.py --out-dir tries --min-steps 8000 --name patient     # a stricter plateau
.venv/bin/python train.py --out-dir tries --animals 1024 --width 4 --fixed-sets 500   # no stopping rule: exactly 500 sets
.venv/bin/python report.py --runs-dir tries --results-dir tries_results
```

The table of `report.py` shows every run, so a changed run can be read beside the baseline. The
rule and charts 1, 2 and 4 keep only one run per vocabulary, width and seed and list the others
as left out, with their scores (see "One run per place" above). A folder of side experiments
mixes recipes on purpose, so the decision rule is not applied to it: the report says `CAREFUL`,
names the runs that differ, and gives no verdict. That is what should happen; the table and the
charts are what such a folder is for.

You can also edit the code. Change a fact in `BASE_ANIMALS` at the top of `language.py`, say by
making most animals brown, and run `.venv/bin/python language.py` again: the bits per animal drop,
because a color that is easy to guess is worth less. Nothing in the code assumes the table is
balanced.

## Honest limits

- **One word is one token here.** Real language models split words into pieces, so their
  "vocabulary" is not a list of words.
- **The score is recall of facts.** It says nothing about fluency, reasoning or generalization.
- **The invented animals have random facts.** Knowing foxes tells a model nothing about `bakos`.
  In a real language the meanings of words are related, and a model can use that.
- **This is a toy.** Models of 5,000 to 175,000 parameters, and a made-up language with eight
  fixed sentence frames. A result here does not transfer automatically to large language models.
- **A positive result would support one consequence of the hypothesis, not the whole of it.**
  "More words need a bigger model" could be true for other reasons than "a model is a
  representation of the meaning of words".
- **A plateau is a limit under this training recipe, not a proof.** Another learning rate, more
  layers or more patience might get further. That is why the rule also asks for a rescue by the
  next width up on the same sets.
- **The stopping rule has soft spots.** A plateau is a judgment from a few numbers. With the
  default word table and one season a set is 53 training steps, so the earliest grading at which
  a plateau can be called is set 45 (2,385 steps), at every vocabulary size, and only if the
  model knows 5 facts out of 100 by then. With four seasons a set is 295 steps and the earliest
  is set 8, or set 32 for 1,024 animals, where the 400 sightings come later. The floor of 2,000
  steps alone was not enough in calibration, which is why `started_at` was added. The price is
  that a model which never begins to learn facts can never be called flat (rule 6). The rule also
  asks only whether the score still *gains*: a score that falls passes that test and is saved as
  a plateau. `train.py` and `report.py` say so when a run ends well below its own best score. A
  plateau that looks doubtful can be checked by training much longer with `--fixed-sets`.
- **Small things in the instrument.** The oracle ignores how much room is left in a row, so a gap
  a hair below 0 (`language.py` puts it at about 0.002 bits per word at the default row length,
  more for shorter rows) is not an error. Bits known is a lower bound, and words known is not
  corrected for chance. `frogs` and `toads` have identical facts in the base table, which `probe.py`
  will see as two identical animals. And the model reads a word's row through a step (a layer
  norm) that removes the row's average, so at width 4 it can read about three numbers' worth
  from a row, not four.
- **`probe.py` measures with few animals.** With 16 or 64 animals its numbers move by luck as
  much as by anything in the word table. It prints that luck beside each number (`+/-`); read a
  result against it. Under zipf usage it counts every animal once, however rarely it is used.
- **The recipe was calibrated on little.** See the status at the top. The learning rate was
  chosen on the 16-animal language. If a default is changed after runs exist, every run has to
  use the new one: `report.py` gives no verdict for a group that mixes recipes, and every
  `summary.json` carries a mark of the code that made it (`code`), so that runs made by
  different versions of the code can be told apart.

## Files

| file | what it is |
|---|---|
| `language.py` | the toy language, the exact training sets, and the oracle |
| `model.py` | the tiny GPT |
| `score.py` | grading a model against the oracle |
| `train.py` | one training run on sets, with the stopping rule |
| `compare.py` | the same model on a small and on a large vocabulary |
| `sweep.py` | the named rounds of runs |
| `report.py` | table, charts and the decision rule, from finished runs |
| `probe.py` | a look inside the trained word table |
| `ask.py` | type a prompt, see the model's next word beside the oracle's |
| `tests/` | the tests (`.venv/bin/python -m pytest`) |
| `runs/`, `results/` | generated: one folder per run; charts and tables |

Each file starts with a short text saying what it is for and how to run it.

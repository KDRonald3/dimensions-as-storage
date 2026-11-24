# Third experiment: plan (DRAFT for the owner to correct)

Drafted on 22 November 2026. Nothing in it has been run. Points marked **TO DECIDE** are open.

## The claim (the owner's, 22 November)

> "the thing of interest here is not the width of the model but the dimensions in which the
> token is represented. simply, there are more weights to help distinguish the tokens as the
> dimensions increase, despite no change in width."

The second experiment cannot show this, because its stages were trained differently (different
learning rates and lengths) and it used one number of columns only (32).

## The grid

| What | Values | How many |
|---|---|---|
| Columns of the word table (the dimensions of a token) | 8, 16, 32, 64 | 4 |
| Body width (neurons in the middle part of each layer) | 4, 8, 16, 32, 64, 128 | 6 |
| Bits per creature (2 bits for each kind of fact) | 8, 16, 32, 64, 128 | 5 |
| Creatures | 16, 64, 256, 1,024 | 4 |
| Word use | equal, unequal | 2 |
| Repeats | seeds 0, 1, 2 | 3 |

That is 2,880 runs. At 32 columns the grid contains the whole second experiment again, plus
body width 4 and 128 bits.

## Held constant in every run

- The language, the facts, the held-back pairs and the exam (as in `v2/PLAN.md`, Part B). 128
  bits needs 64 kinds of fact, so 32 more kinds of words are added in the same pattern.
- One sentence per line of text, 512 lines per step.
- 2 layers, 2 attention heads, a trained word table, a separate output layer.
- The starting word table: one table with 64 columns is drawn once; a model with fewer columns
  starts from its first columns. So a word starts from the same numbers at every column count.
- **The text** (decided by the owner on 22 November: "the text should be kept constant across
  all the variants that train on it"). Every language, that is every combination of creatures,
  bits and word use, has ONE fixed training text: the same sentences in the same order, 512 at
  a step. Every run that trains on that language reads exactly this text, whatever its
  columns, body width, learning rate or repeat. A run's seed sets only its starting weights.
  (In the second experiment the seed set the text as well. The programs need a separate text
  seed for this, fixed at 0.)
- Everything else about training is as in the second experiment: the same optimiser, the same
  warm-up, the averaged copy.

## What may differ between runs (decided by the owner on 22 November)

Only one thing differs between runs besides what the grid varies: how long a run trains, and
that is set by one rule for all.

- **Learning rate: the same for every run** (the owner, 22 November: "given the results are
  best compared with everything constant, it is better to keep the same learning rate for all
  models"). Practice runs (seed 100, never counted) with three rates at the corners of the grid
  fix this one rate before the counted runs: the rate with the most facts known at the end,
  averaged over all practice settings.
- **Whether another rate would have done better is tested afterwards** (the owner: "if the
  experiment is done and there is still compute time available, you can test this by picking
  one model seed across the different widths, and using 1 or 2 lower learning rates, on all
  the variables that affect the performance of the model", and then: "probably the best
  performing seed"). So, only after the counted runs and only if card time is left: for every
  setting, the repeat that scored best in the counted runs is run again from the same starting
  point at a third of the chosen rate, and then, if time still remains, at a tenth. "Best" is
  most facts known; a tie goes to the better score on questions about one creature, and then
  to the lower seed. These runs are reported separately.
- **Training length: one rule for every run** (the owner left the choice to the agent: "do
  what is better"). A fixed length left the hardest settings of the second experiment still
  rising when training ended, so that "cannot hold" could not be told from "not learned yet".
  So a run is looked at when it reaches 40,000 steps, and again each time its length has
  doubled (80,000, 160,000, 320,000, 640,000). It ends at the first of these points where one
  of three things is true:
  1. *Learned:* facts known and questions about one creature are both 99 or more.
  2. *No more gain:* facts known rose by less than 1 point over the last doubling AND by less
     than 1 point over the doubling before it.
  3. *The limit:* it has reached 640,000 steps. Such a run is marked "still rising".
  Every run's result is its scores at the point where it ended, with the reason. The rule is
  the same for every run and is applied by the program, not by judgement.
- **A known weakness of this rule:** in the first experiment one run stood still for two whole
  doublings and then learned. A run like that would be ended as "no more gain". Nothing of the
  kind has been seen in the new language so far.
- **Showing that a run was not stopped too early** (agreed with the owner on 22 November, to
  "prevent gaps in our data and the rigor of the work"):
  1. *The evidence is kept for every run.* A run that ends as "no more gain" records its
     scores at a quarter, a half and the whole of its length and how far its training loss
     still fell. The report shows these for all such runs.
  2. *A fixed sample is carried on anyway.* Every tenth run in the fixed list of counted runs
     is marked beforehand. If a marked run ends as "no more gain", it is trained for two more
     doublings regardless (or to the limit). Its result stays the one at the point where the
     rule ended it; the later scores are recorded beside it. The report states how many of
     the runs carried on gained a point or more. That number is the measured error of the rule.
  3. *Every run that is ended keeps its checkpoint,* so it can be carried on later.
  Not adopted: trying a lower learning rate before stopping. It would change the learning rate
  during a run, and the owner asked for training to stay the same.
- No other training setting differs between runs.

## One thing that cannot stay constant (decided by the owner on 22 November)

In this model the number of columns is also the width of the stream that every layer reads
and writes. So more columns mean three things at once: a longer row for every word, larger
attention parts (their size goes with columns x columns), and a longer input to the middle
parts. Only the number of middle neurons (the body width) stays the same.

**Decision: change only the `columns` setting.** At 32 columns this repeats the second
experiment exactly. If more columns help, these runs cannot say whether the longer rows or
the larger attention parts did it. The owner: "if it is observed that this is the case, then
we can narrow it further if that is possible."

**Rejected:** keeping the body at one fixed width and adding a shared step that turns a row of
8, 16, 32 or 64 numbers into that width. The owner: "this does not solve the problem. that
would be mapping a larger space to a smaller one, which maintains the space a model sees."
The claim is about the space the model works in, so that space is what must change.

## Predictions, to be fixed before the runs

- **The owner's:** with body width, bits and creatures held fixed, the scores rise with the
  number of columns.
- **The agent's (from the second experiment):** what decides is the bits each number of a
  word's row must hold, that is bits per creature divided by columns. At 1 or less the facts
  are learned at every body width and every vocabulary. At 2 learning is slow and may not
  finish. At 4 or more it fails. Body width makes little difference; more creatures need more
  training but do not lower the final score at 1 or less.

**The numbers, fixed on 22 November before any run.** "Score" is facts known at the point where
a run ended, and a setting's score is the middle one of its three repeats.

- **C1 (the owner's claim).** For a body width, bits and number of creatures held fixed, the
  score never falls by 5 points or more from one column count to the next larger one, and in
  at least a quarter of those fixed settings it is 10 points or more higher at 64 columns
  than at 8.
- **C2 (the owner's claim, the other half).** Body width does not do this: in fewer than a
  tenth of the settings (columns, bits and creatures held fixed) is the score 10 points or
  more higher at body width 128 than at body width 4.
- **A1 (the agent's).** Where bits per creature divided by columns is 1 or less, every setting
  ends as "learned", at every body width and every number of creatures.
- **A2 (the agent's).** Where that figure is 4 or more, no setting with 1,024 creatures reaches
  95.
- **A3 (the agent's).** Two settings with the same bits divided by columns (for example
  16 bits in 8 columns and 64 bits in 32 columns), the same body width and the same number of
  creatures end within 10 points of each other.

The report states for each of the five whether it held, with the count of settings on each
side. C1 and A1 to A3 can all hold together; they describe the same thing from two sides.

## Where and how long

- On Kaggle's free graphics cards (a session has two T4 cards; 30 hours a week), with the
  trainer that trains many models in one pass (`v2/batch.py`, being built and reviewed).
- Estimated work: about 357 million training steps, about 13 times the second experiment.
  Estimated card time: 17 to 33 hours on one card, 8 to 17 with both. The card speed is not
  measured yet.
- Before any counted run: a timing test on a card, and a check that the new trainer gives the
  same results as the old one on a small part of the second experiment.

---

## Part B. What the programs must do (written 22 November, before any run)

The programs of the second experiment (`v2/`) are extended. Every new behaviour sits behind a
setting whose default keeps the second experiment's behaviour exactly, so all existing tests
must keep passing unchanged.

### B.1 New settings of a run

- `kinds` may also be 64 (128 bits): 32 more kinds of fact, each with one linking word and four
  values, in the same pattern as kinds 5 to 32, all plain words that are different from every
  other token.
- `columns` is 8, 16, 32 or 64.
- `table_start: "nested64"`: one word table with 64 columns is drawn once, with a fixed seed,
  for the whole vocabulary (64 kinds, 1,024 creatures). A model uses the rows of its own tokens
  and the first `columns` columns. So a word starts from the same numbers at every column
  count, kind count and vocabulary.
- `text_seed`: when given, the training text depends on (text_seed, step) and not on `seed`.
  `seed` then sets the starting weights only.
- `length_rule: "doubling"` with `first_look` (40,000) and `steps` as the limit (640,000). The
  run is looked at when it reaches `first_look` and each time its length has doubled, up to
  the limit. It ends at the first look where (1) facts known and questions about one creature
  are both 99 or more ("learned"), or (2) facts known rose by less than 1 point over the last
  doubling and by less than 1 point over the doubling before it ("no more gain"), or (3) it
  has reached the limit ("limit"). There are gradings at a quarter and a half of `first_look`
  and at every look, besides the usual ones. The summary gains
  `"ended": {"step": N, "reason": "..."}`; `final`, `steps` and `model.pt` are those of the
  step at which the run ended.
- Runs in one group of the many-models trainer may have different learning rates. A group is
  then defined by kinds, body, columns, layers, heads, table form and the settings above, not
  by the rate. When a member ends, the group goes on without it.

### B.2 The runs

Names: `d{columns}_c{creatures}_b{body}_{usage}_k{kinds}_s{seed}`, with `_lr{rate}` added for
practice runs. Every run has `text_seed: 0` and `table_start: "nested64"`, and every run has
2 layers and 2 attention heads at every column count (the owner, 22 November: "make sure you
maintain the same number of transformer heads across the different dimensions"). The number
of heads is fixed; the size of each head is half the columns, so it grows with them.

- **Practice** (never counted): for every columns (8, 16, 32, 64) and kinds (4, 8, 16, 32, 64):
  body 4 and 128, creatures 16 and 1,024, rates 0.003, 0.01 and 0.03, equal use, seed 100,
  a fixed 40,000 steps. That is 240 runs in 40 groups of 6.
- **Choosing the one rate** for the whole experiment, by program: the rate with the highest
  facts known at the end, averaged over all 80 practice settings. Rates within half a point
  of the best are tied; a tie goes to the higher average of questions about one creature, and
  then to the lower rate. (An earlier draft chose a rate for each combination of columns and
  kinds. The owner decided on one rate for all.)
- **Counted runs:** columns x body (4, 8, 16, 32, 64, 128) x kinds x creatures (16, 64, 256,
  1,024) x usage (equal, zipf) x seeds (0, 1, 2), with the chosen rate and the doubling rule.
  That is 2,880 runs in 120 groups of 24. Every tenth run in this fixed list (positions 0, 10,
  20, ...) carries `carry_on: true`: if it ends as "no more gain", it is trained for two more
  doublings regardless, or to the limit. Its `final`, `ended` and `model.pt` stay those of
  the point where the rule ended it, and the later gradings are kept in its curve and
  summarised under `"carried_on"` in its summary.
- **Rate check** (a separate stage, started only after the counted runs and only if card time
  is left): for every setting, the seed of its best counted run (most facts known; a tie goes
  to the better questions-about-one-creature score, then to the lower seed) again, at a third
  of the chosen rate; and a further stage at a tenth. 960 runs each, names ending in
  `_lr{rate}`, results in their own folder.

### B.3 Running on Kaggle

- A Kaggle session has two graphics cards and may last 12 hours; the account has 30 session
  hours a week. A session is given a list of groups and a time budget. It trains one group on
  each card at a time, stops cleanly at a checkpoint when its budget is used, and leaves its
  finished results and its checkpoints as its output.
- A program on the owner's computer sends the sessions, as many at once as Kaggle allows,
  watches them, downloads the results into `runs_v3/`, and sends further sessions for groups
  that are not finished, which pick up from the checkpoints. It never starts a session that
  the remaining weekly quota cannot cover. It can be stopped and started again without losing
  or repeating work.
- Order of the counted groups, so that an early stop still gives a balanced picture: body
  widths in the order 128, 4, 32, 8, 64, 16; inside a body width, kinds from 4 to 64; inside
  those, columns from 8 to 64.

### B.4 Running on Colab (added on 23 November; the owner moved the experiment to Colab Pro+)

The owner bought Colab Pro+ and asked for the result in hours and for no credits to be wasted.
The machines are paid by the hour whether they work or not, so:

- **The fastest cards free at the moment of launch are taken,** by asking for every card type
  at once (v3/colab_gpu.py). Getting the machine is the test.
- **Breadth first.** Every run is taken to its first look (40,000 steps) before any run goes
  further; then the runs that have not ended go on to 80,000, 160,000 and so on. Many runs
  end at the first look (few bits per creature, few creatures); the runs that carry much per
  word in a narrow word table go on longest.
- **Cards stay at full load, and machines are given back as the work shrinks** (the owner, 23
  November: "you should run it such that, in the end, all of the longer-running
  configurations run on the same set of GPUs that are at full load (so that we don't have a
  lot of credits going for no good reason)"). After every round the launcher works out how
  many machines the remaining groups need to keep each card fully busy, moves the remaining
  groups (with their checkpoints) onto that many machines, and stops the others at once. A
  machine is never kept running with a card that is mostly idle.
  *As built, and accepted by the owner on 23 November ("I think the way it handles the longer
  runs is fine"):* the launcher hands out the groups, moves waiting groups to machines that
  have become idle, and stops each machine as soon as it has nothing left to do. It does not
  regroup the remaining long runs onto fewer machines after every round.
- **A reserve of compute units** is never touched (default 100): the launcher stops cleanly
  before the balance falls below it.

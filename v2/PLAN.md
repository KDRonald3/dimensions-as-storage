# Second experiment: plan and specification

Written on 22 November 2026, before any run of this experiment. Part A is the plan, with the
predictions fixed beforehand. Part B is the exact specification the programs must follow.

## Part A. The plan

### The claim under test (the owner's)

A model of a given size can hold the meaning of only so many words. A small vocabulary needs
only a small model, and a larger vocabulary at the same model size does worse.

### What the first experiment got wrong, and what changes

| First experiment | This experiment |
|---|---|
| Each word had a trainable row in the word table | Unchanged, by the owner's decision of 22 November: the table is trained, as in ordinary models. Every run starts from the same table. A frozen table is kept as a switch, to be tested later |
| "Width" changed the length of every word's row and the body together | The table has 32 columns in every run. Model size is the body only: the neurons in the middle part of each layer (8, 16, 32, 64, 128) |
| 20 sentences per animal, each read thousands of times | Sentences are about up to four creatures at once: 215 thousand valid sentences for 16 creatures, 5.6 trillion for 1,024 |
| About ten sentences packed into one row, so a model could look back at an earlier sentence | One sentence per row |
| Runs ended when they looked "flat", which proved unreliable | Every run has the same fixed length, set beforehand from a trial |
| 3 repeats, 1 for most of the harder round | 5 repeats of everything |
| Scored on sentences it had read | Also scored on combinations of creatures it has never read together |
| More bits came with season words, a different kind of sentence | More bits come as more kinds of fact, in the same kind of sentence |

### The grid

- Creatures: 16, 64, 256, 1,024.
- Body width: 8, 16, 32, 64, 128.
- Word use: equal, and unequal (Zipf), as in the first experiment.
- Bits per creature: 8 first (4 kinds of fact). Then 16 (8 kinds), 32 (16 kinds) and, added by
  the owner on 22 November, 64 (32 kinds). The stages after 8 bits use seeds 0 to 2.
- Repeats: seeds 0 to 4. Practice runs use seeds 100 and above and are never counted.

At 8 bits that is 4 x 5 x 2 x 5 = 200 runs.

### Predictions, fixed before the runs

"Facts known" is the main score (B.5, score S1), read at the end of the fixed training length.
A prediction holds for a setting when it holds in at least 4 of its 5 repeats.

- **P1 (a small vocabulary needs only a small model).** With 16 creatures every body width
  reaches 95 or more.
- **P2 (a larger vocabulary at the same size does worse).** At body width 8, the score on 1,024
  creatures is at least 10 points below the score on 16 creatures.
- **P3 (a larger model holds more).** On 1,024 creatures, body width 128 scores at least 10
  points above body width 8.
- **P4 (the agent's expectation, which is the opposite of P2).** With a trained word table the
  number of creatures makes little difference: at every body width the score on 1,024 creatures
  is within 10 points of the score on 16 creatures. The reason given beforehand: each creature
  has its own row of 32 trained numbers, which is room enough for its 8 bits, so a new word
  brings its own storage. If anything fails, the agent expects it to follow the bits per
  creature and the body width, not the number of creatures.
- **For the later test with a frozen table** (not part of the counted runs): the agent expects
  width 8 to hold 64 creatures but not 256, width 32 to hold 256 but not 1,024, and width 128
  to hold 1,024. "Holds" means 95 or more.

P2 and P4 cannot both hold. The runs decide between them.

**Verdict.** The claim is *supported* if P1, P2 and P3 hold with equal use and with unequal
use. It is *contradicted* if at every body width the score on 1,024 creatures is within 10
points of the score on 16 creatures. Anything else is *mixed*, and the report says which parts
held. A low score can mean "cannot hold" or "not learned yet": the report shows every training
curve, and the settings that fail get one run four times as long.

### Rules

- Training length, learning rate and every other setting are fixed from practice runs before
  the counted runs start, and are written at the end of this file with the date.
- Every counted run is reported. Nothing is rerun to get a better number.
- The report lists everything done beyond this plan.

---

## Part B. Specification

> **Changed on 22 November at about 13:30, at the owner's instruction, before any run:** the
> word table is TRAINED by default (`table: "trained"`). The frozen form stays available as
> `table: "frozen"` for a later test. Programs written before this change implement only the
> frozen form; the switch described in B.6 and B.8 must be added, with tests for both forms.

Everything lives in the folder `v2/` and must not import from the first experiment's programs
(`train.py`, `language.py`, `model.py`, `score.py`): copy what is needed. The first experiment's
files must not be changed. Code must be plain and readable, with comments in plain English.

### B.1 The words

- `<pad>` (row 0).
- 14 fixed words: `.` `?` `,` `and` `not` `are` `do` `eat` `live` `in` `wake` `at` `yes` `no`.
- Kinds of fact. Each kind has a linking phrase and four values:
  1. `are`: red brown green black
  2. `eat`: meat plants insects seeds
  3. `live in`: woods fields rivers caves
  4. `wake at`: dawn noon dusk night
  5. `fear`: fire water noise light
  6. `like`: sun rain snow wind
  7. `carry`: sticks stones leaves shells
  8. `drink`: dew sap milk nectar
  9. `climb`: trees rocks walls hills
  10. `dig`: holes tunnels pits dens
  11. `catch`: flies moths mice fish
  12. `build`: nests dams webs mounds
  13. `follow`: tracks scents sounds stars
  14. `avoid`: roads farms towns camps
  15. `cross`: bridges marshes dunes ice
  16. `guard`: eggs cubs paths pools
  17. `smell`: smoke rot musk salt
  18. `hear`: thunder steps calls wings
  19. `chew`: bark roots grass bones
  20. `watch`: clouds shadows flames waves
  21. `dodge`: traps nets snares hooks
  22. `seek`: shade warmth shelter company
  23. `gather`: twigs moss feathers pebbles
  24. `hoard`: nuts berries grain honey
  25. `visit`: ponds cliffs meadows ruins
  26. `shed`: fur skin horns claws
  27. `mark`: logs stumps boulders fences
  28. `raid`: hives barns orchards burrows
  29. `ride`: currents gusts tides floods
  30. `dread`: frost drought storms eclipses
  31. `swallow`: gravel sand mud clay
  32. `trail`: herds flocks swarms packs
- `--kinds K` uses the first K kinds (4, 8, 16 or 32, which is 8, 16, 32 or 64 bits per
  creature). Words of unused kinds are not in the vocabulary.
- Every token must be a different word. If a word above is also one of the 1,024 creature
  names, replace the word above with another plain word and say so in the report.
- Creature names: the same 1,024 names, in the same order, as the first experiment
  (copy the list or the rule that makes it from `language.py`). `--creatures N` uses the first N.
- Token order: `<pad>`, the fixed words, the linking words and values of the kinds in use, then
  all 1,024 creature names (always all 1,024 rows, so creature i has the same row in every
  vocabulary).

### B.2 The facts

For each kind, exactly N/4 creatures have each value, assigned by a random shuffle. The shuffle
uses a fixed seed that depends only on (N, kind), so every run with N creatures has the same
facts, and the facts of a kind are the same whatever K is.

### B.3 The sentences

`LIST` is 1 to 4 different creatures: `a` | `a and b` | `a , b and c` | `a , b , c and d`.
`VP` is the linking phrase of a kind other than the first.

| | first kind | other kinds |
|---|---|---|
| statement | `LIST are VAL .` | `LIST VP VAL .` |
| "not" statement | `LIST are not VAL .` | `LIST do not VP VAL .` |
| question | `are LIST VAL ? yes` or `no` | `do LIST VP VAL ? yes` or `no` |

Truth: a statement is true when every creature in the list has the value. A "not" statement is
true when no creature in the list has it. The answer to a question is `yes` when every creature
in the list has the value, otherwise `no`. Only true sentences, and questions with the right
answer, are ever in the training text.

The longest sentence is 13 tokens. A row is one sentence, padded with `<pad>` to 13.

**Held-back pairs.** One tenth of all unordered pairs of the N creatures (rounded down, at least
1), chosen with a fixed seed that depends only on N. No training sentence has both creatures of
a held-back pair in its list.

**Drawing a training sentence** (all draws from one generator; "by use" means with the
creature weights of B.4):

1. Type: statement 40%, "not" statement 20%, question with `yes` 20%, question with `no` 20%.
2. List length k: 1 with chance 40%, and 2, 3, 4 with 20% each.
3. Kind: equal chance.
4. Creatures:
   - statement and `yes` question: draw the value with equal chance, then k different
     creatures that have it, by use.
   - "not" statement: k different creatures by use; then a value none of them has, with equal
     chance (if there is none, draw the creatures again).
   - `no` question: draw the value with equal chance; draw f, the number of creatures that do
     NOT have it, with equal chance from 1 to k; draw k - f creatures that have it and f that
     do not, by use; put the k in random order.
5. If the list contains a held-back pair, draw the creatures again. If a list of length k
   cannot be made after 50 tries, lower k by one and go on.

### B.4 Word use

- `equal`: every creature in use has the same weight.
- `zipf`: exactly the weights the first experiment used for `--usage zipf` (read `language.py`
  and copy the rule), with the creature of rank 1 the most used.

### B.5 The exam (scores, all from 0 to 100, all on the averaged copy of the model)

The exam sentences are fixed for a language (fixed seed) and do not depend on the run's seed.

- **S1 facts known (the main score).** For every creature in use and every kind in use, the
  prompt `c are` or `c VP`. Right when the true value has the highest score among the four
  values of that kind. Chance is 25. Also reported per kind.
- **S2 questions about one creature.** For every creature and kind: the `yes` question on the
  true value and the `no` questions on the three other values. The answer is whichever of `yes`
  and `no` has the higher score at the answer place. Score = % of `yes` questions right
  + % of `no` questions right - 100, not below 0. Chance is 0.
- **S3 questions about new combinations.** Up to 2,000 questions whose list (2, 3 or 4
  creatures, in equal shares as far as possible) contains at least one held-back pair. Half
  `yes`, half `no` as far as possible; for `no`, f is drawn as in B.3. Creatures drawn with
  equal weights. Scored as S2. Also reported: % right for `yes`, for `no` with f = 1, and for
  `no` with f > 1.
- **S4 questions about lists without a held-back pair.** Built and scored exactly as S3 but with
  no held-back pair in any list. For comparison with S3.
- **S5 "not" known.** For every creature and kind, the prompt `c are not` or `c do not VP`.
  Right when the true value has the LOWEST score among the four values of that kind. Chance
  is 25.
- The average training loss since the last grading.

### B.6 The model

- Word table: (number of tokens) x 32, drawn once from a normal distribution with standard
  deviation 1 and a fixed seed, by drawing the full table for K = 32 and 1,024 creatures and
  taking the rows needed, so a word has the same starting row in every run.
  - `table: "trained"` (the default): the table is a parameter and is trained with everything
    else. Rows of creatures that are not in use receive no gradient and must stay unchanged.
  - `table: "frozen"`: the table is a buffer, never trained, not given to the optimizer.
  The two forms must differ in nothing else.
- Position table: 13 x 32, trained.
- 2 layers, each: normalise, attention with 2 heads, add; normalise, middle network, add
  (the same arrangement as the first experiment's `model.py`). The middle network is
  32 -> `body` -> 32 with the same activation as the first experiment. `body` is 8, 16, 32, 64
  or 128.
- A last normalise, then a trained output layer 32 -> (number of non-creature tokens, without
  `<pad>`), with bias. The model can never output a creature name.
- Starting values of trained parts as in the first experiment.
- Apart from the word table, the number of trained numbers must be identical for every number
  of creatures. With `table: "frozen"` the whole trained part is therefore the same size for
  every vocabulary; with `table: "trained"` it grows by 32 numbers for each word in use.

### B.7 Training

- Batch: 512 rows of 13 tokens. The target at each place is the next token. No loss where the
  target is `<pad>` or a creature name.
- AdamW, no weight decay, learning rate `lr` (default 0.01) constant after a 100-step warm-up,
  gradient clipping as in the first experiment, and an averaged copy of the weights (0.998) as
  in the first experiment, which is the copy that is graded and saved.
- Fixed length: `steps`. No early ending of any kind.
- Grading at steps 100 x 2^(i/2), rounded, for i = 0, 1, 2, ... up to `steps`, and at `steps`.
- The batch of step t depends only on (seed, t), so a run can be picked up again exactly.
- A checkpoint (model, averaged copy, optimizer, step, curve so far) is written every 1,000
  steps, first to a temporary file and then renamed. A run that finds a checkpoint goes on from
  it. A run that finds a finished `summary.json` returns it without training.
- `seed` sets the starting values and the training text. Facts, held-back pairs, word table
  and exam do not depend on it.

### B.8 Interfaces (other programs rely on these exactly)

`v2/run.py`:

```python
def run_one(config: dict, out_root: str) -> dict:
    """Train one model. Writes out_root/<name>/summary.json and model.pt. Returns the summary."""
```

Config keys: `creatures`, `kinds`, `usage` ("equal" or "zipf"), `body`, `seed`, `steps`, `lr`,
`threads`, and optionally `table` ("trained", the default, or "frozen"), `columns` (32),
`layers` (2), `heads` (2), `batch` (512), `warmup` (100), `average` (0.998), `name`. The default
name is `c{creatures}_b{body}_{usage}_k{kinds}_s{seed}`, with `_frozen` added at the end when
the table is frozen. The command line takes `--table trained|frozen`.

Command line: `python -m v2.run --creatures 64 --body 32 --usage equal --kinds 4 --seed 0
--steps 20000 --lr 0.01 --threads 2 --out-root runs_v2`.

`summary.json`:

```json
{"name": "...", "config": {...all keys with their values...},
 "parameters": {"trained": 0, "trained_without_table": 0, "table_rows_in_use": 0,
                "table_numbers": 0, "body_middle": 0},
 "steps": 0, "sentences": 0, "minutes": 0.0, "steps_per_second": 0.0,
 "final": {"facts": 0.0, "facts_by_kind": [0.0], "questions_one": 0.0,
           "questions_new": 0.0, "questions_new_detail": {"yes": 0.0, "no_one": 0.0, "no_more": 0.0, "count": 0},
           "questions_seen": 0.0, "questions_seen_detail": {"yes": 0.0, "no_one": 0.0, "no_more": 0.0, "count": 0},
           "not_known": 0.0, "loss": 0.0},
 "curve": [{"step": 100, "sentences": 51200, "facts": 0.0, "questions_one": 0.0,
            "questions_new": 0.0, "questions_seen": 0.0, "not_known": 0.0, "loss": 0.0}]}
```

`v2/grid.py`: `def grid(stage: str) -> list[dict]` returns the configs of a stage
(`smoke`, `pilot`, `main8`, `bits16`, `bits32`, `bits64` (kinds 32, seeds 0 to 2), `long`, and
`frozen8`, which is the `main8` grid with `table: "frozen"`). All other stages use `table: "trained"`. Settings that the pilot decides
(`steps`, `lr`) are read from `v2/settings.json`.

Results are kept in `runs_v2/<stage>/<name>/`.

---

## Settings fixed from the practice runs

### 8 bits: fixed on 22 November 2026 at 14:10 PST, before any counted run

- **Learning rate 0.01** for every body width. Three were tried (0.003, 0.01, 0.03) on 15
  practice runs with seed 100 (16, 256 and 1,024 creatures with body width 8; 256 and 1,024
  with body width 128; equal use; 20,000 steps). Every one of the 15 ended at 100 facts known.
  0.01 was the only rate with which all five settings also ended at 100 on questions about one
  creature. 0.03 was worse on the questions (87.5 on 1,024 creatures with body width 8) and
  0.003 was slower.
- **Training length 40,000 steps** (20.5 million sentences) for every run. With equal use and
  rate 0.01 the slowest practice run (1,024 creatures, body width 8) knew the facts (97) by
  step 3,200 and the questions (90) by step 6,400. No practice run used unequal use, where the
  rarest of 1,024 creatures is met about 7.5 times less often, so the length is set at about
  six times the slowest.
- **Not perfect in practice:** with 16 creatures and rate 0.01, questions about new
  combinations ended at 83. That score rests on only 58 questions with 16 creatures.
- **Machines:** body widths 8, 16 and 32 train on 1 thread, 64 and 128 on 2 threads. This
  changes the cost, not the method.
- **Done beyond the plan so far:** the value word `mice` was also a creature name and was
  replaced by `gnats`; four short timing runs (stage `speed`) measured the cloud machines.

### 64 bits: fixed on 22 November 2026 at 14:35 PST, before any counted 64-bit run

- **Practice runs** (seed 100; 1,024 creatures; equal use; 40,000 steps; body widths 8 and 128;
  three learning rates). None had finished learning. Facts known ended at 90.5 and 93.6 with
  rate 0.003 (still rising), at 87.5 and 84.7 with 0.01 (levelling off), and at 52 and 53 with
  0.03. The body width made little difference.
- **Learning rate 0.003:** the rate with the most facts known at the end of practice, for both
  body widths.
- **Training length 160,000 steps,** four times the practice length, because practice had not
  finished learning.
- **Cut down to fit the cloud credit** (a change from the plan, which had the full grid with
  three repeats): creatures 16, 256 and 1,024; body widths 8, 32 and 128; both kinds of word
  use; seeds 0 and 1. That is 36 runs.

### 16 and 32 bits: fixed on 22 November 2026 at 15:45 PST, before any counted run of them

- **Practice runs** (stage `pilot2`; seed 100; 1,024 creatures; equal use; 40,000 steps; body
  widths 8 and 128; learning rates 0.003 and 0.01). They ran on the owner's computer.
- **16 bits: learning rate 0.01, 40,000 steps.** With 0.01 both body widths ended at 100 facts
  known and 100 on questions about one creature; the facts were known (97) by step 6,400 and
  the questions (90) by step 12,800 at the latest. 0.003 was about twice as slow.
- **32 bits: learning rate 0.01, 60,000 steps.** With 0.01 the facts ended at 100 and 99.8 and
  questions about one creature at 99.8 and 97.3; the facts were known by step 18,102 and the
  questions by step 18,102 at the latest. Questions about new combinations were at 98 (body
  width 8) and 83 (body width 128) and still rising, so the length is one and a half times the
  practice length. 0.003 was slower on everything.
- **Kept as planned:** the full grid with seeds 0 to 2, 120 runs for each of the two stages.
- **Where they run** (a matter of cost and time, not of method): Modal's own meter stood at
  $10.30 when these stages started, about a third above the agent's estimates. To stay inside
  the owner's $30, the owner's computer does part of seed 2 (16 bits: body widths 8, 16, 32
  and 128; 32 bits: body widths 8 and 128) and Modal does the rest. Body width 64 now trains
  on 1 thread on Modal, like the smaller ones.

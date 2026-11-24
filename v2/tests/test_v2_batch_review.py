"""Second checks of the group trainer (v2/batch.py, v2/batch_text.py), written by a reviewer.

Everything runs on the processor and is kept small (about a minute in all).

  1. Training: a group ends with the same weights and averaged copies as separate trainings by
     v2/run.py, past the warm-up (130 steps), with members of different creatures and use, for
     the table trained and frozen. Both trainers are run with 64-bit numbers here, so that
     rounding cannot hide a real difference: they must agree to about 1e-9.
  2. Clipping is per member: one member with a huge gradient does not change the others.
  3. Rows of creatures not in use stay exactly as they started (32-bit numbers, past step 100).
  4. The sentence maker, read by a reader of our own (not the one in v2/lang.py): every
     sentence is true and has no held-back pair; the shares agree with the reference maker and
     with the chances of PLAN.md B.3; the shorter-list rule with 16 creatures; and a member's
     sentences are the same whatever other members share its group.
  5. Grading: the scores of run_group equal the reference grading of the saved model.pt.
  6. v2/analyze.py reads a folder that holds only results of the group trainer.
  7. Picking up again across a grading point is exact.
"""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import v2.net
from v2 import batch, run
from v2.batch_text import BatchText
from v2.lang import KINDS, LENGTH_CHANCES, TYPE_CHANCES, Language, grade
from v2.net import make_net
from v2.tests.test_v2_batch import chi_square_p

torch.set_num_threads(4)  # the computer is busy with other work; keep to a few threads
ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------------------------
# 1. Training agrees with the reference, past the warm-up
# ------------------------------------------------------------------------------------------------

STEPS = 130  # past the 100 warm-up steps
MEMBERS = [(16, "equal", 0), (64, "zipf", 1), (1024, "zipf", 2)]


def small_configs(table, steps=STEPS, batch_size=64):
    return [{"creatures": n, "kinds": 4, "usage": u, "body": 8, "seed": s, "steps": steps,
             "lr": 0.01, "threads": 1, "table": table, "batch": batch_size}
            for n, u, s in MEMBERS]


def reference_batches(languages, seeds, steps, size):
    """The batches of the reference sentence maker: [members, size, 13] for every step."""
    return [np.stack([lang.sample_batch(seed, step, size) for lang, seed in zip(languages, seeds)])
            for step in range(steps)]


def reference_training(config, lang, batches, member, dtype):
    """One member trained by the code of v2/run.py, step by step as run.train does it."""
    torch.manual_seed(config["seed"])
    model = make_net(lang, config["body"], table_form=config["table"]).to(dtype)
    averaged = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=run.WEIGHT_DECAY)
    for step, tokens in enumerate(batches):
        inputs, targets = run.inputs_and_targets(lang, tokens[member])
        run.train_step(model, optimizer, inputs, targets, run.learning_rate_at(step, config))
        run.update_average(averaged, model, step, config)
    return model, averaged


@pytest.mark.parametrize("table", ["trained", "frozen"])
def test_same_training_as_reference_in_64_bits(tmp_path, monkeypatch, table):
    # Both trainers build their models with 64-bit numbers for this test only.
    plain_make_net = v2.net.make_net
    monkeypatch.setattr(batch, "make_net", lambda *a, **k: plain_make_net(*a, **k).double())
    configs = small_configs(table)
    languages = [Language(n, 4, u) for n, u, _ in MEMBERS]
    batches = reference_batches(languages, [s for *_, s in MEMBERS], STEPS, 64)
    _, state = batch.train_group(configs, tmp_path, "cpu", quiet=True,
                                 batch_maker=lambda step: torch.from_numpy(batches[step]))
    for i, (config, lang) in enumerate(zip(configs, languages)):
        config = run.complete_config(config)
        model, averaged = reference_training(config, lang, batches, i, torch.float64)
        for ours, theirs in ((state["params"], model), (state["averaged"], averaged)):
            assert set(ours) == {name for name, _ in theirs.named_parameters()}
            for name, reference in theirs.named_parameters():
                assert ours[name].dtype == torch.float64
                difference = (ours[name][i] - reference).abs().max().item()
                assert difference < 1e-9, (i, name, difference)


def test_unused_rows_never_move_in_32_bits(tmp_path):
    configs = small_configs("trained", steps=110, batch_size=32)
    _, state = batch.train_group(configs, tmp_path, "cpu", quiet=True)
    for i, (n, usage, _) in enumerate(MEMBERS):
        lang = Language(n, 4, usage)
        start = make_net(lang, 8).word_table.detach()
        unused = slice(lang.rows_in_use, None)
        if n < 1024:
            assert torch.equal(state["params"]["word_table"][i][unused], start[unused])
            assert torch.equal(state["averaged"]["word_table"][i][unused], start[unused])
        # The rows in use did move (so the test is not empty).
        assert not torch.equal(state["averaged"]["word_table"][i][:lang.rows_in_use],
                               start[:lang.rows_in_use])


# ------------------------------------------------------------------------------------------------
# 2. Clipping is per member
# ------------------------------------------------------------------------------------------------


def test_clipping_is_per_member(monkeypatch):
    plain_make_net = v2.net.make_net
    monkeypatch.setattr(batch, "make_net", lambda *a, **k: plain_make_net(*a, **k).double())
    configs = [run.complete_config(c) for c in small_configs("trained", steps=5)]
    languages = [Language(n, 4, u) for n, u, _ in MEMBERS]
    batches = reference_batches(languages, [s for *_, s in MEMBERS], 5, 64)

    def trained(members, blow_up):
        trainer = batch.GroupTrainer([configs[m] for m in members], "cpu")
        if blow_up is not None:  # a member with huge output weights: a huge gradient
            with torch.no_grad():
                trainer.params["output.weight"][blow_up] *= 1e4
        lengths = []
        original = trainer.clip_per_member

        def clip_and_measure():  # notes each member's gradient length before clipping
            grads = [p.grad.reshape(len(members), -1) for p in trainer.params.values()]
            lengths.append(torch.cat(grads, dim=1).norm(dim=1))
            original()

        trainer.clip_per_member = clip_and_measure
        for step in range(5):
            trainer.train_step(torch.from_numpy(batches[step][members]))
        return trainer, lengths

    together, lengths = trained([0, 1, 2], blow_up=1)
    assert all(length[1] > 10 * run.GRADIENT_CLIP for length in lengths)  # really huge
    for m in (0, 2):
        alone, _ = trained([m], blow_up=None)
        for name in alone.params:
            assert (together.params[name][m] - alone.params[name][0]).abs().max() < 1e-12, name
            assert (together.averaged[name][m] - alone.averaged[name][0]).abs().max() < 1e-12

    # The rule itself, on made-up gradients: the same as torch's clip_grad_norm_ per member.
    trainer = batch.GroupTrainer(configs, "cpu")
    generator = torch.Generator().manual_seed(0)
    scale = torch.tensor([1e-3, 1e3, 0.5], dtype=torch.float64)
    for p in trainer.params.values():
        p.grad = torch.randn(p.shape, generator=generator, dtype=p.dtype) * scale.view(
            -1, *([1] * (p.dim() - 1))) / math.sqrt(p[0].numel())
    expected = []
    for m in range(3):
        copies = [p.grad[m].clone().requires_grad_(False) for p in trainer.params.values()]
        holders = [torch.nn.Parameter(torch.zeros_like(c)) for c in copies]
        for holder, c in zip(holders, copies):
            holder.grad = c
        torch.nn.utils.clip_grad_norm_(holders, run.GRADIENT_CLIP)
        expected.append([h.grad for h in holders])
    trainer.clip_per_member()
    for m in range(3):
        for p, want in zip(trainer.params.values(), expected[m]):
            assert torch.allclose(p.grad[m], want, rtol=1e-12, atol=0)


# ------------------------------------------------------------------------------------------------
# 4. The sentence maker, read by our own reader
# ------------------------------------------------------------------------------------------------


def own_reader(lang):
    """A reader of token rows, written from PLAN.md B.3 alone. Returns a function
    ids -> (type, creatures, kind, value, answer) or raises AssertionError."""
    value_of = {}
    for kind, (_, values) in enumerate(KINDS[:lang.kinds]):
        for value, word in enumerate(values):
            value_of[word] = (kind, value)
    creature_of = {name: i for i, name in enumerate(lang.names)}

    def read_list(words):
        # a | a and b | a , b and c | a , b , c and d
        names = words[0::2]
        joins = words[1::2]
        assert len(words) % 2 == 1 and 1 <= len(names) <= 4, words
        assert joins == [","] * (len(names) - 2) + ["and"] if len(names) > 1 else not joins
        creatures = [creature_of[w] for w in names]
        assert all(c < lang.creatures for c in creatures), "a creature not in use"
        assert len(set(creatures)) == len(creatures), "a creature twice"
        return creatures

    def read(ids):
        ids = list(ids)
        while ids and ids[-1] == 0:
            ids.pop()
        assert 0 not in ids
        words = [lang.words[i] for i in ids]
        question = words[-2] == "?"
        if question:
            answer = words[-1]
            assert answer in ("yes", "no")
            body = words[:-2]
            kind, value = value_of[body[-1]]
            if kind == 0:
                assert body[0] == "are"
                creatures = read_list(body[1:-1])
            else:
                link = KINDS[kind][0]
                assert body[0] == "do" and body[-1 - len(link):-1] == link
                creatures = read_list(body[1:-1 - len(link)])
            return (3 if answer == "no" else 2), creatures, kind, value, answer
        assert words[-1] == "."
        body = words[:-1]
        kind, value = value_of[body[-1]]
        if kind == 0:
            negated = body[-3:-1] == ["are", "not"]
            end = -3 if negated else -2
            assert body[end] == "are"
        else:
            link = KINDS[kind][0]
            negated = body[-3 - len(link):-1 - len(link)] == ["do", "not"]
            assert body[-1 - len(link):-1] == link
            end = -3 - len(link) if negated else -1 - len(link)
        return (1 if negated else 0), read_list(body[:end]), kind, value, None

    return read


def check_and_count(lang, rows):
    """Reads every row with our own reader, checks truth and held-back pairs, and counts."""
    read = own_reader(lang)
    counts = {"type": np.zeros(4), "length": np.zeros(4), "type x length": np.zeros(16),
              "kind": np.zeros(lang.kinds), "type x value": np.zeros(16),
              "length x f": np.zeros(16), "creature": np.zeros(lang.creatures)}
    for ids in rows:
        t, creatures, kind, value, answer = read(ids)
        has = [lang.facts[c, kind] == value for c in creatures]
        if t == 0:
            assert all(has)
        elif t == 1:
            assert not any(has)
        else:
            assert answer == ("yes" if all(has) else "no")
        for a in creatures:
            for b in creatures:
                assert not lang.held[a, b], "a held-back pair"
        k = len(creatures)
        counts["type"][t] += 1
        counts["length"][k - 1] += 1
        counts["type x length"][4 * t + k - 1] += 1
        counts["kind"][kind] += 1
        counts["type x value"][4 * t + value] += 1
        if t == 3:
            counts["length x f"][4 * (k - 1) + sum(not h for h in has) - 1] += 1
        for c in creatures:
            counts["creature"][c] += 1
    return counts


def z_score(count, total, chance):
    return (count - total * chance) / math.sqrt(total * chance * (1 - chance))


SETTINGS = [(16, "equal"), (16, "zipf"), (1024, "zipf"), (256, "equal")]
SEEDS = [5, 6, 7, 8]
BATCHES = 24  # 24 batches of 512 = 12,288 sentences per member


@pytest.fixture(scope="module")
def made():
    languages = [Language(n, 4, u) for n, u in SETTINGS]
    maker = BatchText(languages, SEEDS, batch=512, device="cpu")
    parts = [maker.draw(step) for step in range(BATCHES)]
    tokens = torch.cat([p["tokens"] for p in parts], dim=1).numpy()
    return languages, tokens


def test_sentences_true_and_shares_match(made):
    languages, tokens = made
    failures = []
    for m, lang in enumerate(languages):
        ours = check_and_count(lang, tokens[m].tolist())
        reference = lang.draw_sentences(np.random.default_rng([123, m]), tokens.shape[1])
        theirs = check_and_count(lang, reference["tokens"].tolist())
        for what in ours:
            p = chi_square_p(ours[what], theirs[what])
            if p < 1e-4:
                failures.append(f"{lang.creatures} {lang.usage}: {what} p = {p:.2g}")
        # The chances of PLAN.md B.3 (the type never changes; kind and value are equal chance
        # for statements and questions; with 256 or more creatures lists are almost never made
        # shorter, so the lengths keep their chances too).
        total = ours["type"].sum()
        for t in range(4):
            assert abs(z_score(ours["type"][t], total, TYPE_CHANCES[t])) < 4.5
        for kind in range(lang.kinds):
            assert abs(z_score(ours["kind"][kind], total, 1 / lang.kinds)) < 4.5
        for t in (0, 2, 3):
            row = ours["type x value"][4 * t:4 * t + 4]
            for value in range(4):
                assert abs(z_score(row[value], row.sum(), 0.25)) < 4.5
        if lang.creatures >= 256:
            for k in range(4):
                assert abs(z_score(ours["length"][k], total, LENGTH_CHANCES[k])) < 4.5
            # f of "no" questions: equal chance from 1 to k (for k = 1, f is always 1).
            for k in range(2, 5):
                row = ours["length x f"][4 * (k - 1):4 * (k - 1) + k]
                for f in range(k):
                    assert abs(z_score(row[f], row.sum(), 1 / k)) < 4.5
    assert not failures, failures


def test_sixteen_creatures_shorter_lists(made):
    """With 16 creatures each value is held by exactly 4 creatures of a kind, so a statement
    with a list of 4 can only name those 4. Where they contain a held-back pair, such a list
    can never be made and must be made shorter; where they do not, 50 tries always find it."""
    languages, tokens = made
    for m in (0, 1):
        lang = languages[m]
        read = own_reader(lang)
        four_seen = set()
        for ids in tokens[m].tolist():
            t, creatures, kind, value, _ = read(ids)
            if t == 0 and len(creatures) == 4:
                four_seen.add((kind, value))
        for kind in range(lang.kinds):
            for value in range(4):
                group = np.nonzero(lang.facts[:, kind] == value)[0]
                clean = not lang.held[np.ix_(group, group)].any()
                if not clean:
                    assert (kind, value) not in four_seen
                elif lang.usage == "equal":
                    assert (kind, value) in four_seen  # expected about 190 times; never 0


def test_member_sentences_do_not_depend_on_the_group(made):
    """Member i's sentences depend only on its seed, its language and the step: the same when
    it is trained alone or in a group with other members (no information passes between them)."""
    languages, tokens = made
    for m in range(len(languages)):
        alone = BatchText([languages[m]], [SEEDS[m]], batch=512, device="cpu")
        for step in (0, 3):
            expected = tokens[m, step * 512:(step + 1) * 512]
            assert np.array_equal(alone.make(step)[0].numpy(), expected)
    # A different seed gives different sentences.
    other = BatchText([languages[0]], [SEEDS[0] + 1], batch=512, device="cpu")
    assert not np.array_equal(other.make(0)[0].numpy(), tokens[0, :512])


# ------------------------------------------------------------------------------------------------
# 5, 6 and 7. Grading, the files, and picking up again
# ------------------------------------------------------------------------------------------------


def grid_like_configs(steps):
    # Names as v2/grid.py gives them, so analyze.py sees ordinary main8 runs.
    return [{"creatures": n, "kinds": 4, "usage": u, "body": 8, "seed": 0, "steps": steps,
             "lr": 0.01, "threads": 1, "batch": 64} for n, u in ((16, "equal"), (64, "zipf"))]


def test_grading_and_files_as_reference(tmp_path):
    stage = tmp_path / "runs" / "main8"
    summaries = batch.run_group(grid_like_configs(150), str(stage), "cpu")
    assert [p["step"] for p in summaries[0]["curve"]] == [100, 141, 150]
    for summary in summaries:
        config = summary["config"]
        saved = torch.load(stage / summary["name"] / "model.pt", weights_only=True)
        lang = Language(config["creatures"], config["kinds"], config["usage"])
        model = make_net(lang, config["body"], table_form=config["table"])
        model.load_state_dict(saved["model"])  # the reference model reads model.pt as it is
        model.eval()
        graded = grade(lang, run.predictor(model))
        for key, value in graded.items():
            assert summary["final"][key] == pytest.approx(value, abs=1e-9), key
        assert summary["curve"][-1]["facts"] == summary["final"]["facts"]

    # analyze.py, unchanged, reads a folder that holds only these results.
    report = tmp_path / "report"
    done = subprocess.run([sys.executable, str(ROOT / "v2" / "analyze.py"), "--runs-root",
                           str(tmp_path / "runs"), "--report-dir", str(report)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    facts = json.loads((report / "facts.json").read_text())
    assert facts["runs_found"]["main8"] == 2


def test_pick_up_again_across_a_grading(tmp_path, monkeypatch):
    configs = grid_like_configs(150)
    whole, whole_state = batch.train_group(configs, tmp_path / "a", "cpu", quiet=True)
    monkeypatch.setattr(batch, "CHECKPOINT_EVERY", 120)  # after the gradings at 100 and 141 ...
    stopped, _ = batch.train_group(configs, tmp_path / "b", "cpu", stop_after=120, quiet=True)
    assert stopped is None
    resumed, resumed_state = batch.train_group(configs, tmp_path / "b", "cpu", quiet=True)
    for name in whole_state["params"]:
        assert torch.equal(whole_state["params"][name], resumed_state["params"][name])
        assert torch.equal(whole_state["averaged"][name], resumed_state["averaged"][name])
    for a, b in zip(whole, resumed):
        assert a["curve"] == b["curve"] and a["final"] == b["final"]

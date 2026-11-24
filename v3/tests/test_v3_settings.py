"""Tests of the third experiment's new settings in the v2 programs (v3/PLAN.md, Part B.1).

Everything runs on the processor with few threads and tiny models (well under a minute in all).

  1. 64 kinds: every word different, true sentences (both sentence makers), an oracle scores 100.
  2. The nested 64-column table: a word starts from the same numbers at every column count.
  3. text_seed: runs with different seeds and the same text_seed read the same text; the same
     seed with a different text_seed does not (one-model trainer and group trainer).
  4. The doubling length rule on made-up curves, and in a short run of v2/run.py.
  5. A member trained in a group with other rates and with members that end early ends exactly
     like the same member trained alone.
  6. Stopping on request (a deadline), and picking up again: the same end as without a stop.
"""

import json
import time

import numpy as np
import pytest
import torch

from v2 import batch, net, run
from v2 import lang as L
from v2.batch_text import BatchText

torch.set_num_threads(2)  # the computer is busy with other work


# ------------------------------------------------------------------------------------------------
# 1. 64 kinds
# ------------------------------------------------------------------------------------------------


def test_64_kinds_words_are_all_different():
    lang = L.Language(1024, 64)
    assert len(L.ALL_KINDS) == 64 and L.ALL_KINDS[:32] == L.KINDS
    assert len(set(lang.words)) == len(lang.words)
    names = set(L.creature_names())
    assert not any(word in names for word in lang.words[:lang.creature_base])
    # The new kinds follow the pattern of kinds 5 to 32: one linking word, four values.
    assert all(len(link) == 1 and len(values) == 4 for link, values in L.MORE_KINDS)
    # Every word of 4 to 32 kinds keeps its token id.
    small = L.Language(16, 32)
    assert lang.words[:small.creature_base] == small.words[:small.creature_base]


def oracle(lang):
    """A predictor that knows every fact (as in v2/tests, but for all 64 kinds)."""
    yes, no = lang.yes_id, lang.no_id

    def predict(prompts, last):
        scores = np.zeros((len(prompts), lang.n_out))
        for i, (row, end) in enumerate(zip(prompts, last)):
            row = [int(t) for t in row[:end + 1]]
            if lang.words[row[-1]] == "?":
                read = L.read_sentence(lang, row + [yes])
                every = all(lang.facts[c, read["kind"]] == read["value"]
                            for c in read["creatures"])
                scores[i, (yes if every else no) - 1] = 1.0
                continue
            creature = row[0] - lang.creature_base
            words = [lang.words[t] for t in row[1:]]
            negated = "not" in words
            link = [w for w in words if w not in ("do", "not")]
            kind = [k for k in range(lang.kinds) if L.ALL_KINDS[k][0] == link][0]
            true_value = lang.value_ids[kind, lang.facts[creature, kind]]
            if negated:
                scores[i, lang.value_ids[kind] - 1] = 1.0
                scores[i, true_value - 1] = 0.0
            else:
                scores[i, true_value - 1] = 1.0
        return scores

    return predict


def test_64_kinds_sentences_are_true_and_the_oracle_scores_100():
    lang = L.Language(16, 64, "zipf")
    made = lang.draw_sentences(np.random.default_rng([5]), 3000)
    assert all(L.is_true(lang, row) for row in made["tokens"])
    assert set(made["kinds"].tolist()) == set(range(64))  # the new kinds are used too
    maker = BatchText([lang, L.Language(64, 64)], [0, 1], batch=1024, device="cpu")
    tokens = maker.make(3).numpy()
    for m, member in enumerate([lang, L.Language(64, 64)]):
        assert all(L.is_true(member, row) for row in tokens[m])
    best = L.grade(lang, oracle(lang))
    for key in ("facts", "questions_one", "questions_new", "questions_seen", "not_known"):
        assert best[key] == 100.0, key
    assert best["facts_by_kind"] == [100.0] * 64


def test_64_kinds_need_the_nested_table():
    lang = L.Language(16, 64)
    with pytest.raises(ValueError):
        net.make_net(lang, 4)  # the second experiment's table has no rows for kinds 33 to 64
    model = net.make_net(lang, 4, columns=16, table_start="nested64")
    assert model.word_table.shape == (lang.n_tokens, 16)


# ------------------------------------------------------------------------------------------------
# 2. The nested table
# ------------------------------------------------------------------------------------------------


def test_nested_table_same_numbers_at_every_column_count():
    full = net.nested_word_table()
    assert full.shape == (L.NESTED_TOKENS, 64)
    assert abs(float(full.std()) - 1.0) < 0.05
    for kinds in (4, 16, 64):
        for creatures in (16, 1024):
            lang = L.Language(creatures, kinds)
            tables = {d: net.word_table(lang, d, "nested64") for d in (8, 16, 32, 64)}
            for d, table in tables.items():
                assert np.array_equal(table, tables[64][:, :d])
            # The same word has the same row whatever the kinds and creatures.
            for word in ("are", "red", "foxes", lang.names[15]):
                row = tables[64][lang.words.index(word)]
                other = L.Language(16, 4)
                assert np.array_equal(row, net.word_table(other, 64, "nested64")[
                    other.words.index(word)])
    # The model starts from these rows.
    lang = L.Language(16, 8)
    model = net.make_net(lang, 4, columns=8, table_start="nested64")
    assert np.array_equal(model.word_table.detach().numpy(), net.word_table(lang, 8, "nested64"))
    # Without table_start nothing changes.
    assert np.array_equal(net.make_net(lang, 4).word_table.detach().numpy(),
                          net.word_table(lang))


# ------------------------------------------------------------------------------------------------
# 3. text_seed
# ------------------------------------------------------------------------------------------------

TINY = {"creatures": 16, "kinds": 4, "usage": "equal", "body": 4, "steps": 12, "lr": 0.01,
        "threads": 1, "batch": 16, "columns": 8, "table_start": "nested64"}


def texts_read_by_run_one(monkeypatch, tmp_path, configs):
    """The batches each run of v2/run.py reads, recorded step by step."""
    read = []
    original = L.Language.sample_batch

    def recording(self, seed, step, size=512):
        tokens = original(self, seed, step, size)
        read[-1].append(tokens.copy())
        return tokens

    monkeypatch.setattr(L.Language, "sample_batch", recording)
    for i, config in enumerate(configs):
        read.append([])
        run.train(config, str(tmp_path / str(i)), quiet=True)
    return [np.stack(r) for r in read]


def test_text_seed_in_run_one(monkeypatch, tmp_path):
    a, b, c = texts_read_by_run_one(monkeypatch, tmp_path, [
        dict(TINY, seed=0, text_seed=0), dict(TINY, seed=1, text_seed=0),
        dict(TINY, seed=1, text_seed=5)])
    assert np.array_equal(a, b)
    assert not np.array_equal(b, c)
    # The seed still sets the starting weights.
    name = lambda seed: run.complete_config(dict(TINY, seed=seed, text_seed=0))["name"]
    one = torch.load(tmp_path / "0" / name(0) / "model.pt", weights_only=False)["model"]
    two = torch.load(tmp_path / "1" / name(1) / "model.pt", weights_only=False)["model"]
    assert not torch.equal(one["output.weight"], two["output.weight"])
    # Without text_seed, the seed sets the text (the second experiment).
    plain_a, plain_b = texts_read_by_run_one(monkeypatch, tmp_path / "plain", [
        {k: v for k, v in TINY.items() if k != "table_start"} | {"seed": 0},
        {k: v for k, v in TINY.items() if k != "table_start"} | {"seed": 1}])
    assert not np.array_equal(plain_a, plain_b)


def test_text_seed_in_the_group_trainer():
    configs = [run.complete_config(dict(TINY, seed=s, text_seed=t, name=f"m{i}"))
               for i, (s, t) in enumerate(((0, 0), (1, 0), (1, 7)))]
    trainer = batch.GroupTrainer(configs, torch.device("cpu"))
    tokens = trainer.make_text(4)
    assert torch.equal(tokens[0], tokens[1])
    assert not torch.equal(tokens[1], tokens[2])
    # A member's text does not depend on the group, nor on the sentences being made once for
    # members that share them.
    alone = batch.GroupTrainer(configs[2:], torch.device("cpu")).make_text(4)
    assert torch.equal(alone[0], tokens[2])
    assert torch.equal(batch.GroupTrainer(configs[:1], torch.device("cpu")).make_text(4)[0],
                       tokens[1])
    zipf = run.complete_config(dict(TINY, seed=0, text_seed=0, usage="zipf", name="z"))
    mixed = batch.GroupTrainer(configs + [zipf], torch.device("cpu")).make_text(4)
    assert torch.equal(mixed[:3], tokens) and not torch.equal(mixed[3], tokens[0])


# ------------------------------------------------------------------------------------------------
# 4. The doubling rule
# ------------------------------------------------------------------------------------------------

RULE = run.complete_config(dict(TINY, seed=0, steps=640000, length_rule="doubling"))


def curve(points):
    """A made-up curve: {step: (facts, questions_one)}."""
    return [{"step": s, "facts": f, "questions_one": q} for s, (f, q) in sorted(points.items())]


def test_looks_and_gradings_of_the_rule():
    assert run.looks(RULE) == [40000, 80000, 160000, 320000, 640000]
    points = run.grading_points(RULE)
    assert {10000, 20000, 40000, 80000, 160000, 320000, 640000} <= set(points)
    assert set(run.grading_steps(640000)) <= set(points)
    assert run.looks(run.complete_config(dict(TINY, seed=0))) == []  # a fixed length
    with pytest.raises(ValueError):  # the limit must be the first look doubled
        run.complete_config(dict(TINY, seed=0, steps=100000, length_rule="doubling"))


def test_the_rule_on_made_up_curves():
    # Learned at the first look.
    learned = curve({10000: (60, 20), 20000: (90, 70), 40000: (99.5, 99)})
    assert run.end_reason(RULE, 40000, learned) == "learned"
    # Not at a step that is not a look.
    assert run.end_reason(RULE, 20000, learned) is None
    # Facts 99 but questions below 99: not learned; but still rising, so it goes on.
    almost = curve({10000: (60, 20), 20000: (90, 70), 40000: (99.5, 98)})
    assert run.end_reason(RULE, 40000, almost) is None
    # No more gain: less than 1 point over each of the last two doublings.
    flat = curve({10000: (50.0, 0), 20000: (50.6, 0), 40000: (51.2, 0)})
    assert run.end_reason(RULE, 40000, flat) == "no more gain"
    # A one-doubling stall is NOT ended: the doubling before it gained a point or more.
    stall = curve({20000: (40, 0), 40000: (60, 0), 80000: (60.4, 0)})
    assert run.end_reason(RULE, 80000, stall) is None
    # ... and if the next doubling stalls too, it is.
    stall2 = curve({20000: (40, 0), 40000: (60, 0), 80000: (60.4, 0), 160000: (60.9, 0)})
    assert run.end_reason(RULE, 160000, stall2) == "no more gain"
    # Exactly one point is a gain (it must be LESS than 1 to count as none).
    one_point = curve({10000: (50, 0), 20000: (51, 0), 40000: (52, 0)})
    assert run.end_reason(RULE, 40000, one_point) is None
    # The limit: still rising at 640,000.
    rising = curve({160000: (50, 0), 320000: (60, 0), 640000: (70, 0)})
    assert run.end_reason(RULE, 640000, rising) == "limit"
    # Learned at the limit is "learned", not "limit".
    done = curve({160000: (50, 0), 320000: (60, 0), 640000: (100, 100)})
    assert run.end_reason(RULE, 640000, done) == "learned"


def test_run_one_with_the_rule(tmp_path, monkeypatch):
    config = dict(TINY, seed=0, text_seed=0, steps=160, first_look=40, length_rule="doubling")
    # A made-up rule decision, so the test does not depend on what a tiny model learns: the run
    # ends at its second look.
    monkeypatch.setattr(run, "end_reason",
                        lambda c, step, curve: "no more gain" if step == 80 else None)
    summary = run.run_one(config, str(tmp_path))
    assert summary["ended"] == {"step": 80, "reason": "no more gain"}
    assert summary["steps"] == 80 and summary["curve"][-1]["step"] == 80
    assert [p["step"] for p in summary["curve"]] == [10, 20, 40, 80]
    assert summary["final"]["facts"] == summary["curve"][-1]["facts"]
    on_disk = json.loads((tmp_path / summary["name"] / "summary.json").read_text())
    assert on_disk == summary
    assert (tmp_path / summary["name"] / "model.pt").exists()
    # A run with the defaults has no "ended".
    plain = run.run_one(dict(TINY, seed=0, steps=20), str(tmp_path / "plain"))
    assert "ended" not in plain


# ------------------------------------------------------------------------------------------------
# 5. Different rates, and members that end early
# ------------------------------------------------------------------------------------------------


def rule_config(name, creatures, usage, seed, lr):
    return {"creatures": creatures, "kinds": 4, "usage": usage, "body": 4, "seed": seed,
            "steps": 160, "lr": lr, "threads": 1, "batch": 32, "columns": 8, "text_seed": 0,
            "table_start": "nested64", "length_rule": "doubling", "first_look": 40,
            "name": name}


A = rule_config("A", 16, "equal", 0, 0.01)
B = rule_config("B", 64, "zipf", 1, 0.03)
C = rule_config("C", 16, "zipf", 2, 0.003)


def without_time(summary):
    return {k: v for k, v in summary.items() if k not in ("minutes", "steps_per_second")}


def same_files(folder_a, folder_b, name):
    a = torch.load(folder_a / name / "model.pt", weights_only=False)
    b = torch.load(folder_b / name / "model.pt", weights_only=False)
    return a["config"] == b["config"] and all(torch.equal(a["model"][k], b["model"][k])
                                              for k in a["model"])


def test_member_in_a_mixed_group_ends_as_alone(tmp_path, monkeypatch):
    real = run.end_reason

    def rule(config, step, curve):  # C ends at its first look, whatever it scored
        return "made up" if config["name"] == "C" and step == 40 else real(config, step, curve)

    monkeypatch.setattr(batch, "end_reason", rule)
    together, _ = batch.train_group([A, B, C], tmp_path / "together", "cpu", quiet=True)
    alone, _ = batch.train_group([A], tmp_path / "alone", "cpu", quiet=True)
    assert without_time(together[0]) == without_time(alone[0])
    assert same_files(tmp_path / "together", tmp_path / "alone", "A")
    assert together[2]["ended"] == {"step": 40, "reason": "made up"}
    assert together[2]["steps"] == 40
    # B, alone, also ends as in the group (and with another rate than A).
    alone_b, _ = batch.train_group([B], tmp_path / "alone_b", "cpu", quiet=True)
    assert without_time(together[1]) == without_time(alone_b[0])
    assert all("ended" in s and s["trainer"] == "batch" for s in together)
    assert not list((tmp_path / "together" / "_groups").glob("*/checkpoint.pt"))


def test_own_adamw_is_torch_adamw():
    """One step of a group with two rates equals torch.optim.AdamW, member by member."""
    configs = [run.complete_config(c) for c in (A, B)]
    trainer = batch.GroupTrainer(configs, torch.device("cpu"), own_adamw=True)
    reference = [{k: v[i].detach().clone().requires_grad_(True)
                  for k, v in trainer.params.items()} for i in range(2)]
    optimizers = [torch.optim.AdamW(list(r.values()), lr=c["lr"], weight_decay=0.0)
                  for r, c in zip(reference, configs)]
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        grads = {k: torch.randn(v.shape, generator=generator) for k, v in trainer.params.items()}
        for k, p in trainer.params.items():
            p.grad = grads[k].clone()
        trainer.own_adamw_step([c["lr"] for c in configs])
        for i, (r, optimizer) in enumerate(zip(reference, optimizers)):
            for k, p in r.items():
                p.grad = grads[k][i].clone()
            optimizer.step()
    for i in range(2):
        for k in trainer.params:
            assert torch.equal(trainer.params[k][i], reference[i][k]), k


# ------------------------------------------------------------------------------------------------
# 6. Stopping on request and picking up again
# ------------------------------------------------------------------------------------------------


def short_sittings(monkeypatch, whole_seconds, parts=5):
    """Make every sitting with a deadline end after about 1/parts of the time the whole training
    took, counted from the moment the sitting starts training. (A deadline a fixed 1.5 s ahead
    no longer stops anything: the group now trains in under a second, and setting up a sitting
    takes part of that time.) The clock's own guesses are made small too."""
    monkeypatch.setattr(batch, "DEADLINE_SAFETY_SECONDS", 0.0)
    monkeypatch.setattr(batch, "FIRST_STEP_GUESS_SECONDS", 0.001)
    monkeypatch.setattr(batch, "FIRST_SAVE_GUESS_SECONDS", 0.001)
    monkeypatch.setattr(batch, "FIRST_GRADING_GUESS_SECONDS", 0.001)
    budget = max(whole_seconds / parts, 0.05)
    real = batch.DeadlineClock

    class Clock(real):
        def __init__(self, deadline):
            super().__init__(None if deadline is None else time.time() + budget)

    monkeypatch.setattr(batch, "DeadlineClock", Clock)


@pytest.mark.parametrize("rates", ["mixed", "one"])
def test_stop_on_request_and_pick_up(tmp_path, monkeypatch, rates):
    configs = [dict(c, steps=320, first_look=80) for c in (A, B, C)]
    if rates == "one":  # one rate: torch's AdamW over the whole group (as in the counted runs)
        configs = [dict(c, lr=0.01) for c in configs]
    started = time.perf_counter()
    whole, _ = batch.train_group(configs, tmp_path / "whole", "cpu", quiet=True)

    # A short deadline stops the group in the middle, again and again.
    short_sittings(monkeypatch, time.perf_counter() - started)
    stops = 0
    result = None
    while result is None or result["unfinished"]:
        result = batch.run_group_until(configs, tmp_path / "parts", "cpu",
                                       deadline=time.time() + 1.5, quiet=True)
        stops += 1
        assert stops < 60
        if result["unfinished"]:
            assert list((tmp_path / "parts" / "_groups").glob("*/checkpoint.pt"))
            assert set(result["unfinished"]) | set(result["finished"]) == {"A", "B", "C"}
    assert stops > 1, "the deadline never stopped the group; the test proves nothing"
    for summary in whole:
        again = json.loads((tmp_path / "parts" / summary["name"] / "summary.json").read_text())
        assert without_time(again) == without_time(summary)
        assert same_files(tmp_path / "whole", tmp_path / "parts", summary["name"])
    assert not list((tmp_path / "parts" / "_groups").glob("*/checkpoint.pt"))

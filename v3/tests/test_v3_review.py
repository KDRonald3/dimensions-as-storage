"""Tests added by the review of the third experiment's programs (v3/PLAN.md, Part B).

  1. The rate chooser: ONE rate for the whole experiment, averaged over all 80 practice settings.
  2. carry_on: a marked run that ends as "no more gain" is trained two more doublings, but its
     result (final, ended, steps, model.pt) stays that of the point where the rule ended it; in
     both trainers, also when the group is stopped and picked up while carrying on.
  3. A member leaving a group with ONE rate (torch's AdamW, as in the counted runs) does not
     disturb the others.
  4. The text of a language is the same whatever the columns, body width and rate; the
     starting weights still differ by seed.

Everything runs on the processor with tiny models and few threads.
"""

import json
import time

import pytest
import torch

from v2 import batch, run
from v3 import choose_rates, grid

torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------
# 1. The rate chooser
# ------------------------------------------------------------------------------------------------


def made_up(scores):
    """Practice summaries with made-up scores: scores(config) -> (facts, questions_one)."""
    found = {}
    for config in grid.practice():
        facts, one = scores(config)
        found[config["name"]] = {"name": config["name"], "config": config,
                                 "final": {"facts": facts, "questions_one": one}}
    return found


def my_choice(summaries):
    """The rule as I read B.2, written separately."""
    rates = sorted({c["lr"] for c in grid.practice()})
    facts, one = {}, {}
    for rate in rates:
        finals = [summaries[c["name"]]["final"] for c in grid.practice() if c["lr"] == rate]
        assert len(finals) == 80
        facts[rate] = sum(f["facts"] for f in finals) / 80
        one[rate] = sum(f["questions_one"] for f in finals) / 80
    best = max(facts.values())
    tied = [r for r in rates if best - facts[r] <= 0.5 + 1e-9]
    top = max(one[r] for r in tied)
    return min(r for r in tied if one[r] == top)


def test_one_rate_for_all_from_80_settings():
    # 0.01 best in most combinations, but 0.03 far better in one: the average over all 80
    # settings decides, so it is one rate for everything.
    def scores(c):
        if c["columns"] == 8 and c["kinds"] == 4:
            return {0.003: 0, 0.01: 0, 0.03: 100}[c["lr"]], 0
        return {0.003: 50, 0.01: 60, 0.03: 58}[c["lr"]], 0
    chosen = choose_rates.choose(made_up(scores))
    # 0.01 averages 57, 0.03 averages 0.95 * 58 + 0.05 * 100 = 60.1
    assert chosen["rate"] == 0.03 == my_choice(made_up(scores))


@pytest.mark.parametrize("case", range(40))
def test_rate_chooser_agrees_with_my_reading(case):
    import random
    rng = random.Random(case)
    base = {r: rng.choice([50, 50.25, 50.5, 50.75, 51]) for r in (0.003, 0.01, 0.03)}
    q = {r: rng.choice([10, 20]) for r in (0.003, 0.01, 0.03)}
    summaries = made_up(lambda c: (base[c["lr"]], q[c["lr"]]))
    assert choose_rates.choose(summaries)["rate"] == my_choice(summaries)


def test_settings_file_feeds_the_counted_grid(tmp_path):
    practice_dir = tmp_path / "practice"
    for name, summary in made_up(lambda c: ({0.003: 70, 0.01: 80, 0.03: 79.6}[c["lr"]],
                                            {0.003: 0, 0.01: 10, 0.03: 30}[c["lr"]])).items():
        (practice_dir / name).mkdir(parents=True)
        (practice_dir / name / "summary.json").write_text(json.dumps(summary))
    settings_path = tmp_path / "settings.json"
    choose_rates.main(["--practice-dir", str(practice_dir), "--settings", str(settings_path)])
    settings = json.loads(settings_path.read_text())
    assert settings["rate"] == 0.03  # tied within half a point; better questions about one
    runs = grid.counted(settings)
    assert {c["lr"] for c in runs} == {0.03}
    (practice_dir / grid.practice()[5]["name"] / "summary.json").unlink()
    with pytest.raises(ValueError):
        choose_rates.main(["--practice-dir", str(practice_dir), "--settings",
                           str(tmp_path / "other.json")])


# ------------------------------------------------------------------------------------------------
# 2. carry_on
# ------------------------------------------------------------------------------------------------


def member(name, creatures, usage, seed, lr=0.01, **more):
    config = {"creatures": creatures, "kinds": 4, "usage": usage, "body": 4, "seed": seed,
              "steps": 320, "lr": lr, "threads": 1, "batch": 32, "columns": 8, "text_seed": 0,
              "table_start": "nested64", "length_rule": "doubling", "first_look": 40,
              "name": name}
    config.update(more)
    return config


A = member("A", 16, "equal", 0, carry_on=True)
B = member("B", 64, "zipf", 1)


REAL_END_REASON = run.end_reason


def stall_rule(config, step, curve):
    """A made-up rule decision: A is ended as "no more gain" at its first look (40)."""
    if config["name"] == "A" and step == 40:
        return "no more gain"
    return REAL_END_REASON(config, step, curve)


def weights(path):
    return torch.load(path, weights_only=False)["model"]


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def without_time(summary):
    return {k: v for k, v in summary.items() if k not in ("minutes", "steps_per_second")}


def check_carried(summary, curve_steps_beyond=160):
    assert summary["ended"] == {"step": 40, "reason": "no more gain"}
    assert summary["steps"] == 40
    at_40 = [p for p in summary["curve"] if p["step"] == 40][0]
    assert summary["final"]["facts"] == at_40["facts"]
    assert summary["carried_on"]["step"] == curve_steps_beyond == summary["curve"][-1]["step"]
    assert summary["carried_on"]["why"] == "two more doublings"
    last = summary["curve"][-1]
    assert summary["carried_on"]["facts"] == last["facts"]
    assert summary["carried_on"]["gain"] == last["facts"] - at_40["facts"]
    # The evidence of a "no more gain" run (PLAN.md): facts at a quarter, a half and the whole of
    # its length (40), and the fall of the training loss over the last doubling.
    at = {p["step"]: p for p in summary["curve"]}
    evidence = summary["stop_evidence"]
    assert (evidence["facts_quarter"], evidence["facts_half"], evidence["facts_whole"]) == (
        at[10]["facts"], at[20]["facts"], at[40]["facts"])
    assert evidence["loss_fall"] == at[20]["loss"] - at[40]["loss"]


def test_carry_on_in_the_group_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "end_reason", stall_rule)
    together, _ = batch.train_group([A, B], tmp_path / "carry", "cpu", quiet=True)
    check_carried(together[0])
    # The result is the one of step 40: the same weights as A without carry_on, ended there.
    plain_a = {k: v for k, v in A.items() if k != "carry_on"}
    alone, _ = batch.train_group([plain_a], tmp_path / "plain", "cpu", quiet=True)
    assert alone[0]["ended"] == {"step": 40, "reason": "no more gain"}
    assert same_weights(weights(tmp_path / "carry" / "A" / "model.pt"),
                        weights(tmp_path / "plain" / "A" / "model.pt"))
    assert together[0]["final"] == alone[0]["final"]
    # B is not disturbed by A carrying on.
    b_alone, _ = batch.train_group([B], tmp_path / "b", "cpu", quiet=True)
    assert without_time(together[1]) == without_time(b_alone[0])


def test_carry_on_stop_and_pick_up_is_exact(tmp_path, monkeypatch):
    from v3.tests.test_v3_settings import short_sittings
    monkeypatch.setattr(batch, "end_reason", stall_rule)
    started = time.perf_counter()
    whole, _ = batch.train_group([A, B], tmp_path / "whole", "cpu", quiet=True)
    whole_seconds = time.perf_counter() - started
    # Stopped in the middle of the carrying on (step 100), then picked up.
    monkeypatch.setattr(batch, "CHECKPOINT_EVERY", 50)
    stopped, state = batch.train_group([A, B], tmp_path / "parts", "cpu", quiet=True,
                                       stop_after=100)
    assert stopped is None and "carry" in state
    again, _ = batch.train_group([A, B], tmp_path / "parts", "cpu", quiet=True)
    for one, two in zip(whole, again):
        assert without_time(one) == without_time(two)
        name = one["name"]
        assert same_weights(weights(tmp_path / "whole" / name / "model.pt"),
                            weights(tmp_path / "parts" / name / "model.pt"))
    # And by a deadline (stop on request), many times over.
    short_sittings(monkeypatch, whole_seconds)
    result, sittings = None, 0
    while result is None or result["unfinished"]:
        result = batch.run_group_until([A, B], tmp_path / "deadline", "cpu",
                                       deadline=time.time() + 2, quiet=True)
        sittings += 1
        assert sittings < 80
    assert sittings > 1
    for one in whole:
        two = json.loads((tmp_path / "deadline" / one["name"] / "summary.json").read_text())
        assert without_time(one) == without_time(two)


def test_carry_on_in_run_one(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "end_reason", stall_rule)
    monkeypatch.setattr(run, "CHECKPOINT_EVERY", 50)
    summary = run.run_one(dict(A), str(tmp_path / "whole"))
    check_carried(summary)
    plain = run.run_one({k: v for k, v in A.items() if k != "carry_on"}, str(tmp_path / "plain"))
    assert same_weights(weights(tmp_path / "whole" / "A" / "model.pt"),
                        weights(tmp_path / "plain" / "A" / "model.pt"))
    assert plain["final"] == summary["final"]
    # Stopped while carrying on, and picked up: the same.
    assert run.train(dict(A), str(tmp_path / "parts"), stop_after=100, quiet=True) is None
    again = run.run_one(dict(A), str(tmp_path / "parts"))
    assert without_time(again) == without_time(summary)
    assert same_weights(weights(tmp_path / "whole" / "A" / "model.pt"),
                        weights(tmp_path / "parts" / "A" / "model.pt"))


def test_carry_on_settings():
    with pytest.raises(ValueError):  # only with the doubling rule
        run.complete_config({k: v for k, v in A.items()
                             if k not in ("length_rule", "first_look")})
    with pytest.raises(ValueError):
        run.complete_config(dict(A, carry_on="yes"))
    # At the limit there is nothing to carry on to.
    assert not run.carries_on(run.complete_config(A), 320, "no more gain")
    assert run.carry_until(run.complete_config(A), 40) == 160
    assert run.carry_until(run.complete_config(A), 160) == 320


# ------------------------------------------------------------------------------------------------
# 3. A member leaving a group with one rate
# ------------------------------------------------------------------------------------------------


def test_member_leaving_a_one_rate_group(tmp_path, monkeypatch):
    early = member("E", 16, "zipf", 2)

    def rule(config, step, curve):
        return "learned" if config["name"] == "E" and step == 40 else REAL_END_REASON(
            config, step, curve)

    monkeypatch.setattr(batch, "end_reason", rule)
    together, _ = batch.train_group([early, B], tmp_path / "together", "cpu", quiet=True)
    alone, _ = batch.train_group([B], tmp_path / "alone", "cpu", quiet=True)
    assert together[0]["ended"] == {"step": 40, "reason": "learned"}
    assert without_time(together[1]) == without_time(alone[0])
    assert same_weights(weights(tmp_path / "together" / "B" / "model.pt"),
                        weights(tmp_path / "alone" / "B" / "model.pt"))


# ------------------------------------------------------------------------------------------------
# 4. The text is that of the language only
# ------------------------------------------------------------------------------------------------


def test_text_is_the_same_across_columns_body_and_rate():
    base = run.complete_config(member("x", 64, "zipf", 0))
    texts = []
    for columns, body, lr, seed in ((8, 4, 0.01, 0), (64, 128, 0.03, 1), (16, 32, 0.003, 2)):
        config = dict(base, columns=columns, body=body, lr=lr, seed=seed, name=f"m{columns}")
        trainer = batch.GroupTrainer([config], torch.device("cpu"))
        texts.append(trainer.make_text(7)[0])
    assert all(torch.equal(texts[0], t) for t in texts[1:])
    # (The one-model trainer makes its sentences another way, v2/BATCH.md; its text across
    # seeds is tested in test_v3_settings.py.)
    # The starting weights differ by seed (same shapes: compare two seeds at one size).
    one = batch.GroupTrainer([dict(base, seed=0, name="a"), dict(base, seed=1, name="b")],
                             torch.device("cpu"))
    assert not torch.equal(one.params["output.weight"][0], one.params["output.weight"][1])
    # ... but not the word table, which is the nested table for every seed.
    assert torch.equal(one.params["word_table"][0], one.params["word_table"][1])


# ------------------------------------------------------------------------------------------------
# 5. The orchestrator gives up on a group that never moves (instead of using up the quota)
# ------------------------------------------------------------------------------------------------


class StuckKaggle:
    """A pretend Kaggle whose sessions all end normally but where the group gets nowhere: its
    checkpoint is copied in and out again at the same step."""

    user = "someone"

    def hide(self, text):
        return text

    def output(self, name, folder):
        from v3.orchestrate import Orchestrator
        out = folder / "out"
        (out / "_groups" / Orchestrator.checkpoint_folder(GROUP)).mkdir(parents=True)
        (out / "_groups" / Orchestrator.checkpoint_folder(GROUP) / "checkpoint.pt").write_text("")
        (out / "session.json").write_text("{}")
        (out / "status.json").write_text(json.dumps({
            "minutes": 1, "speed": {}, "finished": [], "failed_groups": [],
            "unfinished_groups": [{"group": 0, "unfinished": ["A"], "started": True,
                                   "step": 100, "checkpoint": True}]}))
        return True, ""


GROUP = [A, B]


def test_a_group_that_never_moves_is_given_up(tmp_path):
    from v3 import orchestrate
    orchestrator = orchestrate.Orchestrator("test", StuckKaggle(), tmp_path / "runs",
                                            tmp_path / "logs", 0.1, 1, 0, quiet=True)
    known = {"next_number": 1, "groups": [{"names": ["A", "B"], "source": None, "failures": 0,
                                           "given_up": False}]}
    for number in range(1, 5):
        session = {"stage": "test", "groups": [0], "state": "running", "hours": 0.1}
        orchestrator.collect(f"s{number}", session, "complete", [GROUP], known)
    # The first session moved it to step 100 (progress); the next three did not.
    assert known["groups"][0]["failures"] == 3 and known["groups"][0]["given_up"]

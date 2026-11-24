"""Tests of the scheduling of the fleet (v3/fleet_plan.py and the planning in v3/colab_fleet.py):
speeds, the end of the stage, which groups move where, and the fleet run with the pretend Colab
tool when cards come and go. Processor only.

  1. The arithmetic: work still to do, speeds (measured before guessed), the planner.
  2. Planning on a fleet whose machines are only pretended (no training): a faster machine takes
     waiting groups; a slow machine is stopped in the middle of its rounds when that clearly
     helps; nothing moves when it would not help; a slow machine is given back when the stage
     does not end later without it.
  3. Whole runs with the pretend Colab tool (real training): availability changing over time
     with a faster card appearing mid-stage; Colab's limit on machines; taking over the state
     file of the launcher as it was before (adoption).
"""

import ast
import json
import re
import time

from v3 import colab_fleet, fleet_plan
from v3.tests.test_v3_fleet import fake, make_fleet, run_until_done, tiny_groups  # noqa: F401
from v3.tests.test_v3_fleet_review import add_owners_session, check_results


# ------------------------------------------------------------------------------------------------
# 1. The arithmetic
# ------------------------------------------------------------------------------------------------


def counted_config(columns=8, kinds=4, body=128):
    return {"name": f"c{columns}_k{kinds}_b{body}", "columns": columns, "kinds": kinds,
            "body": body, "steps": 640000, "first_look": 40000, "length_rule": "doubling"}


def test_work_left():
    config = counted_config()
    # All go on: 24 members x 640,000 steps.
    assert fleet_plan.work_left(config, 0, 24, survival=1.0) == 24 * 640000
    # Half go on after each look: 24 x 40,000, then 12 x 40,000, 6 x 80,000, ...
    expected = 24 * 40000 + 12 * 40000 + 6 * 80000 + 3 * 160000 + 1.5 * 320000
    assert fleet_plan.work_left(config, 0, 24, survival=0.5) == expected
    # In the middle of the last round, with part of it done already.
    assert fleet_plan.work_left(config, 320000, 2, survival=0.5, done_in_round=320000) == 320000
    # Never estimated as nothing while it still trains.
    assert fleet_plan.work_left(config, 320000, 2, done_in_round=10 ** 9) > 0


def test_speeds_measured_before_guessed():
    speeds = fleet_plan.Speeds()
    shape = "8x4x128"
    assert speeds.rate("A100", shape) == 6600  # the A100 guess
    assert speeds.rate("T4", shape) == 6600 * 0.25
    # A round on an A100: 1,000,000 member-steps in 400 s with 2 groups on the card.
    speeds.record("a", "A100", shape, 1_000_000, 400.0, sharing=2)
    assert speeds.rate("A100", shape, "a") == 5000  # this machine
    assert speeds.rate("A100", shape, "b") == 5000  # another A100
    # A T4 of the same shape: from the A100's measurement scaled by the guess of the cards
    assert speeds.rate("T4", shape) == 5000 * 0.25 / speeds.card_factor("A100")
    # Once a T4 has been measured (on another shape), its factor is measured too.
    speeds.record("t", "T4", "64x4x128", 230 * 300, 300.0, sharing=1)
    assert abs(speeds.card_factor("T4") - 0.1) < 1e-9
    assert speeds.rate("T4", "64x4x128", "t") == 230


def plan_of(machines, groups, where, movable):
    plan = fleet_plan.improve(machines, groups, where, movable)
    return plan, fleet_plan.stage_end(fleet_plan.ends_of(machines, groups, plan))


def machine(card, rate, ready=0.0):
    return {"card": card, "ready": ready, "rate": lambda shape: rate}


def group(work, compiled_on=()):
    return {"work": work, "shape": "8x4x128", "compiled_on": set(compiled_on)}


def test_planner_moves_only_when_the_end_comes_earlier():
    machines = {"slow": machine("T4", 1000.0), "fast": machine("A100", 4000.0)}
    groups = {i: group(1_000_000, ["slow"]) for i in range(4)}
    where = {i: "slow" for i in range(4)}
    before = fleet_plan.stage_end(fleet_plan.ends_of(machines, groups, where))
    plan, after = plan_of(machines, groups, where, list(groups))
    assert sum(1 for n in plan.values() if n == "fast") >= 3 and after < before / 2
    # Not movable (in the middle of a round): nothing moves.
    assert plan_of(machines, groups, where, [])[0] == where
    # Small groups: the cost of moving (upload, compile) is larger than the gain.
    small = {i: group(20_000, ["slow"]) for i in range(2)}
    assert plan_of(machines, small, {0: "slow", 1: "slow"}, [0, 1])[0] == {0: "slow", 1: "slow"}
    # Two equal machines with equal work: nothing moves.
    machines = {"a": machine("A100", 4000.0), "b": machine("A100", 4000.0)}
    groups = {0: group(10 ** 6, ["a"]), 1: group(10 ** 6, ["b"])}
    assert plan_of(machines, groups, {0: "a", 1: "b"}, [0, 1])[0] == {0: "a", 1: "b"}


# ------------------------------------------------------------------------------------------------
# 2. Planning on pretended machines
# ------------------------------------------------------------------------------------------------


class PretendColab:
    """Stands in for colab_fleet.Colab: records what is sent and answers as a machine would."""

    tool = "/bin/false"

    def __init__(self):
        self.sent = []

    def execute(self, name, code, timeout=300):
        self.sent.append((name, code))
        if 'print("TAKEN "' in code:
            numbers = ast.literal_eval(re.search(r"^NUMBERS = (.*)$", code, re.M).group(1))
            return True, "TAKEN " + json.dumps(numbers)
        if "GIVEN" in code:
            return True, "GIVEN"
        return True, "SIGNALLED"

    def upload(self, *args):
        return True

    def what(self, name):
        """The kinds of request sent to machine `name`."""
        kinds = []
        for to, code in self.sent:
            if to == name:
                kinds.append("take back" if 'print("TAKEN "' in code else
                             "give" if "GIVEN" in code else
                             re.search(r"^WHAT = '(\w+)'", code, re.M).group(1))
        return kinds


class NoProber:
    pending = {}

    def pending_cards(self):
        return []

    def answers(self):
        return []

    def close(self):
        pass


def pretend_fleet(tmp_path, machines, groups):
    """A fleet of the counted shapes on pretended machines.
    machines: {name: card}; groups: {index: (machine, reached, members, "waiting"/"training")}."""
    configs = [[dict(counted_config(8 * 2 ** (i % 4), 4, 128), name=f"g{i}_m{m}")
                for m in range(24)] for i in range(len(groups))]
    fleet = colab_fleet.Fleet("counted", configs, PretendColab(), runs_root=tmp_path / "runs",
                              state_path=tmp_path / "fleet.json", log=tmp_path / "fleet.log",
                              prober=NoProber())
    answers = {name: {"waiting": [], "taken": [], "progress": {}, "alive": 1, "card": ""}
               for name in machines}
    for name, card in machines.items():
        fleet.state["machines"][name] = {"card": card, "last_answer": time.time(),
                                         "started": True}
    for index, (name, reached, members, how) in groups.items():
        fleet.state["groups"][str(index)] = {"machine": name, "reached": reached,
                                             "saved_step": reached, "step": reached,
                                             "members": members, "compiled_on": [name]}
        answers[name]["taken" if how == "training" else "waiting"].append(index)
    return fleet, answers


def test_a_faster_machine_takes_waiting_groups(tmp_path):
    fleet, answers = pretend_fleet(tmp_path, {"t4": "T4", "new-a100": "A100"},
                                   {i: ("t4", 0, 24, "waiting") for i in range(6)})
    fleet.replan(answers)
    on_a100 = fleet.held_by("new-a100")
    assert len(on_a100) >= 3
    assert sorted(on_a100 + fleet.held_by("t4")) == list(range(6))  # none lost, none twice
    assert fleet.colab.what("t4") == ["take back"] and fleet.colab.what("new-a100") == ["give"]
    assert "move from t4 to new-a100" in (tmp_path / "fleet.log").read_text()


def test_a_slow_machine_is_stopped_mid_round_only_when_it_clearly_helps(tmp_path):
    # Two long groups in the middle of their last round on a T4, an A100 with nothing to do.
    fleet, answers = pretend_fleet(tmp_path, {"t4": "T4", "a100": "A100"},
                                   {0: ("t4", 320000, 24, "training"),
                                    1: ("t4", 320000, 24, "training")})
    fleet.replan(answers)
    assert fleet.colab.what("t4") == ["stop"]
    assert fleet.state["machines"]["t4"]["leaving"]
    assert fleet.held_by("t4") == [0, 1]  # they stay its until it has saved them
    # The same groups near the end of their round: stopping would not help clearly.
    fleet, answers = pretend_fleet(tmp_path / "2", {"t4": "T4", "a100": "A100"},
                                   {0: ("t4", 320000, 24, "training"),
                                    1: ("t4", 320000, 24, "training")})
    for key in ("0", "1"):  # (trained for a very long time already)
        fleet.state["groups"][key]["training_since"] = time.time() - 10 ** 7
    fleet.replan(answers)
    assert fleet.colab.what("t4") == [] and not fleet.state["machines"]["t4"].get("leaving")


def test_no_move_when_it_would_not_help(tmp_path):
    # An A100 busy with a long round, a T4 with a group whose round is almost done.
    fleet, answers = pretend_fleet(tmp_path, {"t4": "T4", "a100": "A100"},
                                   {0: ("a100", 320000, 24, "training"),
                                    1: ("t4", 639000, 2, "waiting")})
    fleet.replan(answers)
    assert fleet.colab.sent == []
    assert fleet.held_by("t4") == [1] and fleet.held_by("a100") == [0]


def test_a_slow_machine_is_given_back_when_the_stage_does_not_end_later(tmp_path):
    # One long round on an A100 sets the end; a second A100 has a short round; the T4's small
    # group can go to the second A100 without the stage ending later, and the T4 is given back.
    fleet, answers = pretend_fleet(tmp_path, {"a": "A100", "b": "A100", "t4": "T4"},
                                   {0: ("a", 320000, 24, "training"),
                                    1: ("b", 600000, 24, "training"),
                                    2: ("t4", 630000, 4, "waiting")})
    fleet.replan(answers)
    assert fleet.held_by("t4") == [] and fleet.held_by("b") == [1, 2]
    assert "it costs less" in (tmp_path / "fleet.log").read_text()


# ------------------------------------------------------------------------------------------------
# 3. Whole runs with the pretend Colab tool
# ------------------------------------------------------------------------------------------------


def slow_guesses(monkeypatch, rate=0.5):
    """Make the tiny test groups look long to the planner (`rate` member-steps a second on an
    A100, never replaced by the measured ones) and moves cheap, so that it moves them as it
    would move the real groups."""
    monkeypatch.setattr(fleet_plan, "A100_GUESS", {8: rate, 16: rate, 32: rate, 64: rate})
    monkeypatch.setattr(fleet_plan.Speeds, "record", lambda *args, **kwargs: None)
    for name in ("MOVE_SECONDS", "STOP_SECONDS", "NEW_MACHINE_SECONDS"):
        monkeypatch.setattr(fleet_plan, name, 1.0)


def set_fake(state_path, **values):
    state = json.loads(state_path.read_text())
    state.update(values)
    state_path.write_text(json.dumps(state))


def machines_of(state_path):
    return json.loads(state_path.read_text())["machines"]


def test_availability_changes_and_a_faster_card_takes_over(tmp_path, fake, monkeypatch):
    tool, state_path = fake
    slow_guesses(monkeypatch)
    add_owners_session(state_path)
    set_fake(state_path, refuse=["H100", "G4", "A100", "L4", "T4"])  # nothing free
    fleet = make_fleet(tmp_path, tool, machines=1, cards=("H100", "G4", "A100", "L4", "T4"))
    fleet.step()
    fleet.step()
    assert not fleet.state["machines"]
    assert fleet.state["availability"]["T4"]["outcome"] == "no card free"
    set_fake(state_path, refuse=["H100", "G4", "A100", "L4"])  # a T4 becomes free
    fleet.step()
    assert [m["card"] for m in fleet.state["machines"].values()] == ["T4"]
    assert sorted(fleet.held_by(next(iter(fleet.state["machines"])))) == [0, 1]
    set_fake(state_path, refuse=["H100", "G4", "L4"])  # an A100 becomes free
    run_until_done(fleet, steps=200)
    check_results(tmp_path, state_path)  # all here, exact, each group finished on one machine
    log = (tmp_path / "fleet.log").read_text()
    assert re.search(r"machine air-a100-\S+ \(A100\) added", log)
    # The T4 handed its groups to the A100 and was given back before the end.
    t4 = re.search(r"machine (air-t4-\S+) \(T4\) added", log).group(1)
    assert re.search(rf"move from {t4} to air-a100|machine {t4}: stopped and given back", log)
    assert re.search(rf"machine {t4}: (stopped and given back|nothing left to do)", log)
    models = list((tmp_path / "colab" / "machines").glob("air-a100-*/air_run/out/test/*/model.pt"))
    assert models  # the A100 trained


def test_colab_limit_is_noted(tmp_path, fake, monkeypatch):
    tool, state_path = fake
    slow_guesses(monkeypatch)
    set_fake(state_path, limit=1, refuse=[])
    fleet = make_fleet(tmp_path, tool, machines=3, cards=("A100",))
    run_until_done(fleet)
    assert fleet.state["limit"]["machines"] == 1
    assert "precondition failed" in (tmp_path / "fleet.log").read_text()
    assert not any(m["running"] for m in machines_of(state_path).values())


def test_adoption_of_the_present_state_file(tmp_path, fake):
    """The launcher as it ran before (state file without the new fields, a machine training)
    is taken over: the machine is kept and goes on; nothing is lost or trained twice."""
    tool, state_path = fake
    add_owners_session(state_path)
    colab = colab_fleet.Colab(tool)
    name = "air-a100-085648-0"
    assert colab.run("--config", "x", "new", "-s", name, "--gpu", "A100")[0]
    # What the launcher before did: start the job, give it every group.
    old = colab_fleet.Fleet("test", tiny_groups(), colab, runs_root=tmp_path / "runs",
                            state_path=tmp_path / "unused.json", log=tmp_path / "old.log",
                            workers=1)
    old.state["machines"][name] = {"card": "A100", "last_answer": time.time(), "started": False}
    assert old.start_job(name)
    old.give(name, [0, 1])
    # Cut off in the middle of a move: group 1 was taken back from the machine, but the state
    # file still gives it to the machine. The new launcher must notice and train it elsewhere.
    taken = colab.execute(name, colab_fleet.with_values(colab_fleet.TAKE_BACK, NUMBER=1))[1]
    assert "TAKEN BACK" in taken
    time.sleep(2)  # some training
    present = {"machines": {name: {"card": "A100", "last_answer": time.time(), "started": True,
                                   "waiting": 1}},
               "groups": {"0": {"machine": name}, "1": {"machine": name}}, "failures": {}}
    (tmp_path / "fleet.json").write_text(json.dumps(present))
    fleet = make_fleet(tmp_path, tool, machines=1, cards=("A100",))
    run_until_done(fleet)
    check_results(tmp_path, state_path)
    log = (tmp_path / "fleet.log").read_text()
    assert f"machine {name}: job running" not in log  # already started: not started again
    assert "group 1: not on air-a100-085648-0 any more" in log
    assert not re.search(r"machine air-\S+ \(A100\) added", log)  # no other machine was needed
    assert [c for c, _ in json.loads(state_path.read_text())["log"]].count("new") == 1  # ours
    assert not machines_of(state_path)[name]["running"]


def test_a_slow_machine_hands_over_in_the_middle_of_a_round(tmp_path, fake, monkeypatch):
    """A T4 is training when an A100 becomes free: the T4 is stopped in the middle of its round
    (it saves first), its groups go on on the A100 from there, and the results are exactly those
    of training straight through."""
    import torch

    from v2 import batch
    from v3.tests.test_v3_fleet import without_time
    tool, state_path = fake
    slow_guesses(monkeypatch)
    groups = [[dict(c, steps=8000, first_look=4000, name="long_" + c["name"]) for c in g]
              for g in tiny_groups()]
    set_fake(state_path, refuse=["A100"])
    fleet = colab_fleet.Fleet("test", groups, colab_fleet.Colab(tool), runs_root=tmp_path / "runs",
                              state_path=tmp_path / "fleet.json", log=tmp_path / "fleet.log",
                              machines=1, workers=1, cards=("A100", "T4"), poll_seconds=0.5,
                              lost_minutes=1)
    for _ in range(60):  # until the T4 trains a group (one worker: one group at a time)
        fleet.step()
        t4 = [n for n, m in fleet.state["machines"].items() if m["card"] == "T4"]
        if t4 and fleet.state["machines"][t4[0]].get("waiting") == 1:
            break
        time.sleep(0.5)
    assert t4
    set_fake(state_path, refuse=[])
    run_until_done(fleet, steps=300)
    log = (tmp_path / "fleet.log").read_text()
    assert f"machine {t4[0]} (T4): stopped in the middle of its rounds" in log
    assert f"machine {t4[0]}: stopped and given back" in log
    assert not any(m["running"] for m in machines_of(state_path).values())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as the machine's workers on the processor
    try:
        for number, configs in enumerate(groups):
            whole, _ = batch.train_group(configs, tmp_path / f"whole{number}", "cpu", quiet=True)
            for summary in whole:
                here = json.loads((tmp_path / "runs" / "test" / summary["name"] /
                                   "summary.json").read_text())
                assert without_time(here) == without_time(summary)
    finally:
        torch.set_num_threads(threads)

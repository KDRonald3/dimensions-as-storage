"""Tests of "move up" (Fleet.move_up and Fleet.settle_parking in v3/colab_fleet.py, the park
signal in v3/machine_job.py): a group in the middle of a round on a slower machine goes to a
faster machine that has a free worker. Processor only.

Every order of events for the park signal, and what must hold (S: the slower machine, F: the
faster one, g: the group). In every case g is never on two machines and never lost, because a
task file is always in exactly one of tasks/, taken/ and parked/ on a machine (all moves are
renames that succeed only once), a worker trains g only while its file is in taken/, and the
launcher keeps g as S's until g's file has been taken off S.

  1. Signal written, the worker sees it mid-round: it saves, g goes to parked/. When the
     checkpoint is home (the step here equals the step reached), the launcher takes g's file
     off S, frees g and gives it to F.                              (test: a faster machine ...)
  2. Signal written, g's round ends before the worker sees it: g is back in tasks/ with the
     signal. The next worker that looks for work parks it instead of taking it; or the launcher
     sees it waiting and takes it from the queue itself (same rename). Then as in 1.
  3. Signal written, g ends (all members done) or fails meanwhile: the worker removes the signal
     (a failed or done group is not put back); the launcher drops the move, and removes any
     signal that is left over.                                       (unit test of the cases)
  4. The request writing the signal fails: the move is dropped at once. If it was written
     after all, the next look finds a signal that no move stands for and removes it.
  5. Launcher cut off after noting the move, before the signal was sent: the next launcher sees
     the note, an answer asked for after it, and g still training with no signal: the move is
     dropped (it may be chosen again).                               (unit test of the cases)
  6. Launcher cut off after the signal, before giving g on: the next launcher reads the note
     from the state file and goes on with 1.                         (test: cut off, variant 1)
  7. Launcher cut off after taking g's file off S, before noting g as free: the next launcher
     finds g neither on S nor done, with the note, in two answers in a row (one answer is not
     enough: a look can miss a group in the instant a worker puts it back in the queue): g is
     freed from its checkpoint here and given on.    (test: cut off, variant 2; and a unit test)
  8. The checkpoint came home but the launcher was cut off before noting it (a look that was
     on its way): it would never be packed again, so after two minutes of waiting the launcher
     asks S to send it again (once per two minutes: the request dates the file now, and a look
     packs only files older than 5 seconds).                         (unit test of the cases)
  9. An answer asked for before the signal was written (a slow look still on its way) cannot
     show the signal: the move is left as it is until a newer answer comes.   (unit test)
  8b. The checkpoint never comes home (after 10 minutes): g goes back into S's queue and trains
     on there; the move is dropped.                                   (unit test)
 10. F has no free worker any more when g's checkpoint is home: g is free and the planner hands
     it out like any free group.                                     (test: worker gone)
 11. F lost, or told to leave, meanwhile: the note is dropped with F (release), and g is free
     once off S. S lost or leaving: release frees g from its checkpoint here (the leaving flow
     waits until everything S saved is home).
 12. A signal that stands for no move (left from any of the above, or written by hand): removed
     at the next look. If a worker parked g by it first, g is taken off S when its checkpoint is
     home and handed out (to S again, if the planner wants: the request that gives g removes any
     signal and parked copy first, so g is not parked again by surprise).  (test: stale signal)
 13. A parked copy of g on S while g is on another machine (or done): removed, never trained.
 14. The job on S was started by an older machine_job.py (no park signal): no signal is sent;
     S is stopped as a whole only when every group it trains would move up (never a fast card).
                                                                     (test: cannot park)
"""

import json
import re
import time

import pytest
import torch

from v2 import batch
from v3 import colab_fleet, fleet_plan
from v3.tests.test_v3_fleet import fake, tiny_groups, without_time  # noqa: F401
from v3.tests.test_v3_fleet_plan import PretendColab, pretend_fleet, set_fake, slow_guesses
from v3.tests.test_v3_fleet_takeover import stretches

FIRST_LOOK, STEPS = 6000, 12000


def long_groups():
    """Two tiny groups of two members with long rounds (6,000 steps, about 20 s each), so that
    a group can be caught in the middle of a round."""
    return [[dict(c, steps=STEPS, first_look=FIRST_LOOK, name="up_" + c["name"]) for c in g]
            for g in tiny_groups()]


def measured(fleet, factors=None):
    """Note speeds as if measured for every shape of the fleet's groups (`factors`: how fast each
    card is against an A100). A move up is decided only on measured speeds, and the tiny test
    groups' rounds are too short to be measured (v3/colab_fleet.py round_ended)."""
    factors = factors or {"A100": 1.0, "H100": 1.7}
    for card, factor in factors.items():
        found = fleet.state["speeds"].setdefault(card, {})
        for configs in fleet.groups:
            shape = fleet_plan.shape_of(configs)
            found[shape] = fleet_plan.a100_guess(shape) * factor
    return fleet


def mover_fleet(tmp_path, tool, groups=None, colab=None):
    """An A100 first (an H100 is refused), then an H100 whose 4 workers share the card. The A100
    is a fast card, so the old rules never stop it in the middle of a round: what moves a group
    from it while it trains can only be the move up."""
    return measured(colab_fleet.Fleet("test", groups or long_groups(),
                                      colab or colab_fleet.Colab(tool),
                                      runs_root=tmp_path / "runs",
                                      state_path=tmp_path / "fleet.json",
                                      log=tmp_path / "fleet.log", machines=1, workers=1,
                                      cards=("H100", "A100"), poll_seconds=0.5, lost_minutes=2,
                                      shared_cards={"H100": 4}))


def run_until_a100_trains(fleet):
    """Looks until the A100 trains one group and has the other waiting."""
    for _ in range(80):
        fleet.step()
        a100 = [n for n, m in fleet.state["machines"].items() if m["card"] == "A100"]
        if a100 and fleet.state["machines"][a100[0]].get("waiting") == 1:
            return a100[0]
        time.sleep(0.5)
    raise AssertionError("the A100 never got busy")


def run_to_end(fleet, steps=400):
    for _ in range(steps):
        if not fleet.step():
            return
        time.sleep(fleet.poll_seconds)
    raise AssertionError("the fleet did not finish")


def check_exact_and_once(tmp_path, state_path, groups):
    """Every run here and equal to training it straight through; no stretch trained twice; every
    machine stopped."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as the machine's workers on the processor
    try:
        for number, configs in enumerate(groups):
            whole, _ = batch.train_group(configs, tmp_path / f"whole{number}", "cpu", quiet=True)
            for summary in whole:
                here = tmp_path / "runs" / "test" / summary["name"]
                assert (here / "model.pt").exists()
                assert without_time(json.loads((here / "summary.json").read_text())) == \
                    without_time(summary)
    finally:
        torch.set_num_threads(threads)
    sittings = stretches(tmp_path)
    assert len(sittings) == len(set(sittings)), f"a stretch was trained twice: {sorted(sittings)}"
    assert not any(m["running"] for m in json.loads(state_path.read_text())["machines"].values())


def moved_mid_round(tmp_path, log):
    """The group that moved up, after checking that it went on on the H100 from a step in the
    middle of a round (not from a look)."""
    found = re.search(r"group (\d+) moves up from air-a100-\S+ \(A100\) to air-h100-\S+ \(H100\) "
                      r"in the middle of its round: it should end about \d+ minutes sooner", log)
    assert found, log
    index = found.group(1)
    on_h100 = []
    for path in (tmp_path / "colab" / "machines").glob("air-h100-*/air_run/out/logs/worker*.log"):
        on_h100 += [int(s.replace(",", "")) for g, s in
                    re.findall(r"group (\d+): from step ([\d,]+) to", path.read_text())
                    if g == index]
    assert any(step % FIRST_LOOK for step in on_h100), on_h100
    return int(index)


# ------------------------------------------------------------------------------------------------
# With the pretend Colab tool (real training on the processor)
# ------------------------------------------------------------------------------------------------


def test_a_faster_machine_takes_a_group_in_the_middle_of_its_round(tmp_path, fake, monkeypatch):
    tool, state_path = fake
    slow_guesses(monkeypatch)
    fleet = mover_fleet(tmp_path, tool)
    run_until_a100_trains(fleet)
    set_fake(state_path, refuse=["G4"])  # an H100 is free now
    run_to_end(fleet)
    log = (tmp_path / "fleet.log").read_text()
    index = moved_mid_round(tmp_path, log)
    assert re.search(rf"group {index}: saved on air-a100-\S+ at step [\d,]+; it goes on on "
                     rf"air-h100-", log)
    check_exact_and_once(tmp_path, state_path, long_groups())


def test_the_free_worker_is_gone_when_the_checkpoint_is_home(tmp_path, fake, monkeypatch):
    tool, state_path = fake
    slow_guesses(monkeypatch)
    fleet = mover_fleet(tmp_path, tool)
    run_until_a100_trains(fleet)
    set_fake(state_path, refuse=["G4"])
    for _ in range(80):  # until a move up has been decided
        fleet.step()
        if any(g.get("parking") for g in fleet.state["groups"].values()):
            break
        time.sleep(0.5)
    else:
        raise AssertionError("no move up was decided")
    # Meanwhile the H100's workers were all taken (here: it is counted as having none free).
    workers_of = fleet.workers_of
    monkeypatch.setattr(fleet, "workers_of", lambda name: 0 if "h100" in name else workers_of(name))
    run_to_end(fleet)
    log = (tmp_path / "fleet.log").read_text()
    assert re.search(r"group \d+: saved on air-a100-\S+ at step [\d,]+; air-h100-\S+ has no free "
                     r"worker any more, so it is handed out like any free group", log), log
    check_exact_and_once(tmp_path, state_path, long_groups())


class CutOff(Exception):
    pass


class CutColab(colab_fleet.Colab):
    """The real (pretend) tool, but this program "dies" right after a request whose answer holds
    `marker` was carried out."""

    def __init__(self, tool, marker):
        super().__init__(tool)
        self.marker = marker

    def execute(self, machine, code, timeout=300):
        worked, printed = super().execute(machine, code, timeout)
        if self.marker in printed:
            raise CutOff()
        return worked, printed


@pytest.mark.parametrize("marker", ["PARK SIGNALLED", "PARKED TAKEN"])
def test_launcher_cut_off_in_the_middle_of_a_move_up(tmp_path, fake, monkeypatch, marker):
    """Cut off right after the park signal was written (before the group was given on), or right
    after the parked group was taken off the slower machine (before that was noted). A new
    launcher takes over the state file; the group ends correctly, once."""
    tool, state_path = fake
    slow_guesses(monkeypatch)
    # (Looks here are a second apart, not 40 s: a checkpoint lost with the cut-off program's last
    # look is asked for again after 10 s instead of 2 minutes.)
    monkeypatch.setattr(colab_fleet, "RESEND_SECONDS", 10.0)
    fleet = mover_fleet(tmp_path, tool, colab=CutColab(tool, marker))
    run_until_a100_trains(fleet)
    set_fake(state_path, refuse=["G4"])
    for _ in range(120):
        try:
            fleet.step()
        except CutOff:
            break
        time.sleep(0.5)
    else:
        raise AssertionError("the launcher was never cut off")
    # The program is gone: its looks still on their way end with it, nothing more is saved.
    fleet.pool.shutdown(wait=True)
    fleet.prober.close()
    again = mover_fleet(tmp_path, tool)
    assert any(g.get("parking") for g in again.state["groups"].values())  # the note survived
    run_to_end(again)
    log = (tmp_path / "fleet.log").read_text()
    moved_mid_round(tmp_path, log)
    assert re.search(r"group \d+: saved on air-a100-\S+ at step [\d,]+; it goes on on air-h100-",
                     log), log
    check_exact_and_once(tmp_path, state_path, long_groups())


def test_a_stale_park_signal_does_no_harm(tmp_path, fake, monkeypatch):
    """Park signals that no move stands for, written on the only machine for every group (the
    one training, the one waiting) and for a group that does not exist."""
    tool, state_path = fake
    fleet = colab_fleet.Fleet("test", tiny_groups(), colab_fleet.Colab(tool),
                              runs_root=tmp_path / "runs", state_path=tmp_path / "fleet.json",
                              log=tmp_path / "fleet.log", machines=1, workers=1,
                              cards=("A100",), poll_seconds=0.5, lost_minutes=2)
    for _ in range(60):
        fleet.step()
        names = list(fleet.state["machines"])
        if names and fleet.state["machines"][names[0]].get("waiting") == 1:
            break
        time.sleep(0.5)
    out = tmp_path / "colab" / "machines" / names[0] / "air_run" / "out"
    for number in (0, 1, 7):
        (out / "park" / str(number)).write_text("")
    run_to_end(fleet)
    check_exact_and_once(tmp_path, state_path, tiny_groups())
    assert not list((out / "park").iterdir()) and not list((out / "parked").iterdir())
    assert "moves up" not in (tmp_path / "fleet.log").read_text()


# ------------------------------------------------------------------------------------------------
# The decisions, on pretended machines (no training)
# ------------------------------------------------------------------------------------------------


class Answering(PretendColab):
    """Also answers the requests of a move up as a machine would."""

    def execute(self, name, code, timeout=300):
        if "PARK SIGNALLED" in code:
            self.sent.append((name, code))
            return True, "PARK SIGNALLED"
        if "taken_back_parked" in code:
            self.sent.append((name, code))
            return True, "PARKED TAKEN"
        if "BACK IN THE QUEUE" in code:
            self.sent.append((name, code))
            return True, "BACK IN THE QUEUE"
        return super().execute(name, code, timeout)

    def kinds(self, name):
        """What was sent to machine `name`, in words."""
        found = []
        for to, code in self.sent:
            if to != name:
                continue
            number = re.search(r"^NUMBERS? = (.*)$", code, re.M)
            for word, mark in (("park", "PARK SIGNALLED"), ("take parked", "taken_back_parked"),
                               ("unpark", "UNPARKED"), ("resend", "RESENT"), ("give", "GIVEN"),
                               ("back in the queue", "BACK IN THE QUEUE"),
                               ("stop", "WHAT = 'stop'")):
                if mark in code:
                    found.append(f"{word} {number.group(1)}" if number else word)
                    break
        return found


def answering_fleet(tmp_path, machines, groups, can_park=True, measure=True):
    fleet, answers = pretend_fleet(tmp_path, machines, groups)
    fleet.colab = Answering()
    if measure:  # (as measured: the guesses of fleet_plan for every card)
        measured(fleet, fleet_plan.CARD_GUESS)
    for name, answer in answers.items():
        answer.update(can_park=can_park, parked=[], park_signals=[], asked_at=time.time())
    return fleet, answers


def test_a_move_up_only_when_the_gain_is_clear(tmp_path):
    # Two long groups training on a T4, an idle H100 with two workers: both move, the one with
    # the larger gain first (group 0: about 38 minutes; group 1, with fewer members expected to
    # go on, about 4 once group 0 has left the T4 and it trains there alone).
    fleet, answers = answering_fleet(tmp_path, {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 0, 24, "training"), 1: ("t4", 0, 24, "training")})
    fleet.move_up(answers)
    assert fleet.colab.kinds("t4") == ["park 0", "park 1"]
    assert fleet.state["groups"]["0"]["parking"]["to"] == "h100"
    assert fleet.state["groups"]["0"]["pinned_until"] > time.time() + 1000
    assert fleet.incoming("h100") == 2 and fleet.free_workers("h100", answers) == 0
    assert re.search(r"group 0 moves up from t4 \(T4\) to h100 \(H100\) in the middle of its "
                     r"round: it should end about 38 minutes sooner", (tmp_path / "fleet.log")
                     .read_text())
    fleet.move_up(answers)  # decided once: not again at the next look
    assert fleet.colab.kinds("t4") == ["park 0", "park 1"]
    # One free worker only: the larger gain takes it.
    fleet, answers = answering_fleet(tmp_path / "1", {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 0, 24, "training"), 1: ("t4", 0, 24, "training"),
                                      2: ("h100", 0, 24, "training")})
    fleet.move_up(answers)
    assert fleet.colab.kinds("t4") == ["park 0"]
    # Nearly finished (the last 40,000 steps of 2 members): the gain is under the margin.
    fleet, answers = answering_fleet(tmp_path / "2", {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 600000, 2, "training"),
                                      1: ("t4", 600000, 2, "training")})
    fleet.move_up(answers)
    assert fleet.colab.sent == [] and not fleet.state["groups"]["0"].get("parking")
    # Long groups, but no free worker on the faster machine: nothing moves.
    fleet, answers = answering_fleet(tmp_path / "3", {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 0, 24, "training"), 1: ("h100", 0, 24, "training"),
                                      2: ("h100", 0, 24, "waiting")})
    fleet.move_up(answers)
    assert fleet.colab.sent == []


def test_a_machine_that_cannot_park(tmp_path):
    # Every group it trains can move up (the H100 has two free workers): it leaves as a whole.
    fleet, answers = answering_fleet(tmp_path, {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 0, 24, "training"), 1: ("t4", 0, 24, "training")},
                                     can_park=False)
    answers["h100"]["can_park"] = True
    fleet.workers = 2
    monkey = fleet.workers_of
    fleet.workers_of = lambda name: 4 if name == "h100" else monkey(name)
    fleet.move_up(answers)
    assert fleet.colab.kinds("t4") == ["stop"]
    assert fleet.state["machines"]["t4"]["leaving"]
    assert not any(g.get("parking") for g in fleet.state["groups"].values())
    assert "cannot park one group, and all its groups [0, 1]" in (tmp_path / "fleet.log").read_text()
    # Only one of its groups can move up (one free worker): nothing happens.
    fleet, answers = answering_fleet(tmp_path / "2", {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 0, 24, "training"), 1: ("t4", 0, 24, "training"),
                                      2: ("h100", 0, 24, "training")}, can_park=False)
    fleet.move_up(answers)
    assert fleet.colab.sent == [] and not fleet.state["machines"]["t4"].get("leaving")
    # A fast card that cannot park leaves as a whole too when all its groups would move up (the
    # owner's rule: the A100s released by hand on 23 November).
    fleet, answers = answering_fleet(tmp_path / "3", {"a100": "A100", "h100": "H100"},
                                     {0: ("a100", 0, 24, "training")}, can_park=False)
    measured(fleet, {"A100": 0.3})  # (its group should end much sooner on the H100)
    fleet.move_up(answers)
    assert fleet.colab.kinds("a100") == ["stop"] and fleet.state["machines"]["a100"]["leaving"]
    # The free workers its groups will need are not given to other moves in the same look: the
    # H100 has 4 free workers, the old T4 leaves (its two groups will need two of them), and the
    # L4, which can park, is given at most the other two.
    fleet, answers = answering_fleet(tmp_path / "4", {"t4": "T4", "l4": "L4", "h100": "H100"},
                                     {0: ("t4", 0, 24, "training"), 1: ("t4", 0, 24, "training"),
                                      2: ("l4", 0, 24, "training"), 3: ("l4", 0, 24, "training")})
    workers_of = fleet.workers_of
    fleet.workers_of = lambda name: 4 if name == "h100" else workers_of(name)
    answers["t4"]["can_park"] = False
    fleet.move_up(answers)
    assert fleet.colab.kinds("t4") == ["stop"]
    assert 1 <= len(fleet.colab.kinds("l4")) <= 2 and fleet.incoming("h100") <= 2


def test_the_orders_of_events_of_a_park(tmp_path):
    """settle_parking in the cases of the list at the top (on pretended machines)."""
    machines = {"t4": "T4", "h100": "H100"}

    def started(tmp, at_offset=0.0):
        fleet, answers = answering_fleet(tmp, machines, {0: ("t4", 40000, 24, "training"),
                                                         1: ("t4", 40000, 24, "training")})
        group = fleet.state["groups"]["0"]
        group["parking"] = {"from": "t4", "to": "h100", "at": time.time() + at_offset}
        for answer in answers.values():
            answer["progress"] = {}
        return fleet, answers, group

    # 9. An answer asked for before the signal was written: the move is left as it is.
    fleet, answers, group = started(tmp_path / "9", at_offset=60.0)
    fleet.settle_parking(answers)
    assert group.get("parking") and fleet.colab.sent == []
    # 5. A newer answer, the group still training, no signal: the move is dropped.
    fleet, answers, group = started(tmp_path / "5", at_offset=-60.0)
    fleet.settle_parking(answers)
    assert not group.get("parking") and fleet.colab.sent == []
    assert "did not reach t4" in (tmp_path / "5" / "fleet.log").read_text()
    # The signal is there: wait for the worker.
    fleet, answers, group = started(tmp_path / "w", at_offset=-60.0)
    answers["t4"]["park_signals"] = [0]
    fleet.settle_parking(answers)
    assert group.get("parking") and fleet.colab.sent == []
    # 3. The group ended meanwhile: the move is dropped and the signal left over removed; and
    # 12. a signal for a group with no move is removed too.
    fleet, answers, group = started(tmp_path / "3", at_offset=-60.0)
    answers["t4"]["taken"] = [1]
    answers["t4"]["park_signals"] = [0, 1, 5]
    answers["t4"]["progress"] = {"0": {"step": 80000, "done": True}}
    fleet.settle_parking(answers)
    assert not group.get("parking") and fleet.colab.kinds("t4") == ["unpark [0, 1, 5]"]
    # 1 and 8. Parked, but its checkpoint is not home: wait; after two minutes, ask again for it
    # (once: asking at every look would keep the file from ever being packed).
    fleet, answers, group = started(tmp_path / "8", at_offset=-60.0)
    answers["t4"].update(taken=[1], parked=[0], progress={"0": {"step": 52345, "done": False}})
    fleet.settle_parking(answers)
    fleet.settle_parking(answers)
    assert fleet.colab.kinds("t4") == [] and group["machine"] == "t4"
    group["parked_since"] -= colab_fleet.RESEND_SECONDS + 1  # (two minutes later)
    fleet.settle_parking(answers)
    fleet.settle_parking(answers)
    assert fleet.colab.kinds("t4") == ["resend 0"] and group["machine"] == "t4"
    # ... home now: taken off the T4, given to the H100, pinned there.
    group["saved_step"] = 52345
    fleet.settle_parking(answers)
    assert fleet.colab.kinds("t4") == ["resend 0", "take parked 0"]
    assert fleet.colab.kinds("h100") == ["give"] and group["machine"] == "h100"
    assert not group.get("parking") and group["pinned_until"] > time.time() + 1000
    # 10. Home, but the H100 has no free worker any more: free, for the planner.
    fleet, answers, group = started(tmp_path / "10", at_offset=-60.0)
    answers["t4"].update(taken=[1], parked=[0], progress={"0": {"step": 52345, "done": False}})
    answers["h100"]["taken"] = [7, 8]
    group["saved_step"] = 52345
    fleet.settle_parking(answers)
    assert group["machine"] is None and group["step"] == 52345 and fleet.colab.kinds("h100") == []
    assert 0 in fleet.free_groups()
    # 2. The round ended before the worker saw the signal: taken from the queue.
    fleet, answers, group = started(tmp_path / "2", at_offset=-60.0)
    answers["t4"].update(taken=[1], waiting=[0], park_signals=[0],
                         progress={"0": {"step": 80000, "done": False}})
    group["saved_step"] = 80000
    fleet.settle_parking(answers)
    assert fleet.colab.kinds("t4")[0] == "take parked 0" and group["machine"] == "h100"
    # 13. A parked copy of a group that another machine has: removed, nothing else.
    fleet, answers, group = started(tmp_path / "13", at_offset=-60.0)
    fleet.state["groups"]["1"]["machine"] = "h100"
    answers["t4"].update(taken=[], parked=[1])
    fleet.settle_parking(answers)
    assert "take parked 1" in fleet.colab.kinds("t4")
    assert fleet.state["groups"]["1"]["machine"] == "h100"
    # 12. A worker parked a group by a signal that no move stands for: once its checkpoint is
    # home it is taken off the machine and is free (for the planner, which may give it back).
    fleet, answers, group = started(tmp_path / "12", at_offset=-60.0)
    other = fleet.state["groups"]["1"]
    answers["t4"].update(taken=[0], parked=[1], progress={"1": {"step": 41000, "done": False}})
    other["saved_step"] = 41000
    fleet.settle_parking(answers)
    assert "take parked 1" in fleet.colab.kinds("t4") and other["machine"] is None
    assert other["step"] == 41000 and 1 in fleet.free_groups()
    # 11. The machine it was going to is lost: the move is dropped with it.
    fleet, answers, group = started(tmp_path / "11", at_offset=-60.0)
    fleet.release("h100")
    assert not group.get("parking") and group["machine"] == "t4"
    assert fleet.incoming("h100") == 0


# ------------------------------------------------------------------------------------------------
# Added in review (23 November 2026)
# ------------------------------------------------------------------------------------------------


def test_no_move_up_on_guessed_or_odd_speeds(tmp_path):
    groups = {0: ("a100", 0, 24, "training"), 1: ("a100", 0, 24, "training")}
    # The H100 has not been measured yet (its speed is only a guess): no move.
    fleet, answers = answering_fleet(tmp_path, {"a100": "A100", "h100": "H100"}, groups,
                                     measure=False)
    measured(fleet, {"A100": 1.0})
    fleet.move_up(answers)
    assert fleet.colab.sent == []
    # Measured, and about as fast as the A100, but one odd reading for this shape on the H100
    # makes it look 5 times as fast: no move either (the H100's typical speed counts).
    fleet, answers = answering_fleet(tmp_path / "2", {"a100": "A100", "h100": "H100"}, groups,
                                     measure=False)
    measured(fleet, {"A100": 1.0, "H100": 1.05})
    shape = fleet_plan.shape_of(fleet.groups[0])
    fleet.state["speeds"]["H100"]["16x4x128"] = 1.05 * fleet_plan.a100_guess("16x4x128")
    fleet.state["speeds"]["H100"]["32x4x128"] = 1.05 * fleet_plan.a100_guess("32x4x128")
    fleet.state["machine_speeds"]["h100"] = {shape: 5 * fleet_plan.a100_guess(shape)}
    fleet.move_up(answers)
    assert fleet.colab.sent == []
    # With the H100 really 1.7 times as fast (measured), the groups move.
    fleet, answers = answering_fleet(tmp_path / "3", {"a100": "A100", "h100": "H100"}, groups)
    fleet.move_up(answers)
    assert "park 0" in fleet.colab.kinds("a100")


def test_a_group_missing_once_is_not_freed(tmp_path):
    """A look can miss a group in the instant a worker puts it back in the queue. A group being
    moved up that is missing from one answer is not freed (else it could train on two machines);
    missing from two answers in a row, it was taken off already and goes on."""
    fleet, answers = answering_fleet(tmp_path, {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 40000, 24, "training"),
                                      1: ("t4", 40000, 24, "training")})
    group = fleet.state["groups"]["0"]
    group["parking"] = {"from": "t4", "to": "h100", "at": time.time() - 60}
    for answer in answers.values():
        answer["progress"] = {}
    answers["t4"]["taken"] = [1]  # group 0: not waiting, not training, not parked
    group["absent"] = 1  # (as look_at counts it after one such answer)
    fleet.settle_parking(answers)
    assert group["machine"] == "t4" and group.get("parking")
    assert fleet.colab.kinds("h100") == []
    group["absent"] = 2
    fleet.settle_parking(answers)
    assert group["machine"] == "h100" and not group.get("parking")


def test_a_checkpoint_that_never_comes_home(tmp_path):
    """A parked group whose checkpoint does not come home goes back into the slower machine's
    queue after PARK_WAIT_SECONDS (it is never left parked for ever)."""
    fleet, answers = answering_fleet(tmp_path, {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 40000, 24, "training"),
                                      1: ("t4", 40000, 24, "training")})
    group = fleet.state["groups"]["0"]
    group["parking"] = {"from": "t4", "to": "h100", "at": time.time() - 60}
    answers["t4"].update(taken=[1], parked=[0], progress={"0": {"step": 52345, "done": False}})
    answers["h100"]["progress"] = {}
    fleet.settle_parking(answers)
    assert fleet.colab.kinds("t4") == []
    group["parked_since"] -= colab_fleet.PARK_WAIT_SECONDS + 1
    fleet.settle_parking(answers)
    assert fleet.colab.kinds("t4") == ["back in the queue 0"]
    assert group["machine"] == "t4" and not group.get("parking")
    assert "did not come home" in (tmp_path / "fleet.log").read_text()


def test_the_look_lists_only_this_stages_parked_groups(tmp_path):
    """A parked file of another stage (left on a machine kept for the next stage) is not listed."""
    import subprocess
    import sys
    out = tmp_path / "out"
    for folder in ("parked", "park", "tasks", "taken", "progress"):
        (out / folder).mkdir(parents=True)
    (out / "parked" / "3.json").write_text(json.dumps({"group": 3, "stage": "test"}))
    (out / "parked" / "4.json").write_text(json.dumps({"group": 4, "stage": "earlier"}))
    (out / "park" / "5").write_text("")
    code = colab_fleet.with_values(colab_fleet.POLL, STAGE="test").replace(
        f"BASE = {colab_fleet.BASE!r}", f"BASE = {str(tmp_path)!r}")
    printed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=tmp_path).stdout
    answer = json.loads([line for line in printed.splitlines()
                         if line.startswith("ANSWER ")][-1][len("ANSWER "):])
    assert answer["parked"] == [3] and answer["park_signals"] == [5]
    assert answer["can_park"] is False  # (no abilities.json: a job of an older machine_job.py)


def test_a_parked_group_that_is_not_there_any_more_is_not_freed_at_once(tmp_path):
    """An answer (asked earlier) says the group is parked, but taking it off finds nothing: it may
    have gone back into the queue meanwhile and be training again. It is not freed on that
    answer (only after two answers in a row without it)."""
    fleet, answers = answering_fleet(tmp_path, {"t4": "T4", "h100": "H100"},
                                     {0: ("t4", 40000, 24, "training"),
                                      1: ("t4", 40000, 24, "training")})
    group = fleet.state["groups"]["0"]
    group["parking"] = {"from": "t4", "to": "h100", "at": time.time() - 60}
    answers["t4"].update(taken=[1], parked=[0], progress={"0": {"step": 52345, "done": False}})
    answers["h100"]["progress"] = {}
    group["saved_step"] = 52345
    answer_as = fleet.colab.execute

    def not_there(name, code, timeout=300):
        if "taken_back_parked" in code:
            fleet.colab.sent.append((name, code))
            return True, "PARKED NOT THERE"
        return answer_as(name, code, timeout)

    fleet.colab.execute = not_there
    fleet.settle_parking(answers)
    assert group["machine"] == "t4" and group.get("parking")
    assert fleet.colab.kinds("h100") == []
    assert "WHERE = ('parked',)" in fleet.colab.sent[-1][1]  # (only from parked/)


def test_a_worker_parks_a_waiting_group_with_a_signal(tmp_path):
    """v3/machine_job.py: a group waiting in the queue with a park signal is parked by the next
    worker that looks for work (not trained), and its signal is removed; another is taken."""
    from v3 import machine_job
    out = tmp_path / "out"
    for folder in machine_job.FOLDERS:
        (out / folder).mkdir(parents=True)
    for number in (3, 4):
        (out / "tasks" / f"{number}.json").write_text(json.dumps({"group": number}))
    (out / "progress" / "3.json").write_text(json.dumps({"step": 0}))
    (out / "progress" / "4.json").write_text(json.dumps({"step": 9}))  # (3 would come first)
    (out / "park" / "3").write_text("")
    claimed, task = machine_job.take_next(out, 0)
    assert task == {"group": 4} and claimed.name == "4.w0.json"
    assert (out / "parked" / "3.json").exists() and not (out / "park" / "3").exists()
    assert not list((out / "tasks").iterdir())

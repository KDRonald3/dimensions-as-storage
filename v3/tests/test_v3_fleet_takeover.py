"""A reviewer's checks of the upgraded launcher (v3/colab_fleet.py) taking over from the launcher
that ran the counted stage on 23 November (kept as v3/tests/launcher_before.py), with the
pretend Colab tool and real training on the processor.

  1. Take-over: the old launcher runs a slow machine (a T4) with groups training and waiting,
     then is stopped (its machine goes on). The new launcher takes over its state file, finds a
     faster card, moves groups (between rounds, or by stopping the T4 in the middle of a round).
     Every run comes home equal to straight training, and no group trains the same stretch twice
     (which would show both a group trained twice and a checkpoint that was lost).
  2. The same while the T4's downloads fail half the time: the T4 is not given back before
     everything it saved has come home.
  3. A grant whose answer was lost (the tool wrote the machine into its file, then failed): the
     launcher finds the machine and takes it in (and stops it at the end); a machine of another
     launcher and one of the owner's are not touched.
"""

import json
import re
import stat
import sys
import time

import pytest
import torch

from v2 import batch
from v3 import colab_fleet, colab_gpu
from v3.grid import make_config, name_of
from v3.tests import launcher_before
from v3.tests.test_v3_fleet import HERE, fake, without_time  # noqa: F401
from v3.tests.test_v3_fleet_plan import set_fake, slow_guesses


def groups_of_four():
    """Four tiny groups of two members, with four rounds each (looks at 200, 400, 800, 1600)."""
    groups = []
    for columns in (8, 16):
        for seeds in ((0, 1), (2, 3)):
            group = []
            for seed in seeds:
                config = make_config(columns, 16, 4, "equal", 4, seed, 0.01, 1600, practice=False)
                config.update(first_look=200, batch=32, name=f"k{columns}_" + name_of(config))
                group.append(config)
            groups.append(group)
    return groups


def tool_with(tmp_path, extra):
    """The pretend tool behind a small wrapper (`extra`: Python run before it, may exit)."""
    wrapper = tmp_path / "colab" / "wrapper.py"
    wrapper.write_text(f"""
import json, os, subprocess, sys
from pathlib import Path
argv = sys.argv[1:]
{extra}
sys.exit(subprocess.run([sys.executable, {str(HERE / "fake_colab.py")!r}] + argv).returncode)
""")
    tool = tmp_path / "colab" / "wrapped"
    tool.write_text(f"#!/bin/sh\nexec {sys.executable} {wrapper} \"$@\"\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return tool


def run_old_launcher_until_busy(tmp_path, tool):
    """The launcher as it ran on 23 November, with only a T4 free, until the T4 has finished a
    round of a group and is training another one. Then it is simply left (as `systemctl stop`)."""
    old = launcher_before.Fleet(
        "test", groups_of_four(), launcher_before.Colab(tool), runs_root=tmp_path / "runs",
        state_path=tmp_path / "fleet.json", log=tmp_path / "old.log", machines=1, workers=1,
        cards=("A100", "T4"), poll_seconds=0.5, lost_minutes=1,
        grab=lambda count: colab_gpu.grab(count, "air", wait_seconds=10, cards=["A100", "T4"],
                                          colab=tool))
    for _ in range(120):
        assert old.step(), "the old launcher finished alone; the test proves nothing"
        groups = old.state["groups"].values()
        if any(g.get("saved_step") for g in groups) and old.state["machines"]:
            return next(iter(old.state["machines"]))
        time.sleep(0.5)
    raise AssertionError("the old launcher never got busy")


def stretches(tmp_path):
    """(group, from step) of every sitting on every machine, from the workers' logs."""
    found = []
    for log in (tmp_path / "colab" / "machines").glob("*/air_run/out/logs/worker*.log"):
        found += re.findall(r"group (\d+): from step ([\d,]+) to", log.read_text())
    return found


def check_exact_and_once(tmp_path, state_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as the machine's workers on the processor
    try:
        for number, configs in enumerate(groups_of_four()):
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
    state = json.loads(state_path.read_text())
    assert not any(m["running"] for m in state["machines"].values())


def take_over(tmp_path, tool, monkeypatch):
    slow_guesses(monkeypatch)
    fleet = colab_fleet.Fleet("test", groups_of_four(), colab_fleet.Colab(tool),
                              runs_root=tmp_path / "runs", state_path=tmp_path / "fleet.json",
                              log=tmp_path / "fleet.log", machines=3, workers=1,
                              cards=("A100", "T4"), poll_seconds=0.5, lost_minutes=1)
    for _ in range(400):
        if not fleet.step():
            return fleet
        time.sleep(0.5)
    raise AssertionError("the new launcher did not finish")


def test_take_over_from_the_running_launcher(tmp_path, fake, monkeypatch):
    tool, state_path = fake
    set_fake(state_path, refuse=["H100", "G4", "A100", "L4"])
    t4 = run_old_launcher_until_busy(tmp_path, tool)
    set_fake(state_path, refuse=["H100", "G4", "L4"])  # an A100 is free now
    take_over(tmp_path, tool, monkeypatch)
    log = (tmp_path / "fleet.log").read_text()
    assert f"machine {t4} (T4)" in log  # taken over, not started again
    assert re.search(r"machine air-a100-\S+ \(A100\) added", log)
    assert re.search(rf"move from {t4} to air-a100|machine {t4} \(T4\): stopped in the middle",
                     log)
    check_exact_and_once(tmp_path, state_path)


def test_a_leaving_machine_waits_until_its_work_is_home(tmp_path, fake, monkeypatch):
    tool, state_path = fake
    counter = tmp_path / "colab" / "downloads"
    machines = tmp_path / "colab" / "machines"
    # Once the T4 has been told to stop, its downloads fail until its job has ended and twice
    # more after that (the connection is bad just when its last checkpoints should come home).
    flaky = tool_with(tmp_path, f"""
name = argv[argv.index("-s") + 1] if "-s" in argv else ""
out = Path({str(machines)!r}) / name / "air_run" / "out"
if "download" in argv and "-t4-" in name and (out / "stop").exists():
    count = Path({str(counter)!r})
    n = int(count.read_text()) + 1 if count.exists() else 1
    if (out / "job.json").exists():
        count.write_text(str(n))
    if not (out / "job.json").exists() or n <= 2:
        print("Error: Connection was lost")
        sys.exit(1)
""")
    set_fake(state_path, refuse=["H100", "G4", "A100", "L4"])
    run_old_launcher_until_busy(tmp_path, tool)
    set_fake(state_path, refuse=["H100", "G4", "L4"])
    take_over(tmp_path, flaky, monkeypatch)
    check_exact_and_once(tmp_path, state_path)


@pytest.mark.parametrize("written", [True, False])
def test_a_grant_whose_answer_was_lost_is_found(tmp_path, fake, monkeypatch, written):
    """The tool got an A100 and wrote it into its file, then failed before saying READY (as
    after a timeout or a crash). Colab still runs it: the launcher must find it and give it
    back, but never touch a machine of the owner's or one another launcher holds."""
    tool, state_path = fake
    configs = tmp_path / "configs"
    lost = tool_with(tmp_path, f"""
state = json.loads(Path(os.environ["FAKE_COLAB_STATE"]).read_text())
if argv[:1] == ["--config"]:
    config, rest = Path(argv[1]), argv[2:]
else:
    config, rest = None, argv
if rest[:1] == ["sessions"]:
    known = {{}}
    for path in Path({str(configs)!r}).glob("*.json"):
        known.update({{s["endpoint"]: n for n, s in json.loads(path.read_text()).items()}})
    for name, m in state["machines"].items():
        if m["running"] and name not in state.get("gone", []):
            print(f"[{{known.get(name, '?')}}] {{name}} | Hardware: {{m['card']}} | Shape: STANDARD")
    sys.exit(0)
if rest[:1] == ["new"] and config is not None:
    name = rest[rest.index("-s") + 1]
    done = subprocess.run([sys.executable, {str(HERE / "fake_colab.py")!r}] + argv,
                          capture_output=True, text=True)
    if "READY" in done.stdout:
        if {written!r}:  # the tool wrote it into its file before failing
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(json.dumps({{name: {{"name": name, "endpoint": name, "token": "x"}}}}))
        if "-a100-" in name:  # the answer is lost
            print("requests.exceptions.ReadTimeout: read timeout=120")
            sys.exit(1)
    print(done.stdout)
    sys.exit(done.returncode)
# (The pretend tool knows a machine by its name; the real one by its address in the file.)
if config is not None and config.exists() and "-s" in argv:
    sessions = json.loads(config.read_text() or "{{}}")
    name = argv[argv.index("-s") + 1]
    if name in sessions and sessions[name].get("endpoint") not in (None, name):
        argv[argv.index("-s") + 1] = sessions[name]["endpoint"]
if rest[:1] == ["stop"] and config is not None and config.exists():
    name = rest[rest.index("-s") + 1]
    config.write_text(json.dumps({{k: v for k, v in json.loads(config.read_text()).items()
                                   if k != name}}))
""")
    # Another launcher's machine (another stage, its own state file next to ours).
    other = "air-t4-070000-9"
    state = json.loads(state_path.read_text())
    state["machines"][other] = {"card": "T4", "running": True}
    state["machines"]["owners-browser"] = {"card": "L4", "running": True}  # no request in doubt
    state_path.write_text(json.dumps(state))
    configs.mkdir(parents=True, exist_ok=True)
    (configs / f"{other}.json").write_text(json.dumps({other: {"name": other, "endpoint": other}}))
    (tmp_path / "fleet_other.json").write_text(json.dumps(
        {"machines": {other: {"card": "T4"}}, "groups": {}, "failures": {}}))
    monkeypatch.setattr(colab_fleet, "STRAY_MINUTES", 0.0)  # (no waiting in a test)
    set_fake(state_path, refuse=["H100", "G4", "L4"])
    fleet = colab_fleet.Fleet("test", groups_of_four()[:1], colab_fleet.Colab(lost),
                              runs_root=tmp_path / "runs", state_path=tmp_path / "fleet_test.json",
                              log=tmp_path / "fleet.log", machines=2, workers=1,
                              cards=("A100", "T4"), poll_seconds=0.5, lost_minutes=1)
    for _ in range(300):
        if not fleet.step():
            break
        time.sleep(0.5)
    machines = json.loads(state_path.read_text())["machines"]
    strays = [n for n in machines if "-a100-" in n]
    assert strays and not any(machines[n]["running"] for n in strays), machines
    assert machines[other]["running"] and machines["owners-browser"]["running"]
    log = (tmp_path / "fleet.log").read_text()
    assert re.search(r"no launcher knew it .*taken as air-a100-", log)
    assert re.search(r"machine air-a100-\S+: (job started|nothing left to do)", log)
    assert "owners-browser" not in log


# ------------------------------------------------------------------------------------------------
# 4. A machine is never stopped while what it saved last is still only there
# ------------------------------------------------------------------------------------------------


def test_no_stop_while_the_last_download_failed(tmp_path):
    from v3.tests.test_v3_fleet_plan import PretendColab, pretend_fleet

    class Colab(PretendColab):
        stopped = []

        def exists(self, name):
            return True

        def stop(self, name):
            self.stopped.append(name)
            return True

    fleet, _ = pretend_fleet(tmp_path, {"t4": "T4"}, {0: ("t4", 400, 24, "training")})
    fleet.colab = Colab()
    fleet.state["machines"]["t4"]["leaving"] = True
    ended = {"new": 1, "waiting": [], "taken": [], "progress": {}, "alive": 0, "card": ""}
    answers = [dict(ended, incomplete=True)] * 4 + [dict(ended)] * 2  # two polls a look
    fleet.poll = lambda name: answers.pop(0)
    colab_fleet.time.sleep, sleep = (lambda s: None), colab_fleet.time.sleep
    try:
        assert fleet.look_at("t4") is None and fleet.colab.stopped == []  # download failed
        assert fleet.look_at("t4") is None and fleet.colab.stopped == []  # (and once more)
        assert fleet.look_at("t4") is None and fleet.colab.stopped == ["t4"]  # all home now
        assert "t4" not in fleet.state["machines"]
        # The clean stop (reserve reached) waits the same way.
        fleet, _ = pretend_fleet(tmp_path / "2", {"t4": "T4"}, {0: ("t4", 400, 24, "training")})
        fleet.colab = Colab()
        Colab.stopped = []
        answers = [dict(ended, incomplete=True), dict(ended, incomplete=True), dict(ended),
                   dict(ended)]
        fleet.poll = lambda name: answers.pop(0)
        fleet.stop_all(wait_minutes=1)
        assert fleet.colab.stopped == ["t4"] and answers == []
        assert "had not finished saving" not in (tmp_path / "2" / "fleet.log").read_text()
    finally:
        colab_fleet.time.sleep = sleep

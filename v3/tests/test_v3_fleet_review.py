"""A reviewer's further checks of the fleet (v3/colab_fleet.py) with the pretend Colab tool.
Processor only.

  1. Lost connections: every third request to a machine is carried out but then reported as
     failed ("Connection was lost"), so the fleet tries it again. Still every run comes home,
     equal to the group trained straight through, no group is trained on two machines, and every
     machine is stopped.
  2. A session of the owner's that the fleet did not create is never stopped or touched.
  3. Stop (--stop) in the middle, then start again: it goes on and finishes, with the same
     results as straight through.
  4. Groups that fail every time: the fleet ends (it does not wait for ever).
"""

import json
import stat
import sys

import torch

from v2 import batch
from v3 import colab_fleet
from v3.tests.test_v3_fleet import (HERE, fake, make_fleet, run_until_done,  # noqa: F401
                                    tiny_groups, without_time)

OTHER = "air-a100-064718-1"  # a session of the owner's (as the practice ones were named)


def flaky_tool(tmp_path):
    """A tool that runs the pretend one, but reports every third `exec` as lost after it ran."""
    wrapper = tmp_path / "colab" / "flaky.py"
    wrapper.write_text(f"""
import subprocess, sys
from pathlib import Path
count = Path({str(tmp_path / "colab" / "count")!r})
done = subprocess.run([sys.executable, {str(HERE / "fake_colab.py")!r}] + sys.argv[1:])
if "exec" in sys.argv:
    n = int(count.read_text()) + 1 if count.exists() else 1
    count.write_text(str(n))
    if n % 3 == 0:
        print("Error: Connection was lost")
        sys.exit(1)
sys.exit(done.returncode)
""")
    tool = tmp_path / "colab" / "flaky"
    tool.write_text(f"#!/bin/sh\nexec {sys.executable} {wrapper} \"$@\"\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return tool


def add_owners_session(state_path):
    state = json.loads(state_path.read_text())
    state["machines"][OTHER] = {"card": "A100", "running": True}
    state_path.write_text(json.dumps(state))


def check_results(tmp_path, state_path):
    """Every run here and equal to straight training; each group on one machine; all of the
    fleet's machines stopped; the owner's session untouched."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as the machine's workers on the processor (sums in their order)
    try:
        wholes = [batch.train_group(configs, tmp_path / f"whole{number}", "cpu", quiet=True)[0]
                  for number, configs in enumerate(tiny_groups())]
    finally:
        torch.set_num_threads(threads)
    for whole in wholes:
        for summary in whole:
            here = tmp_path / "runs" / "test" / summary["name"]
            assert (here / "model.pt").exists()
            assert without_time(json.loads((here / "summary.json").read_text())) == \
                without_time(summary)
    trained = {}
    for model in (tmp_path / "colab" / "machines").glob("*/air_run/out/test/*/model.pt"):
        trained.setdefault(model.parent.name, set()).add(model.parts[-6])
    assert trained and all(len(machines) == 1 for machines in trained.values())
    state = json.loads(state_path.read_text())
    assert state["machines"][OTHER]["running"]
    assert all(not m["running"] for n, m in state["machines"].items() if n != OTHER)
    assert not any(name == OTHER for _, name in state["log"])


def test_lost_connections_and_the_owners_session(tmp_path, fake):
    _, state_path = fake
    add_owners_session(state_path)
    fleet = make_fleet(tmp_path, flaky_tool(tmp_path))
    run_until_done(fleet, steps=200)
    check_results(tmp_path, state_path)


def test_stop_and_start_again(tmp_path, fake):
    tool, state_path = fake
    add_owners_session(state_path)
    fleet = make_fleet(tmp_path, tool)
    for _ in range(6):  # some training, then stopped as by `--stop`
        fleet.step()
    fleet.stop_all(wait_minutes=1)
    assert not fleet.state["machines"]
    assert not any(g.get("machine") for g in fleet.state["groups"].values())
    again = make_fleet(tmp_path, tool)
    run_until_done(again)
    check_results(tmp_path, state_path)


def test_groups_that_always_fail_end_the_fleet(tmp_path, fake, monkeypatch):
    tool, _ = fake
    broken = [[dict(c, kinds=3) for c in group] for group in tiny_groups()]  # 3 kinds: an error
    fleet = colab_fleet.Fleet("test", broken, colab_fleet.Colab(tool), runs_root=tmp_path / "runs",
                              state_path=tmp_path / "fleet.json", log=tmp_path / "fleet.log",
                              machines=1, workers=1, cards=("A100",), poll_seconds=0.3,
                              lost_minutes=5)
    run_until_done(fleet, steps=200)
    assert "failed three times" in (tmp_path / "fleet.log").read_text()
    assert not fleet.state["machines"]

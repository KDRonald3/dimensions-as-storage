"""Tests of the breadth-first job of one machine (v3/machine_job.py), of the group trainer's
pausing and "so far" files (v2/batch.py), and of the fleet (v3/colab_fleet.py) with a pretend
Colab tool (fake_colab.py). Processor only."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from v2 import batch
from v3 import colab_fleet, colab_gpu, machine_job
from v3.grid import make_config, name_of

HERE = Path(__file__).resolve().parent


def tiny_groups():
    """Two tiny groups of two members (as the tiny groups of the test stage): 16 creatures, the
    doubling rule with a first look at step 200 and a limit of 400."""
    groups = []
    for columns in (8, 16):
        group = []
        for seed in (0, 1):
            config = make_config(columns, 16, 4, "equal", 4, seed, 0.01, 400, practice=False)
            config.update(first_look=200, batch=32, name=f"t{columns}_" + name_of(config))
            group.append(config)
        groups.append(group)
    return groups


def without_time(summary):
    return {k: v for k, v in summary.items() if k not in ("minutes", "steps_per_second")}


def test_pause_writes_so_far_and_goes_on_exactly(tmp_path):
    group = tiny_groups()[0]
    whole, _ = batch.train_group(group, tmp_path / "whole", "cpu", quiet=True)
    first = batch.run_group_until(group, tmp_path / "parts", "cpu", pause_at=200,
                                  so_far_root=tmp_path / "so_far", quiet=True)
    assert first["step"] == 200 and sorted(first["unfinished"]) == sorted(c["name"] for c in group)
    note = json.loads((tmp_path / "so_far" / f"{group[0]['name']}.json").read_text())
    assert note["status"] == "still training" and note["step"] == 200
    assert note["curve"][-1]["step"] == 200  # the look at step 200 is in it
    again = batch.run_group_until(group, tmp_path / "parts", "cpu", pause_at=400,
                                  so_far_root=tmp_path / "so_far", quiet=True)
    assert not again["unfinished"]
    assert not list((tmp_path / "so_far").glob("*.json"))  # ended runs are no longer "so far"
    for summary in whole:
        parts = json.loads((tmp_path / "parts" / summary["name"] / "summary.json").read_text())
        assert without_time(parts) == without_time(summary)


def test_next_pause():
    config = {"length_rule": "doubling", "first_look": 40000, "steps": 640000}
    assert [machine_job.next_pause(config, s) for s in (0, 39999, 40000, 80000, 600000)] == \
        [40000, 40000, 80000, 160000, 640000]
    assert machine_job.next_pause({"steps": 40000}, 0) is None


def test_machine_job_is_breadth_first(tmp_path, monkeypatch):
    out = tmp_path / "out"
    for folder in machine_job.FOLDERS:
        (out / folder).mkdir(parents=True)
    groups = tiny_groups()
    for number, configs in enumerate(groups):
        machine_job.write_json(out / "tasks" / f"{number}.json",
                               {"group": number, "stage": "test", "configs": configs})
    (out / "close").write_text("")
    rounds = []
    real = machine_job.train_one_round

    def recorded(task, out_folder, device):
        rounds.append((task["group"], machine_job.reached(out_folder, task["group"])))
        return real(task, out_folder, device)

    monkeypatch.setattr(machine_job, "train_one_round", recorded)
    machine_job.worker(0, out, "cpu")
    assert rounds == [(0, 0), (1, 0), (0, 200), (1, 200)]  # both to 200, then both to 400
    assert not list(out.glob("tasks/*.json")) and not list(out.glob("taken/*.json"))
    for number, configs in enumerate(groups):
        assert json.loads((out / "progress" / f"{number}.json").read_text())["done"]
        whole, _ = batch.train_group(configs, tmp_path / f"whole{number}", "cpu", quiet=True)
        for summary in whole:
            here = json.loads((out / "test" / summary["name"] / "summary.json").read_text())
            assert without_time(here) == without_time(summary)
    assert (out / "handoff" / "0.pt").exists()  # the checkpoint made when it was put back


# ------------------------------------------------------------------------------------------------
# The fleet, with the pretend Colab tool
# ------------------------------------------------------------------------------------------------


@pytest.fixture
def fake(tmp_path, monkeypatch):
    state = tmp_path / "colab" / "state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"machines": {}, "refuse": ["H100", "G4"], "balance": 500.0,
                                 "rate": 0.0}))
    tool = tmp_path / "colab" / "colab"
    tool.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_colab.py'} \"$@\"\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_COLAB_STATE", str(state))
    monkeypatch.setattr(colab_gpu, "CONFIGS", tmp_path / "configs")
    monkeypatch.setattr(colab_fleet, "RETRIES", 1)
    yield tool, state
    import subprocess
    subprocess.run(["pkill", "-f", str(tmp_path / "colab" / "machines")])


def make_fleet(tmp_path, tool, **options):
    settings = dict(machines=2, reserve=100.0, workers=1, cards=("H100", "G4", "A100"),
                    poll_seconds=0.5, lost_minutes=0.05)
    settings.update(options)
    return colab_fleet.Fleet("test", tiny_groups(), colab_fleet.Colab(tool),
                             runs_root=tmp_path / "runs", state_path=tmp_path / "fleet.json",
                             log=tmp_path / "fleet.log", **settings)


def run_until_done(fleet, steps=120):
    for _ in range(steps):
        if not fleet.step():
            return
        import time
        time.sleep(fleet.poll_seconds)
    raise AssertionError("the fleet did not finish")


def every_run_here(tmp_path):
    names = {c["name"] for g in tiny_groups() for c in g}
    return names == {p.parent.name for p in (tmp_path / "runs" / "test").glob("*/summary.json")}


def test_fleet_trains_everything_once_and_stops_every_machine(tmp_path, fake):
    tool, state_path = fake
    fleet = make_fleet(tmp_path, tool)
    run_until_done(fleet)
    assert every_run_here(tmp_path)
    for path in (tmp_path / "runs" / "test").glob("*/summary.json"):
        assert (path.parent / "model.pt").exists()
    state = json.loads(state_path.read_text())
    assert not any(m["running"] for m in state["machines"].values())  # all stopped
    assert all(m["card"] == "A100" for m in state["machines"].values())  # H100, G4 refused
    # Each group was trained by one machine only: every summary on a machine is for one group.
    trained = {}
    for summary in (tmp_path / "colab" / "machines").glob("*/air_run/out/test/*/model.pt"):
        trained.setdefault(summary.parent.name, []).append(summary.parts[-6])
    assert all(len(machines) == 1 for machines in trained.values())
    assert not list((tmp_path / "runs" / "test" / "_so_far").glob("*.json"))


def test_fleet_keeps_the_reserve(tmp_path, fake):
    tool, state_path = fake
    state = json.loads(state_path.read_text())
    state["balance"] = 100.5
    state_path.write_text(json.dumps(state))
    fleet = make_fleet(tmp_path, tool)
    assert fleet.step() is False  # stops at once, and asks for no machine
    assert json.loads(state_path.read_text())["machines"] == {}


def test_fleet_gives_up_a_lost_machine_and_survives_a_restart(tmp_path, fake):
    tool, state_path = fake
    fleet = make_fleet(tmp_path, tool)
    fleet.step()  # two machines, the groups handed out
    held = {name: fleet.held_by(name) for name in fleet.state["machines"]}
    assert sorted(i for indices in held.values() for i in indices) == [0, 1]
    lost = next(iter(held))
    state = json.loads(state_path.read_text())
    state["gone"] = [lost]
    state_path.write_text(json.dumps(state))
    # Started again (a new program reading the same state file): goes on and finishes.
    again = make_fleet(tmp_path, tool, machines=1)
    run_until_done(again)
    assert every_run_here(tmp_path)
    assert lost in (tmp_path / "fleet.log").read_text()
    assert not again.state["machines"]
